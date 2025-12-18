# -----------------------------------------------------------------------------------
# Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement
# Novel architecture combining SwinIR with illumination-guided attention
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# =============================================================================
# Helper Functions (from original SwinIR)
# =============================================================================

def window_partition(x, window_size):
    """
    Partition image into non-overlapping windows.
    
    Args:
        x: (B, H, W, C) Input tensor
        window_size: Window size (int)
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition back to image.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Image height and width
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# NOVEL COMPONENT 1: Illumination Estimation Module (IEM)
# =============================================================================

class IlluminationEstimationModule(nn.Module):
    """
    Estimates illumination map from input low-light image.
    
    This module uses a lightweight CNN to estimate per-pixel illumination levels.
    Based on Retinex theory: I = R * L, where L is the illumination component.
    
    The output is inverted so that:
    - Dark regions → High values (more processing needed)
    - Bright regions → Low values (preserve as-is)
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        hidden_channels: Intermediate feature channels (default: 32)
    """
    
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        
        # Initial illumination estimation using max channel (Retinex-inspired)
        # Max of RGB gives a rough illumination estimate
        
        # Refinement network: small UNet-like structure
        self.refine = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Bottleneck with dilation for larger receptive field
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Decoder
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),  # Output: 1 channel illumination map
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            illum_map: Illumination map (B, 1, H, W) where 0=dark, 1=bright
            dark_mask: Inverted map (B, 1, H, W) where 1=dark, 0=bright
        """
        # Get rough illumination using max channel (Retinex theory)
        rough_illum = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # Refine the illumination estimate
        illum_map = self.refine(x)
        
        # Blend rough and refined estimates
        illum_map = 0.5 * rough_illum + 0.5 * illum_map
        
        # Create dark mask: invert so dark areas have high values
        dark_mask = 1.0 - illum_map
        
        return illum_map, dark_mask


# =============================================================================
# NOVEL COMPONENT 2: Illumination-Guided Attention Module (IGAM)
# =============================================================================

class IlluminationGuidedAttention(nn.Module):
    """
    Novel attention mechanism that modulates features based on illumination.
    
    This is THE KEY NOVELTY of Swin-LLIE!
    
    Mathematical formulation:
        F_out = F_in * (1 + α * M)
    
    Where:
        - F_in: Input features from Swin blocks
        - M: Dark mask (high values for dark regions)
        - α: Learnable adaptive scaling factor
    
    This allows the network to:
        - Apply stronger enhancement to dark regions (M is high)
        - Preserve bright regions as-is (M is low)
    
    Args:
        dim: Number of feature channels
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Channel attention to compute adaptive scaling α
        # Uses Global Average Pooling + FC layers
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
        
        # Learnable base scaling parameter (initialized to 0.5)
        self.base_alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # Optional: spatial attention refinement
        self.spatial_refine = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, features, dark_mask):
        """
        Args:
            features: (B, C, H, W) Feature tensor from Swin blocks
            dark_mask: (B, 1, H, W) Dark mask where dark=1, bright=0
        
        Returns:
            modulated_features: (B, C, H, W) Illumination-modulated features
        """
        B, C, H, W = features.shape
        
        # 1. Compute adaptive channel scaling α
        alpha = self.channel_attention(features)  # (B, C)
        alpha = alpha.view(B, C, 1, 1)  # Reshape for broadcasting
        
        # 2. Resize dark_mask to match feature resolution
        if dark_mask.shape[2:] != features.shape[2:]:
            dark_mask = F.interpolate(dark_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. Refine spatial attention
        refined_mask = torch.sigmoid(self.spatial_refine(dark_mask))
        
        # 4. Apply illumination-guided modulation
        # F_out = F_in * (1 + base_α * α * M)
        modulation = 1.0 + self.base_alpha * alpha * refined_mask
        modulated_features = features * modulation
        
        return modulated_features


# =============================================================================
# Core SwinIR Components (with minor modifications)
# =============================================================================

class Mlp(nn.Module):
    """MLP module used in Transformer blocks."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA) module.
    Standard Swin Transformer attention mechanism.
    
    Args:
        dim: Number of input channels
        window_size: Size of attention window
        num_heads: Number of attention heads
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Compute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with W-MSA and SW-MSA.
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (flattening for Transformer)."""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding (reshaping back to 2D)."""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


# =============================================================================
# NOVEL COMPONENT 3: RSTB with Illumination-Guided Attention (RSTB_IGAM)
# =============================================================================

class RSTB_IGAM(nn.Module):
    """
    Residual Swin Transformer Block with Illumination-Guided Attention.
    
    This is a modified RSTB that incorporates our novel IGAM module.
    The key difference from original RSTB:
        - After Swin processing, apply illumination-guided modulation
        - Dark regions get enhanced processing, bright regions preserved
    
    Args:
        dim: Feature dimension
        input_resolution: Input resolution (H, W)
        depth: Number of Swin Transformer blocks
        num_heads: Number of attention heads
        window_size: Window size for attention
        use_igam: Whether to use illumination-guided attention
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv', use_igam=True):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_igam = use_igam

        # Standard Swin Transformer layer
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        # Residual connection convolution
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # Patch embedding/unembedding for shape handling
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        # NOVEL: Illumination-Guided Attention
        if use_igam:
            self.igam = IlluminationGuidedAttention(dim)

    def forward(self, x, x_size, dark_mask=None):
        """
        Args:
            x: (B, H*W, C) Input features
            x_size: (H, W) Spatial size
            dark_mask: (B, 1, H, W) Dark region mask (optional)
        
        Returns:
            Enhanced features with residual connection
        """
        # Standard RSTB processing
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        residual = self.conv(residual)
        
        # NOVEL: Apply illumination-guided attention
        if self.use_igam and dark_mask is not None:
            residual = self.igam(residual, dark_mask)
        
        residual = self.patch_embed(residual)
        
        return residual + x


# =============================================================================
# NOVEL COMPONENT 4: Cross-Stage Feature Fusion (CSFF)
# =============================================================================

class CrossStageFeatureFusion(nn.Module):
    """
    Cross-Stage Feature Fusion module for U-Net skip connections.
    
    Fuses encoder features with decoder features using learned attention.
    This helps preserve structural information while allowing brightness changes.
    
    Args:
        in_dim: Input dimension from encoder
        out_dim: Output dimension for decoder
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # Channel adjustment if dimensions differ
        self.channel_adjust = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
        # Fusion attention
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )
        
        # Learnable fusion weight
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, out_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: Features from encoder (skip connection)
            decoder_feat: Features from decoder
        
        Returns:
            Fused features
        """
        encoder_feat = self.channel_adjust(encoder_feat)
        
        # Ensure same spatial size
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(encoder_feat, size=decoder_feat.shape[2:], 
                                         mode='bilinear', align_corners=False)
        
        # Concatenate and compute attention gate
        concat = torch.cat([encoder_feat, decoder_feat], dim=1)
        gate_weight = self.gate(concat)
        
        # Fuse with learned weights
        fused = self.fusion_conv(concat)
        output = gate_weight * encoder_feat + (1 - gate_weight) * decoder_feat + fused
        
        return output


# =============================================================================
# MAIN MODEL: SwinLLIE
# =============================================================================

class SwinLLIE(nn.Module):
    """
    Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement
    
    A novel architecture that combines:
    1. SwinIR's powerful feature extraction (RSTB blocks)
    2. Illumination Estimation Module (IEM) for dark region detection
    3. Illumination-Guided Attention Module (IGAM) for adaptive enhancement
    4. U-Net style skip connections for structure preservation
    
    Architecture Overview:
        Input → IEM → Encoder (3 stages) → Decoder (3 stages) → Output
                 ↓           ↓                    ↑
              Dark Mask → IGAM modulation at each stage
    
    Args:
        img_size: Input image size (default: 128)
        patch_size: Patch size for embedding (default: 1)
        in_chans: Input channels (default: 3 for RGB)
        embed_dim: Base embedding dimension (default: 60)
        depths: Number of blocks per stage (default: [4, 4, 4])
        num_heads: Attention heads per stage (default: [6, 6, 6])
        window_size: Attention window size (default: 8)
        mlp_ratio: MLP expansion ratio (default: 2)
        use_igam: Use illumination-guided attention (default: True)
    """
    
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[4, 4, 4], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 use_igam=True, **kwargs):
        super().__init__()
        
        self.img_range = img_range
        self.window_size = window_size
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.use_igam = use_igam
        
        # Normalization mean for RGB images
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        # ========================
        # NOVEL: Illumination Estimation
        # ========================
        self.illumination_estimator = IlluminationEstimationModule(in_channels=in_chans)
        
        # ========================
        # Shallow Feature Extraction
        # ========================
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # ========================
        # Encoder (with IGAM)
        # ========================
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build encoder stages
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_stage in range(self.num_stages):
            # Scale resolution for deeper stages
            stage_resolution = (patches_resolution[0] // (2 ** i_stage),
                              patches_resolution[1] // (2 ** i_stage))
            stage_dim = embed_dim * (2 ** i_stage)
            
            layer = RSTB_IGAM(
                dim=stage_dim,
                input_resolution=stage_resolution,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i_stage),
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_igam=use_igam)
            self.encoder_layers.append(layer)
            
            # Downsample between stages (except last)
            if i_stage < self.num_stages - 1:
                downsample = nn.Conv2d(stage_dim, stage_dim * 2, kernel_size=2, stride=2)
                self.downsample_layers.append(downsample)
        
        # ========================
        # Decoder (with skip connections)
        # ========================
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i_stage in range(self.num_stages - 2, -1, -1):
            stage_resolution = (patches_resolution[0] // (2 ** i_stage),
                              patches_resolution[1] // (2 ** i_stage))
            stage_dim = embed_dim * (2 ** i_stage)
            higher_dim = stage_dim * 2
            
            # Upsample from deeper stage
            upsample = nn.Sequential(
                nn.ConvTranspose2d(higher_dim, stage_dim, kernel_size=2, stride=2),
                nn.LeakyReLU(0.2, inplace=True))
            self.upsample_layers.append(upsample)
            
            # Cross-stage fusion with skip connection
            fusion = CrossStageFeatureFusion(stage_dim, stage_dim)
            self.fusion_layers.append(fusion)
            
            # Decoder RSTB block
            layer = RSTB_IGAM(
                dim=stage_dim,
                input_resolution=stage_resolution,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i_stage),
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_igam=use_igam)
            self.decoder_layers.append(layer)
        
        # ========================
        # Output Reconstruction
        # ========================
        self.norm = norm_layer(embed_dim)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=None)
        
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        """Pad image to be divisible by window size."""
        _, _, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        
        # Also ensure divisible by 2^(num_stages-1) for downsampling
        scale = 2 ** (self.num_stages - 1)
        total_h = h + pad_h
        total_w = w + pad_w
        pad_h += (scale - total_h % scale) % scale
        pad_w += (scale - total_w % scale) % scale
        
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        return x
    
    def forward(self, x):
        """
        Forward pass of Swin-LLIE.
        
        Args:
            x: Input low-light image (B, 3, H, W) normalized to [0, 1]
        
        Returns:
            Enhanced image (B, 3, H, W) in same range as input
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        # Normalize input
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # ========================
        # Step 1: Estimate illumination
        # ========================
        illum_map, dark_mask = self.illumination_estimator((x / self.img_range) + self.mean)
        
        # ========================
        # Step 2: Shallow feature extraction
        # ========================
        x_shallow = self.conv_first(x)
        
        # ========================
        # Step 3: Encoder forward pass
        # ========================
        encoder_features = []
        x_enc = x_shallow
        
        for i_stage in range(self.num_stages):
            # Get current resolution
            curr_h, curr_w = x_enc.shape[2:]
            x_size = (curr_h, curr_w)
            
            # Patch embedding
            x_flat = self.patch_embed.forward(x_enc) if i_stage == 0 else x_enc.flatten(2).transpose(1, 2)
            x_flat = self.pos_drop(x_flat)
            
            # Process through RSTB_IGAM
            x_flat = self.encoder_layers[i_stage](x_flat, x_size, dark_mask)
            
            # Reshape back to 2D
            x_enc = x_flat.transpose(1, 2).view(-1, x_flat.shape[-1], curr_h, curr_w)
            
            # Store for skip connection
            encoder_features.append(x_enc)
            
            # Downsample (except last stage)
            if i_stage < self.num_stages - 1:
                x_enc = self.downsample_layers[i_stage](x_enc)
        
        # ========================
        # Step 4: Decoder forward pass with skip connections
        # ========================
        x_dec = encoder_features[-1]  # Start from bottleneck
        
        for i_dec, i_enc in enumerate(range(self.num_stages - 2, -1, -1)):
            # Upsample
            x_dec = self.upsample_layers[i_dec](x_dec)
            
            # Fuse with encoder features (skip connection)
            x_dec = self.fusion_layers[i_dec](encoder_features[i_enc], x_dec)
            
            # Get current resolution
            curr_h, curr_w = x_dec.shape[2:]
            x_size = (curr_h, curr_w)
            
            # Flatten for transformer
            x_flat = x_dec.flatten(2).transpose(1, 2)
            
            # Process through decoder RSTB_IGAM
            x_flat = self.decoder_layers[i_dec](x_flat, x_size, dark_mask)
            
            # Reshape back to 2D
            x_dec = x_flat.transpose(1, 2).view(-1, x_flat.shape[-1], curr_h, curr_w)
        
        # ========================
        # Step 5: Output reconstruction
        # ========================
        x_out = self.conv_after_body(x_dec) + x_shallow
        x_out = self.conv_last(x_out)
        
        # Add residual connection (low-light input + enhancement)
        x_out = x_out + x
        
        # De-normalize
        x_out = x_out / self.img_range + self.mean
        
        # Crop to original size
        return x_out[:, :, :H, :W]
    
    def get_illumination_map(self, x):
        """
        Get the estimated illumination map for visualization.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            illum_map: Illumination map (B, 1, H, W)
            dark_mask: Dark region mask (B, 1, H, W)
        """
        return self.illumination_estimator(x)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Swin-LLIE Model")
    print("=" * 60)
    
    # Create model
    model = SwinLLIE(
        img_size=128,
        embed_dim=60,
        depths=[4, 4, 4],
        num_heads=[6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        use_igam=True
    )
    
    print(f"\n✓ Model created successfully!")
    print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 128, 128)
    print(f"\n✓ Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"✓ Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch!"
    print(f"\n✓ Forward pass successful!")
    
    # Test illumination map extraction
    illum_map, dark_mask = model.get_illumination_map(x)
    print(f"\n✓ Illumination map shape: {illum_map.shape}")
    print(f"✓ Dark mask shape: {dark_mask.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Swin-LLIE is ready for training.")
    print("=" * 60)
