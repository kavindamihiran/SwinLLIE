# -----------------------------------------------------------------------------------
# Swin-LLIE: Simplified Low-Light Image Enhancement
# Clean, beginner-friendly architecture with simple attention mechanism
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_


# =============================================================================
# Helper Functions
# =============================================================================

def window_partition(x, window_size):
    """
    Split image into non-overlapping windows.
    
    Args:
        x: (B, H, W, C) Input tensor
        window_size: Size of each window
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Merge windows back into image.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Image dimensions
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Illumination Estimation Module (SIMPLIFIED)
# =============================================================================

class IlluminationEstimator(nn.Module):
    """
    Simple CNN to estimate how dark each region is.
    
    Input: RGB image (B, 3, H, W)
    Output: 
        - illum_map: brightness of each pixel (0=dark, 1=bright)
        - dark_mask: inverse (1=dark, 0=bright) - where to enhance
    
    Why we need this:
        - Dark regions need more enhancement
        - Bright regions should stay the same (avoid overexposure)
    """
    
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        
        # Simple 3-layer CNN
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W) in [0, 1]
        
        Returns:
            illum_map: (B, 1, H, W) brightness level
            dark_mask: (B, 1, H, W) darkness level (for attention)
            bright_mask: (B, 1, H, W) bright regions (to protect)
        """
        # Get rough brightness from max RGB channel
        rough_bright = torch.max(x, dim=1, keepdim=True)[0]
        
        # Refine with learned network
        refined = self.net(x)
        
        # Combine: 40% rough + 60% learned
        illum_map = 0.4 * rough_bright + 0.6 * refined
        
        # Dark mask: invert (dark=1, bright=0)
        dark_mask = 1.0 - illum_map
        
        # Bright mask: regions above threshold need protection
        bright_mask = torch.clamp((illum_map - 0.6) / 0.4, 0.0, 1.0)
        
        return illum_map, dark_mask, bright_mask


# =============================================================================
# Simple Illumination-Guided Attention (SIMPLIFIED)
# =============================================================================

class SimpleIllumAttention(nn.Module):
    """
    Simple attention that enhances dark regions more than bright regions.
    
    How it works:
        1. Channel attention: Learn which channels are important
        2. Spatial modulation: Apply more enhancement where dark_mask is high
        3. Residual: Blend with original features
    
    This replaces the complex multi-branch attention with a clean design.
    """
    
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        
        # Channel attention (Squeeze-and-Excitation style)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial modulation: combine features with dark mask
        self.spatial_mod = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
        # Learnable blend weight (starts at 0 = use original)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features, dark_mask, bright_mask=None):
        """
        Args:
            features: (B, C, H, W) feature tensor
            dark_mask: (B, 1, H, W) where to enhance (1=dark)
            bright_mask: (B, 1, H, W) where to protect (1=bright)
        
        Returns:
            modulated: (B, C, H, W) enhanced features
        """
        B, C, H, W = features.shape
        
        # Resize mask to match features
        if dark_mask.shape[2:] != (H, W):
            dark_mask = F.interpolate(dark_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # 1. Channel attention
        ch_att = self.channel_att(features)  # (B, C, 1, 1)
        
        # 2. Spatial modulation guided by dark mask
        combined = torch.cat([features, dark_mask], dim=1)  # (B, C+1, H, W)
        spatial = self.spatial_mod(combined)  # (B, C, H, W)
        
        # 3. Apply channel attention to spatial features
        enhanced = spatial * ch_att
        
        # 4. Weight by dark mask (more enhancement in dark regions)
        enhanced = enhanced * (0.5 + 0.5 * dark_mask)
        
        # 5. Protect bright regions if mask provided
        if bright_mask is not None:
            if bright_mask.shape[2:] != (H, W):
                bright_mask = F.interpolate(bright_mask, size=(H, W), mode='bilinear', align_corners=False)
            # Reduce enhancement in bright regions
            enhanced = enhanced * (1.0 - 0.7 * bright_mask)
        
        # 6. Residual blend with learnable weight
        output = features + self.gamma * enhanced
        
        return output


# =============================================================================
# Standard Swin Transformer Components
# =============================================================================

class Mlp(nn.Module):
    """Simple 2-layer MLP with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
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
    Window-based Multi-head Self Attention.
    
    This is the core attention mechanism from Swin Transformer.
    It computes attention within local windows for efficiency.
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

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
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with window attention and shifted window attention.
    
    Two types alternate:
        - W-MSA: Regular window attention
        - SW-MSA: Shifted window attention (for cross-window connections)
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # Pre-compute attention mask for shifted windows
        if self.shift_size > 0:
            attn_mask = self._compute_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _compute_mask(self, x_size):
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

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self._compute_mask(x_size).to(x.device))

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A stack of Swin Transformer blocks for one stage."""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks with alternating shift
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        return x


# =============================================================================
# Patch Embedding / Unembedding
# =============================================================================

class PatchEmbed(nn.Module):
    """Flatten image to sequence for Transformer."""
    
    def __init__(self, img_size=224, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Reshape sequence back to image."""
    
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


# =============================================================================
# RSTB with Illumination Attention (SIMPLIFIED)
# =============================================================================

class RSTB(nn.Module):
    """
    Residual Swin Transformer Block with optional illumination attention.
    
    Structure: Input -> Swin Blocks -> Conv -> (Illum Attention) -> + Input
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_size=224, patch_size=1, use_illum_att=True):
        super().__init__()
        
        self.dim = dim
        self.use_illum_att = use_illum_att

        # Swin Transformer blocks
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint)

        # Residual conv
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # Patch embed/unembed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                       embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

        # Illumination attention (optional)
        if use_illum_att:
            self.illum_att = SimpleIllumAttention(dim)

    def forward(self, x, x_size, dark_mask=None, bright_mask=None):
        # Swin processing
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        residual = self.conv(residual)
        
        # Apply illumination attention if enabled and mask provided
        if self.use_illum_att and dark_mask is not None:
            residual = self.illum_att(residual, dark_mask, bright_mask)
        
        residual = self.patch_embed.forward(residual)
        
        return residual + x


# =============================================================================
# Cross-Stage Feature Fusion (SIMPLIFIED)
# =============================================================================

class FeatureFusion(nn.Module):
    """Simple feature fusion for skip connections."""
    
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, enc_feat, dec_feat):
        # Match spatial sizes
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=False)
        
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.gate(concat)
        fused = self.conv(concat)
        
        return gate * enc_feat + (1 - gate) * dec_feat + fused


# =============================================================================
# MAIN MODEL: SwinLLIE (SIMPLIFIED)
# =============================================================================

class SwinLLIE(nn.Module):
    """
    Swin-LLIE: Simplified Low-Light Image Enhancement
    
    Architecture:
        1. Estimate illumination (which regions are dark)
        2. Extract features with Swin Transformer
        3. Apply more enhancement to dark regions
        4. Reconstruct enhanced image
    
    Args:
        img_size: Input image size (default: 128)
        embed_dim: Feature dimension (default: 60)
        depths: Blocks per stage (default: [4, 4, 4])
        num_heads: Attention heads per stage (default: [6, 6, 6])
        window_size: Attention window size (default: 8)
        use_igam: Use illumination-guided attention (default: True)
    """
    
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[4, 4, 4], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_range=1., resi_connection='1conv', use_igam=True, **kwargs):
        super().__init__()
        
        self.img_range = img_range
        self.window_size = window_size
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.use_igam = use_igam
        
        # Image normalization
        if in_chans == 3:
            self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        # Step 1: Illumination Estimation
        self.illum_estimator = IlluminationEstimator(in_channels=in_chans)
        
        # Step 2: Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                       embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Step 3: Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.num_stages):
            resolution = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
            dim = embed_dim * (2 ** i)
            
            layer = RSTB(
                dim=dim,
                input_resolution=resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i),
                patch_size=patch_size,
                use_illum_att=use_igam)
            self.encoder_layers.append(layer)
            
            if i < self.num_stages - 1:
                down = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
                self.downsample_layers.append(down)
        
        # Step 4: Decoder with skip connections
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(self.num_stages - 2, -1, -1):
            resolution = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
            dim = embed_dim * (2 ** i)
            higher_dim = dim * 2
            
            up = nn.Sequential(
                nn.ConvTranspose2d(higher_dim, dim, kernel_size=2, stride=2),
                nn.ReLU(inplace=True))
            self.upsample_layers.append(up)
            
            fusion = FeatureFusion(dim)
            self.fusion_layers.append(fusion)
            
            layer = RSTB(
                dim=dim,
                input_resolution=resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i),
                patch_size=patch_size,
                use_illum_att=use_igam)
            self.decoder_layers.append(layer)
        
        # Step 5: Output
        self.norm = norm_layer(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def check_image_size(self, x):
        """Pad image to be divisible by window size at all scales."""
        _, _, h, w = x.size()
        scale = 2 ** (self.num_stages - 1)
        min_size = self.window_size * scale
        
        pad_h = (min_size - h % min_size) % min_size
        pad_w = (min_size - w % min_size) % min_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        return x
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Low-light image (B, 3, H, W) in [0, 1]
        
        Returns:
            Enhanced image (B, 3, H, W)
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        # Normalize
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # Step 1: Estimate illumination
        illum, dark_mask, bright_mask = self.illum_estimator((x / self.img_range) + self.mean)
        
        # Step 2: Shallow features
        x_shallow = self.conv_first(x)
        
        # Step 3: Encoder
        encoder_features = []
        x_enc = x_shallow
        
        for i in range(self.num_stages):
            h, w = x_enc.shape[2:]
            x_flat = self.patch_embed.forward(x_enc) if i == 0 else x_enc.flatten(2).transpose(1, 2)
            x_flat = self.pos_drop(x_flat)
            x_flat = self.encoder_layers[i](x_flat, (h, w), dark_mask, bright_mask)
            x_enc = x_flat.transpose(1, 2).view(-1, x_flat.shape[-1], h, w)
            encoder_features.append(x_enc)
            
            if i < self.num_stages - 1:
                x_enc = self.downsample_layers[i](x_enc)
        
        # Step 4: Decoder
        x_dec = encoder_features[-1]
        
        for i, enc_idx in enumerate(range(self.num_stages - 2, -1, -1)):
            x_dec = self.upsample_layers[i](x_dec)
            x_dec = self.fusion_layers[i](encoder_features[enc_idx], x_dec)
            
            h, w = x_dec.shape[2:]
            x_flat = x_dec.flatten(2).transpose(1, 2)
            x_flat = self.decoder_layers[i](x_flat, (h, w), dark_mask, bright_mask)
            x_dec = x_flat.transpose(1, 2).view(-1, x_flat.shape[-1], h, w)
        
        # Step 5: Output
        x_out = self.conv_after(x_dec) + x_shallow
        x_out = self.conv_last(x_out)
        x_out = x_out + x  # Residual
        
        # Denormalize
        x_out = x_out / self.img_range + self.mean
        
        return x_out[:, :, :H, :W]
    
    def get_illumination_map(self, x):
        """Get illumination maps for visualization."""
        return self.illum_estimator(x)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Keep old class names working
IlluminationEstimationModule = IlluminationEstimator
IlluminationGuidedAttention = SimpleIllumAttention
RSTB_IGAM = RSTB


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Simplified Swin-LLIE Model")
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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 128, 128)
    print(f"\n✓ Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"✓ Output shape: {y.shape}")
    assert y.shape == x.shape
    
    # Test illumination extraction
    illum, dark, bright = model.get_illumination_map(x)
    print(f"\n✓ Illumination map: {illum.shape}")
    print(f"✓ Dark mask: {dark.shape}")
    print(f"✓ Bright mask: {bright.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Model is ready.")
    print("=" * 60)
