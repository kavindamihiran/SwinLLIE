# -----------------------------------------------------------------------------------
# Loss Functions for Swin-LLIE (SIMPLIFIED)
# Clean, beginner-friendly loss functions
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Basic Losses
# =============================================================================

class L1Loss(nn.Module):
    """
    L1 Loss: Average absolute difference between pixels.
    
    Use this as your main reconstruction loss.
    Lower is better: 0 = perfect match.
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """
    L2 Loss: Average squared difference (MSE).
    
    Penalizes large errors more than L1.
    Use for smoother results.
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)


# =============================================================================
# Perceptual Loss
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss: Compare high-level features, not just pixels.
    
    Why it matters:
        - L1/L2 alone → blurry results
        - VGG features → realistic textures preserved
    
    Uses pre-trained VGG19 to extract features at multiple depths.
    """
    
    def __init__(self, layers=[3, 8, 15, 22]):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.vgg_blocks = nn.ModuleList()
        prev = 0
        for idx in layers:
            self.vgg_blocks.append(vgg[prev:idx + 1])
            prev = idx + 1
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        loss = 0.0
        p, t = pred, target
        for block in self.vgg_blocks:
            p = block(p)
            t = block(t)
            loss += F.mse_loss(p, t)
        
        return loss


# =============================================================================
# Color Consistency Loss
# =============================================================================

class ColorConsistencyLoss(nn.Module):
    """
    Color Loss: Ensure colors match using cosine similarity.
    
    Why it matters:
        - Dark images often get washed out (grayish)
        - This loss preserves color hues
    
    Measures angle between RGB vectors, ignoring brightness.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Flatten: (B, 3, H, W) -> (B, 3, H*W)
        pred_flat = pred.view(pred.size(0), 3, -1)
        target_flat = target.view(target.size(0), 3, -1)
        
        # Normalize to unit vectors
        pred_norm = F.normalize(pred_flat, dim=1)
        target_norm = F.normalize(target_flat, dim=1)
        
        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (B, H*W)
        
        # Loss = 1 - average similarity
        return 1.0 - cos_sim.mean()


# =============================================================================
# Edge/Sharpness Loss
# =============================================================================

class EdgeLoss(nn.Module):
    """
    Edge Loss: Preserve edges and sharpness.
    
    Why it matters:
        - Enhancement often blurs edges
        - This ensures edges in output match edges in target
    
    Uses Sobel filters to detect edges.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel edge detection filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def get_edges(self, x):
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    
    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return F.l1_loss(pred_edges, target_edges)


# =============================================================================
# Illumination Smoothness Loss
# =============================================================================

class IlluminationSmoothnessLoss(nn.Module):
    """
    [DEPRECATED - Not used in pure SwinIR implementation]
    
    Total Variation (TV) regularization loss.
    Originally used for smoothness regularization.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, illum_map):
        # Gradients in x and y directions
        grad_x = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1])
        grad_y = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :])
        return grad_x.mean() + grad_y.mean()


# =============================================================================
# Exposure Control Loss
# =============================================================================

class ExposureControlLoss(nn.Module):
    """
    Exposure Loss: Prevent overexposure in bright regions.
    
    Why it matters:
        - Bright regions can get "blown out" (pure white)
        - This protects already-bright regions
    
    Two components:
        1. Penalize pixels above threshold
        2. Match bright regions to target
    """
    
    def __init__(self, threshold=0.95, weight_overexpose=2.0, weight_preserve=1.5):
        super().__init__()
        self.threshold = threshold
        self.w_over = weight_overexpose
        self.w_preserve = weight_preserve
    
    def forward(self, pred, target, bright_mask=None):
        loss = 0.0
        
        # 1. Penalize overexposed pixels
        overexposed = F.relu(pred - self.threshold)
        loss += self.w_over * (overexposed ** 2).mean()
        
        # 2. Preserve bright regions (if mask provided)
        if bright_mask is None:
            # Auto-detect from target
            brightness = torch.max(target, dim=1, keepdim=True)[0]
            bright_mask = (brightness > 0.7).float()
        else:
            if bright_mask.shape[2:] != pred.shape[2:]:
                bright_mask = F.interpolate(bright_mask, size=pred.shape[2:], mode='bilinear', align_corners=False)
        
        if bright_mask.sum() > 0:
            bright_mask_exp = bright_mask.expand_as(pred)
            diff = torch.abs(pred - target) * bright_mask_exp
            loss += self.w_preserve * diff.sum() / (bright_mask_exp.sum() + 1e-8)
        
        return loss


# =============================================================================
# SSIM Loss (Optional)
# =============================================================================

class SSIMLoss(nn.Module):
    """
    SSIM Loss: Structural Similarity Index.
    
    Measures how similar two images are structurally.
    More aligned with human perception than L1/L2.
    
    Loss = 1 - SSIM (so lower is better)
    """
    
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        
        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        self.register_buffer('window', window.expand(3, 1, window_size, window_size).contiguous())
    
    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        
        mu_p = F.conv2d(pred, self.window, padding=pad, groups=3)
        mu_t = F.conv2d(target, self.window, padding=pad, groups=3)
        
        sigma_p = F.conv2d(pred ** 2, self.window, padding=pad, groups=3) - mu_p ** 2
        sigma_t = F.conv2d(target ** 2, self.window, padding=pad, groups=3) - mu_t ** 2
        sigma_pt = F.conv2d(pred * target, self.window, padding=pad, groups=3) - mu_p * mu_t
        
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        
        return 1.0 - ssim.mean()


# =============================================================================
# Combined Hybrid Loss (SIMPLIFIED)
# =============================================================================

class HybridLoss(nn.Module):
    """
    Combined Loss for Swin-LLIE Training.
    
    Combines 5 loss functions:
        1. L1: Main reconstruction (weight: 1.0)
        2. VGG: Perceptual quality (weight: 0.1)
        3. Color: Color preservation (weight: 0.5)
        4. Edge: Sharpness (weight: 0.5)
        5. Exposure: Prevent overexposure (weight: 0.5)
    
    Optional:
        - SSIM: Structural similarity
    
    Usage:
        loss_fn = HybridLoss()
        loss, breakdown = loss_fn(pred, target)
    """
    
    def __init__(self, 
                 lambda_l1=1.0, 
                 lambda_vgg=0.1, 
                 lambda_color=0.5,
                 lambda_edge=0.5,
                 lambda_exposure=0.5,
                 lambda_smooth=0.01,
                 use_ssim=False, 
                 lambda_ssim=0.1):
        super().__init__()
        
        # Store weights
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_color = lambda_color
        self.lambda_edge = lambda_edge
        self.lambda_exposure = lambda_exposure
        self.lambda_smooth = lambda_smooth
        self.use_ssim = use_ssim
        self.lambda_ssim = lambda_ssim
        
        # Initialize losses
        self.l1_loss = L1Loss()
        self.vgg_loss = VGGPerceptualLoss()
        self.color_loss = ColorConsistencyLoss()
        self.edge_loss = EdgeLoss()
        self.exposure_loss = ExposureControlLoss()
        self.smooth_loss = IlluminationSmoothnessLoss()
        
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target, illum_map=None, bright_mask=None):
        """
        Compute combined loss.
        
        Args:
            pred: Enhanced image (B, 3, H, W) in [0, 1]
            target: Ground truth (B, 3, H, W) in [0, 1]
            illum_map: Optional illumination map for smoothness (deprecated)
            bright_mask: Optional bright mask for exposure control (deprecated)
        
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Breakdown of individual losses
        """
        loss_dict = {}
        total = 0.0
        
        # 1. L1 Reconstruction
        l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = l1.item()
        total += self.lambda_l1 * l1
        
        # 2. VGG Perceptual
        if self.lambda_vgg > 0:
            vgg = self.vgg_loss(pred, target)
            loss_dict['vgg'] = vgg.item()
            total += self.lambda_vgg * vgg
        
        # 3. Color Consistency
        if self.lambda_color > 0:
            color = self.color_loss(pred, target)
            loss_dict['color'] = color.item()
            total += self.lambda_color * color
        
        # 4. Edge Sharpness
        if self.lambda_edge > 0:
            edge = self.edge_loss(pred, target)
            loss_dict['edge'] = edge.item()
            total += self.lambda_edge * edge
        
        # 5. Exposure Control (simplified - without bright mask)
        if self.lambda_exposure > 0:
            exposure = self.exposure_loss(pred, target)
            loss_dict['exposure'] = exposure.item()
            total += self.lambda_exposure * exposure
        
        # 6. SSIM (optional)
        if self.use_ssim:
            ssim = self.ssim_loss(pred, target)
            loss_dict['ssim'] = ssim.item()
            total += self.lambda_ssim * ssim
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Keep old names working
EdgeSharpnessLoss = EdgeLoss
HighFrequencyLoss = EdgeLoss  # Removed, but alias to Edge


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Simplified Loss Functions")
    print("=" * 60)
    
    # Create test tensors
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    illum = torch.rand(2, 1, 64, 64)
    bright = torch.rand(2, 1, 64, 64)
    
    # Test individual losses
    print("\n1. L1 Loss:", L1Loss()(pred, target).item())
    print("2. VGG Loss:", VGGPerceptualLoss()(pred, target).item())
    print("3. Color Loss:", ColorConsistencyLoss()(pred, target).item())
    print("4. Edge Loss:", EdgeLoss()(pred, target).item())
    print("5. Exposure Loss:", ExposureControlLoss()(pred, target, bright).item())
    print("6. Smooth Loss:", IlluminationSmoothnessLoss()(illum).item())
    print("7. SSIM Loss:", SSIMLoss()(pred, target).item())
    
    # Test hybrid loss
    print("\n8. Hybrid Loss:")
    hybrid = HybridLoss(use_ssim=True)
    total, breakdown = hybrid(pred, target, illum, bright)
    print(f"   Total: {total:.4f}")
    for k, v in breakdown.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("All loss tests passed!")
    print("=" * 60)
