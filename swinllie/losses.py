# -----------------------------------------------------------------------------------
# Loss Functions for Swin-LLIE: Low-Light Image Enhancement
# Hybrid loss combining reconstruction, perceptual, and color consistency losses
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class L1Loss(nn.Module):
    """
    Standard L1 (Mean Absolute Error) Loss.
    
    Measures pixel-wise difference between prediction and ground truth.
    Good for maintaining overall brightness and structure.
    
    L1 = mean(|prediction - target|)
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted enhanced image (B, C, H, W)
            target: Ground truth normal-light image (B, C, H, W)
        """
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """
    Standard L2 (Mean Squared Error) Loss.
    
    Penalizes large errors more heavily than L1.
    Use when you want smoother results.
    
    L2 = mean((prediction - target)^2)
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss for realistic texture preservation.
    
    Uses pre-trained VGG19 to compare high-level features between
    prediction and target. This ensures the enhanced image looks
    realistic and maintains proper textures.
    
    Why it's important for low-light enhancement:
        - Prevents over-smoothing (common problem with L1/L2 alone)
        - Maintains realistic textures in dark regions
        - Keeps fine details like hair, fabric patterns, etc.
    
    L_perceptual = sum_l ||phi_l(pred) - phi_l(target)||^2
    
    where phi_l is the feature map from VGG layer l.
    
    Args:
        feature_layers: Which VGG layers to use (default: [3, 8, 15, 22])
        use_input_norm: Whether to normalize inputs (default: True)
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], use_input_norm=True):
        super().__init__()
        
        # Load pre-trained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Freeze VGG weights - we don't want to train it
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract specific layers for perceptual comparison
        # Layer indices map to different levels of features:
        # - Early layers (3, 8): edges, colors
        # - Middle layers (15): textures
        # - Deep layers (22): semantic content
        self.feature_layers = feature_layers
        self.vgg_layers = nn.ModuleList()
        
        prev_layer = 0
        for layer_idx in feature_layers:
            self.vgg_layers.append(vgg[prev_layer:layer_idx + 1])
            prev_layer = layer_idx + 1
        
        # ImageNet normalization (VGG expects this)
        self.use_input_norm = use_input_norm
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input to ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]
        
        Returns:
            Perceptual loss (scalar)
        """
        if self.use_input_norm:
            pred = self.normalize(pred)
            target = self.normalize(target)
        
        loss = 0.0
        pred_feat = pred
        target_feat = target
        
        for vgg_block in self.vgg_layers:
            pred_feat = vgg_block(pred_feat)
            target_feat = vgg_block(target_feat)
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss


class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss using Cosine Similarity.
    
    CRITICAL for low-light enhancement!
    
    Problem: Swin Transformers often desaturate colors in low-light images,
    producing grayish outputs. This loss forces the model to preserve
    the color relationship between prediction and target.
    
    Mathematical formulation:
        L_color = 1 - cos_similarity(pred, target)
    
    where cos_similarity is computed per-pixel across RGB channels.
    
    This measures the angle between RGB vectors, ignoring intensity.
    So even if brightness differs, color hue should match.
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
        
        Returns:
            Color consistency loss (scalar)
        """
        # Flatten spatial dimensions: (B, 3, H, W) -> (B, 3, H*W)
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Compute cosine similarity along channel dimension
        # cos_sim = (pred · target) / (||pred|| * ||target||)
        pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + self.eps)
        target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + self.eps)
        
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (B, H*W)
        
        # Loss = 1 - mean(cos_similarity)
        loss = 1.0 - cos_sim.mean()
        
        return loss


class IlluminationSmoothnessLoss(nn.Module):
    """
    Total Variation Loss for Illumination Map Smoothness.
    
    The illumination map should be smooth (no sharp edges within
    regions of similar lighting). This regularization prevents
    the illumination estimator from learning noisy patterns.
    
    L_smooth = ||∇x(M)||_1 + ||∇y(M)||_1
    
    where ∇x and ∇y are horizontal and vertical gradients.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, illum_map):
        """
        Args:
            illum_map: Illumination map (B, 1, H, W)
        
        Returns:
            Smoothness loss (scalar)
        """
        # Compute gradients
        grad_x = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1])
        grad_y = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :])
        
        return grad_x.mean() + grad_y.mean()


class EdgeSharpnessLoss(nn.Module):
    """
    Edge/Sharpness Preservation Loss using Sobel operators.
    
    CRITICAL for preventing blurry outputs!
    
    This loss ensures that edges in the enhanced image match the edges
    in the ground truth. It uses Sobel filters to extract edge information
    and penalizes differences.
    
    Mathematical formulation:
        L_edge = ||Sobel(pred) - Sobel(target)||_1
    
    Also includes a Laplacian sharpness component to ensure overall sharpness.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Laplacian filter for sharpness
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        
        # Register as buffers (will be moved to correct device automatically)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def get_edges(self, x):
        """Extract edges using Sobel operators."""
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edges
    
    def get_laplacian(self, x):
        """Extract Laplacian (second derivative) for sharpness."""
        return F.conv2d(x, self.laplacian, padding=1, groups=3)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
        
        Returns:
            Edge sharpness loss (scalar)
        """
        # Edge preservation loss
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # Laplacian sharpness loss - penalize blurry outputs
        pred_lap = self.get_laplacian(pred)
        target_lap = self.get_laplacian(target)
        sharpness_loss = F.l1_loss(pred_lap, target_lap)
        
        # Combined: edge matching + sharpness preservation
        return edge_loss + 0.5 * sharpness_loss


class HighFrequencyLoss(nn.Module):
    """
    High-Frequency Detail Preservation Loss.
    
    Uses Gaussian blur to separate low and high frequency components,
    then ensures high-frequency details (textures, edges) are preserved.
    
    L_hf = ||HF(pred) - HF(target)||_1
    
    where HF(x) = x - Blur(x) extracts high-frequency components.
    """
    
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # 2D kernel
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        
        self.register_buffer('kernel', kernel)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def get_low_freq(self, x):
        """Extract low-frequency component using Gaussian blur."""
        return F.conv2d(x, self.kernel, padding=self.padding, groups=3)
    
    def get_high_freq(self, x):
        """Extract high-frequency component."""
        return x - self.get_low_freq(x)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
        
        Returns:
            High-frequency loss (scalar)
        """
        pred_hf = self.get_high_freq(pred)
        target_hf = self.get_high_freq(target)
        
        return F.l1_loss(pred_hf, target_hf)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    
    SSIM measures structural similarity between images, which is more
    aligned with human perception than pixel-wise losses.
    
    Components:
        - Luminance comparison
        - Contrast comparison
        - Structure comparison
    
    L_SSIM = 1 - SSIM(pred, target)
    
    Args:
        window_size: Size of Gaussian window (default: 11)
        sigma: Gaussian standard deviation (default: 1.5)
    """
    
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = 3
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window())
    
    def _create_window(self):
        """Create 2D Gaussian window."""
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - self.window_size // 2) ** 2 / (2.0 * self.sigma ** 2)))
            for x in range(self.window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # 2D window
        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window_2d.expand(self.channels, 1, self.window_size, self.window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: Images (B, 3, H, W) in range [0, 1]
        
        Returns:
            SSIM loss (1 - SSIM)
        """
        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute local means
        mu_pred = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channels)
        mu_target = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channels)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute local variances and covariance
        sigma_pred_sq = F.conv2d(pred ** 2, self.window, padding=self.window_size // 2, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, self.window, padding=self.window_size // 2, groups=self.channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=self.channels) - mu_pred_target
        
        # SSIM formula
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1.0 - ssim.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss for Swin-LLIE Training.
    
    Combines multiple loss functions for optimal low-light enhancement:
    
    L_total = λ1 * L_L1 + λ2 * L_VGG + λ3 * L_Color + λ4 * L_Smooth + λ5 * L_Edge + λ6 * L_HF
    
    Default weights (empirically determined):
        - λ1 = 1.0   (L1: main reconstruction)
        - λ2 = 0.1   (VGG: perceptual quality)
        - λ3 = 0.5   (Color: color preservation)
        - λ4 = 0.01  (Smooth: illumination regularization)
        - λ5 = 1.0   (Edge: edge/sharpness preservation) - NEW!
        - λ6 = 0.5   (HF: high-frequency detail preservation) - NEW!
    
    Usage:
        loss_fn = HybridLoss()
        loss = loss_fn(pred_image, target_image, illumination_map)
    
    Args:
        lambda_l1: Weight for L1 loss
        lambda_vgg: Weight for VGG perceptual loss
        lambda_color: Weight for color consistency loss
        lambda_smooth: Weight for illumination smoothness loss
        lambda_edge: Weight for edge sharpness loss (NEW)
        lambda_hf: Weight for high-frequency loss (NEW)
        use_ssim: Whether to include SSIM loss (optional)
        lambda_ssim: Weight for SSIM loss
    """
    
    def __init__(self, lambda_l1=1.0, lambda_vgg=0.1, lambda_color=0.5, 
                 lambda_smooth=0.01, lambda_edge=1.0, lambda_hf=0.5,
                 use_ssim=False, lambda_ssim=0.1):
        super().__init__()
        
        # Store weights
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_color = lambda_color
        self.lambda_smooth = lambda_smooth
        self.lambda_edge = lambda_edge
        self.lambda_hf = lambda_hf
        self.use_ssim = use_ssim
        self.lambda_ssim = lambda_ssim
        
        # Initialize individual losses
        self.l1_loss = L1Loss()
        self.vgg_loss = VGGPerceptualLoss()
        self.color_loss = ColorConsistencyLoss()
        self.smooth_loss = IlluminationSmoothnessLoss()
        self.edge_loss = EdgeSharpnessLoss()  # NEW for sharpness
        self.hf_loss = HighFrequencyLoss()    # NEW for detail preservation
        
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target, illum_map=None):
        """
        Compute hybrid loss.
        
        Args:
            pred: Predicted enhanced image (B, 3, H, W) in [0, 1]
            target: Ground truth normal-light image (B, 3, H, W) in [0, 1]
            illum_map: Optional illumination map for smoothness regularization
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components (for logging)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1 Reconstruction Loss
        l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = l1.item()
        total_loss += self.lambda_l1 * l1
        
        # VGG Perceptual Loss
        if self.lambda_vgg > 0:
            vgg = self.vgg_loss(pred, target)
            loss_dict['vgg'] = vgg.item()
            total_loss += self.lambda_vgg * vgg
        
        # Color Consistency Loss
        if self.lambda_color > 0:
            color = self.color_loss(pred, target)
            loss_dict['color'] = color.item()
            total_loss += self.lambda_color * color
        
        # Illumination Smoothness Loss
        if illum_map is not None and self.lambda_smooth > 0:
            smooth = self.smooth_loss(illum_map)
            loss_dict['smooth'] = smooth.item()
            total_loss += self.lambda_smooth * smooth
        
        # Edge Sharpness Loss (NEW - critical for sharp images)
        if self.lambda_edge > 0:
            edge = self.edge_loss(pred, target)
            loss_dict['edge'] = edge.item()
            total_loss += self.lambda_edge * edge
        
        # High-Frequency Detail Loss (NEW - preserves textures)
        if self.lambda_hf > 0:
            hf = self.hf_loss(pred, target)
            loss_dict['hf'] = hf.item()
            total_loss += self.lambda_hf * hf
        
        # Optional SSIM Loss
        if self.use_ssim:
            ssim = self.ssim_loss(pred, target)
            loss_dict['ssim'] = ssim.item()
            total_loss += self.lambda_ssim * ssim
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    pred = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)
    illum_map = torch.rand(2, 1, 128, 128)
    
    # Test individual losses
    print("\n1. Testing L1 Loss...")
    l1_loss = L1Loss()
    print(f"   L1 Loss: {l1_loss(pred, target):.4f}")
    
    print("\n2. Testing VGG Perceptual Loss...")
    vgg_loss = VGGPerceptualLoss()
    print(f"   VGG Loss: {vgg_loss(pred, target):.4f}")
    
    print("\n3. Testing Color Consistency Loss...")
    color_loss = ColorConsistencyLoss()
    print(f"   Color Loss: {color_loss(pred, target):.4f}")
    
    print("\n4. Testing Illumination Smoothness Loss...")
    smooth_loss = IlluminationSmoothnessLoss()
    print(f"   Smooth Loss: {smooth_loss(illum_map):.4f}")
    
    print("\n5. Testing Edge Sharpness Loss...")
    edge_loss = EdgeSharpnessLoss()
    print(f"   Edge Loss: {edge_loss(pred, target):.4f}")
    
    print("\n6. Testing High-Frequency Loss...")
    hf_loss = HighFrequencyLoss()
    print(f"   HF Loss: {hf_loss(pred, target):.4f}")
    
    print("\n7. Testing SSIM Loss...")
    ssim_loss = SSIMLoss()
    print(f"   SSIM Loss: {ssim_loss(pred, target):.4f}")
    
    print("\n8. Testing Hybrid Loss (with new sharpness losses)...")
    hybrid_loss = HybridLoss(use_ssim=True)
    total, loss_dict = hybrid_loss(pred, target, illum_map)
    print(f"   Total Loss: {total:.4f}")
    print(f"   Loss breakdown: {loss_dict}")
    
    print("\n" + "=" * 60)
    print("All loss function tests passed!")
    print("=" * 60)
