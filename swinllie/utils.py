# -----------------------------------------------------------------------------------
# Utility Functions for Swin-LLIE: Low-Light Image Enhancement
# -----------------------------------------------------------------------------------

import os
import random
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def calculate_psnr(img1, img2, crop_border=0):
    """
    Calculate PSNR between two images (industry standard).
    
    Uses scikit-image's peak_signal_noise_ratio which follows the standard formula:
    PSNR = 10 * log10(data_range^2 / MSE)
    
    Args:
        img1, img2: Images as numpy arrays (H, W, C) in range [0, 1] or [0, 255]
        crop_border: Pixels to crop from border before calculation
    
    Returns:
        PSNR value in dB
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Determine data range: if max value > 1.0, assume [0, 255] range
    # This is consistent with how most papers handle it
    data_range = 255.0 if (img1.max() > 1.0 or img2.max() > 1.0) else 1.0
    
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, crop_border=0):
    """
    Calculate SSIM between two images (industry standard).
    
    Uses scikit-image's structural_similarity with:
    - 11x11 window size (standard from original SSIM paper)
    - Gaussian weighting (as specified in Wang et al. 2004)
    - Proper data range handling
    
    Reference: Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P. (2004)
    "Image Quality Assessment: From Error Visibility to Structural Similarity"
    
    Args:
        img1, img2: Images as numpy arrays (H, W, C) in range [0, 1] or [0, 255]
        crop_border: Pixels to crop from border
    
    Returns:
        SSIM value between -1 and 1 (typically 0 to 1 for similar images)
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Consistent data range detection with PSNR
    data_range = 255.0 if (img1.max() > 1.0 or img2.max() > 1.0) else 1.0
    
    # Industry standard SSIM parameters:
    # - win_size=11: 11x11 window (from original paper)
    # - gaussian_weights=True: Gaussian weighting (from original paper)
    # - sigma=1.5: Standard deviation for Gaussian (from original paper)
    # - K1=0.01, K2=0.03: Stability constants (from original paper)
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        # For RGB images - compute SSIM per channel and average
        ssim_value = structural_similarity(
            img1, img2, 
            data_range=data_range,
            channel_axis=2,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            K1=0.01,
            K2=0.03
        )
    else:
        # For grayscale images
        if len(img1.shape) == 3:
            img1 = img1.squeeze()
            img2 = img2.squeeze()
        ssim_value = structural_similarity(
            img1, img2, 
            data_range=data_range,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            K1=0.01,
            K2=0.03
        )
    
    return ssim_value


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, save_path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_psnr': best_psnr,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_psnr', 0.0)


def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array for visualization."""
    if isinstance(tensor, torch.Tensor):
        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image in batch
        # Convert CHW to HWC
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.detach().cpu().numpy()
    
    # Clip to valid range
    tensor = np.clip(tensor, 0, 1)
    return tensor


def numpy_to_tensor(img, device='cpu'):
    """Convert numpy array to PyTorch tensor for inference."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # HWC to CHW
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    
    tensor = torch.from_numpy(img).float()
    
    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)


def create_comparison_image(low, enhanced, gt=None):
    """Create side-by-side comparison image."""
    import cv2
    
    # Convert to numpy if tensors
    if isinstance(low, torch.Tensor):
        low = tensor_to_numpy(low)
    if isinstance(enhanced, torch.Tensor):
        enhanced = tensor_to_numpy(enhanced)
    if gt is not None and isinstance(gt, torch.Tensor):
        gt = tensor_to_numpy(gt)
    
    # Ensure uint8
    low = (low * 255).astype(np.uint8)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    if gt is not None:
        gt = (gt * 255).astype(np.uint8)
        comparison = np.hstack([low, enhanced, gt])
    else:
        comparison = np.hstack([low, enhanced])
    
    return comparison
