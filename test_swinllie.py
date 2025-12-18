#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# Inference Script for Swin-LLIE: Low-Light Image Enhancement
# -----------------------------------------------------------------------------------

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network_swinllie import SwinLLIE


def calculate_psnr(img1, img2, crop_border=0):
    """Calculate PSNR between two images."""
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0):
    """Calculate SSIM between two images (simplified version)."""
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    
    C1, C2 = 6.5025, 58.5225
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def load_model(checkpoint_path, device, config=None):
    """
    Load trained Swin-LLIE model.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: torch device (cuda/cpu)
        config: Optional model configuration dict
    
    Returns:
        Loaded model in eval mode
    """
    # Default configuration (can be overridden)
    if config is None:
        config = {
            'img_size': 128,
            'patch_size': 1,
            'in_chans': 3,
            'embed_dim': 60,
            'depths': [4, 4, 4],
            'num_heads': [6, 6, 6],
            'window_size': 8,
            'mlp_ratio': 2.0,
            'use_igam': True,
        }
    
    # Create model
    model = SwinLLIE(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_chans=config['in_chans'],
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        mlp_ratio=config['mlp_ratio'],
        use_igam=config['use_igam']
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("  Using randomly initialized model (for testing only)")
    
    model = model.to(device)
    model.eval()
    
    return model


def process_image(model, img_path, device, save_path=None, compare_gt=None, max_size=512):
    """
    Enhance a single low-light image.
    
    Args:
        model: Trained SwinLLIE model
        img_path: Path to input low-light image
        device: torch device
        save_path: Optional path to save enhanced image
        compare_gt: Optional path to ground truth for metrics
        max_size: Maximum size for longest edge (images larger than this are resized)
    
    Returns:
        Enhanced image (numpy array), PSNR, SSIM (if GT provided)
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (W, H)
    
    # Resize large images to prevent GPU OOM
    was_resized = False
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.BICUBIC)
        was_resized = True
        print(f"  ↓ Resized {original_size} → {new_size} for processing")
    
    # Convert to tensor
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Pad image to be divisible by window_size * 8
    _, _, h, w = img_tensor.shape
    window_size = 8
    mod_pad = window_size * 8
    pad_h = (mod_pad - h % mod_pad) % mod_pad
    pad_w = (mod_pad - w % mod_pad) % mod_pad
    
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        enhanced = model(img_tensor)
        inference_time = time.time() - start_time
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        enhanced = enhanced[:, :, :h, :w]
    
    # Convert back to numpy
    enhanced = torch.clamp(enhanced, 0, 1)
    enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced_np = (enhanced_np * 255).astype(np.uint8)
    
    # Upscale back to original size if image was resized
    if was_resized:
        enhanced_np = cv2.resize(enhanced_np, original_size, interpolation=cv2.INTER_CUBIC)
        print(f"  ↑ Upscaled result back to {original_size}")
    
    # Calculate metrics if ground truth provided
    psnr, ssim = None, None
    if compare_gt and os.path.exists(compare_gt):
        gt = Image.open(compare_gt).convert('RGB')
        gt = gt.resize(original_size, Image.BICUBIC)
        gt_np = np.array(gt)
        
        psnr = calculate_psnr(enhanced_np, gt_np, crop_border=0)
        ssim = calculate_ssim(enhanced_np, gt_np, crop_border=0)
    
    # Save result
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, enhanced_bgr)
    
    return enhanced_np, psnr, ssim, inference_time


def process_folder(model, input_folder, output_folder, device, gt_folder=None, max_size=512):
    """
    Enhance all images in a folder.
    
    Args:
        model: Trained SwinLLIE model
        input_folder: Path to folder with low-light images
        output_folder: Path to save enhanced images
        device: torch device
        gt_folder: Optional folder with ground truth images
        max_size: Maximum size for longest edge (images larger than this are resized)
    
    Returns:
        Average PSNR, SSIM, inference time
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return None, None, None
    
    print(f"\nProcessing {len(image_files)} images...")
    print("-" * 60)
    
    psnr_list = []
    ssim_list = []
    time_list = []
    
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        save_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_enhanced.png")
        
        # Get ground truth path if available
        gt_path = None
        if gt_folder:
            for ext in valid_extensions:
                potential_gt = os.path.join(gt_folder, os.path.splitext(img_file)[0] + ext)
                if os.path.exists(potential_gt):
                    gt_path = potential_gt
                    break
        
        # Process image
        _, psnr, ssim, inf_time = process_image(model, img_path, device, save_path, gt_path, max_size)
        time_list.append(inf_time)
        
        # Print results
        if psnr is not None:
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            print(f"  {img_file}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, Time={inf_time*1000:.1f}ms")
        else:
            print(f"  {img_file}: Enhanced, Time={inf_time*1000:.1f}ms")
    
    print("-" * 60)
    
    # Summary
    avg_time = np.mean(time_list)
    if len(psnr_list) > 0:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\nAverage: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
    else:
        avg_psnr, avg_ssim = None, None
    
    print(f"Average inference time: {avg_time*1000:.1f}ms per image")
    print(f"Results saved to: {output_folder}")
    
    return avg_psnr, avg_ssim, avg_time


def create_comparison(low_path, enhanced_path, gt_path=None, save_path=None):
    """
    Create side-by-side comparison image.
    
    Args:
        low_path: Path to low-light image
        enhanced_path: Path to enhanced image
        gt_path: Optional path to ground truth
        save_path: Where to save comparison
    """
    low = cv2.imread(low_path)
    enhanced = cv2.imread(enhanced_path)
    
    # Resize to same height
    h = min(low.shape[0], enhanced.shape[0])
    low = cv2.resize(low, (int(low.shape[1] * h / low.shape[0]), h))
    enhanced = cv2.resize(enhanced, (int(enhanced.shape[1] * h / enhanced.shape[0]), h))
    
    if gt_path and os.path.exists(gt_path):
        gt = cv2.imread(gt_path)
        gt = cv2.resize(gt, (int(gt.shape[1] * h / gt.shape[0]), h))
        comparison = np.concatenate([low, enhanced, gt], axis=1)
        labels = ['Low-Light', 'Enhanced', 'Ground Truth']
    else:
        comparison = np.concatenate([low, enhanced], axis=1)
        labels = ['Low-Light', 'Enhanced']
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    w_per_img = comparison.shape[1] // len(labels)
    for i, label in enumerate(labels):
        x = i * w_per_img + 10
        cv2.putText(comparison, label, (x, 30), font, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, comparison)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Swin-LLIE Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or folder path')
    parser.add_argument('--output', type=str, default='results/swinllie',
                        help='Output folder path')
    parser.add_argument('--checkpoint', type=str, default='experiments/swinllie_lol/checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--gt_folder', type=str, default=None,
                        help='Optional ground truth folder for metrics')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save side-by-side comparisons')
    parser.add_argument('--max_size', type=int, default=512,
                        help='Max size for longest edge (larger images are resized, default: 512)')
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Swin-LLIE Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        img_name = os.path.splitext(os.path.basename(args.input))[0]
        save_path = os.path.join(args.output, f"{img_name}_enhanced.png")
        
        gt_path = None
        if args.gt_folder:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_gt = os.path.join(args.gt_folder, img_name + ext)
                if os.path.exists(potential_gt):
                    gt_path = potential_gt
                    break
        
        enhanced, psnr, ssim, inf_time = process_image(
            model, args.input, device, save_path, gt_path, args.max_size)
        
        print(f"\nInput: {args.input}")
        print(f"Output: {save_path}")
        if psnr is not None:
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")
        print(f"Inference time: {inf_time*1000:.1f} ms")
        
        # Create comparison
        if args.save_comparison:
            comp_path = os.path.join(args.output, f"{img_name}_comparison.png")
            create_comparison(args.input, save_path, gt_path, comp_path)
            print(f"Comparison: {comp_path}")
    
    elif os.path.isdir(args.input):
        # Folder of images
        process_folder(model, args.input, args.output, device, args.gt_folder, args.max_size)
    
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
