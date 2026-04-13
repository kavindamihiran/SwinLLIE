#!/usr/bin/env python3
"""
Adaptive Inference for Swin-LLIE - Prevents overexposure on brighter images.

Key idea: Use brightness-aware blending to mix the enhanced output with the 
original input. Dark regions get full enhancement, bright regions are preserved.

Usage:
    python inference_adaptive.py
    python inference_adaptive.py --blend_strength 0.7   # Less aggressive blending
    python inference_adaptive.py --blend_strength 1.0   # Full adaptive (default)
    python inference_adaptive.py --blend_strength 0.0   # No blending (original behavior)
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from swinllie import SwinLLIE

# Config
CHECKPOINT = './experiments/test_run/checkpoints/best.pth'
INPUT_DIR = './test'
OUTPUT_DIR = './test_results'
WINDOW_SIZE = 8


def compute_brightness_map(img_tensor, kernel_size=31):
    """
    Compute a smooth local brightness map from the input image.
    
    Args:
        img_tensor: (B, 3, H, W) in [0, 1]
        kernel_size: Size of the averaging kernel for smooth brightness
    
    Returns:
        brightness_map: (B, 1, H, W) in [0, 1], higher = brighter
    """
    # Convert to grayscale using luminance weights (human perception)
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img_tensor.device)
    gray = (img_tensor * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    
    # Smooth with average pooling to get local brightness
    pad = kernel_size // 2
    brightness = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(gray, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size, stride=1
    )
    
    return brightness


def adaptive_blend(input_img, enhanced_img, blend_strength=1.0, 
                   dark_threshold=0.25, bright_threshold=0.55):
    """
    Blend enhanced output with input based on local brightness.
    
    - Pixels darker than dark_threshold → use enhanced output (full enhancement)
    - Pixels brighter than bright_threshold → blend toward input (prevent overexposure)
    - Pixels in between → smooth transition
    
    Args:
        input_img: Original input (B, 3, H, W) in [0, 1]
        enhanced_img: Model output (B, 3, H, W) 
        blend_strength: 0.0 = no blending, 1.0 = full adaptive blending
        dark_threshold: Below this brightness, use full enhancement
        bright_threshold: Above this brightness, preserve original more
    
    Returns:
        blended: (B, 3, H, W) adaptively blended result
    """
    if blend_strength == 0.0:
        return enhanced_img
    
    # Get local brightness of input
    brightness = compute_brightness_map(input_img)
    
    # Create blend weight: 1.0 for dark regions (use enhanced), 0.0 for bright (use input)
    # Smooth transition between thresholds
    alpha = 1.0 - torch.clamp(
        (brightness - dark_threshold) / (bright_threshold - dark_threshold + 1e-8),
        0.0, 1.0
    )
    
    # Apply blend strength (0 = no effect, 1 = full adaptive)
    # When blend_strength=1: alpha controls fully
    # When blend_strength=0: always use enhanced (alpha=1)
    effective_alpha = 1.0 - blend_strength * (1.0 - alpha)
    
    # Blend
    blended = effective_alpha * enhanced_img + (1.0 - effective_alpha) * input_img
    
    return blended


def soft_clamp(x, max_val=1.0, knee=0.9):
    """
    Soft clamp to prevent harsh clipping at 1.0.
    Uses a smooth rolloff near the maximum value.
    
    This is better than hard clipping (np.clip) because it preserves
    relative differences in bright regions.
    """
    # For values below knee, pass through
    # For values above knee, compress toward max_val
    above_knee = torch.clamp(x - knee, min=0.0)
    compressed = knee + (max_val - knee) * torch.tanh(above_knee / (max_val - knee))
    
    result = torch.where(x <= knee, x, compressed)
    return torch.clamp(result, 0.0, max_val)


def process_image(model, x, device):
    """Process image on specified device."""
    H, W = x.shape[2], x.shape[3]
    
    # Pad to multiple of window_size * 4 (for U-Net downsampling)
    pad_unit = WINDOW_SIZE * 4
    pad_h = (pad_unit - H % pad_unit) % pad_unit
    pad_w = (pad_unit - W % pad_unit) % pad_unit
    x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Process
    output = model(x_padded)
    
    # Remove padding
    return output[:, :, :H, :W]


def try_gpu_with_fallback(model, x, ckpt):
    """Try GPU first, fall back to CPU if out of memory or incompatible."""
    
    use_cuda = False
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            compute_capability = capability[0] + capability[1] / 10
            if compute_capability >= 7.0:
                use_cuda = True
            else:
                print(f'  GPU incompatible (compute capability {compute_capability:.1f} < 7.0), using CPU')
        except:
            print('  GPU detection failed, using CPU')
    
    if not use_cuda:
        print('  Using CPU')
        model = model.to('cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        with torch.no_grad():
            return process_image(model, x.to('cpu'), 'cpu')
    
    try:
        print('  Using compatible GPU...')
        model = model.to('cuda')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        with torch.no_grad():
            output = process_image(model, x.to('cuda'), 'cuda')
        print('  GPU success!')
        return output
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('  GPU out of memory, switching to CPU...')
            torch.cuda.empty_cache()
            model = model.to('cpu')
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()
            with torch.no_grad():
                output = process_image(model, x.to('cpu'), 'cpu')
            print('  CPU success!')
            return output
        else:
            raise e


def analyze_image(img_np):
    """Print brightness analysis of input image."""
    brightness = 0.2126 * img_np[:,:,0] + 0.7152 * img_np[:,:,1] + 0.0722 * img_np[:,:,2]
    mean_b = brightness.mean()
    dark_pct = (brightness < 0.2).mean() * 100
    bright_pct = (brightness > 0.5).mean() * 100
    
    print(f'  Brightness: mean={mean_b:.3f}, dark_pixels={dark_pct:.1f}%, bright_pixels={bright_pct:.1f}%')
    
    if mean_b > 0.4:
        print(f'  ⚠️  Image is relatively bright — adaptive blending will protect highlights')
    elif mean_b < 0.15:
        print(f'  ✓  Image is very dark — full enhancement will be applied')
    else:
        print(f'  ✓  Image is moderately dark — balanced enhancement')
    
    return mean_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive Swin-LLIE Inference')
    parser.add_argument('--blend_strength', type=float, default=1.0,
                        help='Adaptive blend strength: 0.0=off, 1.0=full (default: 1.0)')
    parser.add_argument('--dark_threshold', type=float, default=0.25,
                        help='Brightness below this gets full enhancement (default: 0.25)')
    parser.add_argument('--bright_threshold', type=float, default=0.55,
                        help='Brightness above this preserves original (default: 0.55)')
    parser.add_argument('--soft_clamp', action='store_true', default=True,
                        help='Use soft clamping instead of hard clip (default: True)')
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Swin-LLIE Adaptive Inference')
    print('=' * 50)
    print(f'Blend strength: {args.blend_strength}')
    print(f'Dark threshold: {args.dark_threshold}')
    print(f'Bright threshold: {args.bright_threshold}')
    print(f'Soft clamp: {args.soft_clamp}')
    print('=' * 50)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Process images
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        print(f'\nProcessing: {fname}')
        
        # Load image
        img_path = os.path.join(args.input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        
        # To tensor
        img_np = np.array(img) / 255.0
        x = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
        
        print(f'  Size: {x.shape[3]}x{x.shape[2]}')
        mean_brightness = analyze_image(img_np)
        
        # Create fresh model
        model = SwinLLIE(
            img_size=128, embed_dim=60, depths=[4, 4, 4],
            num_heads=[6, 6, 6], window_size=WINDOW_SIZE
        )
        
        # Process with GPU/CPU fallback
        output = try_gpu_with_fallback(model, x, ckpt)
        
        # === ADAPTIVE BLENDING ===
        output_device = output.device
        x_on_device = x.to(output_device)
        
        # Apply adaptive blend
        blended = adaptive_blend(
            x_on_device, output,
            blend_strength=args.blend_strength,
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold
        )
        
        # Apply soft or hard clamping
        if args.soft_clamp:
            blended = soft_clamp(blended)
        
        # Save
        enhanced = blended[0].permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        
        out_path = os.path.join(args.output_dir, f'adaptive_{fname}')
        enhanced_img.save(out_path)
        print(f'  -> Saved: {out_path}')
    
    print('\n' + '=' * 50)
    print('Done! Check test_results folder.')
    print('Tip: Compare adaptive_* with enhanced_* to see the difference.')
