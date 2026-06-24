#!/usr/bin/env python3
"""Inference script for Swin-LLIE with highlight-safe output."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from swinllie import SwinLLIE


CHECKPOINT_DEFAULT = './experiments/test_run/checkpoints/best.pth'
INPUT_DIR_DEFAULT = './test'
OUTPUT_DIR_DEFAULT = './test_results'
WINDOW_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for Swin-LLIE')
    parser.add_argument('--input', type=str, default=INPUT_DIR_DEFAULT, help='Input directory')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR_DEFAULT, help='Output directory')
    parser.add_argument('--weights', type=str, default=CHECKPOINT_DEFAULT, help='Model weights path')
    parser.add_argument(
        '--blend-strength',
        type=float,
        default=1.0,
        help='Highlight protection strength: 0.0=raw model output, 1.0=preserve bright input regions',
    )
    parser.add_argument(
        '--dark-threshold',
        type=float,
        default=0.25,
        help='Input luminance below this receives full enhancement',
    )
    parser.add_argument(
        '--bright-threshold',
        type=float,
        default=0.55,
        help='Input luminance above this is increasingly preserved',
    )
    parser.add_argument(
        '--enhance-strength',
        type=float,
        default=None,
        help='Overall enhancement strength. Omit for automatic mixed-exposure handling.',
    )
    parser.add_argument(
        '--chroma-strength',
        type=float,
        default=1.0,
        help='Preserve input color ratios to reduce gray artifacts: 0.0=off, 1.0=full',
    )
    parser.add_argument(
        '--max-color-gain',
        type=float,
        default=10.0,
        help='Maximum luminance gain used by chroma preservation',
    )
    parser.add_argument(
        '--no-soft-clamp',
        dest='soft_clamp',
        action='store_false',
        help='Disable smooth highlight rolloff before final uint8 clipping',
    )
    parser.set_defaults(soft_clamp=True)
    return parser.parse_args()


def normalize_state_dict(checkpoint):
    """Return a strict-loadable state dict, handling DataParallel checkpoints."""
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    return state_dict


def build_model(checkpoint, device):
    model = SwinLLIE(
        img_size=128,
        embed_dim=60,
        depths=[4, 4, 4],
        num_heads=[6, 6, 6],
        window_size=WINDOW_SIZE,
    ).to(device)
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def choose_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    try:
        capability = torch.cuda.get_device_capability()
        compute_capability = capability[0] + capability[1] / 10
        if compute_capability >= 7.0:
            return torch.device('cuda')
        print(f'GPU incompatible (compute capability {compute_capability:.1f} < 7.0), using CPU')
    except RuntimeError:
        print('GPU detection failed, using CPU')

    return torch.device('cpu')


def process_image(model, x):
    """Pad, run the model, then remove padding."""
    height, width = x.shape[2], x.shape[3]

    pad_unit = WINDOW_SIZE * 4
    pad_h = (pad_unit - height % pad_unit) % pad_unit
    pad_w = (pad_unit - width % pad_unit) % pad_unit
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    output = model(x_padded)
    return output[:, :, :height, :width]


def luminance(img_tensor):
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img_tensor.device, dtype=img_tensor.dtype)
    return (img_tensor * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def compute_brightness_map(img_tensor, kernel_size=31):
    """Compute smooth local luminance from an RGB tensor in [0, 1]."""
    gray = luminance(img_tensor)

    pad = kernel_size // 2
    return F.avg_pool2d(
        F.pad(gray, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
    )


def adaptive_blend(input_img, enhanced_img, blend_strength, dark_threshold, bright_threshold):
    """
    Preserve already-bright input regions while enhancing dark regions.

    Alpha is 1 in dark areas, approaches 0 in bright areas, and transitions
    smoothly between the two thresholds.
    """
    if blend_strength <= 0:
        return enhanced_img
    if bright_threshold <= dark_threshold:
        raise ValueError('--bright-threshold must be greater than --dark-threshold')

    brightness = compute_brightness_map(input_img)
    alpha = 1.0 - torch.clamp(
        (brightness - dark_threshold) / (bright_threshold - dark_threshold),
        min=0.0,
        max=1.0,
    )
    effective_alpha = 1.0 - blend_strength * (1.0 - alpha)
    return effective_alpha * enhanced_img + (1.0 - effective_alpha) * input_img


def preserve_input_chroma(input_img, enhanced_img, strength=1.0, max_gain=10.0):
    """Keep model luminance while borrowing stable RGB ratios from the input."""
    if strength <= 0:
        return enhanced_img
    if max_gain <= 0:
        raise ValueError('--max-color-gain must be greater than 0')

    input_lum = luminance(input_img).clamp_min(1e-4)
    enhanced_lum = luminance(enhanced_img).clamp(0.0, 1.0)
    gain = (enhanced_lum / input_lum).clamp(0.0, max_gain)
    chroma_preserved = (input_img * gain).clamp(0.0, 1.0)

    # Very dark pixels have unreliable color; trust input chroma once there is
    # enough signal to avoid amplifying color noise.
    chroma_trust = torch.clamp((compute_brightness_map(input_img, kernel_size=15) - 0.02) / 0.16, 0.0, 1.0)
    chroma_weight = strength * chroma_trust
    return (1.0 - chroma_weight) * enhanced_img + chroma_weight * chroma_preserved


def auto_enhance_strength(mean_lum, bright_pct):
    """Reduce enhancement for mixed-exposure scenes, keep full strength for dark scenes."""
    if mean_lum < 0.09:
        return 1.0

    mean_score = np.clip((mean_lum - 0.09) / 0.12, 0.0, 1.0)
    bright_score = np.clip((bright_pct - 3.0) / 12.0, 0.0, 1.0)
    mixed_exposure_score = max(mean_score, bright_score)
    return float(1.0 - 0.55 * mixed_exposure_score)


def soft_clamp(x, max_val=1.0, knee=0.9):
    """Compress values near white to avoid harsh clipping artifacts."""
    above_knee = torch.clamp(x - knee, min=0.0)
    compressed = knee + (max_val - knee) * torch.tanh(above_knee / (max_val - knee))
    return torch.where(x <= knee, x, compressed).clamp(0.0, max_val)


def print_input_stats(img_np):
    luminance = 0.2126 * img_np[:, :, 0] + 0.7152 * img_np[:, :, 1] + 0.0722 * img_np[:, :, 2]
    mean_lum = float(luminance.mean())
    bright_pct = float((luminance > 0.55).mean() * 100)
    clipped_pct = float((img_np >= 0.99).any(axis=2).mean() * 100)
    print(f'  Input luminance: mean={mean_lum:.3f}, bright={bright_pct:.1f}%, near_clipped={clipped_pct:.1f}%')
    return mean_lum, bright_pct, clipped_pct


def run_with_fallback(model, checkpoint, x, device):
    try:
        with torch.inference_mode():
            return process_image(model, x.to(device)), model, device
    except RuntimeError as exc:
        if device.type == 'cuda' and 'out of memory' in str(exc).lower():
            print('  GPU out of memory, switching to CPU...')
            torch.cuda.empty_cache()
            device = torch.device('cpu')
            model = build_model(checkpoint, device)
            with torch.inference_mode():
                return process_image(model, x.to(device)), model, device
        raise


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print('Swin-LLIE Inference')
    print('=' * 40)
    print(f'Highlight protection: blend_strength={args.blend_strength}')
    print(f'Thresholds: dark={args.dark_threshold}, bright={args.bright_threshold}')
    print(f'Enhance strength: {"auto" if args.enhance_strength is None else args.enhance_strength}')
    print(f'Chroma preservation: strength={args.chroma_strength}, max_gain={args.max_color_gain}')
    print(f'Soft clamp: {args.soft_clamp}')

    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    device = choose_device()
    print(f'Device: {device}')
    model = build_model(checkpoint, device)

    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        print(f'\nProcessing: {fname}')
        img_path = os.path.join(args.input, fname)
        img = Image.open(img_path).convert('RGB')

        img_np = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)

        print(f'  Size: {x.shape[3]}x{x.shape[2]}')
        mean_lum, bright_pct, _ = print_input_stats(img_np)
        enhance_strength = (
            auto_enhance_strength(mean_lum, bright_pct)
            if args.enhance_strength is None
            else args.enhance_strength
        )
        print(f'  Enhancement strength: {enhance_strength:.2f}')

        output, model, device = run_with_fallback(model, checkpoint, x, device)
        x_on_device = x.to(output.device)
        output = preserve_input_chroma(
            x_on_device,
            output,
            strength=args.chroma_strength,
            max_gain=args.max_color_gain,
        )
        output = adaptive_blend(
            x_on_device,
            output,
            blend_strength=args.blend_strength,
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold,
        )

        if args.soft_clamp:
            output = soft_clamp(output)
        output = (x_on_device + enhance_strength * (output - x_on_device)).clamp(0.0, 1.0)

        enhanced = output[0].permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))

        out_path = os.path.join(args.output, f'enhanced_{fname}')
        enhanced_img.save(out_path)
        print(f'  -> Saved: {out_path}')

    print('\n' + '=' * 40)
    print(f'Done. Check {args.output} folder.')


if __name__ == '__main__':
    main()
