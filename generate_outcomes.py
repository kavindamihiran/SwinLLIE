#!/usr/bin/env python3
"""Generate several good Swin-LLIE inference candidates for each input image."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import (
    CHECKPOINT_DEFAULT,
    INPUT_DIR_DEFAULT,
    adaptive_blend,
    auto_enhance_strength,
    build_model,
    choose_device,
    preserve_input_chroma,
    run_with_fallback,
    soft_clamp,
)


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

CANDIDATES = [
    {
        'name': '01_raw_model',
        'description': 'Raw model output. Strongest enhancement, least protection.',
        'mode': 'raw',
    },
    {
        'name': '02_auto_balanced',
        'description': 'Default balanced candidate for mixed real-world images.',
        'strength_offset': 0.0,
        'dark_threshold': 0.25,
        'bright_threshold': 0.55,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '03_auto_soft',
        'description': 'Softer enhancement for backlit scenes and skies.',
        'strength_offset': -0.15,
        'min_strength': 0.35,
        'dark_threshold': 0.25,
        'bright_threshold': 0.55,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '04_auto_bright',
        'description': 'Brighter version while still protecting highlights.',
        'strength_offset': 0.18,
        'dark_threshold': 0.25,
        'bright_threshold': 0.55,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '05_highlight_strict',
        'description': 'More aggressive sky/light preservation.',
        'strength_offset': 0.05,
        'dark_threshold': 0.20,
        'bright_threshold': 0.45,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '06_shadow_only',
        'description': 'Boosts dark regions while leaving mid/bright regions closer to input.',
        'strength_offset': 0.10,
        'max_strength': 0.85,
        'dark_threshold': 0.12,
        'bright_threshold': 0.35,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '07_full_low_light',
        'description': 'Full enhancement with color and highlight protection.',
        'fixed_strength': 1.0,
        'dark_threshold': 0.25,
        'bright_threshold': 0.55,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 10.0,
        'saturation': 1.0,
    },
    {
        'name': '08_color_pop',
        'description': 'Balanced candidate with subtle extra color.',
        'strength_offset': 0.0,
        'dark_threshold': 0.25,
        'bright_threshold': 0.55,
        'blend_strength': 1.0,
        'chroma_strength': 1.0,
        'max_color_gain': 12.0,
        'saturation': 1.12,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multiple Swin-LLIE output candidates.')
    parser.add_argument('--input', type=str, default=INPUT_DIR_DEFAULT, help='Input image or directory')
    parser.add_argument('--weights', type=str, default=CHECKPOINT_DEFAULT, help='Model weights path')
    parser.add_argument('--output-root', type=str, default='./results', help='Root folder for generated candidates')
    parser.add_argument('--run-name', type=str, default=None, help='Output subfolder name')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg'], help='Candidate image format')
    parser.add_argument('--jpeg-quality', type=int, default=95, help='JPEG quality when --format jpg is used')
    return parser.parse_args()


def find_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported image file: {path}')
        return [path]

    if not path.is_dir():
        raise ValueError(f'Input path not found: {path}')

    images = []
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(file_path)
    return sorted(images)


def safe_name(text):
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)
    return cleaned.strip('_') or 'image'


def image_stats(img_np):
    lum = 0.2126 * img_np[:, :, 0] + 0.7152 * img_np[:, :, 1] + 0.0722 * img_np[:, :, 2]
    saturation = (img_np.max(axis=2) - img_np.min(axis=2)) / (img_np.max(axis=2) + 1e-6)
    return {
        'mean_luminance': float(lum.mean()),
        'p95_luminance': float(np.percentile(lum, 95)),
        'bright_percent': float((lum > 0.55).mean() * 100),
        'near_clipped_percent': float((img_np >= 0.99).any(axis=2).mean() * 100),
        'mean_saturation': float(saturation.mean()),
    }


def tensor_to_uint8(tensor):
    array = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def save_image(array, path, jpeg_quality):
    image = Image.fromarray(array)
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)


def resolve_strength(candidate, auto_strength):
    if 'fixed_strength' in candidate:
        strength = candidate['fixed_strength']
    else:
        strength = auto_strength + candidate.get('strength_offset', 0.0)

    min_strength = candidate.get('min_strength', 0.0)
    max_strength = candidate.get('max_strength', 1.0)
    return float(np.clip(strength, min_strength, max_strength))


def boost_saturation(tensor, multiplier):
    if multiplier == 1.0:
        return tensor

    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=tensor.device, dtype=tensor.dtype)
    gray = (tensor * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return (gray + (tensor - gray) * multiplier).clamp(0.0, 1.0)


def apply_candidate(input_tensor, raw_output, candidate, auto_strength):
    if candidate.get('mode') == 'raw':
        return raw_output.clamp(0.0, 1.0), {'enhance_strength': 1.0}

    output = preserve_input_chroma(
        input_tensor,
        raw_output,
        strength=candidate.get('chroma_strength', 1.0),
        max_gain=candidate.get('max_color_gain', 10.0),
    )
    output = adaptive_blend(
        input_tensor,
        output,
        blend_strength=candidate.get('blend_strength', 1.0),
        dark_threshold=candidate.get('dark_threshold', 0.25),
        bright_threshold=candidate.get('bright_threshold', 0.55),
    )
    output = soft_clamp(output)

    enhance_strength = resolve_strength(candidate, auto_strength)
    output = (input_tensor + enhance_strength * (output - input_tensor)).clamp(0.0, 1.0)
    output = boost_saturation(output, candidate.get('saturation', 1.0))

    return output, {'enhance_strength': enhance_strength}


def make_contact_sheet(image_paths, labels, output_path):
    thumb_w = 320
    label_h = 28
    pad = 12
    columns = 3
    font = ImageFont.load_default()

    thumbs = []
    cell_h = 0
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        image.thumbnail((thumb_w, 240), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
        cell_h = max(cell_h, image.height + label_h)

    rows = int(np.ceil(len(thumbs) / columns))
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet = Image.new('RGB', (sheet_w, sheet_h), 'white')
    draw = ImageDraw.Draw(sheet)

    for index, (thumb, label) in enumerate(zip(thumbs, labels)):
        row, col = divmod(index, columns)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (cell_h + pad)
        draw.text((x, y), label[:44], fill='black', font=font)
        image_x = x + (thumb_w - thumb.width) // 2
        sheet.paste(thumb, (image_x, y + label_h))

    sheet.save(output_path)


def write_run_summary(run_dir, records):
    lines = ['Swin-LLIE candidate generation summary', '']
    for record in records:
        lines.append(f"Image: {record['input']}")
        lines.append(f"Folder: {record['folder']}")
        lines.append(f"Auto strength: {record['auto_strength']:.2f}")
        lines.append('')
    (run_dir / 'run_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    input_images = find_images(args.input)
    if not input_images:
        raise ValueError(f'No images found in {args.input}')

    run_name = args.run_name or f'candidates_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.output_root) / safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    print('Swin-LLIE Candidate Generator')
    print('=' * 44)
    print(f'Images: {len(input_images)}')
    print(f'Output: {run_dir}')

    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    device = choose_device()
    print(f'Device: {device}')
    model = build_model(checkpoint, device)

    run_records = []
    extension = f'.{args.format}'

    for image_index, image_path in enumerate(input_images, start=1):
        folder = run_dir / f'{image_index:03d}_{safe_name(image_path.stem)}'
        folder.mkdir(parents=True, exist_ok=True)

        print(f'\n[{image_index}/{len(input_images)}] {image_path}')
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image, dtype=np.float32) / 255.0
        stats = image_stats(img_np)
        auto_strength = auto_enhance_strength(stats['mean_luminance'], stats['bright_percent'])
        print(
            f"  mean={stats['mean_luminance']:.3f}, bright={stats['bright_percent']:.1f}%, "
            f"clip={stats['near_clipped_percent']:.1f}%, auto_strength={auto_strength:.2f}"
        )

        input_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
        raw_output, model, device = run_with_fallback(model, checkpoint, input_tensor, device)
        input_on_device = input_tensor.to(raw_output.device)

        output_paths = []
        labels = []
        original_path = folder / '00_original.png'
        save_image((img_np * 255).astype(np.uint8), original_path, args.jpeg_quality)
        output_paths.append(original_path)
        labels.append('00 original')

        candidates_meta = []
        for candidate in CANDIDATES:
            output, extra_meta = apply_candidate(input_on_device, raw_output, candidate, auto_strength)
            output_array = tensor_to_uint8(output)
            output_path = folder / f"{candidate['name']}{extension}"
            save_image(output_array, output_path, args.jpeg_quality)
            output_paths.append(output_path)
            labels.append(candidate['name'])

            out_stats = image_stats(output_array.astype(np.float32) / 255.0)
            candidates_meta.append({
                'name': candidate['name'],
                'file': output_path.name,
                'description': candidate['description'],
                'settings': {k: v for k, v in candidate.items() if k not in {'name', 'description'}},
                'actual': extra_meta,
                'stats': out_stats,
            })

        contact_sheet_path = folder / 'contact_sheet.png'
        make_contact_sheet(output_paths, labels, contact_sheet_path)

        metadata = {
            'input': str(image_path),
            'weights': args.weights,
            'input_stats': stats,
            'auto_strength': auto_strength,
            'contact_sheet': contact_sheet_path.name,
            'candidates': candidates_meta,
        }
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        run_records.append({
            'input': str(image_path),
            'folder': str(folder),
            'auto_strength': auto_strength,
        })
        print(f'  -> {folder}')

    write_run_summary(run_dir, run_records)
    print('\n' + '=' * 44)
    print(f'Done. Open the contact sheets in: {run_dir}')


if __name__ == '__main__':
    main()
