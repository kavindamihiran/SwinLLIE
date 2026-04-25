#!/usr/bin/env python3
"""
Generate ablation_visual.png for the SwinLLIE research paper.
Creates a side-by-side comparison of enhancement results using different loss configurations.
Output: Guides/ablation_visual.png
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from swinllie import SwinLLIE

WINDOW_SIZE = 8

# --- Configuration ---
# Pick 2 representative test images for the figure (2 rows)
TEST_IMAGES = ['23.png', '778.png']
LOW_DIR = './datasets/LOL/eval15/low'
HIGH_DIR = './datasets/LOL/eval15/high'

# Ablation checkpoints (in order for the figure columns)
ABLATION_CONFIGS = [
    {
        'name': '$\mathcal{L}_{L1}$ only',
        'checkpoint': './experiments/ablation_base_l1/checkpoints/best.pth',
    },
    {
        'name': '$\mathcal{L}_{L1}$+$\mathcal{L}_{VGG}$+$\mathcal{L}_{color}$',
        'checkpoint': './experiments/ablation_base_vgg_color/checkpoints/best.pth',
    },
    {
        'name': 'Full Loss (Ours)',
        'checkpoint': './experiments/ablation_full/checkpoints/best.pth',
    },
]

OUTPUT_PATH = './Guides/ablation_visual.png'


def pad_and_infer(model, img_tensor, device):
    """Run inference with proper padding for window attention."""
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    pad_unit = WINDOW_SIZE * 4
    pad_h = (pad_unit - H % pad_unit) % pad_unit
    pad_w = (pad_unit - W % pad_unit) % pad_unit
    x_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        output = model(x_padded)
    output = output[:, :, :H, :W]
    return output.clamp(0, 1)


def load_model(checkpoint_path, device):
    """Load SwinLLIE model from checkpoint."""
    model = SwinLLIE(
        img_size=128, embed_dim=60,
        depths=[4, 4, 4], num_heads=[6, 6, 6], window_size=8
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_image_tensor(path):
    """Load image as normalized tensor."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_numpy(tensor):
    """Convert tensor to displayable numpy array."""
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_images = len(TEST_IMAGES)
    # Columns: Input | L1 only | L1+VGG+Color | Full Loss (Ours) | Ground Truth
    num_cols = 2 + len(ABLATION_CONFIGS)  # input + ablations + GT

    # Collect all result images
    all_results = []  # list of rows, each row is list of numpy images

    for img_name in TEST_IMAGES:
        low_path = os.path.join(LOW_DIR, img_name)
        high_path = os.path.join(HIGH_DIR, img_name)

        if not os.path.exists(low_path) or not os.path.exists(high_path):
            print(f"Warning: {img_name} not found, skipping")
            continue

        print(f"\nProcessing: {img_name}")
        low_tensor = load_image_tensor(low_path).to(device)
        high_np = np.array(Image.open(high_path).convert('RGB')).astype(np.float32) / 255.0

        row = [tensor_to_numpy(low_tensor)]  # First column: input

        for config in ABLATION_CONFIGS:
            ckpt = config['checkpoint']
            if not os.path.exists(ckpt):
                print(f"  Warning: {ckpt} not found, using black placeholder")
                row.append(np.zeros_like(high_np))
                continue

            print(f"  Running: {config['name']}")
            model = load_model(ckpt, device)
            output = pad_and_infer(model, low_tensor, device)
            row.append(tensor_to_numpy(output))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        row.append(high_np)  # Last column: ground truth
        all_results.append(row)

    if not all_results:
        print("Error: No images processed!")
        return

    # --- Create the figure ---
    col_titles = ['Input'] + [c['name'] for c in ABLATION_CONFIGS] + ['Ground Truth']

    fig, axes = plt.subplots(
        num_images, num_cols,
        figsize=(num_cols * 3.5, num_images * 2.8),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.08}
    )

    if num_images == 1:
        axes = [axes]

    for row_idx, row_images in enumerate(all_results):
        for col_idx, img in enumerate(row_images):
            ax = axes[row_idx][col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on top row only
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight='bold', pad=6)

            # Add colored border for "Ours" column
            if col_titles[col_idx] == 'Full Loss (Ours)':
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e74c3c')
                    spine.set_linewidth(2.5)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')
                    spine.set_linewidth(0.5)

        # Row labels
        axes[row_idx][0].set_ylabel(f'Scene {row_idx + 1}', fontsize=10,
                                     fontweight='bold', rotation=90, labelpad=10)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    print(f"\n✓ Saved ablation visual to: {OUTPUT_PATH}")
    print(f"  Resolution: {num_cols * 3.5 * 300:.0f} x {num_images * 2.8 * 300:.0f} pixels")


if __name__ == '__main__':
    main()
