#!/usr/bin/env python3
"""
Create qualitative comparison images for research paper.
Shows Input | Output | Ground Truth side by side in a single image.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from swinllie import SwinLLIE

# Config
CHECKPOINT = './experiments/test_run/checkpoints/best.pth'
LOW_DIR = './datasets/LOL/eval15/low'
HIGH_DIR = './datasets/LOL/eval15/high'
OUTPUT_DIR = './Guides/Research Paper'
WINDOW_SIZE = 8

# Select images for comparison (all 15 eval images - select best ones later)
SELECTED_IMAGES = [
    '1.png', '22.png', '23.png', '55.png', '79.png',
    '111.png', '146.png', '179.png', '493.png', '547.png',
    '665.png', '669.png', '748.png', '778.png', '780.png'
]

def pad_and_process(model, x, device):
    """Process image with padding for window-based attention."""
    H, W = x.shape[2], x.shape[3]
    
    # Pad to multiple of window_size * 4 (for U-Net downsampling)
    pad_unit = WINDOW_SIZE * 4
    pad_h = (pad_unit - H % pad_unit) % pad_unit
    pad_w = (pad_unit - W % pad_unit) % pad_unit
    x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Process
    output = model(x_padded.to(device))
    
    # Remove padding
    return output[:, :, :H, :W]

def create_comparison_image(low_img, enhanced_img, high_img, output_path, labels=True):
    """
    Create a side-by-side comparison image: Input | Output | Ground Truth
    """
    # Ensure all images are same size
    w, h = low_img.size
    
    # Create combined image with spacing
    spacing = 10
    total_width = w * 3 + spacing * 2
    
    # Add space for labels if needed
    label_height = 40 if labels else 0
    total_height = h + label_height
    
    # Create white background
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste images
    y_offset = label_height
    combined.paste(low_img, (0, y_offset))
    combined.paste(enhanced_img, (w + spacing, y_offset))
    combined.paste(high_img, (w * 2 + spacing * 2, y_offset))
    
    # Add labels
    if labels:
        draw = ImageDraw.Draw(combined)
        try:
            # Try to use a nicer font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        label_texts = ['(a) Input', '(b) SwinLLIE Output', '(c) Ground Truth']
        label_positions = [
            w // 2,
            w + spacing + w // 2,
            w * 2 + spacing * 2 + w // 2
        ]
        
        for text, x_pos in zip(label_texts, label_positions):
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = x_pos - text_width // 2
            draw.text((x, 8), text, fill=(0, 0, 0), font=font)
    
    # Save
    combined.save(output_path, quality=95)
    print(f'Saved: {output_path}')
    
    return combined

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('Creating Qualitative Comparison Images')
    print('=' * 50)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model = SwinLLIE(
        img_size=128, 
        embed_dim=60, 
        depths=[4,4,4], 
        num_heads=[6,6,6], 
        window_size=WINDOW_SIZE
    )
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded from: {CHECKPOINT}')
    print()
    
    for idx, img_name in enumerate(SELECTED_IMAGES, 1):
        print(f'Processing image {idx}: {img_name}')
        
        # Load low-light input
        low_path = os.path.join(LOW_DIR, img_name)
        low_img = Image.open(low_path).convert('RGB')
        
        # Load ground truth
        high_path = os.path.join(HIGH_DIR, img_name)
        high_img = Image.open(high_path).convert('RGB')
        
        # Run enhancement
        img_np = np.array(low_img) / 255.0
        x = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0)
        
        with torch.no_grad():
            output = pad_and_process(model, x, device)
        
        enhanced = output[0].permute(1,2,0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        
        # Create comparison image
        output_path = os.path.join(OUTPUT_DIR, f'qualitative_comparison_{idx}.png')
        create_comparison_image(low_img, enhanced_img, high_img, output_path)
    
    print()
    print('=' * 50)
    print(f'Done! Check {OUTPUT_DIR} folder.')
    print('Images are ready for your research paper.')

if __name__ == '__main__':
    main()
