#!/usr/bin/env python3
"""Inference script for Swin-LLIE - Smart GPU/CPU fallback"""

import os
import torch
import numpy as np
from PIL import Image
from swinllie import SwinLLIE

# Config
CHECKPOINT = './experiments/test_run/checkpoints/best.pth'
INPUT_DIR = './test'
OUTPUT_DIR = './test_results'
WINDOW_SIZE = 8

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
    
    # Check GPU compatibility (CUDA capability must be >= 7.0 for PyTorch 2.0+)
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
    
    # Try GPU first
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
            
            # Move to CPU
            model = model.to('cpu')
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()
            with torch.no_grad():
                output = process_image(model, x.to('cpu'), 'cpu')
            print('  CPU success!')
            return output
        else:
            raise e

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('Swin-LLIE Inference')
    print('=' * 40)
    
    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    
    # Process images
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        print(f'\nProcessing: {fname}')
        
        # Load image
        img_path = os.path.join(INPUT_DIR, fname)
        img = Image.open(img_path).convert('RGB')
        
        # To tensor
        img_np = np.array(img) / 255.0
        x = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0)
        
        print(f'  Size: {x.shape[3]}x{x.shape[2]}')
        
        # Create fresh model for each image (to handle GPU/CPU switching)
        model = SwinLLIE(img_size=128, embed_dim=60, depths=[4,4,4], num_heads=[6,6,6], window_size=WINDOW_SIZE)
        
        # Process with GPU/CPU fallback
        output = try_gpu_with_fallback(model, x, ckpt)
        
        # Save
        enhanced = output[0].permute(1,2,0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        
        out_path = os.path.join(OUTPUT_DIR, f'enhanced_{fname}')
        enhanced_img.save(out_path)
        print(f'  -> Saved: {out_path}')
    
    print('\n' + '=' * 40)
    print('Done! Check test_results folder.')
