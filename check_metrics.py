import torch
import os
import sys
import numpy as np
import argparse
from swinllie import SwinLLIE, get_dataloader
from swinllie.utils import calculate_psnr, calculate_ssim

def check_metrics():
    parser = argparse.ArgumentParser(description='Evaluate Swin-LLIE metrics')
    parser.add_argument('--weights', type=str, default='./experiments/test_run/checkpoints/best.pth', help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='./datasets/LOL', help='Path to dataset root')
    args = parser.parse_args()

    checkpoint_path = args.weights
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        # weights_only=False is required for PyTorch 2.6+ compatibility when loading custom metadata
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("Checkpoint Loaded Successfully.")
        print("-" * 30)
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Saved Best PSNR: {checkpoint.get('psnr', 'N/A')}")
        print(f"Saved Best SSIM: {checkpoint.get('ssim', 'N/A')}")
        print(f"Saved Best Loss: {checkpoint.get('loss', 'N/A')}")
        print("-" * 30)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Check for test dataset
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found. Skipping full evaluation.")
        return

    print("Running full evaluation on LOL test set...")
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model = SwinLLIE(
            img_size=128,
            embed_dim=60,
            depths=[4, 4, 4],
            num_heads=[6, 6, 6],
            window_size=8
        ).to(device)
        
        # Handle state dict keys
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') # remove module.
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # DataLoader
        test_loader = get_dataloader('lol', dataset_path, 'test', batch_size=1, patch_size=None, num_workers=1)
        
        psnr_list = []
        ssim_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if 'low' not in batch or 'high' not in batch:
                    continue
                    
                low = batch['low'].to(device)
                high = batch['high'].to(device)
                
                output = model(low).clamp(0, 1)
                
                # Convert to numpy
                out_np = output[0].cpu().numpy().transpose(1, 2, 0) * 255
                gt_np = high[0].cpu().numpy().transpose(1, 2, 0) * 255
                
                # Calculate metrics
                psnr = calculate_psnr(out_np, gt_np)
                
                # SSIM requires at least 7x7 window, images should be large enough
                # Using the same function as in utils.py
                ssim = calculate_ssim(out_np, gt_np)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if (i+1) % 10 == 0:
                    print(f"Processed {i+1} images...")

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            print("-" * 30)
            print("Full Test Set Evaluation Results:")
            print(f"Average PSNR: {avg_psnr:.4f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Number of images: {len(psnr_list)}")
            print("-" * 30)
        else:
            print("No images processed.")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_metrics()
