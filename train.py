#!/usr/bin/env python3
"""Training script for Swin-LLIE"""

import os
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from swinllie import SwinLLIE, HybridLoss, get_dataloader
from swinllie.utils import calculate_psnr, calculate_ssim

# Config
EPOCHS = 100
BATCH_SIZE = 4
PATCH_SIZE = 96
LR = 2e-4
DATASET = './datasets/LOL'
SAVE_DIR = './experiments/test_run'
EVAL_INTERVAL = 10  # Evaluate every N epochs


def evaluate_single(model, device, dataset_path):
    """Evaluate model on one test image, returns (psnr, ssim)."""
    model.eval()
    test_loader = get_dataloader('lol', dataset_path, 'test', batch_size=1, patch_size=None, num_workers=1)
    
    with torch.no_grad():
        batch = next(iter(test_loader))  # Get first image only
        low, high = batch['low'].to(device), batch['high'].to(device)
        output = model(low).clamp(0, 1)
        
        # Convert to numpy (H, W, C) for metrics
        out_np = output[0].cpu().numpy().transpose(1, 2, 0) * 255
        gt_np = high[0].cpu().numpy().transpose(1, 2, 0) * 255
        
        return calculate_psnr(out_np, gt_np), calculate_ssim(out_np, gt_np)

if __name__ == '__main__':
    os.makedirs(f'{SAVE_DIR}/checkpoints', exist_ok=True)

    # Check GPU compatibility (CUDA capability must be >= 7.0 for PyTorch 2.0+)
    use_cuda = False
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            compute_capability = capability[0] + capability[1] / 10
            if compute_capability >= 7.0:
                use_cuda = True
            else:
                print(f'GPU detected but incompatible (compute capability {compute_capability:.1f} < 7.0)')
                print('Falling back to CPU...')
        except:
            print('GPU detection failed, using CPU...')
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')

    # Model
    model = SwinLLIE(img_size=128, embed_dim=60, depths=[4,4,4], num_heads=[6,6,6], window_size=8).to(device)
    criterion = HybridLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Data
    train_loader = get_dataloader('lol', DATASET, 'train', BATCH_SIZE, PATCH_SIZE, num_workers=2)
    print(f'Training samples: {len(train_loader.dataset)}')

    # Train
    best_psnr = 0
    best_ssim = 0
    best_loss = float('inf')
    interval_best_loss = float('inf')
    interval_best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            low, high = batch['low'].to(device), batch['high'].to(device)
            
            optimizer.zero_grad()
            with autocast():
                output = model(low)
                illum, _ = model.get_illumination_map(low)
                loss, _ = criterion(output, high, illum)  # HybridLoss returns (loss, dict)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
        
        # Track best loss within each 10-epoch interval
        if avg_loss < interval_best_loss:
            interval_best_loss = avg_loss
            interval_best_state = model.state_dict().copy()
        
        # At every 10th epoch, evaluate PSNR/SSIM on best-loss model from interval
        if (epoch + 1) % EVAL_INTERVAL == 0:
            # Load best model from this interval for evaluation
            model.load_state_dict(interval_best_state)
            psnr, ssim = evaluate_single(model, device, DATASET)
            print(f'  -> Test PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Best interval loss: {interval_best_loss:.4f}')
            
            # Save if better overall (PSNR primary, SSIM secondary, loss tertiary)
            is_best = (psnr > best_psnr) or (psnr == best_psnr and ssim > best_ssim) or \
                      (psnr == best_psnr and ssim == best_ssim and interval_best_loss < best_loss)
            if is_best:
                best_psnr, best_ssim, best_loss = psnr, ssim, interval_best_loss
                torch.save({'model_state_dict': interval_best_state, 'epoch': epoch,
                            'psnr': psnr, 'ssim': ssim, 'loss': interval_best_loss}, f'{SAVE_DIR}/checkpoints/best.pth')
                print(f'  -> New best! PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}, Loss: {best_loss:.4f}')
            
            # Reset for next interval
            interval_best_loss = float('inf')
            interval_best_state = None
        
        if (epoch + 1) % 20 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, f'{SAVE_DIR}/checkpoints/epoch_{epoch+1}.pth')

    # Final save
    torch.save({'model_state_dict': model.state_dict(), 'epoch': EPOCHS-1}, f'{SAVE_DIR}/checkpoints/final.pth')
    print(f'\nTraining complete! Best PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}, Loss: {best_loss:.4f}')
