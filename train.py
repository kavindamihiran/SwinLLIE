#!/usr/bin/env python3
"""Training script for Swin-LLIE"""

import os
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from swinllie import SwinLLIE, HybridLoss, get_dataloader

# Config
EPOCHS = 100
BATCH_SIZE = 4
PATCH_SIZE = 96
LR = 2e-4
DATASET = './datasets/LOL'
SAVE_DIR = './experiments/test_run'

if __name__ == '__main__':
    os.makedirs(f'{SAVE_DIR}/checkpoints', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    best_loss = float('inf')
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
        
        # Save checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, f'{SAVE_DIR}/checkpoints/best.pth')
            print(f'  -> New best! Loss: {best_loss:.4f}')
        
        if (epoch + 1) % 20 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, f'{SAVE_DIR}/checkpoints/epoch_{epoch+1}.pth')

    # Final save
    torch.save({'model_state_dict': model.state_dict(), 'epoch': EPOCHS-1}, f'{SAVE_DIR}/checkpoints/final.pth')
    print(f'\nTraining complete! Best loss: {best_loss:.4f}')
