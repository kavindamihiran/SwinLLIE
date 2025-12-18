#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# Training Script for Swin-LLIE: Low-Light Image Enhancement
# -----------------------------------------------------------------------------------

import os
import sys
import argparse
import time
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network_swinllie import SwinLLIE
from losses import HybridLoss
from data.lowlight_dataset import get_dataloader


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


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_psnr': best_psnr,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_psnr', 0)


def get_optimizer(model, config):
    """Create optimizer based on config."""
    opt_config = config['training']
    
    if opt_config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=tuple(opt_config['betas'])
        )
    elif opt_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['learning_rate'],
            betas=tuple(opt_config['betas'])
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['learning_rate'],
            momentum=0.9,
            weight_decay=opt_config['weight_decay']
        )
    
    return optimizer


def get_scheduler(optimizer, config, num_batches):
    """Create learning rate scheduler based on config."""
    opt_config = config['training']
    epochs = opt_config['epochs']
    warmup_epochs = opt_config.get('warmup_epochs', 5)
    min_lr = opt_config.get('min_lr', 1e-6)
    
    if opt_config['scheduler'].lower() == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            warmup_steps = warmup_epochs * num_batches
            total_steps = epochs * num_batches
            
            if current_step < warmup_steps:
                # Linear warmup
                return current_step / warmup_steps
            else:
                # Cosine annealing
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return max(min_lr / opt_config['learning_rate'], 
                          0.5 * (1 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif opt_config['scheduler'].lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, 
                    device, epoch, config, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
    
    for i, batch in enumerate(pbar):
        low_img = batch['low'].to(device)
        high_img = batch['high'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config['training']['use_amp']:
            with autocast():
                # Forward pass
                pred = model(low_img)
                
                # Get illumination map for smoothness loss
                illum_map, _ = model.get_illumination_map(low_img)
                
                # Compute loss
                loss, loss_dict = criterion(pred, high_img, illum_map)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            pred = model(low_img)
            illum_map, _ = model.get_illumination_map(low_img)
            loss, loss_dict = criterion(pred, high_img, illum_map)
            
            loss.backward()
            
            if config['training'].get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        # TensorBoard logging
        if writer and (i + 1) % config['training'].get('log_freq', 100) == 0:
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Train/{k}', v, global_step)
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, dataloader, device, epoch, config, writer=None, save_images=False, save_dir=None):
    """Validate the model."""
    model.eval()
    psnr_list = []
    ssim_list = []
    window_size = config['model']['window_size']
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validating", ncols=100)):
            low_img = batch['low'].to(device)
            high_img = batch['high'].to(device)
            img_name = batch['name'][0] if isinstance(batch['name'], list) else batch['name']
            
            # Pad to be divisible by window_size * 8 (for 3-level U-Net with window attention)
            _, _, h, w = low_img.shape
            mod_pad = window_size * 8
            pad_h = (mod_pad - h % mod_pad) % mod_pad
            pad_w = (mod_pad - w % mod_pad) % mod_pad
            
            if pad_h > 0 or pad_w > 0:
                low_img_padded = torch.nn.functional.pad(low_img, (0, pad_w, 0, pad_h), mode='reflect')
                high_img_padded = torch.nn.functional.pad(high_img, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                low_img_padded = low_img
                high_img_padded = high_img
            
            # Forward pass
            pred = model(low_img_padded)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :h, :w]
                high_img = high_img_padded[:, :, :h, :w]
            
            # Clamp to valid range
            pred = torch.clamp(pred, 0, 1)
            
            # Calculate metrics
            pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            high_np = high_img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            
            pred_np = pred_np.astype(np.uint8)
            high_np = high_np.astype(np.uint8)
            
            psnr = calculate_psnr(pred_np, high_np, crop_border=0)
            ssim = calculate_ssim(pred_np, high_np, crop_border=0)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # Save validation images
            if save_images and save_dir and i < config['validation'].get('num_save_images', 5):
                import cv2
                os.makedirs(save_dir, exist_ok=True)
                
                # Save low, pred, high side by side
                low_np = low_img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
                low_np = low_np.astype(np.uint8)
                
                comparison = np.concatenate([low_np, pred_np, high_np], axis=1)
                comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f'epoch{epoch}_{img_name}.png'), comparison)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        writer.add_scalar('Val/SSIM', avg_ssim, epoch)
    
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='Train Swin-LLIE')
    parser.add_argument('--config', type=str, default='configs/swinllie_lol.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Swin-LLIE Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    
    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_images'), exist_ok=True)
    
    # Copy config to save directory
    import shutil
    shutil.copy(args.config, os.path.join(save_dir, 'config.yaml'))
    
    # TensorBoard
    writer = None
    if config['training'].get('use_tensorboard', True):
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    # Create model
    print(f"\n[Model] Creating SwinLLIE...")
    model_config = config['model']
    model = SwinLLIE(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config['in_chans'],
        embed_dim=model_config['embed_dim'],
        depths=model_config['depths'],
        num_heads=model_config['num_heads'],
        window_size=model_config['window_size'],
        mlp_ratio=model_config['mlp_ratio'],
        qkv_bias=model_config['qkv_bias'],
        drop_rate=model_config['drop_rate'],
        attn_drop_rate=model_config['attn_drop_rate'],
        drop_path_rate=model_config['drop_path_rate'],
        use_checkpoint=model_config['use_checkpoint'],
        resi_connection=model_config['resi_connection'],
        use_igam=model_config['use_igam']
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {num_params:,}")
    
    # Create dataloaders
    print(f"\n[Data] Loading dataset...")
    dataset_config = config['dataset']
    
    train_loader = get_dataloader(
        dataset_type=dataset_config['name'],
        root_dir=dataset_config['root_dir'],
        split='train',
        batch_size=config['training']['batch_size'],
        patch_size=dataset_config['patch_size'],
        num_workers=dataset_config['num_workers']
    )
    
    val_loader = get_dataloader(
        dataset_type=dataset_config['name'],
        root_dir=dataset_config['root_dir'],
        split='test',
        batch_size=1,
        patch_size=dataset_config['patch_size'],
        num_workers=dataset_config['num_workers']
    )
    
    print(f"[Data] Train batches: {len(train_loader)}")
    print(f"[Data] Val batches: {len(val_loader)}")
    
    # Create loss function
    print(f"\n[Loss] Creating hybrid loss...")
    loss_config = config['loss']
    criterion = HybridLoss(
        lambda_l1=loss_config['lambda_l1'],
        lambda_vgg=loss_config['lambda_vgg'],
        lambda_color=loss_config['lambda_color'],
        lambda_smooth=loss_config['lambda_smooth'],
        use_ssim=loss_config['use_ssim'],
        lambda_ssim=loss_config['lambda_ssim']
    )
    criterion = criterion.to(device)
    
    # Create optimizer and scheduler
    print(f"\n[Optimizer] Creating {config['training']['optimizer']}...")
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Resume training
    start_epoch = 0
    best_psnr = 0
    
    if config['resume']['enabled']:
        checkpoint_path = config['resume']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"\n[Resume] Loading checkpoint from {checkpoint_path}")
            start_epoch, best_psnr = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler)
            start_epoch += 1
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, config, writer)
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
        for k, v in train_loss_dict.items():
            print(f"  {k}: {v:.4f}")
        
        # Validate
        if (epoch + 1) % config['validation']['val_freq'] == 0:
            val_psnr, val_ssim = validate(
                model, val_loader, device, epoch, config, writer,
                save_images=config['validation']['save_images'],
                save_dir=os.path.join(save_dir, 'val_images'))
            
            print(f"\nEpoch {epoch}: Val PSNR = {val_psnr:.2f} dB, SSIM = {val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, best_psnr,
                    os.path.join(save_dir, 'checkpoints', 'best.pth'))
                print(f"  -> New best model saved! PSNR: {best_psnr:.2f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_psnr,
                os.path.join(save_dir, 'checkpoints', f'epoch_{epoch}.pth'))
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, epoch, best_psnr,
        os.path.join(save_dir, 'checkpoints', 'final.pth'))
    
    if writer:
        writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {save_dir}/checkpoints/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
