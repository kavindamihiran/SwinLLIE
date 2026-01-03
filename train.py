#!/usr/bin/env python3
"""Training script for Swin-LLIE"""

import os
import argparse
import numpy as np
import yaml
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from swinllie import SwinLLIE, HybridLoss, get_dataloader
from swinllie.utils import calculate_psnr, calculate_ssim


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Swin-LLIE model')
    parser.add_argument('--config', type=str, default='./configs/swinllie_lol.yaml',
                        help='Path to config file')
    return parser.parse_args()


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
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Extract config values
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    train_cfg = config['training']
    loss_cfg = config['loss']
    val_cfg = config['validation']
    resume_cfg = config.get('resume', {'enabled': False})
    
    # Training parameters from config
    EPOCHS = train_cfg['epochs']
    BATCH_SIZE = train_cfg['batch_size']
    PATCH_SIZE = dataset_cfg['patch_size']
    LR = train_cfg['learning_rate']
    DATASET = dataset_cfg['root_dir']
    SAVE_DIR = train_cfg['save_dir']
    EVAL_INTERVAL = val_cfg['val_freq']
    SAVE_FREQ = train_cfg.get('save_freq', 20)
    WARMUP_EPOCHS = train_cfg.get('warmup_epochs', 5)
    MIN_LR = train_cfg.get('min_lr', 1e-6)
    GRAD_CLIP = train_cfg.get('grad_clip', 1.0)
    USE_AMP = train_cfg.get('use_amp', True)
    
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
    print(f'Config: {args.config}')

    # Model - use parameters from config
    model = SwinLLIE(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg.get('patch_size', 1),
        in_chans=model_cfg.get('in_chans', 3),
        embed_dim=model_cfg['embed_dim'],
        depths=model_cfg['depths'],
        num_heads=model_cfg['num_heads'],
        window_size=model_cfg['window_size'],
        mlp_ratio=model_cfg.get('mlp_ratio', 2.0),
        qkv_bias=model_cfg.get('qkv_bias', True),
        drop_rate=model_cfg.get('drop_rate', 0.0),
        attn_drop_rate=model_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
        use_checkpoint=model_cfg.get('use_checkpoint', False),
        resi_connection=model_cfg.get('resi_connection', '1conv')
    ).to(device)
    
    # Loss with config parameters
    criterion = HybridLoss(
        lambda_l1=loss_cfg.get('lambda_l1', 1.0),
        lambda_vgg=loss_cfg.get('lambda_vgg', 0.1),
        lambda_color=loss_cfg.get('lambda_color', 0.5),
        lambda_smooth=loss_cfg.get('lambda_smooth', 0.01),
        lambda_edge=loss_cfg.get('lambda_edge', 1.0),
        lambda_exposure=loss_cfg.get('lambda_exposure', 1.0),
        use_ssim=loss_cfg.get('use_ssim', False),
        lambda_ssim=loss_cfg.get('lambda_ssim', 0.1)
    ).to(device)
    
    # Optimizer with config parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=train_cfg.get('weight_decay', 1e-4),
        betas=tuple(train_cfg.get('betas', [0.9, 0.999]))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    scaler = GradScaler() if USE_AMP else None

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Data - use config parameters
    train_loader = get_dataloader(
        dataset_cfg['name'], 
        DATASET, 
        'train', 
        BATCH_SIZE, 
        PATCH_SIZE, 
        num_workers=dataset_cfg.get('num_workers', 4)
    )
    print(f'Training samples: {len(train_loader.dataset)}')

    # Resume training if enabled
    start_epoch = 0
    if resume_cfg.get('enabled', False) and os.path.exists(resume_cfg.get('checkpoint_path', '')):
        checkpoint = torch.load(resume_cfg['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    # Train
    best_psnr = 0
    best_ssim = 0
    best_loss = float('inf')
    interval_best_loss = float('inf')
    interval_best_state = None
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            low, high = batch['low'].to(device), batch['high'].to(device)
            
            optimizer.zero_grad()
            
            if USE_AMP and scaler is not None:
                with autocast('cuda'):
                    output = model(low)
                    illum, dark_mask, bright_mask = model.get_illumination_map(low)
                    loss, _ = criterion(output, high, illum, bright_mask)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(low)
                illum, dark_mask, bright_mask = model.get_illumination_map(low)
                loss, _ = criterion(output, high, illum, bright_mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
        
        # Track best loss within each 10-epoch interval
        if avg_loss < interval_best_loss:
            interval_best_loss = avg_loss
            interval_best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
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
        
        if (epoch + 1) % SAVE_FREQ == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, f'{SAVE_DIR}/checkpoints/epoch_{epoch+1}.pth')

    # Final save
    torch.save({'model_state_dict': model.state_dict(), 'epoch': EPOCHS-1}, f'{SAVE_DIR}/checkpoints/final.pth')
    print(f'\nTraining complete! Best PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}, Loss: {best_loss:.4f}')
