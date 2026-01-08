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
    num_gpus = 0
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            compute_capability = capability[0] + capability[1] / 10
            if compute_capability >= 7.0:
                use_cuda = True
                num_gpus = torch.cuda.device_count()
            else:
                print(f'GPU detected but incompatible (compute capability {compute_capability:.1f} < 7.0)')
                print('Falling back to CPU...')
        except:
            print('GPU detection failed, using CPU...')
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')
    if num_gpus > 1:
        print(f'Found {num_gpus} GPUs - will use DataParallel for multi-GPU training')
    elif num_gpus == 1:
        print(f'Found 1 GPU')
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
    
    # Enable multi-GPU training if available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f'Model wrapped with DataParallel across {num_gpus} GPUs')
    
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
    
    # Auto-scale batch size and workers for multi-GPU
    base_batch_size = BATCH_SIZE
    base_num_workers = dataset_cfg.get('num_workers', 4)
    
    if num_gpus > 1:
        # Scale batch size by number of GPUs
        BATCH_SIZE = base_batch_size * num_gpus
        # Scale workers by number of GPUs
        num_workers_scaled = base_num_workers * num_gpus
        print(f'Auto-scaling for {num_gpus} GPUs:')
        print(f'  Batch size: {base_batch_size} -> {BATCH_SIZE} ({base_batch_size} per GPU)')
        print(f'  Workers: {base_num_workers} -> {num_workers_scaled} ({base_num_workers} per GPU)')
    else:
        num_workers_scaled = base_num_workers
        print(f'Single GPU setup: batch_size={BATCH_SIZE}, workers={num_workers_scaled}')
    
    # Optimizer with config parameters
    # Scale learning rate with batch size (linear scaling rule)
    effective_lr = LR * (BATCH_SIZE / base_batch_size) if num_gpus > 1 else LR
    if num_gpus > 1:
        print(f'  Learning rate: {LR} -> {effective_lr:.6f}')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
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
        num_workers=num_workers_scaled
    )
    
    # Enable pin_memory for faster data transfer if using CUDA
    if hasattr(train_loader, 'pin_memory') and use_cuda:
        train_loader.pin_memory = True
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Total batch size: {BATCH_SIZE}')

    # Resume training if enabled
    start_epoch = 0
    if resume_cfg.get('enabled', False) and os.path.exists(resume_cfg.get('checkpoint_path', '')):
        checkpoint = torch.load(resume_cfg['checkpoint_path'], map_location=device)
        # Handle both DataParallel and non-DataParallel checkpoints
        state_dict = checkpoint['model_state_dict']
        if num_gpus > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
            # Add 'module.' prefix if loading non-DataParallel checkpoint into DataParallel model
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif num_gpus <= 1 and any(k.startswith('module.') for k in state_dict.keys()):
            # Remove 'module.' prefix if loading DataParallel checkpoint into non-DataParallel model
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
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
            # Use non_blocking=True for async GPU transfer (faster with pin_memory)
            low = batch['low'].to(device, non_blocking=True)
            high = batch['high'].to(device, non_blocking=True)
            
            optimizer.zero_grad()  # Standard version for maximum compatibility
            
            if USE_AMP and scaler is not None:
                with autocast('cuda'):
                    output = model(low)
                    loss, _ = criterion(output, high)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(low)
                loss, _ = criterion(output, high)
                
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
            # Save state_dict without 'module.' prefix for compatibility
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            interval_best_state = {k: v.clone() for k, v in state_dict.items()}
        
        # At every 10th epoch, evaluate PSNR/SSIM on best-loss model from interval
        if (epoch + 1) % EVAL_INTERVAL == 0:
            # Load best model from this interval for evaluation
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(interval_best_state)
            else:
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
            # Save without 'module.' prefix for compatibility
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({'model_state_dict': state_dict, 'epoch': epoch}, f'{SAVE_DIR}/checkpoints/epoch_{epoch+1}.pth')

    # Final save
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({'model_state_dict': state_dict, 'epoch': EPOCHS-1}, f'{SAVE_DIR}/checkpoints/final.pth')
    print(f'\nTraining complete! Best PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}, Loss: {best_loss:.4f}')
