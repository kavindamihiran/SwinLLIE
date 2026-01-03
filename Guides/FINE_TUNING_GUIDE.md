# Fine-Tuning Guide for Swin-LLIE

A practical guide to fine-tuning the simplified Swin-LLIE model on custom datasets.

---

## Table of Contents
1. [Quick Start Fine-Tuning](#quick-start-fine-tuning)
2. [Preparing Your Dataset](#preparing-your-dataset)
3. [Training Configuration](#training-configuration)
4. [Advanced Techniques](#advanced-techniques)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start Fine-Tuning

### Step 1: Load Pretrained Weights

```python
from swinllie import SwinLLIE
import torch

# Create model with same architecture as pretrained
model = SwinLLIE(
    embed_dim=60,
    depths=[4, 4, 4],
    num_heads=[6, 6, 6],
    window_size=8,
    use_igam=True
)

# Load pretrained weights
checkpoint = torch.load('pretrained.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
```

### Step 2: Update Config

```yaml
# configs/finetune_custom.yaml
training:
  learning_rate: 0.00002  # 10x LOWER than training from scratch!
  epochs: 50              # Fewer epochs needed
  batch_size: 4

dataset:
  root_dir: "./datasets/your_custom_data"

resume:
  enabled: true
  checkpoint_path: "./pretrained/best.pth"
```

### Step 3: Run Training

```bash
python train.py --config configs/finetune_custom.yaml
```

---

## Preparing Your Dataset

### Required Structure

```
datasets/YourData/
├── train/
│   ├── low/           # Dark input images
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ...
│   └── high/          # Normal-light target images
│       ├── 001.png    # Same filename as low!
│       ├── 002.png
│       └── ...
└── test/
    ├── low/
    └── high/
```

### Image Requirements

| Aspect | Requirement |
|--------|-------------|
| Format | PNG, JPG, BMP |
| Mode | RGB (3 channels) |
| Size | Any (auto-cropped to patches) |
| Range | [0, 255] or [0, 1] |
| Pairs | Same filename in low/ and high/ |

### Dataset Size Guidelines

| Size | Strategy |
|------|----------|
| < 100 pairs | Fine-tune, heavy augmentation |
| 100-500 pairs | Fine-tune, moderate augmentation |
| > 500 pairs | Can train from scratch |

---

## Training Configuration

### Learning Rate (Most Important!)

```yaml
# Training from scratch
learning_rate: 0.0002   # 2e-4

# Fine-tuning (10x lower!)
learning_rate: 0.00002  # 2e-5
```

**Why lower?** Pretrained weights are already good - we don't want to destroy them with aggressive updates.

### Loss Weights for Different Goals

```yaml
# Default (balanced, optimized for fine details)
lambda_l1: 1.0
lambda_vgg: 0.15
lambda_color: 0.5
lambda_edge: 0.6
lambda_detail: 0.3     # NEW: preserves micro-textures
lambda_exposure: 0.5

# Sharper outputs
lambda_edge: 1.0       # Increase edge weight
lambda_detail: 0.5     # Increase detail weight

# Better colors
lambda_color: 1.0      # Increase color weight

# Preserve fine details (textures, patterns)
lambda_detail: 0.5     # Increase for micro-texture preservation

# Prevent overexposure
lambda_exposure: 1.0   # Increase exposure weight
```

### Batch Size vs GPU Memory

| GPU VRAM | Batch Size | Patch Size |
|----------|------------|------------|
| 4 GB     | 1-2        | 64         |
| 8 GB     | 4          | 96         |
| 12 GB    | 8          | 128        |
| 24 GB    | 16         | 128        |

---

## Advanced Techniques

### 1. Layer Freezing

Freeze early layers, train only later layers:

```python
# Freeze illumination estimator (preserve learned brightness detection)
for param in model.illum_estimator.parameters():
    param.requires_grad = False

# Freeze encoder (preserve feature extraction)
for param in model.encoder_layers.parameters():
    param.requires_grad = False

# Train only decoder and output layers
for param in model.decoder_layers.parameters():
    param.requires_grad = True
for param in model.conv_last.parameters():
    param.requires_grad = True
```

### 2. Progressive Unfreezing

Start frozen, gradually unfreeze:

```python
def unfreeze_layers(model, epoch, total_epochs):
    """Unfreeze layers progressively."""
    if epoch < total_epochs * 0.3:
        # First 30%: Only output layers
        freeze_all_except(model, ['conv_last', 'conv_after'])
    elif epoch < total_epochs * 0.6:
        # Next 30%: Add decoder
        freeze_all_except(model, ['decoder_layers', 'conv_last', 'conv_after'])
    else:
        # Final 40%: Train all
        for param in model.parameters():
            param.requires_grad = True

# In training loop
for epoch in range(EPOCHS):
    unfreeze_layers(model, epoch, EPOCHS)
    # ... train ...
```

### 3. Different Learning Rates per Layer

```python
param_groups = [
    # Frozen
    {'params': model.illum_estimator.parameters(), 'lr': 0},
    
    # Low LR (preserve pretrained)
    {'params': model.encoder_layers.parameters(), 'lr': 1e-5},
    
    # Normal LR (adapt to new data)
    {'params': model.decoder_layers.parameters(), 'lr': 2e-5},
    
    # Higher LR (task-specific)
    {'params': model.conv_last.parameters(), 'lr': 5e-5},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
```

### 4. Data Augmentation

For paired low-light enhancement, use **geometric** augmentations only:

```python
import torchvision.transforms.functional as TF
import random

def augment_pair(low, high):
    """Apply same transforms to both images."""
    
    # Random horizontal flip
    if random.random() > 0.5:
        low = TF.hflip(low)
        high = TF.hflip(high)
    
    # Random vertical flip
    if random.random() > 0.5:
        low = TF.vflip(low)
        high = TF.vflip(high)
    
    # Random 90° rotation
    angle = random.choice([0, 90, 180, 270])
    low = TF.rotate(low, angle)
    high = TF.rotate(high, angle)
    
    return low, high

# DON'T use:
# - ColorJitter (destroys low-light characteristics)
# - Brightness adjustment (that's what we're learning!)
# - Contrast changes
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss explodes to NaN | LR too high | Lower to 1e-5 |
| No improvement | LR too low | Increase to 5e-5 |
| Forgetting pretrained | LR too high | Freeze early layers |
| Overfitting | Dataset too small | More augmentation |
| Blurry outputs | VGG/edge too low | Increase weights |

### Monitoring Tips

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 10:
            print(f"WARNING: {name} has large grad: {grad_norm}")
        if grad_norm < 1e-7:
            print(f"WARNING: {name} has vanishing grad")
```

### Validation Metrics

| Metric | Good Value | Meaning |
|--------|------------|---------|
| PSNR | > 20 dB | Pixel accuracy |
| SSIM | > 0.8 | Structural similarity |
| LPIPS | < 0.3 | Perceptual quality |

---

## Example: Fine-Tune for Indoor Scenes

```yaml
# configs/finetune_indoor.yaml
model:
  embed_dim: 60
  depths: [4, 4, 4]
  num_heads: [6, 6, 6]
  window_size: 8
  use_igam: true

dataset:
  name: "lol"
  root_dir: "./datasets/IndoorScenes"
  patch_size: 96
  num_workers: 4

training:
  batch_size: 4
  epochs: 50
  learning_rate: 0.00002  # Fine-tuning LR
  weight_decay: 0.0001
  use_amp: true
  grad_clip: 1.0
  save_dir: "./experiments/indoor_finetune"

loss:
  lambda_l1: 1.0
  lambda_vgg: 0.15
  lambda_color: 0.5
  lambda_edge: 0.6
  lambda_detail: 0.3    # NEW: micro-texture preservation
  lambda_exposure: 0.5

resume:
  enabled: true
  checkpoint_path: "./experiments/pretrained/best.pth"
```

```bash
python train.py --config configs/finetune_indoor.yaml
```

---

## Quick Reference

| Setting | From Scratch | Fine-Tuning |
|---------|--------------|-------------|
| Learning Rate | 2e-4 | 2e-5 |
| Epochs | 100+ | 30-50 |
| Batch Size | 4-8 | 2-4 |
| Warmup | 5 epochs | 2 epochs |
| Early Layers | Train | Freeze/low LR |
| Late Layers | Train | Train |

---

## Further Reading

- [THEORY_GUIDE.md](THEORY_GUIDE.md) - Understanding the architecture
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- Original paper: [Transfer Learning Survey](https://arxiv.org/abs/1808.01974)
