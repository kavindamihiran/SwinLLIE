# Understanding SwinIR for Low-Light Enhancement

A guide explaining the theory behind low-light image enhancement using pure Swin Transformers.

---

## Table of Contents

1. [The Low-Light Problem](#the-low-light-problem)
2. [Swin Transformer Approach](#swin-transformer-approach)
3. [Multi-Scale Processing](#multi-scale-processing)
4. [Pure SwinIR Architecture](#pure-swinir-architecture)
5. [Loss Functions Explained](#loss-functions-explained)
6. [Training Tips](#training-tips)

---

## The Low-Light Problem

### What happens in dark images?

When you take a photo in low light, several things go wrong:

```
Normal Light Image          Low Light Image
├── Good brightness    →    ├── Too dark (hard to see)
├── Rich colors        →    ├── Washed out colors
├── Sharp edges        →    ├── Blurry edges (noise)
└── Good contrast      →    └── Flat, muddy contrast
```

### The challenge

We want to **enhance** dark regions while **preserving** structure:

- Brighten dark areas and reveal details
- Maintain color fidelity
- Avoid amplifying noise
- Preserve natural-looking results

---

## Swin Transformer Approach

### Why Swin Transformers for images?

Traditional transformers work great for text, but images have special properties:

```
Text:     "The cat sat on the mat"
          ↓ (linear sequence)
Image:    2D grid of pixels with spatial relationships
          ↓ (local patterns + global context)
```

**Swin Transformer** solves this with:

1. **Local attention**: Look at small windows first
2. **Shifted windows**: Connect different regions
3. **Hierarchical**: Start small, grow bigger (like a pyramid)

### Window-based attention

Instead of looking at the entire image at once:

```
Full Attention (too expensive)     Window-based (efficient)
┌─────────────────────┐        ┌───┬───┬───┐
│ Every pixel         │  →     │ A │ B │ C │  Each window
│ looks at            │        ├───┼───┼───┤  attends to
│ every other pixel   │        │ D │ E │ F │  itself only
└─────────────────────┘        └───┴───┴───┘
```

**Efficiency gain:**

- Full attention: 128² × 128² = 268M operations
- Window attention (8×8): 128² × 8² = 1M operations
- **268× faster!**

---

## Multi-Scale Processing

### Encoder-Decoder with Skip Connections

Our model processes images at multiple scales:

```
Input (256×256)
    ↓ Stage 1: Local details (Window=8×8)
    ↓ Downsample ÷2
Features (128×128)
    ↓ Stage 2: Medium patterns
    ↓ Downsample ÷2
Features (64×64)
    ↓ Stage 3: Global context
    ↑ Upsample ×2
    ↑ + Skip connection from Stage 2
    ↑ Upsample ×2
    ↑ + Skip connection from Stage 1
Output (256×256)
```

Why this works:

- **Small scale**: Fine details, textures, edges
- **Medium scale**: Objects, patterns
- **Large scale**: Overall structure, global context

---

## Pure SwinIR Architecture

### Complete Architecture

```
┌────────────────────────────────────────────────┐
│              SwinIR for Low-Light               │
├────────────────────────────────────────────────┤
│  Input Image (B, 3, H, W)                       │
│       ↓                                         │
│  ┌──────────────┐                               │
│  │ Conv First   │ → Initial feature extraction  │
│  └──────────────┘                               │
│       ↓                                         │
│  ══════════════ ENCODER ════════════════        │
│       ↓                                         │
│  ┌──────────────┐                               │
│  │ RSTB Stage 1 │ → dim=60, 4 blocks            │
│  └──────────────┘                               │
│       ↓ (downsample 2×)                         │
│  ┌──────────────┐                               │
│  │ RSTB Stage 2 │ → dim=120, 4 blocks           │
│  └──────────────┘                               │
│       ↓ (downsample 2×)                         │
│  ┌──────────────┐                               │
│  │ RSTB Stage 3 │ → dim=240, 4 blocks           │
│  └──────────────┘ (Bottleneck)                  │
│       ↓                                         │
│  ══════════════ DECODER ════════════════        │
│       ↓ (upsample 2×, skip connection)          │
│  ┌──────────────┐                               │
│  │ RSTB Stage 2'│ → Refine with encoder feats   │
│  └──────────────┘                               │
│       ↓ (upsample 2×, skip connection)          │
│  ┌──────────────┐                               │
│  │ RSTB Stage 1'│ → Final refinement            │
│  └──────────────┘                               │
│       ↓                                         │
│  ┌──────────────┐                               │
│  │  Conv Output │ → RGB reconstruction          │
│  └──────────────┘                               │
│       ↓                                         │
│  Output Image (B, 3, H, W)                      │
└────────────────────────────────────────────────┘
```

### Key Components

1. **WindowAttention**: Efficient self-attention in local windows
2. **SwinTransformerBlock**: Alternates W-MSA and SW-MSA (shifted windows)
3. **RSTB**: Residual Swin Transformer Block (stack of Swin blocks + conv)
4. **FeatureFusion**: Combines encoder and decoder features via skip connections

---

## Loss Functions Explained

### Hybrid Loss Composition

We combine 5 different losses for comprehensive training:

#### 1. L1 Loss (Main Reconstruction)

```python
L1 = mean(|predicted - target|)
```

- **Weight**: 1.0 (highest)
- **Purpose**: Match pixel values accurately
- **Effect**: Overall brightness and color

#### 2. VGG Perceptual Loss

```python
VGG = mean(|VGG_features(pred) - VGG_features(target)|)
```

- **Weight**: 0.1
- **Purpose**: Match high-level features (textures, patterns)
- **Effect**: Natural-looking, perceptually similar results

#### 3. Color Consistency Loss

```python
Color = mean(|mean_color(pred) - mean_color(target)|)
```

- **Weight**: 0.5
- **Purpose**: Preserve color tones
- **Effect**: Prevents color shifts (too blue/yellow)

#### 4. Edge Loss

```python
Edge = mean(|sobel(pred) - sobel(target)|)
```

- **Weight**: 0.5
- **Purpose**: Preserve sharp edges
- **Effect**: Clear boundaries, no blur

#### 5. Exposure Control Loss

```python
Exposure = mean(ReLU(pred - 0.95)²) + preserve_bright_regions
```

- **Weight**: 0.5
- **Purpose**: Prevent overexposure
- **Effect**: No blown-out highlights

### Total Loss

```python
Total = 1.0×L1 + 0.1×VGG + 0.5×Color + 0.5×Edge + 0.5×Exposure
```

---

## Training Tips

### 1. Start with Good Hyperparameters

```yaml
model:
  embed_dim: 60
  depths: [4, 4, 4]
  num_heads: [6, 6, 6]
  window_size: 8

training:
  batch_size: 4
  learning_rate: 0.0002
  epochs: 100
  patch_size: 96
```

### 2. Data Augmentation

```python
augmentations = [
    RandomCrop(96),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation90()
]
```

### 3. Learning Rate Schedule

```python
# Cosine annealing with warmup
warmup_epochs = 5
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

### 4. Monitor Metrics

Track during training:

- **Loss**: Should decrease steadily
- **PSNR**: Higher is better (>20 is good)
- **SSIM**: Closer to 1 is better (>0.8 is good)

### 5. GPU Memory Management

If OOM errors:

```python
# Reduce batch size
batch_size = 2

# Reduce patch size
patch_size = 64

# Use gradient checkpointing
use_checkpoint = True
```

---

## Code Examples

### Quick Start

```python
from swinllie import SwinLLIE, HybridLoss
import torch

# 1. Create model
model = SwinLLIE(
    embed_dim=60,
    depths=[4, 4, 4],
    num_heads=[6, 6, 6],
    window_size=8
)

# 2. Load image
low_light_image = load_image('dark.jpg')  # (B, 3, H, W) in [0,1]

# 3. Enhance
with torch.no_grad():
    enhanced = model(low_light_image)

# 4. Training loss
criterion = HybridLoss()
loss, breakdown = criterion(enhanced, ground_truth)
```

### Model Components

```python
# Pure SwinIR architecture
swinllie/
├── models.py
│   ├── WindowAttention          # Core Swin attention
│   ├── SwinTransformerBlock     # Transformer block
│   ├── BasicLayer               # Stack of blocks
│   ├── RSTB                     # Residual Swin block
│   ├── FeatureFusion            # Skip connection fusion
│   └── SwinLLIE                 # Main model
│
├── losses.py
│   ├── L1Loss                   # Reconstruction
│   ├── VGGPerceptualLoss        # Perceptual quality
│   ├── ColorConsistencyLoss     # Color preservation
│   ├── EdgeLoss                 # Sharpness
│   ├── ExposureControlLoss      # Prevent overexposure
│   └── HybridLoss               # Combined
│
├── data.py                      # Dataset loaders
└── utils.py                     # PSNR, SSIM metrics
```

---

## Summary: Key Concepts

1. **Low-light challenge** = Dark images lose details, have noise, washed colors

2. **Window-based attention** = Efficient self-attention in local windows (268× faster)

3. **Multi-scale processing** = Encoder-decoder handles different scales

4. **Pure SwinIR** = End-to-end learning with proven architecture

5. **Hybrid loss** = L1 + VGG + Color + Edge + Exposure for comprehensive training

6. **Skip connections** = Preserve fine details through direct pathways

---

## Further Reading

- Original SwinIR paper: "SwinIR: Image Restoration Using Swin Transformer"
- Swin Transformer: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- Dataset: LOL (Low-Light) Dataset

---

**Happy Training! 🚀**
