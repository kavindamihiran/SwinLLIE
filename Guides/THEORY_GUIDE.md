# Understanding Swin-LLIE: Theory and Architecture

A beginner-friendly guide explaining the theory behind low-light image enhancement with Swin Transformers.

---

## Table of Contents
1. [The Low-Light Problem](#the-low-light-problem)
2. [Retinex Theory](#retinex-theory)
3. [How Attention Works](#how-attention-works)
4. [Our Simplified Architecture](#our-simplified-architecture)
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

We want to **enhance** dark regions while **preserving** bright regions:
- Dark areas → Brighten and reveal details
- Bright areas → Keep as-is (avoid overexposure)

---

## Retinex Theory

### The basic idea

Retinex theory (Retina + Cortex) says every image is made of two parts:

```
Image = Reflectance × Illumination
  I   =     R      ×      L

Where:
- R = What things actually look like (colors, textures)
- L = How much light is shining on them
```

### Why this matters

In a dark image, the illumination (L) is low. If we can:
1. **Estimate** what L is (which parts are dark)
2. **Boost** L in dark regions
3. **Keep** R unchanged

Then we get a properly lit image!

### Our implementation

```python
class IlluminationEstimator(nn.Module):
    """
    Estimates which regions are dark vs bright.
    
    Output:
        dark_mask = 1 where dark (needs enhancement)
        dark_mask = 0 where bright (preserve as-is)
    """
```

---

## How Attention Works

### Self-Attention (The Key Concept)

Imagine reading a sentence: "The cat sat on the mat because **it** was tired."

To understand what "it" refers to, you look back at the whole sentence and find "cat" is most relevant. This is **attention**.

For images, we do the same thing - each pixel "looks at" other pixels to understand context.

### The Math (Simplified)

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What information do I have?"
- V (Value): "What should I output?"
- d: Dimension (for scaling)
```

### Window-based Attention

**Problem**: Looking at ALL pixels is expensive (O(n²) complexity)

**Solution**: Only look at nearby pixels in a **window** (8×8 by default)

```
Image:                    Windows:
┌─────────────────┐       ┌───┬───┬───┬───┐
│                 │       │ 1 │ 2 │ 3 │ 4 │
│                 │   →   ├───┼───┼───┼───┤
│                 │       │ 5 │ 6 │ 7 │ 8 │
│                 │       ├───┼───┼───┼───┤
└─────────────────┘       │...│...│...│...│
                          └───┴───┴───┴───┘

Each window does attention independently
Complexity: O(n) instead of O(n²)
```

### Shifted Windows

**Problem**: Windows don't talk to each other

**Solution**: Every other layer, shift the windows by half

```
Layer 1:          Layer 2 (shifted):
┌───┬───┐         ╔═══╦═══╗
│ A │ B │         ║ * │ * ║  ← Windows now overlap
├───┼───┤    →    ╠═══╬═══╣     with previous boundaries
│ C │ D │         ║ * │ * ║
└───┴───┘         ╚═══╩═══╝

This lets information flow across the whole image!
```

---

## Our Simplified Architecture

### Overview

```
Input Image (dark)
       ↓
┌──────────────────┐
│  Illumination    │ → Estimates dark_mask (where to enhance)
│    Estimator     │
└──────────────────┘
       ↓
┌──────────────────┐
│  Shallow Feature │ → 3×3 conv to extract initial features
│    Extraction    │
└──────────────────┘
       ↓
┌──────────────────┐
│     ENCODER      │ → 3 stages with Swin Transformer + Illum Attention
│  (downsample 2x) │
└──────────────────┘
       ↓
┌──────────────────┐
│     DECODER      │ → 3 stages with skip connections
│  (upsample 2x)   │
└──────────────────┘
       ↓
┌──────────────────┐
│  Output + Skip   │ → Add to input for residual learning
└──────────────────┘
       ↓
  Enhanced Image
```

### Key Component: SimpleIllumAttention

This is the **core innovation** - adaptive enhancement based on darkness:

```python
class SimpleIllumAttention(nn.Module):
    """
    Enhances dark regions more than bright regions.
    
    Step 1: Channel Attention
            Which feature channels are important?
            
    Step 2: Spatial Modulation  
            Combine features with dark_mask to know WHERE to enhance
            
    Step 3: Apply with learnable weight
            output = features + gamma * enhanced
            (gamma starts at 0, learns during training)
    """
```

**The key equation:**

```
output = features + γ × (enhanced × dark_mask)
         ↑          ↑         ↑         ↑
     original  learnable  enhanced  where to
     features   weight    features  apply
```

### Why residual connections?

```
Output = Input + Enhancement
```

This means:
- Network only needs to learn the **difference** (easier!)
- If enhancement is bad, it outputs 0 (keeps original)
- Gradients flow easily during training

---

## Loss Functions Explained

We use 5 losses that each handle a different aspect:

### 1. L1 Loss (Reconstruction)

```
L1 = mean(|prediction - target|)
```

**What it does**: Ensures pixels are close to ground truth
**Why we need it**: Main signal for brightness correction

### 2. VGG Perceptual Loss

```
L_vgg = ||VGG_features(pred) - VGG_features(target)||²
```

**What it does**: Compares high-level features, not just pixels
**Why we need it**: Prevents blurry outputs, preserves textures

```
L1 Only:                With VGG:
┌────────────────┐      ┌────────────────┐
│  Blurry, but   │      │  Sharp with    │
│  correct avg   │  →   │  proper        │
│  brightness    │      │  textures      │
└────────────────┘      └────────────────┘
```

### 3. Color Consistency Loss

```
L_color = 1 - cosine_similarity(pred, target)
```

**What it does**: Ensures colors match (regardless of brightness)
**Why we need it**: Prevents washed-out, grayish outputs

```
Without Color Loss:     With Color Loss:
┌────────────────┐      ┌────────────────┐
│   Grayish,     │      │   Vibrant,     │
│   desaturated  │  →   │   colorful     │
└────────────────┘      └────────────────┘
```

### 4. Edge Loss

```
L_edge = ||Sobel(pred) - Sobel(target)||₁
```

**What it does**: Ensures edges are sharp
**Why we need it**: Maintains structural sharpness

```
Sobel filters detect edges:
[-1  0  1]     [-1 -2 -1]
[-2  0  2]     [ 0  0  0]
[-1  0  1]     [ 1  2  1]
 horizontal     vertical
```

### 5. Exposure Control Loss

```
L_exp = overexposure_penalty + bright_region_preservation
```

**What it does**: Prevents blowing out bright regions
**Why we need it**: Avoids white patches in already-bright areas

### Combined Loss

```
L_total = 1.0×L1 + 0.1×VGG + 0.5×Color + 0.5×Edge + 0.5×Exposure
```

---

## Training Tips

### 1. Learning Rate

```
Start: 0.0002
End:   0.000001 (cosine annealing)
Warmup: 5 epochs
```

Use **warmup** to avoid early training instability.

### 2. Batch Size

```
GPU Memory    Recommended Batch Size
4 GB          1-2
8 GB          4
12+ GB        8
```

### 3. Patch Size

```
Training:  96×96 patches (random crops)
Inference: Full resolution (with padding)
```

Training on patches is faster and provides data augmentation.

### 4. Common Issues

| Problem | Solution |
|---------|----------|
| Blurry outputs | Increase `lambda_edge` |
| Gray/washed out | Increase `lambda_color` |
| Overexposed spots | Increase `lambda_exposure` |
| Noisy outputs | Decrease learning rate |
| Training unstable | Enable gradient clipping |

---

## Code Walkthrough

### File Structure

```
swinllie/
├── models.py   # Model architecture
│   ├── IlluminationEstimator  # Estimates dark regions
│   ├── SimpleIllumAttention   # Adaptive enhancement attention
│   ├── WindowAttention        # Swin Transformer attention
│   ├── SwinTransformerBlock   # Basic transformer block
│   ├── RSTB                   # Residual Swin block + illum attention
│   └── SwinLLIE               # Full model
│
├── losses.py   # Loss functions
│   ├── L1Loss                 # Pixel reconstruction
│   ├── VGGPerceptualLoss      # Feature matching
│   ├── ColorConsistencyLoss   # Color preservation
│   ├── EdgeLoss               # Sharpness
│   ├── ExposureControlLoss    # Overexposure prevention
│   └── HybridLoss             # Combined loss
│
├── data.py     # Dataset loaders
└── utils.py    # PSNR, SSIM metrics
```

### Quick Start Example

```python
import torch
from swinllie import SwinLLIE, HybridLoss

# Create model
model = SwinLLIE(
    embed_dim=60,
    depths=[4, 4, 4],
    num_heads=[6, 6, 6],
    window_size=8,
    use_igam=True  # Enable illumination attention
)

# Create loss
criterion = HybridLoss()

# Forward pass
low_light_image = torch.rand(1, 3, 128, 128)  # B, C, H, W
enhanced = model(low_light_image)

# Compute loss (during training)
target = torch.rand(1, 3, 128, 128)
illum, dark, bright = model.get_illumination_map(low_light_image)
loss, breakdown = criterion(enhanced, target, illum, bright)
```

---

## Further Reading

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [SwinIR Paper](https://arxiv.org/abs/2108.10257)
- [Retinex Theory](https://en.wikipedia.org/wiki/Color_constancy)
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)

---

## Summary

1. **Low-light enhancement** = brighten dark regions, protect bright regions
2. **Retinex theory** = Image = Reflectance × Illumination
3. **Window attention** = efficient self-attention in local windows
4. **Our key innovation** = SimpleIllumAttention adapts enhancement spatially
5. **5 losses** = L1, VGG, Color, Edge, Exposure work together
