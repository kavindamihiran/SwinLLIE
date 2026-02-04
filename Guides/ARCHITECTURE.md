# SwinIR Architecture

Detailed technical documentation for the pure SwinIR model for low-light image enhancement.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Configuration Options](#configuration-options)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SwinIR Model                                 │
│  Parameters: ~4M                                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input (B, 3, H, W)                                                  │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                         │
│  │    conv_first (3×3)     │ → Shallow features (B, 60, H, W)        │
│  └─────────────────────────┘                                         │
│       │                                                              │
│  ═════╪═════════════════ ENCODER ═══════════════════════════         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                         │
│  │    RSTB Stage 1         │ dim=60, 4 blocks, 6 heads              │
│  └─────────────────────────┘                                         │
│       │ ─────────────────────────────────────────────────┐            │
│       ▼ (downsample 2×)                                  │            │
│  ┌─────────────────────────┐                             │            │
│  │    RSTB Stage 2         │ dim=120, 4 blocks, 6 heads  │            │
│  └─────────────────────────┘                             │            │
│       │ ─────────────────────────────────────────┐       │            │
│       ▼ (downsample 2×)                          │       │            │
│  ┌─────────────────────────┐                     │       │            │
│  │    RSTB Stage 3         │ dim=240, 4 blocks   │       │            │
│  │ (Bottleneck)            │                     │       │            │
│  └─────────────────────────┘                     │       │            │
│       │                                          │       │            │
│  ═════╪═════════════════ DECODER ════════════════╪═══════╪═══         │
│       │                                          │       │            │
│       ▼ (upsample 2×)                            │       │            │
│  ┌─────────────────────────┐                     │       │            │
│  │   FeatureFusion         │ ◄───────────────────┘       │            │
│  │   + RSTB Stage 2'       │ Skip connection             │            │
│  └─────────────────────────┘                             │            │
│       │                                                  │            │
│       ▼ (upsample 2×)                                    │            │
│  ┌─────────────────────────┐                             │            │
│  │   FeatureFusion         │ ◄───────────────────────────┘            │
│  │   + RSTB Stage 1'       │ Skip connection                          │
│  └─────────────────────────┘                                         │
│       │                                                              │
│  ═════╪═════════════════ OUTPUT ═════════════════════════════         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                         │
│  │   conv_after + conv_last│ → (B, 3, H, W)                          │
│  └─────────────────────────┘                                         │
│       │                                                              │
│       ▼                                                              │
│       + ◄──────────────────────────────────────────────────          │
│  (Residual: Output = conv_last + Input)                              │
│                                                                      │
│  Output (B, 3, H, W)                                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Shallow Feature Extraction

**Purpose**: Convert input RGB image to feature representation.

```python
class ConvFirst(nn.Module):
    """Simple 3x3 convolution to extract initial features"""
    def __init__(self, in_channels=3, embed_dim=60):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

    def forward(self, x):
        # Input: (B, 3, H, W)
        # Output: (B, 60, H, W)
        return self.conv(x)
```

**Purpose**: Maps RGB pixels to a higher-dimensional feature space that Swin Transformer blocks can process.

---

### 2. WindowAttention

**Purpose**: Efficient self-attention within local windows.

```python
class WindowAttention(nn.Module):
    """
    Complexity: O(N × window_size²) instead of O(N²)

    For 128×128 image with window_size=8:
    - Full attention: 128² × 128² = 268M operations
    - Window attention: 128² × 8² = 1M operations (268× faster!)
    """
```

---

### 4. SwinTransformerBlock

**Purpose**: Single transformer block with window attention.

```
Input (B, H×W, C)
      │
      ▼
LayerNorm → WindowAttention → DropPath → + (residual)
      │
      ▼
LayerNorm → MLP → DropPath → + (residual)
      │
      ▼
Output (B, H×W, C)
```

Alternating between regular and shifted windows.

---

### 4. RSTB (Residual Swin Transformer Block)

**Purpose**: Stack of Swin Transformer blocks with residual connection.

```
Input
  │
  ▼
┌───────────────────────────┐
│ BasicLayer (N blocks)     │
│ W-MSA → SW-MSA → W-MSA... │
└───────────────────────────┘
  │
  ▼
Conv 3×3 (residual conv)
  │
  ▼
+ Input (residual)
  │
  ▼
Output
```

**Key insight**: The residual connection allows gradients to flow directly, making deep networks trainable.

---

### 5. FeatureFusion

**Purpose**: Fuse encoder skip connection with decoder features.

```python
class FeatureFusion(nn.Module):
    def forward(self, enc_feat, dec_feat):
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        fused = self.conv(concat)
        return enc_feat + dec_feat + fused
```

**Key insight**: Simple addition + learned fusion creates effective multi-scale feature integration.

---

## Data Flow

### Forward Pass Example

```python
# Input: (1, 3, 128, 128) normalized [0, 1]
x = torch.randn(1, 3, 128, 128)

# Step 1: Shallow features
shallow = model.conv_first(x)  # (1, 60, 128, 128)

# Step 2: Encoder - Multi-scale processing
enc1 = RSTB(shallow)                    # (1, 60, 128, 128)
enc2 = RSTB(downsample(enc1))           # (1, 120, 64, 64)
enc3 = RSTB(downsample(enc2))           # (1, 240, 32, 32)

# Step 3: Decoder with skip connections
dec2 = RSTB(fuse(enc2, upsample(enc3))) # (1, 120, 64, 64)
dec1 = RSTB(fuse(enc1, upsample(dec2))) # (1, 60, 128, 128)

# Step 4: Output reconstruction
out = conv_last(conv_after(dec1) + shallow) + x  # (1, 3, 128, 128)
```

**Flow**: Input → Extract Features → Encode (downsample) → Decode (upsample + skip) → Output

---

## Configuration Options

### Model Size Variants

| Variant | embed_dim | depths  | num_heads  | Parameters |
| ------- | --------- | ------- | ---------- | ---------- |
| Tiny    | 48        | [2,2,2] | [4,4,4]    | ~2M        |
| Small   | 60        | [4,4,4] | [6,6,6]    | ~4.7M      |
| Base    | 96        | [6,6,6] | [8,8,8]    | ~12M       |
| Large   | 128       | [8,8,8] | [12,12,12] | ~25M       |

### Key Parameters

| Parameter        | Default | Description                      |
| ---------------- | ------- | -------------------------------- |
| `window_size`    | 8       | Attention window size            |
| `mlp_ratio`      | 2.0     | MLP hidden dim = dim × mlp_ratio |
| `drop_path_rate` | 0.1     | Stochastic depth rate            |

---

## Memory Usage

### Training (batch_size=8, patch_size=96)

| GPU VRAM | Recommended Config |
| -------- | ------------------ |
| 4 GB     | batch=4, embed_dim=48 |
| 8 GB     | batch=8, embed_dim=60 |
| 12 GB    | batch=16, embed_dim=60 |

### Inference

- ~1.5 GB for 512×512 image
- ~3 GB for 1024×1024 image
- Automatic CPU fallback if OOM

---

## Comparison with Original SwinIR

| Aspect       | Original SwinIR  | Our Implementation                  |
| ------------ | ---------------- | ----------------------------------- |
| Task         | Super-resolution | Low-light enhancement               |
| Architecture | Single-scale     | U-Net encoder-decoder               |
| Attention    | Window only      | Window only (pure SwinIR)           |
| Loss         | L1               | Hybrid (L1+VGG+Color+Edge+Exposure) |
| Parameters   | ~12M (large)     | ~4.7M (efficient)                   |
| Training     | Patch-based      | Patch-based with data augmentation  |

---

## 📚 Related Documentation

| Document | Description |
|----------|-------------|
| [THEORY_GUIDE.md](THEORY_GUIDE.md) | Beginner-friendly theory explanation |
| [API_REFERENCE.md](API_REFERENCE.md) | Code API documentation |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Deployment instructions |
| [RESEARCH_PAPER_SWINLLIE.tex](RESEARCH_PAPER_SWINLLIE.tex) | Academic paper (LaTeX) |

---

## 📝 Citation

```bibtex
@article{swinllie2025,
  title={SwinLLIE: Swin Transformer for Low-Light Image Enhancement},
  author={Kavinda Mihiran},
  year={2025}
}
```

