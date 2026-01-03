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

**Purpose**: Convert input image to feature representation. # Simple 3-layer CNN
self.net = nn.Sequential(
nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(hidden_dim, 1, 3, padding=1),
nn.Sigmoid()
)

    def forward(self, x):
        # Combine rough (max RGB) + refined (learned)
        rough = torch.max(x, dim=1, keepdim=True)[0]
        refined = self.net(x)
        illum_map = 0.4 * rough + 0.6 * refined
        dark_mask = 1.0 - illum_map
        bright_mask = clamp((illum_map - 0.6) / 0.4)
        return illum_map, dark_mask, bright_mask

````

**Outputs**:
- `illum_map`: (B, 1, H, W) - brightness level (0=dark, 1=bright)
- `dark_mask`: (B, 1, H, W) - where to enhance (1=dark)
- `bright_mask`: (B, 1, H, W) - where to protect (1=bright)

---

### 2. SimpleIllumAttention

**Purpose**: Adapt enhancement strength based on darkness level.

```python
class SimpleIllumAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        # Channel attention (SE-style)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

        # Spatial modulation
        self.spatial_mod = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),  # +1 for dark_mask
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

        # Learnable blend (starts at 0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features, dark_mask, bright_mask=None):
        # Channel attention
        ch_att = self.channel_att(features)

        # Spatial modulation with dark mask
        combined = torch.cat([features, dark_mask], dim=1)
        spatial = self.spatial_mod(combined)
        enhanced = spatial * ch_att * (0.5 + 0.5 * dark_mask)

        # Protect bright regions
        if bright_mask is not None:
            enhanced = enhanced * (1.0 - 0.7 * bright_mask)

        # Residual with learnable weight
        return features + self.gamma * enhanced
````

**Key insight**: `gamma` starts at 0, so initially the network just passes features through. During training, it learns how much enhancement to apply.

---

### 3. WindowAttention

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

### 5. RSTB (Residual Swin Transformer Block)

**Purpose**: Stack of transformer blocks + illumination attention.

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
SimpleIllumAttention (if enabled)
  │
  ▼
+ Input (residual)
  │
  ▼
Output
```

---

### 6. FeatureFusion

**Purpose**: Fuse encoder skip connection with decoder features.

```python
class FeatureFusion(nn.Module):
    def forward(self, enc_feat, dec_feat):
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.gate(concat)  # Learned attention
        fused = self.conv(concat)
        return gate * enc_feat + (1 - gate) * dec_feat + fused
```

---

## Data Flow

### Forward Pass Example

```python
# Input: (1, 3, 128, 128) normalized [0, 1]
x = torch.randn(1, 3, 128, 128)

# Step 1: Illumination estimation
illum, dark, bright = model.illum_estimator(x)
# dark: (1, 1, 128, 128) - high where dark

# Step 2: Shallow features
shallow = model.conv_first(x)  # (1, 60, 128, 128)

# Step 3: Encoder
enc1 = RSTB(shallow, dark_mask=dark)        # (1, 60, 128, 128)
enc2 = RSTB(downsample(enc1), dark_mask)    # (1, 120, 64, 64)
enc3 = RSTB(downsample(enc2), dark_mask)    # (1, 240, 32, 32)

# Step 4: Decoder with skip connections
dec2 = RSTB(fuse(enc2, upsample(enc3)))     # (1, 120, 64, 64)
dec1 = RSTB(fuse(enc1, upsample(dec2)))     # (1, 60, 128, 128)

# Step 5: Output
out = conv_last(conv_after(dec1) + shallow) + x  # (1, 3, 128, 128)
```

---

## Configuration Options

### Model Size Variants

| Variant | embed_dim | depths  | num_heads  | Parameters |
| ------- | --------- | ------- | ---------- | ---------- |
| Tiny    | 48        | [2,2,2] | [4,4,4]    | ~2M        |
| Small   | 60        | [4,4,4] | [6,6,6]    | ~6.5M      |
| Base    | 96        | [6,6,6] | [8,8,8]    | ~15M       |
| Large   | 128       | [8,8,8] | [12,12,12] | ~30M       |

### Key Parameters

| Parameter        | Default | Description                      |
| ---------------- | ------- | -------------------------------- |
| `window_size`    | 8       | Attention window size            |
| `mlp_ratio`      | 2.0     | MLP hidden dim = dim × mlp_ratio |
| `drop_path_rate` | 0.1     | Stochastic depth rate            |
| `use_igam`       | True    | Enable illumination attention    |

---

## Memory Usage

### Training (batch_size=4, patch_size=96)

| GPU VRAM | Recommended Config    |
| -------- | --------------------- |
| 4 GB     | batch=1, embed_dim=48 |
| 8 GB     | batch=4, embed_dim=60 |
| 12 GB    | batch=8, embed_dim=96 |

### Inference

- ~1.5 GB for 512×512 image
- ~3 GB for 1024×1024 image
- Automatic CPU fallback if OOM

---

## Differences from Original SwinIR

| Aspect        | Original SwinIR  | Swin-LLIE                             |
| ------------- | ---------------- | ------------------------------------- |
| Task          | Super-resolution | Low-light enhancement                 |
| Attention     | Window only      | Window + Illumination-guided          |
| Architecture  | Single-scale     | U-Net encoder-decoder                 |
| Normalization | LayerNorm only   | LayerNorm + InstanceNorm in attention |
| Loss          | L1               | Hybrid (L1+VGG+Color+Edge+Exposure)   |
| Parameters    | ~12M (large)     | ~6.5M (efficient)                     |
