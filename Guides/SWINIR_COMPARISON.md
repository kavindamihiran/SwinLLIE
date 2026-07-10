# Swin-LLIE vs Original SwinIR: Key Differences

This document compares our **Swin-LLIE** implementation with the [original SwinIR repository](https://github.com/JingyunLiang/SwinIR).

---

## 🎯 Purpose / Task Focus

| Aspect | Original SwinIR | Swin-LLIE |
|--------|-----------------|-----------|
| **Primary Task** | Image Super-Resolution (2x/3x/4x), Denoising, JPEG Artifact Removal | Low-Light Image Enhancement |
| **Upscaling** | ✅ Supports 2x, 3x, 4x, 8x | ❌ No upscaling (1:1 resolution) |
| **Model Class** | `SwinIR` | `SwinLLIE` |

---

## 🏗️ Architecture Changes

### Original SwinIR: Flat Stack
```
Input → Conv → [RSTB × N] → Conv → Upsample → Output
```

### Swin-LLIE: U-Net Encoder-Decoder
```
Input → Conv → Encoder (3 stages, downsample)
                    ↓
              Bottleneck (RSTB)
                    ↓
        Decoder (3 stages, upsample + skip fusion)
                    ↓
              Conv → Output
```

| Component | Original SwinIR | Swin-LLIE |
|-----------|-----------------|-----------|
| **Architecture** | Flat RSTB stack | U-Net encoder-decoder |
| **Multi-scale** | Single scale | ✅ 3-level pyramid |
| **Skip Connections** | None | ✅ Encoder-to-decoder fusion |
| **Upsampler** | PixelShuffle / Nearest+Conv | Removed (not needed) |

---

## 🔧 Module-Level Differences

| Module | Original | Swin-LLIE |
|--------|----------|-----------|
| `WindowAttention` | Has `qk_scale` parameter | Removed (uses default scaling) |
| `Mlp` | Configurable `act_layer` | Hardcoded GELU |
| `BasicLayer` | Has `downsample` parameter | Removed from BasicLayer |
| `RSTB` | Supports `1conv` / `3conv` + downsample | Simplified (always `1conv`) |
| **`FeatureFusion`** | ❌ Not present | ✅ NEW: Skip connection fusion |
| **`PatchMerging`** | Present but unused | Used for encoder downsampling |

---

## ⚙️ Default Hyperparameters

| Parameter | Original SwinIR | Swin-LLIE |
|-----------|-----------------|-----------|
| `embed_dim` | 96 | 60 |
| `depths` | `[6, 6, 6, 6]` | `[4, 4, 4]` |
| `num_heads` | `[6, 6, 6, 6]` | `[6, 6, 6]` |
| `window_size` | 7 | 8 |
| `mlp_ratio` | 4.0 | 2.0 |
| **Total Parameters** | ~11.8M | ~4.7M |

---

## 📉 Loss Functions

| Loss Type | Original SwinIR | Swin-LLIE |
|-----------|-----------------|-----------|
| L1 Loss | ✅ | ✅ |
| VGG Perceptual | ❌ | ✅ |
| Color Consistency | ❌ | ✅ (cosine similarity) |
| Edge/Sharpness | ❌ | ✅ (Sobel-based) |
| Exposure Control | ❌ | ✅ (prevent overexposure) |
| SSIM Loss | ❌ | ✅ (optional) |

---

## 🚀 Training Pipeline

| Feature | Original SwinIR | Swin-LLIE |
|---------|-----------------|-----------|
| Training Script | External (BasicSR) | Self-contained `train.py` |
| Config Format | Python dict | YAML |
| Mixed Precision | Not included | ✅ `use_amp: true` |
| GPU Auto-Fallback | Not included | ✅ Auto CPU fallback |
| Dataset Loader | External | Built-in (`data.py`) |

---

## 📁 Repository Structure

```
Original SwinIR:                    Swin-LLIE:
├── models/                         ├── swinllie/
│   └── network_swinir.py          │   ├── models.py
├── utils/                         │   ├── losses.py    ← NEW
├── testsets/                      │   ├── data.py      ← NEW
├── main_test_swinir.py            │   └── utils.py     ← NEW
└── predict.py                     ├── train.py        ← NEW
                                   ├── inference.py    ← NEW
                                   ├── configs/        ← NEW
                                   └── Guides/         ← NEW (docs)
```

---

## 📝 Summary

Swin-LLIE is a **purpose-built derivative** of SwinIR adapted for low-light image enhancement:

1. **Task Adaptation**: Converted from SR to LLIE (no upscaling)
2. **Architecture Redesign**: U-Net encoder-decoder with skip connections
3. **Simplified Components**: Removed unused parameters, reduced model size by ~60%
4. **LLIE-Specific Losses**: Added color, edge, and exposure control losses
5. **Complete Training Pipeline**: Self-contained with YAML configs
6. **Beginner-Friendly**: Added comprehensive documentation

---

## 🔗 References

- [Original SwinIR Repository](https://github.com/JingyunLiang/SwinIR)
- [SwinIR Paper](https://arxiv.org/abs/2108.10257)
