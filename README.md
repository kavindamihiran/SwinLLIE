# SwinIR: Pure Low-Light Image Enhancement

A clean implementation of SwinIR (Swin Transformer for Image Restoration) specifically designed for low-light image enhancement without additional attention mechanisms.

## 🌟 Key Features

- **Pure SwinIR Architecture**: Clean implementation of the core Swin Transformer for image restoration
- **Simple and Efficient**: No additional attention mechanisms - just the proven SwinIR approach
- **Multi-scale Processing**: Encoder-decoder with skip connections
- **Smart GPU/CPU Fallback**: Automatic detection with seamless fallback
- **~4M Parameters**: Efficient for training and inference

---

## 📁 Project Structure

```
SwinIR/
├── swinllie/                 # Main module
│   ├── models.py             # Pure SwinIR model (400 lines, well-commented)
│   ├── losses.py             # Loss functions
│   ├── data.py               # Dataset loaders
│   └── utils.py              # PSNR, SSIM metrics
├── configs/
│   └── swinllie_lol.yaml     # Training configuration
├── datasets/LOL/             # Dataset folder
├── experiments/              # Checkpoints and logs
├── Guides/                   # Documentation
│   ├── THEORY_GUIDE.md       # ⭐ Theory explanation
│   ├── ARCHITECTURE.md       # Model architecture details
│   └── FINE_TUNING_GUIDE.md  # Fine-tuning instructions
├── train.py                  # Training script
└── inference.py              # Inference script
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/kavindamihiran/SwinIR.git
cd SwinIR
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python -c "from swinllie import SwinLLIE; print('✓ Model loaded!')"
python swinllie/models.py   # Run model tests
python swinllie/losses.py   # Run loss tests
```

### 3. Train

```bash
python train.py --config configs/swinllie_lol.yaml
```

### 4. Inference

```bash
# Put images in test/ folder
python inference.py
# Results in test_results/
```

---

## 🧠 How It Works

### The Problem

Dark images have: low brightness, washed colors, noise, blurry edges.

### Our Solution

```
Input → IlluminationEstimator → Swin Encoder → Swin Decoder → Output
              ↓                      ↓              ↓
         dark_mask            Apply more      Skip connections
         (where to            enhancement     preserve details
          enhance)            in dark areas
```

### Key Innovation: SimpleIllumAttention

```python
# Adapts enhancement based on darkness level
output = features + gamma * (enhanced * dark_mask)
#        ↑          ↑           ↑          ↑
#    original   learnable   enhanced   where to
#    features    weight     features    apply
```

**Read more**: [📖 THEORY_GUIDE.md](Guides/THEORY_GUIDE.md)

---

## ⚙️ Model Architecture

| Component               | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `IlluminationEstimator` | 3-layer CNN estimates dark regions             |
| `SimpleIllumAttention`  | Channel + Spatial attention guided by darkness |
| `RSTB`                  | Residual Swin Transformer Block                |
| `SwinLLIE`              | Full U-Net with 3 encoder + decoder stages     |

**Parameters**: 6,488,071 (~6.5M)

---

## 📊 Loss Functions

| Loss     | Weight | Purpose                            |
| -------- | ------ | ---------------------------------- |
| L1       | 1.0    | Main reconstruction                |
| VGG      | 0.1    | Perceptual quality (prevents blur) |
| Color    | 0.5    | Color preservation                 |
| Edge     | 0.5    | Sharpness                          |
| Exposure | 0.5    | Prevent overexposure               |

Optional: Smoothness (0.01), SSIM (0.1)

---

## ⚙️ Configuration

Key settings in `configs/swinllie_lol.yaml`:

```yaml
model:
  embed_dim: 60
  depths: [4, 4, 4]
  num_heads: [6, 6, 6]
  window_size: 8
  use_igam: true # Enable illumination attention

training:
  batch_size: 4
  epochs: 100
  learning_rate: 0.0002
  use_amp: true # Mixed precision

loss:
  lambda_l1: 1.0
  lambda_vgg: 0.1
  lambda_color: 0.5
  lambda_edge: 0.5
  lambda_exposure: 0.5
```

---

## 🔧 Troubleshooting

| Problem           | Solution                          |
| ----------------- | --------------------------------- |
| Blurry outputs    | Increase `lambda_edge` to 1.0     |
| Gray/washed out   | Increase `lambda_color` to 1.0    |
| Overexposed spots | Increase `lambda_exposure` to 1.0 |
| GPU OOM           | Reduce `batch_size` to 2 or 1     |
| Training slow     | Enable `use_amp: true`            |

---

## 📚 Documentation

| Guide                                               | Description                                 |
| --------------------------------------------------- | ------------------------------------------- |
| [THEORY_GUIDE.md](Guides/THEORY_GUIDE.md)           | ⭐ **Start here!** Beginner-friendly theory |
| [ARCHITECTURE.md](Guides/ARCHITECTURE.md)           | Detailed architecture                       |
| [FINE_TUNING_GUIDE.md](Guides/FINE_TUNING_GUIDE.md) | Custom dataset training                     |

---

## 📝 Citation

```bibtex
@article{swinllie2025,
  title={Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement},
  author={Group 10},
  year={2025}
}
```

## 🙏 Acknowledgments

- [SwinIR](https://github.com/JingyunLiang/SwinIR) - Base architecture
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/) - Training data
- Retinex theory - Illumination estimation concept
