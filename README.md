# Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement

A novel deep learning architecture combining SwinIR with illumination-guided attention for low-light image enhancement.

## 🌟 Key Features

- **Illumination Estimation Module (IEM)**: Automatically detects dark regions
- **Illumination-Guided Attention (IGAM)**: Applies stronger enhancement to dark areas while preserving bright regions
- **U-Net Architecture**: Multi-scale processing with skip connections
- **Hybrid Loss Function**: L1 + VGG Perceptual + Color Consistency + Smoothness

## 📁 Project Structure

```
SwinLLIE/
├── models/
│   └── network_swinllie.py   # Main model architecture
├── data/
│   └── lowlight_dataset.py   # Dataset loaders (LOL, generic, unpaired)
├── configs/
│   └── swinllie_lol.yaml     # Training configuration
├── losses.py                  # Hybrid loss functions
├── train_swinllie.py          # Training script
├── test_swinllie.py           # Inference script
├── requirements.txt           # Dependencies
└── venv/                      # Virtual environment
```

## 🚀 Quick Start

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Test Model
```bash
python models/network_swinllie.py
```

### 3. Download LOL Dataset
Download from: https://daooshee.github.io/BMVC2018website/

Extract to `datasets/LOL/` with structure:
```
datasets/LOL/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

### 4. Train
```bash
python train_swinllie.py --config configs/swinllie_lol.yaml
```

### 5. Inference
```bash
python test_swinllie.py --input your_image.jpg --checkpoint experiments/swinllie_lol/checkpoints/best.pth
```

## 📊 Model Info

- **Parameters**: ~4.8M
- **Input**: RGB images (any resolution)
- **Output**: Enhanced RGB images (same resolution)

## 📝 Citation

If you use this work, please cite:
```
@article{swinllie2025,
  title={Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement},
  author={Group 10},
  year={2025}
}
```

## 🙏 Acknowledgments

- Based on [SwinIR](https://github.com/JingyunLiang/SwinIR) architecture
- Inspired by Retinex theory for illumination estimation
