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
├── experiments/              # Saved checkpoints and logs
│   └── swinllie_lol/
│       ├── checkpoints/      # Model checkpoints (.pth files)
│       ├── logs/             # TensorBoard logs
│       └── val_images/       # Validation outputs
├── losses.py                 # Hybrid loss functions
├── train_swinllie.py         # Training script
├── test_swinllie.py          # Inference script
└── requirements.txt          # Dependencies
```

---

## 🚀 Quick Setup

### Step 1: Clone and Create Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SwinIR.git
cd SwinIR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
if use vpn:-  pip install -r requirements.txt --proxy http://127.0.0.1:2080
```

### Step 2: Download LOL Dataset

Download from: https://daooshee.github.io/BMVC2018website/

Extract to `datasets/LOL/` with this structure:

```
datasets/LOL/
├── our485/        # Training set (485 pairs)
│   ├── low/       # Low-light images
│   └── high/      # Normal-light images
└── eval15/        # Test set (15 pairs)
    ├── low/
    └── high/
```

### Step 3: Verify Installation

```bash
# Test if model loads correctly
python models/network_swinllie.py
```

---

## 🏋️ Training

### Start Training from Scratch

```bash
python train_swinllie.py --config configs/swinllie_lol.yaml
```

### Training Options

| Argument   | Description                     | Default                     |
| ---------- | ------------------------------- | --------------------------- |
| `--config` | Path to config file             | `configs/swinllie_lol.yaml` |
| `--seed`   | Random seed for reproducibility | `42`                        |
| `--gpu`    | GPU ID to use                   | `0`                         |

### Monitor Training with TensorBoard

```bash
tensorboard --logdir experiments/swinllie_lol/logs
```

Then open http://localhost:6006 in your browser.

---

## 🔄 Resume Training from Checkpoint

The training script **automatically resumes** from the last checkpoint if `resume.enabled: true` in the config file.

### Option 1: Edit Config File

1. Open `configs/swinllie_lol.yaml`
2. Update the resume section:

```yaml
resume:
  enabled: true
  checkpoint_path: "./experiments/swinllie_lol/checkpoints/epoch_XX.pth" # Your checkpoint
```

3. Run training:

```bash
python train_swinllie.py --config configs/swinllie_lol.yaml
```

### Option 2: Quick Resume (without editing config)

Create a custom config YAML with your checkpoint path, or simply ensure your checkpoint is at:

```
./experiments/swinllie_lol/checkpoints/final.pth
```

### Available Checkpoints

After training, checkpoints are saved in `experiments/swinllie_lol/checkpoints/`:

| File           | Description                                     |
| -------------- | ----------------------------------------------- |
| `best.pth`     | Best model (highest validation PSNR)            |
| `final.pth`    | Last epoch model                                |
| `epoch_XX.pth` | Periodic checkpoints (every `save_freq` epochs) |

### What's Saved in Checkpoints

Each checkpoint contains:

- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Learning rate scheduler state (`scheduler_state_dict`)
- Mixed precision scaler state (`scaler_state_dict`)
- Current epoch number
- Best PSNR achieved

This allows **seamless training continuation** without losing any training progress.

---

## 🖼️ Inference / Testing

### Enhance a Single Image

```bash
python test_swinllie.py \
    --input your_image.jpg \
    --checkpoint experiments/swinllie_lol/checkpoints/best.pth \
    --output results/
```

### Enhance a Folder of Images

```bash
python test_swinllie.py \
    --input path/to/low_light_images/ \
    --checkpoint experiments/swinllie_lol/checkpoints/best.pth \
    --output results/enhanced/
```

### Evaluate on Test Set (with metrics)

```bash
python test_swinllie.py \
    --input datasets/LOL/eval15/low/ \
    --gt_folder datasets/LOL/eval15/high/ \
    --checkpoint experiments/swinllie_lol/checkpoints/best.pth \
    --output results/eval15/
```

### Inference Options

| Argument            | Description                         | Default                    |
| ------------------- | ----------------------------------- | -------------------------- |
| `--input`           | Input image or folder path          | _required_                 |
| `--output`          | Output folder for enhanced images   | `results/swinllie`         |
| `--checkpoint`      | Path to model checkpoint            | `experiments/.../best.pth` |
| `--gt_folder`       | Ground truth folder (for PSNR/SSIM) | `None`                     |
| `--gpu`             | GPU ID to use                       | `0`                        |
| `--max_size`        | Max image dimension (prevents OOM)  | `512`                      |
| `--save_comparison` | Save side-by-side comparison        | `False`                    |

---

## ⚙️ Configuration Reference

Key settings in `configs/swinllie_lol.yaml`:

### Model Settings

```yaml
model:
  embed_dim: 60 # Feature dimension (higher = more capacity)
  depths: [4, 4, 4] # Transformer blocks per stage
  num_heads: [6, 6, 6] # Attention heads per stage
  window_size: 8 # Swin Transformer window size
  use_igam: true # Enable Illumination-Guided Attention
```

### Training Settings

```yaml
training:
  batch_size: 4 # Reduce if GPU OOM
  epochs: 100 # Total training epochs
  learning_rate: 0.0002
  use_amp: true # Mixed precision (faster, less memory)
  save_freq: 10 # Save checkpoint every N epochs
```

### Resume Settings

```yaml
resume:
  enabled: true # Set to true to resume
  checkpoint_path: "./experiments/swinllie_lol/checkpoints/final.pth"
```

---

## 📊 Model Info

- **Parameters**: ~4.8M
- **Input**: RGB images (any resolution)
- **Output**: Enhanced RGB images (same resolution)
- **GPU Memory**: ~2-4GB (depending on image size and batch)

---

## 🔧 Troubleshooting

### GPU Out of Memory (OOM)

- Reduce `batch_size` in config (try 2 or 1)
- Use `--max_size 256` during inference
- Ensure `use_amp: true` is enabled

### Resume Not Working

- Verify the checkpoint file exists at the specified path
- Check that `resume.enabled: true` in the config
- Ensure `weights_only=False` is used in `torch.load()` (already implemented)

### Training Slow

- Enable mixed precision: `use_amp: true`
- Reduce `num_workers` if CPU bottleneck
- Use smaller `patch_size` for training (96 → 64)

---

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
