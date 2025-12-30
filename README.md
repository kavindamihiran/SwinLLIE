# Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement

A novel deep learning architecture combining SwinIR with illumination-guided attention for low-light image enhancement.

## 🌟 Key Features

- **Illumination Estimation Module (IEM)**: Multi-scale CNN with dilated convolutions (2x, 4x) for accurate illumination detection
- **Illumination-Guided Attention Module (IGAM)**: Spatially-adaptive enhancement - stronger processing for dark regions, preservation for bright areas
- **U-Net Architecture**: Three-stage encoder-decoder with skip connections and Swin Transformer blocks
- **Comprehensive Loss Function**: L1 + VGG Perceptual + Color Consistency + Smoothness + Edge + High-Frequency + Exposure Control
- **Smart GPU/CPU Fallback**: Automatic CUDA capability detection (≥7.0 required) with seamless CPU fallback
- **Mixed Precision Training**: AMP support for faster training and reduced memory usage
- **Interval-Based Validation**: Evaluates the best model (by loss) every N epochs for robust checkpoint selection

## 📁 Project Structure

```
SwinLLIE/
├── swinllie/                 # Main module package
│   ├── __init__.py           # Module exports
│   ├── models.py             # SwinLLIE model architecture
│   ├── losses.py             # Hybrid loss functions
│   ├── data.py               # Dataset loaders (LOL, generic, unpaired)
│   └── utils.py              # PSNR, SSIM metrics and utilities
├── configs/
│   └── swinllie_lol.yaml     # Training configuration
├── datasets/                 # Dataset folder
│   └── LOL/
│       ├── our485/           # Training set (485 pairs)
│       └── eval15/           # Test set (15 pairs)
├── experiments/              # Saved checkpoints and logs
│   └── test_run/
│       └── checkpoints/      # Model checkpoints (.pth files)
├── Guides/                   # Documentation
│   ├── ARCHITECTURE.md       # Model architecture details
│   └── FINE_TUNING_GUIDE.md  # Fine-tuning instructions
├── test/                     # Input folder for inference
├── test_results/             # Output folder for inference
├── train.py                  # Training script
├── inference.py              # Inference script
└── requirements.txt          # Dependencies
```

---

## 🚀 Quick Setup

### Step 1: Clone and Create Environment

```bash
# Clone the repository
git clone https://github.com/kavindamihiran/SwinIR.git
cd SwinIR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# If behind a proxy/VPN:
pip install -r requirements.txt --proxy http://127.0.0.1:2080
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
python -c "from swinllie import SwinLLIE; print('Model loaded successfully!')"
```

---

## 🏋️ Training

### Start Training from Scratch

```bash
python train.py --config configs/swinllie_lol.yaml
```

### Training Features

- **Mixed Precision Training (AMP)**: Enabled by default - ~40% faster training and reduced GPU memory
- **Cosine Annealing Scheduler**: Smooth LR decay with configurable warmup period (default: 5 epochs)
- **Gradient Clipping**: Prevents training instability (default threshold: 1.0)
- **Automatic GPU/CPU Detection**: CUDA capability check (≥7.0) with automatic CPU fallback
- **Smart Validation**: Evaluates best-loss model every 5 epochs, saves only if PSNR/SSIM improves
- **AdamW Optimizer**: Weight decay regularization for better generalization

### Training Options

| Argument   | Description         | Default                     |
| ---------- | ------------------- | --------------------------- |
| `--config` | Path to config file | `configs/swinllie_lol.yaml` |

### Monitor Training with TensorBoard

```bash
tensorboard --logdir experiments/test_run
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
  checkpoint_path: "./experiments/test_run/checkpoints/final.pth"
```

3. Run training:

```bash
python train.py --config configs/swinllie_lol.yaml
```

### Available Checkpoints

After training, checkpoints are saved in `experiments/test_run/checkpoints/`:

| File           | Description                                         |
| -------------- | --------------------------------------------------- |
| `best.pth`     | Best model (highest PSNR, evaluated every 5 epochs) |
| `final.pth`    | Last epoch model                                    |
| `epoch_XX.pth` | Periodic checkpoints (every 20 epochs by default)   |

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

The inference script uses a simple folder-based approach with automatic GPU/CPU fallback.

### Quick Start

1. Place your low-light images in the `test/` folder
2. Run inference:

```bash
python inference.py
```

3. Find enhanced images in `test_results/` folder

### Configuration

Edit these variables at the top of `inference.py`:

```python
CHECKPOINT = './experiments/test_run/checkpoints/best.pth'  # Model checkpoint
INPUT_DIR = './test'                                         # Input folder
OUTPUT_DIR = './test_results'                                # Output folder
WINDOW_SIZE = 8                                              # Must match training
```

### Features

- **Smart GPU/CPU Fallback**: Automatically detects GPU compatibility and memory
- **Out-of-Memory Handling**: Falls back to CPU if GPU runs out of memory
- **Automatic Padding**: Handles any image size by padding to match window size
- **Batch Processing**: Processes all images in the input folder

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
  mlp_ratio: 2.0 # MLP expansion ratio
  drop_path_rate: 0.1 # Stochastic depth rate
```

### Training Settings

```yaml
training:
  batch_size: 4 # Reduce if GPU OOM
  epochs: 100 # Total training epochs
  learning_rate: 0.0002 # Initial learning rate
  warmup_epochs: 5 # LR warmup period
  min_lr: 0.000001 # Minimum LR for cosine annealing
  use_amp: true # Mixed precision (faster, less memory)
  grad_clip: 1.0 # Gradient clipping threshold
  save_freq: 10 # Save checkpoint every N epochs
  save_dir: "./experiments/test_run"
```

### Loss Settings

```yaml
loss:
  lambda_l1: 1.0 # L1 reconstruction loss
  lambda_vgg: 0.1 # VGG perceptual loss (features from conv1_2, conv2_2, conv3_3, conv4_3)
  lambda_color: 0.5 # Color consistency loss
  lambda_smooth: 0.01 # Smoothness loss (reduces artifacts)
  lambda_edge: 1.0 # Edge preservation loss (Sobel-based)
  lambda_hf: 0.5 # High-frequency loss (Laplacian filtering)
  lambda_exposure: 1.0 # Exposure control loss (optional)
  use_ssim: false # Enable SSIM loss (additional)
  lambda_ssim: 0.1 # SSIM loss weight (if enabled)
```

### Resume Settings

```yaml
resume:
  enabled: false # Set to true to resume
  checkpoint_path: "./experiments/test_run/checkpoints/final.pth"
```

---

## 📊 Model Info

- **Parameters**: ~4.8M (efficient for real-time processing)
- **Architecture**: U-Net with 3-stage encoder-decoder + Swin Transformer blocks
- **Window Size**: 8x8 patches (configurable)
- **Embed Dimension**: 60 (configurable: 48-96 for different model sizes)
- **Attention Heads**: [6, 6, 6] per stage
- **Transformer Depth**: [4, 4, 4] blocks per stage (configurable)
- **Input**: RGB images (any resolution, automatically padded to multiple of 32)
- **Output**: Enhanced RGB images (same resolution as input)
- **GPU Memory**: ~2-4GB for batch_size=4 with 128x128 patches
- **Minimum CUDA**: Compute capability 7.0+ (GTX 1060+, automatic CPU fallback if not met)
- **Speed**: ~0.1s per 512x512 image on modern GPU

---

## 📚 Documentation

Additional guides are available in the `Guides/` folder:

- [ARCHITECTURE.md](Guides/ARCHITECTURE.md) - Detailed model architecture explanation
- [FINE_TUNING_GUIDE.md](Guides/FINE_TUNING_GUIDE.md) - Guide for fine-tuning on custom datasets

---

## 🔧 Troubleshooting

### GPU Out of Memory (OOM)

- Reduce `batch_size` in config (try 2 or 1)
- The inference script automatically falls back to CPU on OOM
- Ensure `use_amp: true` is enabled for training

### GPU Incompatible

- The scripts automatically detect GPU compute capability
- GPUs with CUDA capability < 7.0 will fall back to CPU
- Check your GPU: `python -c "import torch; print(torch.cuda.get_device_capability())"`

### Resume Not Working

- Verify the checkpoint file exists at the specified path
- Check that `resume.enabled: true` in the config
- Ensure `weights_only=False` is used in `torch.load()` (already implemented)

### Training Slow

- Enable mixed precision: `use_amp: true` (already default)
- Reduce `num_workers` if CPU bottleneck (try 2 instead of 4)
- Use smaller `patch_size` for training (96 → 64)
- Consider reducing `embed_dim` (60 → 48) for faster training
- Ensure you're not running other GPU-intensive processes

### Images Look Wrong After Enhancement

- Check that your images are in the correct format (PNG, JPG, BMP)
- Ensure images aren't already enhanced (the model expects low-light input)
- Verify the checkpoint file is not corrupted
- Try different checkpoint (best.pth vs final.pth)

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{swinllie2025,
  title={Swin-LLIE: Illumination-Aware Swin Transformer for Low-Light Image Enhancement},
  author={Group 10},
  year={2025}
}
```

## 🙏 Acknowledgments

- Based on [SwinIR](https://github.com/JingyunLiang/SwinIR) architecture
- Inspired by Retinex theory for illumination estimation
- LOL Dataset from [Deep Retinex Decomposition](https://daooshee.github.io/BMVC2018website/)
