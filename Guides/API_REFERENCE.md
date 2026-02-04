# 📖 SwinLLIE API Reference

Complete API documentation for the SwinLLIE module.

---

## Quick Import

```python
from swinllie import SwinLLIE, HybridLoss
from swinllie.data import LOLDataset, GenericPairedDataset
from swinllie.utils import calculate_psnr, calculate_ssim
```

---

## Models

### SwinLLIE

The main model class for low-light image enhancement.

```python
class SwinLLIE(nn.Module):
    """
    Swin Transformer for Low-Light Image Enhancement.
    
    Args:
        img_size (int): Input image size for training. Default: 128
        patch_size (int): Patch size. Default: 1
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 60
        depths (list): Number of blocks in each stage. Default: [4, 4, 4]
        num_heads (list): Number of attention heads. Default: [6, 6, 6]
        window_size (int): Window size for attention. Default: 8
        mlp_ratio (float): MLP hidden dim ratio. Default: 2.0
        drop_rate (float): Dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        use_checkpoint (bool): Use gradient checkpointing. Default: False
    
    Example:
        >>> model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)  # (1, 3, 256, 256)
    """
```

#### Methods

| Method | Description |
|--------|-------------|
| `forward(x)` | Enhance low-light image. Input/output: (B, 3, H, W) in [0, 1] |
| `check_image_size(x)` | Pad image to be divisible by window size |

#### Model Variants

```python
# Tiny (~2M params)
model = SwinLLIE(embed_dim=48, depths=[2,2,2], num_heads=[4,4,4])

# Small/Default (~4.7M params)
model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])

# Base (~12M params)
model = SwinLLIE(embed_dim=96, depths=[6,6,6], num_heads=[8,8,8])

# Large (~25M params)
model = SwinLLIE(embed_dim=128, depths=[8,8,8], num_heads=[12,12,12])
```

---

## Loss Functions

### HybridLoss

Combined loss function for training.

```python
class HybridLoss(nn.Module):
    """
    Combined loss for SwinLLIE training.
    
    Args:
        lambda_l1 (float): L1 loss weight. Default: 1.0
        lambda_vgg (float): VGG perceptual loss weight. Default: 0.1
        lambda_color (float): Color consistency loss weight. Default: 0.5
        lambda_edge (float): Edge loss weight. Default: 1.0
        lambda_exposure (float): Exposure control loss weight. Default: 1.0
        use_ssim (bool): Include SSIM loss. Default: False
        lambda_ssim (float): SSIM loss weight. Default: 0.1
    
    Returns:
        tuple: (total_loss, loss_dict)
    
    Example:
        >>> criterion = HybridLoss(lambda_l1=1.0, lambda_vgg=0.1)
        >>> loss, breakdown = criterion(pred, target)
        >>> print(breakdown)
        {'l1': 0.05, 'vgg': 0.12, 'color': 0.01, 'edge': 0.03, ...}
    """
```

### Individual Losses

```python
from swinllie.losses import (
    L1Loss,              # Pixel reconstruction
    VGGPerceptualLoss,   # Perceptual quality
    ColorConsistencyLoss, # Color preservation
    EdgeLoss,            # Edge sharpness
    ExposureControlLoss, # Prevent overexposure
    SSIMLoss             # Structural similarity
)
```

---

## Data

### LOLDataset

Dataset loader for the LOL (Low-Light) dataset.

```python
class LOLDataset(Dataset):
    """
    LOL Dataset loader.
    
    Args:
        root_dir (str): Path to LOL dataset
        split (str): 'train' or 'eval'
        patch_size (int): Crop size for training. Default: 96
        augment (bool): Apply augmentation. Default: True
    
    Returns:
        dict: {'low': tensor, 'high': tensor, 'filename': str}
    
    Example:
        >>> dataset = LOLDataset('datasets/LOL', split='train', patch_size=96)
        >>> sample = dataset[0]
        >>> low, high = sample['low'], sample['high']
    """
```

### GenericPairedDataset

For custom paired datasets.

```python
class GenericPairedDataset(Dataset):
    """
    Generic paired image dataset.
    
    Args:
        low_dir (str): Directory with low-light images
        high_dir (str): Directory with ground truth images
        patch_size (int): Crop size. Default: 96
        augment (bool): Apply augmentation. Default: True
    
    Example:
        >>> dataset = GenericPairedDataset(
        ...     low_dir='data/dark',
        ...     high_dir='data/bright',
        ...     patch_size=128
        ... )
    """
```

---

## Utilities

### Metrics

```python
from swinllie.utils import calculate_psnr, calculate_ssim

# Calculate PSNR
psnr = calculate_psnr(pred, target)  # Returns dB value

# Calculate SSIM
ssim = calculate_ssim(pred, target)  # Returns value in [0, 1]
```

### Image Processing

```python
from swinllie.utils import tensor_to_image, image_to_tensor

# Convert tensor to PIL Image
image = tensor_to_image(tensor)  # (B,C,H,W) -> PIL Image

# Convert PIL Image to tensor
tensor = image_to_tensor(image)  # PIL Image -> (1,3,H,W) in [0,1]
```

---

## Inference Example

```python
import torch
from PIL import Image
from swinllie import SwinLLIE
from swinllie.utils import image_to_tensor, tensor_to_image

# Load model
model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
model.load_state_dict(torch.load('checkpoints/best.pth'))
model.eval()

# Load image
image = Image.open('dark_photo.jpg').convert('RGB')
input_tensor = image_to_tensor(image)

# Enhance
with torch.no_grad():
    output = model(input_tensor)

# Save result
result = tensor_to_image(output)
result.save('enhanced.jpg')
```

---

## Training Example

```python
from swinllie import SwinLLIE, HybridLoss
from swinllie.data import LOLDataset
from torch.utils.data import DataLoader

# Setup
model = SwinLLIE().cuda()
criterion = HybridLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

dataset = LOLDataset('datasets/LOL', split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(100):
    for batch in loader:
        low = batch['low'].cuda()
        high = batch['high'].cuda()
        
        # Forward
        pred = model(low)
        loss, breakdown = criterion(pred, high)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Configuration

Load configuration from YAML:

```python
import yaml

with open('configs/swinllie_lol.yaml') as f:
    config = yaml.safe_load(f)

model = SwinLLIE(**config['model'])
```
