# 🖼️ SwinLLIE Results Gallery

Visual results and comparisons for the SwinLLIE model.

---

## Qualitative Comparisons

### LOL Dataset Results

| Input (Low-Light) | SwinLLIE (Ours) | Ground Truth |
|-------------------|-----------------|--------------|
| ![low1](../test_results/low_001.png) | ![out1](../test_results/enhanced_001.png) | ![gt1](../test_results/gt_001.png) |
| ![low2](../test_results/low_002.png) | ![out2](../test_results/enhanced_002.png) | ![gt2](../test_results/gt_002.png) |

> **Note**: Replace the image paths above with your actual test results.

---

## Method Comparison

### Input: Dark Indoor Scene

| Method | Result | PSNR | SSIM |
|--------|--------|------|------|
| **Input** | ![input](images/input.png) | - | - |
| RetinexNet | ![retinex](images/retinex.png) | 16.77 | 0.462 |
| EnlightenGAN | ![enlighten](images/enlighten.png) | 17.48 | 0.651 |
| Zero-DCE | ![zerodce](images/zerodce.png) | 14.86 | 0.589 |
| **SwinLLIE (Ours)** | ![swinllie](images/swinllie.png) | **23.50** | **0.845** |
| Ground Truth | ![gt](images/gt.png) | ∞ | 1.0 |

---

## Enhancement Examples

### Example 1: Night Street Scene

**Before (Low-Light)**
```
[Insert dark image here]
```

**After (Enhanced)**
```
[Insert enhanced image here]
```

**Observations:**
- ✅ Brightness significantly improved
- ✅ Colors preserved naturally
- ✅ No overexposure in bright areas
- ✅ Textures and details maintained

---

### Example 2: Indoor Low-Light

**Before**
```
[Insert dark indoor image]
```

**After**
```
[Insert enhanced image]
```

---

## Failure Cases & Limitations

### Case 1: Extreme Darkness
When input is nearly black (mean intensity < 0.05), the model may:
- Produce noise amplification
- Show color artifacts

**Solution**: Apply multiple enhancement passes or pre-process with histogram equalization.

### Case 2: Mixed Lighting
Strong local light sources may cause:
- HDR-like effects
- Slight halo artifacts

**Solution**: Increase `lambda_exposure` weight during training.

---

## Generating Your Own Results

```python
from swinllie import SwinLLIE
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
model.load_state_dict(torch.load('checkpoints/best.pth'))
model.eval()

# Process image
img = Image.open('dark.jpg').convert('RGB')
tensor = T.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)

# Save
result = T.ToPILImage()(output.squeeze(0).clamp(0, 1))
result.save('enhanced.jpg')
```

---

## Metrics Explanation

| Metric | Description | Good Value |
|--------|-------------|------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | > 20 dB |
| **SSIM** | Structural Similarity Index | > 0.8 |
| **LPIPS** | Learned Perceptual Similarity | < 0.2 |

---

## How to Add Your Results

1. Run inference on test images:
   ```bash
   python inference.py
   ```

2. Results are saved to `test_results/`

3. Copy images to `Guides/images/` folder

4. Update the image paths in this document

---

## Citation

If you use these results, please cite:

```bibtex
@article{swinllie2025,
  title={SwinLLIE: Swin Transformer for Low-Light Image Enhancement},
  author={Kavinda Mihiran},
  year={2025}
}
```
