# 🚀 SwinLLIE Deployment Guide

Guide for deploying SwinLLIE to various platforms.

---

## Table of Contents

1. [Hugging Face Spaces](#hugging-face-spaces)
2. [Local API Server](#local-api-server)
3. [Edge Device Deployment](#edge-device-deployment)
4. [Model Export](#model-export)

---

## Hugging Face Spaces

### Prerequisites
- Hugging Face account
- Git installed
- Trained model checkpoint

### Step 1: Create Space

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/swinllie-demo
cd swinllie-demo
```

### Step 2: Project Structure

```
swinllie-demo/
├── app.py              # Gradio interface
├── requirements.txt    # Dependencies
├── Dockerfile          # Container config
├── swinllie/           # Model code
│   ├── models.py
│   └── utils.py
└── checkpoints/
    └── best_model.pth  # Trained weights
```

### Step 3: Gradio App (app.py)

```python
import gradio as gr
import torch
from PIL import Image
import numpy as np
from swinllie import SwinLLIE

# Load model
model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location='cpu'))
model.eval()

def enhance(image):
    # Preprocess
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    # Enhance
    with torch.no_grad():
        output = model(img)
    
    # Postprocess
    output = output.squeeze(0).permute(1, 2, 0).numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)

# Gradio interface
demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type="pil", label="Low-Light Image"),
    outputs=gr.Image(type="pil", label="Enhanced Image"),
    title="SwinLLIE: Low-Light Image Enhancement",
    description="Upload a dark image to enhance it using Swin Transformer."
)

demo.launch()
```

### Step 4: Deploy

```bash
git add .
git commit -m "Initial deployment"
git push
```

---

## Local API Server

### FastAPI Server

```python
# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
import io
from swinllie import SwinLLIE

app = FastAPI(title="SwinLLIE API")

# Load model once
model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Process
    # ... (same as above)
    
    # Return
    buf = io.BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Usage

```bash
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@dark_image.jpg" \
  --output enhanced.png
```

---

## Edge Device Deployment

### ONNX Export

```python
import torch
from swinllie import SwinLLIE

model = SwinLLIE(embed_dim=60, depths=[4,4,4], num_heads=[6,6,6])
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Export
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "swinllie.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {2: 'height', 3: 'width'},
                  'output': {2: 'height', 3: 'width'}}
)
```

### TensorRT Optimization

```bash
# Convert ONNX to TensorRT
trtexec --onnx=swinllie.onnx --saveEngine=swinllie.trt --fp16
```

### Mobile (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("swinllie.onnx")

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})
```

---

## Model Export

### PyTorch JIT (TorchScript)

```python
model = SwinLLIE(...)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Trace
example = torch.randn(1, 3, 256, 256)
traced = torch.jit.trace(model, example)
traced.save("swinllie_traced.pt")
```

### Model Size Optimization

| Format | Size | Speed |
|--------|------|-------|
| PyTorch (.pth) | ~19 MB | Baseline |
| TorchScript (.pt) | ~19 MB | 1.1× faster |
| ONNX | ~18 MB | 1.2× faster |
| TensorRT FP16 | ~10 MB | 2-3× faster |
| TensorRT INT8 | ~5 MB | 3-4× faster |

---

## Performance Tips

1. **Batch Processing**: Process multiple images together
2. **Half Precision**: Use `.half()` for 2× speed on GPU
3. **Tiling**: For large images, process in overlapping tiles
4. **GPU Warmup**: Run dummy inference before benchmarking

```python
# Half precision example
model = model.half().cuda()
with torch.no_grad():
    output = model(input.half())
```
