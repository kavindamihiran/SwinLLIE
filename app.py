#!/usr/bin/env python3
"""
SwinIR Low-Light Image Enhancement - Gradio Interface
Hosted on Hugging Face Spaces
"""

import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from swinllie import SwinLLIE

# Model configuration
MODEL_CONFIG = {
    'embed_dim': 60,
    'depths': [4, 4, 4],
    'num_heads': [6, 6, 6],
    'window_size': 8
}

CHECKPOINT_PATH = './models/best.pth'

# Global model variable
model = None
device = None


def load_model():
    """Load the SwinLLIE model once at startup."""
    global model, device
    
    print("Loading SwinLLIE model...")
    
    # Determine device
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            compute_capability = capability[0] + capability[1] / 10
            if compute_capability >= 7.0:
                device = 'cuda'
                print(f"✓ Using GPU (Compute Capability: {compute_capability:.1f})")
            else:
                device = 'cpu'
                print(f"✗ GPU incompatible (Compute Capability: {compute_capability:.1f}), using CPU")
        except:
            device = 'cpu'
            print("✗ GPU detection failed, using CPU")
    else:
        device = 'cpu'
        print("✓ Using CPU")
    
    # Initialize model
    model = SwinLLIE(
        img_size=128,
        embed_dim=MODEL_CONFIG['embed_dim'],
        depths=MODEL_CONFIG['depths'],
        num_heads=MODEL_CONFIG['num_heads'],
        window_size=MODEL_CONFIG['window_size']
    )
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully! ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def process_image(input_image, window_size=8):
    """
    Enhance a low-light image using SwinLLIE.
    
    Args:
        input_image: PIL Image or numpy array
        window_size: Window size for processing (default: 8)
    
    Returns:
        Enhanced PIL Image
    """
    if model is None:
        return None, "Error: Model not loaded!"
    
    try:
        # Convert to PIL if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype('uint8'))
        
        # Convert to RGB
        img = input_image.convert('RGB')
        original_size = img.size  # (W, H)
        
        # Convert to tensor
        img_np = np.array(img) / 255.0
        x = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
        
        # Get dimensions
        H, W = x.shape[2], x.shape[3]
        
        # Pad to multiple of window_size * 4 (for U-Net downsampling)
        pad_unit = window_size * 4
        pad_h = (pad_unit - H % pad_unit) % pad_unit
        pad_w = (pad_unit - W % pad_unit) % pad_unit
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Move to device and process
        x_padded = x_padded.to(device)
        
        with torch.no_grad():
            output = model(x_padded)
        
        # Remove padding
        output = output[:, :, :H, :W]
        
        # Convert back to image
        enhanced = output[0].permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        
        # Info message
        info = f"✓ Enhanced! Original: {W}×{H} | Device: {device.upper()}"
        
        return enhanced_img, info
        
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        print(error_msg)
        return None, error_msg


# Custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.title {
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 0.5em;
    background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.description {
    text-align: center;
    font-size: 1.1em;
    color: #6B7280;
    margin-bottom: 2em;
}
.footer {
    text-align: center;
    margin-top: 2em;
    padding: 1em;
    color: #9CA3AF;
    font-size: 0.9em;
}
"""

# Create Gradio interface
def create_interface():
    """Create the Gradio web interface."""
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML('<h1 class="title">🌟 SwinIR Low-Light Enhancement</h1>')
        gr.HTML('<p class="description">Transform your dark images into bright, clear photos using Swin Transformer technology</p>')
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Low-Light Image",
                    type="pil",
                    height=400
                )
                
                enhance_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column():
                output_image = gr.Image(
                    label="Enhanced Result",
                    type="pil",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
        
        # Examples
        gr.Markdown("### 📸 Try These Examples")
        gr.Examples(
            examples=[
                ["test/1.png"] if os.path.exists("test/1.png") else None,
                ["test/2.png"] if os.path.exists("test/2.png") else None,
            ],
            inputs=input_image,
            outputs=[output_image, status_text],
            fn=process_image,
            cache_examples=False
        )
        
        # Info section
        with gr.Accordion("ℹ️ About SwinIR", open=False):
            gr.Markdown("""
            ### Model Architecture
            - **Base**: Swin Transformer for Image Restoration
            - **Parameters**: ~4.7M
            - **Task**: Low-Light Image Enhancement
            - **Training**: LOL Dataset
            
            ### How It Works
            1. Upload your low-light or dark image
            2. The model processes it through a U-Net encoder-decoder with Swin Transformer blocks
            3. Get an enhanced image with improved brightness, colors, and details
            
            ### Technical Details
            - **Embed Dim**: 60
            - **Depths**: [4, 4, 4]
            - **Num Heads**: [6, 6, 6]
            - **Window Size**: 8
            
            ### Links
            - [GitHub Repository](https://github.com/kavindamihiran/SwinIR)
            - [Original SwinIR Paper](https://arxiv.org/abs/2108.10257)
            """)
        
        # Event handlers
        enhance_btn.click(
            fn=process_image,
            inputs=input_image,
            outputs=[output_image, status_text]
        )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>Built with SwinIR | Powered by 🤗 Hugging Face Spaces</p>
            <p>© 2025 Group 10 | Low-Light Image Enhancement Research</p>
        </div>
        """)
    
    return demo


# Main execution
if __name__ == "__main__":
    # Load model at startup
    load_model()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Required for Docker
        server_port=7860,        # Hugging Face Spaces port
        share=False
    )
