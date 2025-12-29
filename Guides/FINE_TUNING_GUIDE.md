# 🎓 Complete Fine-Tuning Masterclass for Deep Learning

A comprehensive guide to becoming an expert at fine-tuning deep learning models, specifically tailored for the SwinLLIE project.

---

## 📚 Part 1: Understanding Fine-Tuning Fundamentals

### What is Fine-Tuning?

Fine-tuning is adapting a pre-trained model to your specific task by continuing training with:

- **Lower learning rates** (typically 10-100x smaller)
- **Task-specific data** (your domain)
- **Selective layer training** (freeze some layers, train others)

### Training from Scratch vs Fine-Tuning

```python
# Training from scratch (your current setup)
model = SwinLLIE(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Fine-tuning (lower LR, load weights)
model = SwinLLIE(...)
checkpoint = torch.load('pretrained.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # 10x lower!
```

---

## 🔧 Part 2: Hyperparameter Tuning Mastery

### 1. Learning Rate (Most Critical!)

**Rule of thumb:**

- Training from scratch: `1e-4` to `5e-4`
- Fine-tuning: `1e-5` to `5e-5`
- Large models: smaller LR
- Small datasets: smaller LR

**Your config:**

```yaml
learning_rate: 0.0002 # Good for training from scratch
min_lr: 0.000001 # Good minimum
```

**Pro tip - Learning Rate Finder:**

```python
# Add to your training script
def find_lr(model, train_loader, optimizer, device, init_lr=1e-8, final_lr=10):
    """Run to find optimal learning rate"""
    lrs, losses = [], []
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr

    for batch in train_loader:
        low, high = batch['low'].to(device), batch['high'].to(device)
        output = model(low)
        loss, _ = criterion(output, high, None)
        losses.append(loss.item())
        lrs.append(lr)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lr *= 1.1  # Exponential increase
        optimizer.param_groups[0]['lr'] = lr
        if lr > final_lr:
            break

    # Plot and find steepest descent
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.savefig('lr_finder.png')
    print(f"Check lr_finder.png - optimal LR is at steepest descent point")
```

### 2. Batch Size

**Impact:**

- **Small (2-4)**: Better generalization, slower training, less GPU memory
- **Large (16-32)**: Faster training, more stable gradients, needs more GPU memory

**Your setup:**

```yaml
batch_size: 4 # Good for 8GB VRAM
```

**Pro strategy - Gradient Accumulation:**

```python
# Simulate batch_size=16 with batch_size=4
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps  # Scale loss
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 3. Optimizer Selection

| Optimizer          | Best For                 | Pros                          | Cons                      |
| ------------------ | ------------------------ | ----------------------------- | ------------------------- |
| **AdamW**          | Transformers, most tasks | Adaptive LR, weight decay fix | Higher memory             |
| **Adam**           | Quick experiments        | Simple, fast convergence      | No decoupled weight decay |
| **SGD + Momentum** | CNNs, when you have time | Best final performance        | Needs careful LR tuning   |
| **Lion**           | Large models             | Memory efficient              | New, less tested          |

**Your config:**

```yaml
optimizer: "adamw" # ✅ Excellent choice for Swin
weight_decay: 0.0001 # ✅ Good for regularization
betas: [0.9, 0.999] # ✅ Standard values
```

### 4. Learning Rate Schedulers

**Comparison:**

```python
# 1. Cosine Annealing (Your choice - BEST for most cases)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
# Smoothly decreases LR following cosine curve

# 2. Step Decay (Good for fine-tuning)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
# Drops LR by 10x every 30 epochs

# 3. ReduceLROnPlateau (Adaptive - great when loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Reduces LR when validation loss stops improving

# 4. OneCycleLR (Fastest convergence - advanced)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-4, epochs=100, steps_per_epoch=len(train_loader)
)
# LR increases then decreases (super-convergence)
```

**With Warmup (Recommended):**

```python
import math

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Cosine annealing with warmup for stable training start.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs to linearly increase LR
        total_epochs: Total training epochs
        min_lr: Minimum learning rate at the end
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        # Cosine annealing after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## 🎯 Part 3: Advanced Fine-Tuning Techniques

### 1. Layer-wise Learning Rates (Discriminative Fine-Tuning)

```python
def get_parameter_groups(model, base_lr=2e-4):
    """
    Different LR for different parts of the model.

    Strategy:
    - Freeze patch embedding (transfer learning)
    - Low LR for early layers (preserve features)
    - Normal LR for middle layers
    - High LR for task-specific heads
    """
    return [
        # Freeze patch embedding
        {'params': model.patch_embed.parameters(), 'lr': 0.0},

        # Low LR for early Swin layers
        {'params': model.layers[0].parameters(), 'lr': base_lr * 0.1},
        {'params': model.layers[1].parameters(), 'lr': base_lr * 0.3},

        # Normal LR for later layers
        {'params': model.layers[2].parameters(), 'lr': base_lr},

        # Higher LR for task-specific heads
        {'params': model.conv_after_body.parameters(), 'lr': base_lr * 2},
        {'params': model.conv_last.parameters(), 'lr': base_lr * 2},
    ]

# Usage
optimizer = torch.optim.AdamW(get_parameter_groups(model), weight_decay=1e-4)
```

### 2. Progressive Unfreezing

```python
def unfreeze_layers(model, epoch, total_epochs):
    """
    Gradually unfreeze layers during training.

    Strategy:
    - Start with only final layers trainable
    - Progressively unfreeze earlier layers
    - Prevents catastrophic forgetting
    """
    if epoch < total_epochs * 0.2:
        # First 20%: Only train final layers
        for param in model.layers.parameters():
            param.requires_grad = False
        for param in model.conv_after_body.parameters():
            param.requires_grad = True
        for param in model.conv_last.parameters():
            param.requires_grad = True

    elif epoch < total_epochs * 0.5:
        # Next 30%: Unfreeze last Swin layer
        for param in model.layers[2].parameters():
            param.requires_grad = True
    else:
        # Final 50%: Train everything
        for param in model.parameters():
            param.requires_grad = True

# In training loop
for epoch in range(EPOCHS):
    unfreeze_layers(model, epoch, EPOCHS)
    # ... training code ...
```

### 3. Gradient Clipping (Essential for Transformers)

```python
# Your current setup is perfect:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Why it matters:
# - Prevents exploding gradients
# - Stabilizes training
# - Essential for Transformers and deep networks

# Alternative: Gradient value clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

# Your setup (already implemented ✅):
scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(low)
        loss = criterion(output, high)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# Benefits:
# - 2-3x faster training
# - 2x less GPU memory
# - Minimal accuracy loss (<1%)
```

---

## 📊 Part 4: Loss Function Tuning

### Understanding Your Loss Weights

```yaml
loss:
  lambda_l1: 1.0 # Pixel-wise reconstruction
  lambda_vgg: 0.1 # Perceptual quality
  lambda_color: 0.5 # Color consistency
  lambda_smooth: 0.01 # Spatial smoothness
```

### How to Tune Loss Weights

**Step-by-step approach:**

1. **Start with single loss** (L1 only)

   ```yaml
   lambda_l1: 1.0
   lambda_vgg: 0.0
   lambda_color: 0.0
   lambda_smooth: 0.0
   ```

2. **Add perceptual loss** if images look over-smoothed

   ```yaml
   lambda_l1: 1.0
   lambda_vgg: 0.1 # Start small
   ```

3. **Add color loss** if colors drift

   ```yaml
   lambda_color: 0.5
   ```

4. **Add smoothing** if artifacts appear
   ```yaml
   lambda_smooth: 0.01
   ```

### Adaptive Loss Weighting (Advanced)

```python
class AdaptiveLossWeights(nn.Module):
    """
    Automatically balance loss components using uncertainty weighting.

    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    Reference: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_losses=4):
        super().__init__()
        # Log variance for each loss component
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, l1, vgg, color, smooth):
        """
        Uncertainty weighting: balances losses automatically

        The network learns to weight each loss based on its uncertainty.
        Losses with high variance get lower weights automatically.
        """
        precision_l1 = torch.exp(-self.log_vars[0])
        loss = precision_l1 * l1 + self.log_vars[0]

        precision_vgg = torch.exp(-self.log_vars[1])
        loss += precision_vgg * vgg + self.log_vars[1]

        precision_color = torch.exp(-self.log_vars[2])
        loss += precision_color * color + self.log_vars[2]

        precision_smooth = torch.exp(-self.log_vars[3])
        loss += precision_smooth * smooth + self.log_vars[3]

        return loss

# Usage:
adaptive_loss = AdaptiveLossWeights().to(device)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(adaptive_loss.parameters()),
    lr=2e-4
)
```

### When to Use Each Loss

| Loss Type             | Use When                       | Impact                        |
| --------------------- | ------------------------------ | ----------------------------- |
| **L1/L2**             | Always (baseline)              | Pixel-accurate reconstruction |
| **Perceptual (VGG)**  | Need realistic textures        | Prevents over-smoothing       |
| **SSIM**              | Structural similarity critical | Preserves edges, patterns     |
| **Color Consistency** | Color-sensitive tasks          | Prevents color drift          |
| **Adversarial (GAN)** | Photorealistic results         | Most realistic but unstable   |
| **Total Variation**   | Remove noise/artifacts         | Spatial smoothness            |

---

## 🔍 Part 5: Data Strategy

### 1. Data Augmentation for Low-Light Enhancement

```python
import torchvision.transforms as T
import random

class LowLightAugmentation:
    """
    Augmentation pipeline for paired low-light enhancement.

    IMPORTANT: Apply SAME transforms to both low and high images!
    """
    def __init__(self, patch_size=96):
        self.patch_size = patch_size

    def __call__(self, low_img, high_img):
        # 1. Random crop (same location for both)
        i, j, h, w = T.RandomCrop.get_params(
            low_img, output_size=(self.patch_size, self.patch_size)
        )
        low_img = TF.crop(low_img, i, j, h, w)
        high_img = TF.crop(high_img, i, j, h, w)

        # 2. Random horizontal flip
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        # 3. Random vertical flip
        if random.random() > 0.5:
            low_img = TF.vflip(low_img)
            high_img = TF.vflip(high_img)

        # 4. Random 90° rotation
        angle = random.choice([0, 90, 180, 270])
        low_img = TF.rotate(low_img, angle)
        high_img = TF.rotate(high_img, angle)

        # DON'T use ColorJitter - destroys low-light characteristics!
        # DON'T adjust brightness/contrast - that's what we're learning!

        return low_img, high_img
```

### 2. Advanced Augmentation: MixUp

```python
def mixup_data(x1, y1, x2, y2, alpha=0.2):
    """
    MixUp augmentation for better generalization.

    Creates virtual training examples by mixing two samples.
    Reference: "mixup: Beyond Empirical Risk Minimization"
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2

    return mixed_x, mixed_y

# In training loop:
for i, batch in enumerate(train_loader):
    if i % 2 == 1 and i > 0:  # Every other batch
        # Mix with previous batch
        low_mixed, high_mixed = mixup_data(
            prev_low, prev_high,
            batch['low'], batch['high'],
            alpha=0.2
        )
        output = model(low_mixed.to(device))
        loss = criterion(output, high_mixed.to(device))
    else:
        # Normal training
        prev_low, prev_high = batch['low'], batch['high']
        # ... normal training ...
```

### 3. Dataset Size Strategy

| Dataset Size      | Strategy                               | Recommended Actions                                                                                 |
| ----------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **< 100 samples** | Heavy augmentation + transfer learning | - MixUp/CutMix<br>- Heavy dropout (0.3-0.5)<br>- Strong regularization<br>- Load pretrained weights |
| **100-1000**      | Moderate augmentation + fine-tuning    | - Standard augmentation<br>- Medium dropout (0.1-0.2)<br>- Fine-tune from pretrained                |
| **> 1000**        | Light augmentation, train from scratch | - Basic geometric transforms<br>- Low dropout (0.0-0.1)<br>- Can train from scratch                 |

**Your LOL dataset:** 485 training pairs → **Perfect for training from scratch with moderate augmentation!**

### 4. Validation Strategy

```python
# Your config:
validation:
  val_freq: 5              # Every 5 epochs ✅
  save_images: true        # Visual inspection ✅
  num_save_images: 5       # Good sample size ✅

# Pro tip: Track multiple metrics
def validate(model, val_loader, criterion, device, save_dir):
    model.eval()
    metrics = {
        'loss': 0,
        'psnr': 0,
        'ssim': 0,
        'lpips': 0  # Perceptual metric
    }

    with torch.no_grad():
        for batch in val_loader:
            low, high = batch['low'].to(device), batch['high'].to(device)
            output = model(low)

            # Compute metrics
            metrics['loss'] += criterion(output, high)[0].item()
            metrics['psnr'] += compute_psnr(output, high)
            metrics['ssim'] += compute_ssim(output, high)
            metrics['lpips'] += compute_lpips(output, high)

    # Average
    for key in metrics:
        metrics[key] /= len(val_loader)

    return metrics

def compute_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def compute_ssim(pred, target):
    """Structural Similarity Index"""
    from pytorch_msssim import ssim
    return ssim(pred, target, data_range=1.0)
```

---

## 🛠️ Part 6: Debugging & Monitoring

### 1. Essential Training Checks

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_grad_flow(named_parameters):
    """
    Visualize gradient flow through network layers.

    Helps diagnose:
    - Vanishing gradients (all values near zero)
    - Exploding gradients (huge spikes)
    - Dead layers (no gradient flow)
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    print("Gradient flow plot saved to gradient_flow.png")

# Use after loss.backward():
plot_grad_flow(model.named_parameters())
```

```python
def check_activation_stats(model, sample_input):
    """
    Monitor activation statistics to detect training issues.

    Healthy activations should have:
    - Mean near 0 (for layers with normalization)
    - Std around 1 (not too small/large)
    - Reasonable min/max range
    """
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if torch.is_tensor(output):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item()
                }
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    with torch.no_grad():
        model(sample_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print summary
    print("\n" + "="*80)
    print("ACTIVATION STATISTICS")
    print("="*80)
    for name, stats in activations.items():
        print(f"\n{name}:")
        for key, val in stats.items():
            print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")

    return activations

# Use during training:
if epoch % 10 == 0:
    stats = check_activation_stats(model, sample_batch)
```

### 2. Training Red Flags & Solutions

| Problem                 | Symptom                                                 | Solution                                                                                            |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Exploding Gradients** | - Loss becomes NaN<br>- Gradients > 100                 | - Lower learning rate<br>- Add gradient clipping<br>- Check data normalization                      |
| **Vanishing Gradients** | - Loss doesn't decrease<br>- Gradients near 0           | - Higher learning rate<br>- Change activation (ReLU→LeakyReLU)<br>- Add residual connections        |
| **Overfitting**         | - Train loss ↓, Val loss ↑<br>- Train/Val gap increases | - More data augmentation<br>- Add dropout/regularization<br>- Early stopping<br>- Reduce model size |
| **Underfitting**        | - Both train & val loss high<br>- Loss plateaus early   | - Larger model<br>- More epochs<br>- Higher learning rate<br>- Remove regularization                |
| **Mode Collapse**       | - All outputs look similar<br>- Low diversity           | - Change loss function<br>- Add diversity penalty<br>- Check data distribution                      |
| **Unstable Training**   | - Loss oscillates wildly                                | - Lower learning rate<br>- Increase batch size<br>- Add batch normalization                         |

### 3. Advanced TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter(f'{SAVE_DIR}/logs')

# In training loop:
global_step = 0

for epoch in range(EPOCHS):
    for i, batch in enumerate(train_loader):
        low, high = batch['low'].to(device), batch['high'].to(device)

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            output = model(low)
            illum, _ = model.get_illumination_map(low)
            loss, loss_dict = criterion(output, high, illum)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if global_step % 100 == 0:
            # 1. Scalar metrics
            writer.add_scalar('Loss/total', loss.item(), global_step)
            writer.add_scalar('Loss/l1', loss_dict['l1'], global_step)
            writer.add_scalar('Loss/vgg', loss_dict['vgg'], global_step)
            writer.add_scalar('Loss/color', loss_dict['color'], global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            # 2. Image comparisons (every 500 steps)
            if global_step % 500 == 0:
                # Show 4 samples side-by-side
                comparison = torch.cat([
                    low[:4],      # Input
                    output[:4],   # Output
                    high[:4],     # Target
                ], dim=0)
                grid = torchvision.utils.make_grid(
                    comparison, nrow=4, normalize=True
                )
                writer.add_image('Comparison/Low_Output_High', grid, global_step)

                # Illumination maps
                illum_grid = torchvision.utils.make_grid(
                    illum[:4].repeat(1, 3, 1, 1), nrow=4, normalize=True
                )
                writer.add_image('Illumination/maps', illum_grid, global_step)

            # 3. Histograms (every 1000 steps)
            if global_step % 1000 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Weights/{name}', param, global_step)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)

        global_step += 1

    # Epoch-level validation logging
    if (epoch + 1) % 5 == 0:
        val_metrics = validate(model, val_loader, criterion, device)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/PSNR', val_metrics['psnr'], epoch)
        writer.add_scalar('Val/SSIM', val_metrics['ssim'], epoch)

writer.close()
```

---

## 🎓 Part 7: Advanced Topics

### 1. Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    """
    Train a small (student) model from a large (teacher) model.

    Use case: Deploy lightweight model while maintaining accuracy.

    The teacher's "soft" predictions contain more information than
    hard labels, helping the student learn better.
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance between soft and hard loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output, target):
        """
        Args:
            student_output: Small model predictions
            teacher_output: Large model predictions (detached)
            target: Ground truth
        """
        # Hard loss (normal training)
        hard_loss = F.l1_loss(student_output, target)

        # Soft loss (distillation)
        soft_student = F.log_softmax(student_output / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_output.detach() / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Usage:
teacher_model = SwinLLIE(embed_dim=120)  # Large model
student_model = SwinLLIE(embed_dim=60)   # Small model (half size)

teacher_model.load_state_dict(torch.load('teacher.pth'))
teacher_model.eval()

distill_criterion = DistillationLoss(temperature=3.0, alpha=0.5)

for batch in train_loader:
    low, high = batch['low'].to(device), batch['high'].to(device)

    # Teacher inference (no grad)
    with torch.no_grad():
        teacher_output = teacher_model(low)

    # Student training
    student_output = student_model(low)
    loss = distill_criterion(student_output, teacher_output, high)

    loss.backward()
    optimizer.step()
```

### 2. Learning Rate Warmup with Restarts

```python
class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with periodic restarts (SGDR).

    Benefits:
    - Helps escape local minima
    - Multiple chances to find better solutions
    - Often achieves better final accuracy

    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6):
        """
        Args:
            T_0: Number of epochs for first restart
            T_mult: Factor to increase T_i after each restart
            eta_min: Minimum learning rate
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.T_cur = 0
        self.T_i = T_0

    def step(self):
        self.T_cur += 1

        if self.T_cur >= self.T_i:
            # Restart: reset counter and increase period
            self.T_cur = 0
            self.T_i *= self.T_mult

        # Cosine annealing within current period
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# PyTorch built-in version:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # First restart after 10 epochs
    T_mult=2,    # Double period after each restart: 10, 20, 40, 80...
    eta_min=1e-6
)

# Usage in training loop:
for epoch in range(EPOCHS):
    for batch in train_loader:
        # ... training ...
        scheduler.step(epoch + batch_idx / len(train_loader))
```

### 3. Exponential Moving Average (EMA)

```python
class EMA:
    """
    Maintain exponential moving average of model weights.

    Benefits:
    - More stable predictions at inference
    - Often improves final accuracy by 0.1-0.5%
    - Smooths out training noise

    Used by: YOLO, EfficientNet, many SOTA models
    """
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (0.999 = very smooth, 0.9 = less smooth)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Call after each optimizer step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # EMA update: shadow = decay * shadow + (1-decay) * current
                self.shadow[name] = self.decay * self.shadow[name] + \
                                   (1 - self.decay) * param.data

    def apply_shadow(self):
        """Use EMA weights (for validation/inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (back to training)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Usage:
ema = EMA(model, decay=0.999)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        # ... training step ...
        optimizer.step()
        ema.update()  # Update EMA after each step

    # Validation with EMA weights
    ema.apply_shadow()
    model.eval()
    val_loss = validate(model, val_loader)
    ema.restore()

    # Save best model (with EMA)
    if val_loss < best_loss:
        ema.apply_shadow()
        torch.save(model.state_dict(), 'best_ema.pth')
        ema.restore()
```

### 4. Stochastic Weight Averaging (SWA)

```python
from torch.optim.swa_utils import AveragedModel, SWALR

"""
SWA: Average weights from multiple points in training.

Benefits:
- Better generalization than single model
- Wider optima → more robust
- Easy to implement

Reference: "Averaging Weights Leads to Wider Optima and Better Generalization"
"""

# Create SWA model
swa_model = AveragedModel(model)

# Special SWA learning rate scheduler
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

# Training
swa_start = 75  # Start SWA at epoch 75

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        # ... normal training ...
        optimizer.step()

    if epoch >= swa_start:
        # Update SWA model
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        # Normal scheduler before SWA
        scheduler.step()

# At the end: use SWA model for inference
torch.optim.swa_utils.update_bn(train_loader, swa_model)
torch.save(swa_model.state_dict(), 'swa_model.pth')
```

---

## 📈 Part 8: Practical Fine-Tuning Recipes

### Recipe 1: Fine-tuning on Similar Dataset

**Scenario:** You have a pretrained model on LOL, want to adapt to another low-light dataset.

```yaml
# config_finetune_similar.yaml
training:
  batch_size: 4
  epochs: 50 # Fewer epochs
  learning_rate: 0.00002 # 10x lower LR
  weight_decay: 0.0001
  warmup_epochs: 2 # Short warmup

resume:
  enabled: true
  checkpoint_path: "./pretrained/lol_best.pth"

# Fine-tuning strategy
fine_tune:
  freeze_layers: [] # Train all layers
  layer_lr_decay: 0.9 # Earlier layers get 0.9x LR
```

```python
# Implementation
def load_pretrained_finetune(model, checkpoint_path, lr=2e-5):
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained weights from {checkpoint_path}")

    # Layer-wise LR decay
    layer_params = []
    lr_decay = 0.9

    for i, layer in enumerate(model.layers):
        layer_lr = lr * (lr_decay ** i)
        layer_params.append({
            'params': layer.parameters(),
            'lr': layer_lr
        })

    # Higher LR for final layers
    layer_params.append({'params': model.conv_after_body.parameters(), 'lr': lr * 2})
    layer_params.append({'params': model.conv_last.parameters(), 'lr': lr * 2})

    optimizer = torch.optim.AdamW(layer_params, weight_decay=1e-4)
    return optimizer
```

### Recipe 2: Fine-tuning on Very Different Domain

**Scenario:** Pretrained on natural images, fine-tuning on medical/satellite/artistic images.

```yaml
# config_finetune_different.yaml
training:
  batch_size: 4
  epochs: 80 # More epochs
  learning_rate: 0.0001 # Moderate LR
  weight_decay: 0.0002 # Stronger regularization
  warmup_epochs: 5

fine_tune:
  freeze_layers: [0] # Freeze first layer
  progressive_unfreeze: true # Gradually unfreeze
  unfreeze_schedule: [20, 40] # Epochs to unfreeze
```

```python
def progressive_finetune(model, epoch, schedule=[20, 40]):
    """Gradually unfreeze layers"""
    if epoch < schedule[0]:
        # Freeze first layer
        for param in model.layers[0].parameters():
            param.requires_grad = False
    elif epoch < schedule[1]:
        # Unfreeze first, freeze nothing
        for param in model.parameters():
            param.requires_grad = True
    # After schedule[1]: everything trainable
```

### Recipe 3: Few-Shot Learning (< 100 samples)

**Scenario:** Very limited data, need maximum generalization.

```yaml
# config_fewshot.yaml
training:
  batch_size: 2 # Small batch
  epochs: 200 # Many epochs
  learning_rate: 0.00001 # Very low LR
  weight_decay: 0.001 # Strong regularization

data:
  augmentation: heavy # Maximum augmentation
  mixup: true
  mixup_alpha: 0.4
  cutmix: true

model:
  dropout: 0.3 # High dropout
  drop_path_rate: 0.2

loss:
  label_smoothing: 0.1 # Prevent overconfidence
```

```python
class FewShotTrainer:
    """Specialized trainer for few-shot scenarios"""
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data  # Small dataset

        # Aggressive augmentation
        self.aug = A.Compose([
            A.RandomCrop(96, 96),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
        ])

        # Very low learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=1e-3
        )

        # Early stopping (prevent overfitting)
        self.early_stopping = EarlyStopping(patience=20)

    def train_epoch(self):
        for batch in self.train_data:
            # Heavy augmentation + MixUp
            if random.random() < 0.5:
                batch = self.mixup(batch)

            # ... training ...
```

### Recipe 4: Domain Adaptation (Unpaired Data)

**Scenario:** No paired low/high images, need to enhance based on style.

```yaml
# config_domain_adapt.yaml
training:
  mode: "unsupervised"
  batch_size: 8
  epochs: 150
  learning_rate: 0.0002

loss:
  use_adversarial: true # GAN loss
  lambda_cycle: 10.0 # Cycle consistency
  lambda_identity: 5.0 # Identity loss

model:
  add_discriminator: true
```

---

## 🚀 Part 9: Your Immediate Action Plan

### Week 1: Establish Baseline ✅

- [x] Get training working
- [ ] Run for full 100 epochs
- [ ] Save checkpoints (best + periodic)
- [ ] Document metrics: Loss, PSNR, SSIM
- [ ] Save example outputs

**Goal:** Establish baseline performance for comparison.

### Week 2: Learning Rate Optimization

- [ ] Implement LR finder
- [ ] Test learning rates: `[1e-4, 2e-4, 5e-4]`
- [ ] Try different schedulers:
  - Cosine (current) ✅
  - OneCycleLR
  - ReduceLROnPlateau
- [ ] Document best configuration

**Goal:** Find optimal learning rate.

### Week 3: Loss Function Tuning

- [ ] Experiment with loss weights:

  ```yaml
  # Test 1: Conservative
  lambda_l1: 1.0, lambda_vgg: 0.05

  # Test 2: Balanced (current)
  lambda_l1: 1.0, lambda_vgg: 0.1

  # Test 3: Aggressive
  lambda_l1: 0.5, lambda_vgg: 0.2
  ```

- [ ] Add SSIM loss (optional)
- [ ] Visual comparison of results

**Goal:** Balance reconstruction vs perceptual quality.

### Week 4: Advanced Techniques

- [ ] Implement EMA (easy win)
- [ ] Add gradient accumulation (effective batch_size = 16)
- [ ] Try SWA for last 25 epochs
- [ ] Enhanced monitoring (TensorBoard)

**Goal:** Squeeze out final performance gains.

### Week 5: Validation & Analysis

- [ ] Comprehensive testing on all test sets
- [ ] Compute multiple metrics (PSNR, SSIM, LPIPS)
- [ ] Visual quality assessment
- [ ] Compare with baseline
- [ ] Document findings

**Goal:** Thorough evaluation and documentation.

---

## 📊 Part 10: Experiment Tracking Template

### Create experiment log file:

```python
# experiments/experiment_log.py
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_file = f"{exp_dir}/experiments.json"
        self.experiments = self.load_logs()

    def load_logs(self):
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def log_experiment(self, config, results):
        exp = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }
        self.experiments.append(exp)

        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def compare_experiments(self):
        """Compare all experiments"""
        for i, exp in enumerate(self.experiments):
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}: {exp['timestamp']}")
            print(f"{'='*60}")
            print(f"Config: {exp['config']}")
            print(f"Results: {exp['results']}")

# Usage:
logger = ExperimentLogger('./experiments')

# After training
logger.log_experiment(
    config={
        'lr': 2e-4,
        'batch_size': 4,
        'loss_weights': {'l1': 1.0, 'vgg': 0.1}
    },
    results={
        'train_loss': 0.045,
        'val_loss': 0.052,
        'psnr': 23.4,
        'ssim': 0.87
    }
)
```

---

## 📚 Part 11: Essential Resources

### Papers to Read (In Order)

1. **Fundamentals:**

   - "Attention is All You Need" - Transformer architecture
   - "An Image is Worth 16x16 Words: ViT" - Vision Transformers
   - "Swin Transformer" - Hierarchical vision transformer

2. **Training Techniques:**

   - "Bag of Tricks for Image Classification" - Practical tips
   - "Cyclical Learning Rates for Training Neural Networks"
   - "SGDR: Stochastic Gradient Descent with Warm Restarts"

3. **Low-Light Enhancement:**
   - "RetinexNet: Deep Retinex Decomposition"
   - "Zero-DCE: Learning Curve Parameter Maps"
   - "SCI: Self-Calibrated Illumination Learning"

### Books

1. **"Deep Learning" by Goodfellow, Bengio, Courville**

   - Best for theory and fundamentals

2. **"Dive into Deep Learning"** (d2l.ai)

   - Free online, very practical
   - Excellent code examples

3. **"Deep Learning with PyTorch"**
   - PyTorch-specific best practices

### Online Resources

1. **Papers with Code** (paperswithcode.com)

   - Find SOTA methods with implementations

2. **Weights & Biases Blog** (wandb.ai/site/articles)

   - Practical ML engineering articles

3. **PyTorch Forums** (discuss.pytorch.org)
   - Great for debugging help

### Tools to Master

1. **Weights & Biases (wandb)**

   ```python
   import wandb

   wandb.init(project="swinllie", config={
       "learning_rate": 2e-4,
       "batch_size": 4,
       "epochs": 100
   })

   # Log metrics
   wandb.log({"loss": loss, "psnr": psnr})
   ```

2. **Optuna** (Hyperparameter optimization)

   ```python
   import optuna

   def objective(trial):
       lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
       batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

       model = train_model(lr=lr, batch_size=batch_size)
       return validate(model)

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=20)
   ```

3. **PyTorch Lightning** (Cleaner code)

   ```python
   import pytorch_lightning as pl

   class LitSwinLLIE(pl.LightningModule):
       def training_step(self, batch, batch_idx):
           # ... training logic ...
           return loss

       def configure_optimizers(self):
           return torch.optim.AdamW(self.parameters(), lr=2e-4)

   trainer = pl.Trainer(gpus=1, max_epochs=100)
   trainer.fit(model, train_loader)
   ```

---

## 🎯 Part 12: Key Takeaways & Golden Rules

### The 10 Commandments of Fine-Tuning

1. **Learning Rate is King** 👑

   - Most important hyperparameter
   - Use LR finder to find optimal value
   - Lower LR for fine-tuning (10-100x smaller)

2. **Start Simple, Add Complexity** 🔧

   - Begin with single loss (L1)
   - Add components one at a time
   - Validate each addition

3. **Monitor Everything** 📊

   - Log metrics, gradients, activations
   - Save images, not just numbers
   - Use TensorBoard/WandB

4. **Validate Visually** 👁️

   - Metrics can lie, eyes don't
   - Save example outputs regularly
   - Compare across experiments

5. **Be Systematic** 📋

   - Change ONE thing at a time
   - Document all experiments
   - Use config files

6. **Don't Overtrain** ⏱️

   - Watch for overfitting
   - Use early stopping
   - Validate regularly

7. **Use Mixed Precision** ⚡

   - Free 2x speedup
   - Less memory usage
   - Minimal accuracy loss

8. **Augment Wisely** 🎲

   - Domain-appropriate only
   - Don't destroy important features
   - Paired data needs paired augmentation

9. **Regularize Appropriately** 🛡️

   - Weight decay for large models
   - Dropout for small datasets
   - Gradient clipping for stability

10. **Trust the Process** 🧘
    - Deep learning needs patience
    - Some experiments will fail
    - Keep iterating

### Quick Decision Tree

```
Need better performance?
├─ Loss not decreasing?
│  ├─ Check learning rate (try LR finder)
│  ├─ Check gradients (vanishing/exploding?)
│  └─ Check data (normalized? augmented?)
│
├─ Overfitting (val >> train)?
│  ├─ More data augmentation
│  ├─ Add dropout/regularization
│  ├─ Early stopping
│  └─ Reduce model size
│
├─ Underfitting (both losses high)?
│  ├─ Larger model
│  ├─ More epochs
│  ├─ Higher learning rate
│  └─ Remove regularization
│
└─ Close to optimal?
   ├─ Try EMA/SWA
   ├─ Ensemble models
   ├─ Fine-tune loss weights
   └─ Advanced augmentation
```

### Performance Checklist ✅

Before claiming "training is done":

- [ ] Trained for sufficient epochs (loss plateaued)
- [ ] Tried at least 3 different learning rates
- [ ] Validated on held-out test set
- [ ] Saved example outputs
- [ ] Computed multiple metrics (PSNR, SSIM, visual)
- [ ] Compared with baseline/previous best
- [ ] Checked for overfitting
- [ ] Documented configuration
- [ ] Tested on edge cases
- [ ] Model saved and reproducible

---

## 🎓 Final Words

Fine-tuning is both **art and science**:

- **Science:** Systematic experimentation, metric tracking
- **Art:** Intuition about what to try next, visual assessment

**Key to success:**

1. Solid fundamentals (you have this!)
2. Systematic approach (use this guide)
3. Patience and iteration (keep at it!)

Remember: Even SOTA papers typically try 50-100+ experiments before finding the best configuration. Don't get discouraged!

---

## 📞 Quick Reference Card

### Essential Commands

```bash
# Find optimal LR
python train.py --find_lr

# Train with config
python train.py --config configs/swinllie_lol.yaml

# Resume training
python train.py --resume experiments/swinllie_lol/checkpoints/last.pth

# Validate model
python test.py --checkpoint best.pth --save_images

# TensorBoard
tensorboard --logdir experiments/swinllie_lol/logs

# Monitor GPU
watch -n 1 nvidia-smi
```

### Loss Weight Presets

```python
# Conservative (preserve details)
LOSS_CONSERVATIVE = {'l1': 1.0, 'vgg': 0.05, 'color': 0.3, 'smooth': 0.005}

# Balanced (your default)
LOSS_BALANCED = {'l1': 1.0, 'vgg': 0.1, 'color': 0.5, 'smooth': 0.01}

# Aggressive (max quality)
LOSS_AGGRESSIVE = {'l1': 0.5, 'vgg': 0.2, 'color': 0.8, 'smooth': 0.02}
```

### Learning Rate Presets

```python
# Training from scratch
LR_SCRATCH = 2e-4

# Fine-tuning (similar domain)
LR_FINETUNE_SIMILAR = 2e-5

# Fine-tuning (different domain)
LR_FINETUNE_DIFFERENT = 1e-4

# Few-shot learning
LR_FEWSHOT = 1e-5
```

---

**Good luck with your fine-tuning journey!** 🚀

Remember: Every expert was once a beginner. Keep experimenting, keep learning!
