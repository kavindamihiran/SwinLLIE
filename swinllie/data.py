# -----------------------------------------------------------------------------------
# Dataset Loader for Low-Light Image Enhancement
# Supports LOL, SID, VE-LOL and custom datasets
# -----------------------------------------------------------------------------------

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class LowLightDataset(Dataset):
    """
    Dataset loader for low-light image enhancement.
    
    Supports popular low-light datasets:
        1. LOL (LOw-Light) Dataset - Most commonly used
        2. SID (See-in-the-Dark) Dataset
        3. VE-LOL Dataset
        4. Any custom paired dataset
    
    Expected folder structure:
        dataset_root/
        ├── train/
        │   ├── low/       # Low-light images
        │   └── high/      # Normal-light ground truth
        └── test/
            ├── low/
            └── high/
    
    Args:
        root_dir: Path to dataset root
        split: 'train' or 'test'
        patch_size: Size of random crop for training (default: 128)
        augment: Whether to apply data augmentation (default: True for train)
    """
    
    def __init__(self, root_dir, split='train', patch_size=128, augment=True):
        super().__init__()
        
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        
        # Setup paths
        self.low_dir = os.path.join(root_dir, split, 'low')
        self.high_dir = os.path.join(root_dir, split, 'high')
        
        # Check if directories exist
        if not os.path.exists(self.low_dir):
            raise ValueError(f"Low-light image directory not found: {self.low_dir}")
        if not os.path.exists(self.high_dir):
            raise ValueError(f"High-light image directory not found: {self.high_dir}")
        
        # Get image file names
        self.image_names = self._get_image_pairs()
        
        print(f"[{split.upper()}] Loaded {len(self.image_names)} image pairs from {root_dir}")
    
    def _get_image_pairs(self):
        """Get matched low/high image pairs based on filename."""
        low_images = set(os.listdir(self.low_dir))
        high_images = set(os.listdir(self.high_dir))
        
        # Find common filenames (matched pairs)
        # Handle case where filenames might differ slightly
        paired_names = []
        
        for low_name in low_images:
            # Try exact match first
            if low_name in high_images:
                paired_names.append(low_name)
            else:
                # Try matching base name (without extension differences)
                low_base = os.path.splitext(low_name)[0]
                for high_name in high_images:
                    high_base = os.path.splitext(high_name)[0]
                    if low_base == high_base:
                        paired_names.append((low_name, high_name))
                        break
        
        # Convert to list of tuples if needed
        result = []
        for item in paired_names:
            if isinstance(item, tuple):
                result.append(item)
            else:
                result.append((item, item))
        
        if len(result) == 0:
            raise ValueError(f"No matching image pairs found in {self.root_dir}/{self.split}")
        
        return sorted(result)
    
    def _load_image(self, path):
        """Load image and convert to tensor."""
        img = Image.open(path).convert('RGB')
        return img
    
    def _random_crop(self, low_img, high_img):
        """Apply random crop to both images with same location."""
        w, h = low_img.size
        
        if w < self.patch_size or h < self.patch_size:
            # Resize if image is smaller than patch size
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            low_img = low_img.resize((new_w, new_h), Image.BICUBIC)
            high_img = high_img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h
        
        # Random crop coordinates
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        
        low_img = TF.crop(low_img, y, x, self.patch_size, self.patch_size)
        high_img = TF.crop(high_img, y, x, self.patch_size, self.patch_size)
        
        return low_img, high_img
    
    def _augment(self, low_img, high_img):
        """Apply data augmentation (same transform to both images)."""
        # Random horizontal flip
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            low_img = TF.vflip(low_img)
            high_img = TF.vflip(high_img)
        
        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            low_img = TF.rotate(low_img, angle)
            high_img = TF.rotate(high_img, angle)
        
        return low_img, high_img
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """
        Get a training/testing sample.
        
        Returns:
            low_img: Low-light input tensor (3, H, W) in [0, 1]
            high_img: Ground truth tensor (3, H, W) in [0, 1]
            img_name: Image filename (for saving results)
        """
        low_name, high_name = self.image_names[idx]
        
        # Load images
        low_path = os.path.join(self.low_dir, low_name)
        high_path = os.path.join(self.high_dir, high_name)
        
        low_img = self._load_image(low_path)
        high_img = self._load_image(high_path)
        
        # Ensure same size
        if low_img.size != high_img.size:
            high_img = high_img.resize(low_img.size, Image.BICUBIC)
        
        # Training: random crop + augmentation
        if self.split == 'train':
            low_img, high_img = self._random_crop(low_img, high_img)
            if self.augment:
                low_img, high_img = self._augment(low_img, high_img)
        
        # Convert to tensor [0, 1]
        to_tensor = transforms.ToTensor()
        low_tensor = to_tensor(low_img)
        high_tensor = to_tensor(high_img)
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'name': os.path.splitext(low_name)[0]
        }


class LOLDataset(LowLightDataset):
    """
    LOL (LOw-Light) Dataset specific loader.
    
    LOL Dataset structure:
        LOL/
        ├── our485/          # Training set (485 pairs)
        │   ├── low/
        │   └── high/
        └── eval15/          # Test set (15 pairs)
            ├── low/
            └── high/
    
    Download: https://daooshee.github.io/BMVC2018website/
    """
    
    def __init__(self, root_dir, split='train', patch_size=128, augment=True):
        # Map split names to LOL folder structure
        if split == 'train':
            actual_dir = os.path.join(root_dir, 'our485')
        else:
            actual_dir = os.path.join(root_dir, 'eval15')
        
        # LOL has slightly different structure
        self.low_dir = os.path.join(actual_dir, 'low')
        self.high_dir = os.path.join(actual_dir, 'high')
        
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        
        self.image_names = self._get_image_pairs()
        print(f"[LOL-{split.upper()}] Loaded {len(self.image_names)} image pairs")


class UnpairedLowLightDataset(Dataset):
    """
    Dataset for unpaired low-light images (inference only).
    
    Use this when you only have low-light images without ground truth.
    
    Args:
        image_dir: Directory containing low-light images
        resize: Optional resize dimension (None for original size)
    """
    
    def __init__(self, image_dir, resize=None):
        self.image_dir = image_dir
        self.resize = resize
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_names = [
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        self.image_names = sorted(self.image_names)
        
        print(f"[INFERENCE] Found {len(self.image_names)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (W, H)
        
        if self.resize:
            img = img.resize((self.resize, self.resize), Image.BICUBIC)
        
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        
        return {
            'low': img_tensor,
            'name': os.path.splitext(img_name)[0],
            'original_size': original_size
        }


def get_dataloader(dataset_type, root_dir, split='train', batch_size=8, 
                   patch_size=128, num_workers=4, pin_memory=True):
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_type: 'lol', 'generic', or 'unpaired'
        root_dir: Path to dataset root
        split: 'train' or 'test'
        batch_size: Batch size
        patch_size: Training patch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    if dataset_type.lower() == 'lol':
        dataset = LOLDataset(root_dir, split, patch_size)
    elif dataset_type.lower() == 'unpaired':
        dataset = UnpairedLowLightDataset(root_dir)
    else:
        dataset = LowLightDataset(root_dir, split, patch_size)
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches for faster loading
    )
    
    return dataloader


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Dataset Loaders")
    print("=" * 60)
    
    # Test with dummy data structure
    import tempfile
    import shutil
    
    # Create temporary dataset structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create folders
        for split in ['train', 'test']:
            for subdir in ['low', 'high']:
                os.makedirs(os.path.join(tmpdir, split, subdir))
        
        # Create dummy images
        for i in range(5):
            for split in ['train', 'test']:
                # Create dummy low-light image (darker)
                low_img = Image.fromarray(np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8))
                low_img.save(os.path.join(tmpdir, split, 'low', f'img_{i:03d}.png'))
                
                # Create dummy high-light image (brighter)
                high_img = Image.fromarray(np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8))
                high_img.save(os.path.join(tmpdir, split, 'high', f'img_{i:03d}.png'))
        
        print(f"\n✓ Created dummy dataset at {tmpdir}")
        
        # Test LowLightDataset
        print("\n1. Testing LowLightDataset...")
        dataset = LowLightDataset(tmpdir, split='train', patch_size=128)
        sample = dataset[0]
        print(f"   Low shape: {sample['low'].shape}")
        print(f"   High shape: {sample['high'].shape}")
        print(f"   Name: {sample['name']}")
        
        # Test DataLoader
        print("\n2. Testing DataLoader...")
        dataloader = get_dataloader('generic', tmpdir, split='train', batch_size=2)
        batch = next(iter(dataloader))
        print(f"   Batch low shape: {batch['low'].shape}")
        print(f"   Batch high shape: {batch['high'].shape}")
        
        # Test unpaired dataset
        print("\n3. Testing UnpairedLowLightDataset...")
        unpaired_dataset = UnpairedLowLightDataset(os.path.join(tmpdir, 'test', 'low'))
        sample = unpaired_dataset[0]
        print(f"   Image shape: {sample['low'].shape}")
        print(f"   Original size: {sample['original_size']}")
    
    print("\n" + "=" * 60)
    print("All dataset tests passed!")
    print("=" * 60)
