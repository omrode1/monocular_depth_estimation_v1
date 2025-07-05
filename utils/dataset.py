import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MonocularDepthDataset(Dataset):
    """
    Dataset class for monocular depth estimation.
    Supports both supervised (with depth ground truth) and unsupervised training.
    """
    
    def __init__(self, data_path, split='train', transform=None, 
                 load_depth=False, min_depth=0.1, max_depth=100.0):
        """
        Args:
            data_path: Path to the dataset directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            load_depth: Whether to load depth ground truth (for supervised training)
            min_depth: Minimum depth value
            max_depth: Maximum depth value
        """
        self.data_path = data_path
        self.split = split
        self.load_depth = load_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Default transforms
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(height=192, width=640),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=192, width=640),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        
        if self.load_depth:
            self.depth_paths = self._load_depth_paths()
    
    def _load_image_paths(self):
        """Load paths to RGB images."""
        image_dir = os.path.join(self.data_path, self.split, 'images')
        if not os.path.exists(image_dir):
            # Try alternative structure
            image_dir = os.path.join(self.data_path, 'images')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_paths.extend([
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(ext)
            ])
        
        return sorted(image_paths)
    
    def _load_depth_paths(self):
        """Load paths to depth ground truth images."""
        depth_dir = os.path.join(self.data_path, self.split, 'depth')
        if not os.path.exists(depth_dir):
            depth_dir = os.path.join(self.data_path, 'depth')
        
        if not os.path.exists(depth_dir):
            raise ValueError(f"Depth directory not found: {depth_dir}")
        
        depth_paths = []
        for ext in ['png', 'npy']:
            depth_paths.extend([
                os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                if f.lower().endswith(ext)
            ])
        
        return sorted(depth_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        sample = {'image': image, 'image_path': image_path}
        
        # Load depth if available
        if self.load_depth and idx < len(self.depth_paths):
            depth_path = self.depth_paths[idx]
            
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    depth = np.zeros((image.shape[1], image.shape[2]))
            
            # Normalize depth
            depth = np.clip(depth, self.min_depth, self.max_depth)
            depth = 1.0 / depth  # Convert to disparity
            
            # Resize depth to match image
            if depth.shape != (image.shape[1], image.shape[2]):
                depth = cv2.resize(depth, (image.shape[2], image.shape[1]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            depth = torch.from_numpy(depth).float().unsqueeze(0)
            sample['depth'] = depth
            sample['depth_path'] = depth_path
        
        return sample

class KITTIDataset(MonocularDepthDataset):
    """
    KITTI dataset loader for monocular depth estimation.
    """
    
    def __init__(self, data_path, split='train', **kwargs):
        super().__init__(data_path, split, **kwargs)
    
    def _load_image_paths(self):
        """Load KITTI image paths."""
        image_dir = os.path.join(self.data_path, 'image_02', 'data')
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.data_path, 'images')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"KITTI image directory not found: {image_dir}")
        
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_paths.extend([
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(ext)
            ])
        
        return sorted(image_paths)
    
    def _load_depth_paths(self):
        """Load KITTI depth paths."""
        depth_dir = os.path.join(self.data_path, 'gt_depth')
        if not os.path.exists(depth_dir):
            depth_dir = os.path.join(self.data_path, 'depth')
        
        if not os.path.exists(depth_dir):
            raise ValueError(f"KITTI depth directory not found: {depth_dir}")
        
        depth_paths = []
        for ext in ['png', 'npy']:
            depth_paths.extend([
                os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                if f.lower().endswith(ext)
            ])
        
        return sorted(depth_paths) 