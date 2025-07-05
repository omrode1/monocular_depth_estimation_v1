import os
import numpy as np
import cv2
from utils.dataset import MonocularDepthDataset

def test_paired_dataset():
    print("Testing paired dataset...")
    
    # Check if paired data exists
    img_dir = 'data/paired/train/images'
    depth_dir = 'data/paired/train/depth'
    
    if not os.path.exists(img_dir) or not os.path.exists(depth_dir):
        print("❌ Paired dataset directories not found!")
        return False
    
    # Count files
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    
    print(f"📁 Images: {len(img_files)}")
    print(f"📁 Depths: {len(depth_files)}")
    
    if len(img_files) != len(depth_files):
        print("❌ Mismatch in number of images and depth files!")
        return False
    
    # Test loading a few samples
    print("\n🔍 Testing data loading...")
    
    for i in range(min(3, len(img_files))):
        img_path = os.path.join(img_dir, img_files[i])
        depth_path = os.path.join(depth_dir, depth_files[i])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to load image: {img_path}")
            return False
        
        # Load depth
        depth = np.load(depth_path)
        
        print(f"  Sample {i+1}:")
        print(f"    Image: {img.shape} (dtype: {img.dtype})")
        print(f"    Depth: {depth.shape} (dtype: {depth.dtype})")
        print(f"    Depth range: {np.min(depth):.2f} to {np.max(depth):.2f}")
    
    # Test with our dataset class
    print("\n🧪 Testing with MonocularDepthDataset...")
    
    try:
        dataset = MonocularDepthDataset(
            data_path='data/paired',
            split='train',
            load_depth=True,
            transform=None
        )
        
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading one sample
        sample = dataset[0]
        img, depth = sample['image'], sample['depth']
        
        print(f"  Sample shape: {img.shape}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Image range: {img.min():.2f} to {img.max():.2f}")
        print(f"  Depth range: {depth.min():.2f} to {depth.max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

if __name__ == '__main__':
    success = test_paired_dataset()
    if success:
        print("\n🎉 Dataset test passed! Ready for training.")
    else:
        print("\n❌ Dataset test failed!") 