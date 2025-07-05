#!/usr/bin/env python3
"""
Demo script for Monocular Depth Estimation
This script demonstrates the capabilities of the depth estimation model
with sample images, real-time processing, and visualization.
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.monodepth import MonodepthModel
from utils.metrics import visualize_depth

def download_sample_images():
    """Download sample images for demonstration."""
    sample_urls = [
        "https://raw.githubusercontent.com/isl-org/DPT/main/input/room.jpg",
        "https://raw.githubusercontent.com/isl-org/DPT/main/input/road.jpg"
    ]
    
    os.makedirs("demo_images", exist_ok=True)
    
    for i, url in enumerate(sample_urls):
        filename = f"demo_images/sample_{i+1}.jpg"
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
    
    return [f"demo_images/sample_{i+1}.jpg" for i in range(len(sample_urls))]

def create_dummy_model():
    """Create a dummy model for demonstration purposes."""
    print("Creating dummy model for demonstration...")
    
    model = MonodepthModel(
        num_layers=18,
        scales=range(4),
        use_skips=True,
        pretrained=False
    )
    
    # Initialize with random weights
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    
    return model

def preprocess_image(image, width=640, height=192):
    """Preprocess image for model input."""
    # Resize image
    image = cv2.resize(image, (width, height))
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image

def postprocess_depth(depth_map, colormap='plasma'):
    """Postprocess depth map for visualization."""
    if torch.is_tensor(depth_map):
        depth_map = depth_map.detach().cpu().numpy()
    
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    
    # Normalize for visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    return depth_colored, depth_normalized

def demo_single_image(model, image_path, device, output_dir="demo_output"):
    """Demonstrate depth estimation on a single image."""
    print(f"\nProcessing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Preprocess
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = time.time() - start_time
    
    # Get depth prediction
    depth_pred = outputs['disp_0']
    
    # Postprocess
    depth_colored, depth_normalized = postprocess_depth(depth_pred)
    
    # Resize depth to match original image
    original_height, original_width = image.shape[:2]
    depth_colored = cv2.resize(depth_colored, (original_width, original_height))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.jpg'), image)
    
    # Save depth visualization
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_depth.jpg'), depth_colored)
    
    # Create side-by-side comparison
    combined = np.hstack([image, depth_colored])
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_combined.jpg'), combined)
    
    # Create matplotlib visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Depth map
    axes[1].imshow(depth_normalized, cmap='plasma')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    # Combined
    axes[2].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Combined')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Processing completed in {inference_time:.3f} seconds")
    print(f"Results saved to: {output_dir}")
    
    return inference_time

def demo_real_time(model, device, duration=10):
    """Demonstrate real-time depth estimation."""
    print(f"\nStarting real-time demo for {duration} seconds...")
    print("Press 'q' to quit early")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    frame_count = 0
    total_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if time limit reached
        if time.time() - start_time > duration:
            break
        
        # Process frame
        try:
            input_tensor = preprocess_image(frame)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                inference_start = time.time()
                outputs = model(input_tensor)
                inference_time = time.time() - inference_start
            
            depth_pred = outputs['disp_0']
            depth_colored, _ = postprocess_depth(depth_pred)
            
            # Resize depth to match frame
            original_height, original_width = frame.shape[:2]
            depth_colored = cv2.resize(depth_colored, (original_width, original_height))
            
            # Combine original and depth
            combined = np.hstack([frame, depth_colored])
            
            # Add FPS info
            fps = 1.0 / inference_time
            cv2.putText(combined, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f'Time: {time.time() - start_time:.1f}s', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Real-time Monocular Depth Estimation Demo', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
            total_inference_time += inference_time
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_fps = frame_count / total_inference_time
        print(f"Real-time demo completed:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total time: {time.time() - start_time:.2f} seconds")

def demo_model_architecture():
    """Demonstrate the model architecture."""
    print("\n=== Model Architecture Demo ===")
    
    # Create model
    model = MonodepthModel(
        num_layers=18,
        scales=range(4),
        use_skips=True,
        pretrained=False
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 192, 640)
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    return model

def load_trained_model():
    """Load the trained model."""
    model = MonodepthModel(
        num_layers=18,
        scales=range(0, 4),
        use_skips=True,
        pretrained=False
    )
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully!")
    return model

def predict_depth_from_image(model, image_path, output_path=None):
    """Predict depth from a single image."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for model input
    target_size = (640, 192)
    image_resized = cv2.resize(image_rgb, target_size)
    
    # Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    image_norm = (image_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    
    # Predict depth
    with torch.no_grad():
        outputs = model(image_tensor)
        depth = outputs['disp_0'].squeeze()
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(image_resized)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Predicted depth
    depth_np = depth.cpu().numpy()
    im = ax2.imshow(depth_np, cmap='plasma')
    ax2.set_title('Predicted Depth')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
    
    plt.show()
    
    # Print statistics
    print(f"Depth range: {depth_np.min():.3f} to {depth_np.max():.3f}")
    print(f"Depth mean: {depth_np.mean():.3f}")
    print(f"Depth std: {depth_np.std():.3f}")
    
    return depth_np

def main():
    """Main demo function."""
    print("🎯 Monocular Depth Estimation Demo")
    print("=" * 40)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Demo options
    print("\nChoose an option:")
    print("1. Test on sample validation image")
    print("2. Test on your own image")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Test on a sample validation image
        sample_image = "data/paired/val/images/img-060705-17.10.14-p-285t000.jpg"
        if os.path.exists(sample_image):
            print(f"\n📸 Testing on sample image: {os.path.basename(sample_image)}")
            predict_depth_from_image(model, sample_image, "demo_sample_result.png")
        else:
            print("❌ Sample image not found!")
    
    elif choice == "2":
        # Test on user's image
        image_path = input("\nEnter the path to your image: ").strip()
        if os.path.exists(image_path):
            print(f"\n📸 Processing your image: {os.path.basename(image_path)}")
            predict_depth_from_image(model, image_path, "demo_custom_result.png")
        else:
            print("❌ Image not found!")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice!")

if __name__ == '__main__':
    main() 