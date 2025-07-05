import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.monodepth import MonodepthModel
from utils.metrics import visualize_depth

def load_model(checkpoint_path):
    """Load the trained model."""
    # Create model
    model = MonodepthModel(
        num_layers=50,
        scales=range(0, 4),
        use_skips=True,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if checkpoint is a dict with model_state_dict or just the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        # Checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    
    model.eval()
    return model

def resize_preserve_aspect_ratio(image, target_size=(640, 192)):
    """Resize image while preserving aspect ratio and padding if necessary."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor to fit within target size
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)  # Use the smaller scale to fit within bounds
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image with target size
    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image in center
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return padded

def preprocess_image_original_aspect(image_path, max_size=640):
    """Preprocess image while preserving original aspect ratio."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within max_size while preserving aspect ratio
    scale = min(max_size / w, max_size / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Normalize
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Convert to tensor
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, (h, w), (new_h, new_w)

def preprocess_image(image_path, target_size=(640, 192)):
    """Preprocess image for model input."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while preserving aspect ratio
    image = resize_preserve_aspect_ratio(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    return image

def predict_depth(model, image_tensor):
    """Predict depth from image tensor."""
    with torch.no_grad():
        outputs = model(image_tensor)
        depth = outputs['disp_0'].squeeze()  # Get highest resolution prediction
    
    return depth

def visualize_results(original_image, predicted_depth, save_path=None):
    """Visualize original image and predicted depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Predicted depth - handle both tensor and numpy array inputs
    if hasattr(predicted_depth, 'cpu'):
        depth_np = predicted_depth.cpu().numpy()
    else:
        depth_np = predicted_depth
    
    im = ax2.imshow(depth_np, cmap='plasma')
    ax2.set_title('Predicted Depth')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    
    plt.show()

def test_model():
    """Test the trained model on sample images."""
    print("🧪 Testing trained model...")
    
    # Check if model exists
    checkpoint_path = 'checkpoints_improved/best_a1_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        return
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Test on a few sample images
    test_images = [
        'data/paired/val/images/img-060705-17.10.14-p-285t000.jpg',
        'data/paired/val/images/img-060705-17.10.14-p-346t000.jpg',
        'data/paired/val/images/img-060705-17.28.10-p-223t000.jpg',
        'data/paired/val/images/img-060705-17.28.10-p-223t000.jpg',
        'data/paired/val/images/img-060705-17.28.10-p-223t000.jpg',
        'data/paired/val/images/img-060705-17.28.10-p-223t000.jpg'
    ]
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        print(f"\n📸 Testing image {i+1}: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        original_h, original_w = original_image.shape[:2]
        
        # Preprocess for model (preserve aspect ratio)
        image_tensor, (orig_h, orig_w), (model_h, model_w) = preprocess_image_original_aspect(image_path)
        
        # Predict depth
        predicted_depth = predict_depth(model, image_tensor)
        
        # Resize depth prediction to original image dimensions
        predicted_depth_resized = cv2.resize(predicted_depth.cpu().numpy(), (original_w, original_h))
        
        # Resize original image for display (smaller size for visualization)
        display_size = (800, 600)  # Reasonable display size
        display_image = cv2.resize(original_image, display_size)
        
        # Resize depth prediction to match display size
        display_depth = cv2.resize(predicted_depth_resized, display_size)
        
        # Visualize results
        save_path = f'test_results_{i+1}.png'
        visualize_results(display_image, display_depth, save_path)
        
        # Print depth statistics
        depth_np = predicted_depth_resized
        print(f"  Depth range: {depth_np.min():.3f} to {depth_np.max():.3f}")
        print(f"  Depth mean: {depth_np.mean():.3f}")
        print(f"  Depth std: {depth_np.std():.3f}")

if __name__ == '__main__':
    test_model() 