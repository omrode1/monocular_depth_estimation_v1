import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from models.monodepth import MonodepthModel
from utils.metrics import visualize_depth

def parse_args():
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation Inference')
    
    # Input arguments
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image/video path or "webcam" for real-time')
    parser.add_argument('--output', type=str, default='output', 
                       help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--num_layers', type=int, default=18, 
                       choices=[18, 34, 50], help='ResNet depth')
    parser.add_argument('--scales', type=str, default='0,1,2,3', 
                       help='Depth prediction scales')
    parser.add_argument('--use_skips', action='store_true', default=True, 
                       help='Use skip connections')
    
    # Processing arguments
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--height', type=int, default=192, help='Input height')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # Visualization arguments
    parser.add_argument('--colormap', type=str, default='plasma', 
                       help='Colormap for depth visualization')
    parser.add_argument('--save_video', action='store_true', 
                       help='Save output video')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for inference."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def load_model(checkpoint_path, args, device):
    """Load trained model from checkpoint."""
    # Create model
    scales = [int(s) for s in args.scales.split(',')]
    model = MonodepthModel(
        num_layers=args.num_layers,
        scales=scales,
        use_skips=args.use_skips,
        pretrained=False
    )
    
    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image, width, height):
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

def process_image(model, image_path, args, device):
    """Process a single image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess
    input_tensor = preprocess_image(image, args.width, args.height)
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = time.time() - start_time
    
    # Get depth prediction
    depth_pred = outputs['disp_0']
    
    # Postprocess
    depth_colored, depth_normalized = postprocess_depth(depth_pred, args.colormap)
    
    # Resize depth to match original image
    original_height, original_width = image.shape[:2]
    depth_colored = cv2.resize(depth_colored, (original_width, original_height))
    
    return image, depth_colored, depth_normalized, inference_time

def process_video(model, video_path, args, device):
    """Process video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    output_path = None
    video_writer = None
    if args.save_video:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, 'depth_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width * 2, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        try:
            input_tensor = preprocess_image(frame, args.width, args.height)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_tensor)
                inference_time = time.time() - start_time
            
            depth_pred = outputs['disp_0']
            depth_colored, _ = postprocess_depth(depth_pred, args.colormap)
            
            # Resize depth to match frame
            depth_colored = cv2.resize(depth_colored, (width, height))
            
            # Combine original and depth
            combined = np.hstack([frame, depth_colored])
            
            # Save frame
            if video_writer:
                video_writer.write(combined)
            
            # Display
            cv2.imshow('Monocular Depth Estimation', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            total_time += inference_time
            
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_time
                print(f"Processed {frame_count}/{total_frames} frames, Avg FPS: {avg_fps:.2f}")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue
    
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed. Output saved to: {output_path}")

def process_webcam(model, args, device):
    """Process real-time webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        try:
            input_tensor = preprocess_image(frame, args.width, args.height)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_tensor)
                inference_time = time.time() - start_time
            
            depth_pred = outputs['disp_0']
            depth_colored, _ = postprocess_depth(depth_pred, args.colormap)
            
            # Resize depth to match frame
            original_height, original_width = frame.shape[:2]
            depth_colored = cv2.resize(depth_colored, (original_width, original_height))
            
            # Combine original and depth
            combined = np.hstack([frame, depth_colored])
            
            # Add FPS info
            fps = 1.0 / inference_time
            cv2.putText(combined, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Real-time Monocular Depth Estimation', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                os.makedirs(args.output, exist_ok=True)
                save_path = os.path.join(args.output, f'frame_{frame_count}.jpg')
                cv2.imwrite(save_path, combined)
                print(f"Frame saved: {save_path}")
            
            frame_count += 1
            total_time += inference_time
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Average FPS: {avg_fps:.2f}")

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model = load_model(args.checkpoint, args, device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input
    if args.input.lower() == 'webcam':
        process_webcam(model, args, device)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(model, args.input, args, device)
    else:
        # Process single image
        image, depth_colored, depth_normalized, inference_time = process_image(
            model, args.input, args, device
        )
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Save original image
        cv2.imwrite(os.path.join(args.output, f'{base_name}_original.jpg'), image)
        
        # Save depth visualization
        cv2.imwrite(os.path.join(args.output, f'{base_name}_depth.jpg'), depth_colored)
        
        # Save depth map as numpy array
        np.save(os.path.join(args.output, f'{base_name}_depth.npy'), depth_normalized)
        
        # Create side-by-side comparison
        combined = np.hstack([image, depth_colored])
        cv2.imwrite(os.path.join(args.output, f'{base_name}_combined.jpg'), combined)
        
        print(f"Processing completed in {inference_time:.3f} seconds")
        print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main() 