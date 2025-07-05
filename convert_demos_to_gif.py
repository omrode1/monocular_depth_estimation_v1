#!/usr/bin/env python3
"""
Convert Demo Videos to GIF Format
This script converts the large MP4 demo videos to compressed GIF format
for better GitHub compatibility and faster loading.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ OpenCV, NumPy, and PIL are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install opencv-python numpy pillow")
        return False

def convert_video_to_gif_opencv(video_path, output_path, max_frames=50, scale_factor=0.5):
    """
    Convert video to GIF using OpenCV and PIL.
    This is a fallback method when ffmpeg is not available.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        print(f"Converting {video_path} to GIF...")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval to get max_frames
        frame_interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % frame_interval == 0:
                # Resize frame
                height, width = frame.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print("No frames extracted from video")
            return False
        
        # Save as GIF with slower playback
        # Use a fixed duration of 200ms per frame for slower playback
        frame_duration = 200  # milliseconds per frame
        
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,  # Fixed duration for slower playback
            loop=0,
            optimize=True
        )
        
        # Get file sizes
        original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        gif_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✓ Conversion complete!")
        print(f"  Original: {original_size:.1f} MB")
        print(f"  GIF: {gif_size:.1f} MB")
        print(f"  Compression: {((original_size - gif_size) / original_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def convert_with_ffmpeg(video_path, output_path, scale_factor=0.5):
    """
    Convert video to GIF using ffmpeg (if available).
    This method produces better quality GIFs.
    """
    try:
        # Create ffmpeg command for high-quality GIF conversion
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            '-loop', '0',
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"Converting {video_path} to GIF using ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Get file sizes
            original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            gif_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f"✓ Conversion complete!")
            print(f"  Original: {original_size:.1f} MB")
            print(f"  GIF: {gif_size:.1f} MB")
            print(f"  Compression: {((original_size - gif_size) / original_size * 100):.1f}%")
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ffmpeg not found, using OpenCV fallback...")
        return False
    except Exception as e:
        print(f"Error during ffmpeg conversion: {e}")
        return False

def main():
    """Main function to convert all demo videos to GIFs."""
    print("Converting Demo Videos to GIF Format")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Demo video files
    demo_files = [
        ('demo/main_demo.mp4', 'demo/main_demo.gif'),
        ('demo/batch_processing_demo.mp4', 'demo/batch_processing_demo.gif'),
        ('demo/comparison_demo.mp4', 'demo/comparison_demo.gif')
    ]
    
    # Create demo directory if it doesn't exist
    os.makedirs('demo', exist_ok=True)
    
    successful_conversions = 0
    
    for video_path, gif_path in demo_files:
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found, skipping...")
            continue
        
        print(f"\nProcessing: {video_path}")
        
        # Try ffmpeg first, then fallback to OpenCV
        success = convert_with_ffmpeg(video_path, gif_path)
        
        if not success:
            print("Trying OpenCV conversion...")
            success = convert_video_to_gif_opencv(video_path, gif_path, max_frames=60, scale_factor=0.4)
        
        if success:
            successful_conversions += 1
        else:
            print(f"Failed to convert {video_path}")
    
    print(f"\n{'='*50}")
    print(f"Conversion Summary:")
    print(f"  Successful: {successful_conversions}/{len(demo_files)}")
    
    if successful_conversions > 0:
        print(f"\nNext steps:")
        print(f"1. Update README.md to use GIF files instead of MP4")
        print(f"2. Consider removing original MP4 files to save space")
        print(f"3. Add *.mp4 to .gitignore if you don't want to track large video files")
    
    return successful_conversions > 0

if __name__ == "__main__":
    main() 