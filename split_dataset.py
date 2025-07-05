import os
import shutil
import random

def split_dataset(train_img_dir='data/paired/train/images', 
                  train_depth_dir='data/paired/train/depth',
                  val_img_dir='data/paired/val/images',
                  val_depth_dir='data/paired/val/depth',
                  val_split=0.1):
    """Split dataset into training and validation sets."""
    
    # Get all image files
    img_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    
    # Calculate number of validation samples
    num_val = int(len(img_files) * val_split)
    print(f"Total images: {len(img_files)}")
    print(f"Validation samples: {num_val}")
    print(f"Training samples: {len(img_files) - num_val}")
    
    # Randomly select validation samples
    val_files = random.sample(img_files, num_val)
    
    # Move files to validation
    for img_file in val_files:
        # Move image
        src_img = os.path.join(train_img_dir, img_file)
        dst_img = os.path.join(val_img_dir, img_file)
        shutil.move(src_img, dst_img)
        
        # Move corresponding depth file
        depth_file = img_file.replace('.jpg', '.npy')
        src_depth = os.path.join(train_depth_dir, depth_file)
        dst_depth = os.path.join(val_depth_dir, depth_file)
        if os.path.exists(src_depth):
            shutil.move(src_depth, dst_depth)
    
    print("Dataset split completed!")

if __name__ == '__main__':
    split_dataset() 