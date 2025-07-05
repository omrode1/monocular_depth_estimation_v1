import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.monodepth import MonodepthModel
from utils.dataset import MonocularDepthDataset, KITTIDataset
from utils.losses import DepthLoss
from utils.metrics import evaluate_model, visualize_depth, compute_depth_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Improved Monocular Depth Estimation Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset_type', type=str, default='general', 
                       choices=['general', 'kitti'], help='Dataset type')
    parser.add_argument('--supervised', action='store_true', help='Use supervised training')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=50, 
                       choices=[18, 34, 50], help='ResNet encoder depth')
    parser.add_argument('--use_skips', action='store_true', default=True, 
                       help='Use skip connections')
    parser.add_argument('--scales', nargs='+', type=int, default=[0, 1, 2, 3], 
                       help='Output scales')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'step', 'cosine'], help='LR scheduler')
    parser.add_argument('--patience', type=int, default=10, help='LR scheduler patience')
    parser.add_argument('--step_size', type=int, default=30, help='Step LR step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor')
    
    # Loss arguments
    parser.add_argument('--lambda_photo', type=float, default=1.0, help='Photometric loss weight')
    parser.add_argument('--lambda_smooth', type=float, default=1e-3, help='Smoothness loss weight')
    parser.add_argument('--lambda_edge', type=float, default=1e-2, help='Edge-aware loss weight')
    
    # Augmentation arguments
    parser.add_argument('--augment', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[640, 192], help='Crop size')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=5, help='Validation frequency')
    
    return parser.parse_args()

def get_transforms(args):
    """Get data augmentation transforms."""
    if not args.augment:
        return None, None
    
    # Training transforms - ensure consistent size
    train_transform = A.Compose([
        A.Resize(height=args.crop_size[1], width=args.crop_size[0]),  # Always resize to same size
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation transforms
    val_transform = A.Compose([
        A.Resize(height=args.crop_size[1], width=args.crop_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def setup_device():
    """Setup device (GPU/CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def create_model(args, device):
    """Create and setup model."""
    model = MonodepthModel(
        num_layers=args.num_layers,
        scales=args.scales,
        use_skips=args.use_skips,
        pretrained=True  # Use pretrained weights for better initialization
    )
    
    # Load previous checkpoint if exists
    checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def create_optimizer_and_scheduler(model, args):
    """Create optimizer and learning rate scheduler."""
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=args.gamma, 
            patience=args.patience
        )
    elif args.scheduler == 'step':
        scheduler = StepLR(
            optimizer, step_size=args.step_size, 
            gamma=args.gamma, verbose=True
        )
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    
    return optimizer, scheduler

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device) if args.supervised else None
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Debug: print shapes
        if batch_idx == 0:
            print(f"Images shape: {images.shape}")
            print(f"Outputs keys: {outputs.keys()}")
            for key, value in outputs.items():
                print(f"  {key} shape: {value.shape}")
            if args.supervised:
                print(f"Depths shape: {depths.shape}")
                print(f"Depths range: {depths.min():.4f} to {depths.max():.4f}")
        
        # Compute loss
        try:
            if args.supervised:
                losses = criterion(outputs, depths)
                loss = losses['total']  # Extract total loss from dictionary
            else:
                losses = criterion(outputs, input_image=images)
                loss = losses['total']  # Extract total loss from dictionary
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss}")
                continue
                
        except Exception as e:
            print(f"Error computing loss: {e}")
            continue
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device, args):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_abs_rel = 0
    total_a1 = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device) if args.supervised else None
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            if args.supervised:
                losses = criterion(outputs, depths)
                loss = losses['total']  # Extract total loss from dictionary
                total_loss += loss.item()
                
                # Compute metrics
                pred_depth = outputs['disp_0']
                metrics = compute_depth_metrics(pred_depth, depths)
                total_rmse += metrics['rmse']
                total_abs_rel += metrics['abs_rel']
                total_a1 += metrics['a1']
    
    avg_loss = total_loss / num_batches
    avg_rmse = total_rmse / num_batches
    avg_abs_rel = total_abs_rel / num_batches
    avg_a1 = total_a1 / num_batches
    
    return avg_loss, avg_rmse, avg_abs_rel, avg_a1

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(args.log_dir)
    
    # Get transforms
    train_transform, val_transform = get_transforms(args)
    
    # Create datasets
    if args.dataset_type == 'kitti':
        train_dataset = KITTIDataset(
            args.data_path, 'train', transform=train_transform, 
            load_depth=args.supervised
        )
        val_dataset = KITTIDataset(
            args.data_path, 'val', transform=val_transform, 
            load_depth=args.supervised
        )
    else:
        train_dataset = MonocularDepthDataset(
            args.data_path, 'train', transform=train_transform, 
            load_depth=args.supervised
        )
        val_dataset = MonocularDepthDataset(
            args.data_path, 'val', transform=val_transform, 
            load_depth=args.supervised
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(args, device)
    
    # Create loss function
    criterion = DepthLoss(
        supervised_weight=1.0 if args.supervised else 0.0,
        unsupervised_weight=0.0,  # set to >0 if using unsupervised
        smoothness_weight=args.lambda_smooth,
        edge_aware_weight=args.lambda_edge
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    
    # Training loop
    best_val_loss = float('inf')
    best_a1 = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Model: ResNet-{args.num_layers}, Scales: {args.scales}")
    print(f"Augmentation: {args.augment}, Scheduler: {args.scheduler}")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_rmse, val_abs_rel, val_a1 = validate(model, val_loader, criterion, device, args)
            
            # Log metrics
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metrics/RMSE', val_rmse, epoch)
            writer.add_scalar('Metrics/AbsRel', val_abs_rel, epoch)
            writer.add_scalar('Metrics/A1', val_a1, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch+1}/{args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val RMSE: {val_rmse:.4f}')
            print(f'  Val Abs Rel: {val_abs_rel:.4f}')
            print(f'  Val A1: {val_a1:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                print(f'  New best model saved! (Loss: {val_loss:.4f})')
            
            if val_a1 > best_a1:
                best_a1 = val_a1
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_a1_model.pth'))
                print(f'  New best A1 model saved! (A1: {val_a1:.4f})')
            
            # Update scheduler
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        else:
            # Log only training loss
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            print(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'args': args
            }, checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path}')
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    main() 