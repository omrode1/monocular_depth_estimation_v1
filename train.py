import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.monodepth import MonodepthModel
from utils.dataset import MonocularDepthDataset, KITTIDataset
from utils.losses import DepthLoss
from utils.metrics import evaluate_model, visualize_depth, compute_depth_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Monocular Depth Estimation Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset_type', type=str, default='general', 
                       choices=['general', 'kitti'], help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=18, choices=[18, 34, 50], 
                       help='ResNet depth')
    parser.add_argument('--scales', type=str, default='0,1,2,3', help='Depth prediction scales')
    parser.add_argument('--use_skips', action='store_true', default=True, help='Use skip connections')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--supervised', action='store_true', help='Use supervised training')
    parser.add_argument('--supervised_weight', type=float, default=1.0, help='Supervised loss weight')
    parser.add_argument('--unsupervised_weight', type=float, default=1.0, help='Unsupervised loss weight')
    parser.add_argument('--smoothness_weight', type=float, default=0.001, help='Smoothness loss weight')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], 
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'], 
                       help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=30, help='LR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR decay factor')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=100, help='Log frequency (steps)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def create_dataloaders(args):
    """Create training and validation dataloaders."""
    
    # Parse scales
    scales = [int(s) for s in args.scales.split(',')]
    
    # Dataset class
    if args.dataset_type == 'kitti':
        dataset_class = KITTIDataset
    else:
        dataset_class = MonocularDepthDataset
    
    # Training dataset
    train_dataset = dataset_class(
        data_path=args.data_path,
        split='train',
        load_depth=args.supervised,
        min_depth=0.1,
        max_depth=100.0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Validation dataset
    val_dataset = dataset_class(
        data_path=args.data_path,
        split='val',
        load_depth=args.supervised,
        min_depth=0.1,
        max_depth=100.0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def create_model(args):
    """Create the monocular depth estimation model."""
    scales = [int(s) for s in args.scales.split(',')]
    
    model = MonodepthModel(
        num_layers=args.num_layers,
        scales=scales,
        use_skips=args.use_skips,
        pretrained=True
    )
    
    return model

def create_optimizer(model, args):
    """Create optimizer and scheduler."""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    
    return optimizer, scheduler

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['image'].to(device)
        targets = batch['depth'].to(device) if 'depth' in batch else None
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets, images)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Total': f'{total_loss/(batch_idx+1):.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % args.log_freq == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            for loss_name, loss_value in loss_dict.items():
                if loss_name != 'total':
                    writer.add_scalar(f'Train/{loss_name.capitalize()}', loss_value.item(), step)
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device, epoch, args, writer):
    """Validate the model."""
    model.eval()
    total_loss = 0
    metrics_sum = {
        'rmse': 0, 'mae': 0, 'abs_rel': 0, 'sq_rel': 0,
        'a1': 0, 'a2': 0, 'a3': 0
    }
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            targets = batch['depth'].to(device) if 'depth' in batch else None
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss_dict = criterion(outputs, targets, images)
            loss = loss_dict['total']
            total_loss += loss.item()
            
            # Compute metrics if supervised
            if args.supervised and targets is not None:
                pred_depth = outputs['disp_0']
                for i in range(images.shape[0]):
                    sample_metrics = compute_depth_metrics(pred_depth[i], targets[i])
                    for key in metrics_sum:
                        if key in sample_metrics:
                            metrics_sum[key] += sample_metrics[key]
                    num_samples += 1
    
    # Average metrics
    avg_loss = total_loss / len(val_loader)
    if num_samples > 0:
        for key in metrics_sum:
            metrics_sum[key] /= num_samples
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    for metric_name, metric_value in metrics_sum.items():
        writer.add_scalar(f'Val/{metric_name.upper()}', metric_value, epoch)
    
    return avg_loss, metrics_sum

def save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    """Save model checkpoint."""
    os.makedirs(args.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'args': args
    }
    
    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Create loss function
    criterion = DepthLoss(
        supervised_weight=args.supervised_weight,
        unsupervised_weight=args.unsupervised_weight,
        smoothness_weight=args.smoothness_weight
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, args)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, args, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        if args.supervised:
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"Val Abs Rel: {val_metrics['abs_rel']:.4f}")
            print(f"Val A1: {val_metrics['a1']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    main() 