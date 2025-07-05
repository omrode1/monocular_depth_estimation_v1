import torch
import torch.nn.functional as F
import numpy as np

def compute_depth_metrics(pred, target, min_depth=0.1, max_depth=100.0):
    """
    Compute depth estimation metrics.
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        min_depth: Minimum depth value
        max_depth: Maximum depth value
    Returns:
        Dictionary of metrics (all values as Python float)
    """
    # Convert to numpy for easier computation
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Ensure same shape
    if pred.shape != target.shape:
        if torch.is_tensor(pred):
            pred = F.interpolate(pred.unsqueeze(0), 
                               size=target.shape, mode='bilinear', align_corners=False).squeeze(0)
        else:
            pred = F.interpolate(torch.from_numpy(pred).unsqueeze(0), 
                               size=target.shape, mode='bilinear', align_corners=False).squeeze(0).numpy()
    
    # Mask valid pixels
    mask = (target > min_depth) & (target < max_depth)
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) == 0:
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0
        }
    
    # Convert to depth if needed (assuming disparity input)
    if pred.max() <= 1.0:
        pred = 1.0 / (pred + 1e-8)
    if target.max() <= 1.0:
        target = 1.0 / (target + 1e-8)
    
    # Compute metrics
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))
    
    # Relative errors
    abs_rel = float(np.mean(np.abs(pred - target) / target))
    sq_rel = float(np.mean(((pred - target) ** 2) / target))
    
    # Accuracy metrics
    thresh = np.maximum((target / pred), (pred / target))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())
    
    return {
        'rmse': rmse,
        'mae': mae,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }

def compute_scale_invariant_error(pred, target, min_depth=0.1, max_depth=100.0):
    """
    Compute scale-invariant error (SILog).
    Returns a Python float.
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Mask valid pixels
    mask = (target > min_depth) & (target < max_depth)
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) == 0:
        return float('inf')
    
    # Convert to depth if needed
    if pred.max() <= 1.0:
        pred = 1.0 / (pred + 1e-8)
    if target.max() <= 1.0:
        target = 1.0 / (target + 1e-8)
    
    # Compute log difference
    log_diff = np.log(pred) - np.log(target)
    
    # Scale-invariant error
    silog = float(np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2))
    
    return silog

def compute_depth_consistency(pred_depths, scales=None):
    """
    Compute consistency between depth predictions at different scales.
    """
    if scales is None:
        scales = list(pred_depths.keys())
    
    consistency_loss = 0
    count = 0
    
    for i in range(len(scales) - 1):
        scale1, scale2 = scales[i], scales[i + 1]
        pred1, pred2 = pred_depths[scale1], pred_depths[scale2]
        
        # Resize pred2 to match pred1
        if torch.is_tensor(pred2):
            pred2_resized = F.interpolate(pred2, size=pred1.shape[2:], 
                                        mode='bilinear', align_corners=False)
        else:
            pred2_resized = F.interpolate(torch.from_numpy(pred2), size=pred1.shape[2:], 
                                        mode='bilinear', align_corners=False).numpy()
        
        # Compute consistency
        if torch.is_tensor(pred1):
            consistency = torch.mean((pred1 - pred2_resized) ** 2)
        else:
            consistency = np.mean((pred1 - pred2_resized) ** 2)
        
        consistency_loss += consistency
        count += 1
    
    return consistency_loss / count if count > 0 else 0

def evaluate_model(model, dataloader, device, min_depth=0.1, max_depth=100.0):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    metrics_sum = {
        'rmse': 0, 'mae': 0, 'abs_rel': 0, 'sq_rel': 0,
        'a1': 0, 'a2': 0, 'a3': 0, 'silog': 0
    }
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['depth'].to(device) if 'depth' in batch else None
            
            # Forward pass
            outputs = model(images)
            pred_depth = outputs['disp_0']  # Use highest resolution prediction
            
            if targets is not None:
                # Compute metrics for each sample in batch
                for i in range(images.shape[0]):
                    sample_metrics = compute_depth_metrics(
                        pred_depth[i], targets[i], min_depth, max_depth
                    )
                    
                    for key in metrics_sum:
                        if key in sample_metrics:
                            metrics_sum[key] += sample_metrics[key]
                    
                    num_samples += 1
    
    # Average metrics
    if num_samples > 0:
        for key in metrics_sum:
            metrics_sum[key] /= num_samples
    
    return metrics_sum

def visualize_depth(depth_map, colormap='plasma', normalize=True):
    """
    Visualize depth map with colormap.
    """
    import matplotlib.pyplot as plt
    
    if torch.is_tensor(depth_map):
        depth_map = depth_map.detach().cpu().numpy()
    
    if depth_map.ndim == 4:
        depth_map = depth_map.squeeze()
    elif depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    
    if normalize:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf() 