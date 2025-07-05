import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DepthLoss(nn.Module):
    """
    Combined loss function for monocular depth estimation.
    Supports both supervised and unsupervised training.
    """
    
    def __init__(self, supervised_weight=1.0, unsupervised_weight=1.0,
                 smoothness_weight=0.001, edge_aware_weight=0.1):
        super(DepthLoss, self).__init__()
        
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight
        self.smoothness_weight = smoothness_weight
        self.edge_aware_weight = edge_aware_weight
    
    def forward(self, pred_depths, target_depth=None, input_image=None):
        """
        Compute the total loss.
        Args:
            pred_depths: Dictionary of predicted depths at different scales
            target_depth: Ground truth depth (for supervised training)
            input_image: Input RGB image (for unsupervised training)
        Returns:
            Dictionary of losses
        """
        total_loss = 0
        losses = {}
        
        # Supervised loss if ground truth is available
        if target_depth is not None and self.supervised_weight > 0:
            supervised_loss = self.compute_supervised_loss(pred_depths, target_depth)
            losses['supervised'] = supervised_loss
            total_loss += self.supervised_weight * supervised_loss
        
        # Unsupervised loss using photometric consistency
        if input_image is not None and self.unsupervised_weight > 0:
            unsupervised_loss = self.compute_unsupervised_loss(pred_depths, input_image)
            losses['unsupervised'] = unsupervised_loss
            total_loss += self.unsupervised_weight * unsupervised_loss
        
        # Smoothness loss
        if input_image is not None and self.smoothness_weight > 0:
            smoothness_loss = self.compute_smoothness_loss(pred_depths, input_image)
            losses['smoothness'] = smoothness_loss
            total_loss += self.smoothness_weight * smoothness_loss
        
        losses['total'] = total_loss
        return losses
    
    def compute_supervised_loss(self, pred_depths, target_depth):
        """Compute supervised loss using ground truth depth."""
        loss = 0
        count = 0
        
        for scale, pred_depth in pred_depths.items():
            # Resize target to match prediction scale
            target_resized = F.interpolate(target_depth, size=pred_depth.shape[2:], 
                                         mode='bilinear', align_corners=False)
            
            # BerHu loss (robust to outliers)
            diff = torch.abs(pred_depth - target_resized)
            c = 0.2 * torch.max(diff)
            
            berhu_loss = torch.where(diff <= c, diff, 
                                   (diff**2 + c**2) / (2 * c))
            
            loss += torch.mean(berhu_loss)
            count += 1
        
        return loss / count if count > 0 else 0
    
    def compute_unsupervised_loss(self, pred_depths, input_image):
        """Compute unsupervised loss using photometric consistency."""
        # This is a simplified version - in practice, you'd need stereo pairs or video sequences
        # For now, we'll use a consistency loss between different scales
        loss = 0
        count = 0
        
        scales = list(pred_depths.keys())
        for i in range(len(scales) - 1):
            scale1, scale2 = scales[i], scales[i + 1]
            pred1, pred2 = pred_depths[scale1], pred_depths[scale2]
            
            # Resize pred2 to match pred1
            pred2_resized = F.interpolate(pred2, size=pred1.shape[2:], 
                                        mode='bilinear', align_corners=False)
            
            # Consistency loss
            consistency_loss = F.mse_loss(pred1, pred2_resized)
            loss += consistency_loss
            count += 1
        
        return loss / count if count > 0 else 0
    
    def compute_smoothness_loss(self, pred_depths, input_image):
        """Compute edge-aware smoothness loss."""
        loss = 0
        count = 0
        
        # Convert image to grayscale for edge detection
        if input_image.shape[1] == 3:
            gray = 0.299 * input_image[:, 0:1] + 0.587 * input_image[:, 1:2] + 0.114 * input_image[:, 2:3]
        else:
            gray = input_image
        
        for scale, pred_depth in pred_depths.items():
            # Resize gray image to match prediction scale
            gray_resized = F.interpolate(gray, size=pred_depth.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
            # Compute gradients
            grad_x = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
            grad_y = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
            
            # Edge-aware weights
            grad_x_img = torch.abs(gray_resized[:, :, :, :-1] - gray_resized[:, :, :, 1:])
            grad_y_img = torch.abs(gray_resized[:, :, :-1, :] - gray_resized[:, :, 1:, :])
            
            # Exponential weights
            weights_x = torch.exp(-torch.mean(grad_x_img, dim=1, keepdim=True))
            weights_y = torch.exp(-torch.mean(grad_y_img, dim=1, keepdim=True))
            
            smoothness_loss = torch.mean(weights_x * grad_x) + torch.mean(weights_y * grad_y)
            loss += smoothness_loss
            count += 1
        
        return loss / count if count > 0 else 0

class BerHuLoss(nn.Module):
    """
    BerHu loss for depth estimation - robust to outliers.
    """
    
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        c = self.threshold * torch.max(diff)
        
        loss = torch.where(diff <= c, diff, (diff**2 + c**2) / (2 * c))
        return torch.mean(loss)

class SSIMLoss(nn.Module):
    """
    SSIM loss for photometric consistency.
    """
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average) 