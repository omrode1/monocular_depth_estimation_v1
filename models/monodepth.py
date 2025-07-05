import torch
import torch.nn as nn
from models.encoder import ResNetEncoder
from models.decoder import DepthDecoder

class MonodepthModel(nn.Module):
    """
    Complete monocular depth estimation model.
    Combines ResNet encoder with depth decoder for multi-scale depth prediction.
    """
    
    def __init__(self, num_layers=18, scales=range(4), use_skips=True, pretrained=True):
        super(MonodepthModel, self).__init__()
        
        self.encoder = ResNetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=scales,
            use_skips=use_skips
        )
        
        self.scales = scales

    def forward(self, x):
        """
        Forward pass through the complete model.
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W)
        Returns:
            Dictionary of depth predictions at different scales
        """
        features = self.encoder(x)
        depth_outputs = self.decoder(features)
        return depth_outputs
    
    def get_depth(self, x, scale=0):
        """
        Get depth prediction at a specific scale.
        Args:
            x: Input RGB image tensor
            scale: Scale index (0 is highest resolution)
        Returns:
            Depth map at specified scale
        """
        outputs = self.forward(x)
        return outputs[f"disp_{scale}"]
