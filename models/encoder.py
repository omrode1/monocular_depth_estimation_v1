import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    """
    ResNet encoder for monocular depth estimation.
    Uses ResNet-18/34/50 as backbone and extracts multi-scale features.
    """
    
    def __init__(self, num_layers=18, pretrained=True):
        super(ResNetEncoder, self).__init__()
        
        if num_layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif num_layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif num_layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet depth: {num_layers}")
        
        # Remove the final classification layers
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # 64 channels
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4   # 512 channels
        ])
        
        # Store the number of channels for each layer
        if num_layers == 18 or num_layers == 34:
            self.num_ch_enc = [64, 64, 128, 256, 512]
        else:  # ResNet-50
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            features.append(x)
            
            # Apply max pooling after the first layer
            if i == 0:
                x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        return features 