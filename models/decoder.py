import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthDecoder(nn.Module):
    """
    Depth decoder for monocular depth estimation.
    Uses skip connections and produces depth maps at multiple scales.
    """
    
    def __init__(self, num_ch_enc=[64, 64, 128, 256, 512], scales=range(4), use_skips=True):
        super(DepthDecoder, self).__init__()
        
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.use_skips = use_skips
        
        # Number of channels for each decoder stage
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        # Decoder layers
        self.convs = nn.ModuleDict()
        
        for i in range(4, -1, -1):
            # Upconv
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}"] = nn.Sequential(
                nn.ConvTranspose2d(num_ch_in, num_ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(num_ch_out),
                nn.ReLU(inplace=True)
            )
            
            # Conv block - handle skip connections properly
            if self.use_skips and i > 0:
                # With skip connection
                num_ch_in = num_ch_out + self.num_ch_enc[i - 1]
                self.convs[f"conv_{i}"] = nn.Sequential(
                    ConvBlock(num_ch_in, num_ch_out),
                    ConvBlock(num_ch_out, num_ch_out)
                )
            else:
                # Without skip connection
                self.convs[f"conv_{i}"] = nn.Sequential(
                    ConvBlock(num_ch_out, num_ch_out),
                    ConvBlock(num_ch_out, num_ch_out)
                )
        
        # Depth prediction heads
        for s in self.scales:
            self.convs[f"dispconv_{s}"] = nn.Conv2d(self.num_ch_dec[s], 1, kernel_size=3, padding=1)
    
    def forward(self, input_features):
        """
        Forward pass through the decoder.
        Args:
            input_features: List of encoder features [feat1, feat2, feat3, feat4, feat5]
        Returns:
            Dictionary of depth predictions at different scales
        """
        outputs = {}
        
        # Start from the deepest features
        x = input_features[-1]
        
        for i in range(4, -1, -1):
            # Upsample
            x = self.convs[f"upconv_{i}"](x)
            
            # Skip connection
            if self.use_skips and i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)
            
            # Convolution
            x = self.convs[f"conv_{i}"](x)
            
            # Predict depth at this scale
            if i in self.scales:
                outputs[f"disp_{i}"] = torch.sigmoid(self.convs[f"dispconv_{i}"](x))
        
        return outputs 