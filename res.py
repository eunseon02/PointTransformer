
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Convolutional layer (1x1 Conv3D to reduce channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        # Save the original input (residual connection)
        residual = x

        # Pass through convolutional layers
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        residual = self.residual_conv(residual)

        # Add the original input (residual) to the output
        out = out + residual

        return out