import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, im_size, im_channels,
                 conv_channels, kernels, strides, paddings):
        super().__init__()
        self.img_size = im_size
        self.im_channels = im_channels
        activation = nn.LeakyReLU()
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity(),
                nn.Dropout2d(0.2) if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.view(out.size(0), -1).squeeze(1)


