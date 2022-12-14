import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )
        if use_bn:
            self.encode.append(nn.BatchNorm2d(out_channels))
        self.encode.append(nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.encode(x)
