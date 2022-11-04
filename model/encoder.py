import torch
import torch.nn as nn
from model.block import Block


class Encoder(nn.Module):
    def __init__(self, num_blocks, num_channels, message_length):
        super().__init__()
        self.conv_layers = nn.Sequential()
        self.conv_layers.append(Block(3, num_channels))
        for _ in range(num_blocks - 1):
            self.conv_layers.append(Block(num_channels, num_channels))
        self.final_step = nn.Sequential()
        self.final_step.append(Block(num_channels + message_length + 3, num_channels))
        self.final_step.append(nn.Conv2d(num_channels, 3,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0))

    def forward(self, image, message):
        prepared_image = self.conv_layers(image)

        # replicate message spatially
        message = message.unsqueeze(-1).unsqueeze(-1)
        message = message.expand(-1, -1, image.shape[-2], image.shape[-1])

        concat = torch.cat([prepared_image, message, image], dim=1)
        encoded_image = self.final_step(concat)
        return encoded_image
