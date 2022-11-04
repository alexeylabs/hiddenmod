import torch
import torch.nn as nn
from model.block import Block


class Decoder(nn.Module):
    def __init__(self, num_blocks, num_channels, message_length):
        super().__init__()
        self.conv_layers = nn.Sequential()
        self.conv_layers.append(Block(3, num_channels))
        for _ in range(num_blocks - 1):
            self.conv_layers.append(Block(num_channels, num_channels))

        self.final_step = nn.Sequential()
        self.final_step.append(Block(num_channels, message_length))
        self.final_step.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.final_step.append(nn.Flatten())
        self.final_step.append(nn.Linear(message_length, message_length))

    def forward(self, encoded_image):
        encoded_image = self.conv_layers(encoded_image)
        message = self.final_step(encoded_image)

        return message
