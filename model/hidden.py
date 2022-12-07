import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.noiser import Noiser
from model.discriminator import Discriminator


class Hidden(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(num_blocks=config['encoder']['num_blocks'],
                               num_channels=config['encoder']['num_channels'],
                               message_length=config['message_length'],
                               use_bn=config['encoder']['use_bn']).to(config['device'])

        self.decoder = Decoder(num_blocks=config['decoder']['num_blocks'],
                               num_channels=config['decoder']['num_channels'],
                               message_length=config['message_length'],
                               use_bn=config['decoder']['use_bn']).to(config['device'])

        self.noiser = Noiser(config['noiser']).to(config['device'])

    def forward(self, images, messages):
        encoded_images = self.encoder(images, messages)
        noised_images = self.noiser(encoded_images)
        decoded_messages = self.decoder(noised_images)
        return encoded_images, decoded_messages
