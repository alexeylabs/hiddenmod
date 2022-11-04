import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

from model.block import Block
from model.encoder import Encoder
from model.decoder import Decoder
from model.noiser import Noiser
from model.discriminator import Discriminator
from config import config

BATCH_SIZE = 64
MESSAGE_LENGTH = 30

batch = torch.rand(BATCH_SIZE, 3, 128, 128)
message = np.random.randint(2, size=(BATCH_SIZE, MESSAGE_LENGTH))
message = torch.Tensor(message)
print(batch.shape, message.shape)

block = Block(3, 3)
print('block', batch.shape, block(batch).shape)

# enc = Encoder(num_blocks=4,
#               num_channels=64,
#               message_length=30)
# print('enc', batch.shape, enc(batch, message).shape)

# enc = Decoder(num_blocks=4,
#               num_channels=64,
#               message_length=30)
# print('dec', batch.shape, enc(batch).shape)

# noiser = Noiser(config['noiser'])
#
# image = Image.open('./debug/test.jpg')
# x = transforms.ToTensor()(image)
# x = torch.unsqueeze(x, 0)
# print('x', x.shape)
#
# image_2 = noiser(x)
# out = transforms.ToPILImage()(image_2[0])
# out.show()

# dis = Discriminator(num_blocks=3,
#                     num_channels=64)
# print('dis', batch.shape, dis(batch).shape)


