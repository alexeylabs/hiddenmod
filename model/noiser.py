import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from model.jpeg import JpegCompression


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        noised_image = image
        return noised_image


class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        noised_image = F.hflip(image)
        return noised_image


class Rotate(nn.Module):
    def __init__(self, rotation_range):
        super().__init__()
        self.angle_min = rotation_range[0]
        self.angle_max = rotation_range[1]

    def forward(self, image):
        angle = random.randrange(self.angle_min, self.angle_max)
        noised_image = F.rotate(image, angle)
        return noised_image


class CenterCrop(nn.Module):
    def __init__(self, crop_range):
        super().__init__()
        self.crop_min = crop_range[0]
        self.crop_max = crop_range[1]

    def forward(self, image):
        crop_size = random.randrange(int(self.crop_min*image.shape[-1]),
                                     int(self.crop_max*image.shape[-1]))
        noised_image = F.center_crop(image, crop_size)
        return noised_image


class Noiser(nn.Module):
    def __init__(self, noiser_config):
        super().__init__()
        self.noises = []
        if 'identity' in noiser_config.keys():
            self.noises.append(Identity())
        if 'flip' in noiser_config.keys():
            self.noises.append(Flip())
        if 'rotate' in noiser_config.keys():
            self.noises.append(Rotate(noiser_config['rotate']))
        if 'center_crop' in noiser_config.keys():
            self.noises.append(CenterCrop(noiser_config['center_crop']))
        if 'jpeg' in noiser_config.keys():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.noises.append(JpegCompression(device))
        print('Using noises:', self.noises)

    def forward(self, image):
        noised_image = random.choice(self.noises)(image)
        return noised_image
