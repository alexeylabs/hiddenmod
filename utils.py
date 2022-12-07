from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image, make_grid
import torch


class ImageDataset(Dataset):
    def __init__(self, images_path, size):
        self.filenames = glob.glob(images_path + '/**/*.jpg')
        print('Found files:', len(self.filenames))
        self.transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = image.convert('RGB')
        image = self.transforms(image)
        return image


def get_data_loaders(config):
    train_dataset = ImageDataset(config['train']['train_images'],
                                 config['image_size'])
    train_loader = DataLoader(train_dataset,
                              batch_size=config['train']['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=config['pin_memory'])

    valid_dataset = ImageDataset(config['train']['val_images'],
                                 config['image_size'])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['train']['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'],
                              pin_memory=config['pin_memory'])
    return train_loader, valid_loader


def save_examples(images, filename):
    denormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                           std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                      transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                           std=[1., 1., 1.]),
                                      ])
    Path('results').mkdir(parents=True, exist_ok=True)
    save_image(make_grid(denormalize(images)), 'results/' + filename)


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
