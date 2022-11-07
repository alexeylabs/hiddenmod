import torch

config = {'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
          'experiment_name': 'base',
          'message_length': 30,
          'image_size': 128,
          'encoder': {
              'num_blocks': 7,
              'num_channels': 64
          },
          'decoder': {
              'num_blocks': 4,
              'num_channels': 64
          },
          'discriminator': {
              'num_blocks': 3,
              'num_channels': 64
          },
          'noiser': {
              'identity': None,
              'flip': None,
              'rotate': (-10, 10),
              'center_crop': (0.5, 0.9),
          },
          'train': {
              'epochs': 10,
              'batch_size': 32,
              'train_images': '/Users/alexey/data/coco_128/train',
              'val_images': '/Users/alexey/data/coco_128/val',
          }
          }