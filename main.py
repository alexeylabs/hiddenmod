from model.hidden import Hidden
from config import config
from model.noiser import Noiser
from train import train
from utils import get_data_loaders
from validate import validate


def main():
    hidden = Hidden(config)
    train_loader, val_loader = get_data_loaders(config)

    train(hidden, config, train_loader, val_loader)

    for noise in config.noiser.keys():
        noiser_config = {
            noise: config.noiser[noise],
        }
        hidden.noiser = Noiser(noiser_config)
        validate(hidden, val_loader, config['message_length'], config['device'])


main()
