from model.hidden import Hidden
from config import config
from train import train
from utils import get_data_loaders


def main():
    hidden = Hidden(config)
    train_loader, val_loader = get_data_loaders(config)

    train(hidden, config, train_loader, val_loader)


main()
