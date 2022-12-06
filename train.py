import torch
import numpy as np
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import time
from datetime import datetime
from tensorflow import summary

from model.discriminator import Discriminator
from model.hidden import Hidden
from validate import validate
from utils import save_examples, save_model


def train(hidden: Hidden, config, train_loader, val_loader):

    if config['use_tb']:
        current_time = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        train_log_dir = 'logs/tensorboard/train/' + current_time
        test_log_dir = 'logs/tensorboard/test/' + current_time
        train_summary_writer = summary.create_file_writer(train_log_dir)
        test_summary_writer = summary.create_file_writer(test_log_dir)

    discriminator = Discriminator(num_blocks=config['discriminator']['num_blocks'],
                                  num_channels=config['discriminator']['num_channels'],
                                  use_bn=config['use_bn']).to(config['device'])

    num_epochs = config['train']['epochs']
    msg_length = config['message_length']
    device = config['device']

    image_distortion_criterion = nn.MSELoss().to(device)
    message_distortion_criterion = nn.MSELoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

    adversarial_loss_weight = config['train']['adversarial_loss_weight']
    distortion_loss_weight = config['train']['distortion_loss_weight']

    hidden_optimizer = torch.optim.Adam(hidden.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())

    for epoch in tqdm(range(1, num_epochs+1)):

        image_distortion_history = []
        message_distortion_history = []
        adversarial_history = []
        ber_history = []

        for images in train_loader:
            hidden.train()
            discriminator.train()

            images = images.to(device)
            messages = np.random.randint(2, size=(images.shape[0], msg_length))
            messages = torch.Tensor(messages).to(device)

            encoded_images, decoded_messages = hidden(images, messages)

            # train discriminator
            discriminator_optimizer.zero_grad()

            discriminator_prediction = discriminator(torch.cat((images.detach(),
                                                                encoded_images.detach()), 0))
            true_prediction = torch.cat((torch.full((images.shape[0], 1), 1.),
                                         torch.full((images.shape[0], 1), 0.)), 0).to(device)
            discriminator_loss = adversarial_criterion(discriminator_prediction, true_prediction)

            discriminator_loss.backward()
            discriminator_optimizer.step()

            # train encoder-decoder
            hidden_optimizer.zero_grad()

            messages_loss = message_distortion_criterion(messages, decoded_messages)
            images_loss = image_distortion_criterion(images, encoded_images)

            discriminator_prediction = discriminator(encoded_images)
            adversarial_loss_enc = adversarial_criterion(discriminator_prediction,
                                                         torch.full((images.shape[0], 1), 1.).to(device))

            hidden_loss = (distortion_loss_weight * images_loss +
                           adversarial_loss_weight * adversarial_loss_enc +
                           messages_loss)
            hidden_loss.backward()
            hidden_optimizer.step()

            decoded_messages = torch.clip(decoded_messages.detach().round(), 0., 1.).to(device)
            ber = torch.abs(messages - decoded_messages).sum() / (images.shape[0] * msg_length)

            image_distortion_history.append(images_loss.item())
            message_distortion_history.append(messages_loss.item())
            adversarial_history.append(discriminator_loss.item())
            ber_history.append(ber.item())

        print()
        img_loss = sum(image_distortion_history) / len(image_distortion_history)
        msg_loss = sum(message_distortion_history) / len(message_distortion_history)
        adv_loss = sum(adversarial_history) / len(adversarial_history)
        train_ber = sum(ber_history) / len(ber_history)

        val_ber, val_psnr = validate(hidden, val_loader, msg_length, device)

        print('image_distortion:  ', img_loss)
        print('message_distortion:', msg_loss)
        print('adversarial:       ', adv_loss)
        print('train_ber:         ', train_ber)

        print('validation_ber:    ', val_ber)
        print('validation_psnr:   ', val_psnr)
        print()
        time.sleep(1)

        if config['use_tb']:
            with train_summary_writer.as_default():
                summary.scalar('img_loss', img_loss, step=epoch)
                summary.scalar('msg_loss', msg_loss, step=epoch)
                summary.scalar('adv_loss', adv_loss, step=epoch)
                summary.scalar('train_ber', train_ber, step=epoch)
            with test_summary_writer.as_default():
                summary.scalar('val_ber', val_ber, step=epoch)
                summary.scalar('val_psnr', val_psnr, step=epoch)

        save_examples(images, str(epoch) + '_original.jpg')
        save_examples(encoded_images, str(epoch) + '_encoded.jpg')
        save_model(hidden, config['experiment_name'] + str(epoch)+'.pth')
