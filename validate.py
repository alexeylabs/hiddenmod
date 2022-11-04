import torch
import numpy as np

from model.hidden import Hidden


def validate(hidden: Hidden, val_loader, msg_length, device):
    ber_history = []
    with torch.no_grad():
        for images in val_loader:

            hidden.eval()

            messages = np.random.randint(2, size=(images.shape[0], msg_length))
            messages = torch.Tensor(messages).to(device)
            images = images.to(device)

            encoded_images, decoded_messages = hidden(images, messages)

            decoded_messages = torch.clip(decoded_messages.detach().round(), 0., 1.).to(device)
            ber = torch.abs(messages - decoded_messages).sum() / (images.shape[0] * msg_length)
            ber_history.append(ber.item())

    return sum(ber_history) / len(ber_history)
