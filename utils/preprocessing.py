import torch
import numpy as np
from config import WINDOW, STEP


def create_mask(length, ann):

    mask = np.zeros(length)

    current_class = None
    start = None

    for sym, sample in zip(ann.symbol, ann.sample):
        if sym == "(":
            start = sample

        elif sym in ["p", "N", "t"]:
            if sym == "p":
                current_class = 1
            elif sym == "N":
                current_class = 2
            elif sym == "t":
                current_class = 3

        elif sym == ")":
            end = sample
            if start is not None and current_class is not None:
                mask[start:end] = current_class

    return mask

def split_windows(signal, mask, augment=True, device="cpu"):
    if not isinstance(signal, torch.Tensor):
        signal = torch.from_numpy(signal).float()
    signal = signal.to(device)

    signal = (signal - signal.mean(dim=1, keepdim=True)) / (
        signal.std(dim=1, keepdim=True) + 1e-8
    )

    X, Y = [], []
    length = signal.shape[1]

    for i in range(0, length - WINDOW + 1, STEP):
        win = signal[:, i : i + WINDOW].clone()
        label = mask[i : i + WINDOW].copy()

        if augment:
            win = win * torch.empty_like(win).uniform_(0.975, 1.025)
            win = win + torch.randn_like(win) * 0.005

        X.append(win.cpu())
        Y.append(torch.from_numpy(label).long())

    return X, Y

# Сигнал: |-------|-------|-------|-------|-------|
#         0      500     1000    1500    2000    2500

# Окно 1: [=======]
#          0-999

# Окно 2:     [=======]
#            500-1499

# Окно 3:         [=======]
#               1000-1999
