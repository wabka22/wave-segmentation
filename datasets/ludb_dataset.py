import os
import wfdb
import torch
import numpy as np

from torch.utils.data import Dataset
from utils.preprocessing import create_mask, split_windows
from utils.preprocessing import bandpass_filter, bandpass_filter_gpu


class LUDBDataset(Dataset):
    
    def __init__(self, path, records):

        self.X = [] # окна сигналов
        self.Y = [] # маски окон 

        for rec in records: 
            record = wfdb.rdrecord(os.path.join(path, rec))
            # Аннотация - разметка
            ann = wfdb.rdann(os.path.join(path, rec), "ii") 

            signal = record.p_signal.T

            signal = torch.from_numpy(signal).float()   # ← перевод в torch

            signal = (signal - signal.mean()) / signal.std()

            signal = bandpass_filter_gpu(signal)        # ← сразу весь батч каналов

            signal = signal.numpy()                     # ← обратно в numpy (если нужно)
            mask = create_mask(signal.shape[1], ann)

            xs, ys = split_windows(signal, mask)

            self.X.extend(xs)
            self.Y.extend(ys)

        self.X = torch.from_numpy(np.array(self.X)).float()
        self.Y = torch.from_numpy(np.array(self.Y)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
