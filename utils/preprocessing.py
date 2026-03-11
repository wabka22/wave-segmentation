import numpy as np
from config import WINDOW, STEP
from scipy.signal import butter, filtfilt

def bandpass_filter(signal):

    fs = 500

    low = 0.5 / (fs/2)
    high = 40 / (fs/2)

    b, a = butter(4, [low, high], btype='band')

    return filtfilt(b, a, signal)

def create_mask(length, ann):

    mask = np.zeros(length)

    current_class = None
    start = None

    for sym, sample in zip(ann.symbol, ann.sample):

        if sym == '(':
            start = sample

        elif sym in ['p','N','t']:
            if sym == 'p':
                current_class = 1
            elif sym == 'N':
                current_class = 2
            elif sym == 't':
                current_class = 3

        elif sym == ')':
            end = sample
            if start is not None and current_class is not None:
                mask[start:end] = current_class

    return mask


def split_windows(signal, mask, augment=True):
    X = []
    Y = []

    for i in range(0, len(signal) - WINDOW, STEP):
        win = signal[i:i+WINDOW].copy()
        label = mask[i:i+WINDOW]

        win = (win - win.mean()) / (win.std() + 1e-8)

        if augment:
            win = win * np.random.uniform(0.95, 1.05)
            win = win + np.random.normal(0, 0.01, size=win.shape)

        X.append(win)
        Y.append(label)

    return X, Y

