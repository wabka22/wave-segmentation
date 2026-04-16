from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from ecg_signal_processor import load_sample


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """
    Исходные метки:
    -1 -> background
     0 -> QRS
     1 -> SPIKES
     2 -> P
     3 -> noise

    Новые метки:
    0 -> background
    1 -> QRS
    2 -> SPIKES
    """
    new_labels = np.zeros_like(labels, dtype=np.int64)

    new_labels[labels == 0] = 1
    new_labels[labels == 1] = 2

    return new_labels


class ECGDataset(Dataset):
    def __init__(
        self,
        signal_dir: str | Path,
        markup_dir: str | Path,
        file_ids: list[str],
        label_channel: int = 0,
        background_value: int = -1,
        window: int | None = None,
        step: int | None = None,
    ):
        self.signal_dir = Path(signal_dir)
        self.markup_dir = Path(markup_dir)
        self.file_ids = file_ids
        self.label_channel = label_channel
        self.background_value = background_value
        self.window = window if window is not None else config.WINDOW
        self.step = step if step is not None else config.STEP

        self.X = []
        self.Y = []

        for file_id in self.file_ids:
            signal_path = self.signal_dir / f"{file_id}.npy"
            markup_path = self.markup_dir / f"{file_id}.json"

            if not signal_path.exists() or not markup_path.exists():
                continue

            signal, labels = load_sample(
                signal_path=signal_path,
                markup_path=markup_path,
                background_value=self.background_value,
            )

            if signal.ndim != 2 or labels.ndim != 2:
                continue

            if self.label_channel >= labels.shape[0]:
                raise ValueError(
                    f"label_channel={self.label_channel} вне диапазона. "
                    f"Число каналов разметки: {labels.shape[0]}"
                )

            labels = remap_labels(labels)
            target = labels[self.label_channel]
            target = self.expand_segments(target, cls=2, radius=4)
            xs, ys = self._split_windows(signal, target)
            self.X.extend(xs)
            self.Y.extend(ys)

        if len(self.X) == 0:
            raise ValueError("Dataset is empty. Проверь пути, разметку и параметры окна.")

        self.X = torch.stack(self.X).float()   # [N, C, W]
        self.Y = torch.stack(self.Y).long()    # [N, W]

    def _split_windows(self, signal: np.ndarray, target: np.ndarray):
        xs = []
        ys = []

        length = signal.shape[1]

        if length < self.window:
            return xs, ys

        for start in range(0, length - self.window + 1, self.step):
            end = start + self.window

            x_win = signal[:, start:end]
            y_win = target[start:end]

            xs.append(torch.from_numpy(x_win.copy()))
            ys.append(torch.from_numpy(y_win.copy()))

        return xs, ys
    
    def expand_segments(self,mask: np.ndarray, cls: int, radius: int) -> np.ndarray:
        mask = mask.copy()
        idx = np.where(mask == cls)[0]

        if len(idx) == 0:
            return mask

        start = idx[0]
        prev = idx[0]

        segments = []

        for i in idx[1:]:
            if i != prev + 1:
                segments.append((start, prev + 1))
                start = i
            prev = i

        segments.append((start, prev + 1))

        for s, e in segments:
            new_s = max(0, s - radius)
            new_e = min(len(mask), e + radius)
            mask[new_s:new_e] = np.where(mask[new_s:new_e] == 0, cls, mask[new_s:new_e])

        return mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]