from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .core import load_sample

LABEL_MAP = {
    0: 1,  # QRS
    1: 2,  # SPIKES
    4: 3,  # QRS_AFTER_SPIKE TODO: замени 4 на реальный Type
}

def remap_labels(labels: np.ndarray) -> np.ndarray:
    new_labels = np.zeros_like(labels, dtype=np.int64)

    for old_type, new_type in LABEL_MAP.items():
        new_labels[labels == old_type] = new_type

    return new_labels

class ECGDataset(Dataset):
    def __init__(
        self,
        signal_dir: str | Path,
        markup_dir: str | Path,
        background_value: int = -1,
    ):
        self.signal_dir = Path(signal_dir)
        self.markup_dir = Path(markup_dir)
        self.background_value = background_value

        self.samples = []

        for signal_path in sorted(self.signal_dir.glob("*.npy")):
            file_id = signal_path.stem
            markup_path = self.markup_dir / f"{file_id}.json"

            if markup_path.exists():
                self.samples.append((signal_path, markup_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal_path, markup_path = self.samples[idx]

        signal, labels = load_sample(
            signal_path=signal_path,
            markup_path=markup_path,
            background_value=self.background_value,
        )

        labels = remap_labels(labels)

        return signal.astype(np.float32), labels.astype(np.int64)