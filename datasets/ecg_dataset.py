from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import config
from ecg_signal_processor.core import load_sample, load_signal


# Единая схема модели:
# 0 -> background
# 1 -> обычный QRS
# 2 -> SPIKES
# 3 -> QRS_AFTER_SPIKE

LABEL_MAP_JSON = {
    0: 1,  # обычный QRS
    1: 2,  # SPIKES
    4: 3,  # QRS_AFTER_SPIKE
}

LABEL_MAP_MASK = {
    2: 1,  # обычный QRS
}


def remap_labels(labels: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    new_labels = np.zeros_like(labels, dtype=np.int64)

    for old_type, new_type in label_map.items():
        new_labels[labels == old_type] = new_type

    return new_labels


class ECGDataset(Dataset):
    def __init__(
        self,
        json_signal_dir: str | Path | None = None,
        json_markup_dir: str | Path | None = None,
        mask_datasets: list[tuple[str | Path, str | Path]] | None = None,
        background_value: int = -1,
        window: int | None = None,
        step: int | None = None,
        json_repeat: int = 3,
    ):
        self.background_value = background_value
        self.window = window if window is not None else config.WINDOW
        self.step = step if step is not None else config.STEP

        self.samples = []

        if json_signal_dir is not None and json_markup_dir is not None:
            json_signal_dir = Path(json_signal_dir)
            json_markup_dir = Path(json_markup_dir)

            for signal_path in sorted(json_signal_dir.glob("*.npy")):
                file_id = signal_path.stem
                markup_path = json_markup_dir / f"{file_id}.json"

                if markup_path.exists():
                    for _ in range(json_repeat):
                        self.samples.append(
                            {
                                "type": "json",
                                "signal_path": signal_path,
                                "label_path": markup_path,
                            }
                        )

        if mask_datasets is not None:
            for signal_dir, mask_dir in mask_datasets:
                signal_dir = Path(signal_dir)
                mask_dir = Path(mask_dir)

                for signal_path in sorted(signal_dir.glob("*.npy")):
                    file_id = signal_path.stem
                    mask_path = mask_dir / f"{file_id}.npy"

                    if mask_path.exists():
                        self.samples.append(
                            {
                                "type": "mask",
                                "signal_path": signal_path,
                                "label_path": mask_path,
                            }
                        )
        
        json_count = sum(1 for s in self.samples if s["type"] == "json")
        mask_count = sum(1 for s in self.samples if s["type"] == "mask")

        print(f"Dataset samples:")
        print(f"  json samples: {json_count}")
        print(f"  mask samples: {mask_count}")
        print(f"  total:        {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("Dataset is empty. Проверь пути к данным.")

    def __len__(self):
        return len(self.samples)

    def _load_item(self, sample: dict) -> tuple[np.ndarray, np.ndarray]:
        if sample["type"] == "json":
            signal, labels = load_sample(
                signal_path=sample["signal_path"],
                markup_path=sample["label_path"],
                background_value=self.background_value,
            )
            labels = remap_labels(labels, LABEL_MAP_JSON)

        elif sample["type"] == "mask":
            signal = load_signal(sample["signal_path"])
            labels = np.load(sample["label_path"], allow_pickle=False)
            labels = remap_labels(labels, LABEL_MAP_MASK)

        else:
            raise ValueError(f"Unknown sample type: {sample['type']}")

        return signal.astype(np.float32), labels.astype(np.int64)

    def _random_window(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if labels.ndim == 1:
            label_length = labels.shape[0]
        elif labels.ndim == 2:
            label_length = labels.shape[1]
        else:
            raise ValueError(f"Unexpected labels shape: {labels.shape}")

        signal_length = signal.shape[1]
        length = min(signal_length, label_length)

        signal = signal[:, :length]

        if labels.ndim == 1:
            labels = labels[:length]
            labels_1d = labels
        else:
            labels = labels[:, :length]
            labels_1d = labels[0]

        if length < self.window:
            pad_len = self.window - length

            signal = np.pad(
                signal,
                pad_width=((0, 0), (0, pad_len)),
                mode="constant",
                constant_values=0,
            )

            labels_1d = np.pad(
                labels_1d,
                pad_width=(0, pad_len),
                mode="constant",
                constant_values=0,
            )

            length = self.window

        max_start = length - self.window

        positive_idx = np.where((labels_1d == 2) | (labels_1d == 3))[0]

        if len(positive_idx) > 0 and np.random.rand() < 0.75:
            center = int(np.random.choice(positive_idx))

            # небольшой сдвиг, чтобы событие не всегда было строго по центру
            shift = np.random.randint(-self.window // 4, self.window // 4 + 1)
            start = center - self.window // 2 + shift
            start = max(0, min(start, max_start))
        else:
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        end = start + self.window

        signal_win = signal[:, start:end]
        labels_win = labels_1d[start:end]

        return signal_win, labels_win

    def __getitem__(self, idx):
        sample = self.samples[idx]

        signal, labels = self._load_item(sample)
        signal_win, labels_win = self._random_window(signal, labels)

        signal_win = np.nan_to_num(
            signal_win,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        mean = signal_win.mean(axis=1, keepdims=True)
        std = signal_win.std(axis=1, keepdims=True)
        signal_win = (signal_win - mean) / (std + 1e-8)

        labels_win = np.nan_to_num(
            labels_win,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return signal_win.astype(np.float32), labels_win.astype(np.int64)