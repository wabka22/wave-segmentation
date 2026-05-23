import json
import shutil
from pathlib import Path

import numpy as np


SIGNAL_DIR = Path("data/data_with_spikes/ecs_short")
MARKUP_DIR = Path("data/data_with_spikes/markings")

OUT_SIGNAL_DIR = Path("data/data_with_spikes/ecs_aug")
OUT_MARKUP_DIR = Path("data/data_with_spikes/markings_aug")

SAMPLE_RATE = 500


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_spike_markup(markup: dict) -> bool:
    """
    Type 1 -> SPIKES
    Type 4 -> QRS_AFTER_SPIKE

    Аугментируем только файлы, где есть spike или QRS_AFTER_SPIKE.
    """
    for channel_segments in markup.get("Segments", []):
        for seg in channel_segments:
            seg_type = int(seg.get("Type", -1))

            if seg_type in (1, 4):
                return True

    return False


def augment_ecg(signal: np.ndarray, sample_rate: int = 500) -> np.ndarray:
    """
    Очень лёгкая аугментация:
    - меняем только амплитуду;
    - иногда инвертируем сигнал;
    - временная ось НЕ меняется;
    - JSON-разметка остаётся полностью той же.
    """
    aug = signal.astype(np.float32).copy()

    if aug.ndim != 2:
        raise ValueError(f"Expected signal shape [channels, samples], got {aug.shape}")

    n_channels, _ = aug.shape

    # Общий gain
    gain = np.random.uniform(0.60, 1.10)
    aug *= gain

    # Небольшой gain по каналам
    channel_gain = np.random.uniform(0.97, 1.03, size=(n_channels, 1))
    aug *= channel_gain

    # Иногда инверсия всего сигнала
    if np.random.rand() < 0.10:
        aug *= -1.0

    return aug.astype(np.float32)


def main():
    OUT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MARKUP_DIR.mkdir(parents=True, exist_ok=True)

    num_aug_per_file = 3

    processed = 0
    skipped = 0
    created = 0

    for signal_path in sorted(SIGNAL_DIR.glob("*.npy")):
        file_id = signal_path.stem
        markup_path = MARKUP_DIR / f"{file_id}.json"

        if not markup_path.exists():
            skipped += 1
            continue

        markup = load_json(markup_path)

        if not has_spike_markup(markup):
            skipped += 1
            continue

        signal = np.load(signal_path, allow_pickle=False)

        # Копируем оригинал как есть
        shutil.copy2(signal_path, OUT_SIGNAL_DIR / signal_path.name)
        shutil.copy2(markup_path, OUT_MARKUP_DIR / markup_path.name)

        for aug_id in range(num_aug_per_file):
            aug_signal = augment_ecg(signal, sample_rate=SAMPLE_RATE)

            new_name = f"{file_id}_aug{aug_id + 1}"
            new_signal_path = OUT_SIGNAL_DIR / f"{new_name}.npy"
            new_markup_path = OUT_MARKUP_DIR / f"{new_name}.json"

            np.save(new_signal_path, aug_signal)

            shutil.copy2(markup_path, new_markup_path)

            created += 1

        processed += 1

    print("Done")
    print("processed files:", processed)
    print("created augmented files:", created)
    print("skipped files:", skipped)
    print("out signals:", OUT_SIGNAL_DIR)
    print("out markups:", OUT_MARKUP_DIR)


if __name__ == "__main__":
    main()