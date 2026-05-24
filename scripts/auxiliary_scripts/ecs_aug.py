import json
import shutil
from pathlib import Path

import numpy as np


SIGNAL_DIR = Path("data/data_with_spikes/ecs_short")
MARKUP_DIR = Path("data/data_with_spikes/markings")

OUT_SIGNAL_DIR = Path("data/data_with_spikes/ecs_aug")
OUT_MARKUP_DIR = Path("data/data_with_spikes/markings_aug")

SAMPLE_RATE = 500
NUM_AUG_PER_FILE = 3


def load_json(path: Path) -> dict:
    """
    Загружает JSON-файл с разметкой.

    Возвращает содержимое файла как словарь Python.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_spike_markup(markup: dict) -> bool:
    """
    Проверяет, есть ли в JSON-разметке SPIKE или QRS_AFTER_SPIKE.

    В исходной разметке:
        Type 1 -> SPIKES
        Type 4 -> QRS_AFTER_SPIKE

    Аугментация выполняется только для файлов,
    где есть хотя бы один из этих типов сегментов.
    """
    for channel_segments in markup.get("Segments", []):
        for seg in channel_segments:
            seg_type = int(seg.get("Type", -1))

            if seg_type in (1, 4):
                return True

    return False


def augment_ecg(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Выполняет лёгкую аугментацию ЭКГ-сигнала.

    Изменяется только амплитуда сигнала:
        - применяется общий коэффициент усиления;
        - применяется небольшой коэффициент усиления по каналам;
        - иногда выполняется инверсия всего сигнала.

    Временная ось не изменяется, поэтому JSON-разметку можно
    копировать без изменений.

    Args:
        signal: ЭКГ-сигнал формы [channels, samples].
        sample_rate: Частота дискретизации сигнала.

    Returns:
        Аугментированный сигнал формы [channels, samples].
    """
    aug = signal.astype(np.float32).copy()

    if aug.ndim != 2:
        raise ValueError(
            f"Expected signal shape [channels, samples], got {aug.shape}"
        )

    n_channels, _ = aug.shape

    # Общий gain для всего сигнала.
    gain = np.random.uniform(0.60, 1.10)
    aug *= gain

    # Небольшой gain отдельно для каждого канала.
    channel_gain = np.random.uniform(0.97, 1.03, size=(n_channels, 1))
    aug *= channel_gain

    # Иногда инвертируем весь сигнал.
    if np.random.rand() < 0.10:
        aug *= -1.0

    return aug.astype(np.float32)


def copy_original_files(signal_path: Path, markup_path: Path) -> None:
    """
    Копирует оригинальный сигнал и JSON-разметку в выходные папки.
    """
    shutil.copy2(signal_path, OUT_SIGNAL_DIR / signal_path.name)
    shutil.copy2(markup_path, OUT_MARKUP_DIR / markup_path.name)


def save_augmented_sample(
    signal: np.ndarray,
    markup_path: Path,
    file_id: str,
    aug_id: int,
) -> None:
    """
    Создаёт и сохраняет один аугментированный вариант сигнала.

    Разметка копируется без изменений, так как аугментация
    не меняет временную ось сигнала.
    """
    aug_signal = augment_ecg(signal, sample_rate=SAMPLE_RATE)

    new_name = f"{file_id}_aug{aug_id + 1}"
    new_signal_path = OUT_SIGNAL_DIR / f"{new_name}.npy"
    new_markup_path = OUT_MARKUP_DIR / f"{new_name}.json"

    np.save(new_signal_path, aug_signal)
    shutil.copy2(markup_path, new_markup_path)


def main() -> None:
    """
    Создаёт аугментированную версию датасета со spike-разметкой.

    Скрипт проходит по всем .npy сигналам, ищет соответствующий JSON,
    проверяет наличие SPIKES или QRS_AFTER_SPIKE и создаёт несколько
    аугментированных копий таких сигналов.

    Оригинальные файлы также копируются в выходные папки.
    """
    OUT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MARKUP_DIR.mkdir(parents=True, exist_ok=True)

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

        # Копируем оригинал как есть.
        copy_original_files(signal_path, markup_path)

        for aug_id in range(NUM_AUG_PER_FILE):
            save_augmented_sample(
                signal=signal,
                markup_path=markup_path,
                file_id=file_id,
                aug_id=aug_id,
            )
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