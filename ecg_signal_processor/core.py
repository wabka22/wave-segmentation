import json
from pathlib import Path

import numpy as np


def load_markup(markup_path: str | Path) -> dict:
    """
    Загружает JSON-файл с разметкой ЭКГ.

    Возвращает содержимое JSON как словарь Python.
    """
    markup_path = Path(markup_path)

    with open(markup_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_signal(signal_path: str | Path) -> np.ndarray:
    """
    Загружает ЭКГ-сигнал из .npy файла.

    Если обычная загрузка не удалась, пытается восстановить .npy файл,
    найдя внутри него заголовок NUMPY.
    """
    signal_path = Path(signal_path)

    try:
        return np.load(signal_path, allow_pickle=False)
    except Exception:
        pass

    raw = signal_path.read_bytes()
    marker = b"NUMPY"
    pos = raw.find(marker)

    if pos == -1:
        raise ValueError(
            f"Не удалось загрузить сигнал и не найден заголовок NPY: {signal_path}"
        )

    recovered = b"\x93" + raw[pos:]
    tmp_path = signal_path.with_suffix(".recovered.npy")
    tmp_path.write_bytes(recovered)

    return np.load(tmp_path, allow_pickle=False)


def build_label_matrix(
    signal: np.ndarray,
    markup: dict,
    background_value: int = -1,
) -> np.ndarray:
    """
    Создаёт матрицу меток для сигнала на основе JSON-разметки.

    Возвращает массив формы [channels, samples], где каждому отсчёту
    соответствует тип сегмента из разметки или background_value для фона.
    """
    if signal.ndim != 2:
        raise ValueError(
            f"Ожидался сигнал формы [channels, samples], получено: {signal.shape}"
        )

    n_channels, n_samples = signal.shape
    labels = np.full((n_channels, n_samples), background_value, dtype=np.int32)

    segments_by_channel = markup.get("Segments")
    if segments_by_channel is None:
        raise ValueError("В JSON отсутствует ключ 'Segments'")

    for ch_idx, channel_segments in enumerate(segments_by_channel):
        if ch_idx >= n_channels:
            break

        for seg in channel_segments:
            seg_type = int(seg["Type"])
            start = max(0, int(seg["StartMark"]))
            end = min(n_samples - 1, int(seg["EndMark"]))

            if start <= end:
                labels[ch_idx, start:end + 1] = seg_type

    return labels


def load_sample(
    signal_path: str | Path,
    markup_path: str | Path,
    background_value: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает один обучающий пример: ЭКГ-сигнал и его разметку.

    Сигнал берётся из .npy файла, разметка — из JSON.
    Возвращает пару: signal и labels.
    """
    signal = load_signal(signal_path)
    markup = load_markup(markup_path)
    labels = build_label_matrix(signal, markup, background_value)

    return signal, labels