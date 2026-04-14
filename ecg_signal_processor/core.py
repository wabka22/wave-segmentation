import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_markup(markup_path: str | Path) -> dict:
    markup_path = Path(markup_path)
    with open(markup_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_signal(signal_path: str | Path) -> np.ndarray:
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
    background_value: int = -1
) -> np.ndarray:
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


def build_dataframe(signal: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n_channels, n_samples = signal.shape
    data = {"sample": np.arange(n_samples)}

    for ch in range(n_channels):
        data[f"ch{ch}_signal"] = signal[ch]
        data[f"ch{ch}_label"] = labels[ch]

    return pd.DataFrame(data)


def load_sample(
    signal_path: str | Path,
    markup_path: str | Path,
    background_value: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    signal = load_signal(signal_path)
    markup = load_markup(markup_path)
    labels = build_label_matrix(signal, markup, background_value)
    return signal, labels


def process_file(
    signal_path: str | Path,
    markup_path: str | Path,
    output_dir: str | Path,
    background_value: int = -1,
    save_csv: bool = False,
    save_signal: bool = True,
    save_labels: bool = True,
) -> dict:
    signal, labels = load_sample(signal_path, markup_path, background_value)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_signal:
        np.save(output_dir / "signal.npy", signal)

    if save_labels:
        np.save(output_dir / "labels.npy", labels)

    if save_csv:
        df = build_dataframe(signal, labels)
        df.to_csv(output_dir / "data.csv", index=False)

    return {
        "signal_path": str(signal_path),
        "markup_path": str(markup_path),
        "output_dir": str(output_dir),
        "shape": signal.shape,
    }


def process_dataset(
    signal_dir: str | Path,
    markup_dir: str | Path,
    output_base_dir: str | Path,
    background_value: int = -1,
    save_csv: bool = False,
    save_signal: bool = True,
    save_labels: bool = True,
) -> tuple[int, int]:
    signal_dir = Path(signal_dir)
    markup_dir = Path(markup_dir)
    output_base_dir = Path(output_base_dir)

    if not signal_dir.exists():
        raise FileNotFoundError(f"Папка с сигналами не найдена: {signal_dir}")

    if not markup_dir.exists():
        raise FileNotFoundError(f"Папка с разметкой не найдена: {markup_dir}")

    signal_files = sorted(signal_dir.glob("*.npy"))

    processed_count = 0
    skipped_count = 0

    for signal_path in signal_files:
        file_id = signal_path.stem
        markup_path = markup_dir / f"{file_id}.json"

        if not markup_path.exists():
            skipped_count += 1
            continue

        output_dir = output_base_dir / file_id
        process_file(
            signal_path=signal_path,
            markup_path=markup_path,
            output_dir=output_dir,
            background_value=background_value,
            save_csv=save_csv,
            save_signal=save_signal,
            save_labels=save_labels,
        )
        processed_count += 1

    return processed_count, skipped_count