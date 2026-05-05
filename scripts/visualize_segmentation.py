from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np


# Типы сегментов
BACKGROUND = -1
QRS = 0
SPIKE = 1
OTHER_SPIKE = 2
QRS_AFTER_SPIKE = 4

# Цвета
COLORS = {
    QRS: "green",
    SPIKE: "red",
    OTHER_SPIKE: "purple",      # если есть Type 2
    QRS_AFTER_SPIKE: "orange",
}

LABEL_NAMES = {
    QRS: "QRS",
    SPIKE: "SPIKE",
    OTHER_SPIKE: "SPIKE_TYPE_2",
    QRS_AFTER_SPIKE: "QRS_AFTER_SPIKE",
}


def load_json_labels(
    json_path: str | Path,
    signal_shape: tuple,
    background_value: int = -1,
) -> np.ndarray:
    """
    Преобразует json-разметку в массив labels shape=(channels, length)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = np.full(signal_shape, background_value, dtype=np.int64)

    for channel_segments in data["Segments"]:
        for seg in channel_segments:
            ch = seg["Channel"]
            seg_type = seg["Type"]
            start = seg["StartMark"]
            end = seg["EndMark"]

            labels[ch, start:end + 1] = seg_type

    return labels


def plot_signal_with_json_labels(
    signal_path: str | Path,
    labels_path: str | Path,
    channel: int = 0,
) -> None:
    signal = np.load(signal_path)

    # signal shape: (channels, samples)
    labels = load_json_labels(labels_path, signal.shape)

    sig = signal[channel]
    lab = labels[channel]

    plt.figure(figsize=(18, 6))
    plt.plot(sig, label="ECG signal", linewidth=1)

    unique_labels = np.unique(lab)
    unique_labels = unique_labels[unique_labels != BACKGROUND]

    used_labels = set()

    for label in unique_labels:
        indices = np.where(lab == label)[0]

        if len(indices) == 0:
            continue

        start = indices[0]

        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                end = indices[i - 1]

                plot_label = LABEL_NAMES.get(label, f"Type {label}") if label not in used_labels else None

                plt.axvspan(
                    start,
                    end,
                    alpha=0.3,
                    color=COLORS.get(label, "gray"),
                    label=plot_label,
                )

                used_labels.add(label)
                start = indices[i]

        plot_label = LABEL_NAMES.get(label, f"Type {label}") if label not in used_labels else None

        plt.axvspan(
            start,
            indices[-1],
            alpha=0.3,
            color=COLORS.get(label, "gray"),
            label=plot_label,
        )

        used_labels.add(label)

    plt.title(f"Channel {channel} | {Path(signal_path).name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    signal_path = project_root / "data" / "data_with_spikes" / "ecs_short" / "3.npy"
    labels_path = project_root / "data" / "data_with_spikes" / "markings" / "3.json"

    print("Project root:", project_root)
    print("Signal path:", signal_path)
    print("Labels path:", labels_path)

    plot_signal_with_json_labels(signal_path, labels_path, channel=0)