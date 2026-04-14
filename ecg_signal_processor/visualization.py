from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_with_labels(
    signal_path: str | Path,
    labels_path: str | Path,
    channel: int = 0,
) -> None:
    signal = np.load(signal_path)
    labels = np.load(labels_path)

    sig = signal[channel]
    lab = labels[channel]

    plt.figure(figsize=(15, 5))
    plt.plot(sig, label="ECG signal", linewidth=1)

    unique_labels = np.unique(lab)
    unique_labels = unique_labels[unique_labels != -1]

    colors = {
        0: "green",
        1: "red",
        2: "blue",
        3: "orange",
    }

    used_labels = set()

    for label in unique_labels:
        indices = np.where(lab == label)[0]
        if len(indices) == 0:
            continue

        start = indices[0]

        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                end = indices[i - 1]
                plot_label = f"label {label}" if label not in used_labels else None
                plt.axvspan(start, end, alpha=0.3, color=colors.get(label, "gray"), label=plot_label)
                used_labels.add(label)
                start = indices[i]

        plot_label = f"label {label}" if label not in used_labels else None
        plt.axvspan(start, indices[-1], alpha=0.3, color=colors.get(label, "gray"), label=plot_label)
        used_labels.add(label)

    plt.title(f"Channel {channel}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()