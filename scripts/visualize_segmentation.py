import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIGNAL_DIR = Path("data/data_with_spikes/ecs_short")
MARKUP_DIR = Path("data/data_with_spikes/markings")


LABEL_NAMES = {
    0: "Type 0",
    1: "Type 1",
    2: "Type 2",
    3: "Type 3",
    4: "Type 4",
}


COLORS = {
    0: "green",
    1: "orange",
    2: "blue",
    3: "red",
    4: "purple",
}


def load_json_markup(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_json_file(
    file_id: str,
    channel: int = 0,
    start: int = 0,
    end: int | None = None,
):
    signal_path = SIGNAL_DIR / f"{file_id}.npy"
    markup_path = MARKUP_DIR / f"{file_id}.json"

    signal = np.load(signal_path, allow_pickle=False)
    markup = load_json_markup(markup_path)

    sig = signal[channel]

    if end is None:
        end = len(sig)

    sig = sig[start:end]

    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(start, end), sig, linewidth=1, label=f"Signal ch{channel}")

    segments = markup["Segments"][channel]

    print(f"\nFILE: {file_id}")
    print(f"CHANNEL: {channel}")

    used = set()

    for seg in segments:
        seg_type = int(seg["Type"])
        seg_start = int(seg["StartMark"])
        seg_end = int(seg["EndMark"])

        if seg_end < start or seg_start > end:
            continue

        draw_start = max(seg_start, start)
        draw_end = min(seg_end, end)

        print(
            f"Type {seg_type}: {seg_start}-{seg_end}"
        )

        plt.axvspan(
            draw_start,
            draw_end,
            alpha=0.3,
            color=COLORS.get(seg_type, "gray"),
            label=LABEL_NAMES.get(seg_type, f"Type {seg_type}")
            if seg_type not in used
            else None,
        )

        used.add(seg_type)

    plt.title(
        f"{file_id}.json | channel {channel} | samples {start}:{end}"
    )
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    files = sorted(SIGNAL_DIR.glob("*.npy"))

    print("Available files:")
    for f in files[:20]:
        print(f.stem)

    # поменяй при необходимости
    plot_json_file(
        file_id="0",
        channel=0,
        start=0,
        end=3000,
    )