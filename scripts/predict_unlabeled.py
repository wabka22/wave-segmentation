import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from ecg_signal_processor import load_signal
from models.unet1d import UNet1D


def clip_long_segments(mask: np.ndarray, max_len: int = 60) -> np.ndarray:
    mask = mask.copy()
    start = None
    current = 0

    for i in range(len(mask)):
        if mask[i] != 0 and start is None:
            start = i
            current = mask[i]
        elif start is not None and mask[i] != current:
            if i - start > max_len:
                mask[start:i] = 0
            start = None

    if start is not None and len(mask) - start > max_len:
        mask[start:len(mask)] = 0

    return mask


def remove_small_segments(mask: np.ndarray, cls: int, min_len: int) -> np.ndarray:
    mask = mask.copy()
    start = None

    for i in range(len(mask)):
        if mask[i] == cls and start is None:
            start = i
        elif mask[i] != cls and start is not None:
            if i - start < min_len:
                mask[start:i] = 0
            start = None

    if start is not None and len(mask) - start < min_len:
        mask[start:len(mask)] = 0

    return mask


def find_unlabeled_signal(signal_dir: str | Path, markup_dir: str | Path) -> Path | None:
    signal_dir = Path(signal_dir)
    markup_dir = Path(markup_dir)

    for signal_path in sorted(signal_dir.glob("*.npy")):
        file_id = signal_path.stem
        markup_path = markup_dir / f"{file_id}.json"

        if not markup_path.exists():
            return signal_path

    return None


def predict_full_signal_probs(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: str,
    window: int,
    step: int,
) -> np.ndarray:
    """
    signal: [C, L]
    return: probs_avg [classes, L]
    """
    signal = signal.astype(np.float32)
    length = signal.shape[1]

    scores_sum = None
    counts = np.zeros(length, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, length - window + 1, step):
            end = start + window

            x_win = signal[:, start:end]
            x_tensor = torch.from_numpy(x_win).float().unsqueeze(0).to(device)

            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            if scores_sum is None:
                num_classes = probs.shape[0]
                scores_sum = np.zeros((num_classes, length), dtype=np.float32)

            scores_sum[:, start:end] += probs
            counts[start:end] += 1

        if np.any(counts == 0):
            last_start = max(0, length - window)
            last_end = length

            x_win = signal[:, last_start:last_end]
            x_tensor = torch.from_numpy(x_win).float().unsqueeze(0).to(device)

            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            if scores_sum is None:
                num_classes = probs.shape[0]
                scores_sum = np.zeros((num_classes, length), dtype=np.float32)

            scores_sum[:, last_start:last_end] += probs
            counts[last_start:last_end] += 1

    counts[counts == 0] = 1.0
    probs_avg = scores_sum / counts[None, :]
    return probs_avg


def probs_to_mask(
    probs_avg: np.ndarray,
    spikes_thr: float = 0.75,
) -> np.ndarray:
    pred_mask = probs_avg.argmax(axis=0).astype(np.int32)

    spikes_prob = probs_avg[2]
    pred_mask[(pred_mask == 2) & (spikes_prob < spikes_thr)] = 0

    return pred_mask


def mask_to_segments_full(mask: np.ndarray, channel: int) -> list[dict]:
    segments = []
    start = None
    current_cls = 0

    for i, val in enumerate(mask):
        val = int(val)

        if val != 0 and start is None:
            start = i
            current_cls = val

        elif start is not None and val != current_cls:
            segments.append(
                {
                    "Channel": int(channel),
                    "Type": int(current_cls - 1),  # 1->0(QRS), 2->1(SPIKES)
                    "StartMark": int(start),
                    "EndMark": int(i - 1),
                    "SegmentationAgent": 1,
                    "ComplexMark": None,
                }
            )

            start = None
            current_cls = 0

            if val != 0:
                start = i
                current_cls = val

    if start is not None:
        segments.append(
            {
                "Channel": int(channel),
                "Type": int(current_cls - 1),
                "StartMark": int(start),
                "EndMark": int(len(mask) - 1),
                "SegmentationAgent": 1,
                "ComplexMark": None,
            }
        )

    return segments


def save_prediction_full_json(
    pred_mask: np.ndarray,
    signal: np.ndarray,
    signal_path: str | Path,
    output_dir: str | Path,
    label_channel: int,
    sample_rate: int = 500,
) -> Path:
    signal_path = Path(signal_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{signal_path.stem}.json"

    n_channels = signal.shape[0]
    all_channels = [[] for _ in range(n_channels)]
    all_channels[label_channel] = mask_to_segments_full(pred_mask, label_channel)

    markup = {
        "SignalName": signal_path.name,
        "SampleRate": int(sample_rate),
        "SignalFileSize": int(signal.size),
        "UsedModel": "UNet1D",
        "Segments": all_channels,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(markup, f, ensure_ascii=False, indent=2)

    return output_path


def plot_all_in_one(
    signal: np.ndarray,
    probs_avg: np.ndarray,
    pred_mask: np.ndarray,
    channel: int = 0,
):
    sig = signal[channel]
    qrs_prob = probs_avg[1]
    spikes_prob = probs_avg[2]

    plt.figure(figsize=(16, 5))

    plt.plot(sig, label="ECG signal", linewidth=1, color="black")

    scale = np.max(np.abs(sig)) * 0.5 if np.max(np.abs(sig)) > 0 else 1.0
    plt.plot(qrs_prob * scale, label="QRS prob", color="red", alpha=0.7)
    plt.plot(spikes_prob * scale, label="SPIKES prob", color="orange", alpha=0.7)

    colors = {
        1: "red",
        2: "green",
    }

    names = {
        1: "QRS",
        2: "SPIKES",
    }

    used = set()

    for cls in [1, 2]:
        idx = np.where(pred_mask == cls)[0]
        if len(idx) == 0:
            continue

        start = idx[0]

        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                end = idx[i - 1]
                label = names[cls] if cls not in used else None
                plt.axvspan(start, end, alpha=0.2, color=colors[cls], label=label)
                used.add(cls)
                start = idx[i]

        label = names[cls] if cls not in used else None
        plt.axvspan(start, idx[-1], alpha=0.2, color=colors[cls], label=label)
        used.add(cls)

    plt.title(f"Signal + prediction + probabilities (channel {channel})")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    model = UNet1D(classes=3, in_channels=12).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    signal_path = Path(config.SIGNAL_DIR) / "40.npy"
    # signal_path = find_unlabeled_signal(config.SIGNAL_DIR, config.MARKUP_DIR)

    if signal_path is None or not signal_path.exists():
        raise ValueError(f"Не найден файл сигнала: {signal_path}")

    print("Using unlabeled signal:", signal_path)

    signal = load_signal(signal_path)

    probs_avg = predict_full_signal_probs(
        model=model,
        signal=signal,
        device=device,
        window=config.WINDOW,
        step=config.STEP,
    )

    pred_mask = probs_to_mask(
        probs_avg,
        spikes_thr=0.65,
    )

    pred_mask = remove_small_segments(pred_mask, cls=1, min_len=12)  # QRS
    pred_mask = remove_small_segments(pred_mask, cls=2, min_len=8)   # SPIKES
    pred_mask = clip_long_segments(pred_mask, max_len=60)

    output_json = save_prediction_full_json(
        pred_mask=pred_mask,
        signal=signal,
        signal_path=signal_path,
        output_dir="data/prediction_markin",
        label_channel=config.LABEL_CHANNEL,
        sample_rate=500,
    )

    print("Saved JSON to:", output_json)
    print("Unique predicted classes:", np.unique(pred_mask))
    print("Mean QRS prob:", probs_avg[1].mean())
    print("Mean SPIKES prob:", probs_avg[2].mean())
    print("Max QRS prob:", probs_avg[1].max())
    print("Max SPIKES prob:", probs_avg[2].max())

    plot_all_in_one(signal, probs_avg, pred_mask, channel=config.LABEL_CHANNEL)


if __name__ == "__main__":
    main()