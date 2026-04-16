import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
from models.unet1d import UNet1D
from ecg_signal_processor import load_signal


def merge_small_segments(mask: np.ndarray, min_len: int = 4) -> np.ndarray:
    mask = mask.copy()

    segments = []
    in_seg = False
    start = 0
    cls = 0

    for i, val in enumerate(mask):
        if val != 0 and not in_seg:
            start = i
            in_seg = True
            cls = val
        elif in_seg and val != cls:
            end = i

            if end - start < min_len:
                if segments:
                    segments[-1] = (segments[-1][0], end, segments[-1][2])
                else:
                    segments.append((start, end, cls))
            else:
                segments.append((start, end, cls))

            in_seg = False

    if in_seg:
        segments.append((start, len(mask), cls))

    new_mask = np.zeros_like(mask)
    for s, e, cls in segments:
        new_mask[s:e] = cls

    return new_mask


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
            x_tensor = torch.from_numpy(x_win).float().unsqueeze(0).to(device)  # [1, C, W]

            logits = model(x_tensor)  # [1, classes, W]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [classes, W]

            if scores_sum is None:
                num_classes = probs.shape[0]
                scores_sum = np.zeros((num_classes, length), dtype=np.float32)

            scores_sum[:, start:end] += probs
            counts[start:end] += 1

        # хвост
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
    qrs_thr: float = 0.20,
    spikes_thr: float = 0.10,
    apply_merge: bool = True,
    min_len: int = 4,
) -> np.ndarray:
    """
    classes:
    0 = background
    1 = QRS
    2 = SPIKES
    """
    qrs_prob = probs_avg[1]
    spikes_prob = probs_avg[2]

    pred_mask = np.zeros(qrs_prob.shape[0], dtype=np.int32)

    pred_mask[qrs_prob >= qrs_thr] = 1
    pred_mask[spikes_prob >= spikes_thr] = 2  # SPIKES приоритетнее

    if apply_merge:
        pred_mask = merge_small_segments(pred_mask, min_len=min_len)

    return pred_mask


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

    scale = np.max(np.abs(sig)) * 0.5
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

def clip_long_segments(mask, max_len=50):
    mask = mask.copy()
    current = 0
    start = None

    for i in range(len(mask)):
        if mask[i] != 0 and start is None:
            start = i
            current = mask[i]

        elif mask[i] != current and start is not None:
            if i - start > max_len:
                mask[start:i] = 0
            start = None

    return mask


def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    model = UNet1D(classes=3, in_channels=12).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    signal_path = find_unlabeled_signal(config.SIGNAL_DIR, config.MARKUP_DIR)
    if signal_path is None:
        raise ValueError("Не найден ни один неразмеченный .npy файл.")

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
        qrs_thr=0.30,
        spikes_thr=0.50,
        apply_merge=True,
        min_len=4,
    )
    
    pred_mask = clip_long_segments(pred_mask, max_len=80)

    print("Unique predicted classes:", np.unique(pred_mask))
    print("Mean QRS prob:", probs_avg[1].mean())
    print("Mean SPIKES prob:", probs_avg[2].mean())
    print("Max QRS prob:", probs_avg[1].max())
    print("Max SPIKES prob:", probs_avg[2].max())

    plot_all_in_one(signal, probs_avg, pred_mask, channel=0)


if __name__ == "__main__":
    main()