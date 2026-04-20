import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from ecg_signal_processor import load_signal
from models.unet1d import UNet1D


def clip_long_segments(mask: np.ndarray, max_len: int = 80) -> np.ndarray:
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
    qrs_thr: float = 0.2,
    spikes_thr: float = 0.6,
) -> np.ndarray:
    bg_prob = probs_avg[0]
    qrs_prob = probs_avg[1]
    spikes_prob = probs_avg[2]

    pred_mask = np.zeros(qrs_prob.shape[0], dtype=np.int32)

    pred_mask[qrs_prob > bg_prob] = 1

    pred_mask[
        (spikes_prob > qrs_prob + 0.05) &
        (spikes_prob > bg_prob) &
        (spikes_prob > spikes_thr)
    ] = 2

    return pred_mask


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    # QRS не трогаем
    mask = remove_small_segments(mask, cls=2, min_len=8)   # SPIKES
    mask = clip_long_segments(mask, max_len=80)
    return mask


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


def save_prediction_all_channels_json(
    all_masks: list[np.ndarray],
    signal: np.ndarray,
    signal_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 500,
) -> Path:
    signal_path = Path(signal_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{signal_path.stem}.json"

    all_channels = []
    for ch, mask in enumerate(all_masks):
        all_channels.append(mask_to_segments_full(mask, ch))

    markup = {
        "SignalName": signal_path.name,
        "SampleRate": int(sample_rate),
        "SignalFileSize": int(signal.size),
        "UsedModel": "UNet1D_channel_emphasis",
        "Segments": all_channels,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(markup, f, ensure_ascii=False, indent=2)

    return output_path


def build_channel_emphasis_input(
    signal: np.ndarray,
    channel: int,
    other_scale: float = 0.2,
) -> np.ndarray:
    """
    Целевой канал оставляем как есть, остальные ослабляем.
    Это мягче, чем зануление остальных каналов.
    """
    x = signal.astype(np.float32).copy()
    for ch in range(x.shape[0]):
        if ch != channel:
            x[ch] *= other_scale
    return x


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

    signal_path = Path(config.SIGNAL_DIR) / "38.npy"

    if not signal_path.exists():
        raise ValueError(f"Не найден файл сигнала: {signal_path}")

    print("Using signal:", signal_path)

    signal = load_signal(signal_path)
    n_channels = signal.shape[0]

    all_masks = []
    debug_channel = 0
    debug_probs = None
    debug_mask = None

    for ch in range(n_channels):
        signal_ch = build_channel_emphasis_input(signal, ch, other_scale=0.2)

        probs_avg = predict_full_signal_probs(
            model=model,
            signal=signal_ch,
            device=device,
            window=config.WINDOW,
            step=config.STEP,
        )

        pred_mask = probs_to_mask(
            probs_avg,
            qrs_thr=0.25,
            spikes_thr=0.65,
        )

        pred_mask = postprocess_mask(pred_mask)
        all_masks.append(pred_mask)

        print(
            f"Channel {ch}: classes={np.unique(pred_mask)}, "
            f"mean_qrs={probs_avg[1].mean():.4f}, "
            f"mean_spikes={probs_avg[2].mean():.4f}, "
            f"max_qrs={probs_avg[1].max():.4f}, "
            f"max_spikes={probs_avg[2].max():.4f}"
        )

        if ch == debug_channel:
            debug_probs = probs_avg
            debug_mask = pred_mask

    output_json = save_prediction_all_channels_json(
        all_masks=all_masks,
        signal=signal,
        signal_path=signal_path,
        output_dir="data/prediction_markin",
        sample_rate=500,
    )

    print("Saved JSON to:", output_json)

    if debug_probs is not None and debug_mask is not None:
        plot_all_in_one(signal, debug_probs, debug_mask, channel=debug_channel)


if __name__ == "__main__":
    main()