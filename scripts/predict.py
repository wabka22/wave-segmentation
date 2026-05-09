import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from ecg_signal_processor import load_signal
from models.unet1d import UNet1D


MODEL_TO_JSON_TYPE = {
    1: 0,  # QRS
    2: 1,  # SPIKES
    3: 4,  # QRS_AFTER_SPIKE
}


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)

    return (signal - mean) / (std + 1e-8)


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


def clip_long_segments(mask: np.ndarray, cls: int, max_len: int) -> np.ndarray:
    mask = mask.copy()
    start = None

    for i in range(len(mask)):
        if mask[i] == cls and start is None:
            start = i
        elif mask[i] != cls and start is not None:
            if i - start > max_len:
                mask[start:i] = 0
            start = None

    if start is not None and len(mask) - start > max_len:
        mask[start:len(mask)] = 0

    return mask


def predict_full_signal_probs(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: str,
    window: int,
    step: int,
) -> np.ndarray:
    signal = normalize_signal(signal)
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
    return scores_sum / counts[None, :]


def probs_to_mask(
    probs_avg: np.ndarray,
    qrs_thr: float = 0.50,
    spikes_thr: float = 0.25,
    qrs_after_thr: float = 0.50,
) -> np.ndarray:
    bg_prob = probs_avg[0]
    qrs_prob = probs_avg[1]
    spikes_prob = probs_avg[2]
    qrs_after_prob = probs_avg[3]

    pred_mask = np.zeros(qrs_prob.shape[0], dtype=np.int32)

    # SPIKES
    spikes_mask = (
        (spikes_prob >= spikes_thr)
        & (spikes_prob >= qrs_prob * 0.75)
        & (spikes_prob >= bg_prob)
    )
    pred_mask[spikes_mask] = 2

    # QRS_AFTER_SPIKE
    qrs_after_mask = (
        (qrs_after_prob >= qrs_after_thr)
        & (qrs_after_prob >= bg_prob)
        & (qrs_after_prob >= qrs_prob * 0.75)
        & (pred_mask == 0)
    )
    pred_mask[qrs_after_mask] = 3

    # QRS
    qrs_mask = (
        (qrs_prob >= qrs_thr)
        & (qrs_prob >= bg_prob)
        & (pred_mask == 0)
    )
    pred_mask[qrs_mask] = 1

    return pred_mask

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = remove_small_segments(mask, cls=1, min_len=5)
    mask = remove_small_segments(mask, cls=2, min_len=1)
    mask = remove_small_segments(mask, cls=3, min_len=8)

    mask = clip_long_segments(mask, cls=1, max_len=60)
    mask = clip_long_segments(mask, cls=2, max_len=40)
    mask = clip_long_segments(mask, cls=3, max_len=100)

    return mask

def keep_qrs_after_only_near_spike(
    mask: np.ndarray,
    max_dist_after_spike: int = 160,
) -> np.ndarray:
    mask = mask.copy()

    spike_idx = np.where(mask == 2)[0]

    if len(spike_idx) == 0:
        mask[mask == 3] = 1
        return mask

    qrs_after_idx = np.where(mask == 3)[0]

    for i in qrs_after_idx:
        has_spike_before = np.any(
            (spike_idx >= i - max_dist_after_spike) &
            (spike_idx < i)
        )

        if not has_spike_before:
            mask[i] = 1

    return mask

def mask_to_segments(mask: np.ndarray, channel: int) -> list[dict]:
    segments = []
    start = None
    current_cls = 0

    for i, val in enumerate(mask):
        val = int(val)

        if val != 0 and start is None:
            start = i
            current_cls = val

        elif start is not None and val != current_cls:
            if current_cls in MODEL_TO_JSON_TYPE:
                segments.append(
                    {
                        "Channel": int(channel),
                        "Type": int(MODEL_TO_JSON_TYPE[current_cls]),
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

    if start is not None and current_cls in MODEL_TO_JSON_TYPE:
        segments.append(
            {
                "Channel": int(channel),
                "Type": int(MODEL_TO_JSON_TYPE[current_cls]),
                "StartMark": int(start),
                "EndMark": int(len(mask) - 1),
                "SegmentationAgent": 1,
                "ComplexMark": None,
            }
        )

    return segments


def save_prediction_all_channels_json(
    pred_mask: np.ndarray,
    signal: np.ndarray,
    signal_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 500,
) -> Path:
    signal_path = Path(signal_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{signal_path.stem}.json"

    n_channels = signal.shape[0]
    all_channels = []

    for ch in range(n_channels):
        all_channels.append(mask_to_segments(pred_mask, ch))

    markup = {
        "SignalName": signal_path.name,
        "SampleRate": int(sample_rate),
        "SignalFileSize": int(signal.size),
        "UsedModel": "UNet1D_4classes_last_model",
        "Segments": all_channels,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(markup, f, ensure_ascii=False, indent=2)

    return output_path


def get_segments(mask: np.ndarray, cls: int) -> list[tuple[int, int]]:
    segments = []
    start = None

    for i, val in enumerate(mask):
        if val == cls and start is None:
            start = i
        elif val != cls and start is not None:
            segments.append((start, i - 1))
            start = None

    if start is not None:
        segments.append((start, len(mask) - 1))

    return segments


def make_qrs_after_spikes(
    mask: np.ndarray,
    min_dist: int = 0,
    max_dist: int = 260,
) -> np.ndarray:
    mask = mask.copy()

    spike_segments = get_segments(mask, cls=2)
    qrs_segments = get_segments(mask, cls=1)
    qrs_after_segments = get_segments(mask, cls=3)

    used_qrs = set()

    for sp_start, sp_end in spike_segments:
        best_seg = None
        best_dist = None
        best_kind = None

        for idx, (q_start, q_end) in enumerate(qrs_segments):
            if idx in used_qrs:
                continue

            dist = q_start - sp_end

            if min_dist <= dist <= max_dist:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_seg = (q_start, q_end)
                    best_kind = ("qrs", idx)

        for q_start, q_end in qrs_after_segments:
            dist = q_start - sp_end

            if min_dist <= dist <= max_dist:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_seg = (q_start, q_end)
                    best_kind = ("qrs_after", None)

        if best_seg is not None:
            q_start, q_end = best_seg
            mask[q_start:q_end + 1] = 3

            if best_kind[0] == "qrs":
                used_qrs.add(best_kind[1])

    return mask

def plot_prediction(
    signal: np.ndarray,
    probs_avg: np.ndarray,
    pred_mask: np.ndarray,
    channel: int = 0,
):
    sig = signal[channel]

    qrs_prob = probs_avg[1]
    spikes_prob = probs_avg[2]
    qrs_after_prob = probs_avg[3]

    plt.figure(figsize=(18, 6))
    plt.plot(sig, label="ECG signal", linewidth=1, color="black")

    scale = np.max(np.abs(sig)) * 0.5 if np.max(np.abs(sig)) > 0 else 1.0

    plt.plot(qrs_prob * scale, label="QRS prob", color="red", alpha=0.7)
    plt.plot(spikes_prob * scale, label="SPIKES prob", color="green", alpha=0.7)
    plt.plot(qrs_after_prob * scale, label="QRS_AFTER_SPIKE prob", color="orange", alpha=0.7)

    colors = {
        1: "red",
        2: "green",
        3: "orange",
    }

    names = {
        1: "QRS",
        2: "SPIKES",
        3: "QRS_AFTER_SPIKE",
    }

    used = set()

    for cls in [1, 2, 3]:
        idx = np.where(pred_mask == cls)[0]

        if len(idx) == 0:
            continue

        start = idx[0]

        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                end = idx[i - 1]
                plt.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=colors[cls],
                    label=names[cls] if cls not in used else None,
                )
                used.add(cls)
                start = idx[i]

        plt.axvspan(
            start,
            idx[-1],
            alpha=0.2,
            color=colors[cls],
            label=names[cls] if cls not in used else None,
        )
        used.add(cls)

    plt.title(f"Signal + prediction + probabilities | channel {channel}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

def enforce_spike_then_qrs_after(
    mask: np.ndarray,
    clear_before: int = 35,
    clear_after: int = 20,
    qrs_search_after: int = 260,
) -> np.ndarray:
    mask = mask.copy()

    spike_segments = get_segments(mask, cls=2)

    for sp_start, sp_end in spike_segments:

        left = max(0, sp_start - clear_before)
        right = min(len(mask), sp_end + clear_after + 1)

        qrs_near_spike = mask[left:right] == 1
        part = mask[left:right]
        part[qrs_near_spike] = 0
        mask[left:right] = part

        search_left = sp_end + 1
        search_right = min(len(mask), sp_end + qrs_search_after + 1)

        candidate_start = None
        candidate_end = None

        i = search_left
        while i < search_right:
            if mask[i] == 1 or mask[i] == 3:
                candidate_start = i
                current_cls = mask[i]

                while i < search_right and mask[i] == current_cls:
                    i += 1

                candidate_end = i - 1
                break

            i += 1

        if candidate_start is not None:
            mask[candidate_start:candidate_end + 1] = 3

            after_left = candidate_end + 1
            after_right = min(len(mask), candidate_end + 40)

            part = mask[after_left:after_right]
            part[part == 1] = 0
            mask[after_left:after_right] = part

    return mask


def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    model = UNet1D(classes=4, in_channels=12).to(device)
    model.load_state_dict(
        torch.load("checkpoints/best_model.pth", map_location=device)
    )
    model.eval()

    signal_path = Path("data/data_with_spikes/ecs_short") / "122.npy"
    # signal_path = Path("data/segmentation/signals") / "12.npy"

    if not signal_path.exists():
        raise ValueError(f"Не найден файл сигнала: {signal_path}")

    print("Using signal:", signal_path)

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
        qrs_thr=0.45,
        spikes_thr=0.25,
        qrs_after_thr=0.60,
    )
    
    pred_mask = keep_qrs_after_only_near_spike(
        pred_mask,
        max_dist_after_spike=260,
    )

    pred_mask = enforce_spike_then_qrs_after(
        pred_mask,
        clear_before=35,
        clear_after=20,
        qrs_search_after=260,
    )

    # pred_mask = postprocess_mask(pred_mask)
    
    print(
        f"Prediction: "
        f"classes={np.unique(pred_mask)}, "
        f"mean_qrs={probs_avg[1].mean():.4f}, "
        f"mean_spikes={probs_avg[2].mean():.4f}, "
        f"mean_qrs_after={probs_avg[3].mean():.4f}, "
        f"max_qrs={probs_avg[1].max():.4f}, "
        f"max_spikes={probs_avg[2].max():.4f}, "
        f"max_qrs_after={probs_avg[3].max():.4f}"
    )

    output_json = save_prediction_all_channels_json(
        pred_mask=pred_mask,
        signal=signal,
        signal_path=signal_path,
        output_dir="data/data_with_spikes/prediction_markin",
        sample_rate=500,
    )

    print("Saved JSON to:", output_json)

    plot_prediction(
        signal=signal,
        probs_avg=probs_avg,
        pred_mask=pred_mask,
        channel=0,
    )


if __name__ == "__main__":
    main()