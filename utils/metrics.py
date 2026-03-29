import torch
import numpy as np
from sklearn.metrics import f1_score


def mask_to_segments(mask, cls):
    segments = []
    in_seg = False
    start = 0

    for i in range(len(mask)):
        if mask[i] == cls and not in_seg:
            start = i
            in_seg = True
        elif mask[i] != cls and in_seg:
            segments.append((start, i))
            in_seg = False

    if in_seg:
        segments.append((start, len(mask)))

    return segments


def iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    return inter / union if union > 0 else 0


def match_segments(pred_segs, true_segs, iou_thr=0.3, tol=10):
    matched = 0
    used = set()

    for p in pred_segs:
        for i, t in enumerate(true_segs):
            if i in used:
                continue

            # overlap OR tolerance
            overlap = not (p[1] < t[0] - tol or t[1] < p[0] - tol)

            if overlap or iou(p, t) > iou_thr:
                matched += 1
                used.add(i)
                break

    return matched


def segment_f1(pred_mask, true_mask, cls):
    pred_segs = mask_to_segments(pred_mask, cls)
    true_segs = mask_to_segments(true_mask, cls)

    if len(pred_segs) == 0 and len(true_segs) == 0:
        return 1.0

    tp = match_segments(pred_segs, true_segs)

    fp = len(pred_segs) - tp
    fn = len(true_segs) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return 2 * precision * recall / (precision + recall + 1e-6)


def merge_small_segments(mask, min_len=10):
    mask = mask.copy() if isinstance(mask, np.ndarray) else mask.cpu().numpy()

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
                    segments[-1] = (segments[-1][0], end)
                else:
                    segments.append((start, end))
            else:
                segments.append((start, end))

            in_seg = False

    if in_seg:
        segments.append((start, len(mask)))

    new_mask = np.zeros_like(mask)
    for s, e in segments:
        new_mask[s:e] = mask[s]

    return new_mask


def evaluate(model, loader, device, min_seg_len=10):
    model.eval()

    preds, trues = [], []
    seg_f1_scores = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy() # Предсказания модели
            y = y.cpu().numpy() # Истинные метки

            # пост-обработка сегментов
            for i in range(p.shape[0]):
                p[i] = merge_small_segments(p[i], min_len=min_seg_len)
                y[i] = merge_small_segments(y[i], min_len=min_seg_len)

                for cls in [1, 2, 3]:
                    seg_f1_scores[cls].append(segment_f1(p[i], y[i], cls))

            preds.extend(p.flatten())
            trues.extend(y.flatten())

    f1 = f1_score(trues, preds, average=None)

    print("\n--- Point-wise F1 ---")
    print("F1 P:", f1[1])
    print("F1 QRS:", f1[2])
    print("F1 T:", f1[3])

    print("\n--- Segment F1 (post-processed) ---")
    print("F1 P:", np.mean(seg_f1_scores[1]))
    print("F1 QRS:", np.mean(seg_f1_scores[2]))
    print("F1 T:", np.mean(seg_f1_scores[3]))
    
    return seg_f1_scores
