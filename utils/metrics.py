import torch
import numpy as np
from sklearn.metrics import f1_score


def mask_to_segments(mask, cls):
    """
    Преобразует одномерную маску классов в список сегментов заданного класса.

    Например, если mask содержит непрерывный участок класса cls,
    функция вернёт его границы в формате (start, end),
    где start включается, а end не включается.
    """
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
    """
    Считает IoU для двух сегментов.

    IoU = длина пересечения сегментов / длина их объединения.
    Значение близко к 1 означает сильное совпадение,
    значение 0 означает отсутствие пересечения.
    """
    s1, e1 = seg1
    s2, e2 = seg2

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    return inter / union if union > 0 else 0.0


def match_segments(pred_segs, true_segs, iou_thr=0.3, tol=10):
    """
    Считает количество предсказанных сегментов, совпавших с истинными.

    Сегмент считается найденным, если он пересекается с истинным сегментом
    с учётом допуска tol или если IoU между сегментами больше iou_thr.
    Один истинный сегмент может быть сопоставлен только один раз.
    """
    matched = 0
    used = set()

    for p in pred_segs:
        for i, t in enumerate(true_segs):
            if i in used:
                continue

            overlap = not (p[1] < t[0] - tol or t[1] < p[0] - tol)

            if overlap or iou(p, t) > iou_thr:
                matched += 1
                used.add(i)
                break

    return matched


def segment_f1(pred_mask, true_mask, cls):
    """
    Считает F1-score на уровне сегментов для одного класса.

    Сначала маски переводятся в списки сегментов.
    Затем считается количество совпавших сегментов:
    tp — найденные правильные сегменты,
    fp — лишние предсказанные сегменты,
    fn — пропущенные истинные сегменты.

    Возвращает F1-score для заданного класса.
    """
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


def merge_small_segments(mask, min_len=4):
    """
    Выполняет простую постобработку предсказанной маски.

    Короткие ненулевые сегменты длиной меньше min_len
    объединяются с предыдущим сегментом, если он есть.
    Это уменьшает влияние коротких шумовых предсказаний
    перед расчётом segment F1.
    """
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


def evaluate(model, loader, device, min_seg_len=4):
    """
    Оценивает качество модели на validation или test DataLoader.

    Считает два типа метрик:
    1. Point-wise F1 — F1 по отдельным точкам маски.
    2. Segment F1 — F1 по целым сегментам для классов
       QRS, SPIKES и QRS_AFTER_SPIKE.

    Для segment F1 перед оценкой постобрабатывается только prediction,
    истинная разметка остаётся без изменений.

    Возвращает словарь списков segment F1 по классам:
    {1: [...], 2: [...], 3: [...]}.
    """
    model.eval()

    preds, trues = [], []

    # 1 = QRS, 2 = SPIKES, 3 = QRS_AFTER_SPIKE
    seg_f1_scores = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            p = model(x).argmax(1).cpu().numpy()
            y = y.cpu().numpy()

            for i in range(p.shape[0]):
                pred_mask = p[i].copy()
                true_mask = y[i].copy()

                # постобрабатываем только prediction
                pred_mask = merge_small_segments(pred_mask, min_len=min_seg_len)

                for cls in [1, 2, 3]:
                    seg_f1_scores[cls].append(
                        segment_f1(pred_mask, true_mask, cls)
                    )

                preds.extend(pred_mask.flatten())
                trues.extend(true_mask.flatten())

    preds = np.array(preds)
    trues = np.array(trues)

    f1 = f1_score(
        trues,
        preds,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    print("\n--- Point-wise F1 ---")
    print("F1 background:", f1[0])
    print("F1 QRS:", f1[1])
    print("F1 SPIKES:", f1[2])
    print("F1 QRS_AFTER_SPIKE:", f1[3])

    print("\n--- Segment F1 (post-processed) ---")
    print("F1 QRS:", np.mean(seg_f1_scores[1]) if len(seg_f1_scores[1]) > 0 else 0.0)
    print("F1 SPIKES:", np.mean(seg_f1_scores[2]) if len(seg_f1_scores[2]) > 0 else 0.0)
    print(
        "F1 QRS_AFTER_SPIKE:",
        np.mean(seg_f1_scores[3]) if len(seg_f1_scores[3]) > 0 else 0.0
    )

    return seg_f1_scores