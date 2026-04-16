import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ecg_dataset import ECGDataset
from models.unet1d import UNet1D
from utils.metrics import evaluate
import config

cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]

    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 2, 1).float()

    intersection = (pred * target_onehot).sum(dim=2)
    union = pred.sum(dim=2) + target_onehot.sum(dim=2)

    dice = (2 * intersection + smooth) / (union + smooth)

    # фон убираем
    dice = dice[:, 1:]

    return 1 - dice.mean()


def compute_loss(pred, y, weights):
    dice = dice_loss(pred, y)

    num_classes = pred.shape[1]
    logits = pred.permute(0, 2, 1).reshape(-1, num_classes)
    targets = y.reshape(-1)

    ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")

    pt = torch.exp(-ce)
    focal = ((1 - pt) ** 2 * ce).mean()

    loss = 0.5 * focal + 0.5 * dice
    return loss, ce.mean(), dice


def get_available_file_ids(signal_dir, markup_dir):
    signal_dir = Path(signal_dir)
    markup_dir = Path(markup_dir)

    file_ids = []

    for signal_path in sorted(signal_dir.glob("*.npy")):
        file_id = signal_path.stem
        markup_path = markup_dir / f"{file_id}.json"

        if markup_path.exists():
            file_ids.append(file_id)

    return file_ids


def split_file_ids(file_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    if len(file_ids) < 3:
        raise ValueError("Слишком мало файлов для train/val/test split")

    rng = np.random.default_rng(seed)
    shuffled = list(file_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    return train_ids, val_ids, test_ids


def create_loaders():
    file_ids = get_available_file_ids(config.SIGNAL_DIR, config.MARKUP_DIR)

    if len(file_ids) == 0:
        raise ValueError("Не найдено ни одной пары .npy + .json")

    train_ids = config.TRAIN_IDS
    val_ids = config.VAL_IDS
    test_ids = config.TEST_IDS

    print("Found files:", file_ids)
    print("Train ids:", train_ids)
    print("Val ids:", val_ids)
    print("Test ids:", test_ids)

    train_dataset = ECGDataset(
        signal_dir=config.SIGNAL_DIR,
        markup_dir=config.MARKUP_DIR,
        file_ids=train_ids,
        label_channel=config.LABEL_CHANNEL,
        window=config.WINDOW,
        step=config.STEP,
    )

    val_dataset = ECGDataset(
        signal_dir=config.SIGNAL_DIR,
        markup_dir=config.MARKUP_DIR,
        file_ids=val_ids,
        label_channel=config.LABEL_CHANNEL,
        window=config.WINDOW,
        step=config.STEP,
    )

    test_dataset = ECGDataset(
        signal_dir=config.SIGNAL_DIR,
        markup_dir=config.MARKUP_DIR,
        file_ids=test_ids,
        label_channel=config.LABEL_CHANNEL,
        window=config.WINDOW,
        step=config.STEP,
    )

    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows:   {len(val_dataset)}")
    print(f"Test windows:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, weights, use_amp):
    model.train()

    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0

    progress = tqdm(loader, desc="train", leave=False)

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss, ce, dice = compute_loss(pred, y, weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        ce_sum += ce.item()
        dice_sum += dice.item()

    n = len(loader)
    return loss_sum / n, ce_sum / n, dice_sum / n


def validate(model, loader, device):
    seg_f1_scores = evaluate(model, loader, device)

    val_f1_qrs = float(np.mean(seg_f1_scores[1]))
    val_f1_spikes = float(np.mean(seg_f1_scores[2]))
    val_mean_seg_f1 = (val_f1_qrs + val_f1_spikes) / 2.0

    return {
        "val_f1_qrs": val_f1_qrs,
        "val_f1_spikes": val_f1_spikes,
        "val_mean_seg_f1": val_mean_seg_f1,
    }

def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = create_loaders()

    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    model = UNet1D(classes=3, in_channels=12).to(device)
    
    weights = torch.tensor([0.03, 0.48, 0.49], dtype=torch.float32, device=device)      
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_epoch = -1
    patience = 10
    bad_epochs = 0
    history = []

    print(f"Device: {device}")
    print("Start training...")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_ce, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            weights=weights,
            use_amp=use_amp,
        )

        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(
            f"train_loss: {train_loss:.4f} | "
            f"train_ce: {train_ce:.4f} | "
            f"train_dice: {train_dice:.4f}"
        )

        val_metrics = validate(model, val_loader, device)
        current_score = val_metrics["val_mean_seg_f1"]

        print(
            f"val_seg_f1 -> "
            f"QRS: {val_metrics['val_f1_qrs']:.4f} | "
            f"SPIKES: {val_metrics['val_f1_spikes']:.4f} | "
            f"mean: {current_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce": train_ce,
                "train_dice": train_dice,
                "val_f1_qrs": val_metrics["val_f1_qrs"],
                "val_f1_spikes": val_metrics["val_f1_spikes"],
                "val_mean_seg_f1": current_score,
            }
        )

        pd.DataFrame(history).to_csv("history.csv", index=False)

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            bad_epochs = 0

            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"Best model saved at epoch {epoch} with score {best_score:.4f}")
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{patience}")

        if bad_epochs >= patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest epoch: {best_epoch}, best val mean segment F1: {best_score:.4f}")

    print("\nLoading best model and evaluating on TEST...")
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    test_metrics = validate(model, test_loader, device)

    print("\nFinal TEST metrics:")
    print(f"QRS segment F1:    {test_metrics['val_f1_qrs']:.4f}")
    print(f"SPIKES segment F1: {test_metrics['val_f1_spikes']:.4f}")
    print(f"Mean segment F1:   {test_metrics['val_mean_seg_f1']:.4f}")


if __name__ == "__main__":
    set_seed(config.SEED)
    main()