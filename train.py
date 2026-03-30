import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ludb_dataset import LUDBDataset
from models.unet1d import UNet1D
from utils.metrics import evaluate
import config

cudnn.benchmark = True


def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]

    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 2, 1).float()

    intersection = (pred * target_onehot).sum(dim=2)
    union = pred.sum(dim=2) + target_onehot.sum(dim=2)

    dice = (2 * intersection + smooth) / (union + smooth)

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


def create_loaders():
    records = [str(i) for i in range(1, 201)]

    # train / val / test
    train_records = records[:140]
    val_records = records[140:170]
    test_records = records[170:]

    train_dataset = LUDBDataset(config.DATA_PATH, train_records)
    val_dataset = LUDBDataset(config.DATA_PATH, val_records)
    test_dataset = LUDBDataset(config.DATA_PATH, test_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
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

    val_f1_p = float(np.mean(seg_f1_scores[1]))
    val_f1_qrs = float(np.mean(seg_f1_scores[2]))
    val_f1_t = float(np.mean(seg_f1_scores[3]))
    val_mean_seg_f1 = (val_f1_p + val_f1_qrs + val_f1_t) / 3.0

    return {
        "val_f1_p": val_f1_p,
        "val_f1_qrs": val_f1_qrs,
        "val_f1_t": val_f1_t,
        "val_mean_seg_f1": val_mean_seg_f1,
    }


def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = create_loaders()

    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    model = UNet1D().to(device)

    weights = torch.tensor([0.05, 0.4, 0.4, 0.15], dtype=torch.float32, device=device)

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
            f"P: {val_metrics['val_f1_p']:.4f} | "
            f"QRS: {val_metrics['val_f1_qrs']:.4f} | "
            f"T: {val_metrics['val_f1_t']:.4f} | "
            f"mean: {current_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce": train_ce,
                "train_dice": train_dice,
                "val_f1_p": val_metrics["val_f1_p"],
                "val_f1_qrs": val_metrics["val_f1_qrs"],
                "val_f1_t": val_metrics["val_f1_t"],
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
    print(f"P segment F1:   {test_metrics['val_f1_p']:.4f}")
    print(f"QRS segment F1: {test_metrics['val_f1_qrs']:.4f}")
    print(f"T segment F1:   {test_metrics['val_f1_t']:.4f}")
    print(f"Mean segment F1:{test_metrics['val_mean_seg_f1']:.4f}")


if __name__ == "__main__":
    main()