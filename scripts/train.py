import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm

import config
from datasets.ecg_dataset import ECGDataset
from models.unet1d import UNet1D
from utils.metrics import evaluate

cudnn.benchmark = True


def set_seed(seed=42):
    """
    Фиксирует генераторы случайных чисел для повторяемости обучения.

    Это помогает получать более похожие результаты при повторных запусках:
    одинаковое перемешивание данных, одинаковую инициализацию весов
    и более детерминированное поведение CUDA/cuDNN.
    """
    random.seed(seed)  # фиксирует Python random
    np.random.seed(seed)  # фиксирует NumPy random
    torch.manual_seed(seed)  # фиксирует PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # фиксирует PyTorch GPU
    torch.backends.cudnn.deterministic = True  # просит детерминированные CUDA-алгоритмы
    torch.backends.cudnn.benchmark = False  # отключает автоподбор быстрых CUDA-алгоритмов


def dice_loss(pred, target, smooth=1e-6):
    """
    Считает Dice Loss для задачи сегментации ЭКГ.

    pred — выход модели формы [B, classes, L].
    target — правильные метки формы [B, L].

    Dice измеряет совпадение предсказанных и правильных областей.
    Фон исключается из расчёта, чтобы он не доминировал над редкими классами.
    """
    # Dice измеряет, насколько хорошо совпали две области:
    num_classes = pred.shape[1]

    pred = F.softmax(pred, dim=1)  # значения нейронки в вероятности классов

    target_onehot = F.one_hot(target, num_classes).permute(0, 2, 1).float()  # перевод в матрицу

    intersection = (pred * target_onehot).sum(dim=2)
    union = pred.sum(dim=2) + target_onehot.sum(dim=2)

    dice = (2 * intersection + smooth) / (union + smooth)

    # фон убираем то есть первый класс из наших классов
    dice = dice[:, 1:]

    return 1 - dice.mean()  # Dice — это метрика качества, её надо максимизировать


def compute_loss(pred, y, weights):
    """
    Считает итоговую функцию потерь для обучения модели.

    Используются две части:
    1. CrossEntropy — ошибка классификации каждой отдельной точки.
    2. Dice Loss — ошибка совпадения целых сегментов.

    Итоговый loss — взвешенная сумма CrossEntropy и Dice Loss.
    """
    num_classes = pred.shape[1]

    logits = pred.permute(0, 2, 1).reshape(-1, num_classes)
    targets = y.reshape(-1)

    ce = F.cross_entropy(
        logits,
        targets,
        weight=weights,
        reduction="mean",
        label_smoothing=0.005,
    )

    dice = dice_loss(pred, y)

    loss = 0.65 * ce + 0.35 * dice

    # CE = ошибка классификации каждой отдельной точки
    # DiceLoss = ошибка совпадения целых сегментов
    return loss, ce, dice


def print_split_info(name, dataset):
    """
    Печатает информацию о train/val/test subset.

    Показывает, сколько json-примеров и mask-примеров попало в split,
    а также выводит несколько имён файлов для проверки.
    """
    json_count = 0
    mask_count = 0
    examples = []

    base_dataset = dataset.dataset
    indices = dataset.indices

    for idx in indices:
        sample = base_dataset.samples[int(idx)]

        if sample["type"] == "json":
            json_count += 1
        elif sample["type"] == "mask":
            mask_count += 1

        if len(examples) < 10:
            examples.append(
                f'{sample["type"]}: {sample["signal_path"].name}'
            )

    print(f"\n{name}:")
    print(f"  json: {json_count}")
    print(f"  mask: {mask_count}")
    print("  examples:")
    for ex in examples:
        print(f"    {ex}")


def create_loaders():
    """
    Создаёт train, validation и test DataLoader.

    Внутри:
    1. Создаётся полный ECGDataset.
    2. Отдельно собираются индексы json и mask-примеров.
    3. Данные перемешиваются и делятся на train/val/test.
    4. json-примеры дополнительно повторяются, чтобы редкие классы чаще попадали в обучение.
    5. Создаются DataLoader для обучения, валидации и теста.
    """
    full_dataset = ECGDataset(
        json_signal_dir="data/data_with_spikes/ecs_short",
        json_markup_dir="data/data_with_spikes/markings",
        mask_datasets=[
            ("data/segmentation/signals", "data/segmentation/masks"),
            ("data/segmentation_kvachadze_npy/signals", "data/segmentation_kvachadze_npy/masks"),
        ],
        background_value=-1,
        json_repeat=1,
    )

    x, y = full_dataset[0]

    print(type(x), type(y))
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("samples[0]:", full_dataset.samples[0])
    print("dataset len:", len(full_dataset))

    json_indices = [
        i for i, sample in enumerate(full_dataset.samples)
        if sample["type"] == "json"
    ]

    mask_indices = [
        i for i, sample in enumerate(full_dataset.samples)
        if sample["type"] == "mask"
    ]

    rng = np.random.default_rng(config.SEED)
    # перемешать элементы списка в случайном порядке.
    rng.shuffle(json_indices)
    rng.shuffle(mask_indices)

    def split_indices(indices):
        """
        Делит список индексов на train, validation и test.

        Размеры частей задаются через config.TRAIN_RATIO и config.VAL_RATIO.
        Test получает оставшиеся индексы.
        """
        n = len(indices)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = train_end + int(n * config.VAL_RATIO)

        return (
            indices[:train_end],
            indices[train_end:val_end],
            indices[val_end:],
        )

    json_train, json_val, json_test = split_indices(json_indices)
    mask_train, mask_val, mask_test = split_indices(mask_indices)

    json_repeat = 35

    mask_train = mask_train[:int(len(mask_train) * 0.7)]
    mask_val = mask_val[:int(len(mask_val) * 0.7)]
    mask_test = mask_test[:int(len(mask_test) * 0.7)]

    train_indices = json_train * json_repeat + mask_train
    val_indices = json_val * 5 + mask_val
    test_indices = json_test * 5 + mask_test

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    print_split_info("TRAIN", train_dataset)
    print_split_info("VAL", val_dataset)
    print_split_info("TEST", test_dataset)

    print(f"Total real samples: {len(full_dataset)}")
    print(f"Train samples:      {len(train_dataset)}")
    print(f"Val samples:        {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, weights, use_amp):
    """
    Обучает модель одну эпоху.

    Одна эпоха — это один полный проход по всем батчам train DataLoader.
    Внутри для каждого батча выполняются:
    forward pass, расчёт loss, backward pass и обновление весов модели.
    """
    model.train()

    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0

    progress = tqdm(loader, desc="train", leave=False)

    for x, y in progress:
        # x — входные данные для модели
        # y — правильные ответы / разметка
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # перед каждым новым батчем надо обнулить старые градиенты
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss, ce, dice = compute_loss(pred, y, weights)

        scaler.scale(loss).backward()  # Вычисляем градиенты, масштабируя loss для стабильности при использовании смешанной точности
        scaler.step(optimizer)  # Обновляем веса модели
        scaler.update()  # Обновляем scaler

        loss_sum += loss.item()
        ce_sum += ce.item()
        dice_sum += dice.item()

    n = len(loader)
    return loss_sum / n, ce_sum / n, dice_sum / n  # возвращаем средние значения по эпохе для loss, ce и dice


def validate(model, loader, device):
    """
    Оценивает модель на validation или test DataLoader.

    Использует функцию evaluate(), которая возвращает segment F1
    для каждого класса. Затем считается отдельный F1 для QRS, SPIKES,
    QRS_AFTER_SPIKE и общий взвешенный score.
    """
    seg_f1_scores = evaluate(model, loader, device)

    val_f1_qrs = float(np.mean(seg_f1_scores[1]))
    val_f1_spikes = float(np.mean(seg_f1_scores[2]))
    val_f1_qrs_after_spike = float(np.mean(seg_f1_scores[3]))

    val_mean_seg_f1 = (
        0.2 * val_f1_qrs +
        0.4 * val_f1_spikes +
        0.4 * val_f1_qrs_after_spike
    )

    return {
        "val_f1_qrs": val_f1_qrs,
        "val_f1_spikes": val_f1_spikes,
        "val_f1_qrs_after_spike": val_f1_qrs_after_spike,
        "val_mean_seg_f1": val_mean_seg_f1,
    }


def main():
    """
    Основная функция обучения.

    Создаёт DataLoader, модель, веса классов, optimizer и scaler.
    Затем запускает обучение на несколько эпох, сохраняет last_model
    после каждой эпохи и best_model по лучшему validation score.
    После обучения загружает лучшую модель и оценивает её на test set.
    """
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = create_loaders()

    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    use_amp = False

    model = UNet1D(classes=4, in_channels=12).to(device)

    weights = torch.tensor(
        [0.02, 0.25, 0.45, 0.65],
        dtype=torch.float32,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) #объект, который обновляет веса нейросети после того, как PyTorch посчитал градиенты.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_score = -1.0
    best_model_path = "checkpoints/best_model.pth"
    last_model_path = "checkpoints/last_model.pth"

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
            f"QRS_AFTER_SPIKE: {val_metrics['val_f1_qrs_after_spike']:.4f} | "
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
                "val_f1_qrs_after_spike": val_metrics["val_f1_qrs_after_spike"],
                "val_mean_seg_f1": current_score,
            }
        )

        torch.save(model.state_dict(), last_model_path)
        print(f"Last model updated at epoch {epoch}")

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model updated at epoch {epoch} | "
                f"best_val_mean_seg_f1: {best_score:.4f}"
            )

    print("\nTraining completed.")

    print("\nLoading BEST model and evaluating on TEST...")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    model.eval()

    test_metrics = validate(model, test_loader, device)

    print("\nFinal TEST metrics:")
    print(f"QRS segment F1:             {test_metrics['val_f1_qrs']:.4f}")
    print(f"SPIKES segment F1:          {test_metrics['val_f1_spikes']:.4f}")
    print(f"QRS_AFTER_SPIKE segment F1: {test_metrics['val_f1_qrs_after_spike']:.4f}")
    print(f"Mean segment F1:            {test_metrics['val_mean_seg_f1']:.4f}")


if __name__ == "__main__":
    set_seed(config.SEED)

    main()