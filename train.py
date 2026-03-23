import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.ludb_dataset import LUDBDataset
from models.unet1d import UNet1D
from utils.metrics import evaluate
import config
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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


def main():
    records = [str(i) for i in range(1,201)]
    train = records[:160]
    test = records[160:]

    train_dataset = LUDBDataset(config.DATA_PATH, train)
    test_dataset = LUDBDataset(config.DATA_PATH, test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    model = UNet1D().to(device)

    weights = torch.tensor([0.05, 0.4, 0.4, 0.15]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")
    

    for epoch in range(config.EPOCHS):
        model.train()
        loss_sum = 0

        for x, y in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(x)

                dice = dice_loss(pred, y)

                ce = criterion(
                    pred.permute(0,2,1).reshape(-1,4),
                    y.reshape(-1)
                )

                loss = ce + 0.5 * dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

        print("epoch", epoch, "loss", loss_sum)

        evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()