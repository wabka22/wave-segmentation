import torch
from torch.utils.data import DataLoader

from datasets.ludb_dataset import LUDBDataset
from models.unet1d import UNet1D
from utils.visualization import plot_ecg
import config


def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    records = [str(i) for i in range(1, 201)]
    test_records = records[170:]

    test_dataset = LUDBDataset(config.DATA_PATH, test_records)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
    )

    model = UNet1D().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    print("Best model loaded.")

    num_samples = 5

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= num_samples:
                break

            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()[0]
            true = y.numpy()[0]

            # первый канал ECG
            signal = x.cpu().numpy()[0, 0]

            plot_ecg(signal, pred, true)


if __name__ == "__main__":
    main()