import torch
import config

from pathlib import Path

from models.unet1d import UNet1D


def export_model():
    checkpoint_path = Path("checkpoints/best_model.pth")
    output_path = Path("checkpoints/best_model.onnx")

    device = "cpu"

    model = UNet1D(classes=3, in_channels=12).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 12, config.WINDOW, device=device)


    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "signal_length",
            },
            "output": {
                0: "batch_size",
                2: "signal_length",
            },
        },
    )

    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    export_model()