import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ecg(signal, pred, true=None):
    """
    signal: (L,)
    pred: (L,)
    true: (L,) or None
    """

    signal = signal.squeeze()

    plt.figure(figsize=(15, 5))
    plt.plot(signal, label="ECG", color="black")

    colors = {
        1: "blue",   # P
        2: "red",    # QRS
        3: "green"   # T
    }

    for cls in [1, 2, 3]:
        mask = pred == cls
        plt.fill_between(
            np.arange(len(signal)),
            signal.min(),
            signal.max(),
            where=mask,
            color=colors[cls],
            alpha=0.2,
            label=f"Pred {cls}"
        )

    if true is not None:
        for cls in [1, 2, 3]:
            mask = true == cls
            plt.fill_between(
                np.arange(len(signal)),
                signal.min(),
                signal.max(),
                where=mask,
                color=colors[cls],
                alpha=0.1,
                hatch='//',
                label=f"True {cls}"
            )

    plt.legend(loc="upper right")
    plt.title("ECG segmentation")
    plt.show()