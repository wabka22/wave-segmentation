import torch
from sklearn.metrics import f1_score


def evaluate(model, loader, device):

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for x,y in loader:

            x = x.to(device)

            p = model(x).argmax(1).cpu()

            preds.extend(p.numpy().flatten())
            trues.extend(y.numpy().flatten())

    f1 = f1_score(trues, preds, average=None)

    print("F1 P:",f1[1])
    print("F1 QRS:",f1[2])
    print("F1 T:",f1[3])
