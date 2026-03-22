import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):

    def __init__(self, classes=4, in_channels=12):

        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ConvBlock(32,64)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ConvBlock(64,128)

        self.up1 = nn.ConvTranspose1d(128,64,2,stride=2)
        self.dec1 = ConvBlock(128,64)

        self.up2 = nn.ConvTranspose1d(64,32,2,stride=2)
        self.dec2 = ConvBlock(64,32)

        self.out = nn.Conv1d(32,classes,1)

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        d1 = self.up1(e3)
        d1 = torch.cat([d1,e2],dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2,e1],dim=1)
        d2 = self.dec2(d2)

        return self.out(d2)
