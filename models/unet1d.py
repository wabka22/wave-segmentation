import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            # Свертка: извлекаем признаки из локального окна (размер 3)
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),  # нормализация (ускоряет и стабилизирует обучение)
            nn.ReLU(),              # нелинейность

            # Вторая свертка — уточняем признаки
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        # Просто пропускаем вход через блок
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, classes=4, in_channels=12):

        super().__init__()

        # ===== ENCODER (сжатие, извлечение признаков) =====
        self.enc1 = ConvBlock(in_channels, 32)  # [B,12,L] -> [B,32,L]
        self.pool1 = nn.MaxPool1d(2)            # L -> L/2

        self.enc2 = ConvBlock(32, 64)           # [B,32,L/2] -> [B,64,L/2]
        self.pool2 = nn.MaxPool1d(2)            # L/2 -> L/4

        self.enc3 = ConvBlock(64, 128)          # [B,64,L/4] -> [B,128,L/4]

        # ===== DECODER (восстановление сигнала) =====
        self.up1 = nn.ConvTranspose1d(128, 64, 2, stride=2)  # L/4 -> L/2
        self.dec1 = ConvBlock(128, 64)  # после concat: 64 + 64 = 128 каналов

        self.up2 = nn.ConvTranspose1d(64, 32, 2, stride=2)   # L/2 -> L
        self.dec2 = ConvBlock(64, 32)  # после concat: 32 + 32 = 64 каналов

        # Финальный слой: перевод признаков в классы
        self.out = nn.Conv1d(32, classes, 1)  # [B,32,L] -> [B,4,L]

    def forward(self, x):
        # ===== ENCODER =====
        e1 = self.enc1(x)              # [B,12,L] -> [B,32,L]
        e2 = self.enc2(self.pool1(e1)) # [B,32,L] -> pool -> [B,32,L/2] -> [B,64,L/2]

        e3 = self.enc3(self.pool2(e2)) # [B,64,L/2] -> pool -> [B,64,L/4] -> [B,128,L/4]

        # ===== DECODER =====
        d1 = self.up1(e3)              # [B,128,L/4] -> [B,64,L/2]

        # Skip connection: добавляем признаки из encoder (e2)
        d1 = torch.cat([d1, e2], dim=1)  # [B,64,L/2] + [B,64,L/2] -> [B,128,L/2]
        d1 = self.dec1(d1)               # -> [B,64,L/2]

        d2 = self.up2(d1)              # [B,64,L/2] -> [B,32,L]

        # Skip connection с самым ранним уровнем
        d2 = torch.cat([d2, e1], dim=1)  # [B,32,L] + [B,32,L] -> [B,64,L]
        d2 = self.dec2(d2)               # -> [B,32,L]

        # Финальный прогноз: для каждой точки 4 класса
        return self.out(d2)  # [B,32,L] -> [B,4,L]