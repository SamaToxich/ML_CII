import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Блок: Свёртка → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Блок с skip connection: y = Conv(Conv(x)) + x."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )

    def forward(self, x):
        return self.block(x) + x


class FaceEncoder(nn.Module):
    """
    Принимает лицо [B, 3, 112, 112] → эмбеддинг [B, 512].
    """
    def __init__(self, embedding_size=512):
        super().__init__()

        # Блок 1: вход
        self.stem = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64, stride=2)    # [B, 64, 56, 56]
        )

        # Блок 2
        self.layer1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)               # [B, 64, 56, 56]
        )

        # Блок 3
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),   # [B, 128, 28, 28]
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Блок 4
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),  # [B, 256, 14, 14]
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Блок 5
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),  # [B, 512, 7, 7]
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # Голова
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, 512, 1, 1]
        self.embedding = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)                # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 512]
        x = self.embedding(x)           # [B, 512]
        x = self.bn(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x