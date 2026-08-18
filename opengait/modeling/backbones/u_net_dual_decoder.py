import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class U_NetDualDecoder(nn.Module):
    """共享编码器并分别预测概率图 P 与阈值图 T。"""

    def __init__(self, in_channels=3, p_out_channels=1, t_out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)

        self.p_up3 = UpConv(128, 64)
        self.p_dec3 = ConvBlock(128, 64)
        self.p_up2 = UpConv(64, 32)
        self.p_dec2 = ConvBlock(64, 32)
        self.p_up1 = UpConv(32, 16)
        self.p_dec1 = ConvBlock(32, 16)
        self.p_head = nn.Conv2d(16, p_out_channels, 1)

        self.t_up3 = UpConv(128, 64)
        self.t_dec3 = ConvBlock(128, 64)
        self.t_up2 = UpConv(64, 32)
        self.t_dec2 = ConvBlock(64, 32)
        self.t_up1 = UpConv(32, 16)
        self.t_dec1 = ConvBlock(32, 16)
        self.t_head = nn.Conv2d(16, t_out_channels, 1)

    def _encode(self, inputs):
        x1 = self.enc1(inputs)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

    @staticmethod
    def _decode(x1, x2, x3, x4, up3, dec3, up2, dec2, up1, dec1, head):
        d3 = dec3(torch.cat((x3, up3(x4)), dim=1))
        d2 = dec2(torch.cat((x2, up2(d3)), dim=1))
        d1 = dec1(torch.cat((x1, up1(d2)), dim=1))
        return head(d1)

    def forward(self, inputs):
        features = self._encode(inputs)
        p_logits = self._decode(
            *features,
            self.p_up3,
            self.p_dec3,
            self.p_up2,
            self.p_dec2,
            self.p_up1,
            self.p_dec1,
            self.p_head,
        )
        t_logits = self._decode(
            *features,
            self.t_up3,
            self.t_dec3,
            self.t_up2,
            self.t_dec2,
            self.t_up1,
            self.t_dec1,
            self.t_head,
        )
        return p_logits, t_logits
