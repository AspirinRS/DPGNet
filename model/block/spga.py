import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        t = int(abs(math.log2(channels) / gamma) + b / gamma)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                               # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)            # [B, 1, C]
        y = self.conv(y)                               # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)          # [B, C, 1, 1]
        return x * self.sigmoid(y).expand_as(x)


class DilatedDWEncoder(nn.Module):
    DILATIONS = [1, 2, 4, 8]

    def __init__(self, channels):
        super(DilatedDWEncoder, self).__init__()
        self.dw_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3,
                          padding=d, dilation=d, groups=channels, bias=False),
                nn.GELU(),
            ) for d in self.DILATIONS
        ])
        self.eca_layers = nn.ModuleList([ECA(channels) for _ in self.DILATIONS])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.DILATIONS), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = []
        D_k = x
        for dw, eca in zip(self.dw_layers, self.eca_layers):
            D_k = dw(D_k)
            feats.append(eca(D_k))
        return self.fuse(torch.cat(feats, dim=1))


class SemanticPriorGenerator(nn.Module):
    def __init__(self, channels):
        super(SemanticPriorGenerator, self).__init__()
        self.conv_w = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.conv_prior = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, D_f_cur, D_f_up):
        w = self.conv_w(torch.cat([D_f_cur, D_f_up], dim=1))
        D_p = w * D_f_cur + (1 - w) * D_f_up
        return self.conv_prior(D_p)


class SPGA(nn.Module):
    def __init__(self, channels):
        super(SPGA, self).__init__()
        self.encoder = DilatedDWEncoder(channels)
        self.prior_gen = SemanticPriorGenerator(channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),   # 2C → C
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        _, _, H2, W2 = x2.shape
        D_f = self.encoder(x2)                    # [B, C, H2, W2]
        x1_up = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)
        M_prior = self.prior_gen(D_f, x1_up)      # [B, 1, H2, W2]
        D_cat = torch.cat([D_f, x1_up], dim=1)   # [B, 2C, H2, W2]
        D_cat = D_cat * M_prior
        D_out = self.conv_out(D_cat)
        return D_out, M_prior


if __name__ == '__main__':
    x1 = torch.randn(2, 128, 8, 8)
    x2 = torch.randn(2, 128, 16, 16)
    model = SPGA(128)
    out, prior = model(x1, x2)
    print(f"D_out: {out.shape}, M_prior: {prior.shape}")
