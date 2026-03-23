import torch
import torch.nn as nn
from .sprm_components import DifferenceAwareAttention


class SPRM(nn.Module):
    def __init__(self, in_d, out_d):
        super(SPRM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.x_concat = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        self.diff_attn = DifferenceAwareAttention(self.in_d)

        self.proj = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x_sub = torch.abs(x1 - x2)
        x_cat = self.x_concat(torch.cat([x1, x2], dim=1))
        D_i = self.diff_attn(x_sub, x_cat, x1, x2)   # [B, C, H, W]
        out = self.proj(D_i)
        return out, x1, x2

if __name__ == '__main__':
    x1 = torch.randn((32, 128, 8, 8))
    x2 = torch.randn((32, 128, 8, 8))
    model = SPRM(128, 128)
    out, x1_cal, x2_cal = model(x1, x2)
    print(f"diff: {out.shape}, x1_cal: {x1_cal.shape}, x2_cal: {x2_cal.shape}")
