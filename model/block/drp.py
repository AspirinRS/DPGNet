import torch
import torch.nn as nn
import torch.nn.functional as F


class DRP(nn.Module):
    def __init__(self, in_dim, ds=4):
        super(DRP, self).__init__()
        self.d_k = in_dim
        self.ds  = ds

        self.conv_re1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_re2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

        self.conv_k1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.conv_v1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.conv_k2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.conv_v2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.conv_q  = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

        self.softmax = nn.Softmax(dim=-1)
        if ds > 1:
            self.pool = nn.AvgPool2d(ds)

    def forward(self, f1, f2, D):
        B, C, H, W = f1.shape

        f1_1, f1_2 = torch.chunk(f1, 2, dim=1)
        f2_1, f2_2 = torch.chunk(f2, 2, dim=1)
        F_re1 = self.conv_re1(torch.cat([f1_1, f2_2], dim=1)) + f1   # [B, C, H, W]
        F_re2 = self.conv_re2(torch.cat([f2_1, f1_2], dim=1)) + f2   # [B, C, H, W]

        if self.ds > 1:
            F_re1_d = self.pool(F_re1)
            F_re2_d = self.pool(F_re2)
            D_d     = self.pool(D)
        else:
            F_re1_d, F_re2_d, D_d = F_re1, F_re2, D
        _, _, h, w = F_re1_d.shape
        N = h * w
        scale = self.d_k ** -0.5

        K1 = self.conv_k1(F_re1_d).view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        V1 = self.conv_v1(F_re1_d).view(B, C, N)                   # [B, C, N]
        K2 = self.conv_k2(F_re2_d).view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        V2 = self.conv_v2(F_re2_d).view(B, C, N)                   # [B, C, N]

        Q = self.conv_q(D_d).view(B, C, N).permute(0, 2, 1)         # [B, N, C]

        A1 = self.softmax(torch.bmm(Q, K1.permute(0, 2, 1)) * scale)  # [B, N, N]
        A2 = self.softmax(torch.bmm(Q, K2.permute(0, 2, 1)) * scale)  # [B, N, N]

        R1 = torch.bmm(V1, A1.permute(0, 2, 1)).view(B, C, h, w)    # [B, C, h, w]
        R2 = torch.bmm(V2, A2.permute(0, 2, 1)).view(B, C, h, w)    # [B, C, h, w]

        if self.ds > 1:
            R1 = F.interpolate(R1, size=(H, W), mode='bilinear', align_corners=False)
            R2 = F.interpolate(R2, size=(H, W), mode='bilinear', align_corners=False)

        S = torch.abs(R1 - R2)
        D_r = self.conv_fuse(torch.cat([D, S], dim=1)) + D

        return D_r


if __name__ == '__main__':
    B, C, H, W = 2, 128, 32, 32
    f1  = torch.randn(B, C, H, W)
    f2  = torch.randn(B, C, H, W)
    D   = torch.randn(B, C, H, W)
    drp = DRP(in_dim=C, ds=4)
    out = drp(f1, f2, D)
    print(f"D_r: {out.shape}")   # [2, 128, 32, 32]
