import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class StripDSConv(nn.Module):
    def __init__(self, channels):
        super(StripDSConv, self).__init__()
        self.dsc_1x7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dsc_7x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.dsc_1x7(x) + self.dsc_7x1(x))


class SpectralDifferenceGenerator(nn.Module):
    def __init__(self, in_channels):
        super(SpectralDifferenceGenerator, self).__init__()
        self.strip_conv = StripDSConv(in_channels)

    def forward(self, x1, x2):
        f1 = self.strip_conv(x1)
        f2 = self.strip_conv(x2)

        S1_mu  = f1.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        S2_mu  = f2.mean(dim=[2, 3], keepdim=True)
        S1_var = f1.var( dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        S2_var = f2.var( dim=[2, 3], keepdim=True)

        S1_mu_phi = F.relu(S1_mu)
        S2_mu_phi = F.relu(S2_mu)

        P_com = torch.cat([S1_mu_phi * S2_mu_phi, S1_var * S2_var], dim=1)  # [B, 2C, 1, 1]

        eps = 1e-8
        dot_mu   = (S1_mu_phi * S2_mu_phi).sum(dim=1, keepdim=True)          # [B, 1, 1, 1]
        norm_mu  = (S1_mu_phi.norm(p=2, dim=1, keepdim=True) *
                    S2_mu_phi.norm(p=2, dim=1, keepdim=True)).clamp(min=eps)
        angle_mu = torch.acos((dot_mu / norm_mu).clamp(-1 + eps, 1 - eps)) / math.pi  # [B, 1, 1, 1]
        dot_var  = (S1_var * S2_var).sum(dim=1, keepdim=True)                # [B, 1, 1, 1]
        norm_var = (S1_var.norm(p=2, dim=1, keepdim=True) *
                    S2_var.norm(p=2, dim=1, keepdim=True)).clamp(min=eps)
        angle_var = torch.acos((dot_var / norm_var).clamp(-1 + eps, 1 - eps)) / math.pi
        P_diff = torch.cat([angle_mu, angle_var], dim=1)                     # [B, 2, 1, 1]

        return P_com, P_diff


class DPSBranch(nn.Module):
    def __init__(self, channels):
        super(DPSBranch, self).__init__()
        self.coord_att = CoordAtt(inp=channels, oup=channels, reduction=16)
        self.conv_expand = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pdiff_proj = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_sub, p_diff):
        B, C, H, W = x_sub.shape
        d_low  = F.avg_pool2d(x_sub, kernel_size=3, stride=1, padding=1)
        d_low  = F.interpolate(d_low, size=(H, W), mode='bilinear', align_corners=False)
        d_edge = x_sub - d_low
        d_coordatt = self.coord_att(d_edge) + x_sub
        feat = self.conv_expand(d_coordatt)          # [B, C, H, W]
        pdiff_weight = self.pdiff_proj(p_diff)       # [B, C, 1, 1]
        feat = feat * pdiff_weight
        sde = feat + x_sub
        return sde


class CPF(nn.Module):
    def __init__(self, dim):
        super(CPF, self).__init__()
        self.layer_norm = nn.GroupNorm(1, dim)
        self.H0 = nn.Parameter(torch.randn(dim, 1, 1, 2, dtype=torch.float32) * 0.02)
        self.g_theta = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim),
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, p_com):
        B, C, H, W = x.shape
        x_in = x
        x = self.layer_norm(x)
        x_fft = torch.fft.rfft2(x.float(), dim=(2, 3), norm='ortho')  # [B, C, H, W//2+1] complex
        delta_H = self.g_theta(p_com)
        H0_complex = torch.view_as_complex(self.H0.contiguous())       # [C, 1, 1] complex
        modulation = 1.0 + self.alpha * torch.tanh(delta_H)            # [B, C, 1, 1] real
        H_i        = H0_complex * modulation                           # [B, C, 1, 1] complex
        x_fft = x_fft * H_i
        cde   = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')
        return cde + x_in


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                          # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)              # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w.expand(-1, -1, h, w) * a_h.expand(-1, -1, h, w)


class ProgressiveDSBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ProgressiveDSBlock, self).__init__()
        pad = kernel_size // 2
        self.dsconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.dsconv(x) + self.conv1x1(x)


class DifferenceAwareAttention(nn.Module):
    def __init__(self, gate_channel):
        super(DifferenceAwareAttention, self).__init__()
        self.prior_gen = SpectralDifferenceGenerator(gate_channel)
        self.cpf = CPF(dim=gate_channel)
        self.dps = DPSBranch(channels=gate_channel)
        self.refine_7x7 = ProgressiveDSBlock(gate_channel, kernel_size=7)
        self.refine_5x5 = ProgressiveDSBlock(gate_channel, kernel_size=5)
        self.refine_3x3 = ProgressiveDSBlock(gate_channel, kernel_size=3)

    def forward(self, x_sub, x_cat, x1_orig, x2_orig):
        p_com, p_diff = self.prior_gen(x1_orig, x2_orig)
        cde = self.cpf(x_cat, p_com)
        sde = self.dps(x_sub, p_diff)
        D0 = cde + sde
        D1  = self.refine_7x7.dsconv(D0) + self.refine_7x7.conv1x1(D0)
        D2  = self.refine_5x5.dsconv(D1) + self.refine_5x5.conv1x1(D0)
        D_i = self.refine_3x3.dsconv(D2) + self.refine_3x3.conv1x1(D0)
        return D_i


if __name__ == '__main__':
    x1 = torch.randn(2, 128, 32, 32)
    x2 = torch.randn(2, 128, 32, 32)
    x_sub = torch.abs(x1 - x2)
    import torch.nn as nn
    x_cat = nn.Conv2d(256, 128, 1)(torch.cat([x1, x2], dim=1))
    diff_attn = DifferenceAwareAttention(gate_channel=128)
    out = diff_attn(x_sub, x_cat, x1, x2)
    print(out.shape)  # [2, 128, 32, 32]
