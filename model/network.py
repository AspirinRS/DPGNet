import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbone.mobilenetv2 import mobilenet_v2
from .block.spga import SPGA
from .util import init_method
from .block.heads import FCNHead, GatedResidualUpHead
from .block.sprm import SPRM
from .block.drp import DRP


def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == 'resnet18d':
        backbone = timm.create_model('resnet18d', pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class Detector(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_channels=128,
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super().__init__()
        self.backbone = get_backbone(backbone_name)

        self.lateral_d1 = nn.Conv2d(self.backbone.channels[1], fpn_channels, 1)
        self.lateral_d2 = nn.Conv2d(self.backbone.channels[2], fpn_channels, 1)
        self.lateral_d3 = nn.Conv2d(self.backbone.channels[3], fpn_channels, 1)
        self.lateral_d4 = nn.Conv2d(self.backbone.channels[4], fpn_channels, 1)

        self.d4_to_d3 = SPGA(fpn_channels)
        self.d3_to_d2 = SPGA(fpn_channels)
        self.d2_to_d1 = SPGA(fpn_channels)

        self.diff_sprm_d1 = SPRM(128, 128)
        self.diff_sprm_d2 = SPRM(128, 128)
        self.diff_sprm_d3 = SPRM(128, 128)
        self.diff_sprm_d4 = SPRM(128, 128)

        self.drp_d1 = DRP(fpn_channels, ds=4)
        self.drp_d2 = DRP(fpn_channels, ds=4)
        self.drp_d3 = DRP(fpn_channels, ds=4)
        self.drp_d4 = DRP(fpn_channels, ds=4)

        self.d4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.d3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.d2_head = nn.Conv2d(fpn_channels, 2, 1)
        self.d1_head = nn.Conv2d(fpn_channels, 2, 1)
        self.project = nn.Sequential(
            nn.Conv2d(fpn_channels*4, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(True)
        )
        self.head = GatedResidualUpHead(fpn_channels, 2, dropout_rate=dropout_rate)

        self.debug_step = 0

    def forward(self, x1, x2):
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)

        t1_d1 = self.lateral_d1(t1_c2)
        t1_d2 = self.lateral_d2(t1_c3)
        t1_d3 = self.lateral_d3(t1_c4)
        t1_d4 = self.lateral_d4(t1_c5)

        t2_d1 = self.lateral_d1(t2_c2)
        t2_d2 = self.lateral_d2(t2_c3)
        t2_d3 = self.lateral_d3(t2_c4)
        t2_d4 = self.lateral_d4(t2_c5)

        diff_d1, t1_cal_d1, t2_cal_d1 = self.diff_sprm_d1(t1_d1, t2_d1)
        diff_d2, _, _ = self.diff_sprm_d2(t1_d2, t2_d2)
        diff_d3, _, _ = self.diff_sprm_d3(t1_d3, t2_d3)
        diff_d4, _, _ = self.diff_sprm_d4(t1_d4, t2_d4)

        final_diff_d1 = self.drp_d1(t1_d1, t2_d1, diff_d1)
        final_diff_d2 = self.drp_d2(t1_d2, t2_d2, diff_d2)
        final_diff_d3 = self.drp_d3(t1_d3, t2_d3, diff_d3)
        final_diff_d4 = self.drp_d4(t1_d4, t2_d4, diff_d4)

        fea_d4 = final_diff_d4
        pred_d4 = self.d4_head(fea_d4)

        fea_d3, prior_d3 = self.d4_to_d3(fea_d4, final_diff_d3)
        pred_d3 = self.d3_head(fea_d3)

        fea_d2, prior_d2 = self.d3_to_d2(fea_d3, final_diff_d2)
        pred_d2 = self.d2_head(fea_d2)

        fea_d1, prior_d1 = self.d2_to_d1(fea_d2, final_diff_d1)
        pred_d1 = self.d1_head(fea_d1)

        pred = self.head(fea_d1)

        pred_d1 = F.interpolate(pred_d1, size=(256, 256), mode='bilinear', align_corners=False)
        pred_d2 = F.interpolate(pred_d2, size=(256, 256), mode='bilinear', align_corners=False)
        pred_d3 = F.interpolate(pred_d3, size=(256, 256), mode='bilinear', align_corners=False)
        pred_d4 = F.interpolate(pred_d4, size=(256, 256), mode='bilinear', align_corners=False)

        return pred, pred_d1, pred_d2, pred_d3, pred_d4


if __name__ == '__main__':
    x1 = torch.randn((2, 3, 256, 256))
    x2 = torch.randn((2, 3, 256, 256))
    model = Detector()
    pred, pred_d1, pred_d2, pred_d3, pred_d4 = model(x1, x2)
    print(f"pred:     {pred.shape}")
    print(f"pred_d1:  {pred_d1.shape}")
    print(f"pred_d2:  {pred_d2.shape}")
    print(f"pred_d3:  {pred_d3.shape}")
    print(f"pred_d4:  {pred_d4.shape}")
