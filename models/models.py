from distutils.command.sdist import show_formats

import torch.nn as nn
from pyexpat import features
from torch import device
from torch.nn import MultiheadAttention

from models.modules import *

# test code
import numpy as np


class BIEN(nn.Module):
    def __init__(self, length):
        super(BIEN, self).__init__()
        self.embedding = nn.Embedding(40, 64)
        self.position_embedding = PositionalEncoding(64, max_len=int(length * 50))

        self.OriginalBackbone = nn.ModuleList([
            EXT(64, 128),
            EXT(128, 128),
            EXT(128, 256),
            EXT(256, 256),
        ])

        self.CalibrationBackbone = nn.ModuleList([
            EXT(64, 128),
            EXT(128, 128),
            EXT(128, 256),
            EXT(256, 256),
        ])

        self.CVIM = nn.ModuleList([
            CVIM(128),
            CVIM(128),
            CVIM(256),
            CVIM(256),
        ])

        self.backbone_R = None  # 骨干网占位变量
        self.backbone_S = None
        self.backbone_output = None

        self.Neck = Attention(512, 0.3)

        self.Head = Classify(512, 2)

    def forward(self, row_Original, row_Calibration, return_tsne_feat=False):
        # 嵌入和位置编码

        em_Original = self.embedding(row_Original)
        em_Calibration = self.embedding(row_Calibration)

        Original = self.position_embedding(em_Original)
        Calibration = self.position_embedding(em_Calibration)

        # 调整形状
        Original = Original.view(Original.size(0), -1, Original.size(3))
        Calibration = Calibration.view(Calibration.size(0), -1, Calibration.size(3))

        # 初始化中间变量
        r, s = Original, Calibration

        # 使用循环处理四个阶段
        for i in range(4):
            # 通过各自的主干网络
            r = self.OriginalBackbone[i](r)
            s = self.CalibrationBackbone[i](s)
            r, s = self.CVIM[i](r, s)

        backbone_out = torch.cat((r, s), dim=2)
        self.backbone_R = r
        self.backbone_S = s
        self.backbone_output = backbone_out

        # 此时的c、r、s是(b,s,c)的组织形式,数据将通过Attention注意力头
        Neck_out = self.Neck(backbone_out)

        # 经过注意力neck后c、r、s仍为是(b,s,c)的组织形式
        result = self.Head(Neck_out)

        if return_tsne_feat:
            feat = Neck_out.mean(dim=1)  # (B, C)
            return result, feat
        else:
            return result
