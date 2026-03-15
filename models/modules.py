import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.conv_tasnet import ConvBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class SeparableConv1d(nn.Module):
    """Separate convolution layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        padding (int): Padding size
        bias (bool): bias or not

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(SeparableConv1d, self).__init__()
        self.depth_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, bias=bias)
        self.point_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)

        return out


class PAEM(nn.Module):
    """Pivotal-Feature Adaptive Enhancement Module

    Args:
        in_channels (int): Number of input channels

    """

    def __init__(self, in_channels):
        super(PAEM, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.fc2 = nn.Linear(in_channels // 2, in_channels, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1, bias=False)
        self.sigmoid2 = nn.Sigmoid()



    def forward(self, x):
        spatial_scale = self.sigmoid1(self.conv(x))
        out = x * spatial_scale

        channel_squeeze = self.pool(out)
        channel_squeeze = self.sigmoid2(self.fc2(self.relu(self.fc1(channel_squeeze.permute(0, 2, 1)))))
        channel_squeeze = channel_squeeze.permute(0, 2, 1)

        out = out * channel_squeeze

        return out


class EXT(nn.Module):
    def __init__(self, in_channels, out_channels, FF_channels=2048, dropout=0.1):
        super(EXT, self).__init__()
        self.linear1 = nn.Linear(in_channels, FF_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(FF_channels)

        self.linear2 = nn.Linear(FF_channels, out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(out_channels)

        self.Sep_conv1 = SeparableConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.Sep_dropout1 = nn.Dropout(dropout)
        self.Sep_norm1 = nn.LayerNorm(out_channels)

        self.Sep_conv2 = SeparableConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.Sep_dropout2 = nn.Dropout(dropout)
        self.Sep_norm2 = nn.LayerNorm(out_channels)

        self.combined_norm0 = nn.LayerNorm(out_channels * 2)
        self.combined_conv = nn.Conv1d(2 * out_channels, out_channels, kernel_size=3, padding=1)

        self.combined_norm1 = nn.LayerNorm(out_channels)
        self.combined_dropout1 = nn.Dropout(dropout)
        self.paem = PAEM(out_channels)
        self.combined_norm2 = nn.LayerNorm(out_channels)
        self.combined_dropout2 = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(out_channels)

        # 两种残差变换
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.projectio = nn.Linear(in_channels, out_channels)

        # self.bn = nn.BatchNorm1d(num_features=BN_DIM)
        # Add regularization term for weight decay
        # self.weight_decay = 1e-5

        self.pre_norm_feat = None  # 特征直方图临时变量

    def forward(self, x):
        # src = src.transpose(0, 1).transpose(1, 2)
        # (S, B, E) -> (B, E, S) 其中(S, B, E)是多头注意力的输出和输入

        # 原始x为(batch, seq_length, feature)
        temp_y1 = self.linear1(x)
        temp_y1 = F.gelu(self.norm1(temp_y1))
        temp_y1 = self.linear2(temp_y1)
        y1 = F.gelu(self.norm2(temp_y1))

        # 原始x为(batch,seq,feature)
        # 由于conv1d要求是(batch,feature,seq_length)所以需要转换
        conv_x = x.permute(0, 2, 1)
        temp_y2 = self.Sep_conv1(conv_x)
        temp_y2 = F.gelu(self.norm2(temp_y2.permute(0, 2, 1)))
        temp_y2 = self.Sep_dropout1(temp_y2)

        temp_y2 = temp_y2.permute(0, 2, 1)
        temp_y2 = self.Sep_conv2(temp_y2)
        temp_y2 = F.gelu(self.norm2(temp_y2.permute(0, 2, 1)))
        y2 = self.Sep_dropout2(temp_y2)
        # 特征已经被还原为原始的x(batch,seq,feature)

        concatenated = torch.cat((y1, y2), dim=2)
        # 这里应该有layernorm
        concatenated = self.combined_norm0(concatenated)
        # 后面PAE还是PAE与卷积的混合体有待研究，如果是PAE需要进行一步.permute(0, 2, 1)变换

        concatenated = concatenated.permute(0, 2, 1)
        combined = self.combined_conv(concatenated)
        combined = F.gelu(self.combined_norm1(combined.permute(0, 2, 1)))
        combined = self.combined_dropout1(combined)

        combined = combined.permute(0, 2, 1)
        combined = self.paem(combined)
        combined = F.gelu(self.combined_norm2(combined.permute(0, 2, 1)))
        combined = self.combined_dropout2(combined)

        if x.shape[-1] != combined.shape[-1]:
            x = x.permute(0, 2, 1)
            x = self.shortcut(x)
            x = x.permute(0, 2, 1)

        output = x + combined
        self.pre_norm_feat = output #直方图变量
        output = self.output_norm(output)

        return output  # 最终输出同原始的x(batch, seq_length, feature)


class IFFN(nn.Module):
    """Inverted FFN module

    Args:
        in_channels (int): Number of input channels
        drop(float): dropout probability

    """

    def __init__(self, in_channels, drop=0.0):
        super(IFFN, self).__init__()
        self.linear1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1)
        self.linear2 = nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)  # 这里进行了特征缩放
        self.conv = nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1,
                              groups=in_channels * 2)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

        self.drop_out = nn.Dropout(drop)

    def forward(self, x):
        out = self.gelu1(self.linear1(x.permute(0, 2, 1)))
        out = self.drop_out(out)

        out = self.gelu2(self.conv(out))
        out = self.linear2(out)
        out = self.drop_out(out)

        return out.permute(0, 2, 1)


class Attention(nn.Module):
    """Attention Neck module

    Args:
        out_channels (int): Number of output channels
        drop(float): dropout probability

    """

    def __init__(self, out_channels, drop=0.2):
        super(Attention, self).__init__()
        self.ln1 = nn.LayerNorm(out_channels)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=16, dropout=drop, add_bias_kv=True)
        self.ln2 = nn.LayerNorm(out_channels)
        #self.ffn = IFFN(out_channels, drop * 2)
        self.casa=CASAB(in_channels=out_channels)

    def forward(self, x):
        out = self.ln1(x)
        out = out.permute(1, 0, 2)  # (b,s,c)->(s,b,c)
        out, _ = self.attn(out, out, out)
        out2 = out.permute(1, 0, 2) + x  # (s,b,c)->(b,s,c)
        out = self.ln2(out2)  # (b,s,c)
        #out = self.ffn(out)  # (b,s,c)
        out = self.casa(out)
        out = out + out2

        return out


class CB(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(CB, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 2, 1) #b s c 输入 转 B  c  s 进入卷积

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)

        x = x.permute(0, 2, 1) #B c s 转 b s c

        return x


class ChannelAttention(nn.Module):
    """
    输入维度: (B, Seq, C)
    输出维度: (B, Seq, C)
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 平均池化：取每个通道的平均值
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 最大池化：取每个通道的最大值
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 两层全连接（用1x1卷积代替）实现通道权重的学习
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),  # SiLU 激活，平滑版 ReLU
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出范围 [0,1]，作为权重系数
        )


    def forward(self, x):
        # x: (B, Seq, C)
        # B, S, C = x.shape
        x=x.permute(0, 2, 1)  #b s c 转 b c s

        # 平均池化结果送入全连接
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化结果送入全连接
        max_out = self.fc(self.max_pool(x))
        # 两个结果相加，形成最终的通道注意力
        scale = avg_out + max_out
        # 输入特征乘以注意力权重

        x =x * scale
        x = x.permute(0, 2, 1)

        return x  #BCS->B S C


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=7, padding=3, groups=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x=x.permute(0, 2, 1) # b s c -> BCS

        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        sum_out = torch.sum(x, dim=1, keepdim=True)

        pool = torch.cat((mean_out, max_out, min_out, sum_out), dim=1)

        attention = self.conv(pool)
        x * attention

        x = x.permute(0, 2, 1) # BCS -> BSC

        return x


class CASAB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CASAB, self).__init__()
        self.convblock = CB(in_channels=in_channels, out_channels=in_channels)
        self.chanel_attention = ChannelAttention(in_channels=in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.convblock(x)
        ca = self.chanel_attention(x)
        sa = self.spatial_attention(x)
        return ca + sa


class Classify(nn.Module):
    """Classify Head module

    Args:
        out_channels (int): Number of output channels
        drop(float): dropout probability

    """

    def __init__(self, in_channels=256, out_channels=2, drop=0.5):
        super(Classify, self).__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(in_channels, out_channels)  # 是否为隐写为两个类别

    def forward(self, x):
        out = self.pooling(x.permute(0, 2, 1)).squeeze(2)  # bsc->bcs
        out = self.dropout(out)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out


class CVIM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        # 1D conv → 序列维度 T 做局部卷积
        self.l_proj1 = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=1, padding=0, bias=True),
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        )
        self.r_proj1 = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=1, padding=0, bias=True),
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        )

        self.l_proj2 = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=1, padding=0, bias=True),
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=1, padding=0, bias=True),
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        )

        self.l_proj3 = nn.Conv1d(c, c, kernel_size=1, padding=0)
        self.r_proj3 = nn.Conv1d(c, c, kernel_size=1, padding=0)

    def forward(self, x_l, x_r):
        """
        x_l, x_r: (B, T, C)
        """

        # Convert to (B, C, T) for conv1d
        x_l = x_l.permute(0, 2, 1)
        x_r = x_r.permute(0, 2, 1)

        # Q: (B, C, T) → (B, T, C)
        Q_l = self.l_proj1(x_l).permute(0, 2, 1)
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1).transpose(1, 2)

        # V: (B, T, C)
        V_l = self.l_proj2(x_l).permute(0, 2, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 1)

        # Attention (B, T, C) × (B, C, T) = (B, T, T)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        # Cross-view attention
        F_r2l = torch.matmul(F.softmax(attention, dim=-1), V_r)  # (B, T, C)
        F_l2r = torch.matmul(F.softmax(attention.transpose(1, 2), dim=-1), V_l)

        # back to (B, C, T)
        F_r2l = self.l_proj3(F_r2l.permute(0, 2, 1))
        F_l2r = self.r_proj3(F_l2r.permute(0, 2, 1))

        # output (B, T, C)
        out_l = (x_l + F_r2l).permute(0, 2, 1)
        out_r = (x_r + F_l2r).permute(0, 2, 1)

        return out_l, out_r
