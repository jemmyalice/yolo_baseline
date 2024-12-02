import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

__all__ = ['FusionNetwork']

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out

class DualStreamFusion(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ir, out_channels):
        super(DualStreamFusion, self).__init__()
        self.se_rgb = SEAttention(in_channels_rgb)
        self.se_ir = SEAttention(in_channels_ir)
        self.conv_rgb = nn.Conv2d(in_channels_rgb, in_channels_rgb, kernel_size=3, stride=1, padding=1)
        self.norm_rgb = nn.BatchNorm2d(in_channels_rgb)  # 添加归一化层
        self.conv_ir = nn.Conv2d(in_channels_ir, in_channels_ir, kernel_size=3, stride=1, padding=1)
        self.norm_ir = nn.BatchNorm2d(in_channels_ir)  # 添加归一化层
        self.finalrgb_conv = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.finalrgb_norm = nn.BatchNorm2d(out_channels // 2)  # 添加归一化层
        self.finalir_conv = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.finalir_norm = nn.BatchNorm2d(out_channels // 2)  # 添加归一化层
        self.se_final = SEAttention(out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # 添加BatchNorm初始化
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, r1, i1):
        f1_r, f1_i = self.sam(r1, i1)
        r3 = self.se_rgb(f1_r)
        i3 = self.se_ir(f1_i)
        r4 = self.conv_rgb(r3)
        r4 = self.norm_rgb(r4)  # 添加归一化
        i4 = self.conv_ir(i3)
        i4 = self.norm_ir(i4)  # 添加归一化

        r4 = r3 * r4
        i4 = i3 * i4
        r4 = r4 + r1
        i4 = i4 + i1

        r4 = self.finalrgb_conv(r4)
        r4 = self.finalrgb_norm(r4)  # 添加归一化
        i4 = self.finalir_conv(i4)
        i4 = self.finalir_norm(i4)  # 添加归一化
        fused = torch.cat((r4, i4), dim=1)
        fused = self.se_final(fused)
        return fused

# 初始化网络
class FusionNetwork(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ir, out_channels):
        super(FusionNetwork, self).__init__()
        self.dual_stream_fusion = DualStreamFusion(in_channels_rgb, in_channels_ir, out_channels)

    def forward(self, r1, i1):
        return self.dual_stream_fusion(r1, i1)