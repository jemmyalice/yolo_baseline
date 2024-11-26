import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SEAttention', 'FusionNetwork']

# SAM模块：对两个通道进行相减后使用某种操作（比如简单卷积或差异增强）生成新的特征
class SAM_Module(nn.Module):
    def __init__(self):
        super(SAM_Module, self).__init__()

    def forward(self, x1, x2):
        # x1 - x2是两个输入图像的差异图，模拟一些特征增强
        f1 = x1 - x2
        # 对差异图进行某种处理（假设是卷积处理）
        f1 = F.relu(f1)
        # 返回处理后的两个图（这里直接使用相同的f1做示范）
        return f1, f1

# SEAttention模块：自适应通道注意力机制
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SEAttention, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):  # 为 SEAttention 的全连接层添加初始化逻辑
                nn.init.xavier_normal_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # 通道加权
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class InSEAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super(InSEAttention, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # 通道加权
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class DualStreamFusion(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ir, out_channels):
        super(DualStreamFusion, self).__init__()
        # SAM模块
        self.sam = SAM_Module()
        # SE注意力模块
        self.se_rgb = InSEAttention(in_channels_rgb)
        self.se_ir = InSEAttention(in_channels_ir)
        # 卷积操作
        self.conv_rgb = nn.Conv2d(in_channels_rgb, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_ir = nn.Conv2d(in_channels_ir, out_channels, kernel_size=3, stride=1, padding=1)
        # 输出融合模块
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.se_final = InSEAttention(out_channels)
        # 初始化卷积层和全连接层
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # 对SAM模块的卷积层进行初始化（SiLU激活）
            if isinstance(m, nn.Conv2d) and isinstance(m, SAM_Module):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')  # SiLU激活
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对SE模块的卷积层进行初始化（ReLU激活）
            if isinstance(m, nn.Conv2d) and isinstance(m, InSEAttention):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # ReLU激活
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对SE模块的全连接层进行初始化（ReLU激活）
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, r1, i1):
        # 步骤 1: 通过SAM模块得到两个图（r2, i2）
        f1_r, f1_i = self.sam(r1, i1)
        # 步骤 2: SE注意力机制
        r3 = self.se_rgb(f1_r)  # RGB图像
        i3 = self.se_ir(f1_i)  # 红外图像
        # 步骤 3: 添加原始图像并卷积
        r3 = self.conv_rgb(r3 + r1)
        i3 = self.conv_ir(i3 + i1)
        # 步骤 4: 特征拼接和SE注意力
        fused = torch.cat((r3, i3), dim=1)  # 拼接两个通道的特征图
        fused = self.final_conv(fused)
        fused = self.se_final(fused)  # SE注意力

        return fused

# 初始化网络
class FusionNetwork(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ir, out_channels):
        super(FusionNetwork, self).__init__()
        self.dual_stream_fusion = DualStreamFusion(in_channels_rgb, in_channels_ir, out_channels)

    def forward(self, r1, i1):
        return self.dual_stream_fusion(r1, i1)