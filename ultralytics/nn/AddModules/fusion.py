import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SEAttention', 'FusionNetwork']


# SAM模块：对两个通道进行相减后使用某种操作（比如简单卷积或差异增强）生成新的特征
class SAM_Module(nn.Module):
    def __init__(self):
        super(SAM_Module, self).__init__()
        # 7x7卷积层处理拼接后的池化结果
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.norm = nn.BatchNorm2d(1)  # 添加归一化层

    def forward(self, x1, x2):
        f1 = x1 - x2
        # 全局最大池化：对每个像素点的通道进行池化，返回最大值
        max_pool = torch.max(f1, dim=1, keepdim=True)[0]  # 对每个像素点的通道维度进行最大池化
        # 全局平均池化：对每个像素点的通道进行池化，返回平均值
        avg_pool = torch.mean(f1, dim=1, keepdim=True)  # 对每个像素点的通道维度进行平均池化
        # 拼接最大池化和平均池化的结果
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # 拼接后的形状: [batch_size, 2, height, width]
        # 通过卷积处理拼接后的结果
        f1 = self.conv(pooled)  # 输出形状: [batch_size, 1, height, width]
        f1 = self.norm(f1)  # 添加归一化
        # 将卷积结果加回原始图像（对r1和i1分别加）
        result_r = x1 + f1
        result_i = x2 + f1
        return result_r, result_i


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
        self.sam = SAM_Module()
        self.se_rgb = InSEAttention(in_channels_rgb)
        self.se_ir = InSEAttention(in_channels_ir)
        self.conv_rgb = nn.Conv2d(in_channels_rgb, in_channels_rgb, kernel_size=3, stride=1, padding=1)
        self.norm_rgb = nn.BatchNorm2d(in_channels_rgb)  # 添加归一化层
        self.conv_ir = nn.Conv2d(in_channels_ir, in_channels_ir, kernel_size=3, stride=1, padding=1)
        self.norm_ir = nn.BatchNorm2d(in_channels_ir)  # 添加归一化层
        self.finalrgb_conv = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.finalrgb_norm = nn.BatchNorm2d(out_channels // 2)  # 添加归一化层
        self.finalir_conv = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.finalir_norm = nn.BatchNorm2d(out_channels // 2)  # 添加归一化层
        self.se_final = InSEAttention(out_channels)
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