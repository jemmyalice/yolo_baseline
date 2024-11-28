import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块
import torch.nn.functional as F

__all__ = ['FusionNetwork']

class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=1):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)

        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果

        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SEAttention, self).__init__()
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
        # SE注意力模块
        self.ema_rgb = EMA(in_channels_rgb)
        self.ema_ir = EMA(in_channels_ir)
        # 卷积操作
        self.conv_rgb = nn.Conv2d(in_channels_rgb, in_channels_rgb, kernel_size=3, stride=1, padding=1)
        self.conv_ir = nn.Conv2d(in_channels_ir, in_channels_ir, kernel_size=3, stride=1, padding=1)
        # 输出融合模块
        self.final_conv_rgb = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.final_conv_ir = nn.Conv2d(in_channels_ir, out_channels // 2, kernel_size=3, stride=1, padding=1)

        self.se_final = SEAttention(out_channels)
        # 初始化卷积层和全连接层
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # 对SAM模块的卷积层进行初始化（SiLU激活）
            if isinstance(m, nn.Conv2d) and isinstance(m, EMA):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')  # SiLU激活
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对SE模块的卷积层进行初始化（ReLU激活）
            if isinstance(m, nn.Conv2d) and isinstance(m, SEAttention):
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
        #ema注意力机制
        r2 = self.ema_rgb(r1)  # RGB图像
        i2 = self.ema_ir(i1)  # 红外图像

        #卷积
        r3 = self.conv_rgb(r2)
        i3 = self.conv_ir(i2)

        r3 = r3 * r2
        i3 = i3 * i2

        r3 = r3 + r1
        i3 = i3 + i1

        # 最后卷积
        r4 = self.final_conv_rgb(r3)
        i4 = self.final_conv_ir(i3)

        #特征拼接和SE注意力
        fused = torch.cat((r4, i4), dim=1)  # 拼接两个通道的特征图
        fused = self.se_final(fused)  # SE注意力

        return fused

# 初始化网络
class FusionNetwork(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ir, out_channels):
        super(FusionNetwork, self).__init__()
        self.dual_stream_fusion = DualStreamFusion(in_channels_rgb, in_channels_ir, out_channels)

    def forward(self, r1, i1):
        return self.dual_stream_fusion(r1, i1)