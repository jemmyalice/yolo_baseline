import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict
import math

__all__ = ['ACDFBlock']


class ECA(nn.Module):

    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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
        y=self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y=y.squeeze(-1).permute(0,2,1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y=self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y=self.sigmoid(y) # 生成权重表示: (B,1,C)
        y=y.permute(0,2,1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return x*y.expand_as(x)  # 权重对输入的通道进行重新加权: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)

class ESA(nn.Module):

    def __init__(self, kernel_size=3):
        super(ESA, self).__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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
        y=self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y=y.squeeze(-1).permute(0,2,1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y=self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y=self.sigmoid(y) # 生成权重表示: (B,1,C)
        y=y.permute(0,2,1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return x*y.expand_as(x)  # 权重对输入的通道进行重新加权: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)

class CDM(nn.Module):
    def __init__(self, in_channels):
        """
        Cross-Modal Difference Module (CDM)
        Args:
            in_channels (int): Number of input channels.
        """
        super(CDM, self).__init__()
        self.global_pool_x = nn.AdaptiveAvgPool2d((1, None))  # Global Pool in X-direction
        self.global_pool_y = nn.AdaptiveAvgPool2d((None, 1))  # Global Pool in Y-direction
        # Two parallel convolutions for visible and infrared inputs
        self.shared_conv_visible = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # Path for visible
        self.shared_conv_infrared = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # Path for infrared
        self.sigmoid = nn.Sigmoid()

    def forward(self, visible, infrared):
        # Compute the difference between the two modalities
        diff_visible = visible - infrared  # Difference for visible
        diff_infrared = infrared - visible  # Difference for infrared

        # Perform global pooling for both visible and infrared differences
        pool_x_visible = self.global_pool_x(diff_visible)
        pool_y_visible = self.global_pool_y(diff_visible)

        pool_x_infrared = self.global_pool_x(diff_infrared)
        pool_y_infrared = self.global_pool_y(diff_infrared)

        # Apply shared convolutions to each modality's difference
        conv_visible = self.shared_conv_visible(pool_x_visible + pool_y_visible)
        conv_infrared = self.shared_conv_infrared(pool_x_infrared + pool_y_infrared)

        # Combine the results of both paths
        combined = conv_visible + conv_infrared  # Element-wise addition
        attention = self.sigmoid(combined)  # Apply sigmoid to generate attention weights

        # Enhance the difference features using the attention weights
        enhanced = attention * (diff_visible + diff_infrared)

        # Return the final fused features (visible + enhanced difference)
        fused_visible = visible + enhanced
        fused_infrared = infrared + enhanced
        return fused_visible, fused_infrared

class ACDFBlock(nn.Module):
    def __init__(self, rgb_channels=3, ir_channels=1, reduction=16, fused_channels=64):
        """
        ACDF Block: Combines ESA and CDM, followed by Concat and ECA.
        Args:
            rgb_channels (int): Number of channels in RGB input.
            ir_channels (int): Number of channels in IR input.
            reduction (int): Reduction ratio for ESA.
            fused_channels (int): Number of channels after fusion.
        """
        super(ACDFBlock, self).__init__()
        # Convolutional layers to match the channel sizes
        self.rgb_preconv = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False)
        self.ir_preconv = nn.Conv2d(ir_channels, ir_channels, kernel_size=1, bias=False)
        self.esa_visible = ESA(fused_channels, reduction)
        self.esa_infrared = ESA(fused_channels, reduction)
        self.final_conv_rgb = nn.Conv2d(rgb_channels, fused_channels // 2, kernel_size = 3, padding = 1, bias = False)
        self.final_conv_ir = nn.Conv2d(ir_channels, fused_channels // 2, kernel_size = 3, padding = 1, bias = False)
        self.cdm = CDM(fused_channels)
        self.eca = ECA(fused_channels)  # ECA layer after concatenation

    def forward(self, visible, infrared):
        #esa 输出
        esa_visible = self.esa_visible(visible)
        esa_infrared = self.esa_infrared(infrared)

        # 输出之后经过了conv
        enhanced_visible = self.rgb_preconv(esa_visible)  # Apply RGB convolution
        enhanced_infrared = self.ir_preconv(esa_infrared)  # Apply IR convolution

        # 对应元素乘， 尺寸要一样
        enhanced_visible = esa_visible * enhanced_visible
        enhanced_infrared = esa_infrared * enhanced_infrared
        enhanced_visible = enhanced_visible + visible
        enhanced_infrared = enhanced_infrared + infrared

        enhanced_visible = self.final_conv_rgb(enhanced_visible)
        enhanced_infrared = self.final_conv_ir(enhanced_infrared)

        # Fuse the enhanced features using CDM
        fused_visible, fused_visible = self.cdm(enhanced_visible, enhanced_infrared)

        # Concatenate the original visible and infrared inputs with the fused features
        concat_features = torch.cat([fused_visible, fused_visible], dim=1)  # Concatenate along channels

        # Apply ECA after concatenation
        final_output = self.eca(concat_features)

        return final_output

if __name__ == '__main__':
    # (B, C, H, W)
    input=torch.randn(1,512,7,7)
    Model = ECA(kernel_size=3)
    output=Model(input)
    print(output.shape)