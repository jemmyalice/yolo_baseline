import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

__all__ = ["MF11"]
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

class SE_Block1(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=(3 - 1) // 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=(1 - 1) // 2)
        self.gap1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        y1 = self.conv(x)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y1 = self.gap(y1).view(b, c, 1, 1)
        y2 = self.conv1(x)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y2 = self.gap1(y2).view(b, c, 1, 1)
        y = y + y1 + y2
        return x * y.expand_as(x) # 注意力作用每一个通道上

class MF11(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(MF11, self).__init__()
        self.catconvA = nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True)
        self.catconvB = nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True)
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        # self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        # self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
        self.se = SE_Block(64, 16)
        self.se_r = SE_Block1(3,3)
        self.se_i = SE_Block1(3,3)
        # self.se_i = SE_Block(1,1)

    def forward(self, x, y):  # B * C * H * W #x_   left, x_right
        x_left_ori, x_right_ori = x, y
        b, c, h, w = x_left_ori.shape
        x_left = self.se_r(x_left_ori)
        x_right = self.se_i(x_right_ori)
        x_left = x_left * 0.5
        x_right = x_right * 0.5
        # x_left = x_left_ori * 0.5
        # x_right = x_right_ori * 0.5

        #########start
        # x_diff = x_left - x_right.expend_as(x_left)
        x_diff = x_right - x_left
        x_diffA = self.catconvA((torch.cat([x_diff, x_left], dim=1)))
        x_diffB = self.catconvB((torch.cat([x_diff, x_right], dim=1)))
        x_mask_left = torch.mul(self.mask_map_r(x_diffA).repeat(1, 3, 1, 1), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_diffB), x_right)

        #########end
        # x_mask_left = torch.mul(self.mask_map_r(x_left).repeat(1, 3, 1, 1), x_left)
        # x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))
        # import scipy.io as sio
        # sio.savemat('features/output.mat', mdict={'data':out.cpu().numpy()})

        return out