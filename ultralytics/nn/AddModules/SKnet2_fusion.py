import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

__all__ = ["MF3"]
def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)  # (1, 3, 4, 4) ->(1, 3, 1, 1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
    def forward(self, F_vi, F_ir):
        # 计算视觉与红外特征差分
        sub_vi_ir =F_vi - F_ir.repeat(1, 3, 1, 1)
        sub_w_vi_ir = self.avg_pool1(sub_vi_ir)  # Global Average Pooling
        w_vi_ir = torch.sigmoid(sub_w_vi_ir)
        # 计算红外与视觉特征差分
        sub_ir_vi = F_ir.repeat(1, 3, 1, 1) - F_vi
        sub_w_ir_vi = self.avg_pool2(sub_ir_vi)  # Global Average Pooling
        w_ir_vi = torch.sigmoid(sub_w_ir_vi)
        # 加权差分特征
        F_dvi = w_vi_ir * sub_ir_vi  # 放大差分信号
        F_dir = w_ir_vi * sub_vi_ir
        # 生成融合特征
        F_fvi = F_vi + F_dir
        F_fir = F_ir + F_dvi[:, :16, :, :]
        return F_fvi, F_fir

class MF3(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(MF3, self).__init__()
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        # self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        # self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
        self.se = SE_Block(64, 16)
        self.cmd = CMD()
        self.se_r = SE_Block(3, 3)
        self.se_i = SE_Block(3, 3)
        # self.se_i = SE_Block(1,1)

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
        # x_diff = x_right - x_left
        # x_diffA = self.catconvA((torch.cat([x_diff, x_left], dim=1)))
        # x_diffB = self.catconvB((torch.cat([x_diff, x_right], dim=1)))
        # x_mask_left = torch.mul(self.mask_map_r(x_diffA).repeat(1, 3, 1, 1), x_left)
        # x_mask_right = torch.mul(self.mask_map_i(x_diffB), x_right)
        #########end
        x_mask_left = torch.mul(self.mask_map_r(x_left).repeat(1, 3, 1, 1), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB

        #########start
        out_RGB, out_IR = self.cmd(out_RGB, out_IR)

        out = self.se(torch.cat([out_RGB, out_IR], 1))
        # import scipy.io as sio
        # sio.savemat('features/output.mat', mdict={'data':out.cpu().numpy()})

        return out