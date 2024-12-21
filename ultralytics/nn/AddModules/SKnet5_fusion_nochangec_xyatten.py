import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

__all__ = ["MF6"]
def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
class DSConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block with Leaky ReLU activation.
    """
    def __init__(self, in_channel, out_channel, negative_slope=0.01):
        super(DSConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x
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
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
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
"Coordinate Attention for Efficient Mobile Network Design"
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x

        B, C, H, W = x.size()
        x_h = self.pool_h(x)  # 压缩水平方向: (B, C, H, W) --> (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 压缩垂直方向: (B, C, H, W) --> (B, C, 1, W) --> (B,C,W,1)

        # 坐标注意力生成
        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的向量: (B,C,H+W,1)
        y = self.conv1(y)  # 通过Conv进行变换,并降维: (B,C,H+W,1)--> (B,d,H+W,1)
        y = self.bn1(y)  # BatchNorm操作: (B,d,H+W,1)
        y = self.relu(y)  # Relu操作: (B,d,H+W,1)

        x_h, x_w = torch.split(y, [H, W], dim=2)  # 沿着空间方向重新分割为两部分: (B,d,H+W,1)--> x_h:(B,d,H,1); x_w:(B,d,W,1)
        x_w = x_w.permute(0, 1, 3, 2)  # x_w: (B,d,W,1)--> (B,d,1,W)

        a_h = self.conv_h(x_h).sigmoid()  # 恢复与输入相同的通道数,并生成垂直方向的权重: (B,d,H,1)-->(B,C,H,1)
        a_w = self.conv_w(x_w).sigmoid()  # 恢复与输入相同的通道数,并生成水平方向的权重: (B,d,1,W)-->(B,C,1,W)

        out = identity * a_w * a_h  # 将垂直、水平方向权重应用于输入,从而反映感兴趣的对象是否存在于相应的行和列中: (B,C,H,W) * (B,C,1,W) * (B,C,H,1) = (B,C,H,W)

        return out

class CMD(nn.Module):
    def __init__(self, channels):
        super(CMD, self).__init__()
        # self.co1 = CoordAtt(channels, channels)
        self.co2 = CoordAtt(channels, channels)

    def forward(self, F_vi, F_ir):
        # 计算视觉与红外特征差分
        # sub_vi_ir =F_vi - F_ir.repeat(1, 3, 1, 1)
        # F_dvi = self.co1(sub_vi_ir)
        # 计算红外与视觉特征差分
        sub_ir_vi = F_ir.repeat(1, 3, 1, 1) - F_vi
        F_dir = self.co2(sub_ir_vi)
        # 生成融合特征
        F_fvi = F_vi + F_dir #  F_dir变为48
        # F_fir = F_ir + F_dvi[:, :16, :, :]
        F_fir = F_ir + F_dir[:, :16, :, :]
        return F_fvi, F_fir

class MF6(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(MF6, self).__init__()
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        # self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        # self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        # self.bottleneck1 = DSConvBlock(channels, 16)
        # self.bottleneck2 = DSConvBlock(channels, 48)
        self.cmd = CMD(48)
        self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
        self.se = SE_Block(64, 16)

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