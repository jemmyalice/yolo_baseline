import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

__all__ = ["MF2"]
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

class SKAttention(nn.Module):
    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=8,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        # 有几个卷积核,就有几个尺度, 每个尺度对应的卷积层由Conv-bn-relu实现
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        # 将全局向量降维
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)
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
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs=[]
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  #每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats=torch.stack(conv_outs,0) # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U=sum(conv_outs) #(K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S=U.mean(-1).mean(-1)
        # 降低通道数,提高计算效率: (B,C)-->(B,d)
        Z=self.fc(S)

        # 将紧凑特征Z通过K个全连接层得到K个尺度对应的通道描述符表示, 然后基于K个通道描述符计算注意力权重
        weights=[]
        for fc in self.fcs:
            weight=fc(Z) #恢复预输入相同的通道数: (B,d)-->(B,C)
            weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        scale_weight=torch.stack(weights,0) #将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight=self.softmax(scale_weight) #在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V=(scale_weight*feats).sum(0) # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V

class MF2(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(MF2, self).__init__()
        self.catconvA = dsconv_3x3(channels * 2, channels)
        self.catconvB = dsconv_3x3(channels * 2, channels)
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        # self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        # self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
        self.se = SE_Block(64, 16)
        self.sk_r = SKAttention(3,reduction = 3)
        self.sk_i = SKAttention(3,reduction = 3)
        # self.se_i = SE_Block(1,1)

    def forward(self, x, y):  # B * C * H * W #x_   left, x_right
        x_left_ori, x_right_ori = x, y
        b, c, h, w = x_left_ori.shape
        x_left = self.sk_r(x_left_ori)
        x_right = self.sk_i(x_right_ori)
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
        out = self.se(torch.cat([out_RGB, out_IR], 1))
        # import scipy.io as sio
        # sio.savemat('features/output.mat', mdict={'data':out.cpu().numpy()})

        return out