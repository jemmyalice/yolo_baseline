import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
# 没有dwconv也没有cdm，需要直接取消注释就行了,这个CDM是eca版本的
# inception
__all__ = ["MF9"]
# 没有dwconv也没有cdm，需要直接取消注释就行了,这个CDM是eca版本的
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
class bottleblock(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel,stride):
        super(bottleblock, self).__init__()
        self.midchannel=mid_channel
        output=out_channel-in_channel
        self.stride=stride

        self.pointwise_conv1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=mid_channel,kernel_size=1,stride=1,bias=False),
                                           nn.BatchNorm2d(mid_channel),
                                           nn.ReLU(inplace=True))
        self.depth_conv=nn.Sequential(nn.Conv2d(in_channels=mid_channel,out_channels=mid_channel,kernel_size=3,padding=1,stride=stride,groups=mid_channel,bias=False),
                                      nn.BatchNorm2d(mid_channel))
        self.pointwise_conv2=nn.Sequential(nn.Conv2d(in_channels=mid_channel,out_channels=output,kernel_size=1,stride=1,bias=False),
                                           nn.BatchNorm2d(output),
                                           nn.ReLU(inplace=True))
        if stride==2:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=3,padding=1,stride=stride,groups=in_channel,bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1,stride=1,bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.ReLU(inplace=True))
        else:
            self.shortcut=nn.Sequential()
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
    def forward(self,x):
        if self.stride==2:
            residual=self.shortcut(x)
            x=self.pointwise_conv1(x)
            x=self.depth_conv(x)
            x=self.pointwise_conv2(x)
            return torch.cat((residual,x),dim=1)
        elif self.stride==1:
            x1,x2=self.channel_shuffle(x)
            residual=self.shortcut(x2)
            x1=self.pointwise_conv1(x1)
            x1=self.depth_conv(x1)
            x1=self.pointwise_conv2(x1)
            return torch.cat((residual,x1),dim=1)
class SE_Block1(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, vis_out=0, ir_out=0, **kwargs):
        super(SE_Block1, self).__init__(**kwargs)
        # 4，单1x1卷积层
        self.p4_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        # 线路2，1x1卷积层后接3x3卷积层
        if vis_out != 0:
            self.p2_1 = bottleblock(in_channels, in_channels, vis_out, 1)
        elif ir_out != 0:
            self.p2_1 = bottleblock(in_channels, in_channels, ir_out, 1)
        # 线路1，3x3最大汇聚层后接1x1卷积层
        self.p1_1 = SE_Block(in_channels, in_channels)
    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_1(x)
        p4 = self.p4_1(x)
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p4), dim=1)
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
class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)  # (1, 3, 4, 4) ->(1, 3, 1, 1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.eca1 = ECAAttention()
        self.eca2 = ECAAttention()
    def forward(self, F_vi, F_ir):
        # 计算视觉与红外特征差分
        sub_vi_ir =F_vi - F_ir.repeat(1, 3, 1, 1)
        F_dvi = self.eca1(sub_vi_ir)
        # 计算红外与视觉特征差分
        sub_ir_vi = F_ir.repeat(1, 3, 1, 1) - F_vi
        F_dir = self.eca2(sub_ir_vi)
        # 生成融合特征
        F_fvi = F_vi + F_dir #  F_dir变为48
        F_fir = F_ir + F_dvi[:, :16, :, :]
        return F_fvi, F_fir

class MF9(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(MF9, self).__init__()
        self.mask_map_r = nn.Conv2d(channels*4, 1, 1, 1, 0, bias=True)
        # self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels*4, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        # self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bottleneck1 = nn.Conv2d(channels*4, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels*4, 48, 3, 1, 1, bias=False)
        self.se = SE_Block(64,16)
        # self.cmd = CMD()
        self.se_r = SE_Block1(4,vis_out = 48)
        self.se_i = SE_Block1(4,ir_out = 16)
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
        x_left = self.se_r(torch.concat([x_left_ori, (x_left_ori - x_right_ori)[:,:1,:,:]], dim = 1))
        x_right = self.se_i(torch.concat([x_right_ori, (x_right_ori - x_left_ori)[:,:1,:,:]], dim = 1))
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
        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori.repeat(1, 4, 1, 1))
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori.repeat(1, 4, 1, 1))  # RGB

        #########start
        # out_RGB, out_IR = self.cmd(out_RGB, out_IR)

        out = self.se(torch.cat([out_RGB, out_IR], 1))
        return out