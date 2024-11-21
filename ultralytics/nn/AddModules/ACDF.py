import torch
from torch import nn
from torch.nn.parameter import Parameter

__all__ = ['ACDF', 'eca_layer']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 卷积操作中的 padding 通过 autopad 函数计算，
        # 意味p为None时，代码将自动计算填充数，p = (k-1)*d/2，确保输入和输出特征图大小一致。
        # 这种自适应填充在目标检测中较为常见，用来灵活适配不同大小的卷积核。
        # groups和dilation用于 分组卷积和空洞卷积 为1则是普通卷积
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # act是激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # nn.Identity()，相当于跳过激活层。这种设计提升了模块的灵活性，可以适应不同的实验需求。

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    # forward_fuse 是专门为推理阶段准备的。
    # 通常在模型推理时可以将卷积和批归一化合并，减少计算量。
    # 这个方法略过了 BatchNorm 层，将批归一化与卷积权重融合。减少计算量，提高推理速度
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class eca_layer(nn.Module):
    """
    Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ACDF(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=None):
        super(ACDF, self).__init__()
        self.eca1 = eca_layer(in_channel, k_size=k)
        self.eca2 = eca_layer(in_channel, k_size=k)
        self.conv1 = Conv(in_channel, in_channel, k=3, s=1, p=1)
        self.conv2 = Conv(in_channel, in_channel, k=3, s=1, p=1)

        self.cout1 = Conv(in_channel, out_channel, k=3, s=1, p=1)
        self.cout2 = Conv(in_channel, out_channel, k=3, s=1, p=1)

    def forward(self, irx, rgbx):
        irxa = self.eca1(irx)
        rgbxa = self.eca2(rgbx)

        irxa_c = self.conv1(irxa)
        rgbxa_c = self.conv2(rgbxa)

        # conv和attention element mutilply And add
        irx1 = irxa * irxa_c
        rgbx1 = rgbxa * rgbxa_c
        irx1 = irx + irx1
        rgbx1 = rgbx + rgbx1

        irx1 = self.cout1(irx1)
        rgbx1 = self.cout2(rgbx1)

        return torch.cat([irx1, rgbx1], dim = 1)