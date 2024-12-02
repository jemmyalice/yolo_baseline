import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块
import torch.nn.functional as F

__all__ = ['My_add']

class My_add(nn.Module):
    def __init__(self):
        super(My_add, self).__init__()
    def forward(self, x, y):
        return x + y.expand_as(x)
