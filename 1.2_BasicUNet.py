import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 下采样，三层卷积
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        # 上采样，三层卷积
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.ReLU()  # 激活函数
        self.downscale = nn.MaxPool2d(2)  # 下采样，最大池化层实现空间分辨率的减半
        self.upscale = nn.Upsample(scale_factor=2)  # 双线性插值的上采样层，将空间分辨率放大两倍

    def forward(self, x):
        h=[]
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
                h.append(x)  # 累积保存下采样过程中的特征，用于后续的跳跃连接
                x = self.downscale(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()    # 跳跃连接，将编码器的高分辨率特征（输入）与解码器的低分辨率特征（上采样后的特征）结合，增强解码器的细节恢复能力（保留输入的原始信息，避免信息丢失）
            x = self.act(l(x))

        return x
    
# 测试BasicUNet
net = BasicUNet()
x = torch.rand(8, 1, 28, 28)  # 输入形状为(批量大小, 通道数, 高度, 宽度)
print(net(x).shape)           # 输出应与输入形状相同
print(sum([p.numel() for p in net.parameters()]))  # 输出模型参数总数
