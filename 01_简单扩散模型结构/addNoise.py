import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 下载训练用的数据集
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

# 创建一个数据加载器
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_loader))
print('Input shape:', x.shape)
print('Labels:', y)

# # 可视化样本，检查数据加载器中样本的形状与标签是否正确
# plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Grays')
# plt.title('Original MNIST Images')
# plt.show()

"""根据amount为输入x加入噪声，即退化过程"""
def corrupt(x, amount):
    noise = torch.rand_like(x)              # 用来控制输入的噪声量的参数
    amount = amount.view(-1, 1, 1, 1)       # 调整amount的形状，以匹配x的形状，以保证广播机制不出错
    return x*(1-amount) + noise*amount

# 加入噪声
amount = torch.linspace(0, 1, x.shape[0])    # 从0到1增加噪声，退化得更强烈
noisy_x = corrupt(x, amount)

# # 绘制输入数据和加入噪声后的样本
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# # 原始数据
# axs[0].set_title('Input data')
# axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

# # 加入噪声后的数据
# axs[1].set_title('Corrupted data (-- amount increases -->)')
# axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0], cmap='Greys')

# plt.show()
