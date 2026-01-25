import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BasicUNet import BasicUNet

from addNoise import corrupt



# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# 下载数据集
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
# 数据加载器
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
x, y = next(iter(train_dataloader))

# 设置训练运行周期
n_epochs = 3

# 创建网络
net = BasicUNet()
net.to(device)  # 递归地将模型的所有参数和缓冲区（如权重和偏置）移动到目标设备上

# 指定损失函数
loss_fn = nn.MSELoss()

# 指定优化器
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# 记录训练过程中的损失
losses = []

# 训练
for epoch in range(n_epochs):
    for x, y in train_dataloader:

        # 获取数据并退化处理
        x = x.to(device)    # 将数据加载到GPU
        noise_amount = torch.rand(x.shape[0]).to(device)    # 随机选取噪声量
        noisy_x = corrupt(x, noise_amount)  # 创建带噪的输入

        # 得到预测结果
        pred = net(noisy_x)

        # 计算损失函数
        loss = loss_fn(pred, x)

        # 反向传播并更新参数
        opt.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算当前批次的梯度
        opt.step()  # 更新模型参数

        # 记录损失
        losses.append(loss.item())
    
    # 输出在每个周期训练得到的损失的均值 
    avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

# 查看损失曲线
plt.plot(losses)
plt.ylim(0, 0.1)
plt.show()

# 查看输入、加噪、输出对比图像
x, y = next(iter(train_dataloader))
x = x[:8]   # 选择前8条数据

# 在(0，1)区间选择退化量
amount = torch.linspace(0, 1, x.shape[0])
noised_x = corrupt(x, amount)

# 得到预测结果
with torch.no_grad():
    pred = net(noised_x.to(device)).detach().cpu()


# # 绘图
# # 绘制输入数据和加入噪声后的样本
# fig, axs = plt.subplots(3, 1, figsize=(12, 8))
# # 原始数据
# axs[0].set_title('Input data')
# axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0,1), cmap='Greys')
# # 加入噪声后的数据
# axs[1].set_title('Corrupted data (-- amount increases -->)')
# axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0,1), cmap='Greys')
# # 预测输出
# axs[2].set_title('Network Predictions')
# axs[2].imshow(torchvision.utils.make_grid(pred)[0].clip(0,1), cmap='Greys')
# plt.show()

# 采样处理
n_steps = 5
x = torch.rand(8, 1, 28, 28).to(device)
step_history = [x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():
        pred = net(x)
    pred_output_history.append(pred.detach().cpu())
    mix_factor = 1/(n_steps-i)
    x = x*(1-mix_factor) + pred*mix_factor
    step_history.append(x.detach().cpu())

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
axs[0, 0].set_title('x (model input)')
axs[0, 1].set_title('model prediction')
for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0,1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0,1), cmap='Greys')
plt.show()
