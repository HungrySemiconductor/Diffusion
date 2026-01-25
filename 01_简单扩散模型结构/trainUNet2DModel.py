import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from diffusers import UNet2DModel
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
net = UNet2DModel(
    sample_size=28,          # 目标图像的分辨率
    in_channels=1,          # 输入通道数，RGB图像通道数为3，灰度图像通道数为1
    out_channels=1,         # 输出图像的通道数  
    layers_per_block=2,     # 设置在一个UNet块中使用多少个ResNet层
    block_out_channels=(32, 64, 64), # 每个UNet块的输出通道数，与BasicNet基本相同
    down_block_types=(
        "DownBlock2D",      # 标准的ResNet下采样模块
        "AttnDownBlock2D",  # 带有空域维度self-att的ResNet下采样模块
        "AttnDownBlock2D", 
    ),
    up_block_types=(
        "AttnUpBlock2D",    
        "AttnUpBlock2D",    # 带有空域维度self-att的ResNet上采样模块
        "UpBlock2D",        # 标准的ResNet上采样模块
    ),
)
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
        
        # 将噪声量转换为时间步长（假设噪声量0对应时间步长0，噪声量1对应最大时间步长）
        # 通常时间步长是整数，这里我们需要根据噪声量计算对应的时间步长
        # 假设我们有1000个时间步长
        num_timesteps = 1000
        timesteps = torch.round(noise_amount * (num_timesteps - 1)).long().to(device)

        # 得到预测结果（需要传入时间步长参数）
        pred = net(noisy_x, timesteps).sample

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
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()

# 查看输入、加噪、输出对比图像
x, y = next(iter(train_dataloader))
x = x[:8]   # 选择前8条数据

# 在(0，1)区间选择退化量
amount = torch.linspace(0, 1, x.shape[0])
noised_x = corrupt(x, amount)

# 将噪声量转换为时间步长
timesteps = torch.round(amount * (num_timesteps - 1)).long().to(device)

# 得到预测结果
with torch.no_grad():
    preds = net(noised_x.to(device), timesteps).sample.cpu()


# 绘图
# 绘制输入数据和加入噪声后的样本
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
# 原始数据
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0,1), cmap='Greys')
axs[0].axis('off')
# 加入噪声后的数据
axs[1].set_title('Corrupted data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0,1), cmap='Greys')
axs[1].axis('off')
# 预测输出
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0,1), cmap='Greys')
axs[2].axis('off')
plt.tight_layout()
plt.show()

# 采样处理（DDPM采样过程）
n_steps = 5
x = torch.rand(64, 1, 28, 28).to(device)

for i in range(n_steps):
    # 计算当前时间步（从大到小）
    timestep_value = num_timesteps - 1 - int(i * num_timesteps / n_steps)
    timesteps = torch.full((x.shape[0],), timestep_value, dtype=torch.long, device=device)
    
    with torch.no_grad():
        pred = net(x, timesteps).sample
    
    # 简单的更新策略（更复杂的扩散模型有特定的更新公式）
    mix_factor = 1/(n_steps - i)
    x = x*(1 - mix_factor) + pred*mix_factor

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.imshow(torchvision.utils.make_grid(x.detach().cpu(), nrow=8)[0].clip(0,1), cmap='Greys')
ax.set_title('Generated Samples')
ax.axis('off')
plt.tight_layout()
plt.show()