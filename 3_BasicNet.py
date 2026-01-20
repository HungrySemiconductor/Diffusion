import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_loader))
print('Input shape:', x.shape)
print('Lables:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='gray');


