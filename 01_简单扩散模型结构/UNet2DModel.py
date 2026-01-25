from diffusers import UNet2DModel

model = UNet2DModel(
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

# print(model)    # 输出模型结构
print(sum([p.numel() for p in model.parameters()]))  # 输出模型参数总数