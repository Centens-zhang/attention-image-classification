"""
CBAM注意力模块 (Convolutional Block Attention Module)
参考论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
论文链接: https://arxiv.org/abs/1807.06521
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    
    通过全局平均池化和全局最大池化捕获通道间的关系，
    使用共享的MLP网络生成通道注意力图。
    
    Args:
        channels: 输入特征图的通道数
        reduction: 降维比例，默认为16
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP网络
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            通道注意力图 [B, C, 1, 1]
        """
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x))
        # 合并两个分支并应用sigmoid
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    通过在通道维度上进行平均池化和最大池化，
    捕获空间位置的重要性。
    
    Args:
        kernel_size: 卷积核大小，默认为7
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        # 保证padding使输出尺寸不变
        padding = kernel_size // 2
        
        # 卷积层：将2通道特征图转换为1通道的注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            空间注意力图 [B, 1, H, W]
        """
        # 在通道维度上进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接两个池化结果
        x = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # 通过卷积层生成空间注意力图
        x = self.conv(x)  # [B, 1, H, W]
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module)
    
    结合通道注意力和空间注意力的复合注意力模块。
    先应用通道注意力，再应用空间注意力。
    
    工作流程:
    1. 输入特征 F
    2. 通道注意力: F' = F ⊗ Mc(F)
    3. 空间注意力: F'' = F' ⊗ Ms(F')
    4. 输出 F''
    
    Args:
        channels: 输入特征图的通道数
        reduction: 通道注意力的降维比例，默认为16
        kernel_size: 空间注意力的卷积核大小，默认为7
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channels, reduction)
        # 空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            输出特征图 [B, C, H, W]
        """
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x


if __name__ == '__main__':
    # 测试CBAM模块
    print("测试CBAM注意力模块...")
    
    # 创建测试输入
    batch_size = 4
    channels = 64
    height, width = 32, 32
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"输入形状: {x.shape}")
    
    # 测试通道注意力
    print("\n1. 测试通道注意力模块:")
    ca = ChannelAttention(channels=channels, reduction=16)
    ca_out = ca(x)
    print(f"   通道注意力输出形状: {ca_out.shape}")
    assert ca_out.shape == (batch_size, channels, 1, 1), "通道注意力输出形状错误!"
    
    # 测试空间注意力
    print("\n2. 测试空间注意力模块:")
    sa = SpatialAttention(kernel_size=7)
    sa_out = sa(x)
    print(f"   空间注意力输出形状: {sa_out.shape}")
    assert sa_out.shape == (batch_size, 1, height, width), "空间注意力输出形状错误!"
    
    # 测试完整CBAM模块
    print("\n3. 测试完整CBAM模块:")
    cbam = CBAM(channels=channels, reduction=16, kernel_size=7)
    y = cbam(x)
    print(f"   CBAM输出形状: {y.shape}")
    assert y.shape == x.shape, "输出形状应该与输入相同!"
    
    # 计算参数量
    total_params = sum(p.numel() for p in cbam.parameters())
    print(f"   参数量: {total_params}")
    
    print("\n✓ CBAM模块测试通过!")
    print(f"✓ 输入形状 {x.shape} -> 输出形状 {y.shape}")
