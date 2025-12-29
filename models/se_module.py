"""
SE注意力模块 (Squeeze-and-Excitation Module)
参考论文: Squeeze-and-Excitation Networks (CVPR 2018)
论文链接: https://arxiv.org/abs/1709.01507
"""
import torch
import torch.nn as nn


class SEModule(nn.Module):
    """
    SE (Squeeze-and-Excitation) 注意力模块
    
    通过显式地建模通道之间的相互依赖关系，自适应地重新校准通道特征响应。
    
    工作流程:
    1. Squeeze: 全局平均池化，将空间维度压缩为1x1
    2. Excitation: 两层全连接网络学习通道注意力权重
       - 第一层: 降维 (channels -> channels/reduction)
       - 第二层: 升维 (channels/reduction -> channels)
    3. Scale: 将学到的权重与原始特征相乘
    
    Args:
        channels: 输入特征图的通道数
        reduction: 降维比例，默认为16
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEModule, self).__init__()
        
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 两层全连接网络
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图，形状为 [batch_size, channels, height, width]
        
        Returns:
            输出特征图，形状与输入相同
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # 展平为 [B, C]
        y = y.view(batch_size, channels)
        
        # Excitation: 学习通道注意力权重 [B, C] -> [B, C]
        y = self.fc(y)
        
        # 恢复维度 [B, C] -> [B, C, 1, 1]
        y = y.view(batch_size, channels, 1, 1)
        
        # Scale: 将权重应用到原始特征图
        return x * y.expand_as(x)


if __name__ == '__main__':
    # 测试SE模块
    print("测试SE注意力模块...")
    
    # 创建测试输入
    batch_size = 4
    channels = 64
    height, width = 32, 32
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"输入形状: {x.shape}")
    
    # 创建SE模块
    se = SEModule(channels=channels, reduction=16)
    
    # 前向传播
    y = se(x)
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状
    assert y.shape == x.shape, "输出形状应该与输入相同!"
    
    # 计算参数量
    total_params = sum(p.numel() for p in se.parameters())
    print(f"参数量: {total_params}")
    
    # 理论参数量: (C * C/r) + (C/r * C) = 2 * C^2 / r
    expected_params = 2 * channels * (channels // 16)
    print(f"理论参数量: {expected_params}")
    
    print("\n✓ SE模块测试通过!")
    print(f"✓ 输入形状 {x.shape} -> 输出形状 {y.shape}")
