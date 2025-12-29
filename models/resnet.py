"""
ResNet模型及其注意力机制变体
适配CIFAR-10数据集（32x32输入）
"""
import torch
import torch.nn as nn
from typing import Optional, Type, List
from .se_module import SEModule
from .cbam_module import CBAM


class BasicBlock(nn.Module):
    """
    ResNet的基本残差块
    
    结构:
    x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
    |_______________________________________________|
    
    支持插入SE和CBAM注意力模块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        use_se: 是否使用SE模块
        use_cbam: 是否使用CBAM模块
    """
    expansion = 1  # BasicBlock的通道扩展因子
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        use_se: bool = False,
        use_cbam: bool = False
    ):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力模块
        self.use_se = use_se
        self.use_cbam = use_cbam
        
        if use_se:
            self.se = SEModule(out_channels, reduction=16)
        
        if use_cbam:
            self.cbam = CBAM(out_channels, reduction=16, kernel_size=7)
        
        # 快捷连接（shortcut）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果输入输出尺寸不同，需要调整shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力模块（在BN之后，加法之前）
        if self.use_se:
            out = self.se(out)
        
        if self.use_cbam:
            out = self.cbam(out)
        
        # 快捷连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet模型（适配CIFAR-10）
    
    与标准ResNet不同之处：
    1. 初始卷积层使用3x3卷积，stride=1（而非7x7, stride=2）
    2. 不使用最大池化层
    3. 适配32x32输入尺寸
    
    Args:
        block: 残差块类型（BasicBlock）
        num_blocks: 每个stage的block数量
        num_classes: 分类类别数
        use_se: 是否使用SE模块
        use_cbam: 是否使用CBAM模块
    """
    
    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 10,
        use_se: bool = False,
        use_cbam: bool = False
    ):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.use_se = use_se
        self.use_cbam = use_cbam
        
        # 初始卷积层（适配CIFAR-10）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 4个stage
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(
        self, 
        block: Type[BasicBlock], 
        out_channels: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """构建ResNet的一个stage"""
        layers = []
        
        # 第一个block可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, self.use_se, self.use_cbam))
        self.in_channels = out_channels * block.expansion
        
        # 后续block
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, 1, self.use_se, self.use_cbam))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 4个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类器
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    工厂函数：根据模型名称返回对应的模型
    
    Args:
        model_name: 模型名称
            - 'resnet18': 纯ResNet18
            - 'resnet18_se': ResNet18 + SE
            - 'resnet18_cbam': ResNet18 + CBAM
            - 'resnet18_se_cbam': ResNet18 + SE + CBAM
        num_classes: 分类类别数，默认10（CIFAR-10）
    
    Returns:
        对应的模型实例
    """
    # ResNet18配置：[2, 2, 2, 2]表示每个stage有2个BasicBlock
    num_blocks = [2, 2, 2, 2]
    
    if model_name == 'resnet18':
        # 纯ResNet18
        model = ResNet(BasicBlock, num_blocks, num_classes, use_se=False, use_cbam=False)
    
    elif model_name == 'resnet18_se':
        # ResNet18 + SE
        model = ResNet(BasicBlock, num_blocks, num_classes, use_se=True, use_cbam=False)
    
    elif model_name == 'resnet18_cbam':
        # ResNet18 + CBAM
        model = ResNet(BasicBlock, num_blocks, num_classes, use_se=False, use_cbam=True)
    
    elif model_name == 'resnet18_se_cbam':
        # ResNet18 + SE + CBAM
        model = ResNet(BasicBlock, num_blocks, num_classes, use_se=True, use_cbam=True)
    
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    return model


if __name__ == '__main__':
    # 测试所有模型变体
    print("测试ResNet模型及其变体...")
    
    model_names = ['resnet18', 'resnet18_se', 'resnet18_cbam', 'resnet18_se_cbam']
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print('='*60)
        
        # 创建模型
        model = get_model(model_name, num_classes=10)
        
        # 测试前向传播
        x = torch.randn(4, 3, 32, 32)  # CIFAR-10输入尺寸
        y = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 验证输出形状
        assert y.shape == (4, 10), f"输出形状错误: {y.shape}"
    
    print("\n✓ 所有模型测试通过!")
