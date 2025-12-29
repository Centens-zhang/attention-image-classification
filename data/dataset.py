"""
数据加载模块 - CIFAR-10数据集
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    获取CIFAR-10数据集的DataLoader
    
    Args:
        config: 配置对象，包含batch_size, num_workers等参数
    
    Returns:
        train_loader, test_loader: 训练集和测试集的DataLoader
    """
    # CIFAR-10数据集的均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 测试集变换（只做归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 训练集变换
    if config.use_data_augmentation:
        # 强训练：包含数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # 弱训练：不使用数据增强
        train_transform = test_transform
    
    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    return train_loader, test_loader


def get_class_names() -> list:
    """
    获取CIFAR-10的类别名称
    
    Returns:
        类别名称列表
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == '__main__':
    # 测试数据加载
    import sys
    sys.path.append('..')
    from config import Exp2Config
    
    print("测试数据加载模块...")
    config = Exp2Config()
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(config)
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch形状:")
    print(f"  图像: {images.shape}")  # [batch_size, 3, 32, 32]
    print(f"  标签: {labels.shape}")  # [batch_size]
    print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n类别名称:")
    print(get_class_names())
    
    print("\n✓ 数据加载模块测试通过!")
