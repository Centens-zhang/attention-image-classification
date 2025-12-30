"""
数据加载模块 - CIFAR-10数据集
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List
import numpy as np
from torch.utils.data import Subset

def get_balanced_subset(targets: List[int], num_classes: int, subset_size: int) -> List[int]:
    """
    获取类别平衡的子集索引
    
    Args:
        targets: 数据集的所有标签
        num_classes: 类别总数
        subset_size: 目标子集大小
    
    Returns:
        选中的样本索引列表
    """
    samples_per_class = subset_size // num_classes
    indices = []
    
    # 将索引按类别分组
    class_indices = [[] for _ in range(num_classes)]
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)
        
    # 每个类别均匀采样
    for c in range(num_classes):
        # 如果该类样本不足，则取全部；否则随机采样
        curr_indices = class_indices[c]
        if len(curr_indices) > samples_per_class:
            # 随机选择
            selected = np.random.choice(curr_indices, samples_per_class, replace=False)
            indices.extend(selected)
        else:
            indices.extend(curr_indices)
            
    # 打乱索引
    np.random.shuffle(indices)
    return indices


import os
import tarfile
import subprocess
from torchvision.datasets.utils import check_integrity

def download_cifar10_manually(root_dir):
    """
    手动下载CIFAR-10数据集（使用清华镜像）
    """
    # 确保数据目录存在
    os.makedirs(root_dir, exist_ok=True)
    
    # CIFAR-10文件信息
    filename = "cifar-10-python.tar.gz"
    url = "https://mirrors.tuna.tsinghua.edu.cn/pytorch/cifar-10-python.tar.gz"
    filepath = os.path.join(root_dir, filename)
    
    # 检查文件是否已解压
    extracted_dir = os.path.join(root_dir, "cifar-10-batches-py")
    if os.path.exists(extracted_dir):
        print(f"数据集已存在: {extracted_dir}")
        return

    # 检查压缩包是否存在，如果不存在则下载
    if not os.path.exists(filepath):
        print(f"正在从清华镜像下载CIFAR-10数据集...")
        print(f"下载地址: {url}")
        try:
            subprocess.check_call(["wget", "-O", filepath, url])
            print("下载完成!")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请尝试手动下载并放置在 data/ 目录下")
            return
            
    # 解压文件
    print("正在解压数据集...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=root_dir)
    print("解压完成!")


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    获取CIFAR-10数据集的DataLoader
    
    Args:
        config: 配置对象，包含batch_size, num_workers等参数
    
    Returns:
        train_loader, test_loader: 训练集和测试集的DataLoader
    """
    # 尝试手动下载数据集
    download_cifar10_manually(config.data_root)
    
    # CIFAR-10数据集的均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 测试集变换（只做归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 训练集变换
    # 注意：config中属性名为use_augmentation
    if getattr(config, 'use_augmentation', False):
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
    
    # 加载完整训练集
    train_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=False,  # 已手动下载
        transform=train_transform
    )
    
    # 加载完整测试集
    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=False,  # 已手动下载
        transform=test_transform
    )
    
    # --- 核心修改：应用子集采样 ---
    if getattr(config, 'use_subset', False):
        print(f"⚠️ 启用子集模式：训练集{config.subset_train_size}张，测试集{config.subset_test_size}张")
        
        # 1. 获取训练集子集
        train_indices = get_balanced_subset(
            train_dataset.targets, 
            config.num_classes, 
            config.subset_train_size
        )
        train_dataset = Subset(train_dataset, train_indices)
        
        # 2. 获取测试集子集
        test_indices = get_balanced_subset(
            test_dataset.targets, 
            config.num_classes, 
            config.subset_test_size
        )
        test_dataset = Subset(test_dataset, test_indices)
    # ---------------------------
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # pin_memory=True if config.device == 'cuda' else False # config可能没有device属性，暂且注释或改为默认True
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
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
    from config import get_config
    
    print("测试数据加载模块...")
    config = get_config('exp2')
    # 强制启用子集以便测试
    config.use_subset = True
    config.subset_train_size = 100
    config.subset_test_size = 20
    
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
