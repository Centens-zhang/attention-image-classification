"""
配置文件 - 定义所有实验的超参数
"""
from typing import Dict, Any
import torch


class BaseConfig:
    """基础配置类"""
    # 数据相关
    data_root = './data'
    dataset = 'CIFAR10'
    num_classes = 10
    
    # 训练相关
    batch_size = 128
    num_workers = 4
    
    # 优化器相关
    optimizer = 'SGD'
    momentum = 0.9
    weight_decay = 5e-4
    
    # 学习率调度
    scheduler = 'CosineAnnealingLR'
    
    # 其他
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 保存路径
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    result_dir = './results'
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


class Exp1Config(BaseConfig):
    """实验1: ResNet18基线（弱训练）
    - 50 epochs
    - 无数据增强
    - 学习率: 0.01
    """
    exp_name = 'exp1'
    model_name = 'resnet18'
    epochs = 50
    lr = 0.01
    use_data_augmentation = False
    description = 'ResNet18基线（弱训练：50 epochs，无数据增强）'


class Exp2Config(BaseConfig):
    """实验2: ResNet18基线（强训练）
    - 100 epochs
    - 有数据增强
    - 学习率: 0.1
    """
    exp_name = 'exp2'
    model_name = 'resnet18'
    epochs = 100
    lr = 0.1
    use_data_augmentation = True
    description = 'ResNet18基线（强训练：100 epochs，有数据增强）'


class Exp3Config(BaseConfig):
    """实验3: ResNet18 + SE模块
    - 100 epochs
    - 有数据增强
    - 学习率: 0.1
    """
    exp_name = 'exp3'
    model_name = 'resnet18_se'
    epochs = 100
    lr = 0.1
    use_data_augmentation = True
    description = 'ResNet18 + SE注意力模块'


class Exp4Config(BaseConfig):
    """实验4: ResNet18 + SE + CBAM（本文提出的方法）
    - 100 epochs
    - 有数据增强
    - 学习率: 0.1
    """
    exp_name = 'exp4'
    model_name = 'resnet18_se_cbam'
    epochs = 100
    lr = 0.1
    use_data_augmentation = True
    description = 'ResNet18 + SE + CBAM（本文提出的方法）'


def get_config(exp_name: str) -> BaseConfig:
    """
    根据实验名称获取对应的配置
    
    Args:
        exp_name: 实验名称 ('exp1', 'exp2', 'exp3', 'exp4')
    
    Returns:
        对应的配置对象
    """
    configs = {
        'exp1': Exp1Config(),
        'exp2': Exp2Config(),
        'exp3': Exp3Config(),
        'exp4': Exp4Config(),
    }
    
    if exp_name not in configs:
        raise ValueError(f"未知的实验名称: {exp_name}. 可选: {list(configs.keys())}")
    
    return configs[exp_name]


if __name__ == '__main__':
    # 测试配置
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        config = get_config(exp)
        print(f"\n{exp.upper()}:")
        print(f"  模型: {config.model_name}")
        print(f"  训练轮数: {config.epochs}")
        print(f"  学习率: {config.lr}")
        print(f"  数据增强: {config.use_data_augmentation}")
        print(f"  描述: {config.description}")
