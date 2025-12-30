"""
配置文件 - 所有超参数配置
支持小数据集训练（1万训练集 + 2千测试集）
"""

class Config:
    """基础配置"""
    # 数据集配置
    dataset = 'CIFAR10'
    data_root = './data'
    num_classes = 10
    
    # 小数据集配置（核心修改）
    use_subset = False           # 使用数据集子集
    subset_train_size = 10000   # 训练集：1万张
    subset_test_size = 2000     # 测试集：2千张
    
    # 训练配置
    batch_size = 128
    num_epochs = 100
    num_workers = 4
    
    # 优化器配置
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # 学习率调度
    lr_scheduler = 'cosine'
    warmup_epochs = 5
    
    # 数据增强
    use_augmentation = True
    
    # 模型配置
    model_name = 'resnet18'
    
    # 路径配置
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    result_dir = './results'
    
    # 设备配置
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 随机种子
    seed = 42


# 四组实验配置
EXPERIMENTS = {
    'exp1': {
        'model_name': 'resnet18',
        'num_epochs': 50,
        'use_augmentation': False,
        'description': '弱基线：ResNet18，无数据增强，50epochs'
    },
    'exp2': {
        'model_name': 'resnet18',
        'num_epochs': 100,
        'use_augmentation': True,
        'description': '强基线：ResNet18，有数据增强，100epochs'
    },
    'exp3': {
        'model_name': 'resnet18_se',
        'num_epochs': 100,
        'use_augmentation': True,
        'description': 'ResNet18 + SE模块'
    },
    'exp4': {
        'model_name': 'resnet18_se_cbam',
        'num_epochs': 100,
        'use_augmentation': True,
        'description': 'ResNet18 + SE + CBAM（本文方法）'
    }
}


def get_config(exp_name: str = None) -> Config:
    """
    获取配置对象
    
    Args:
        exp_name: 实验名称 (exp1/exp2/exp3/exp4)
    
    Returns:
        配置对象
    """
    config = Config()
    
    if exp_name and exp_name in EXPERIMENTS:
        exp_config = EXPERIMENTS[exp_name]
        config.exp_name = exp_name
        config.model_name = exp_config['model_name']
        config.num_epochs = exp_config['num_epochs']
        config.use_augmentation = exp_config['use_augmentation']
        config.description = exp_config['description']
        
        # 设置特定实验的日志和检查点路径
        import os
        config.checkpoint_dir = os.path.join(config.checkpoint_dir)
        config.log_dir = os.path.join(config.log_dir, exp_name)
        
        # 确保目录存在
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.result_dir, exist_ok=True)
        
    return config
