"""
训练脚本 - 主训练循环
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import get_config
from data.dataset import get_dataloaders
from models import get_model
from utils import Logger, accuracy, AverageMeter, count_parameters, compute_flops


def set_seed(seed: int):
    """
    设置随机种子，确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> tuple:
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
    
    Returns:
        (平均损失, 平均准确率)
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = accuracy(outputs, targets, topk=(1,))[0]
        
        # 更新统计量
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, acc_meter.avg


def test_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> tuple:
    """
    测试一个epoch
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
    
    Returns:
        (平均损失, Top-1准确率, Top-5准确率)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    pbar = tqdm(test_loader, desc=f'Epoch {epoch} [Test]')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 计算准确率
            top1_acc, top5_acc = accuracy(outputs, targets, topk=(1, 5))
            
            # 更新统计量
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'top1': f'{top1_meter.avg:.2f}%',
                'top5': f'{top5_meter.avg:.2f}%'
            })
    
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def main(args):
    """主训练函数"""
    
    # 获取配置
    config = get_config(args.exp)
    print(f"\n{'='*60}")
    print(f"实验: {config.exp_name}")
    print(f"描述: {config.description}")
    print(f"{'='*60}")
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n加载数据集...")
    train_loader, test_loader = get_dataloaders(config)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print(f"\n创建模型: {config.model_name}")
    model = get_model(config.model_name, num_classes=config.num_classes)
    model = model.to(device)
    
    # 打印模型信息
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    flops = compute_flops(model, input_size=(1, 3, 32, 32))
    if flops > 0:
        print(f"FLOPs: {flops:.2f} G")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )
    
    # 创建日志记录器
    logger = Logger(config.log_dir, config.exp_name)
    
    # 支持断点续训
    start_epoch = 1
    if args.resume:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.exp_name}_latest.pth')
        start_epoch = logger.load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    # 训练循环
    print(f"\n开始训练 (共{config.epochs}个epoch)...")
    print(f"学习率: {config.lr}, Batch Size: {config.batch_size}")
    print(f"数据增强: {'是' if config.use_data_augmentation else '否'}")
    
    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch: {epoch}/{config.epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print('='*60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 测试
        test_loss, test_top1_acc, test_top5_acc = test_epoch(
            model, test_loader, criterion, device, epoch
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_top1_acc)
        
        # 保存checkpoint
        is_best = test_top1_acc > logger.best_acc
        logger.save_checkpoint(
            epoch, model, optimizer, scheduler,
            test_top1_acc, config.checkpoint_dir, is_best
        )
        
        # 打印Top-5准确率
        print(f"Top-5 Accuracy: {test_top5_acc:.2f}%")
    
    # 关闭日志记录器
    logger.close()
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳模型: Epoch {logger.best_epoch}, Top-1 Acc: {logger.best_acc:.2f}%")
    print(f"模型保存在: {config.checkpoint_dir}/{config.exp_name}_best.pth")
    print(f"日志保存在: {config.log_dir}/{config.exp_name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练图像分类模型')
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        choices=['exp1', 'exp2', 'exp3', 'exp4'],
        help='实验名称 (exp1/exp2/exp3/exp4)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从最新的checkpoint恢复训练'
    )
    
    args = parser.parse_args()
    main(args)
