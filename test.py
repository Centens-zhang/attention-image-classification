"""
测试脚本 - 模型评估
"""
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from data.dataset import get_dataloaders, get_class_names
from models import get_model
from utils import (
    accuracy,
    AverageMeter,
    compute_confusion_matrix,
    plot_confusion_matrix,
    count_parameters,
    compute_flops
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str
) -> tuple:
    """
    在测试集上评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        (Top-1准确率, Top-5准确率, 平均损失)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    pbar = tqdm(test_loader, desc='评估中')
    
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
    
    return top1_meter.avg, top5_meter.avg, loss_meter.avg


def main(args):
    """主测试函数"""
    
    # 获取配置
    config = get_config(args.exp)
    print(f"\n{'='*60}")
    print(f"测试实验: {config.exp_name}")
    print(f"模型: {config.model_name}")
    print(f"{'='*60}")
    
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n加载测试数据集...")
    _, test_loader = get_dataloaders(config)
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
    
    # 加载模型权重
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.exp_name}_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 模型文件不存在: {checkpoint_path}")
        print("请先训练模型或指定正确的checkpoint路径")
        return
    
    print(f"\n加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"模型来自Epoch: {checkpoint['epoch']}")
    if 'test_acc' in checkpoint:
        print(f"训练时测试准确率: {checkpoint['test_acc']:.2f}%")
    
    # 评估模型
    print("\n" + "="*60)
    print("开始评估...")
    print("="*60)
    
    top1_acc, top5_acc, avg_loss = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("评估结果:")
    print(f"{'='*60}")
    print(f"测试损失: {avg_loss:.4f}")
    print(f"Top-1 准确率: {top1_acc:.2f}%")
    print(f"Top-5 准确率: {top5_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # 生成混淆矩阵
    if args.plot_cm:
        print("生成混淆矩阵...")
        cm = compute_confusion_matrix(
            model, test_loader, device, num_classes=config.num_classes
        )
        
        # 保存混淆矩阵
        os.makedirs(config.result_dir, exist_ok=True)
        cm_path = os.path.join(config.result_dir, f'confusion_matrix_{config.exp_name}.png')
        
        class_names = get_class_names()
        plot_confusion_matrix(cm, class_names, cm_path, normalize=True)
        
        print(f"✓ 混淆矩阵已保存: {cm_path}")
    
    print("\n测试完成!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试图像分类模型')
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        choices=['exp1', 'exp2', 'exp3', 'exp4'],
        help='实验名称 (exp1/exp2/exp3/exp4)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（默认使用best模型）'
    )
    parser.add_argument(
        '--plot-cm',
        action='store_true',
        help='是否生成混淆矩阵图'
    )
    
    args = parser.parse_args()
    main(args)
