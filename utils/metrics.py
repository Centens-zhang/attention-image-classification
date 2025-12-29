"""
评估指标计算模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list:
    """
    计算Top-K准确率
    
    Args:
        output: 模型输出 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        topk: 要计算的top-k值，例如 (1, 5)
    
    Returns:
        准确率列表，对应每个k值
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # 获取top-k预测结果
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # 比较预测和真实标签
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def compute_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int = 10
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数
    
    Returns:
        混淆矩阵 [num_classes, num_classes]
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = False
):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 混淆矩阵已保存: {save_path}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    计算模型参数量
    
    Args:
        model: 模型
    
    Returns:
        (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """
    计算模型的FLOPs（浮点运算次数）
    
    Args:
        model: 模型
        input_size: 输入尺寸 (batch_size, channels, height, width)
    
    Returns:
        FLOPs (单位: G)
    """
    try:
        from thop import profile
        
        input_tensor = torch.randn(input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        # 转换为G（十亿）
        flops_g = flops / 1e9
        
        return flops_g
    
    except ImportError:
        print("警告: 未安装thop库，无法计算FLOPs")
        return 0.0


class AverageMeter:
    """
    计算并存储平均值和当前值
    用于记录训练过程中的loss和accuracy
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新统计量
        
        Args:
            val: 当前值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # 测试评估指标
    print("测试评估指标模块...")
    
    # 测试Top-K准确率
    print("\n1. 测试Top-K准确率:")
    output = torch.randn(8, 10)  # 8个样本，10个类别
    target = torch.randint(0, 10, (8,))
    
    top1_acc, top5_acc = accuracy(output, target, topk=(1, 5))
    print(f"   Top-1准确率: {top1_acc:.2f}%")
    print(f"   Top-5准确率: {top5_acc:.2f}%")
    
    # 测试AverageMeter
    print("\n2. 测试AverageMeter:")
    meter = AverageMeter()
    for i in range(1, 6):
        meter.update(i * 10, n=1)
        print(f"   更新{i}: 当前值={meter.val}, 平均值={meter.avg:.2f}")
    
    # 测试参数量计算
    print("\n3. 测试参数量计算:")
    import sys
    sys.path.append('..')
    from models import get_model
    
    model = get_model('resnet18', num_classes=10)
    total_params, trainable_params = count_parameters(model)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 测试FLOPs计算
    print("\n4. 测试FLOPs计算:")
    flops = compute_flops(model, input_size=(1, 3, 32, 32))
    if flops > 0:
        print(f"   FLOPs: {flops:.2f} G")
    
    print("\n✓ 评估指标模块测试通过!")
