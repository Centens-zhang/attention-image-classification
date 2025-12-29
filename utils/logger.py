"""
训练日志记录模块
使用TensorBoard记录训练过程
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional


class Logger:
    """
    训练日志记录器
    
    功能:
    1. 使用TensorBoard记录训练曲线
    2. 保存最佳模型checkpoint
    3. 支持断点续训
    
    Args:
        log_dir: 日志保存目录
        exp_name: 实验名称
    """
    
    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir = os.path.join(log_dir, exp_name)
        self.exp_name = exp_name
        
        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # 记录最佳模型信息
        self.best_acc = 0.0
        self.best_epoch = 0
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        记录标量指标
        
        Args:
            metrics: 指标字典，例如 {'loss': 0.5, 'acc': 0.9}
            step: 当前步数（epoch或iteration）
            prefix: 指标前缀，例如 'train' 或 'test'
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)
    
    def log_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        train_acc: float,
        test_loss: float, 
        test_acc: float
    ):
        """
        记录一个epoch的训练和测试结果
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            train_acc: 训练准确率
            test_loss: 测试损失
            test_acc: 测试准确率
        """
        # 记录训练指标
        self.log_scalars({
            'loss': train_loss,
            'accuracy': train_acc
        }, epoch, prefix='train')
        
        # 记录测试指标
        self.log_scalars({
            'loss': test_loss,
            'accuracy': test_acc
        }, epoch, prefix='test')
        
        # 打印日志
        print(f"Epoch [{epoch}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        test_acc: float,
        checkpoint_dir: str,
        is_best: bool = False
    ):
        """
        保存模型checkpoint
        
        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            test_acc: 测试准确率
            checkpoint_dir: checkpoint保存目录
            is_best: 是否为最佳模型
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 构建checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存最新的checkpoint
        latest_path = os.path.join(checkpoint_dir, f'{self.exp_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(checkpoint_dir, f'{self.exp_name}_best.pth')
            torch.save(checkpoint, best_path)
            self.best_acc = test_acc
            self.best_epoch = epoch
            print(f"✓ 保存最佳模型! Epoch: {epoch}, Accuracy: {test_acc:.2f}%")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> int:
        """
        加载checkpoint，支持断点续训
        
        Args:
            checkpoint_path: checkpoint文件路径
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        
        Returns:
            开始的epoch数
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint不存在: {checkpoint_path}")
            return 0
        
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载学习率调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复最佳模型信息
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ 从epoch {start_epoch}继续训练, 最佳准确率: {self.best_acc:.2f}%")
        
        return start_epoch
    
    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()
        print(f"\n训练完成!")
        print(f"最佳模型: Epoch {self.best_epoch}, Accuracy: {self.best_acc:.2f}%")
        print(f"日志保存在: {self.log_dir}")


if __name__ == '__main__':
    # 测试Logger
    print("测试Logger模块...")
    
    # 创建临时logger
    logger = Logger(log_dir='./test_logs', exp_name='test_exp')
    
    # 模拟训练过程
    for epoch in range(1, 4):
        train_loss = 1.0 / epoch
        train_acc = 80.0 + epoch * 2
        test_loss = 0.8 / epoch
        test_acc = 85.0 + epoch * 2
        
        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
    
    logger.close()
    
    print("\n✓ Logger模块测试通过!")
    print("可以运行 'tensorboard --logdir=./test_logs' 查看日志")
