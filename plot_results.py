"""
结果可视化脚本
读取所有实验的训练日志，生成对比图表
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_tensorboard_logs(log_dir: str) -> Dict[str, List]:
    """
    读取TensorBoard日志
    
    Args:
        log_dir: 日志目录
    
    Returns:
        包含训练数据的字典
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': []
    }
    
    # 读取训练loss
    if 'train/loss' in ea.Tags()['scalars']:
        for event in ea.Scalars('train/loss'):
            if event.step not in data['epochs']:
                data['epochs'].append(event.step)
            data['train_loss'].append(event.value)
    
    # 读取训练accuracy
    if 'train/accuracy' in ea.Tags()['scalars']:
        for event in ea.Scalars('train/accuracy'):
            data['train_acc'].append(event.value)
    
    # 读取测试loss
    if 'test/loss' in ea.Tags()['scalars']:
        for event in ea.Scalars('test/loss'):
            data['test_loss'].append(event.value)
    
    # 读取测试accuracy
    if 'test/accuracy' in ea.Tags()['scalars']:
        for event in ea.Scalars('test/accuracy'):
            data['test_acc'].append(event.value)
    
    return data


def plot_training_curves(all_data: Dict[str, Dict], save_path: str):
    """
    绘制训练曲线对比图
    
    Args:
        all_data: 所有实验的数据
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp_names = {
        'exp1': 'Exp1: ResNet18 (Weak)',
        'exp2': 'Exp2: ResNet18 (Strong)',
        'exp3': 'Exp3: ResNet18+SE',
        'exp4': 'Exp4: ResNet18+SE+CBAM'
    }
    
    colors = {
        'exp1': '#1f77b4',
        'exp2': '#ff7f0e',
        'exp3': '#2ca02c',
        'exp4': '#d62728'
    }
    
    # 绘制训练loss
    ax = axes[0, 0]
    for exp_name, data in all_data.items():
        if data['train_loss']:
            ax.plot(data['epochs'], data['train_loss'], 
                   label=exp_names[exp_name], color=colors[exp_name], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制测试loss
    ax = axes[0, 1]
    for exp_name, data in all_data.items():
        if data['test_loss']:
            ax.plot(data['epochs'], data['test_loss'], 
                   label=exp_names[exp_name], color=colors[exp_name], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制训练accuracy
    ax = axes[1, 0]
    for exp_name, data in all_data.items():
        if data['train_acc']:
            ax.plot(data['epochs'], data['train_acc'], 
                   label=exp_names[exp_name], color=colors[exp_name], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制测试accuracy
    ax = axes[1, 1]
    for exp_name, data in all_data.items():
        if data['test_acc']:
            ax.plot(data['epochs'], data['test_acc'], 
                   label=exp_names[exp_name], color=colors[exp_name], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练曲线已保存: {save_path}")


def plot_accuracy_comparison(all_data: Dict[str, Dict], save_path: str):
    """
    绘制准确率对比柱状图
    
    Args:
        all_data: 所有实验的数据
        save_path: 保存路径
    """
    # 提取最终准确率
    exp_names = []
    accuracies = []
    
    exp_labels = {
        'exp1': 'ResNet18\n(Weak)',
        'exp2': 'ResNet18\n(Strong)',
        'exp3': 'ResNet18\n+SE',
        'exp4': 'ResNet18\n+SE+CBAM'
    }
    
    for exp_name in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp_name in all_data and all_data[exp_name]['test_acc']:
            exp_names.append(exp_labels[exp_name])
            accuracies.append(max(all_data[exp_name]['test_acc']))
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(exp_names, accuracies, color=colors[:len(exp_names)], 
                  width=0.6, edgecolor='black', linewidth=1.5)
    
    # 在柱子上方显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 准确率对比图已保存: {save_path}")


def create_ablation_table(all_data: Dict[str, Dict], save_path: str):
    """
    创建消融实验表格
    
    Args:
        all_data: 所有实验的数据
        save_path: 保存路径
    """
    # 从checkpoint文件读取参数量信息
    from models import get_model
    from utils import count_parameters
    
    table_data = []
    
    model_configs = {
        'exp1': ('resnet18', 'ResNet18 (Weak Training)'),
        'exp2': ('resnet18', 'ResNet18 (Strong Training)'),
        'exp3': ('resnet18_se', 'ResNet18 + SE'),
        'exp4': ('resnet18_se_cbam', 'ResNet18 + SE + CBAM')
    }
    
    for exp_name in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp_name in all_data and all_data[exp_name]['test_acc']:
            model_name, description = model_configs[exp_name]
            
            # 获取参数量
            model = get_model(model_name, num_classes=10)
            total_params, _ = count_parameters(model)
            
            # 获取最佳准确率
            best_acc = max(all_data[exp_name]['test_acc'])
            
            table_data.append({
                'Experiment': exp_name.upper(),
                'Model': description,
                'Parameters': f'{total_params:,}',
                'Data Aug.': 'Yes' if exp_name != 'exp1' else 'No',
                'Epochs': 100 if exp_name != 'exp1' else 50,
                'Best Acc (%)': f'{best_acc:.2f}'
            })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(table_data)
    df.to_csv(save_path, index=False)
    
    print(f"✓ 消融实验表格已保存: {save_path}")
    print("\n消融实验结果:")
    print(df.to_string(index=False))


def main():
    """主函数"""
    print("\n" + "="*60)
    print("结果可视化")
    print("="*60)
    
    log_dir = './logs'
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    
    # 读取所有实验的日志
    all_data = {}
    exp_names = ['exp1', 'exp2', 'exp3', 'exp4']
    
    print("\n读取训练日志...")
    for exp_name in exp_names:
        exp_log_dir = os.path.join(log_dir, exp_name)
        
        if os.path.exists(exp_log_dir):
            print(f"  读取 {exp_name}...")
            try:
                data = read_tensorboard_logs(exp_log_dir)
                all_data[exp_name] = data
                
                if data['test_acc']:
                    print(f"    最佳准确率: {max(data['test_acc']):.2f}%")
            except Exception as e:
                print(f"    警告: 读取失败 - {str(e)}")
        else:
            print(f"  跳过 {exp_name} (日志不存在)")
    
    if not all_data:
        print("\n错误: 没有找到任何训练日志!")
        print("请先运行训练脚本生成日志。")
        return
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 1. 训练曲线对比图
    training_curves_path = os.path.join(result_dir, 'training_curves.png')
    plot_training_curves(all_data, training_curves_path)
    
    # 2. 准确率对比柱状图
    accuracy_comparison_path = os.path.join(result_dir, 'accuracy_comparison.png')
    plot_accuracy_comparison(all_data, accuracy_comparison_path)
    
    # 3. 消融实验表格
    ablation_table_path = os.path.join(result_dir, 'ablation_study.csv')
    create_ablation_table(all_data, ablation_table_path)
    
    print("\n" + "="*60)
    print("可视化完成!")
    print(f"结果保存在: {result_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
