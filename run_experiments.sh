#!/bin/bash
# 批量运行所有实验的脚本

echo "================================"
echo "开始运行所有实验"
echo "================================"

# 实验1: ResNet18基线（弱训练）
echo ""
echo "================================"
echo "运行实验1: ResNet18 (弱训练)"
echo "================================"
python train.py --exp exp1

# 实验2: ResNet18基线（强训练）
echo ""
echo "================================"
echo "运行实验2: ResNet18 (强训练)"
echo "================================"
python train.py --exp exp2

# 实验3: ResNet18 + SE
echo ""
echo "================================"
echo "运行实验3: ResNet18 + SE"
echo "================================"
python train.py --exp exp3

# 实验4: ResNet18 + SE + CBAM
echo ""
echo "================================"
echo "运行实验4: ResNet18 + SE + CBAM"
echo "================================"
python train.py --exp exp4

# 生成可视化结果
echo ""
echo "================================"
echo "生成可视化结果"
echo "================================"
python plot_results.py

echo ""
echo "================================"
echo "所有实验完成!"
echo "================================"
echo "查看结果:"
echo "  - 模型: ./checkpoints/"
echo "  - 日志: ./logs/"
echo "  - 可视化: ./results/"
echo "================================"
