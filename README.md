# 基于多注意力机制融合的图像分类算法研究

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目是本科毕业论文《基于多注意力机制融合的图像分类算法研究》的实验代码，实现了基于ResNet18的多种注意力机制（SE、CBAM）融合方法，并在CIFAR-10数据集上进行了消融实验。

## 📋 项目简介

本项目探索了注意力机制在图像分类任务中的应用，主要包括：

- **SE注意力模块** (Squeeze-and-Excitation): 通过显式建模通道间的相互依赖关系，自适应地重新校准通道特征响应
- **CBAM注意力模块** (Convolutional Block Attention Module): 结合通道注意力和空间注意力的复合注意力机制
- **多注意力融合**: 将SE和CBAM模块融合到ResNet18中，提升分类性能

## 🏗️ 项目结构

```
attention-image-classification/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖库
├── config.py                    # 配置文件（超参数）
├── data/
│   └── dataset.py              # CIFAR-10数据加载
├── models/
│   ├── __init__.py
│   ├── se_module.py            # SE注意力模块
│   ├── cbam_module.py          # CBAM注意力模块
│   └── resnet.py               # ResNet18及变体
├── utils/
│   ├── __init__.py
│   ├── logger.py               # 训练日志记录
│   └── metrics.py              # 评估指标计算
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── plot_results.py              # 结果可视化
├── run_experiments.sh           # 一键运行所有实验
├── checkpoints/                 # 模型权重保存目录
├── logs/                        # TensorBoard日志目录
└── results/                     # 实验结果保存目录
```

## 🚀 快速开始

### 环境配置

#### 本地环境

```bash
# 克隆项目
git clone https://github.com/Centens-zhang/attention-image-classification.git
cd attention-image-classification

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```


### 运行实验

#### 方式1: 运行单个实验

```bash
# 实验1: ResNet18基线（弱训练：50 epochs，无数据增强）
python train.py --exp exp1

# 实验2: ResNet18基线（强训练：100 epochs，有数据增强）
python train.py --exp exp2

# 实验3: ResNet18 + SE注意力模块
python train.py --exp exp3

# 实验4: ResNet18 + SE + CBAM（本文提出的方法）
python train.py --exp exp4
```

#### 方式2: 一键运行所有实验

```bash
bash run_experiments.sh
```

#### 断点续训

```bash
# 从最新的checkpoint恢复训练
python train.py --exp exp2 --resume
```

### 测试模型

```bash
# 测试指定实验的模型
python test.py --exp exp4

# 测试并生成混淆矩阵
python test.py --exp exp4 --plot-cm

# 使用指定的checkpoint文件
python test.py --exp exp4 --checkpoint ./checkpoints/exp4_best.pth
```

### 可视化结果

```bash
# 生成训练曲线对比图、准确率对比图和消融实验表格
python plot_results.py
```

### 查看TensorBoard日志

```bash
# 查看所有实验的训练日志
tensorboard --logdir=./logs

# 查看指定实验的日志
tensorboard --logdir=./logs/exp4
```

## 🧪 实验设置

### 四组对比实验

| 实验 | 模型 | 训练轮数 | 数据增强 | 学习率 | 描述 |
|------|------|----------|----------|--------|------|
| Exp1 | ResNet18 | 50 | ❌ | 0.01 | 弱基线 |
| Exp2 | ResNet18 | 100 | ✅ | 0.1 | 强基线 |
| Exp3 | ResNet18+SE | 100 | ✅ | 0.1 | 加入SE模块 |
| Exp4 | ResNet18+SE+CBAM | 100 | ✅ | 0.1 | 本文方法 |

### 训练配置

- **数据集**: CIFAR-10 (50,000训练 + 10,000测试)
- **优化器**: SGD with momentum=0.9, weight_decay=5e-4
- **学习率调度**: CosineAnnealingLR
- **Batch Size**: 128
- **数据增强**: RandomCrop(32, padding=4) + RandomHorizontalFlip
- **损失函数**: CrossEntropyLoss

### 模型架构

- **基础网络**: ResNet18（适配CIFAR-10的32×32输入）
- **SE模块**: reduction ratio = 16
- **CBAM模块**: kernel size = 7
- **注意力位置**: 每个残差块之后

## 📊 预期结果

### 性能指标（CIFAR-10）

| 模型 | 参数量 | Top-1 Acc | Top-5 Acc |
|------|--------|-----------|-----------|
| ResNet18 (弱) | ~11M | ~85% | ~99% |
| ResNet18 (强) | ~11M | ~88% | ~99.5% |
| ResNet18+SE | ~11M | ~89% | ~99.6% |
| ResNet18+SE+CBAM | ~11M | ~90% | ~99.7% |

### 输出文件

运行实验后，将生成以下文件：

```
checkpoints/
├── exp1_best.pth           # 最佳模型权重
├── exp1_latest.pth         # 最新模型权重（用于断点续训）
├── exp2_best.pth
├── exp3_best.pth
└── exp4_best.pth

logs/
├── exp1/                   # TensorBoard日志
├── exp2/
├── exp3/
└── exp4/

results/
├── training_curves.png     # 训练曲线对比图
├── accuracy_comparison.png # 准确率对比柱状图
├── ablation_study.csv      # 消融实验表格
└── confusion_matrix_exp4.png # 混淆矩阵
```

## 🔬 模块测试

每个模块都包含测试代码，可以独立测试：

```bash
# 测试配置模块
python config.py

# 测试数据加载模块
python data/dataset.py

# 测试SE注意力模块
python models/se_module.py

# 测试CBAM注意力模块
python models/cbam_module.py

# 测试ResNet模型
python models/resnet.py

# 测试日志模块
python utils/logger.py

# 测试评估指标模块
python utils/metrics.py
```

## 📝 代码特点

- ✅ **完整的docstring**: 所有函数都有详细的文档字符串
- ✅ **中文注释**: 关键步骤包含中文注释，便于理解
- ✅ **类型提示**: 使用Python type hints提高代码可读性
- ✅ **PEP8规范**: 遵循Python代码规范
- ✅ **模块化设计**: 各模块功能独立，易于复用
- ✅ **错误处理**: 提供清晰的错误提示信息
- ✅ **可复现性**: 设置随机种子，保存完整训练日志

## 📚 参考论文

1. **SE-Net**: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR 2018)
   ```
   @inproceedings{hu2018squeeze,
     title={Squeeze-and-excitation networks},
     author={Hu, Jie and Shen, Li and Sun, Gang},
     booktitle={CVPR},
     pages={7132--7141},
     year={2018}
   }
   ```

2. **CBAM**: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) (ECCV 2018)
   ```
   @inproceedings{woo2018cbam,
     title={Cbam: Convolutional block attention module},
     author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
     booktitle={ECCV},
     pages={3--19},
     year={2018}
   }
   ```

3. **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)
   ```
   @inproceedings{he2016deep,
     title={Deep residual learning for image recognition},
     author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
     booktitle={CVPR},
     pages={770--778},
     year={2016}
   }
   ```

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可证

本项目采用 MIT 许可证。

## 👨‍💻 作者

Centens Zhang

## 🙏 致谢

感谢PyTorch团队提供的优秀深度学习框架，以及CIFAR-10数据集的贡献者。
