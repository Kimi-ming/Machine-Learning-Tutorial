# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个中文机器学习教学项目，使用纯 Python 从零实现核心机器学习和深度学习算法。项目强调教学清晰度而非性能，包含详细的原理解释和可视化功能。

**核心理念**：所有算法都不依赖 sklearn/tensorflow 等库来展示内部工作机制。可选支持 numpy/matplotlib 以获得更好的性能和可视化效果。

## 代码运行

### 基本执行方式

```bash
# 直接运行各个算法模块
python ml_linear_regression.py
python ml_neural_network.py
python ml_random_forest.py
python deep_learning_cnn.py

# 运行简化版线性回归演示
python ml_linear_regression_simple.py
```

**注意**：虽然 README 中提到了 `python main.py`，但仓库中实际上没有 main.py 文件。每个算法模块都是独立的，可以直接执行。

### 测试

本仓库没有正式的测试文件。每个算法模块包含演示代码，直接运行时会展示：
- 理论解释（输出到控制台）
- 示例数据集和预测结果
- 性能指标
- ASCII 可视化（如果没有 matplotlib）

## 代码架构

### 算法模块结构

每个 `ml_*.py` 和 `deep_learning_*.py` 文件都遵循以下模式：

1. **理论函数**：`{algorithm}_theory()` - 打印详细的中文解释
2. **核心算法类**：实现算法，包含 `fit()` 和 `predict()` 方法
3. **演示函数**：各种 `*_example()` 或 `*_demo()` 函数
4. **主函数**：`main()` 函数，直接运行模块时会编排所有演示

### 双实现模式

许多模块实现了回退逻辑：

```python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # 后面使用纯 Python 实现
```

**重要**：修改算法代码时，需要同时维护 numpy 和纯 Python 两种代码路径。

### 线性回归的特殊情况

存在两个实现版本：
- `ml_linear_regression.py` - 完整版，包含梯度下降、可视化、学习率实验
- `ml_linear_regression_simple.py` - 简化版，使用闭式解，采用 dataclass 和类型提示

简化版展示了更现代的 Python 实践（dataclass、类型提示），并包含：
- 自动学习率估计（`learning_rate='auto'`）
- 带 patience 机制的稳健早停
- 基于文本的损失曲线可视化（不依赖 matplotlib）
- 特征标准化支持

## 模块分类

**基础算法**：
- 线性回归、逻辑回归、决策树、K均值聚类

**高级算法**：
- 神经网络（多层感知机，反向传播）
- 支持向量机（简化 SMO 算法）
- 随机森林（Bootstrap 采样，特征重要性）

**深度学习**：
- `deep_learning_cnn.py` - 卷积神经网络
- `deep_learning_rnn.py` - 循环神经网络、LSTM
- `deep_learning_exercises.py` - 实践练习
- `deep_learning_advanced.py` - GAN、Transformer、强化学习、联邦学习

**工具模块**：
- `ml_evaluation_tools.py` - 交叉验证、性能指标（混淆矩阵、R² 等）
- `ml_visualization.py` - ASCII 图表、学习曲线、决策边界
- `ml_summary.py` - 算法对比表

## 重要约定

### 语言使用
- 所有代码注释、文档字符串和控制台输出都使用**中文**
- 变量名使用英文
- 理论解释详细且使用中文

### 数值稳定性
- 激活函数（sigmoid、tanh）包含溢出预防检查
- 线性回归在拟合前检查接近零的方差
- 数据验证中包含 NaN/Inf 值的错误处理

### 教学特性
线性回归实现包含以下教学功能：
- 学习率对比实验，展示不同学习率的影响
- 损失历史跟踪用于可视化
- 详细输出显示迭代进度
- 带 patience 机制的早停

### 不使用外部机器学习库
不要引入以下依赖：
- scikit-learn
- tensorflow/keras
- pytorch
- pandas（用于机器学习操作）

可能已经使用的可选依赖：
- numpy（用于数值计算）
- matplotlib（用于可视化）

当这些库不可用时，回退到纯 Python 实现。

## Git 提交约定

**重要**：在创建提交时，不要添加任何 AI 工具的署名信息（如 "Generated with Claude Code" 或 "Co-Authored-By: Claude" 等）。保持提交信息简洁明了，只包含实际的更改描述。
