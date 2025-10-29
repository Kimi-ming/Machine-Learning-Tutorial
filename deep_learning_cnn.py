# -*- coding: utf-8 -*-
# 深度学习教程 - 卷积神经网络 (CNN)
# Convolutional Neural Networks: 专门处理图像数据的深度学习架构

import random
import math

def cnn_theory():
    """
    卷积神经网络原理解释
    
    CNN是专门设计用来处理具有网格结构数据（如图像）的深度学习架构。
    
    核心概念：
    - 卷积层：使用卷积核提取局部特征
    - 池化层：降低空间维度，减少参数
    - 全连接层：最终的分类或回归
    - 特征图：卷积操作的输出
    
    CNN的优势：
    1. 局部连接：每个神经元只连接局部区域
    2. 权重共享：同一个卷积核在整个图像上共享
    3. 平移不变性：对图像的平移具有不变性
    4. 层次化特征：从边缘到复杂模式的层次特征
    
    典型架构：
    输入图像 -> 卷积层 -> 激活函数 -> 池化层 -> ... -> 全连接层 -> 输出
    
    重要参数：
    - 卷积核大小：通常使用3x3或5x5
    - 步长(Stride)：卷积核移动的步长
    - 填充(Padding)：边缘填充方式
    - 通道数：输入和输出的通道数量
    """
    print("=== 卷积神经网络 (CNN) 原理 ===")
    print("专门处理图像数据的深度学习架构")
    print()
    print("核心思想：")
    print("• 局部连接：关注图像的局部特征")
    print("• 权重共享：减少参数数量")
    print("• 层次化：从低级特征到高级特征")
    print()

class SimpleCNN:
    """
    简化的卷积神经网络实现
    用于教学目的，展示CNN的基本原理
    """
    
    def __init__(self, input_shape, conv_filters, fc_layers):
        """
        初始化CNN
        input_shape: (height, width, channels)
        conv_filters: 卷积层配置列表 [(filter_num, kernel_size, stride), ...]
        fc_layers: 全连接层神经元数量列表
        """
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.fc_layers = fc_layers
        
        # 初始化卷积核
        self.conv_weights = []
        self.conv_biases = []
        
        in_channels = input_shape[2]
        for filter_num, kernel_size, stride in conv_filters:
            # 初始化卷积核权重
            weights = []
            for _ in range(filter_num):
                kernel = []
                for _ in range(in_channels):
                    channel_kernel = []
                    for _ in range(kernel_size):
                        row = [random.gauss(0, 0.1) for _ in range(kernel_size)]
                        channel_kernel.append(row)
                    kernel.append(channel_kernel)
                weights.append(kernel)
            self.conv_weights.append(weights)
            
            # 初始化偏置
            biases = [0.1 for _ in range(filter_num)]
            self.conv_biases.append(biases)
            
            in_channels = filter_num
        
        print(f"CNN初始化完成:")
        print(f"输入形状: {input_shape}")
        print(f"卷积层: {len(conv_filters)}层")
        print(f"全连接层: {fc_layers}")
    
    def conv2d(self, input_map, kernel, bias, stride=1, padding=0):
        """
        2D卷积操作
        input_map: 输入特征图 (height, width)
        kernel: 卷积核 (kernel_height, kernel_width)
        stride: 步长
        padding: 填充大小
        """
        input_h, input_w = len(input_map), len(input_map[0])
        kernel_h, kernel_w = len(kernel), len(kernel[0])

        # 如果需要padding，先填充输入
        if padding > 0:
            padded = [[0] * (input_w + 2 * padding) for _ in range(input_h + 2 * padding)]
            for i in range(input_h):
                for j in range(input_w):
                    padded[i + padding][j + padding] = input_map[i][j]
            input_map = padded
            input_h, input_w = len(input_map), len(input_map[0])

        output_h = (input_h - kernel_h) // stride + 1
        output_w = (input_w - kernel_w) // stride + 1

        output = []
        for i in range(output_h):
            row = []
            for j in range(output_w):
                # 计算卷积
                conv_sum = 0
                for ki in range(kernel_h):
                    for kj in range(kernel_w):
                        input_i = i * stride + ki
                        input_j = j * stride + kj
                        conv_sum += input_map[input_i][input_j] * kernel[ki][kj]
                conv_sum += bias
                row.append(conv_sum)
            output.append(row)

        return output
    
    def relu(self, feature_map):
        """对特征图应用ReLU激活"""
        return [[max(0, val) for val in row] for row in feature_map]
    
    def max_pooling(self, feature_map, pool_size=2, stride=2):
        """
        最大池化操作
        feature_map: 输入特征图
        pool_size: 池化窗口大小
        stride: 步长
        """
        input_h, input_w = len(feature_map), len(feature_map[0])
        output_h = (input_h - pool_size) // stride + 1
        output_w = (input_w - pool_size) // stride + 1

        output = []
        for i in range(output_h):
            row = []
            for j in range(output_w):
                # 在池化窗口中找最大值
                max_val = float('-inf')
                for pi in range(pool_size):
                    for pj in range(pool_size):
                        input_i = i * stride + pi
                        input_j = j * stride + pj
                        if input_i < input_h and input_j < input_w:
                            max_val = max(max_val, feature_map[input_i][input_j])
                row.append(max_val)
            output.append(row)

        return output

    def average_pooling(self, feature_map, pool_size=2, stride=2):
        """
        平均池化操作
        feature_map: 输入特征图
        pool_size: 池化窗口大小
        stride: 步长
        """
        input_h, input_w = len(feature_map), len(feature_map[0])
        output_h = (input_h - pool_size) // stride + 1
        output_w = (input_w - pool_size) // stride + 1

        output = []
        for i in range(output_h):
            row = []
            for j in range(output_w):
                # 在池化窗口中计算平均值
                pool_sum = 0
                count = 0
                for pi in range(pool_size):
                    for pj in range(pool_size):
                        input_i = i * stride + pi
                        input_j = j * stride + pj
                        if input_i < input_h and input_j < input_w:
                            pool_sum += feature_map[input_i][input_j]
                            count += 1
                avg_val = pool_sum / count if count > 0 else 0
                row.append(avg_val)
            output.append(row)

        return output

    def batch_normalization(self, feature_map, epsilon=1e-5):
        """
        批归一化操作（简化版）
        feature_map: 输入特征图
        epsilon: 防止除零的小常数
        """
        # 计算均值和方差
        all_values = []
        for row in feature_map:
            all_values.extend(row)

        mean = sum(all_values) / len(all_values)
        variance = sum((x - mean) ** 2 for x in all_values) / len(all_values)
        std = math.sqrt(variance + epsilon)

        # 归一化
        normalized = []
        for row in feature_map:
            normalized_row = [(val - mean) / std for val in row]
            normalized.append(normalized_row)

        return normalized

    def dropout(self, feature_map, drop_rate=0.5, training=True):
        """
        Dropout正则化
        feature_map: 输入特征图
        drop_rate: 丢弃比例
        training: 是否在训练模式
        """
        if not training:
            return feature_map

        output = []
        for row in feature_map:
            output_row = []
            for val in row:
                if random.random() > drop_rate:
                    # 保留并缩放
                    output_row.append(val / (1 - drop_rate))
                else:
                    # 丢弃
                    output_row.append(0)
            output.append(output_row)

        return output
    
    def flatten(self, feature_maps):
        """将多维特征图展平为一维向量"""
        flattened = []
        for feature_map in feature_maps:
            for row in feature_map:
                for val in row:
                    flattened.append(val)
        return flattened
    
    def forward(self, input_image):
        """
        前向传播
        input_image: 输入图像或特征图
        """
        current = input_image

        # 卷积层处理
        for i, (filter_num, kernel_size, stride) in enumerate(self.conv_filters):
            # 应用所有卷积核
            conv_outputs = []
            for f in range(filter_num):
                if i == 0:
                    # 第一层：直接从输入图像卷积
                    output = self.conv2d(current,
                                       self.conv_weights[i][f][0],  # 第一个通道的核
                                       self.conv_biases[i][f],
                                       stride,
                                       padding=1)  # 添加padding保持尺寸
                else:
                    # 后续层：处理多通道（简化实现：对所有通道求平均）
                    multi_channel_sum = None
                    for ch_idx, channel_map in enumerate(current):
                        if ch_idx < len(self.conv_weights[i][f]):
                            ch_output = self.conv2d(channel_map,
                                                   self.conv_weights[i][f][ch_idx],
                                                   0,  # 偏置只在最后加一次
                                                   stride,
                                                   padding=1)
                            if multi_channel_sum is None:
                                multi_channel_sum = ch_output
                            else:
                                # 逐元素相加
                                for row_idx in range(len(ch_output)):
                                    for col_idx in range(len(ch_output[0])):
                                        multi_channel_sum[row_idx][col_idx] += ch_output[row_idx][col_idx]

                    # 添加偏置
                    output = multi_channel_sum
                    for row_idx in range(len(output)):
                        for col_idx in range(len(output[0])):
                            output[row_idx][col_idx] += self.conv_biases[i][f]

                # 应用激活函数
                output = self.relu(output)
                conv_outputs.append(output)

            # 池化
            pooled_outputs = []
            for output in conv_outputs:
                pooled = self.max_pooling(output)
                pooled_outputs.append(pooled)

            current = pooled_outputs

        # 展平
        flattened = self.flatten(current)

        return flattened

def conv_operation_demo():
    """卷积操作演示"""
    print("\n=== 卷积操作演示 ===")
    
    # 简单的输入图像 (5x5)
    input_image = [
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6]
    ]
    
    # 边缘检测卷积核 (3x3)
    edge_kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    
    print("输入图像 (5x5):")
    for row in input_image:
        print([f"{val:2d}" for val in row])
    
    print("\n边缘检测卷积核 (3x3):")
    for row in edge_kernel:
        print([f"{val:2d}" for val in row])
    
    # 执行卷积
    cnn = SimpleCNN((5, 5, 1), [(1, 3, 1)], [10])
    output = cnn.conv2d(input_image, edge_kernel, 0)
    
    print(f"\n卷积输出 ({len(output)}x{len(output[0])}):")
    for row in output:
        print([f"{val:6.1f}" for val in row])
    
    print("\n卷积操作解释:")
    print("• 卷积核在输入图像上滑动")
    print("• 每个位置计算元素对应相乘再求和")
    print("• 边缘检测核能够突出边缘信息")

def pooling_demo():
    """池化操作演示"""
    print("\n=== 池化操作演示 ===")
    
    # 特征图 (4x4)
    feature_map = [
        [1.5, 2.3, 0.8, 1.2],
        [3.1, 4.7, 2.1, 0.9],
        [0.6, 1.8, 3.4, 2.7],
        [2.2, 0.4, 1.1, 3.8]
    ]
    
    print("输入特征图 (4x4):")
    for row in feature_map:
        print([f"{val:4.1f}" for val in row])
    
    cnn = SimpleCNN((4, 4, 1), [(1, 2, 1)], [5])
    pooled = cnn.max_pooling(feature_map, pool_size=2, stride=2)
    
    print(f"\n最大池化输出 (2x2):")
    for row in pooled:
        print([f"{val:4.1f}" for val in row])
    
    print("\n池化操作解释:")
    print("• 在2x2窗口中选择最大值")
    print("• 减少特征图尺寸，保留重要特征")
    print("• 增加平移不变性，减少过拟合")

def feature_hierarchy_demo():
    """特征层次演示"""
    print("\n=== CNN特征层次演示 ===")
    
    print("CNN学习的特征层次:")
    print()
    
    layers = [
        ("第1层", "边缘和线条", ["水平线", "垂直线", "斜线", "曲线"]),
        ("第2层", "简单形状", ["角落", "圆形", "方形", "T型连接"]),
        ("第3层", "复杂模式", ["纹理", "简单对象部分", "重复模式"]),
        ("第4层", "对象部分", ["眼睛", "轮子", "窗户", "门"]),
        ("第5层", "完整对象", ["人脸", "汽车", "房子", "动物"])
    ]
    
    for layer, description, features in layers:
        print(f"{layer}: {description}")
        print(f"    特征例子: {', '.join(features)}")
        print()
    
    print("特征演化过程:")
    print("• 低层：检测基本的边缘和纹理")
    print("• 中层：组合成简单的形状和模式")  
    print("• 高层：识别完整的对象和场景")
    print("• 每层建立在前一层的基础上")

def famous_cnn_architectures():
    """著名的CNN架构介绍"""
    print("\n=== 著名CNN架构 ===")
    
    architectures = {
        "LeNet-5 (1998)": {
            "贡献": "第一个成功的CNN架构",
            "结构": "Conv-Pool-Conv-Pool-FC-FC",
            "应用": "手写数字识别",
            "特点": ["使用Sigmoid激活", "较小的网络"]
        },
        
        "AlexNet (2012)": {
            "贡献": "深度学习在ImageNet的突破",
            "结构": "更深的CNN + ReLU + Dropout",
            "应用": "大规模图像分类",
            "特点": ["使用ReLU激活", "Dropout正则化", "GPU训练"]
        },
        
        "VGG (2014)": {
            "贡献": "证明深度的重要性",
            "结构": "多个3x3卷积堆叠",
            "应用": "特征提取backbone",
            "特点": ["统一使用3x3卷积", "网络更深", "结构简洁"]
        },
        
        "ResNet (2015)": {
            "贡献": "解决深度网络退化问题",
            "结构": "残差连接 + 批归一化",
            "应用": "超深网络训练",
            "特点": ["跳跃连接", "150+层深度", "批归一化"]
        },
        
        "EfficientNet (2019)": {
            "贡献": "平衡准确性和效率",
            "结构": "复合缩放 + 移动卷积",
            "应用": "高效图像分类",
            "特点": ["复合缩放", "自动搜索", "高效设计"]
        }
    }
    
    for name, info in architectures.items():
        print(f"【{name}】")
        print(f"贡献: {info['贡献']}")
        print(f"结构: {info['结构']}")
        print(f"应用: {info['应用']}")
        print(f"特点: {', '.join(info['特点'])}")
        print()

def cnn_applications():
    """CNN应用领域"""
    print("=== CNN主要应用领域 ===")
    
    applications = {
        "图像分类": {
            "任务": "判断图像属于哪个类别",
            "例子": ["动物识别", "物体分类", "场景识别"],
            "架构": "标准CNN + 全连接分类器"
        },
        
        "目标检测": {
            "任务": "定位和识别图像中的多个对象",
            "例子": ["人脸检测", "车辆检测", "行人检测"],
            "架构": "R-CNN, YOLO, SSD等"
        },
        
        "图像分割": {
            "任务": "像素级别的图像理解",
            "例子": ["医学图像分割", "自动驾驶", "背景去除"],
            "架构": "U-Net, DeepLab, Mask R-CNN"
        },
        
        "风格迁移": {
            "任务": "将一种艺术风格应用到另一张图像",
            "例子": ["艺术风格化", "照片卡通化", "图像增强"],
            "架构": "生成对抗网络 + CNN"
        },
        
        "医学影像": {
            "任务": "辅助医学诊断",
            "例子": ["肺部CT分析", "皮肤癌检测", "眼底病变"],
            "架构": "定制化CNN + 注意力机制"
        }
    }
    
    for field, info in applications.items():
        print(f"【{field}】")
        print(f"任务: {info['任务']}")
        print(f"应用: {', '.join(info['例子'])}")
        print(f"常用架构: {info['架构']}")
        print()

def cnn_training_tips():
    """CNN训练技巧"""
    print("=== CNN训练技巧 ===")
    
    print("数据预处理:")
    print("• 图像归一化: 将像素值缩放到[0,1]或[-1,1]")
    print("• 数据增强: 旋转、缩放、裁剪、翻转等")
    print("• 批量处理: 使用合适的批大小")
    print()
    
    print("网络设计:")
    print("• 使用3x3卷积核作为基础构建块")
    print("• 添加批归一化加速训练")
    print("• 使用残差连接训练深层网络")
    print("• 合理设置通道数和层数")
    print()
    
    print("训练策略:")
    print("• 使用预训练模型进行迁移学习")
    print("• 采用学习率衰减策略")
    print("• 监控验证集避免过拟合")
    print("• 使用合适的损失函数")
    print()
    
    print("调试技巧:")
    print("• 可视化卷积核和特征图")
    print("• 监控梯度流动")
    print("• 检查数据加载和预处理")
    print("• 从简单模型开始逐步增加复杂度")

def padding_demo():
    """Padding操作演示"""
    print("\n=== Padding操作演示 ===")

    # 简单的输入图像 (3x3)
    input_image = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("原始输入图像 (3x3):")
    for row in input_image:
        print([f"{val:2d}" for val in row])

    # 演示不同padding
    cnn = SimpleCNN((3, 3, 1), [(1, 3, 1)], [10])

    # 3x3卷积核
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]

    print("\n卷积核 (3x3) - 垂直边缘检测:")
    for row in kernel:
        print([f"{val:2d}" for val in row])

    # 无padding
    output_no_pad = cnn.conv2d(input_image, kernel, 0, stride=1, padding=0)
    print(f"\n无Padding输出 ({len(output_no_pad)}x{len(output_no_pad[0])}):")
    for row in output_no_pad:
        print([f"{val:6.1f}" for val in row])

    # padding=1
    output_pad1 = cnn.conv2d(input_image, kernel, 0, stride=1, padding=1)
    print(f"\nPadding=1输出 ({len(output_pad1)}x{len(output_pad1[0])}):")
    for row in output_pad1:
        print([f"{val:6.1f}" for val in row])

    print("\nPadding作用:")
    print("• 无Padding: 输出尺寸缩小 (3x3 -> 1x1)")
    print("• Padding=1: 保持输出尺寸 (3x3 -> 3x3)")
    print("• 使用Padding可以构建更深的网络")

def batch_norm_demo():
    """批归一化演示"""
    print("\n=== 批归一化演示 ===")

    # 特征图 (3x3)
    feature_map = [
        [100, 150, 200],
        [120, 180, 220],
        [110, 160, 210]
    ]

    print("原始特征图:")
    for row in feature_map:
        print([f"{val:6.1f}" for val in row])

    cnn = SimpleCNN((3, 3, 1), [(1, 2, 1)], [5])
    normalized = cnn.batch_normalization(feature_map)

    print("\n批归一化后:")
    for row in normalized:
        print([f"{val:6.3f}" for val in row])

    print("\n批归一化效果:")
    print("• 将特征值标准化为均值0、方差1")
    print("• 加速训练收敛")
    print("• 缓解梯度消失/爆炸问题")
    print("• 具有轻微的正则化效果")

def pooling_comparison_demo():
    """不同池化方法对比"""
    print("\n=== 池化方法对比 ===")

    feature_map = [
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [2.0, 1.0, 3.0, 2.0],
        [4.0, 5.0, 6.0, 7.0]
    ]

    print("输入特征图 (4x4):")
    for row in feature_map:
        print([f"{val:4.1f}" for val in row])

    cnn = SimpleCNN((4, 4, 1), [(1, 2, 1)], [5])

    # 最大池化
    max_pooled = cnn.max_pooling(feature_map, pool_size=2, stride=2)
    print("\n最大池化 (2x2):")
    for row in max_pooled:
        print([f"{val:4.1f}" for val in row])

    # 平均池化
    avg_pooled = cnn.average_pooling(feature_map, pool_size=2, stride=2)
    print("\n平均池化 (2x2):")
    for row in avg_pooled:
        print([f"{val:4.1f}" for val in row])

    print("\n池化方法对比:")
    print("• 最大池化: 保留最显著特征,常用于分类任务")
    print("• 平均池化: 保留整体信息,常用于特征降维")
    print("• 最大池化对噪声更鲁棒")

def receptive_field_demo():
    """感受野演示"""
    print("\n=== 感受野演示 ===")

    print("感受野（Receptive Field）：")
    print("输出特征图中的一个神经元能够'看到'的输入图像的区域大小")
    print()

    layers_info = [
        ("输入层", "7x7图像", 1, "每个像素看到自己"),
        ("第1层卷积", "3x3卷积,stride=1", 3, "每个输出看到3x3输入"),
        ("第1层池化", "2x2池化,stride=2", 4, "感受野增加到4x4"),
        ("第2层卷积", "3x3卷积,stride=1", 8, "感受野增加到8x8"),
        ("第2层池化", "2x2池化,stride=2", 10, "感受野增加到10x10")
    ]

    print("网络结构与感受野变化:")
    print(f"{'层':<12} {'操作':<20} {'感受野':<8} {'说明'}")
    print("-" * 70)
    for layer, operation, rf, desc in layers_info:
        print(f"{layer:<12} {operation:<20} {rf}x{rf:<6} {desc}")

    print("\n感受野的重要性:")
    print("• 更大的感受野能够捕获更全局的信息")
    print("• 深层网络的感受野呈指数增长")
    print("• 空洞卷积可以在不增加参数的情况下扩大感受野")

def kernel_visualization():
    """卷积核可视化演示"""
    print("\n=== 常见卷积核可视化 ===")

    kernels = {
        "垂直边缘检测": [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        "水平边缘检测": [
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ],
        "锐化": [
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ],
        "模糊": [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ],
        "浮雕": [
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ]
    }

    for name, kernel in kernels.items():
        print(f"\n【{name}】")
        for row in kernel:
            print("  " + " ".join(f"{val:6.2f}" for val in row))

    print("\n卷积核特点:")
    print("• 边缘检测核: 突出图像中的边缘信息")
    print("• 锐化核: 增强图像细节")
    print("• 模糊核: 平滑图像,去除噪声")
    print("• CNN能自动学习最优的卷积核")

def simple_image_classification_example():
    """简单的图像分类示例"""
    print("\n=== 简单图像分类示例 ===")
    
    # 模拟简单的二分类问题：识别图像是否包含边缘
    print("任务：识别5x5图像是否包含强边缘")
    print()
    
    # 生成训练数据
    def generate_edge_data():
        """生成包含边缘的训练数据"""
        # 有边缘的图像（垂直边缘）
        edge_images = [
            [[1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0]],
            
            [[0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0]]
        ]
        
        # 无边缘的图像（均匀分布）
        no_edge_images = [
            [[0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5]],
             
            [[0.3, 0.4, 0.3, 0.4, 0.3],
             [0.4, 0.3, 0.4, 0.3, 0.4],
             [0.3, 0.4, 0.3, 0.4, 0.3],
             [0.4, 0.3, 0.4, 0.3, 0.4],
             [0.3, 0.4, 0.3, 0.4, 0.3]]
        ]
        
        X = edge_images + no_edge_images
        y = [1, 1, 0, 0]  # 1表示有边缘，0表示无边缘
        
        return X, y
    
    X_train, y_train = generate_edge_data()
    
    print("训练数据:")
    for i, (image, label) in enumerate(zip(X_train, y_train)):
        print(f"图像{i+1} (标签: {'有边缘' if label else '无边缘'}):")
        for row in image:
            print("  " + " ".join(f"{val:3.1f}" for val in row))
        print()
    
    # 使用简单的特征提取方法
    def extract_edge_features(image):
        """提取边缘特征"""
        # 计算水平和垂直梯度
        h_grad = 0
        v_grad = 0
        
        for i in range(len(image)-1):
            for j in range(len(image[0])-1):
                # 水平梯度
                h_grad += abs(image[i][j+1] - image[i][j])
                # 垂直梯度
                v_grad += abs(image[i+1][j] - image[i][j])
        
        return [h_grad, v_grad]
    
    # 提取所有图像的特征
    features = [extract_edge_features(img) for img in X_train]
    
    print("提取的边缘特征 (水平梯度, 垂直梯度):")
    for i, (feat, label) in enumerate(zip(features, y_train)):
        print(f"图像{i+1}: {feat[0]:.2f}, {feat[1]:.2f} -> {'有边缘' if label else '无边缘'}")
    
    print("\n分析:")
    print("• 有边缘的图像梯度值较大")
    print("• 无边缘的图像梯度值较小")
    print("• CNN可以自动学习这种特征")

def main():
    """主函数"""
    print("卷积神经网络 (CNN) 教程")
    print("=" * 50)

    # 基础理论
    cnn_theory()

    # 核心操作演示
    conv_operation_demo()
    padding_demo()
    pooling_demo()
    pooling_comparison_demo()

    # 高级概念
    batch_norm_demo()
    kernel_visualization()
    receptive_field_demo()
    feature_hierarchy_demo()

    # 架构与应用
    famous_cnn_architectures()
    cnn_applications()

    # 实践示例
    simple_image_classification_example()
    cnn_training_tips()

    print("\n" + "=" * 50)
    print("CNN学习要点总结")
    print()
    print("核心概念:")
    print("• 卷积操作：提取局部特征,支持Padding保持尺寸")
    print("• 池化操作：降维和抽象(最大池化/平均池化)")
    print("• 批归一化：加速训练,稳定梯度")
    print("• 特征层次：从简单边缘到复杂对象")
    print("• 权重共享：大幅减少参数数量")
    print("• 感受野：理解网络'看'多大范围的输入")
    print()
    print("实践建议:")
    print("• 理解卷积和池化的数学原理")
    print("• 学习经典CNN架构(LeNet, AlexNet, ResNet等)")
    print("• 掌握数据增强技术提升泛化能力")
    print("• 使用预训练模型进行迁移学习")
    print("• 可视化卷积核和特征图辅助调试")
    print()
    print("下一步学习:")
    print("• 实现完整的图像分类项目")
    print("• 学习目标检测算法(YOLO, Faster R-CNN)")
    print("• 了解语义分割技术(U-Net, DeepLab)")
    print("• 探索生成对抗网络(GANs)")
    print("• 研究注意力机制和Transformer在视觉中的应用")

if __name__ == "__main__":
    main()