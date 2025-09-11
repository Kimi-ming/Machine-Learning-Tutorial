# -*- coding: utf-8 -*-
# 深度学习实践练习
# Deep Learning Exercises: 动手实践深度学习项目

import random
import math
import json

def exercises_introduction():
    """深度学习练习介绍"""
    print("=== 深度学习实践练习 ===")
    print("通过动手实践巩固深度学习知识")
    print()
    print("练习特点:")
    print("• 循序渐进：从基础到高级")
    print("• 实践导向：解决实际问题")
    print("• 代码实现：亲手编写算法")
    print("• 深入理解：掌握底层原理")
    print()

class DeepLearningProject:
    """深度学习项目基类"""
    
    def __init__(self, name, difficulty, description):
        self.name = name
        self.difficulty = difficulty  # 1-5: 简单到困难
        self.description = description
        self.completed = False
    
    def get_info(self):
        """获取项目信息"""
        difficulty_stars = "★" * self.difficulty + "☆" * (5 - self.difficulty)
        status = "✅ 已完成" if self.completed else "⏳ 待完成"
        
        return f"""
【{self.name}】
难度: {difficulty_stars} ({self.difficulty}/5)
状态: {status}
描述: {self.description}
"""

def exercise_1_perceptron():
    """练习1：实现感知机"""
    print("\n" + "="*50)
    print("练习 1: 实现单层感知机")
    print("="*50)
    
    class Perceptron:
        """单层感知机实现"""
        
        def __init__(self, input_size, learning_rate=0.1):
            # 随机初始化权重和偏置
            self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
            self.bias = random.uniform(-1, 1)
            self.learning_rate = learning_rate
            self.training_errors = []
            
        def activation(self, x):
            """阶跃激活函数"""
            return 1 if x >= 0 else 0
            
        def predict(self, inputs):
            """预测"""
            weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
            weighted_sum += self.bias
            return self.activation(weighted_sum)
            
        def train(self, training_data, epochs=100):
            """训练感知机"""
            for epoch in range(epochs):
                errors = 0
                for inputs, target in training_data:
                    prediction = self.predict(inputs)
                    error = target - prediction
                    
                    # 更新权重和偏置
                    if error != 0:
                        errors += 1
                        for i in range(len(self.weights)):
                            self.weights[i] += self.learning_rate * error * inputs[i]
                        self.bias += self.learning_rate * error
                
                self.training_errors.append(errors)
                if errors == 0:
                    print(f"训练在第 {epoch + 1} 轮收敛！")
                    break
    
    # 任务：学习AND逻辑门
    print("\n任务：学习AND逻辑门")
    print("AND逻辑真值表：")
    print("输入1 | 输入2 | 输出")
    print("-----|-------|-----")
    print("  0  |   0   |  0")
    print("  1  |   0   |  0") 
    print("  0  |   1   |  0")
    print("  1  |   1   |  1")
    
    # 训练数据
    and_data = [
        ([0, 0], 0),
        ([1, 0], 0),
        ([0, 1], 0),
        ([1, 1], 1)
    ]
    
    # 创建和训练感知机
    perceptron = Perceptron(input_size=2)
    print(f"\n初始权重: [{perceptron.weights[0]:.3f}, {perceptron.weights[1]:.3f}]")
    print(f"初始偏置: {perceptron.bias:.3f}")
    
    perceptron.train(and_data, epochs=100)
    
    print(f"\n最终权重: [{perceptron.weights[0]:.3f}, {perceptron.weights[1]:.3f}]")
    print(f"最终偏置: {perceptron.bias:.3f}")
    
    # 测试结果
    print(f"\n测试结果：")
    print("输入 | 预测 | 实际")
    print("----|------|----")
    for inputs, target in and_data:
        prediction = perceptron.predict(inputs)
        print(f"{inputs} |  {prediction}   |  {target}")
    
    print(f"\n练习总结：")
    print("• 感知机能够学习线性可分的问题")
    print("• 权重更新规则：w = w + α(t-y)x")
    print("• AND门是线性可分的，所以能够收敛")
    print("• 尝试XOR门会发现无法收敛（需要多层网络）")

def exercise_2_mlp_xor():
    """练习2：多层感知机解决XOR问题"""
    print("\n" + "="*50)
    print("练习 2: 多层感知机解决XOR问题")
    print("="*50)
    
    class MLP:
        """多层感知机"""
        
        def __init__(self):
            # 网络结构: 2输入 -> 2隐藏 -> 1输出
            # 隐藏层权重
            self.W1 = [[random.uniform(-2, 2) for _ in range(2)] for _ in range(2)]
            self.b1 = [random.uniform(-1, 1) for _ in range(2)]
            
            # 输出层权重  
            self.W2 = [random.uniform(-2, 2) for _ in range(2)]
            self.b2 = random.uniform(-1, 1)
            
            self.learning_rate = 1.0
            
        def sigmoid(self, x):
            """Sigmoid激活函数"""
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
            
        def sigmoid_derivative(self, x):
            """Sigmoid导数"""
            s = self.sigmoid(x)
            return s * (1 - s)
            
        def forward(self, inputs):
            """前向传播"""
            # 隐藏层
            z1 = []
            for i in range(2):
                z = sum(self.W1[i][j] * inputs[j] for j in range(2)) + self.b1[i]
                z1.append(z)
            a1 = [self.sigmoid(z) for z in z1]
            
            # 输出层
            z2 = sum(self.W2[i] * a1[i] for i in range(2)) + self.b2
            a2 = self.sigmoid(z2)
            
            return z1, a1, z2, a2
            
        def backward(self, inputs, target):
            """反向传播"""
            # 前向传播
            z1, a1, z2, a2 = self.forward(inputs)
            
            # 输出层误差
            output_error = (a2 - target) * self.sigmoid_derivative(z2)
            
            # 隐藏层误差
            hidden_errors = []
            for i in range(2):
                error = output_error * self.W2[i] * self.sigmoid_derivative(z1[i])
                hidden_errors.append(error)
            
            # 更新输出层权重
            for i in range(2):
                self.W2[i] -= self.learning_rate * output_error * a1[i]
            self.b2 -= self.learning_rate * output_error
            
            # 更新隐藏层权重
            for i in range(2):
                for j in range(2):
                    self.W1[i][j] -= self.learning_rate * hidden_errors[i] * inputs[j]
                self.b1[i] -= self.learning_rate * hidden_errors[i]
            
            return a2
            
        def train(self, training_data, epochs=5000):
            """训练网络"""
            for epoch in range(epochs):
                total_error = 0
                for inputs, target in training_data:
                    output = self.backward(inputs, target)
                    total_error += 0.5 * (output - target) ** 2
                    
                if epoch % 1000 == 0:
                    print(f"轮次 {epoch:4d}: 错误 = {total_error:.6f}")
    
    print("\n任务：学习XOR逻辑门")
    print("XOR逻辑真值表：")
    print("输入1 | 输入2 | 输出")
    print("-----|-------|-----")
    print("  0  |   0   |  0")
    print("  1  |   0   |  1")
    print("  0  |   1   |  1") 
    print("  1  |   1   |  0")
    
    # XOR训练数据
    xor_data = [
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 1], 0)
    ]
    
    # 训练网络
    mlp = MLP()
    print(f"\n开始训练多层感知机...")
    mlp.train(xor_data, epochs=5000)
    
    # 测试结果
    print(f"\n测试结果：")
    print("输入    | 输出   | 目标")
    print("--------|-------|-----")
    for inputs, target in xor_data:
        _, _, _, output = mlp.forward(inputs)
        print(f"[{inputs[0]}, {inputs[1]}] | {output:.3f} | {target}")
    
    print(f"\n练习总结：")
    print("• XOR问题是非线性的，需要隐藏层")
    print("• 多层感知机通过反向传播学习")
    print("• 隐藏层使网络能够学习复杂模式")
    print("• Sigmoid激活函数引入非线性")

def exercise_3_mini_cnn():
    """练习3：迷你CNN实现"""
    print("\n" + "="*50)
    print("练习 3: 迷你卷积神经网络")
    print("="*50)
    
    class MiniCNN:
        """迷你CNN用于理解卷积概念"""
        
        def __init__(self):
            # 简单的3x3卷积核
            self.conv_filter = [
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ]
            
        def convolution(self, image, kernel, stride=1):
            """2D卷积操作"""
            img_h, img_w = len(image), len(image[0])
            ker_h, ker_w = len(kernel), len(kernel[0])
            
            out_h = (img_h - ker_h) // stride + 1
            out_w = (img_w - ker_w) // stride + 1
            
            output = []
            for i in range(out_h):
                row = []
                for j in range(out_w):
                    conv_sum = 0
                    for ki in range(ker_h):
                        for kj in range(ker_w):
                            img_i = i * stride + ki
                            img_j = j * stride + kj
                            conv_sum += image[img_i][img_j] * kernel[ki][kj]
                    row.append(conv_sum)
                output.append(row)
            
            return output
            
        def relu(self, feature_map):
            """ReLU激活"""
            return [[max(0, val) for val in row] for row in feature_map]
            
        def max_pooling(self, feature_map, pool_size=2):
            """最大池化"""
            img_h, img_w = len(feature_map), len(feature_map[0])
            out_h = img_h // pool_size
            out_w = img_w // pool_size
            
            output = []
            for i in range(out_h):
                row = []
                for j in range(out_w):
                    max_val = float('-inf')
                    for pi in range(pool_size):
                        for pj in range(pool_size):
                            img_i = i * pool_size + pi
                            img_j = j * pool_size + pj
                            if img_i < img_h and img_j < img_w:
                                max_val = max(max_val, feature_map[img_i][img_j])
                    row.append(max_val)
                output.append(row)
                
            return output
    
    print("\n任务：边缘检测卷积")
    
    # 创建测试图像（包含边缘）
    test_image = [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ]
    
    print("输入图像 (5x5):")
    for row in test_image:
        print("  " + " ".join(f"{val}" for val in row))
    
    # 创建CNN并处理
    cnn = MiniCNN()
    
    print(f"\n边缘检测卷积核:")
    for row in cnn.conv_filter:
        print("  " + " ".join(f"{val:2d}" for val in row))
    
    # 卷积操作
    conv_output = cnn.convolution(test_image, cnn.conv_filter)
    print(f"\n卷积输出 ({len(conv_output)}x{len(conv_output[0])}):")
    for row in conv_output:
        print("  " + " ".join(f"{val:4.0f}" for val in row))
    
    # ReLU激活
    relu_output = cnn.relu(conv_output)
    print(f"\nReLU后:")
    for row in relu_output:
        print("  " + " ".join(f"{val:4.0f}" for val in row))
    
    # 最大池化
    pooled_output = cnn.max_pooling(relu_output)
    print(f"\n最大池化 ({len(pooled_output)}x{len(pooled_output[0])}):")
    for row in pooled_output:
        print("  " + " ".join(f"{val:4.0f}" for val in row))
    
    print(f"\n练习总结：")
    print("• 卷积操作提取局部特征")
    print("• 边缘检测核突出边缘信息")
    print("• ReLU去除负值，增加非线性")
    print("• 池化降低维度，保留重要特征")

def exercise_4_simple_rnn():
    """练习4：简单RNN实现"""
    print("\n" + "="*50)
    print("练习 4: 简单循环神经网络")
    print("="*50)
    
    class SimpleRNN:
        """简单RNN实现"""
        
        def __init__(self, input_size, hidden_size):
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # 权重矩阵
            self.Wxh = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] 
                       for _ in range(input_size)]
            self.Whh = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] 
                       for _ in range(hidden_size)]
            self.bh = [0.0 for _ in range(hidden_size)]
            
        def tanh(self, x):
            """tanh激活函数"""
            return math.tanh(x)
            
        def matrix_vector_mult(self, matrix, vector):
            """矩阵向量乘法"""
            result = []
            for row in matrix:
                dot = sum(m * v for m, v in zip(row, vector))
                result.append(dot)
            return result
            
        def forward(self, inputs):
            """前向传播"""
            hidden = [0.0] * self.hidden_size
            outputs = []
            
            for input_vec in inputs:
                # h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
                h_input = self.matrix_vector_mult(self.Wxh, input_vec)
                h_hidden = self.matrix_vector_mult(self.Whh, hidden)
                
                new_hidden = []
                for i in range(self.hidden_size):
                    h_val = h_input[i] + h_hidden[i] + self.bh[i]
                    new_hidden.append(self.tanh(h_val))
                
                hidden = new_hidden
                outputs.append(hidden[:])  # 复制当前隐藏状态
                
            return outputs
    
    print("\n任务：序列记忆测试")
    print("输入序列：[1, 0, 1] -> 期望RNN能记住序列模式")
    
    # 创建RNN
    rnn = SimpleRNN(input_size=1, hidden_size=3)
    
    # 输入序列（每个元素转为向量）
    input_sequence = [[1], [0], [1]]
    
    print(f"输入序列：")
    for i, inp in enumerate(input_sequence):
        print(f"  时刻{i+1}: {inp}")
    
    # 前向传播
    hidden_states = rnn.forward(input_sequence)
    
    print(f"\n隐藏状态演化：")
    for i, hidden in enumerate(hidden_states):
        print(f"  时刻{i+1}: [{', '.join(f'{h:.3f}' for h in hidden)}]")
    
    print(f"\n分析：")
    print("• 每个时刻的隐藏状态都包含了历史信息")
    print("• 隐藏状态随着输入序列不断更新")
    print("• RNN通过隐藏状态传递序列信息")
    
    # 测试序列变化的影响
    print(f"\n测试不同序列：")
    test_sequences = [
        [[1], [1], [1]],
        [[0], [0], [0]], 
        [[1], [0], [0]]
    ]
    
    for seq in test_sequences:
        final_hidden = rnn.forward(seq)[-1]
        seq_str = ''.join(str(x[0]) for x in seq)
        print(f"序列 [{seq_str}] -> 最终状态: [{', '.join(f'{h:.3f}' for h in final_hidden)}]")
    
    print(f"\n练习总结：")
    print("• RNN能够处理变长序列")
    print("• 隐藏状态承载序列记忆")
    print("• 不同序列产生不同的最终状态")
    print("• 为处理语言、时间序列等提供基础")

def exercise_5_gradient_descent():
    """练习5：梯度下降可视化"""
    print("\n" + "="*50)
    print("练习 5: 梯度下降优化可视化")
    print("="*50)
    
    def quadratic_function(x):
        """二次函数 f(x) = x^2 - 4x + 3"""
        return x**2 - 4*x + 3
        
    def quadratic_derivative(x):
        """二次函数导数 f'(x) = 2x - 4"""
        return 2*x - 4
    
    class GradientDescent:
        """梯度下降优化器"""
        
        def __init__(self, learning_rate=0.1):
            self.learning_rate = learning_rate
            self.history = []
            
        def optimize(self, initial_x, max_iterations=50):
            """执行梯度下降"""
            x = initial_x
            
            for i in range(max_iterations):
                # 计算函数值和梯度
                fx = quadratic_function(x)
                grad = quadratic_derivative(x)
                
                # 记录历史
                self.history.append((x, fx, grad))
                
                # 检查收敛
                if abs(grad) < 1e-6:
                    print(f"在第 {i+1} 次迭代时收敛！")
                    break
                
                # 更新参数
                x = x - self.learning_rate * grad
                
            return x
    
    print("\n任务：找到函数 f(x) = x² - 4x + 3 的最小值")
    print("理论最小值在 x = 2 处，f(2) = -1")
    
    # 测试不同学习率
    learning_rates = [0.01, 0.1, 0.5]
    starting_point = 0.0
    
    for lr in learning_rates:
        print(f"\n--- 学习率: {lr} ---")
        optimizer = GradientDescent(learning_rate=lr)
        final_x = optimizer.optimize(starting_point)
        final_fx = quadratic_function(final_x)
        
        print(f"最终位置: x = {final_x:.4f}")
        print(f"最终函数值: f(x) = {final_fx:.4f}")
        print(f"迭代次数: {len(optimizer.history)}")
        
        # 显示前几次迭代
        print("前5次迭代:")
        print("迭代 |    x    |  f(x)  | 梯度  ")
        print("----|---------|--------|-------")
        for i in range(min(5, len(optimizer.history))):
            x, fx, grad = optimizer.history[i]
            print(f" {i:2d} | {x:7.3f} | {fx:6.3f} | {grad:5.2f}")
    
    print(f"\n分析不同学习率的影响：")
    print("• 学习率太小：收敛慢，需要更多迭代")
    print("• 学习率适中：快速收敛到最优解")
    print("• 学习率太大：可能震荡或发散")
    
    # 演示不同起始点
    print(f"\n测试不同起始点（学习率=0.1）：")
    start_points = [-2, 0, 5, 10]
    
    for start_x in start_points:
        optimizer = GradientDescent(learning_rate=0.1)
        final_x = optimizer.optimize(start_x)
        final_fx = quadratic_function(final_x)
        
        print(f"起始点 {start_x:2d} -> 最终点 {final_x:.3f} (f={final_fx:.3f}) 用了 {len(optimizer.history)} 次迭代")
    
    print(f"\n练习总结：")
    print("• 梯度下降沿负梯度方向移动")
    print("• 学习率控制步长大小")
    print("• 凸函数能保证收敛到全局最优")
    print("• 深度学习中梯度下降优化网络权重")

def exercise_6_backpropagation():
    """练习6：手工计算反向传播"""
    print("\n" + "="*50)
    print("练习 6: 反向传播算法手工计算")
    print("="*50)
    
    print("网络结构：输入层(2) -> 隐藏层(2) -> 输出层(1)")
    print("激活函数：Sigmoid")
    print("损失函数：均方误差")
    
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # 网络参数（固定值用于演示）
    print(f"\n网络参数：")
    
    # 隐藏层权重和偏置
    W1 = [[0.5, 0.3], [0.2, 0.8]]  # 2x2
    b1 = [0.1, 0.4]
    
    # 输出层权重和偏置  
    W2 = [0.6, 0.7]  # 1x2
    b2 = 0.2
    
    print(f"隐藏层权重 W1:")
    for i, row in enumerate(W1):
        print(f"  神经元{i+1}: {row}")
    print(f"隐藏层偏置 b1: {b1}")
    print(f"输出层权重 W2: {W2}")
    print(f"输出层偏置 b2: {b2}")
    
    # 输入和目标
    inputs = [0.8, 0.2]
    target = 0.9
    
    print(f"\n输入: {inputs}")
    print(f"目标输出: {target}")
    
    print(f"\n=== 前向传播 ===")
    
    # 隐藏层计算
    print(f"隐藏层计算:")
    z1 = []
    a1 = []
    for i in range(2):
        z = sum(W1[i][j] * inputs[j] for j in range(2)) + b1[i]
        z1.append(z)
        a = sigmoid(z)
        a1.append(a)
        print(f"  神经元{i+1}: z={z:.4f}, a=sigmoid({z:.4f})={a:.4f}")
    
    # 输出层计算
    print(f"输出层计算:")
    z2 = sum(W2[i] * a1[i] for i in range(2)) + b2
    a2 = sigmoid(z2)
    print(f"  z={z2:.4f}, output=sigmoid({z2:.4f})={a2:.4f}")
    
    # 损失计算
    loss = 0.5 * (a2 - target)**2
    print(f"\n损失: L = 0.5*(预测-目标)² = 0.5*({a2:.4f}-{target})² = {loss:.4f}")
    
    print(f"\n=== 反向传播 ===")
    
    # 输出层误差
    print(f"输出层误差:")
    delta2 = (a2 - target) * sigmoid_derivative(z2)
    print(f"  δ₂ = (a₂-t) * σ'(z₂) = ({a2:.4f}-{target}) * {sigmoid_derivative(z2):.4f} = {delta2:.4f}")
    
    # 隐藏层误差
    print(f"隐藏层误差:")
    delta1 = []
    for i in range(2):
        d = delta2 * W2[i] * sigmoid_derivative(z1[i])
        delta1.append(d)
        print(f"  δ₁[{i+1}] = δ₂ * W₂[{i+1}] * σ'(z₁[{i+1}]) = {delta2:.4f} * {W2[i]} * {sigmoid_derivative(z1[i]):.4f} = {d:.4f}")
    
    # 权重梯度
    print(f"\n权重梯度:")
    learning_rate = 0.5
    
    # 输出层权重梯度
    print(f"输出层权重梯度:")
    for i in range(2):
        grad = delta2 * a1[i]
        new_weight = W2[i] - learning_rate * grad
        print(f"  ∂L/∂W₂[{i+1}] = δ₂ * a₁[{i+1}] = {delta2:.4f} * {a1[i]:.4f} = {grad:.4f}")
        print(f"  W₂[{i+1}] = {W2[i]} - {learning_rate}*{grad:.4f} = {new_weight:.4f}")
    
    # 隐藏层权重梯度
    print(f"隐藏层权重梯度:")
    for i in range(2):
        for j in range(2):
            grad = delta1[i] * inputs[j]
            new_weight = W1[i][j] - learning_rate * grad
            print(f"  ∂L/∂W₁[{i+1}][{j+1}] = δ₁[{i+1}] * x[{j+1}] = {delta1[i]:.4f} * {inputs[j]} = {grad:.4f}")
            print(f"  W₁[{i+1}][{j+1}] = {W1[i][j]} - {learning_rate}*{grad:.4f} = {new_weight:.4f}")
    
    print(f"\n练习总结：")
    print("• 反向传播使用链式法则计算梯度")
    print("• 误差从输出层向输入层传播")
    print("• 每层的梯度依赖于下一层的误差")
    print("• 权重更新使用梯度下降规则")

def create_practice_roadmap():
    """创建深度学习实践路线图"""
    print("\n" + "="*50)
    print("深度学习实践路线图")
    print("="*50)
    
    roadmap = {
        "初级阶段": [
            ("感知机实现", "理解基本的线性分类器"),
            ("多层感知机", "掌握反向传播算法"),
            ("激活函数对比", "理解非线性变换的作用"),
            ("损失函数实验", "掌握不同任务的损失选择"),
            ("梯度下降调优", "理解优化算法原理")
        ],
        
        "中级阶段": [
            ("卷积神经网络", "实现基本的图像分类"),
            ("循环神经网络", "处理序列数据"),
            ("LSTM/GRU", "解决长序列问题"),
            ("批归一化", "加速训练和提高稳定性"),
            ("Dropout正则化", "防止过拟合")
        ],
        
        "高级阶段": [
            ("注意力机制", "理解注意力的计算过程"),
            ("Transformer", "掌握现代NLP架构"),
            ("生成对抗网络", "学习生成模型"),
            ("变分自编码器", "理解潜在空间建模"),
            ("强化学习基础", "探索智能体学习")
        ],
        
        "项目实战": [
            ("图像分类项目", "完整的CNN项目流程"),
            ("文本分类项目", "NLP实践应用"),
            ("时间序列预测", "RNN在实际数据上的应用"),
            ("推荐系统", "深度学习在推荐中的应用"),
            ("生成模型项目", "创建艺术作品或文本")
        ]
    }
    
    for stage, exercises in roadmap.items():
        print(f"\n【{stage}】")
        for i, (name, desc) in enumerate(exercises, 1):
            print(f"{i}. {name}")
            print(f"   目标: {desc}")
    
    print(f"\n学习建议:")
    print("• 循序渐进，不要跳跃式学习")
    print("• 每个练习都要动手实现")
    print("• 理解原理比记住代码更重要")
    print("• 多做实验，观察参数变化的影响")
    print("• 结合理论学习和实践练习")

def main():
    """主函数"""
    print("🎯 深度学习实践练习")
    print("=" * 60)
    
    exercises_introduction()
    exercise_1_perceptron()
    exercise_2_mlp_xor() 
    exercise_3_mini_cnn()
    exercise_4_simple_rnn()
    exercise_5_gradient_descent()
    exercise_6_backpropagation()
    create_practice_roadmap()
    
    print("\n" + "=" * 60)
    print("🎓 练习总结")
    print()
    print("通过这些练习你学到了:")
    print("• 感知机的线性分类能力和局限性")
    print("• 多层网络解决非线性问题的原理")
    print("• 卷积操作提取图像特征的过程")
    print("• RNN处理序列数据的记忆机制")
    print("• 梯度下降优化算法的工作原理")
    print("• 反向传播计算梯度的详细步骤")
    print()
    print("继续学习建议:")
    print("• 实现更复杂的网络架构")
    print("• 在真实数据集上验证算法")
    print("• 学习使用深度学习框架")
    print("• 关注最新的研究进展")
    print("• 参与开源项目贡献代码")
    print()
    print("记住：理解原理比记住代码更重要！")

if __name__ == "__main__":
    main()