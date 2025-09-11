# 机器学习算法教程 - 神经网络
# Neural Network: 从零实现多层感知机

import random
import math

def neural_network_theory():
    """
    神经网络原理解释
    
    神经网络是模拟人脑神经元工作方式的机器学习算法。
    
    核心概念：
    - 神经元(节点)：接收输入，计算加权和，通过激活函数产生输出
    - 层(Layer)：输入层、隐藏层、输出层
    - 权重(Weight)：连接强度，决定信号传递的重要性
    - 偏置(Bias)：调节神经元激活的阈值
    - 激活函数：引入非线性，常用Sigmoid、ReLU等
    
    前向传播：
    1. 输入数据从输入层流向输出层
    2. 每层计算：输出 = 激活函数(权重*输入 + 偏置)
    3. 最终得到网络预测结果
    
    反向传播：
    1. 计算预测误差
    2. 误差从输出层反向传播到输入层
    3. 使用链式法则计算每个参数的梯度
    4. 更新权重和偏置以减小误差
    
    优点：
    - 能够学习复杂的非线性关系
    - 通用逼近器，理论上可以逼近任何函数
    - 适应性强，可处理多种类型的问题
    
    缺点：
    - 需要大量数据
    - 训练时间长
    - 容易过拟合
    - 黑盒模型，解释性差
    """
    print("=== 神经网络算法原理 ===")
    print("目标：通过多层非线性变换学习复杂映射关系")
    print("方法：前向传播 + 反向传播 + 梯度下降")
    print("应用：图像识别、自然语言处理、语音识别等")
    print()

def sigmoid(x):
    """Sigmoid激活函数"""
    # 防止数值溢出
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU激活函数"""
    return max(0, x)

def relu_derivative(x):
    """ReLU函数的导数"""
    return 1 if x > 0 else 0

def tanh(x):
    """Tanh激活函数"""
    if x > 500:
        return 1.0
    elif x < -500:
        return -1.0
    return math.tanh(x)

def tanh_derivative(x):
    """Tanh函数的导数"""
    t = tanh(x)
    return 1 - t * t

class SimpleNeuralNetwork:
    """
    简单的多层感知机实现
    支持任意层数和神经元数量
    """
    
    def __init__(self, layers, learning_rate=0.1, activation='sigmoid'):
        """
        初始化神经网络
        layers: 每层神经元数量列表，如[2, 4, 1]表示输入层2个、隐藏层4个、输出层1个
        learning_rate: 学习率
        activation: 激活函数类型
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # 为每相邻两层之间初始化权重矩阵
        for i in range(len(layers) - 1):
            # Xavier初始化：权重在[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]范围内
            limit = math.sqrt(6.0 / (layers[i] + layers[i + 1]))
            weight_matrix = []
            for j in range(layers[i + 1]):  # 下一层神经元数
                neuron_weights = []
                for k in range(layers[i]):  # 当前层神经元数
                    weight = random.uniform(-limit, limit)
                    neuron_weights.append(weight)
                weight_matrix.append(neuron_weights)
            self.weights.append(weight_matrix)
            
            # 初始化偏置为小随机值
            bias_vector = [random.uniform(-0.1, 0.1) for _ in range(layers[i + 1])]
            self.biases.append(bias_vector)
        
        # 训练历史
        self.loss_history = []
        
        print(f"神经网络结构: {' -> '.join(map(str, layers))}")
        print(f"激活函数: {activation}")
        print(f"总参数数量: {self.count_parameters()}")
    
    def count_parameters(self):
        """计算总参数数量"""
        total = 0
        for i in range(len(self.weights)):
            total += len(self.weights[i]) * len(self.weights[i][0])  # 权重数量
            total += len(self.biases[i])  # 偏置数量
        return total
    
    def activate(self, x):
        """激活函数"""
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        elif self.activation == 'tanh':
            return tanh(x)
        else:
            return x  # 线性激活
    
    def activate_derivative(self, x):
        """激活函数的导数"""
        if self.activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation == 'relu':
            return relu_derivative(x)
        elif self.activation == 'tanh':
            return tanh_derivative(x)
        else:
            return 1  # 线性激活的导数
    
    def forward_propagation(self, inputs):
        """
        前向传播
        返回每层的输出和加权和（用于反向传播）
        """
        activations = [inputs]  # 存储每层的激活值
        z_values = []  # 存储每层的加权和
        
        current_input = inputs
        
        for i in range(len(self.weights)):
            # 计算加权和 z = W*x + b
            z = []
            for j in range(len(self.weights[i])):
                weighted_sum = sum(w * inp for w, inp in zip(self.weights[i][j], current_input))
                weighted_sum += self.biases[i][j]
                z.append(weighted_sum)
            
            z_values.append(z)
            
            # 计算激活值
            activation = [self.activate(zi) for zi in z]
            activations.append(activation)
            current_input = activation
        
        return activations, z_values
    
    def backward_propagation(self, inputs, targets, activations, z_values):
        """反向传播计算梯度"""
        # 存储权重和偏置的梯度
        weight_gradients = []
        bias_gradients = []
        
        # 初始化梯度为0
        for i in range(len(self.weights)):
            weight_grad = [[0 for _ in range(len(self.weights[i][0]))] 
                          for _ in range(len(self.weights[i]))]
            bias_grad = [0 for _ in range(len(self.biases[i]))]
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)
        
        # 计算输出层误差
        output_errors = []
        for i in range(len(targets)):
            error = (activations[-1][i] - targets[i]) * self.activate_derivative(z_values[-1][i])
            output_errors.append(error)
        
        # 从输出层反向计算梯度
        errors = [output_errors]
        
        # 反向传播误差
        for i in range(len(self.weights) - 1, 0, -1):
            prev_errors = []
            for j in range(len(activations[i])):
                error = 0
                for k in range(len(errors[0])):
                    error += self.weights[i][k][j] * errors[0][k]
                error *= self.activate_derivative(z_values[i-1][j])
                prev_errors.append(error)
            errors.insert(0, prev_errors)
        
        # 计算权重和偏置梯度
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weight_gradients[i][j][k] = errors[i][j] * activations[i][k]
                bias_gradients[i][j] = errors[i][j]
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """更新权重和偏置"""
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.learning_rate * weight_gradients[i][j][k]
                self.biases[i][j] -= self.learning_rate * bias_gradients[i][j]
    
    def calculate_loss(self, predictions, targets):
        """计算均方误差损失"""
        return sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(targets)
    
    def train(self, X_train, y_train, epochs=1000, verbose=True):
        """
        训练神经网络
        X_train: 输入数据列表
        y_train: 目标输出列表
        epochs: 训练轮数
        """
        print(f"\n开始训练神经网络...")
        print(f"训练样本: {len(X_train)}个")
        print(f"训练轮数: {epochs}")
        
        for epoch in range(epochs):
            total_loss = 0
            
            # 对每个样本进行训练
            for i in range(len(X_train)):
                inputs = X_train[i]
                targets = y_train[i] if isinstance(y_train[i], list) else [y_train[i]]
                
                # 前向传播
                activations, z_values = self.forward_propagation(inputs)
                predictions = activations[-1]
                
                # 计算损失
                loss = self.calculate_loss(predictions, targets)
                total_loss += loss
                
                # 反向传播
                weight_gradients, bias_gradients = self.backward_propagation(
                    inputs, targets, activations, z_values)
                
                # 更新参数
                self.update_parameters(weight_gradients, bias_gradients)
            
            # 记录平均损失
            avg_loss = total_loss / len(X_train)
            self.loss_history.append(avg_loss)
            
            # 打印训练进度
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"轮次 {epoch + 1:4d}/{epochs}: 损失 = {avg_loss:.6f}")
        
        print("训练完成！")
    
    def predict(self, inputs):
        """预测单个样本"""
        activations, _ = self.forward_propagation(inputs)
        return activations[-1]
    
    def predict_batch(self, X):
        """批量预测"""
        predictions = []
        for inputs in X:
            pred = self.predict(inputs)
            predictions.append(pred[0] if len(pred) == 1 else pred)
        return predictions

def activation_functions_demo():
    """激活函数演示"""
    print("\n=== 激活函数演示 ===")
    
    test_values = [-3, -2, -1, 0, 1, 2, 3]
    
    print("输入值  | Sigmoid | ReLU   | Tanh   ")
    print("-" * 35)
    
    for x in test_values:
        sig = sigmoid(x)
        r = relu(x)
        t = tanh(x)
        print(f"{x:6d}  | {sig:7.3f} | {r:6.3f} | {t:6.3f}")
    
    print("\n激活函数特点：")
    print("- Sigmoid: 输出范围(0,1)，存在梯度消失问题")
    print("- ReLU: 计算简单，缓解梯度消失，但存在死神经元")
    print("- Tanh: 输出范围(-1,1)，0中心化，但仍有梯度消失")

def xor_problem_example():
    """XOR问题：经典的非线性分类问题"""
    print("\n=== XOR问题演示 ===")
    print("XOR是经典的非线性问题，单层感知机无法解决")
    print("需要隐藏层来学习非线性映射关系")
    
    # XOR训练数据
    X_train = [
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ]
    
    y_train = [0, 1, 1, 0]  # XOR真值表
    
    print("\nXOR真值表:")
    print("输入1 | 输入2 | 输出")
    print("-" * 18)
    for i in range(len(X_train)):
        print(f"  {X_train[i][0]}   |   {X_train[i][1]}   |  {y_train[i]}")
    
    # 创建神经网络：2个输入，4个隐藏神经元，1个输出
    nn = SimpleNeuralNetwork([2, 4, 1], learning_rate=0.5, activation='sigmoid')
    
    # 训练网络
    nn.train(X_train, y_train, epochs=1000, verbose=False)
    
    # 测试结果
    print(f"\n训练后的预测结果:")
    print("输入1 | 输入2 | 期望输出 | 预测输出 | 错误")
    print("-" * 45)
    
    total_error = 0
    for i in range(len(X_train)):
        prediction = nn.predict(X_train[i])[0]
        error = abs(prediction - y_train[i])
        total_error += error
        
        print(f"  {X_train[i][0]}   |   {X_train[i][1]}   |    {y_train[i]}     | {prediction:8.4f} | {error:.4f}")
    
    print(f"\n平均绝对误差: {total_error / len(X_train):.4f}")
    
    # 显示学习曲线
    print(f"\n学习过程（每100轮显示）:")
    for i in range(0, len(nn.loss_history), 100):
        print(f"轮次 {i+1:4d}: 损失 = {nn.loss_history[i]:.6f}")

def regression_example():
    """神经网络回归示例：拟合非线性函数"""
    print("\n=== 神经网络回归演示 ===")
    print("目标：用神经网络拟合函数 y = x^2 - 2x + 1")
    
    # 生成训练数据
    X_train = []
    y_train = []
    
    for i in range(50):
        x = random.uniform(-2, 3)  # x在[-2, 3]范围内
        y = x * x - 2 * x + 1 + random.gauss(0, 0.1)  # 添加小量噪声
        X_train.append([x])
        y_train.append([y])
    
    print(f"生成了{len(X_train)}个训练样本")
    print("前10个样本:")
    print("   x   |   y  ")
    print("-" * 15)
    for i in range(10):
        print(f"{X_train[i][0]:6.2f} | {y_train[i][0]:5.2f}")
    
    # 创建神经网络：1个输入，8个隐藏神经元，1个输出
    nn = SimpleNeuralNetwork([1, 8, 1], learning_rate=0.01, activation='tanh')
    
    # 训练网络
    nn.train(X_train, y_train, epochs=2000, verbose=False)
    
    # 测试结果
    test_points = [-2, -1, 0, 1, 2, 3]
    print(f"\n测试结果:")
    print("  x   | 真实y | 预测y | 误差")
    print("-" * 30)
    
    total_error = 0
    for x in test_points:
        true_y = x * x - 2 * x + 1
        pred_y = nn.predict([x])[0]
        error = abs(true_y - pred_y)
        total_error += error
        
        print(f"{x:5.1f} | {true_y:5.2f} | {pred_y:5.2f} | {error:.3f}")
    
    print(f"\n平均绝对误差: {total_error / len(test_points):.3f}")

def network_architecture_comparison():
    """不同网络结构对比"""
    print("\n=== 网络结构对比 ===")
    
    # 准备相同的数据
    X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [0, 1, 1, 0]  # XOR问题
    
    architectures = [
        ([2, 2, 1], "浅层小网络"),
        ([2, 4, 1], "浅层中网络"), 
        ([2, 8, 1], "浅层大网络"),
        ([2, 4, 4, 1], "深层网络")
    ]
    
    print("测试不同网络结构解决XOR问题的效果:")
    print("网络结构      | 描述       | 最终损失")
    print("-" * 40)
    
    for arch, desc in architectures:
        nn = SimpleNeuralNetwork(arch, learning_rate=0.5, activation='sigmoid')
        nn.train(X_data, y_data, epochs=1000, verbose=False)
        final_loss = nn.loss_history[-1] if nn.loss_history else float('inf')
        
        arch_str = ' -> '.join(map(str, arch))
        print(f"{arch_str:<12} | {desc:<10} | {final_loss:.6f}")

def neural_network_tips():
    """神经网络使用技巧"""
    print("\n=== 神经网络使用技巧 ===")
    
    print("1. 网络结构设计:")
    print("   - 隐藏层数：一般1-2层足够，复杂问题可增加")
    print("   - 神经元数：通常在输入和输出维度之间")
    print("   - 过宽或过深的网络容易过拟合")
    
    print("\n2. 激活函数选择:")
    print("   - 隐藏层推荐ReLU或Tanh")
    print("   - 输出层根据问题选择（回归用线性，分类用Sigmoid/Softmax）")
    print("   - 避免在深层网络中使用Sigmoid（梯度消失）")
    
    print("\n3. 学习率调整:")
    print("   - 开始可以用较大值（0.1-0.01）")
    print("   - 训练过程中可以逐渐减小")
    print("   - 观察损失曲线调整合适的值")
    
    print("\n4. 权重初始化:")
    print("   - Xavier初始化适用于Sigmoid/Tanh")
    print("   - He初始化适用于ReLU")
    print("   - 避免全零初始化")
    
    print("\n5. 防止过拟合:")
    print("   - 使用验证集监控训练过程")
    print("   - 增加训练数据")
    print("   - 简化网络结构")
    print("   - 应用正则化技术")

if __name__ == "__main__":
    neural_network_theory()
    activation_functions_demo()
    xor_problem_example()
    regression_example()
    network_architecture_comparison()
    neural_network_tips()
    
    print("\n=== 总结 ===")
    print("神经网络是强大的机器学习工具：")
    print("• 能够学习复杂的非线性映射关系")
    print("• 通过前向传播和反向传播进行训练")
    print("• 激活函数引入非线性变换")
    print("• 网络结构设计对性能影响很大")
    print("• 是深度学习的基础")
    
    print("\n实践建议：")
    print("• 从简单问题开始理解原理")
    print("• 尝试不同的网络结构和参数")
    print("• 学习使用现代深度学习框架")
    print("• 关注数据质量和预处理")
    
    print("\n下一步：学习CNN、RNN等专用网络结构！")