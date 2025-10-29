"""
真实项目案例：手写数字识别系统
使用神经网络实现OCR（光学字符识别）

项目背景：
自动识别手写数字在银行支票处理、邮政编码识别、
表单自动化等场景中广泛应用。

数据说明：
- 输入：8x8像素的灰度图像（简化版MNIST）
- 每个像素值：0-16的灰度值
- 输出：数字0-9的分类

学习目标：
1. 理解神经网络的工作原理
2. 学习反向传播算法
3. 掌握多分类问题处理
4. 了解激活函数的作用
"""

import random
import math

def sigmoid(x):
    """Sigmoid激活函数"""
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU激活函数"""
    return max(0, x)

def relu_derivative(x):
    """ReLU导数"""
    return 1 if x > 0 else 0

def softmax(x_list):
    """Softmax函数（用于多分类输出层）"""
    max_val = max(x_list)
    exp_x = [math.exp(x - max_val) for x in x_list]  # 数值稳定性
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


class DigitRecognitionNN:
    """
    手写数字识别神经网络
    三层结构：输入层 -> 隐藏层 -> 输出层
    """

    def __init__(self, input_size=64, hidden_size=32, output_size=10, learning_rate=0.01):
        """
        初始化神经网络
        input_size: 输入层大小（8x8=64个像素）
        hidden_size: 隐藏层大小
        output_size: 输出层大小（10个数字）
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重（Xavier初始化）
        self.weights_ih = [[random.gauss(0, math.sqrt(2.0 / input_size))
                           for _ in range(hidden_size)]
                          for _ in range(input_size)]
        self.bias_h = [0.0] * hidden_size

        self.weights_ho = [[random.gauss(0, math.sqrt(2.0 / hidden_size))
                           for _ in range(output_size)]
                          for _ in range(hidden_size)]
        self.bias_o = [0.0] * output_size

        self.loss_history = []

    def forward(self, inputs):
        """
        前向传播
        返回：隐藏层输出、最终输出
        """
        # 隐藏层
        hidden = []
        for j in range(self.hidden_size):
            z = sum(inputs[i] * self.weights_ih[i][j] for i in range(self.input_size))
            z += self.bias_h[j]
            hidden.append(relu(z))  # 使用ReLU激活

        # 输出层
        output = []
        for k in range(self.output_size):
            z = sum(hidden[j] * self.weights_ho[j][k] for j in range(self.hidden_size))
            z += self.bias_o[k]
            output.append(z)

        output = softmax(output)  # Softmax得到概率分布
        return hidden, output

    def backward(self, inputs, hidden, output, target):
        """
        反向传播
        target: one-hot编码的目标
        """
        # 输出层误差（交叉熵损失的梯度）
        output_errors = [output[k] - target[k] for k in range(self.output_size)]

        # 隐藏层误差
        hidden_errors = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            error = sum(output_errors[k] * self.weights_ho[j][k]
                       for k in range(self.output_size))
            hidden_errors[j] = error * (1 if hidden[j] > 0 else 0)  # ReLU导数

        # 更新输出层权重
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_ho[j][k] -= self.learning_rate * output_errors[k] * hidden[j]

        for k in range(self.output_size):
            self.bias_o[k] -= self.learning_rate * output_errors[k]

        # 更新隐藏层权重
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_ih[i][j] -= self.learning_rate * hidden_errors[j] * inputs[i]

        for j in range(self.hidden_size):
            self.bias_h[j] -= self.learning_rate * hidden_errors[j]

    def fit(self, X, y, epochs=50, batch_size=32, verbose=True):
        """
        训练网络
        X: 输入数据列表
        y: 标签列表（0-9的数字）
        epochs: 训练轮数
        batch_size: 批次大小
        """
        n_samples = len(X)

        if verbose:
            print(f"\n开始训练手写数字识别模型...")
            print(f"样本数量: {n_samples}")
            print(f"网络结构: {self.input_size}-{self.hidden_size}-{self.output_size}")
            print(f"学习率: {self.learning_rate}, 训练轮数: {epochs}")
            print("-" * 60)

        for epoch in range(epochs):
            # 打乱数据
            indices = list(range(n_samples))
            random.shuffle(indices)

            epoch_loss = 0
            correct = 0

            # 小批量训练
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                for idx in batch_indices:
                    # One-hot编码
                    target = [0.0] * self.output_size
                    target[y[idx]] = 1.0

                    # 前向传播
                    hidden, output = self.forward(X[idx])

                    # 计算损失（交叉熵）
                    loss = -sum(target[k] * math.log(output[k] + 1e-10)
                               for k in range(self.output_size))
                    epoch_loss += loss

                    # 反向传播
                    self.backward(X[idx], hidden, output, target)

                    # 统计准确率
                    pred = output.index(max(output))
                    if pred == y[idx]:
                        correct += 1

            avg_loss = epoch_loss / n_samples
            accuracy = correct / n_samples
            self.loss_history.append(avg_loss)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}: 损失={avg_loss:.4f}, 准确率={accuracy:.4f}")

        if verbose:
            print("-" * 60)
            print("训练完成！")

    def predict(self, X):
        """
        预测
        返回：预测的数字标签
        """
        if isinstance(X[0], (int, float)):
            X = [X]

        predictions = []
        for x in X:
            _, output = self.forward(x)
            predictions.append(output.index(max(output)))

        return predictions[0] if len(predictions) == 1 else predictions

    def predict_proba(self, X):
        """
        预测概率分布
        """
        if isinstance(X[0], (int, float)):
            X = [X]

        probabilities = []
        for x in X:
            _, output = self.forward(x)
            probabilities.append(output)

        return probabilities[0] if len(probabilities) == 1 else probabilities


def generate_digit_pattern(digit):
    """
    生成8x8的数字图案
    使用预定义的模板加上噪声
    """
    # 简化的数字模板（8x8）
    templates = {
        0: [
            [0,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ],
        1: [
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,1,0,0,0],
            [0,1,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,1,1,1,1,1,1,0],
        ],
        2: [
            [0,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [1,1,1,1,1,1,1,0],
        ],
        3: [
            [1,1,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [1,1,1,1,1,1,0,0],
        ],
        4: [
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,1,0,0],
            [0,0,1,0,1,1,0,0],
            [0,1,0,0,1,1,0,0],
            [1,0,0,0,1,1,0,0],
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0],
        ],
        5: [
            [1,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ],
        6: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ],
        7: [
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,0,0,0,0],
            [0,0,1,1,0,0,0,0],
        ],
        8: [
            [0,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ],
        9: [
            [0,1,1,1,1,1,0,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,1,1,1,1,0,0,0],
        ],
    }

    template = templates.get(digit, templates[0])

    # 添加噪声和变化
    pattern = []
    for row in template:
        for val in row:
            # 基础值 + 随机噪声
            noisy_val = val * random.uniform(8, 16) if val else random.uniform(0, 2)
            pattern.append(noisy_val)

    return pattern


def print_digit(pattern):
    """
    打印8x8数字图像（ASCII艺术）
    """
    chars = ' .:-=+*#%@'
    for i in range(8):
        row = pattern[i*8:(i+1)*8]
        # 归一化到0-9范围
        max_val = max(row) if max(row) > 0 else 1
        normalized = [int(val / max_val * 9) for val in row]
        print(''.join(chars[val] for val in normalized))


def generate_dataset(n_samples_per_digit=100):
    """生成数字数据集"""
    print(f"生成数字数据集（每个数字{n_samples_per_digit}个样本）...")

    X = []
    y = []

    for digit in range(10):
        for _ in range(n_samples_per_digit):
            pattern = generate_digit_pattern(digit)
            # 归一化到0-1
            max_val = max(pattern) if max(pattern) > 0 else 1
            pattern = [p / max_val for p in pattern]
            X.append(pattern)
            y.append(digit)

    # 打乱数据
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return list(X), list(y)


def demo_digit_recognition():
    """完整的数字识别演示"""
    print("\n" + "=" * 60)
    print("项目演示：手写数字识别系统")
    print("=" * 60)

    # 1. 生成数据
    X, y = generate_dataset(n_samples_per_digit=100)

    # 显示示例
    print("\n数字样本示例：")
    print("=" * 60)
    for digit in range(10):
        idx = y.index(digit)
        print(f"\n数字 {digit}:")
        print_digit(X[idx])

    # 2. 划分数据集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n\n数据集划分：")
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")

    # 3. 训练模型
    model = DigitRecognitionNN(input_size=64, hidden_size=32,
                               output_size=10, learning_rate=0.1)
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # 4. 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = sum(1 for true, pred in zip(y_train, y_pred_train) if true == pred) / len(y_train)
    test_acc = sum(1 for true, pred in zip(y_test, y_pred_test) if true == pred) / len(y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 混淆矩阵（简化版）
    print(f"\n每个数字的识别准确率：")
    for digit in range(10):
        digit_indices = [i for i, label in enumerate(y_test) if label == digit]
        if digit_indices:
            digit_correct = sum(1 for i in digit_indices if y_pred_test[i] == digit)
            digit_acc = digit_correct / len(digit_indices)
            bar = '█' * int(digit_acc * 30)
            print(f"  数字 {digit}: {digit_acc:.2%} {bar}")

    # 5. 预测示例
    print("\n" + "=" * 60)
    print("预测示例")
    print("=" * 60)

    # 随机选择几个测试样本
    test_samples = random.sample(range(len(X_test)), min(5, len(X_test)))

    for idx in test_samples:
        print(f"\n样本 {idx+1}:")
        print("图像:")
        print_digit(X_test[idx])

        proba = model.predict_proba(X_test[idx])
        pred = proba.index(max(proba))

        print(f"\n真实数字: {y_test[idx]}")
        print(f"预测数字: {pred} {'✓' if pred == y_test[idx] else '✗'}")
        print(f"置信度: {max(proba):.2%}")

        # 显示前3个最可能的数字
        top3 = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:3]
        print("\n预测概率分布（Top 3）:")
        for digit, prob in top3:
            bar = '█' * int(prob * 30)
            print(f"  数字 {digit}: {prob:.2%} {bar}")

    # 6. 应用场景
    print("\n" + "=" * 60)
    print("应用场景和扩展建议")
    print("=" * 60)
    print("""
当前应用场景：
1. 银行支票自动处理
2. 邮政编码自动识别
3. 表单数字自动录入
4. 验证码识别

模型特点：
✓ 三层神经网络结构
✓ ReLU激活函数（隐藏层）
✓ Softmax输出（多分类）
✓ 交叉熵损失函数
✓ 小批量梯度下降

改进方向：
1. 增加隐藏层数量和神经元（深度网络）
2. 使用卷积神经网络（CNN）提取特征
3. 数据增强（旋转、缩放、平移）
4. Dropout防止过拟合
5. 使用真实MNIST数据集（28x28像素）

性能对比：
- 当前模型: ~{:.1%} 准确率
- CNN模型: ~99% 准确率
- 人类: ~98% 准确率
    """.format(test_acc))


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        机器学习实战项目：手写数字识别系统              ║
║        基于神经网络的OCR技术                           ║
╚══════════════════════════════════════════════════════════╝

项目亮点：
✓ 完整实现前向传播和反向传播
✓ 多分类问题（10个数字）
✓ Softmax + 交叉熵
✓ ASCII艺术可视化
    """)

    demo_digit_recognition()

    print("\n" + "=" * 60)
    print("感谢使用数字识别系统！")
    print("=" * 60)


if __name__ == "__main__":
    main()
