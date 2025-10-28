# 机器学习算法教程 - 逻辑回归
# Logistic Regression: 代码实现 + 原理解释

import math
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_numpy(data: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name}必须是一维序列，但得到的是形状 {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name}不能为空")
    return array


def _validate_binary_labels(y: np.ndarray) -> None:
    unique = np.unique(y)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError("标签必须只包含0或1")


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError("真实标签和预测标签的形状必须一致")

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return accuracy, precision, recall

def logistic_regression_theory():
    """
    逻辑回归原理解释
    
    逻辑回归是用于分类问题的线性模型。
    
    核心思想：
    - 使用Sigmoid函数将线性回归的输出映射到(0,1)区间
    - 输出可以解释为属于正类的概率
    - 通过设定阈值(通常是0.5)来进行分类决策
    
    数学公式：
    1. 线性部分：z = w*x + b
    2. Sigmoid函数：σ(z) = 1 / (1 + e^(-z))
    3. 预测概率：p(y=1|x) = σ(w*x + b)
    
    损失函数(对数似然损失)：
    Loss = -[y*log(p) + (1-y)*log(1-p)]
    
    为什么不用均方误差？
    - Sigmoid函数是非线性的，均方误差会产生非凸函数
    - 对数似然损失是凸函数，保证全局最优解
    """
    print("=== 逻辑回归算法原理 ===")
    print("目标：预测样本属于某个类别的概率")
    print("方法：Sigmoid函数 + 最大似然估计")
    print("应用：邮件分类、医疗诊断、营销预测等")
    print()

class LogisticRegression:
    """逻辑回归算法实现，使用梯度下降优化参数。"""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        if learning_rate <= 0:
            raise ValueError("学习率必须为正数")
        if max_iterations <= 0:
            raise ValueError("最大迭代次数必须为正整数")
        if tolerance < 0:
            raise ValueError("容差不能为负数")

        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

        self.weight: float = 0.0
        self.bias: float = 0.0
        self.cost_history: list[float] = []

    @staticmethod
    def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Sigmoid 激活函数，带有数值稳定处理。"""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        X_arr = _to_numpy(X, name="特征")
        y_arr = _to_numpy(y, name="标签")
        _validate_binary_labels(y_arr)

        if X_arr.shape != y_arr.shape:
            raise ValueError("特征和标签必须长度一致")

        n_samples = X_arr.size
        print(f"开始训练逻辑回归：{n_samples}个样本")
        print(f"正样本数：{int(np.sum(y_arr))}, 负样本数：{n_samples - int(np.sum(y_arr))}")

        self.weight = 0.0
        self.bias = 0.0
        self.cost_history.clear()

        previous_cost = math.inf

        for iteration in range(self.max_iterations):
            z = self.weight * X_arr + self.bias
            y_prob = self.sigmoid(z)

            cost = log_loss(y_arr, y_prob)
            self.cost_history.append(cost)

            dw = float(np.mean((y_prob - y_arr) * X_arr))
            db = float(np.mean(y_prob - y_arr))

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if iteration % 100 == 0:
                print(f"迭代 {iteration}: 损失={cost:.4f}, w={self.weight:.4f}, b={self.bias:.4f}")

            if self.tolerance > 0 and abs(previous_cost - cost) < self.tolerance:
                print(f"在第 {iteration} 次迭代时收敛，损失变化 {abs(previous_cost - cost):.6f}")
                break
            previous_cost = cost

        print(f"训练完成！最终参数: w={self.weight:.4f}, b={self.bias:.4f}")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X, name="特征")
        return self.sigmoid(self.weight * X_arr + self.bias)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("阈值必须在(0, 1)之间")

        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def decision_boundary_value(self) -> Union[float, None]:
        return -self.bias / self.weight if self.weight != 0 else None

def sigmoid_function_demo():
    """
    演示Sigmoid函数的特性
    """
    print("\n=== Sigmoid函数特性演示 ===")
    
    # 生成测试数据
    z_values = np.linspace(-10, 10, 100)
    sigmoid_values = 1 / (1 + np.exp(-z_values))
    
    print("Sigmoid函数 σ(z) = 1 / (1 + e^(-z)) 的特点：")
    print("• 输出范围：(0, 1)")
    print("• 在z=0时，σ(0) = 0.5")
    print("• 单调递增")
    print("• S型曲线")
    
    # 关键点值
    key_points = [-2, -1, 0, 1, 2]
    print(f"\n关键点的Sigmoid值：")
    for z in key_points:
        sigmoid_val = 1 / (1 + math.exp(-z))
        print(f"σ({z:2d}) = {sigmoid_val:.4f}")
    
    plt.figure(figsize=(10, 4))

    # 子图1: Sigmoid函数
    plt.subplot(1, 2, 1)
    plt.plot(z_values, sigmoid_values, 'b-', linewidth=2, label='Sigmoid函数')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='决策阈值=0.5')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('z = wx + b')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid激活函数')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2: 导数
    sigmoid_derivative = sigmoid_values * (1 - sigmoid_values)
    plt.subplot(1, 2, 2)
    plt.plot(z_values, sigmoid_derivative, 'g-', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
    plt.xlabel('z')
    plt.ylabel("σ'(z)")
    plt.title('Sigmoid函数的导数')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def generate_classification_data():
    """
    生成二分类示例数据：学生成绩 vs 是否通过考试
    """
    print("\n生成示例数据：学生学习时间(小时) -> 是否通过考试(1通过/0不通过)")
    
    np.random.seed(42)
    
    # 学习时间 (0-20小时)
    study_hours = np.random.uniform(0, 20, 200)
    
    # 通过概率随学习时间增加
    # 使用真实的逻辑回归关系
    true_weight = 0.3
    true_bias = -3
    
    z = true_weight * study_hours + true_bias
    true_probabilities = 1 / (1 + np.exp(-z))
    
    # 根据概率生成标签
    passed = np.random.binomial(1, true_probabilities)
    
    print(f"真实关系参数: w={true_weight}, b={true_bias}")
    print(f"数据分布: 通过{np.sum(passed)}人, 不通过{len(passed)-np.sum(passed)}人")
    
    return study_hours, passed

def practical_example():
    """
    实际应用示例：学生考试通过率预测
    """
    print("\n=== 实际应用：学生考试通过率预测 ===")
    
    # 生成训练数据
    X_train, y_train = generate_classification_data()
    
    # 训练模型
    model = LogisticRegression(learning_rate=0.1, max_iterations=2000, tolerance=1e-6)
    model.fit(X_train, y_train)
    
    # 预测新学生
    test_hours = [2, 5, 8, 12, 15, 18]
    print(f"\n预测结果：")
    print(f"决策边界：学习时间 = {model.decision_boundary_value():.1f}小时")
    print("学习时间    通过概率    预测结果")
    print("-" * 35)
    
    for hours in test_hours:
        prob = model.predict_proba(hours)
        prediction = "通过" if prob >= 0.5 else "不通过"
        print(f"{hours:6.0f}小时    {prob:8.3f}    {prediction}")
    
    # 模型评估
    train_predictions = model.predict(X_train)
    accuracy, precision, recall = classification_metrics(y_train, train_predictions)
    
    print(f"\n模型性能：")
    print(f"训练集准确率: {accuracy:.3f}")
    print(f"精确率(Precision): {precision:.3f}")
    print(f"召回率(Recall): {recall:.3f}")
    
    # 可视化
    visualize_logistic_regression(X_train, y_train, model)

def visualize_logistic_regression(X, y, model):
    """
    可视化逻辑回归结果
    """
    plt.figure(figsize=(15, 5))

    # 子图1: 数据点和决策边界
    plt.subplot(1, 3, 1)

    # 绘制数据点
    passed_idx = y == 1
    failed_idx = y == 0

    plt.scatter(X[passed_idx], y[passed_idx], color='green', alpha=0.6, label='通过考试', s=30)
    plt.scatter(X[failed_idx], y[failed_idx], color='red', alpha=0.6, label='未通过考试', s=30)

    # 决策边界
    boundary = model.decision_boundary_value()
    if boundary is not None:
        plt.axvline(x=boundary, color='blue', linestyle='--', linewidth=2, label=f'决策边界: {boundary:.1f}小时')

    plt.xlabel('学习时间(小时)')
    plt.ylabel('考试结果')
    plt.title('逻辑回归分类结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 概率曲线
    plt.subplot(1, 3, 2)
    x_line = np.linspace(0, 20, 200)
    prob_line = model.predict_proba(x_line)

    plt.plot(x_line, prob_line, 'b-', linewidth=2, label='通过概率')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='决策阈值=0.5')
    if boundary is not None:
        plt.axvline(x=boundary, color='b', linestyle='--', alpha=0.7)

    plt.scatter(X[passed_idx], np.ones(np.sum(passed_idx)), color='green', alpha=0.3, s=20)
    plt.scatter(X[failed_idx], np.zeros(np.sum(failed_idx)), color='red', alpha=0.3, s=20)

    plt.xlabel('学习时间(小时)')
    plt.ylabel('通过概率')
    plt.title('Sigmoid概率函数')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 损失函数收敛
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history, 'g-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('对数似然损失')
    plt.title('训练过程中的损失下降')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_with_linear_regression():
    """
    对比线性回归和逻辑回归的区别
    """
    print("\n=== 线性回归 vs 逻辑回归对比 ===")
    
    comparison = {
        "问题类型": {"线性回归": "回归问题(预测连续值)", "逻辑回归": "分类问题(预测类别)"},
        "输出范围": {"线性回归": "(-∞, +∞)", "逻辑回归": "(0, 1)"},
        "激活函数": {"线性回归": "无(线性)", "逻辑回归": "Sigmoid函数"},
        "损失函数": {"线性回归": "均方误差(MSE)", "逻辑回归": "对数似然损失"},
        "决策方式": {"线性回归": "直接输出预测值", "逻辑回归": "概率>阈值则分类为正类"},
        "应用场景": {"线性回归": "房价预测、销量预测", "逻辑回归": "邮件分类、疾病诊断"}
    }
    
    print(f"{'特征':<10} | {'线性回归':<20} | {'逻辑回归':<20}")
    print("-" * 55)
    for feature, values in comparison.items():
        print(f"{feature:<10} | {values['线性回归']:<20} | {values['逻辑回归']:<20}")

def mathematical_insights():
    """
    数学原理深入解释
    """
    print("\n=== 数学原理深入 ===")
    
    
    print("1. 为什么使用Sigmoid函数？")
    print("   • 将任意实数映射到(0,1)区间，可解释为概率")
    print("   • 可微分，便于梯度计算")
    print("   • 在两端饱和，中间敏感的特性符合分类需求")
    
    print("\n2. 对数似然损失函数推导：")
    print("   • 假设：P(y=1|x) = σ(wx+b), P(y=0|x) = 1-σ(wx+b)")
    print("   • 似然函数：L = ∏P(yi|xi)")
    print("   • 对数似然：log L = Σ[yi*log(pi) + (1-yi)*log(1-pi)]")
    print("   • 损失函数：J = -log L (最小化负对数似然)")
    
    print("\n3. 梯度计算：")
    print("   • ∂J/∂w = (1/m)Σ(σ(wx+b) - y)x")
    print("   • ∂J/∂b = (1/m)Σ(σ(wx+b) - y)")
    print("   • 形式与线性回归类似，但σ(wx+b)替代了wx+b")

if __name__ == "__main__":
    # 运行完整的逻辑回归教程
    logistic_regression_theory()
    sigmoid_function_demo()
    practical_example()
    compare_with_linear_regression()
    mathematical_insights()
    
    print("\n=== 总结 ===")
    print("逻辑回归扩展了线性模型到分类问题，核心概念包括：")
    print("• Sigmoid函数：将线性输出转换为概率")
    print("• 对数似然损失：适合概率预测的损失函数")
    print("• 决策边界：分离不同类别的界限")
    print("• 概率解释：输出可以理解为置信度")
    print("\n下一步：学习决策树，了解非线性分类方法！")