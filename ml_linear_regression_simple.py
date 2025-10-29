# 机器学习算法教程 - 线性回归 (纯Python版)
# Linear Regression: 代码实现 + 原理解释

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Union

Number = Union[int, float]


def _ensure_sequence(values: Iterable[Number], *, allow_empty: bool = False) -> List[float]:
    values_list = [float(v) for v in values]
    if not allow_empty and not values_list:
        raise ValueError("输入数据不能为空")
    # 检查NaN和Inf
    if not all(math.isfinite(v) for v in values_list):
        raise ValueError("输入数据包含NaN或Inf，请检查数据质量")
    return values_list


def _check_lengths(X: Sequence[Number], y: Sequence[Number]) -> None:
    if len(X) != len(y):
        raise ValueError("特征和目标数据的长度必须一致")


def mean_squared_error(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """计算均方误差（MSE）。"""
    _check_lengths(y_true, y_pred)
    errors = [(pred - actual) for pred, actual in zip(y_pred, y_true)]
    return sum(error * error for error in errors) / len(errors)


def r2_score(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """计算R²分数（决定系数）。"""
    _check_lengths(y_true, y_pred)
    y_mean = sum(y_true) / len(y_true)
    ss_total = sum((y - y_mean) ** 2 for y in y_true)
    ss_residual = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0


def fit_closed_form(X: Sequence[Number], y: Sequence[Number]) -> tuple[float, float]:
    """使用闭式解(OLS)直接计算线性回归参数: w=Cov(X,y)/Var(X), b=mean(y)-w*mean(X)"""
    X_list = _ensure_sequence(X)
    y_list = _ensure_sequence(y)
    _check_lengths(X_list, y_list)
    
    n = len(X_list)
    x_mean = sum(X_list) / n
    y_mean = sum(y_list) / n
    
    covariance = sum((x - x_mean) * (y_val - y_mean) for x, y_val in zip(X_list, y_list))
    variance = sum((x - x_mean) ** 2 for x in X_list)
    
    if variance < 1e-10:
        raise ValueError("特征方差接近0，无法拟合")
    
    weight = covariance / variance
    bias = y_mean - weight * x_mean
    return weight, bias


def linear_regression_theory():
    """
    线性回归原理解释
    
    线性回归是最基础的机器学习算法，用于预测连续值。
    
    核心思想：
    - 假设目标变量y与特征x之间存在线性关系
    - 通过一条直线来拟合数据点
    - 目标是找到最佳的直线参数（斜率和截距）
    
    数学公式：
    y = w * x + b
    其中：w是权重(斜率)，b是偏置(截距)
    
    损失函数(均方误差MSE)：
    Loss = (1/n) * Σ(y_predicted - y_actual)²
    
    优化方法：
    1. 梯度下降法 - 迭代优化
    2. 正规方程法 - 直接计算最优解
    """
    print("=== 线性回归算法原理 ===")
    print("目标：找到最佳直线 y = w*x + b 来拟合数据")
    print("方法：最小化预测值与真实值之间的均方误差")
    print("应用：房价预测、股价预测、销售预测等")
    print()

@dataclass
class SimpleLinearRegression:
    """线性回归算法实现（纯Python版）。"""

    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 0.0
    verbose: int = 0  # 添加 verbose 属性，0=静默，1=基本信息，2=详细信息
    weight: float = field(default=0.0, init=False)
    bias: float = field(default=0.0, init=False)
    cost_history: List[float] = field(default_factory=list, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("学习率必须为正数")
        if self.max_iterations <= 0:
            raise ValueError("最大迭代次数必须为正整数")
        if self.tolerance < 0:
            raise ValueError("容差不能为负数")

    def fit(self, X: Iterable[Number], y: Iterable[Number]) -> "SimpleLinearRegression":
        """训练模型，返回self支持链式调用。"""
        X_list = _ensure_sequence(X)
        y_list = _ensure_sequence(y)
        _check_lengths(X_list, y_list)

        n_samples = len(X_list)
        if self.verbose >= 1:
            print(f"开始训练：{n_samples}个样本，学习率={self.learning_rate}")

        self.cost_history.clear()
        previous_cost = math.inf

        for iteration in range(self.max_iterations):
            # 向量化计算预测值和误差
            predictions = [self.weight * x + self.bias for x in X_list]
            errors = [pred - actual for pred, actual in zip(predictions, y_list)]

            cost = sum(e * e for e in errors) / n_samples
            self.cost_history.append(cost)

            # 向量化梯度计算
            gradient_w = (2 / n_samples) * sum(error * x for error, x in zip(errors, X_list))
            gradient_b = (2 / n_samples) * sum(errors)

            self.weight -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            if self.verbose >= 2 and iteration % 100 == 0:
                print(f"迭代 {iteration}: 损失={cost:.4f}, w={self.weight:.4f}, b={self.bias:.4f}")

            if self.tolerance > 0 and abs(previous_cost - cost) < self.tolerance:
                if self.verbose >= 1:
                    print(f"满足提前停止条件（迭代 {iteration}），损失变化 {abs(previous_cost - cost):.6f}")
                break
            previous_cost = cost

        if self.verbose >= 1:
            print(f"训练完成！最终参数: w={self.weight:.4f}, b={self.bias:.4f}")
        
        self._is_fitted = True
        return self

    def predict(self, X: Union[Number, Sequence[Number]]) -> Union[float, List[float]]:
        """预测函数。"""
        if not self._is_fitted:
            raise RuntimeError("模型未训练，请先调用fit()方法")
        
        if isinstance(X, (int, float)):
            return self.weight * float(X) + self.bias

        values = _ensure_sequence(X, allow_empty=True)
        return [self.weight * x + self.bias for x in values]
    
    def score(self, X: Sequence[Number], y: Sequence[Number]) -> float:
        """计算模型的R²分数（sklearn兼容API）"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def summary(self) -> None:
        """打印模型摘要信息"""
        if not self._is_fitted:
            print("模型状态: 未训练")
            return
        
        print("=" * 50)
        print("模型摘要")
        print("=" * 50)
        print(f"模型类型: 简单线性回归")
        print(f"训练状态: 已训练")
        print(f"模型公式: y = {self.weight:.6f} * x + {self.bias:.6f}")
        print(f"参数:")
        print(f"  - 权重(w): {self.weight:.6f}")
        print(f"  - 偏置(b): {self.bias:.6f}")
        print(f"训练信息:")
        print(f"  - 学习率: {self.learning_rate}")
        print(f"  - 迭代次数: {len(self.cost_history)}")
        print(f"  - 最终损失: {self.cost_history[-1]:.6f}")
        print("=" * 50)

def generate_sample_data():
    """
    生成示例数据：房屋面积 vs 价格
    """
    print("生成示例数据：房屋面积(平方米) -> 价格(万元)")
    
    random.seed(42)

    # 生成100个房屋面积数据点 (50-200平方米)
    areas = [random.uniform(50, 200) for _ in range(100)]

    # 真实关系：价格 = 0.5 * 面积 + 10 + 噪声
    true_slope = 0.5
    true_intercept = 10

    prices = [true_slope * area + true_intercept + random.gauss(0, 5) for area in areas]

    print(f"真实关系：价格 = {true_slope} * 面积 + {true_intercept} + 噪声")
    print(f"数据范围：面积 {min(areas):.0f}-{max(areas):.0f}平方米，价格 {min(prices):.1f}-{max(prices):.1f}万元")

    return areas, prices

def practical_example():
    """
    实际应用示例：房价预测
    """
    print("\n=== 实际应用：房价预测系统 ===")
    
    # 生成训练数据
    X_train, y_train = generate_sample_data()
    
    # 训练模型 - 对于未归一化的数据，需要使用较小的学习率
    model = SimpleLinearRegression(learning_rate=0.00001, max_iterations=2000, tolerance=1e-6)
    model.fit(X_train, y_train)
    
    # 预测新房价
    test_areas = [75, 100, 150, 180]
    print(f"\n预测结果（基于训练的模型）：")
    print(f"模型公式：价格 = {model.weight:.3f} * 面积 + {model.bias:.3f}")
    
    for area in test_areas:
        predicted_price = model.predict(area)
        print(f"{area}平方米的房子 -> 预测价格: {predicted_price:.1f}万元")
    
    # 模型评估
    train_predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, train_predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_train, train_predictions)

    print(f"\n模型性能评估：")
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"均方根误差(RMSE): {rmse:.2f} 万元")
    print(f"R2分数: {r2:.4f} (越接近1越好)")
    
    # 显示损失下降过程
    print(f"\n损失函数变化（前10次迭代）：")
    for i in range(min(10, len(model.cost_history))):
        print(f"迭代 {i}: {model.cost_history[i]:.4f}")

def demonstrate_different_learning_rates():
    """
    演示不同学习率的效果
    """
    print("\n=== 不同学习率效果演示 ===")
    
    # 生成相同的训练数据
    X_train, y_train = generate_sample_data()
    
    learning_rates = [0.00005, 0.00001, 0.000001]
    
    for lr in learning_rates:
        print(f"\n--- 学习率 = {lr} ---")
        model = SimpleLinearRegression(learning_rate=lr, max_iterations=200, tolerance=1e-6, verbose=0)
        model.fit(X_train, y_train)
        
        # 显示最终损失
        final_cost = model.cost_history[-1] if model.cost_history else float('inf')
        print(f"最终损失: {final_cost:.4f}")
        
        # 显示是否收敛
        if len(model.cost_history) >= 2:
            last_change = abs(model.cost_history[-1] - model.cost_history[-2])
            if last_change < 0.001:
                print("模型已收敛")
            else:
                print("模型可能需要更多迭代")

def mathematical_insights():
    """
    数学原理深入解释
    """
    print("\n=== 数学原理深入 ===")
    
    print("1. 梯度下降算法步骤：")
    print("   a) 初始化参数 w, b")
    print("   b) 计算预测值：y_pred = w*x + b")
    print("   c) 计算损失：MSE = (1/n)Sigma(y_pred - y_true)^2")
    print("   d) 计算梯度：")
    print("      d_MSE/d_w = (2/n)Sigma(y_pred - y_true) * x")
    print("      d_MSE/d_b = (2/n)Sigma(y_pred - y_true)")
    print("   e) 更新参数：")
    print("      w = w - alpha * d_MSE/d_w")
    print("      b = b - alpha * d_MSE/d_b")
    print("   f) 重复b-e直到收敛")
    
    print("\n2. 学习率选择原则：")
    print("   - 太大：可能跳过最优点，损失震荡")
    print("   - 太小：收敛缓慢，需要更多迭代")
    print("   - 典型值：0.001 - 0.1")
    
    print("\n3. 为什么使用均方误差？")
    print("   - 可微分：便于梯度计算")
    print("   - 凸函数：保证全局最优解")
    print("   - 对大误差惩罚更重")
    print("   - 数学性质良好")

def toy_example():
    """玩具示例：5个点的线性回归，手算验证"""
    print("\n=== 玩具示例：5个点的线性回归 ===")
    X = [1, 2, 3, 4, 5]
    y = [2.1, 3.9, 6.2, 8.0, 9.8]  # y ≈ 2x + 噪声
    
    print(f"数据: X={X}")
    print(f"      y={y}")
    
    # 闭式解
    w_ols, b_ols = fit_closed_form(X, y)
    print(f"\n闭式解: w={w_ols:.4f}, b={b_ols:.4f}")
    print(f"公式: y = {w_ols:.4f}*x + {b_ols:.4f}")
    
    # 梯度下降（使用较大学习率因为数据已标准化）
    model = SimpleLinearRegression(learning_rate=0.01, max_iterations=500, tolerance=1e-6, verbose=2)
    print("\n梯度下降训练：")
    model.fit(X, y)
    
    print(f"\n参数对比: Δw={abs(model.weight - w_ols):.6f}, Δb={abs(model.bias - b_ols):.6f}")
    print(f"两种方法得到的参数几乎一致！")


def simple_gradient_descent_demo():
    """
    简单的梯度下降可视化演示
    """
    print("\n=== 梯度下降过程演示 ===")
    
    # 简单的一维优化问题：最小化 f(x) = (x-3)^2
    print("演示问题：最小化 f(x) = (x-3)^2，最优解 x = 3")
    
    x = 0  # 起始点
    learning_rate = 0.1
    iterations = 10
    
    print(f"初始位置: x = {x}")
    print(f"学习率: {learning_rate}")
    print(f"\n梯度下降过程:")
    print("迭代  |  x值  | f(x)值 | 梯度 | 更新步长")
    print("-" * 45)
    
    for i in range(iterations):
        fx = (x - 3) ** 2  # 函数值
        gradient = 2 * (x - 3)  # 梯度 f'(x) = 2(x-3)
        step = learning_rate * gradient
        
        print(f"{i:4d}  | {x:5.2f} | {fx:6.3f} | {gradient:5.2f} | {step:6.3f}")
        
        x = x - step  # 更新参数
    
    print(f"\n最终结果: x = {x:.3f}，理论最优解: x = 3.000")

if __name__ == "__main__":
    # 运行完整的线性回归教程
    linear_regression_theory()
    simple_gradient_descent_demo()
    toy_example()  # 新增：玩具示例
    practical_example()
    demonstrate_different_learning_rates()
    mathematical_insights()
    
    print("\n=== 总结 ===")
    print("线性回归是机器学习的基础，通过本教程你学会了：")
    print("- 线性回归的数学原理和几何意义")
    print("- 梯度下降算法的具体实现")
    print("- 损失函数的作用和计算方法")
    print("- 学习率对训练过程的影响")
    print("- 如何评估模型性能")
    print("\n建议安装numpy和matplotlib以获得更好的学习体验：")
    print("pip install numpy matplotlib")
    print("\n下一步：学习逻辑回归，了解分类问题的解决方法！")