# 机器学习算法教程 - 线性回归
# Linear Regression: 代码实现 + 原理解释

import random
import math

# 尝试导入numpy，如果没有就使用纯Python实现
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("警告：numpy未安装，使用纯Python实现（性能可能较慢）")
    print("建议运行：pip install numpy matplotlib 获得更好体验")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

class LinearRegression:
    """
    线性回归算法实现
    使用梯度下降法优化参数
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        初始化参数
        learning_rate: 学习率，控制每次参数更新的步长
        max_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weight = 0  # 权重w
        self.bias = 0    # 偏置b
        self.cost_history = []  # 记录损失函数值
        
    def fit(self, X, y):
        """
        训练模型
        X: 输入特征 (n_samples,)
        y: 目标值 (n_samples,)
        """
        n_samples = len(X)
        
        print(f"开始训练：{n_samples}个样本，学习率={self.learning_rate}")
        
        # 梯度下降迭代
        for i in range(self.max_iterations):
            # 前向传播：计算预测值
            y_predicted = self.predict(X)
            
            # 计算损失函数(均方误差)
            cost = np.mean((y_predicted - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (2/n_samples) * np.sum((y_predicted - y) * X)  # 对w的偏导
            db = (2/n_samples) * np.sum(y_predicted - y)        # 对b的偏导
            
            # 更新参数
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代打印一次进度
            if i % 100 == 0:
                print(f"迭代 {i}: 损失={cost:.4f}, w={self.weight:.4f}, b={self.bias:.4f}")
        
        print(f"训练完成！最终参数: w={self.weight:.4f}, b={self.bias:.4f}")
        
    def predict(self, X):
        """
        预测函数
        返回预测值 y = w*x + b
        """
        return self.weight * X + self.bias
    
    def plot_results(self, X, y):
        """
        可视化结果（如果matplotlib可用）
        """
        try:
            plt.figure(figsize=(12, 4))
            
            # 子图1：数据点和拟合直线
            plt.subplot(1, 2, 1)
            plt.scatter(X, y, color='blue', alpha=0.6, label='实际数据')
            
            # 绘制拟合直线
            x_line = np.linspace(min(X), max(X), 100)
            y_line = self.predict(x_line)
            plt.plot(x_line, y_line, color='red', linewidth=2, label=f'拟合直线: y={self.weight:.2f}x+{self.bias:.2f}')
            
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title('线性回归拟合结果')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2：损失函数下降曲线
            plt.subplot(1, 2, 2)
            plt.plot(self.cost_history, color='green', linewidth=2)
            plt.xlabel('迭代次数')
            plt.ylabel('损失函数值')
            plt.title('训练过程中的损失变化')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法显示图表")
            print("可以运行: pip install matplotlib 来安装")

def generate_sample_data():
    """
    生成示例数据：房屋面积 vs 价格
    """
    print("生成示例数据：房屋面积(平方米) -> 价格(万元)")
    
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    # 生成100个房屋面积数据点 (50-200平方米)
    areas = np.random.uniform(50, 200, 100)
    
    # 真实关系：价格 = 0.5 * 面积 + 噪声
    true_slope = 0.5
    true_intercept = 10
    noise = np.random.normal(0, 5, 100)  # 添加噪声
    
    prices = true_slope * areas + true_intercept + noise
    
    print(f"真实关系：价格 = {true_slope} * 面积 + {true_intercept} + 噪声")
    print(f"数据范围：面积 {areas.min():.0f}-{areas.max():.0f}平方米，价格 {prices.min():.1f}-{prices.max():.1f}万元")
    
    return areas, prices

def demonstrate_overfitting_underfitting():
    """
    演示过拟合和欠拟合概念
    """
    print("\n=== 过拟合与欠拟合演示 ===")
    
    # 生成训练数据
    X_train, y_train = generate_sample_data()
    
    # 三种不同学习率的模型
    models = {
        "学习率过高(可能不收敛)": LinearRegression(learning_rate=0.1, max_iterations=100),
        "学习率适中": LinearRegression(learning_rate=0.01, max_iterations=1000),
        "学习率过低(收敛缓慢)": LinearRegression(learning_rate=0.001, max_iterations=1000)
    }
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        
        # 计算最终损失
        final_cost = model.cost_history[-1] if model.cost_history else float('inf')
        print(f"最终损失: {final_cost:.4f}")

def practical_example():
    """
    实际应用示例：房价预测
    """
    print("\n=== 实际应用：房价预测系统 ===")
    
    # 生成训练数据
    X_train, y_train = generate_sample_data()
    
    # 训练模型
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
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
    mse = np.mean((train_predictions - y_train) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\n模型性能评估：")
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"均方根误差(RMSE): {rmse:.2f} 万元")
    
    # 可视化结果
    model.plot_results(X_train, y_train)

def mathematical_insights():
    """
    数学原理深入解释
    """
    print("\n=== 数学原理深入 ===")
    
    print("1. 梯度下降算法原理：")
    print("   - 目标：最小化损失函数 J(w,b) = (1/2m)Σ(hw,b(x^i) - y^i)²")
    print("   - 方法：沿着梯度的反方向更新参数")
    print("   - 更新公式：")
    print("     w := w - α * ∂J/∂w")
    print("     b := b - α * ∂J/∂b")
    print("   - α是学习率，控制每次更新的步长")
    
    print("\n2. 为什么使用均方误差？")
    print("   - 可微分：便于计算梯度")
    print("   - 凸函数：保证全局最优解")
    print("   - 对大误差惩罚更重")
    
    print("\n3. 学习率的选择：")
    print("   - 过大：可能跳过最优点，不收敛")
    print("   - 过小：收敛缓慢，需要更多迭代")
    print("   - 经验值：通常在0.001-0.1之间")

if __name__ == "__main__":
    # 运行完整的线性回归教程
    linear_regression_theory()
    practical_example()
    demonstrate_overfitting_underfitting()
    mathematical_insights()
    
    print("\n=== 总结 ===")
    print("线性回归是机器学习的基础，掌握了它你就理解了：")
    print("• 监督学习的基本流程")
    print("• 损失函数的概念")
    print("• 梯度下降优化算法")
    print("• 过拟合和欠拟合问题")
    print("• 模型评估方法")
    print("\n下一步：学习逻辑回归，了解分类问题的解决方法！")