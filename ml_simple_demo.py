# 机器学习算法入门演示 - 简化版
# 专注于理解原理，避免数值计算问题

import random
import math

def linear_regression_theory():
    """线性回归原理解释"""
    print("=== 线性回归算法原理 ===")
    print("目标：找到最佳直线 y = w*x + b 来拟合数据")
    print("方法：最小化预测值与真实值之间的均方误差")
    print("应用：房价预测、股价预测、销售预测等")
    print()

class SimpleLinearRegression:
    """线性回归算法实现（稳定版）"""
    
    def __init__(self, learning_rate=0.001, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weight = 0.0
        self.bias = 0.0
        self.cost_history = []
        
    def normalize_data(self, X, y):
        """数据标准化，避免数值溢出"""
        X_mean = sum(X) / len(X)
        X_std = math.sqrt(sum((x - X_mean) ** 2 for x in X) / len(X))
        y_mean = sum(y) / len(y)
        y_std = math.sqrt(sum((yi - y_mean) ** 2 for yi in y) / len(y))
        
        X_norm = [(x - X_mean) / X_std for x in X]
        y_norm = [(yi - y_mean) / y_std for yi in y]
        
        return X_norm, y_norm, X_mean, X_std, y_mean, y_std
        
    def fit(self, X, y):
        """训练模型"""
        n_samples = len(X)
        
        # 数据标准化
        X_norm, y_norm, self.X_mean, self.X_std, self.y_mean, self.y_std = self.normalize_data(X, y)
        
        print(f"开始训练：{n_samples}个样本")
        print("数据已标准化，避免数值溢出")
        
        for i in range(self.max_iterations):
            # 前向传播
            predictions = [self.weight * x + self.bias for x in X_norm]
            
            # 计算损失
            cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y_norm)) / n_samples
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = sum((pred - actual) * x for pred, actual, x in zip(predictions, y_norm, X_norm)) * (2 / n_samples)
            db = sum(pred - actual for pred, actual in zip(predictions, y_norm)) * (2 / n_samples)
            
            # 更新参数
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每20次迭代打印一次
            if i % 20 == 0:
                print(f"迭代 {i:2d}: 损失={cost:.4f}, w={self.weight:.4f}, b={self.bias:.4f}")
        
        print("训练完成！")
        
    def predict(self, X):
        """预测函数（反标准化）"""
        if isinstance(X, (int, float)):
            X = [X]
            single_value = True
        else:
            single_value = False
            
        # 标准化输入
        X_norm = [(x - self.X_mean) / self.X_std for x in X]
        
        # 预测标准化的输出
        predictions_norm = [self.weight * x + self.bias for x in X_norm]
        
        # 反标准化
        predictions = [pred * self.y_std + self.y_mean for pred in predictions_norm]
        
        return predictions[0] if single_value else predictions

def simple_example():
    """简单示例：温度预测"""
    print("\n=== 简单示例：根据海拔预测温度 ===")
    
    # 简单数据：海拔高度(米) -> 温度(°C)
    # 规律：海拔每升高100米，温度下降0.6°C
    altitudes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    temperatures = [25, 24.4, 23.8, 23.2, 22.6, 22.0, 21.4, 20.8, 20.2, 19.6, 19.0]
    
    print("训练数据：")
    print("海拔(米)  温度(°C)")
    for alt, temp in zip(altitudes, temperatures):
        print(f"{alt:6d}    {temp:5.1f}")
    
    # 训练模型
    model = SimpleLinearRegression(learning_rate=0.01, max_iterations=100)
    model.fit(altitudes, temperatures)
    
    # 预测新数据
    test_altitudes = [150, 350, 750]
    print(f"\n预测结果：")
    for alt in test_altitudes:
        pred_temp = model.predict(alt)
        print(f"海拔 {alt}米 -> 预测温度: {pred_temp:.1f}°C")

def gradient_descent_visualization():
    """梯度下降可视化"""
    print("\n=== 梯度下降算法演示 ===")
    print("问题：最小化函数 f(x) = (x-5)^2")
    print("理论最优解：x = 5")
    
    x = 0  # 起始点
    learning_rate = 0.2
    
    print(f"\n起始点: x = {x}")
    print(f"学习率: {learning_rate}")
    print("\n迭代过程:")
    print("步骤  |   x值   | f(x)值 | 梯度  | 更新量")
    print("-" * 40)
    
    for i in range(8):
        fx = (x - 5) ** 2
        gradient = 2 * (x - 5)  # f'(x) = 2(x-5)
        update = learning_rate * gradient
        
        print(f"{i:4d}  | {x:6.2f}  | {fx:6.2f} | {gradient:5.1f} | {update:6.2f}")
        
        x = x - update  # 参数更新
    
    print(f"\n最终结果: x = {x:.3f}")
    print("可以看到x逐渐接近理论最优解5.000")

def loss_function_demo():
    """损失函数演示"""
    print("\n=== 损失函数概念演示 ===")
    
    # 真实数据点 (x, y)
    true_data = [(1, 3), (2, 5), (3, 7), (4, 9)]  # y = 2x + 1
    
    print("真实数据点: (x, y)")
    for x, y in true_data:
        print(f"({x}, {y})")
    
    # 测试不同的直线参数
    test_params = [
        (1.5, 2),    # 斜率太小
        (2, 1),      # 正确斜率，正确截距
        (2.5, 0.5)   # 斜率太大
    ]
    
    print(f"\n测试不同的直线参数:")
    print("参数(w,b)  |  直线方程    | 均方误差")
    print("-" * 35)
    
    for w, b in test_params:
        # 计算预测值和误差
        mse = 0
        for x, y_true in true_data:
            y_pred = w * x + b
            mse += (y_pred - y_true) ** 2
        mse /= len(true_data)
        
        print(f"({w:3.1f}, {b:3.1f}) | y = {w}x + {b}   | {mse:8.2f}")
    
    print("\n可以看到参数(2, 1)对应的直线y=2x+1误差最小")

def machine_learning_workflow():
    """机器学习完整流程演示"""
    print("\n=== 机器学习完整流程 ===")
    
    print("1. 问题定义：根据房屋面积预测价格")
    print("2. 数据收集：收集面积和价格的历史数据")
    print("3. 数据预处理：检查数据质量，处理异常值")
    print("4. 模型选择：选择线性回归模型")
    print("5. 模型训练：使用梯度下降优化参数")
    print("6. 模型评估：计算预测误差")
    print("7. 模型应用：对新房屋进行价格预测")
    
    # 模拟完整流程
    print(f"\n模拟房价预测:")
    
    # 步骤2-3: 数据
    areas = [60, 80, 100, 120, 150]  # 房屋面积
    prices = [50, 65, 80, 95, 120]  # 房价（万元）
    
    print("历史数据:")
    for area, price in zip(areas, prices):
        print(f"面积: {area}平米, 价格: {price}万元")
    
    # 步骤4-5: 训练模型
    model = SimpleLinearRegression(learning_rate=0.01, max_iterations=50)
    model.fit(areas, prices)
    
    # 步骤6: 评估
    predictions = model.predict(areas)
    mse = sum((pred - actual) ** 2 for pred, actual in zip(predictions, prices)) / len(prices)
    print(f"\n模型性能: MSE = {mse:.2f}")
    
    # 步骤7: 应用
    new_area = 110
    predicted_price = model.predict(new_area)
    print(f"预测: {new_area}平米房屋价格为 {predicted_price:.1f}万元")

if __name__ == "__main__":
    linear_regression_theory()
    gradient_descent_visualization()
    loss_function_demo()
    simple_example()
    machine_learning_workflow()
    
    print("\n=== 总结 ===")
    print("通过这个教程，你学到了：")
    print("• 线性回归的基本概念和数学原理")
    print("• 梯度下降算法的工作过程")
    print("• 损失函数的作用和重要性")
    print("• 机器学习的完整工作流程")
    print("• 如何从数学公式到代码实现")
    
    print("\n下一步建议：")
    print("• 学习更复杂的算法（逻辑回归、决策树等）")
    print("• 练习处理真实数据集")
    print("• 了解数据预处理技术")
    print("• 学习模型评估和验证方法")