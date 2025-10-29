"""
真实项目案例：房价预测系统
使用线性回归预测房屋价格

项目背景：
房地产中介需要一个快速估价系统，根据房屋的基本特征（面积、房龄、楼层等）
来预测房价，帮助客户快速了解房屋的市场价值。

数据说明：
- 面积（平方米）：50-200
- 房龄（年）：0-30
- 楼层：1-30
- 距离地铁（公里）：0.1-5
- 价格（万元）：目标变量

学习目标：
1. 理解多元线性回归的实际应用
2. 学会特征工程和数据预处理
3. 掌握模型评估和误差分析
4. 了解如何进行预测和解释结果
"""

import random
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("提示：建议安装 numpy 以获得更好性能")

class HousePricePredictor:
    """
    多元线性回归房价预测器
    使用多个特征预测房价
    """

    def __init__(self, learning_rate=0.0001, max_iterations=1000):
        """
        初始化预测器
        learning_rate: 学习率（多元回归需要较小的学习率）
        max_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None  # 权重向量
        self.bias = 0  # 偏置
        self.cost_history = []  # 损失历史
        self.feature_names = []  # 特征名称
        self.feature_means = []  # 特征均值（用于标准化）
        self.feature_stds = []  # 特征标准差
        self.target_mean = 0  # 目标均值
        self.target_std = 1  # 目标标准差

    def standardize_features(self, X, fit=True):
        """
        特征标准化：(x - mean) / std
        这对于梯度下降非常重要！
        """
        if HAS_NUMPY:
            X = np.array(X)
            if fit:
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.std(X, axis=0) + 1e-8  # 避免除零
            X_normalized = (X - self.feature_means) / self.feature_stds
            return X_normalized
        else:
            # 纯Python实现
            if fit:
                n_features = len(X[0])
                self.feature_means = [sum(row[j] for row in X) / len(X) for j in range(n_features)]
                self.feature_stds = []
                for j in range(n_features):
                    variance = sum((row[j] - self.feature_means[j]) ** 2 for row in X) / len(X)
                    self.feature_stds.append(math.sqrt(variance) + 1e-8)

            X_normalized = []
            for row in X:
                normalized_row = [(val - mean) / std for val, mean, std in
                                 zip(row, self.feature_means, self.feature_stds)]
                X_normalized.append(normalized_row)
            return X_normalized

    def standardize_target(self, y, fit=True):
        """目标变量标准化"""
        if HAS_NUMPY:
            y = np.array(y)
            if fit:
                self.target_mean = np.mean(y)
                self.target_std = np.std(y) + 1e-8
            return (y - self.target_mean) / self.target_std
        else:
            if fit:
                self.target_mean = sum(y) / len(y)
                variance = sum((val - self.target_mean) ** 2 for val in y) / len(y)
                self.target_std = math.sqrt(variance) + 1e-8
            return [(val - self.target_mean) / self.target_std for val in y]

    def fit(self, X, y, feature_names=None):
        """
        训练模型
        X: 特征矩阵 [[面积, 房龄, 楼层, 距离地铁], ...]
        y: 目标价格列表
        feature_names: 特征名称列表
        """
        if feature_names:
            self.feature_names = feature_names

        # 标准化特征和目标
        X_normalized = self.standardize_features(X, fit=True)
        y_normalized = self.standardize_target(y, fit=True)

        if HAS_NUMPY:
            X_normalized = np.array(X_normalized)
            y_normalized = np.array(y_normalized)
            n_samples, n_features = X_normalized.shape

            # 初始化权重
            self.weights = np.zeros(n_features)

            print(f"\n开始训练房价预测模型...")
            print(f"样本数量: {n_samples}, 特征数量: {n_features}")
            print(f"学习率: {self.learning_rate}, 最大迭代: {self.max_iterations}")
            print("-" * 60)

            # 梯度下降
            for iteration in range(self.max_iterations):
                # 预测
                y_pred = np.dot(X_normalized, self.weights) + self.bias

                # 计算损失
                cost = np.mean((y_pred - y_normalized) ** 2)
                self.cost_history.append(cost)

                # 计算梯度
                dw = (2 / n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
                db = (2 / n_samples) * np.sum(y_pred - y_normalized)

                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # 打印进度
                if iteration % 100 == 0:
                    # 反标准化以显示真实误差
                    real_error = cost * (self.target_std ** 2)
                    print(f"迭代 {iteration:4d}: 损失={cost:.6f}, 真实RMSE={math.sqrt(real_error):.2f}万元")

            print("-" * 60)
            print("训练完成！")
            self._print_model_summary()

        else:
            # 纯Python实现（简化版）
            n_samples = len(X_normalized)
            n_features = len(X_normalized[0])
            self.weights = [0.0] * n_features

            print(f"\n开始训练（纯Python模式）...")

            for iteration in range(self.max_iterations):
                # 预测
                y_pred = [sum(x[j] * self.weights[j] for j in range(n_features)) + self.bias
                         for x in X_normalized]

                # 计算损失
                cost = sum((pred - actual) ** 2 for pred, actual in zip(y_pred, y_normalized)) / n_samples
                self.cost_history.append(cost)

                # 计算梯度并更新
                for j in range(n_features):
                    dw = (2 / n_samples) * sum((pred - actual) * x[j]
                                              for pred, actual, x in zip(y_pred, y_normalized, X_normalized))
                    self.weights[j] -= self.learning_rate * dw

                db = (2 / n_samples) * sum(pred - actual for pred, actual in zip(y_pred, y_normalized))
                self.bias -= self.learning_rate * db

                if iteration % 100 == 0:
                    real_error = cost * (self.target_std ** 2)
                    print(f"迭代 {iteration:4d}: RMSE={math.sqrt(real_error):.2f}万元")

    def predict(self, X):
        """
        预测房价
        X: 特征矩阵或单个样本
        返回: 预测价格（万元）
        """
        # 判断是单个样本还是多个样本
        is_single = isinstance(X[0], (int, float))
        if is_single:
            X = [X]

        # 标准化特征
        X_normalized = self.standardize_features(X, fit=False)

        if HAS_NUMPY:
            X_normalized = np.array(X_normalized)
            y_pred_normalized = np.dot(X_normalized, self.weights) + self.bias
        else:
            n_features = len(X_normalized[0])
            y_pred_normalized = [sum(x[j] * self.weights[j] for j in range(n_features)) + self.bias
                                for x in X_normalized]

        # 反标准化得到真实价格
        if HAS_NUMPY:
            y_pred = y_pred_normalized * self.target_std + self.target_mean
            return float(y_pred[0]) if is_single else y_pred.tolist()
        else:
            y_pred = [val * self.target_std + self.target_mean for val in y_pred_normalized]
            return y_pred[0] if is_single else y_pred

    def _print_model_summary(self):
        """打印模型摘要信息"""
        print("\n模型参数摘要：")
        print("=" * 60)
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if HAS_NUMPY:
                    # 计算真实系数（考虑标准化）
                    real_coef = self.weights[i] * self.target_std / self.feature_stds[i]
                    print(f"{name:12s}: 系数={real_coef:8.2f} (每单位变化对价格的影响)")
                else:
                    print(f"{name:12s}: 权重={self.weights[i]:.4f}")
        print(f"{'截距':12s}: {self.bias:.4f}")
        print("=" * 60)

    def evaluate(self, X, y_true):
        """
        评估模型性能
        返回: R²分数, RMSE, MAE
        """
        y_pred = self.predict(X)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]

        # 计算指标
        n = len(y_true)

        # RMSE
        rmse = math.sqrt(sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true)) / n)

        # MAE
        mae = sum(abs(pred - true) for pred, true in zip(y_pred, y_true)) / n

        # R²
        y_mean = sum(y_true) / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_true)
        ss_res = sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return r2, rmse, mae


def generate_realistic_house_data(n_samples=200):
    """
    生成模拟的真实房价数据

    定价规则（基于实际市场逻辑）：
    - 基础价格：50万
    - 面积：每平方米 +0.6万
    - 房龄：每年 -0.8万
    - 楼层：6-15层加成 +5万，1-3层或高层 -3万
    - 地铁距离：每公里 -4万
    - 随机波动：±10万
    """
    print("生成模拟房价数据...")
    print("=" * 60)

    data = []
    for _ in range(n_samples):
        # 特征
        area = random.uniform(50, 200)  # 面积：50-200平米
        age = random.uniform(0, 30)  # 房龄：0-30年
        floor = random.randint(1, 30)  # 楼层：1-30层
        metro_distance = random.uniform(0.1, 5.0)  # 地铁距离：0.1-5公里

        # 价格计算（真实市场逻辑）
        base_price = 50
        price = base_price
        price += area * 0.6  # 面积贡献
        price -= age * 0.8  # 房龄折旧

        # 楼层加成
        if 6 <= floor <= 15:
            price += 5  # 黄金楼层
        elif floor <= 3 or floor >= 25:
            price -= 3  # 低层或高层

        price -= metro_distance * 4  # 地铁距离影响

        # 添加随机波动（模拟市场不确定性）
        price += random.gauss(0, 10)

        # 确保价格为正
        price = max(price, 30)

        data.append([area, age, floor, metro_distance, price])

    return data


def demo_house_price_prediction():
    """演示完整的房价预测项目流程"""
    print("\n" + "=" * 60)
    print("项目演示：智能房价预测系统")
    print("=" * 60)

    # 1. 生成数据
    data = generate_realistic_house_data(n_samples=200)

    # 分离特征和标签
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    # 划分训练集和测试集（80/20）
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\n数据集划分：")
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")

    # 2. 训练模型
    model = HousePricePredictor(learning_rate=0.01, max_iterations=1000)
    feature_names = ['面积(㎡)', '房龄(年)', '楼层', '地铁距离(km)']
    model.fit(X_train, y_train, feature_names=feature_names)

    # 3. 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    r2_train, rmse_train, mae_train = model.evaluate(X_train, y_train)
    print(f"\n训练集性能：")
    print(f"  R² 分数:  {r2_train:.4f} (越接近1越好)")
    print(f"  RMSE:     {rmse_train:.2f} 万元")
    print(f"  MAE:      {mae_train:.2f} 万元")

    r2_test, rmse_test, mae_test = model.evaluate(X_test, y_test)
    print(f"\n测试集性能：")
    print(f"  R² 分数:  {r2_test:.4f}")
    print(f"  RMSE:     {rmse_test:.2f} 万元")
    print(f"  MAE:      {mae_test:.2f} 万元")

    # 4. 实际预测示例
    print("\n" + "=" * 60)
    print("实际预测案例")
    print("=" * 60)

    test_cases = [
        {
            "描述": "高档小区新房",
            "特征": [120, 0, 10, 0.5],  # 120㎡, 新房, 10层, 地铁500米
        },
        {
            "描述": "老旧小区",
            "特征": [80, 25, 3, 2.5],  # 80㎡, 25年, 3层, 地铁2.5公里
        },
        {
            "描述": "中等户型房",
            "特征": [95, 10, 15, 1.0],  # 95㎡, 10年, 15层, 地铁1公里
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}: {case['描述']}")
        features = case['特征']
        print(f"  面积: {features[0]:.0f}㎡")
        print(f"  房龄: {features[1]:.0f}年")
        print(f"  楼层: {features[2]}层")
        print(f"  地铁距离: {features[3]:.1f}公里")

        predicted_price = model.predict(features)
        print(f"  预测价格: {predicted_price:.2f} 万元")

    # 5. 误差分析
    print("\n" + "=" * 60)
    print("误差分析（前10个测试样本）")
    print("=" * 60)
    print(f"{'实际价格':<12} {'预测价格':<12} {'误差':<12} {'误差率'}")
    print("-" * 60)

    predictions = model.predict(X_test[:10])
    if not isinstance(predictions, list):
        predictions = [predictions]

    for true_val, pred_val in zip(y_test[:10], predictions):
        error = pred_val - true_val
        error_rate = abs(error / true_val) * 100
        print(f"{true_val:<12.2f} {pred_val:<12.2f} {error:+12.2f} {error_rate:6.2f}%")

    # 6. 使用建议
    print("\n" + "=" * 60)
    print("模型使用建议")
    print("=" * 60)
    print("""
1. 模型适用范围：
   - 面积：50-200平方米
   - 房龄：0-30年
   - 楼层：1-30层
   - 地铁距离：0.1-5公里

2. 预测精度：
   - 平均误差约 ±{:.2f} 万元
   - 在合理范围内可作为参考

3. 局限性：
   - 未考虑装修、朝向、学区等因素
   - 仅使用线性模型，无法捕捉复杂非线性关系
   - 需要更多真实数据来提升准确性

4. 改进方向：
   - 添加更多特征（装修、学区、配套设施等）
   - 使用多项式特征捕捉非线性关系
   - 尝试更复杂的模型（决策树、神经网络等）
    """.format(rmse_test))


def interactive_prediction():
    """交互式预测功能"""
    print("\n" + "=" * 60)
    print("交互式房价预测")
    print("=" * 60)

    # 先训练一个模型
    data = generate_realistic_house_data(n_samples=200)
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    model = HousePricePredictor(learning_rate=0.01, max_iterations=500)
    feature_names = ['面积(㎡)', '房龄(年)', '楼层', '地铁距离(km)']
    model.fit(X, y, feature_names=feature_names)

    print("\n模型训练完成！请输入房屋信息进行预测：")
    print("（输入 'q' 退出）\n")

    while True:
        try:
            area_input = input("请输入面积（平方米，50-200）: ")
            if area_input.lower() == 'q':
                break
            area = float(area_input)

            age = float(input("请输入房龄（年，0-30）: "))
            floor = int(input("请输入楼层（1-30）: "))
            metro = float(input("请输入距离地铁距离（公里，0.1-5）: "))

            # 预测
            features = [area, age, floor, metro]
            price = model.predict(features)

            print(f"\n{'='*40}")
            print(f"预测结果: {price:.2f} 万元")
            print(f"{'='*40}\n")

        except ValueError:
            print("输入错误，请输入数字！\n")
        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        机器学习实战项目：房价预测系统                  ║
║        基于多元线性回归的智能估价工具                  ║
╚══════════════════════════════════════════════════════════╝
    """)

    print("请选择运行模式：")
    print("1. 完整演示模式（推荐）")
    print("2. 交互式预测模式")
    print("3. 两者都运行")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == '1':
        demo_house_price_prediction()
    elif choice == '2':
        interactive_prediction()
    elif choice == '3':
        demo_house_price_prediction()
        interactive_prediction()
    else:
        print("选择无效，运行默认演示...")
        demo_house_price_prediction()

    print("\n" + "=" * 60)
    print("感谢使用房价预测系统！")
    print("=" * 60)


if __name__ == "__main__":
    main()
