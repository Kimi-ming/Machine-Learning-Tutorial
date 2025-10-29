"""
真实项目案例：信用卡欺诈检测系统
使用逻辑回归识别异常交易

项目背景：
银行每天处理数百万笔信用卡交易，其中少数可能是欺诈交易。
需要一个实时检测系统来识别可疑交易，保护客户资金安全。

数据说明：
- 交易金额（元）
- 交易时间（小时，0-23）
- 距离上次交易的时间（小时）
- 距离常用地点的距离（公里）
- 商户类型风险评分（0-1）
- 是否境外交易（0/1）
- 是否欺诈：0=正常，1=欺诈（目标变量）

学习目标：
1. 理解二分类问题的实际应用
2. 学习处理不平衡数据集
3. 掌握分类模型的评估指标
4. 了解精确率、召回率的权衡
"""

import random
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("提示：建议安装 numpy 以获得更好性能")


def sigmoid(x):
    """Sigmoid激活函数"""
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class FraudDetector:
    """
    信用卡欺诈检测器
    基于逻辑回归的二分类模型
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, threshold=0.5):
        """
        初始化欺诈检测器
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        threshold: 分类阈值（默认0.5，可调整以平衡精确率和召回率）
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = 0
        self.cost_history = []
        self.feature_names = []

        # 用于标准化
        self.feature_means = []
        self.feature_stds = []

    def standardize_features(self, X, fit=True):
        """特征标准化"""
        if HAS_NUMPY:
            X = np.array(X)
            if fit:
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.std(X, axis=0) + 1e-8
            return (X - self.feature_means) / self.feature_stds
        else:
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

    def fit(self, X, y, feature_names=None):
        """
        训练模型
        X: 特征矩阵
        y: 标签 (0=正常, 1=欺诈)
        """
        if feature_names:
            self.feature_names = feature_names

        # 标准化特征
        X_normalized = self.standardize_features(X, fit=True)

        if HAS_NUMPY:
            X_normalized = np.array(X_normalized)
            y = np.array(y)
            n_samples, n_features = X_normalized.shape
            self.weights = np.zeros(n_features)

            # 统计类别分布
            n_fraud = np.sum(y == 1)
            n_normal = np.sum(y == 0)

            print(f"\n开始训练欺诈检测模型...")
            print(f"样本数量: {n_samples} (正常: {n_normal}, 欺诈: {n_fraud})")
            print(f"欺诈率: {n_fraud/n_samples*100:.2f}%")
            print(f"学习率: {self.learning_rate}, 迭代次数: {self.max_iterations}")
            print("-" * 60)

            # 梯度下降
            for iteration in range(self.max_iterations):
                # 计算预测概率
                z = np.dot(X_normalized, self.weights) + self.bias
                y_pred_prob = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

                # 计算交叉熵损失
                epsilon = 1e-8
                cost = -np.mean(y * np.log(y_pred_prob + epsilon) +
                               (1 - y) * np.log(1 - y_pred_prob + epsilon))
                self.cost_history.append(cost)

                # 计算梯度
                dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred_prob - y))
                db = (1 / n_samples) * np.sum(y_pred_prob - y)

                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # 打印进度
                if iteration % 100 == 0:
                    y_pred = (y_pred_prob >= self.threshold).astype(int)
                    accuracy = np.mean(y_pred == y)
                    print(f"迭代 {iteration:4d}: 损失={cost:.6f}, 准确率={accuracy:.4f}")

            print("-" * 60)
            print("训练完成！")
            self._print_model_summary()

        else:
            # 纯Python实现
            n_samples = len(X_normalized)
            n_features = len(X_normalized[0])
            self.weights = [0.0] * n_features

            n_fraud = sum(y)
            print(f"\n开始训练（纯Python模式）...")
            print(f"样本数: {n_samples}, 欺诈样本: {n_fraud}")

            for iteration in range(self.max_iterations):
                # 预测
                y_pred_prob = []
                for x in X_normalized:
                    z = sum(x[j] * self.weights[j] for j in range(n_features)) + self.bias
                    y_pred_prob.append(sigmoid(z))

                # 损失
                epsilon = 1e-8
                cost = -sum(y_i * math.log(p + epsilon) + (1 - y_i) * math.log(1 - p + epsilon)
                           for y_i, p in zip(y, y_pred_prob)) / n_samples
                self.cost_history.append(cost)

                # 梯度
                for j in range(n_features):
                    dw = sum((p - y_i) * x[j] for p, y_i, x in zip(y_pred_prob, y, X_normalized)) / n_samples
                    self.weights[j] -= self.learning_rate * dw

                db = sum(p - y_i for p, y_i in zip(y_pred_prob, y)) / n_samples
                self.bias -= self.learning_rate * db

                if iteration % 100 == 0:
                    accuracy = sum(1 for p, y_i in zip(y_pred_prob, y)
                                 if (p >= self.threshold) == y_i) / n_samples
                    print(f"迭代 {iteration:4d}: 准确率={accuracy:.4f}")

    def predict_proba(self, X):
        """
        预测欺诈概率
        返回: 0-1之间的概率值
        """
        is_single = isinstance(X[0], (int, float))
        if is_single:
            X = [X]

        X_normalized = self.standardize_features(X, fit=False)

        if HAS_NUMPY:
            X_normalized = np.array(X_normalized)
            z = np.dot(X_normalized, self.weights) + self.bias
            proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return float(proba[0]) if is_single else proba.tolist()
        else:
            n_features = len(X_normalized[0])
            probas = []
            for x in X_normalized:
                z = sum(x[j] * self.weights[j] for j in range(n_features)) + self.bias
                probas.append(sigmoid(z))
            return probas[0] if is_single else probas

    def predict(self, X):
        """
        预测类别
        返回: 0(正常) 或 1(欺诈)
        """
        probas = self.predict_proba(X)
        if isinstance(probas, float):
            return 1 if probas >= self.threshold else 0
        else:
            return [1 if p >= self.threshold else 0 for p in probas]

    def _print_model_summary(self):
        """打印模型摘要"""
        print("\n模型参数摘要：")
        print("=" * 60)
        print(f"{'特征名称':<20} {'权重':>15} {'影响'}")
        print("-" * 60)

        if self.feature_names and HAS_NUMPY:
            weights_with_names = list(zip(self.feature_names, self.weights))
            weights_with_names.sort(key=lambda x: abs(x[1]), reverse=True)

            for name, weight in weights_with_names:
                influence = "增加欺诈风险" if weight > 0 else "降低欺诈风险"
                print(f"{name:<20} {weight:>15.4f}  {influence}")

        print(f"\n{'截距':<20} {self.bias:>15.4f}")
        print(f"{'分类阈值':<20} {self.threshold:>15.4f}")
        print("=" * 60)

    def evaluate(self, X, y_true):
        """
        全面评估模型性能
        返回: 准确率, 精确率, 召回率, F1分数
        """
        y_pred = self.predict(X)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]

        # 计算混淆矩阵
        tp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 1)  # 真阳性
        tn = sum(1 for pred, true in zip(y_pred, y_true) if pred == 0 and true == 0)  # 真阴性
        fp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 0)  # 假阳性
        fn = sum(1 for pred, true in zip(y_pred, y_true) if pred == 0 and true == 1)  # 假阴性

        # 计算指标
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }


def generate_fraud_data(n_samples=1000, fraud_rate=0.05):
    """
    生成模拟的信用卡交易数据

    欺诈交易特征：
    - 金额通常较大
    - 常在深夜发生
    - 距离上次交易时间短（连续多笔）
    - 距离常用地点远
    - 高风险商户
    - 境外交易概率高
    """
    print(f"生成模拟交易数据 (欺诈率: {fraud_rate*100:.1f}%)...")
    print("=" * 60)

    data = []
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # 生成正常交易
    for _ in range(n_normal):
        amount = random.uniform(10, 500)  # 正常金额：10-500元
        hour = random.choice(list(range(8, 23)))  # 白天交易
        time_since_last = random.uniform(24, 168)  # 较长时间间隔
        distance_from_home = random.uniform(0, 20)  # 常用地点附近
        merchant_risk = random.uniform(0, 0.3)  # 低风险商户
        is_foreign = 0  # 国内交易
        is_fraud = 0

        data.append([amount, hour, time_since_last, distance_from_home,
                    merchant_risk, is_foreign, is_fraud])

    # 生成欺诈交易
    for _ in range(n_fraud):
        amount = random.uniform(500, 5000)  # 异常大额
        hour = random.choice([0, 1, 2, 3, 4, 23])  # 深夜
        time_since_last = random.uniform(0.1, 5)  # 短时间内多笔
        distance_from_home = random.uniform(50, 500)  # 远离常用地点
        merchant_risk = random.uniform(0.5, 1.0)  # 高风险商户
        is_foreign = random.choice([0, 1])  # 可能境外
        is_fraud = 1

        data.append([amount, hour, time_since_last, distance_from_home,
                    merchant_risk, is_foreign, is_fraud])

    # 打乱数据
    random.shuffle(data)
    return data


def demo_fraud_detection():
    """演示完整的欺诈检测项目流程"""
    print("\n" + "=" * 60)
    print("项目演示：信用卡欺诈检测系统")
    print("=" * 60)

    # 1. 生成数据
    data = generate_fraud_data(n_samples=2000, fraud_rate=0.05)

    # 分离特征和标签
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    # 划分训练集和测试集
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\n数据集划分：")
    print(f"  训练集: {len(X_train)} 笔交易")
    print(f"  测试集: {len(X_test)} 笔交易")
    print(f"  训练集欺诈率: {sum(y_train)/len(y_train)*100:.2f}%")

    # 2. 训练模型
    model = FraudDetector(learning_rate=0.1, max_iterations=1000, threshold=0.5)
    feature_names = ['交易金额', '交易时间', '距上次交易', '距常用地点', '商户风险', '是否境外']
    model.fit(X_train, y_train, feature_names=feature_names)

    # 3. 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    metrics_train = model.evaluate(X_train, y_train)
    print(f"\n训练集性能：")
    print(f"  准确率:   {metrics_train['accuracy']:.4f}")
    print(f"  精确率:   {metrics_train['precision']:.4f} (预测为欺诈的交易中真正是欺诈的比例)")
    print(f"  召回率:   {metrics_train['recall']:.4f} (实际欺诈交易中被检测出的比例)")
    print(f"  F1分数:   {metrics_train['f1']:.4f}")

    cm = metrics_train['confusion_matrix']
    print(f"\n  混淆矩阵:")
    print(f"    真阳性(TP): {cm['TP']} - 正确识别的欺诈")
    print(f"    真阴性(TN): {cm['TN']} - 正确识别的正常")
    print(f"    假阳性(FP): {cm['FP']} - 误判为欺诈的正常交易")
    print(f"    假阴性(FN): {cm['FN']} - 漏判的欺诈交易")

    metrics_test = model.evaluate(X_test, y_test)
    print(f"\n测试集性能：")
    print(f"  准确率:   {metrics_test['accuracy']:.4f}")
    print(f"  精确率:   {metrics_test['precision']:.4f}")
    print(f"  召回率:   {metrics_test['recall']:.4f}")
    print(f"  F1分数:   {metrics_test['f1']:.4f}")

    # 4. 阈值调整分析
    print("\n" + "=" * 60)
    print("阈值调整分析")
    print("=" * 60)
    print("\n调整分类阈值可以平衡精确率和召回率：")
    print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'建议场景'}")
    print("-" * 70)

    thresholds = [0.3, 0.5, 0.7, 0.9]
    scenarios = [
        "宽松检测，减少漏报",
        "平衡性能（默认）",
        "严格检测，减少误报",
        "极严格，最小化误报"
    ]

    for threshold, scenario in zip(thresholds, scenarios):
        model.threshold = threshold
        metrics = model.evaluate(X_test, y_test)
        print(f"{threshold:<8.1f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {scenario}")

    # 恢复默认阈值
    model.threshold = 0.5

    # 5. 实际案例检测
    print("\n" + "=" * 60)
    print("实际交易检测案例")
    print("=" * 60)

    test_cases = [
        {
            "描述": "正常白天购物",
            "特征": [150, 14, 48, 5, 0.1, 0],
            "预期": "正常"
        },
        {
            "描述": "深夜大额境外交易",
            "特征": [3000, 2, 0.5, 200, 0.8, 1],
            "预期": "欺诈"
        },
        {
            "描述": "连续小额交易",
            "特征": [100, 10, 0.2, 10, 0.6, 0],
            "预期": "可疑"
        },
        {
            "描述": "周末休闲消费",
            "特征": [300, 18, 72, 15, 0.2, 0],
            "预期": "正常"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}: {case['描述']}")
        features = case['特征']
        print(f"  交易金额: {features[0]:.0f}元")
        print(f"  交易时间: {features[1]}点")
        print(f"  距上次交易: {features[2]:.1f}小时")
        print(f"  距常用地点: {features[3]:.0f}公里")
        print(f"  商户风险: {features[4]:.2f}")
        print(f"  是否境外: {'是' if features[5] else '否'}")

        prob = model.predict_proba(features)
        pred = model.predict(features)

        print(f"  欺诈概率: {prob:.4f}")
        print(f"  检测结果: {'🚨 欺诈' if pred == 1 else '✓ 正常'}")
        print(f"  预期结果: {case['预期']}")

        # 风险等级
        if prob < 0.3:
            risk_level = "低风险"
        elif prob < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
        print(f"  风险等级: {risk_level}")

    # 6. 业务建议
    print("\n" + "=" * 60)
    print("系统部署建议")
    print("=" * 60)
    print(f"""
1. 性能指标理解：
   - 当前召回率: {metrics_test['recall']:.2%} - 能检测出 {metrics_test['recall']:.2%} 的欺诈交易
   - 当前精确率: {metrics_test['precision']:.2%} - 预警交易中 {metrics_test['precision']:.2%} 确实是欺诈
   - 假阳性率: {metrics_test['confusion_matrix']['FP']/(metrics_test['confusion_matrix']['FP']+metrics_test['confusion_matrix']['TN']):.2%} - 正常交易被误判的比例

2. 部署策略：
   - 高风险交易(>0.7): 立即拦截，人工审核
   - 中风险交易(0.3-0.7): 短信验证或额外认证
   - 低风险交易(<0.3): 正常放行，异步监控

3. 持续优化：
   - 定期用新数据重新训练模型
   - 收集用户反馈改进特征工程
   - 监控模型性能变化

4. 成本收益：
   - 假阳性成本: 客户体验下降
   - 假阴性成本: 实际欺诈损失
   - 建议根据业务需求调整阈值
    """)


def interactive_detection():
    """交互式欺诈检测"""
    print("\n" + "=" * 60)
    print("交互式欺诈检测")
    print("=" * 60)

    # 训练模型
    data = generate_fraud_data(n_samples=2000, fraud_rate=0.05)
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    model = FraudDetector(learning_rate=0.1, max_iterations=500)
    feature_names = ['交易金额', '交易时间', '距上次交易', '距常用地点', '商户风险', '是否境外']
    model.fit(X, y, feature_names=feature_names)

    print("\n模型训练完成！请输入交易信息进行检测：")
    print("（输入 'q' 退出）\n")

    while True:
        try:
            amount_input = input("交易金额（元）: ")
            if amount_input.lower() == 'q':
                break
            amount = float(amount_input)

            hour = int(input("交易时间（小时，0-23）: "))
            time_since_last = float(input("距上次交易时间（小时）: "))
            distance = float(input("距常用地点距离（公里）: "))
            merchant_risk = float(input("商户风险评分（0-1）: "))
            is_foreign = int(input("是否境外交易（0=否，1=是）: "))

            features = [amount, hour, time_since_last, distance, merchant_risk, is_foreign]
            prob = model.predict_proba(features)
            pred = model.predict(features)

            print(f"\n{'='*50}")
            print(f"欺诈概率: {prob:.4f}")
            print(f"检测结果: {'🚨 欺诈警告！' if pred == 1 else '✓ 正常交易'}")
            if prob > 0.7:
                print(f"建议操作: 立即拦截，联系持卡人确认")
            elif prob > 0.3:
                print(f"建议操作: 发送验证短信")
            else:
                print(f"建议操作: 正常放行")
            print(f"{'='*50}\n")

        except ValueError:
            print("输入错误，请输入有效数字！\n")
        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        机器学习实战项目：信用卡欺诈检测系统            ║
║        基于逻辑回归的实时风险评估                      ║
╚══════════════════════════════════════════════════════════╝
    """)

    print("请选择运行模式：")
    print("1. 完整演示模式（推荐）")
    print("2. 交互式检测模式")
    print("3. 两者都运行")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == '1':
        demo_fraud_detection()
    elif choice == '2':
        interactive_detection()
    elif choice == '3':
        demo_fraud_detection()
        interactive_detection()
    else:
        print("选择无效，运行默认演示...")
        demo_fraud_detection()

    print("\n" + "=" * 60)
    print("感谢使用欺诈检测系统！")
    print("=" * 60)


if __name__ == "__main__":
    main()
