"""
真实项目案例：电信客户流失预测系统
使用决策树预测客户流失风险

项目背景：
电信运营商每年流失大量客户，造成巨大损失。
需要预测哪些客户可能流失，提前采取挽留措施。

数据说明：
- 合约时长（月）：客户使用服务的时间
- 月费（元）：每月消费金额
- 通话时长（分钟/月）：平均通话时间
- 客服投诉次数：过去6个月投诉次数
- 套餐类型：0=基础, 1=标准, 2=高级
- 是否有宽带：0=否, 1=是
- 年龄：客户年龄
- 是否流失：0=未流失, 1=流失（目标变量）

学习目标：
1. 理解决策树的决策过程
2. 学会解释模型预测结果
3. 掌握特征重要性分析
4. 了解过拟合控制方法
"""

import random
import math
from collections import Counter

class ChurnDecisionTree:
    """
    客户流失预测决策树
    使用CART算法（分类与回归树）
    """

    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        """
        初始化决策树
        max_depth: 最大深度（防止过拟合）
        min_samples_split: 节点分裂需要的最小样本数
        min_samples_leaf: 叶子节点的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = []
        self.feature_importances = {}

    def _gini_impurity(self, y):
        """
        计算基尼不纯度
        Gini = 1 - Σ(p_i²)
        p_i 是类别i的概率
        """
        if len(y) == 0:
            return 0
        counter = Counter(y)
        n = len(y)
        gini = 1.0 - sum((count / n) ** 2 for count in counter.values())
        return gini

    def _split_data(self, X, y, feature_idx, threshold):
        """
        根据特征和阈值分割数据
        """
        left_mask = [x[feature_idx] <= threshold for x in X]
        right_mask = [not m for m in left_mask]

        X_left = [X[i] for i in range(len(X)) if left_mask[i]]
        y_left = [y[i] for i in range(len(y)) if left_mask[i]]
        X_right = [X[i] for i in range(len(X)) if right_mask[i]]
        y_right = [y[i] for i in range(len(y)) if right_mask[i]]

        return X_left, y_left, X_right, y_right

    def _find_best_split(self, X, y):
        """
        寻找最佳分割点
        遍历所有特征和可能的阈值，选择基尼增益最大的分割
        """
        n_features = len(X[0])
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        current_gini = self._gini_impurity(y)

        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            values = sorted(set(x[feature_idx] for x in X))

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)

                # 检查分割是否有效
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                # 计算加权基尼不纯度
                n = len(y)
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)
                weighted_gini = (len(y_left) / n) * gini_left + (len(y_right) / n) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, current_gini - best_gini

    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        """
        n_samples = len(y)
        n_classes = len(set(y))

        # 停止条件
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # 创建叶子节点
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return {
                'is_leaf': True,
                'class': most_common,
                'probability': counter[most_common] / n_samples,
                'samples': n_samples,
                'distribution': dict(counter)
            }

        # 寻找最佳分割
        best_feature, best_threshold, gain = self._find_best_split(X, y)

        if best_feature is None:
            # 无法分割，创建叶子节点
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return {
                'is_leaf': True,
                'class': most_common,
                'probability': counter[most_common] / n_samples,
                'samples': n_samples
            }

        # 记录特征重要性
        if self.feature_names and best_feature < len(self.feature_names):
            feature_name = self.feature_names[best_feature]
            self.feature_importances[feature_name] = \
                self.feature_importances.get(feature_name, 0) + gain * n_samples

        # 分割数据
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)

        # 递归构建子树
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
            'samples': n_samples
        }

    def fit(self, X, y, feature_names=None):
        """
        训练决策树
        """
        if feature_names:
            self.feature_names = feature_names
            self.feature_importances = {name: 0 for name in feature_names}

        print(f"\n开始训练客户流失预测模型...")
        print(f"样本数量: {len(X)}")
        print(f"流失客户: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"最大深度: {self.max_depth}")
        print("-" * 60)

        self.tree = self._build_tree(X, y)

        # 归一化特征重要性
        if self.feature_importances:
            total = sum(self.feature_importances.values())
            if total > 0:
                self.feature_importances = {
                    k: v / total for k, v in self.feature_importances.items()
                }

        print("训练完成！")
        self._print_tree_info()

    def _predict_single(self, x, node):
        """
        预测单个样本
        """
        if node['is_leaf']:
            return node['class'], node['probability']

        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """
        预测类别
        """
        if isinstance(X[0], (int, float)):
            X = [X]

        predictions = []
        for x in X:
            pred_class, _ = self._predict_single(x, self.tree)
            predictions.append(pred_class)

        return predictions[0] if len(predictions) == 1 else predictions

    def predict_proba(self, X):
        """
        预测概率
        """
        if isinstance(X[0], (int, float)):
            X = [X]

        probabilities = []
        for x in X:
            _, prob = self._predict_single(x, self.tree)
            probabilities.append(prob)

        return probabilities[0] if len(probabilities) == 1 else probabilities

    def _print_tree_info(self):
        """
        打印树的基本信息
        """
        print("\n模型信息：")
        print("=" * 60)
        print(f"树的深度: {self._get_tree_depth(self.tree)}")
        print(f"叶子节点数: {self._count_leaves(self.tree)}")

        if self.feature_importances:
            print("\n特征重要性排序：")
            sorted_features = sorted(self.feature_importances.items(),
                                    key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                bar = '█' * int(importance * 50)
                print(f"  {feature:<15}: {importance:.4f} {bar}")
        print("=" * 60)

    def _get_tree_depth(self, node):
        """计算树的深度"""
        if node['is_leaf']:
            return 1
        return 1 + max(self._get_tree_depth(node['left']),
                      self._get_tree_depth(node['right']))

    def _count_leaves(self, node):
        """计算叶子节点数量"""
        if node['is_leaf']:
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])

    def print_tree(self, node=None, depth=0, prefix=""):
        """
        打印决策树结构
        """
        if node is None:
            node = self.tree

        if node['is_leaf']:
            churn_label = "流失" if node['class'] == 1 else "留存"
            print(f"{prefix}→ 预测: {churn_label} (概率: {node['probability']:.2%}, "
                  f"样本: {node['samples']})")
        else:
            feature_name = (self.feature_names[node['feature']]
                          if self.feature_names else f"特征{node['feature']}")
            print(f"{prefix}[{feature_name} <= {node['threshold']:.2f}?] "
                  f"(样本: {node['samples']})")
            print(f"{prefix}├─ 是:")
            self.print_tree(node['left'], depth + 1, prefix + "│  ")
            print(f"{prefix}└─ 否:")
            self.print_tree(node['right'], depth + 1, prefix + "   ")

    def explain_prediction(self, x):
        """
        解释单个预测的决策路径
        """
        print("\n" + "=" * 60)
        print("决策路径分析")
        print("=" * 60)

        path = []
        node = self.tree

        while not node['is_leaf']:
            feature_name = (self.feature_names[node['feature']]
                          if self.feature_names else f"特征{node['feature']}")
            value = x[node['feature']]
            threshold = node['threshold']

            if value <= threshold:
                decision = f"{feature_name}={value:.2f} <= {threshold:.2f}"
                path.append((decision, '→ 左分支'))
                node = node['left']
            else:
                decision = f"{feature_name}={value:.2f} > {threshold:.2f}"
                path.append((decision, '→ 右分支'))
                node = node['right']

        # 打印路径
        for i, (decision, branch) in enumerate(path, 1):
            print(f"步骤 {i}: {decision} {branch}")

        # 最终结果
        churn_label = "流失" if node['class'] == 1 else "留存"
        print(f"\n最终预测: {churn_label}")
        print(f"置信度: {node['probability']:.2%}")
        print(f"基于 {node['samples']} 个训练样本")
        print("=" * 60)


def generate_churn_data(n_customers=1000):
    """
    生成模拟的客户流失数据

    流失规则：
    - 合约时长短 + 月费高 → 容易流失
    - 投诉次数多 → 容易流失
    - 通话时长少 + 无宽带 → 容易流失
    - 套餐级别低 + 年龄大 → 容易流失
    """
    print(f"生成 {n_customers} 位客户的数据...")

    data = []

    for _ in range(n_customers):
        # 特征
        tenure = random.uniform(1, 72)  # 合约时长：1-72月
        monthly_fee = random.uniform(50, 300)  # 月费：50-300元
        call_minutes = random.uniform(100, 2000)  # 通话时长
        complaints = random.randint(0, 10)  # 投诉次数
        plan_type = random.randint(0, 2)  # 套餐类型
        has_internet = random.choice([0, 1])  # 是否有宽带
        age = random.randint(18, 70)  # 年龄

        # 流失概率计算（基于业务规则）
        churn_score = 0

        # 规则1: 短期高价客户容易流失
        if tenure < 12 and monthly_fee > 200:
            churn_score += 30

        # 规则2: 投诉多的客户
        churn_score += complaints * 5

        # 规则3: 使用少的客户
        if call_minutes < 500 and has_internet == 0:
            churn_score += 20

        # 规则4: 基础套餐老年客户
        if plan_type == 0 and age > 55:
            churn_score += 15

        # 规则5: 长期客户不易流失
        if tenure > 48:
            churn_score -= 25

        # 规则6: 高级套餐客户不易流失
        if plan_type == 2:
            churn_score -= 20

        # 添加随机性
        churn_score += random.uniform(-10, 10)

        # 转换为二分类
        churn = 1 if churn_score > 30 else 0

        data.append([tenure, monthly_fee, call_minutes, complaints,
                    plan_type, has_internet, age, churn])

    return data


def demo_churn_prediction():
    """完整的客户流失预测演示"""
    print("\n" + "=" * 60)
    print("项目演示：电信客户流失预测系统")
    print("=" * 60)

    # 1. 生成数据
    data = generate_churn_data(n_customers=1000)

    # 分离特征和标签
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n数据集划分：")
    print(f"  训练集: {len(X_train)} 位客户")
    print(f"  测试集: {len(X_test)} 位客户")

    # 2. 训练模型
    feature_names = ['合约时长', '月费', '通话时长', '投诉次数',
                    '套餐类型', '有宽带', '年龄']
    model = ChurnDecisionTree(max_depth=5, min_samples_split=20, min_samples_leaf=10)
    model.fit(X_train, y_train, feature_names=feature_names)

    # 3. 打印决策树
    print("\n" + "=" * 60)
    print("决策树结构（前3层）")
    print("=" * 60)
    model.print_tree()

    # 4. 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算指标
    def calculate_metrics(y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1, (tp, tn, fp, fn)

    acc_train, prec_train, rec_train, f1_train, _ = calculate_metrics(y_train, y_pred_train)
    acc_test, prec_test, rec_test, f1_test, cm = calculate_metrics(y_test, y_pred_test)

    print(f"训练集性能：")
    print(f"  准确率: {acc_train:.4f}")
    print(f"  精确率: {prec_train:.4f}")
    print(f"  召回率: {rec_train:.4f}")
    print(f"  F1分数: {f1_train:.4f}")

    print(f"\n测试集性能：")
    print(f"  准确率: {acc_test:.4f}")
    print(f"  精确率: {prec_test:.4f} (预测流失中真实流失的比例)")
    print(f"  召回率: {rec_test:.4f} (真实流失中被预测出的比例)")
    print(f"  F1分数: {f1_test:.4f}")

    print(f"\n混淆矩阵：")
    print(f"  真阳性(TP): {cm[0]} - 正确预测的流失客户")
    print(f"  真阴性(TN): {cm[1]} - 正确预测的留存客户")
    print(f"  假阳性(FP): {cm[2]} - 误判为流失的留存客户")
    print(f"  假阴性(FN): {cm[3]} - 漏判的流失客户 ⚠️")

    # 5. 预测案例
    print("\n" + "=" * 60)
    print("客户流失风险预测案例")
    print("=" * 60)

    test_cases = [
        {
            "描述": "新用户高价套餐",
            "特征": [3, 280, 1500, 0, 2, 1, 28],  # 3月, 280元, 1500分钟, 无投诉, 高级套餐, 有宽带, 28岁
        },
        {
            "描述": "老用户投诉多",
            "特征": [36, 150, 800, 8, 1, 1, 45],  # 36月, 150元, 800分钟, 8次投诉
        },
        {
            "描述": "短期基础套餐用户",
            "特征": [6, 100, 300, 2, 0, 0, 60],  # 6月, 100元, 低使用, 基础套餐, 无宽带
        },
        {
            "描述": "长期VIP客户",
            "特征": [60, 250, 1800, 0, 2, 1, 35],  # 60月, 250元, 高使用, 高级套餐
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}: {case['描述']}")
        features = case['特征']
        labels = ['合约时长', '月费', '通话时长', '投诉次数', '套餐类型', '有宽带', '年龄']

        for label, value in zip(labels, features):
            if label == '套餐类型':
                types = ['基础', '标准', '高级']
                print(f"  {label}: {types[value]}")
            elif label == '有宽带':
                print(f"  {label}: {'是' if value else '否'}")
            else:
                print(f"  {label}: {value}")

        pred = model.predict(features)
        prob = model.predict_proba(features)

        print(f"\n  流失风险: {'⚠️ 高风险' if pred == 1 else '✓ 低风险'}")
        print(f"  流失概率: {prob:.2%}")

        # 解释预测
        if i == 2:  # 详细解释一个案例
            model.explain_prediction(features)

    # 6. 挽留策略建议
    print("\n" + "=" * 60)
    print("客户挽留策略建议")
    print("=" * 60)
    print("""
基于模型分析，针对不同风险等级的客户：

【高风险客户】(流失概率 > 70%)
1. 立即指派客户经理一对一跟进
2. 提供专属优惠：免费升级套餐或费用减免
3. 赠送增值服务：流量包、会员权益等
4. 快速响应和解决投诉问题

【中风险客户】(流失概率 30%-70%)
1. 发送关怀短信/邮件
2. 推荐更适合的套餐
3. 邀请参加会员活动
4. 提供积分优惠

【低风险客户】(流失概率 < 30%)
1. 保持常规服务质量
2. 定期推送新产品信息
3. 维护良好客户关系

重点关注特征：
""")

    if model.feature_importances:
        sorted_features = sorted(model.feature_importances.items(),
                                key=lambda x: x[1], reverse=True)[:3]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"{i}. {feature} - 重要性: {importance:.2%}")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        机器学习实战项目：客户流失预测系统              ║
║        基于决策树的可解释AI                            ║
╚══════════════════════════════════════════════════════════╝

决策树优势：
✓ 高度可解释 - 清晰的决策规则
✓ 不需要特征标准化
✓ 可处理非线性关系
✓ 适合业务人员理解
    """)

    demo_churn_prediction()

    print("\n" + "=" * 60)
    print("感谢使用客户流失预测系统！")
    print("=" * 60)


if __name__ == "__main__":
    main()
