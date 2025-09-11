
# 机器学习算法教程 - 决策树
# Decision Tree: 代码实现 + 原理解释

import math
import random

def decision_tree_theory():
    """
    决策树原理解释
    
    决策树是一种基于规则的分类算法，通过一系列if-else条件来做决策。
    
    核心思想：
    - 从根节点开始，每个内部节点表示一个特征上的测试
    - 每个分支代表一个测试结果
    - 每个叶子节点代表一个类别标签
    - 通过递归分割数据来构建树
    
    关键概念：
    1. 信息增益(Information Gain)：衡量特征分割的效果
    2. 熵(Entropy)：衡量数据的混乱程度
    3. 基尼不纯度(Gini Impurity)：另一种衡量不纯度的方法
    
    优点：
    - 易于理解和解释
    - 不需要数据预处理
    - 可以处理数值型和类别型特征
    - 可以捕获非线性关系
    
    缺点：
    - 容易过拟合
    - 对数据变化敏感
    - 倾向于选择有更多水平的特征
    """
    print("=== 决策树算法原理 ===")
    print("目标：通过一系列判断条件对数据进行分类")
    print("方法：递归分割数据，最大化信息增益")
    print("应用：医疗诊断、信用评估、推荐系统等")
    print()

def calculate_entropy(labels):
    """
    计算熵值
    Entropy = -Σ(p_i * log2(p_i))
    其中 p_i 是类别 i 的概率
    """
    if not labels:
        return 0
    
    # 统计各类别的数量
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 计算熵
    entropy = 0
    total = len(labels)
    
    for count in label_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_information_gain(parent_labels, left_labels, right_labels):
    """
    计算信息增益
    Information Gain = Entropy(parent) - Weighted_Average(Entropy(children))
    """
    parent_entropy = calculate_entropy(parent_labels)
    
    total_samples = len(parent_labels)
    left_weight = len(left_labels) / total_samples
    right_weight = len(right_labels) / total_samples
    
    weighted_entropy = (left_weight * calculate_entropy(left_labels) + 
                       right_weight * calculate_entropy(right_labels))
    
    return parent_entropy - weighted_entropy

class DecisionTreeNode:
    """决策树节点"""
    def __init__(self):
        self.feature_index = None  # 分割特征的索引
        self.threshold = None      # 分割阈值
        self.left = None          # 左子树
        self.right = None         # 右子树
        self.prediction = None    # 叶子节点的预测值
        self.is_leaf = False      # 是否为叶子节点

class SimpleDecisionTree:
    """
    简单决策树实现
    只处理数值型特征的二分类问题
    """
    
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        """训练决策树"""
        self.feature_names = feature_names or [f"特征{i}" for i in range(len(X[0]))]
        print(f"开始构建决策树，最大深度={self.max_depth}")
        self.root = self._build_tree(X, y, depth=0)
        print("决策树构建完成！")
        
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        node = DecisionTreeNode()
        
        # 停止条件
        if (depth >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) < self.min_samples_split):
            
            node.is_leaf = True
            # 选择多数类作为预测
            label_counts = {}
            for label in y:
                label_counts[label] = label_counts.get(label, 0) + 1
            node.prediction = max(label_counts.keys(), key=lambda x: label_counts[x])
            return node
        
        # 寻找最佳分割
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain == 0:  # 无法进一步分割
            node.is_leaf = True
            label_counts = {}
            for label in y:
                label_counts[label] = label_counts.get(label, 0) + 1
            node.prediction = max(label_counts.keys(), key=lambda x: label_counts[x])
            return node
        
        # 记录分割信息
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        print(f"深度{depth}: 使用{self.feature_names[best_feature]} <= {best_threshold:.2f} 分割，信息增益={best_gain:.4f}")
        
        # 分割数据
        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
        
        # 递归构建子树
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        
        return node
    
    def _find_best_split(self, X, y):
        """寻找最佳分割特征和阈值"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # 遍历所有特征
        for feature_idx in range(len(X[0])):
            # 获取该特征的所有唯一值作为候选阈值
            feature_values = [sample[feature_idx] for sample in X]
            unique_values = sorted(set(feature_values))
            
            # 尝试不同的阈值
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # 分割数据
                left_y = []
                right_y = []
                
                for j, sample in enumerate(X):
                    if sample[feature_idx] <= threshold:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])
                
                # 计算信息增益
                if len(left_y) > 0 and len(right_y) > 0:
                    gain = calculate_information_gain(y, left_y, right_y)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _split_data(self, X, y, feature_index, threshold):
        """根据特征和阈值分割数据"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i, sample in enumerate(X):
            if sample[feature_index] <= threshold:
                left_X.append(sample)
                left_y.append(y[i])
            else:
                right_X.append(sample)
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def predict(self, X):
        """预测"""
        if isinstance(X[0], (int, float)):  # 单个样本
            return self._predict_single(self.root, X)
        else:  # 多个样本
            return [self._predict_single(self.root, sample) for sample in X]
    
    def _predict_single(self, node, sample):
        """预测单个样本"""
        if node.is_leaf:
            return node.prediction
        
        if sample[node.feature_index] <= node.threshold:
            return self._predict_single(node.left, sample)
        else:
            return self._predict_single(node.right, sample)
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """打印决策树结构"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf:
            print(f"{indent}{prefix}: 预测 = {node.prediction}")
        else:
            feature_name = self.feature_names[node.feature_index]
            print(f"{indent}{prefix}: {feature_name} <= {node.threshold:.2f}?")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "是")
            if node.right:
                self.print_tree(node.right, depth + 1, "否")

def entropy_demo():
    """熵的概念演示"""
    print("\n=== 熵(Entropy)概念演示 ===")
    
    test_cases = [
        ([1, 1, 1, 1], "全部同类"),
        ([1, 0, 1, 0], "完全混乱"),
        ([1, 1, 1, 0], "轻微混乱"),
        ([1, 1, 1, 1, 1, 0], "偏向一类")
    ]
    
    print("熵值计算示例：")
    print("数据集        | 描述     | 熵值")
    print("-" * 35)
    
    for labels, description in test_cases:
        entropy = calculate_entropy(labels)
        print(f"{str(labels):<12} | {description:<8} | {entropy:.3f}")
    
    print("\n熵的意义：")
    print("- 熵值为0：数据完全纯净，全部属于同一类")
    print("- 熵值为1：数据完全混乱，各类别数量相等")
    print("- 熵值越小，数据越纯净")

def information_gain_demo():
    """信息增益演示"""
    print("\n=== 信息增益演示 ===")
    
    # 示例：根据天气预测是否外出
    print("示例：根据天气情况预测是否外出")
    
    # 原始数据：[晴天, 阴天, 雨天] -> [外出, 不外出]
    weather = ["晴", "晴", "阴", "雨", "雨", "雨", "阴", "晴", "晴", "雨"]
    go_out = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]  # 1=外出, 0=不外出
    
    print("原始数据:")
    for w, g in zip(weather, go_out):
        print(f"天气: {w}, 外出: {'是' if g else '否'}")
    
    # 计算原始熵
    original_entropy = calculate_entropy(go_out)
    print(f"\n原始数据熵值: {original_entropy:.3f}")
    
    # 按天气类型分割
    sunny_labels = [go_out[i] for i, w in enumerate(weather) if w == "晴"]
    cloudy_labels = [go_out[i] for i, w in enumerate(weather) if w == "阴"]
    rainy_labels = [go_out[i] for i, w in enumerate(weather) if w == "雨"]
    
    print(f"\n按天气分割后:")
    print(f"晴天: {sunny_labels}, 熵值: {calculate_entropy(sunny_labels):.3f}")
    print(f"阴天: {cloudy_labels}, 熵值: {calculate_entropy(cloudy_labels):.3f}")
    print(f"雨天: {rainy_labels}, 熵值: {calculate_entropy(rainy_labels):.3f}")
    
    # 计算加权平均熵
    total = len(go_out)
    weighted_entropy = (len(sunny_labels)/total * calculate_entropy(sunny_labels) +
                       len(cloudy_labels)/total * calculate_entropy(cloudy_labels) +
                       len(rainy_labels)/total * calculate_entropy(rainy_labels))
    
    information_gain = original_entropy - weighted_entropy
    print(f"\n信息增益: {original_entropy:.3f} - {weighted_entropy:.3f} = {information_gain:.3f}")

def practical_example():
    """实际应用示例：学生成绩预测"""
    print("\n=== 实际应用：学生是否能通过考试 ===")
    
    # 特征：[学习时间(小时), 作业完成率(%), 出勤率(%)]
    # 标签：1=通过, 0=不通过
    
    training_data = [
        ([8, 85, 90], 1), ([12, 95, 95], 1), ([15, 90, 85], 1),
        ([3, 60, 70], 0), ([5, 70, 80], 0), ([2, 50, 65], 0),
        ([10, 88, 92], 1), ([6, 75, 85], 0), ([14, 92, 88], 1),
        ([4, 65, 75], 0), ([11, 85, 90], 1), ([7, 80, 82], 1)
    ]
    
    # 分离特征和标签
    X = [sample[0] for sample in training_data]
    y = [sample[1] for sample in training_data]
    feature_names = ["学习时间", "作业完成率", "出勤率"]
    
    print("训练数据:")
    print("学习时间 | 作业完成率 | 出勤率 | 通过考试")
    print("-" * 40)
    for i, (features, label) in enumerate(training_data):
        result = "是" if label else "否"
        print(f"{features[0]:6d}   | {features[1]:8d}   | {features[2]:4d}   | {result}")
    
    # 训练决策树
    tree = SimpleDecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(X, y, feature_names)
    
    print(f"\n决策树结构:")
    tree.print_tree()
    
    # 预测新学生
    test_students = [
        [9, 80, 85],   # 中等学生
        [13, 90, 95],  # 优秀学生
        [3, 55, 60]    # 较差学生
    ]
    
    print(f"\n预测结果:")
    for i, student in enumerate(test_students):
        prediction = tree.predict(student)
        result = "通过" if prediction else "不通过"
        print(f"学生{i+1}：学习{student[0]}小时，作业{student[1]}%，出勤{student[2]}% -> 预测：{result}")

def decision_tree_vs_linear_models():
    """决策树与线性模型对比"""
    print("\n=== 决策树 vs 线性模型对比 ===")
    
    comparison = {
        "模型类型": {"线性模型": "参数化模型", "决策树": "非参数化模型"},
        "决策边界": {"线性模型": "线性分割", "决策树": "轴平行矩形分割"},
        "可解释性": {"线性模型": "系数含义明确", "决策树": "规则清晰易懂"},
        "数据要求": {"线性模型": "需要数值化", "决策树": "处理混合类型"},
        "过拟合": {"线性模型": "较少过拟合", "决策树": "容易过拟合"},
        "非线性": {"线性模型": "无法捕获", "决策树": "自然处理"},
        "训练速度": {"线性模型": "较快", "决策树": "中等"},
        "预测速度": {"线性模型": "很快", "决策树": "快"},
    }
    
    print(f"{'特征':<12} | {'线性模型':<20} | {'决策树':<20}")
    print("-" * 60)
    for feature, values in comparison.items():
        print(f"{feature:<12} | {values['线性模型']:<20} | {values['决策树']:<20}")

if __name__ == "__main__":
    decision_tree_theory()
    entropy_demo()
    information_gain_demo()
    practical_example()
    decision_tree_vs_linear_models()
    
    print("\n=== 总结 ===")
    print("决策树算法的核心要点：")
    print("• 通过递归分割构建分类规则")
    print("• 使用信息增益选择最佳分割特征")
    print("• 熵衡量数据的混乱程度")
    print("• 易于理解但容易过拟合")
    print("• 可以处理非线性关系")
    
    print("\n实际应用建议：")
    print("• 设置合适的最大深度防止过拟合")
    print("• 可以结合随机森林提高性能")
    print("• 适合作为基线模型和特征选择")
    print("• 在规则清晰的业务场景中效果很好")