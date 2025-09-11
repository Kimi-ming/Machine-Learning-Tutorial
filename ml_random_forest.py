# 机器学习算法教程 - 随机森林
# Random Forest: 集成学习的经典算法

import random
import math
from collections import Counter

def random_forest_theory():
    """
    随机森林原理解释
    
    随机森林是一种集成学习算法，基于多个决策树的组合。
    
    核心思想：
    - 构建多个决策树，每个树在不同的训练子集上训练
    - 对于分类问题，使用多数投票进行预测
    - 对于回归问题，使用平均值进行预测
    
    关键技术：
    1. Bootstrap采样：有放回地抽样训练数据
    2. 特征随机选择：每次分裂时只考虑部分特征
    3. 多数投票/平均：集成多个决策树的结果
    
    随机性来源：
    - 训练样本的随机性（Bootstrap采样）
    - 特征选择的随机性（每次分裂随机选择特征子集）
    
    优点：
    - 减少过拟合，泛化能力强
    - 对异常值和噪声数据robust
    - 可以处理大型数据集
    - 提供特征重要性评估
    - 训练可以并行化
    
    缺点：
    - 解释性不如单个决策树
    - 在某些数据集上可能过拟合
    - 对类别不平衡敏感
    - 内存消耗较大
    """
    print("=== 随机森林算法原理 ===")
    print("目标：通过集成多个决策树提高预测性能")
    print("方法：Bootstrap采样 + 特征随机选择 + 投票机制")
    print("应用：分类和回归问题，特征选择，异常检测")
    print()

class BootstrapSampler:
    """Bootstrap采样器"""
    
    @staticmethod
    def sample(X, y, n_samples=None, random_state=None):
        """
        Bootstrap采样：有放回地抽样
        """
        if random_state:
            random.seed(random_state)
        
        n_original = len(X)
        if n_samples is None:
            n_samples = n_original
        
        # 有放回抽样
        indices = [random.randint(0, n_original - 1) for _ in range(n_samples)]
        
        X_bootstrap = [X[i] for i in indices]
        y_bootstrap = [y[i] for i in indices]
        
        # 记录袋外样本（Out-of-Bag samples）
        oob_indices = []
        for i in range(n_original):
            if i not in indices:
                oob_indices.append(i)
        
        X_oob = [X[i] for i in oob_indices]
        y_oob = [y[i] for i in oob_indices]
        
        return X_bootstrap, y_bootstrap, X_oob, y_oob

class RandomDecisionTree:
    """随机决策树（用于随机森林）"""
    
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 每次分裂时考虑的特征数
        self.tree = None
        self.feature_importances = {}
    
    def _calculate_entropy(self, y):
        """计算熵"""
        if not y:
            return 0
        
        counter = Counter(y)
        total = len(y)
        entropy = 0
        
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_information_gain(self, y, left_y, right_y):
        """计算信息增益"""
        if not left_y or not right_y:
            return 0
        
        parent_entropy = self._calculate_entropy(y)
        total_samples = len(y)
        
        left_weight = len(left_y) / total_samples
        right_weight = len(right_y) / total_samples
        
        weighted_entropy = (left_weight * self._calculate_entropy(left_y) + 
                           right_weight * self._calculate_entropy(right_y))
        
        return parent_entropy - weighted_entropy
    
    def _get_random_features(self, n_total_features):
        """随机选择特征子集"""
        if self.max_features is None:
            # 默认使用sqrt(总特征数)个特征
            n_features = int(math.sqrt(n_total_features))
        else:
            n_features = min(self.max_features, n_total_features)
        
        return random.sample(range(n_total_features), n_features)
    
    def _find_best_split(self, X, y):
        """寻找最佳分裂点（只在随机特征子集中）"""
        n_samples, n_features = len(X), len(X[0])
        
        if n_samples < self.min_samples_split:
            return None, None, 0
        
        # 随机选择特征子集
        feature_indices = self._get_random_features(n_features)
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            # 获取该特征的所有唯一值
            feature_values = [sample[feature_idx] for sample in X]
            unique_values = sorted(set(feature_values))
            
            if len(unique_values) < 2:
                continue
            
            # 尝试不同的分割点
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # 按阈值分割数据
                left_y = []
                right_y = []
                
                for j, sample in enumerate(X):
                    if sample[feature_idx] <= threshold:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # 计算信息增益
                gain = self._calculate_information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件
        if (depth >= self.max_depth or 
            len(set(y)) <= 1 or 
            len(y) < self.min_samples_split):
            
            # 返回叶节点（多数类）
            counter = Counter(y)
            prediction = counter.most_common(1)[0][0]
            return {'prediction': prediction, 'samples': len(y)}
        
        # 寻找最佳分割
        feature_idx, threshold, gain = self._find_best_split(X, y)
        
        if feature_idx is None or gain == 0:
            # 无法进一步分割
            counter = Counter(y)
            prediction = counter.most_common(1)[0][0]
            return {'prediction': prediction, 'samples': len(y)}
        
        # 分割数据
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i, sample in enumerate(X):
            if sample[feature_idx] <= threshold:
                left_X.append(sample)
                left_y.append(y[i])
            else:
                right_X.append(sample)
                right_y.append(y[i])
        
        # 更新特征重要性
        if feature_idx not in self.feature_importances:
            self.feature_importances[feature_idx] = 0
        self.feature_importances[feature_idx] += gain * len(y)
        
        # 递归构建子树
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': len(y)
        }
    
    def fit(self, X, y):
        """训练决策树"""
        self.feature_importances = {}
        self.tree = self._build_tree(X, y)
        
        # 标准化特征重要性
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            for feature_idx in self.feature_importances:
                self.feature_importances[feature_idx] /= total_importance
    
    def _predict_sample(self, sample, node):
        """预测单个样本"""
        if 'prediction' in node:
            return node['prediction']
        
        feature_value = sample[node['feature_idx']]
        
        if feature_value <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])
    
    def predict(self, X):
        """预测"""
        if self.tree is None:
            raise ValueError("树未训练")
        
        if isinstance(X[0], (int, float)):
            # 单个样本
            return self._predict_sample(X, self.tree)
        else:
            # 多个样本
            return [self._predict_sample(sample, self.tree) for sample in X]

class RandomForest:
    """随机森林实现"""
    
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, 
                 max_features=None, random_state=None):
        """
        初始化随机森林
        n_estimators: 树的数量
        max_depth: 单个树的最大深度
        min_samples_split: 分裂所需的最小样本数
        max_features: 每次分裂时考虑的特征数
        random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None
    
    def fit(self, X, y):
        """训练随机森林"""
        print(f"开始训练随机森林：{self.n_estimators}棵树")
        
        if self.random_state:
            random.seed(self.random_state)
        
        self.trees = []
        n_features = len(X[0])
        
        # 初始化特征重要性
        total_feature_importances = {i: 0.0 for i in range(n_features)}
        
        # 训练每棵树
        for i in range(self.n_estimators):
            # Bootstrap采样
            X_bootstrap, y_bootstrap, _, _ = BootstrapSampler.sample(
                X, y, random_state=None if self.random_state is None 
                else self.random_state + i)
            
            # 创建并训练决策树
            tree = RandomDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # 累加特征重要性
            for feature_idx, importance in tree.feature_importances.items():
                total_feature_importances[feature_idx] += importance
            
            if (i + 1) % max(1, self.n_estimators // 5) == 0:
                print(f"已训练 {i + 1}/{self.n_estimators} 棵树")
        
        # 计算平均特征重要性
        self.feature_importances_ = {}
        for feature_idx, total_importance in total_feature_importances.items():
            self.feature_importances_[feature_idx] = total_importance / self.n_estimators
        
        # 计算袋外得分
        self._calculate_oob_score(X, y)
        
        print("随机森林训练完成！")
        print(f"袋外得分: {self.oob_score_:.4f}" if self.oob_score_ else "")
    
    def _calculate_oob_score(self, X, y):
        """计算袋外得分"""
        oob_predictions = {}
        oob_counts = {}
        
        # 对每棵树计算其袋外样本的预测
        for tree_idx, tree in enumerate(self.trees):
            # 重新生成Bootstrap样本以获得袋外样本
            random.seed(self.random_state + tree_idx if self.random_state else tree_idx)
            _, _, X_oob, y_oob = BootstrapSampler.sample(X, y)
            
            if not X_oob:
                continue
            
            # 预测袋外样本
            oob_pred = tree.predict(X_oob)
            
            # 找出袋外样本在原始数据中的索引
            for i, sample in enumerate(X_oob):
                # 简单的样本匹配（实际应该用更robust的方法）
                for j, orig_sample in enumerate(X):
                    if sample == orig_sample:
                        if j not in oob_predictions:
                            oob_predictions[j] = {}
                            oob_counts[j] = {}
                        
                        pred = oob_pred[i]
                        if pred not in oob_predictions[j]:
                            oob_predictions[j][pred] = 0
                        oob_predictions[j][pred] += 1
                        break
        
        # 计算最终的袋外预测和准确率
        correct = 0
        total = 0
        
        for sample_idx, pred_counts in oob_predictions.items():
            if pred_counts:
                final_pred = max(pred_counts.keys(), key=lambda k: pred_counts[k])
                if final_pred == y[sample_idx]:
                    correct += 1
                total += 1
        
        self.oob_score_ = correct / total if total > 0 else None
    
    def predict(self, X):
        """预测"""
        if not self.trees:
            raise ValueError("模型未训练")
        
        if isinstance(X[0], (int, float)):
            # 单个样本
            return self._predict_single(X)
        else:
            # 多个样本
            return [self._predict_single(sample) for sample in X]
    
    def _predict_single(self, sample):
        """预测单个样本（多数投票）"""
        votes = {}
        
        for tree in self.trees:
            prediction = tree.predict(sample)
            if prediction not in votes:
                votes[prediction] = 0
            votes[prediction] += 1
        
        # 返回得票最多的类别
        return max(votes.keys(), key=lambda k: votes[k])
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.trees:
            raise ValueError("模型未训练")
        
        if isinstance(X[0], (int, float)):
            return self._predict_proba_single(X)
        else:
            return [self._predict_proba_single(sample) for sample in X]
    
    def _predict_proba_single(self, sample):
        """预测单个样本的概率"""
        votes = {}
        
        for tree in self.trees:
            prediction = tree.predict(sample)
            if prediction not in votes:
                votes[prediction] = 0
            votes[prediction] += 1
        
        # 转换为概率
        total_votes = sum(votes.values())
        probabilities = {label: count / total_votes for label, count in votes.items()}
        
        return probabilities
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.feature_importances_ is None:
            return {}
        return self.feature_importances_.copy()

def bootstrap_sampling_demo():
    """Bootstrap采样演示"""
    print("\n=== Bootstrap采样演示 ===")
    
    # 创建示例数据
    X = [[i, i*2] for i in range(10)]  # 10个样本
    y = [i % 2 for i in range(10)]     # 二分类标签
    
    print("原始数据:")
    print("样本索引  特征         标签")
    print("-" * 30)
    for i, (sample, label) in enumerate(zip(X, y)):
        print(f"    {i}     {sample}    {label}")
    
    # 进行Bootstrap采样
    print(f"\n进行Bootstrap采样（样本数={len(X)}）：")
    X_boot, y_boot, X_oob, y_oob = BootstrapSampler.sample(X, y, random_state=42)
    
    print(f"Bootstrap样本数: {len(X_boot)}")
    print(f"袋外样本数: {len(X_oob)}")
    
    print(f"\nBootstrap样本 (前10个):")
    for i, (sample, label) in enumerate(zip(X_boot[:10], y_boot[:10])):
        # 找出原样本索引
        orig_idx = X.index(sample) if sample in X else "?"
        print(f"  {sample} (标签={label}, 原索引={orig_idx})")
    
    print(f"\n袋外样本:")
    for sample, label in zip(X_oob, y_oob):
        orig_idx = X.index(sample)
        print(f"  {sample} (标签={label}, 原索引={orig_idx})")
    
    print(f"\n统计信息:")
    print(f"原数据中每个样本被选中的次数:")
    boot_count = Counter(X.index(sample) for sample in X_boot)
    for i in range(len(X)):
        count = boot_count.get(i, 0)
        status = "袋外" if count == 0 else f"{count}次"
        print(f"  样本{i}: {status}")

def feature_importance_demo():
    """特征重要性演示"""
    print("\n=== 特征重要性演示 ===")
    
    # 创建示例数据：其中某些特征更重要
    random.seed(42)
    X = []
    y = []
    
    for i in range(200):
        # 特征1：最重要的特征
        f1 = random.uniform(0, 10)
        # 特征2：中等重要
        f2 = random.uniform(0, 5)
        # 特征3：噪声特征
        f3 = random.gauss(0, 1)
        # 特征4：与特征1相关
        f4 = f1 * 0.3 + random.gauss(0, 0.5)
        
        # 标签主要由特征1和特征2决定
        if f1 > 6 or f2 > 3:
            label = 1
        else:
            label = 0
        
        # 添加10%的噪声
        if random.random() < 0.1:
            label = 1 - label
        
        X.append([f1, f2, f3, f4])
        y.append(label)
    
    print(f"生成数据：{len(X)}个样本，4个特征")
    print("特征设计:")
    print("  特征1: 最重要（主要决定因子）")
    print("  特征2: 中等重要（次要决定因子）")
    print("  特征3: 噪声特征（随机）")
    print("  特征4: 与特征1相关")
    
    # 训练随机森林
    rf = RandomForest(n_estimators=20, max_depth=6, random_state=42)
    rf.fit(X, y)
    
    # 分析特征重要性
    importance = rf.get_feature_importance()
    
    print(f"\n特征重要性排序:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (feature_idx, imp) in enumerate(sorted_features):
        feature_names = ["特征1(重要)", "特征2(中等)", "特征3(噪声)", "特征4(相关)"]
        print(f"  第{rank+1}名: {feature_names[feature_idx]} - 重要性: {imp:.4f}")
    
    # 可视化特征重要性
    print(f"\n特征重要性柱状图:")
    max_importance = max(importance.values())
    
    for feature_idx in range(4):
        imp = importance.get(feature_idx, 0)
        bar_length = int((imp / max_importance) * 30)
        bar = "█" * bar_length
        feature_names = ["特征1", "特征2", "特征3", "特征4"]
        print(f"{feature_names[feature_idx]}  |{bar:<30}| {imp:.4f}")

def random_forest_vs_single_tree():
    """随机森林与单棵树性能比较"""
    print("\n=== 随机森林 vs 单决策树比较 ===")
    
    # 生成带噪声的分类数据
    random.seed(42)
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    # 训练数据
    for i in range(300):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(0, 10)
        
        # 复杂的分类规则
        if (x1 > 5 and x2 > 5) or (x1 < 3 and x2 < 3 and x3 > 7):
            label = 1
        else:
            label = 0
        
        # 添加15%噪声
        if random.random() < 0.15:
            label = 1 - label
        
        X_train.append([x1, x2, x3])
        y_train.append(label)
    
    # 测试数据
    for i in range(100):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(0, 10)
        
        if (x1 > 5 and x2 > 5) or (x1 < 3 and x2 < 3 and x3 > 7):
            label = 1
        else:
            label = 0
        
        X_test.append([x1, x2, x3])
        y_test.append(label)
    
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    print(f"分类分布 - 训练集: {Counter(y_train)}")
    print(f"分类分布 - 测试集: {Counter(y_test)}")
    
    # 训练单决策树
    print(f"\n训练单决策树...")
    single_tree = RandomDecisionTree(max_depth=8, min_samples_split=5)
    single_tree.fit(X_train, y_train)
    
    # 训练随机森林
    print(f"\n训练随机森林...")
    rf = RandomForest(n_estimators=20, max_depth=6, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # 评估性能
    # 训练集性能
    train_pred_tree = single_tree.predict(X_train)
    train_pred_rf = rf.predict(X_train)
    
    train_acc_tree = sum(1 for t, p in zip(y_train, train_pred_tree) if t == p) / len(y_train)
    train_acc_rf = sum(1 for t, p in zip(y_train, train_pred_rf) if t == p) / len(y_train)
    
    # 测试集性能
    test_pred_tree = single_tree.predict(X_test)
    test_pred_rf = rf.predict(X_test)
    
    test_acc_tree = sum(1 for t, p in zip(y_test, test_pred_tree) if t == p) / len(y_test)
    test_acc_rf = sum(1 for t, p in zip(y_test, test_pred_rf) if t == p) / len(y_test)
    
    # 结果对比
    print(f"\n性能对比:")
    print(f"{'模型':<12} | {'训练准确率':<10} | {'测试准确率':<10} | {'过拟合程度'}")
    print("-" * 55)
    
    overfitting_tree = train_acc_tree - test_acc_tree
    overfitting_rf = train_acc_rf - test_acc_rf
    
    print(f"{'单决策树':<12} | {train_acc_tree:<10.4f} | {test_acc_tree:<10.4f} | {overfitting_tree:>10.4f}")
    print(f"{'随机森林':<12} | {train_acc_rf:<10.4f} | {test_acc_rf:<10.4f} | {overfitting_rf:>10.4f}")
    
    print(f"\n分析:")
    if test_acc_rf > test_acc_tree:
        print("✓ 随机森林的泛化性能更好")
    if overfitting_rf < overfitting_tree:
        print("✓ 随机森林的过拟合程度更低")
    
    print(f"袋外得分: {rf.oob_score_:.4f}" if rf.oob_score_ else "")

def hyperparameter_tuning_demo():
    """超参数调优演示"""
    print("\n=== 随机森林超参数调优演示 ===")
    
    # 生成数据
    random.seed(42)
    X, y = [], []
    
    for i in range(500):
        x1 = random.uniform(0, 20)
        x2 = random.uniform(0, 20)
        x3 = random.uniform(0, 20)
        
        # 复杂的分类规则
        if x1**2 + x2**2 < 100 or (x1 > 15 and x3 > 15):
            label = 1
        else:
            label = 0
        
        X.append([x1, x2, x3])
        y.append(label)
    
    # 分割训练集和验证集
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"数据集大小: 训练集{len(X_train)}，验证集{len(X_val)}")
    
    # 超参数网格
    param_grid = {
        'n_estimators': [5, 10, 20],
        'max_depth': [3, 5, 8],
        'max_features': [1, 2, 3]
    }
    
    print(f"\n超参数网格搜索:")
    print(f"n_estimators: {param_grid['n_estimators']}")
    print(f"max_depth: {param_grid['max_depth']}")
    print(f"max_features: {param_grid['max_features']}")
    
    best_score = 0
    best_params = None
    results = []
    
    print(f"\n搜索结果:")
    print(f"{'n_trees':<8} | {'depth':<5} | {'features':<8} | {'验证准确率':<10}")
    print("-" * 45)
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for max_f in param_grid['max_features']:
                # 训练模型
                rf = RandomForest(
                    n_estimators=n_est,
                    max_depth=max_d,
                    max_features=max_f,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                
                # 验证集评估
                val_pred = rf.predict(X_val)
                val_acc = sum(1 for t, p in zip(y_val, val_pred) if t == p) / len(y_val)
                
                results.append((n_est, max_d, max_f, val_acc))
                print(f"{n_est:<8} | {max_d:<5} | {max_f:<8} | {val_acc:<10.4f}")
                
                if val_acc > best_score:
                    best_score = val_acc
                    best_params = (n_est, max_d, max_f)
    
    print(f"\n最佳参数组合:")
    print(f"n_estimators: {best_params[0]}")
    print(f"max_depth: {best_params[1]}")
    print(f"max_features: {best_params[2]}")
    print(f"最佳验证准确率: {best_score:.4f}")

def ensemble_methods_comparison():
    """集成方法比较"""
    print("\n=== 集成学习方法对比 ===")
    
    comparison_data = [
        ("方法", "基学习器", "采样策略", "特征选择", "组合方式", "主要优点"),
        ("随机森林", "决策树", "Bootstrap", "随机子集", "投票/平均", "减少过拟合，并行化"),
        ("梯度提升", "决策树", "全样本", "全特征", "加权组合", "预测精度高"),
        ("AdaBoost", "弱分类器", "重采样", "全特征", "加权投票", "适应性强"),
        ("Bagging", "任意", "Bootstrap", "全特征", "投票/平均", "方差减少"),
        ("Extra Trees", "极随机树", "全样本", "随机分割", "投票/平均", "训练快速")
    ]
    
    # 打印对比表
    col_widths = [12, 12, 10, 10, 10, 18]
    
    for i, row in enumerate(comparison_data):
        if i == 0:  # 表头
            print("=" * sum(col_widths))
        
        for j, cell in enumerate(row):
            print(f"{cell:<{col_widths[j]}}", end="")
        print()
        
        if i == 0:  # 表头后的分隔线
            print("=" * sum(col_widths))
    
    print("=" * sum(col_widths))
    
    print(f"\n随机森林的特点:")
    print("1. 双重随机性：Bootstrap采样 + 特征随机选择")
    print("2. 袋外估计：利用未被选中的样本进行验证")
    print("3. 特征重要性：基于信息增益的特征排序")
    print("4. 并行训练：每棵树可独立训练")
    print("5. 鲁棒性强：对噪声和异常值不敏感")

def random_forest_applications():
    """随机森林应用场景"""
    print("\n=== 随机森林应用场景 ===")
    
    applications = {
        "生物信息学": [
            "基因表达数据分析",
            "蛋白质结构预测", 
            "药物发现",
            "疾病诊断"
        ],
        "金融领域": [
            "信用评分",
            "欺诈检测",
            "风险管理",
            "投资组合优化"
        ],
        "电子商务": [
            "推荐系统",
            "用户行为预测",
            "价格优化",
            "库存管理"
        ],
        "图像识别": [
            "目标检测",
            "医学影像分析",
            "遥感图像分类",
            "人脸识别"
        ],
        "自然语言处理": [
            "文本分类",
            "情感分析",
            "垃圾邮件检测",
            "信息抽取"
        ]
    }
    
    for field, tasks in applications.items():
        print(f"\n{field}:")
        for task in tasks:
            print(f"  • {task}")

if __name__ == "__main__":
    random_forest_theory()
    bootstrap_sampling_demo()
    feature_importance_demo()
    random_forest_vs_single_tree()
    hyperparameter_tuning_demo()
    ensemble_methods_comparison()
    random_forest_applications()
    
    print("\n=== 总结 ===")
    print("随机森林是优秀的集成学习算法：")
    print("• 通过Bootstrap采样增加样本多样性")
    print("• 通过特征随机选择增加模型多样性")  
    print("• 有效减少过拟合，提高泛化能力")
    print("• 提供可靠的特征重要性评估")
    print("• 训练可并行化，适合大规模数据")
    
    print("\n实践建议：")
    print("• 适合作为baseline模型")
    print("• 数据预处理要求相对较低")
    print("• 注意调优树的数量和深度")
    print("• 利用袋外得分进行模型验证")
    print("• 结合特征重要性进行特征选择")