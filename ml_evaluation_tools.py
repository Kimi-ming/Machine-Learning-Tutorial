# 机器学习评估工具
# Model Evaluation Tools: 交叉验证、性能评估、模型比较

import random
import math
from collections import Counter

def evaluation_theory():
    """
    模型评估理论
    
    为什么需要模型评估？
    - 评估模型的泛化能力
    - 比较不同模型的性能
    - 选择最优的超参数
    - 检测过拟合和欠拟合
    
    常用评估方法：
    1. 留出验证(Hold-out Validation)
    2. 交叉验证(Cross Validation)
    3. 留一验证(Leave-One-Out)
    4. 自助法(Bootstrap)
    
    评估指标：
    - 回归：MAE, MSE, RMSE, R²
    - 分类：Accuracy, Precision, Recall, F1-Score
    - 混淆矩阵(Confusion Matrix)
    """
    print("=== 模型评估方法原理 ===")
    print("目标：客观评估模型的真实性能")
    print("原则：训练集训练，验证集调参，测试集评估")
    print("重点：避免数据泄露，确保评估的公正性")
    print()

class DataSplitter:
    """数据分割工具"""
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        """
        将数据分割为训练集和测试集
        """
        if random_state:
            random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # 创建索引并随机打乱
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        # 分割索引
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # 分割数据
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=None):
        """
        将数据分割为训练集、验证集和测试集
        """
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = DataSplitter.train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # 再从剩余数据中分离验证集
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = DataSplitter.train_test_split(
            X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

class KFoldCrossValidator:
    """K折交叉验证"""
    
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = random_state
    
    def split(self, X, y):
        """
        生成K个训练-验证数据对
        """
        if self.random_state:
            random.seed(self.random_state)
        
        n_samples = len(X)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        fold_size = n_samples // self.k
        folds = []
        
        for i in range(self.k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.k - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = indices[:start_idx] + indices[end_idx:]
            
            X_train = [X[j] for j in train_indices]
            X_val = [X[j] for j in val_indices]
            y_train = [y[j] for j in train_indices]
            y_val = [y[j] for j in val_indices]
            
            folds.append((X_train, X_val, y_train, y_val))
        
        return folds

class RegressionMetrics:
    """回归评估指标"""
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """平均绝对误差"""
        return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """均方误差"""
        return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """均方根误差"""
        mse = RegressionMetrics.mean_squared_error(y_true, y_pred)
        return math.sqrt(mse)
    
    @staticmethod
    def r_squared(y_true, y_pred):
        """R²决定系数"""
        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum((y - y_mean) ** 2 for y in y_true)
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        
        if ss_tot == 0:
            return 1 if ss_res == 0 else 0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def evaluate_regression(y_true, y_pred, print_results=True):
        """综合回归评估"""
        mae = RegressionMetrics.mean_absolute_error(y_true, y_pred)
        mse = RegressionMetrics.mean_squared_error(y_true, y_pred)
        rmse = RegressionMetrics.root_mean_squared_error(y_true, y_pred)
        r2 = RegressionMetrics.r_squared(y_true, y_pred)
        
        if print_results:
            print("=== 回归模型评估结果 ===")
            print(f"平均绝对误差(MAE): {mae:.4f}")
            print(f"均方误差(MSE): {mse:.4f}")
            print(f"均方根误差(RMSE): {rmse:.4f}")
            print(f"R²决定系数: {r2:.4f}")
            
            # 解释R²
            if r2 >= 0.9:
                print("模型拟合效果: 优秀")
            elif r2 >= 0.7:
                print("模型拟合效果: 良好")
            elif r2 >= 0.5:
                print("模型拟合效果: 中等")
            else:
                print("模型拟合效果: 需要改进")
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2
        }

class ClassificationMetrics:
    """分类评估指标"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """准确率"""
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """混淆矩阵"""
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        
        n_labels = len(labels)
        matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        for true, pred in zip(y_true, y_pred):
            true_idx = label_to_idx[true]
            pred_idx = label_to_idx[pred]
            matrix[true_idx][pred_idx] += 1
        
        return matrix, labels
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred, positive_label=1):
        """计算精确率、召回率、F1分数（二分类）"""
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == positive_label and pred == positive_label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true != positive_label and pred == positive_label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == positive_label and pred != positive_label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, print_results=True):
        """综合分类评估"""
        acc = ClassificationMetrics.accuracy(y_true, y_pred)
        matrix, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': acc,
            'confusion_matrix': matrix,
            'labels': labels
        }
        
        # 如果是二分类，计算更多指标
        if len(labels) == 2:
            precision, recall, f1 = ClassificationMetrics.precision_recall_f1(y_true, y_pred)
            results.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        if print_results:
            print("=== 分类模型评估结果 ===")
            print(f"准确率(Accuracy): {acc:.4f}")
            
            if len(labels) == 2:
                print(f"精确率(Precision): {precision:.4f}")
                print(f"召回率(Recall): {recall:.4f}")
                print(f"F1分数: {f1:.4f}")
            
            print("\n混淆矩阵:")
            print("预测\\真实", end="")
            for label in labels:
                print(f"{label:6}", end="")
            print()
            
            for i, label in enumerate(labels):
                print(f"{label:8}", end="")
                for j in range(len(labels)):
                    print(f"{matrix[i][j]:6}", end="")
                print()
        
        return results

class ModelComparator:
    """模型比较工具"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_result(self, model_name, metrics):
        """添加模型评估结果"""
        self.results[model_name] = metrics
    
    def compare_regression_models(self):
        """比较回归模型"""
        if not self.results:
            print("没有模型结果可比较")
            return
        
        print("=== 回归模型性能比较 ===")
        print(f"{'模型名称':<15} | {'MAE':<8} | {'MSE':<8} | {'RMSE':<8} | {'R²':<8}")
        print("-" * 65)
        
        best_models = {
            'mae': (float('inf'), ""),
            'mse': (float('inf'), ""),
            'rmse': (float('inf'), ""),
            'r2': (-float('inf'), "")
        }
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<15} | {metrics['mae']:<8.4f} | {metrics['mse']:<8.4f} | "
                  f"{metrics['rmse']:<8.4f} | {metrics['r2']:<8.4f}")
            
            # 更新最优模型
            if metrics['mae'] < best_models['mae'][0]:
                best_models['mae'] = (metrics['mae'], model_name)
            if metrics['mse'] < best_models['mse'][0]:
                best_models['mse'] = (metrics['mse'], model_name)
            if metrics['rmse'] < best_models['rmse'][0]:
                best_models['rmse'] = (metrics['rmse'], model_name)
            if metrics['r2'] > best_models['r2'][0]:
                best_models['r2'] = (metrics['r2'], model_name)
        
        print(f"\n最优模型:")
        print(f"MAE最小: {best_models['mae'][1]} ({best_models['mae'][0]:.4f})")
        print(f"MSE最小: {best_models['mse'][1]} ({best_models['mse'][0]:.4f})")
        print(f"RMSE最小: {best_models['rmse'][1]} ({best_models['rmse'][0]:.4f})")
        print(f"R²最大: {best_models['r2'][1]} ({best_models['r2'][0]:.4f})")
    
    def compare_classification_models(self):
        """比较分类模型"""
        if not self.results:
            print("没有模型结果可比较")
            return
        
        print("=== 分类模型性能比较 ===")
        
        # 检查是否有二分类指标
        has_binary_metrics = all('precision' in metrics for metrics in self.results.values())
        
        if has_binary_metrics:
            print(f"{'模型名称':<15} | {'准确率':<8} | {'精确率':<8} | {'召回率':<8} | {'F1分数':<8}")
            print("-" * 65)
        else:
            print(f"{'模型名称':<15} | {'准确率':<8}")
            print("-" * 30)
        
        best_accuracy = (0, "")
        
        for model_name, metrics in self.results.items():
            if has_binary_metrics:
                print(f"{model_name:<15} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<8.4f} | "
                      f"{metrics['recall']:<8.4f} | {metrics['f1_score']:<8.4f}")
            else:
                print(f"{model_name:<15} | {metrics['accuracy']:<8.4f}")
            
            if metrics['accuracy'] > best_accuracy[0]:
                best_accuracy = (metrics['accuracy'], model_name)
        
        print(f"\n最优模型:")
        print(f"准确率最高: {best_accuracy[1]} ({best_accuracy[0]:.4f})")

def cross_validation_demo():
    """交叉验证演示"""
    print("\n=== 交叉验证演示 ===")
    
    # 模拟简单的分类数据
    print("生成模拟分类数据...")
    X = []
    y = []
    
    random.seed(42)
    for i in range(100):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        # 简单的分类规则：x1 + x2 > 10 为类别1，否则为类别0
        label = 1 if x1 + x2 > 10 else 0
        # 添加一些噪声
        if random.random() < 0.1:  # 10%的噪声
            label = 1 - label
        
        X.append([x1, x2])
        y.append(label)
    
    print(f"数据集大小: {len(X)}个样本")
    print(f"类别分布: {Counter(y)}")
    
    # 简单的逻辑回归模型（用于演示）
    class SimpleLogisticRegression:
        def __init__(self, learning_rate=0.1, max_iterations=100):
            self.learning_rate = learning_rate
            self.max_iterations = max_iterations
            self.weights = [0.0, 0.0]
            self.bias = 0.0
        
        def sigmoid(self, z):
            z = max(min(z, 500), -500)  # 防止溢出
            return 1 / (1 + math.exp(-z))
        
        def fit(self, X, y):
            for _ in range(self.max_iterations):
                for i in range(len(X)):
                    z = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                    pred = self.sigmoid(z)
                    error = pred - y[i]
                    
                    # 更新权重
                    for j in range(len(self.weights)):
                        self.weights[j] -= self.learning_rate * error * X[i][j]
                    self.bias -= self.learning_rate * error
        
        def predict(self, X):
            predictions = []
            for x in X:
                z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
                prob = self.sigmoid(z)
                predictions.append(1 if prob >= 0.5 else 0)
            return predictions
    
    # 执行5折交叉验证
    cv = KFoldCrossValidator(k=5, random_state=42)
    folds = cv.split(X, y)
    
    print(f"\n执行5折交叉验证...")
    accuracies = []
    
    for i, (X_train, X_val, y_train, y_val) in enumerate(folds):
        # 训练模型
        model = SimpleLogisticRegression()
        model.fit(X_train, y_train)
        
        # 预测验证集
        y_pred = model.predict(X_val)
        
        # 计算准确率
        accuracy = ClassificationMetrics.accuracy(y_val, y_pred)
        accuracies.append(accuracy)
        
        print(f"第{i+1}折: 训练样本{len(X_train)}个, 验证样本{len(X_val)}个, 准确率{accuracy:.4f}")
    
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = math.sqrt(sum((acc - mean_accuracy) ** 2 for acc in accuracies) / len(accuracies))
    
    print(f"\n交叉验证结果:")
    print(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"最高准确率: {max(accuracies):.4f}")
    print(f"最低准确率: {min(accuracies):.4f}")

def model_selection_demo():
    """模型选择演示"""
    print("\n=== 模型选择演示 ===")
    
    # 生成回归数据
    print("生成回归数据进行模型比较...")
    X = []
    y = []
    
    random.seed(42)
    for i in range(80):
        x = random.uniform(0, 10)
        # 真实关系是二次函数
        noise = random.gauss(0, 2)
        y_true = 0.5 * x * x - 2 * x + 3 + noise
        X.append([x])
        y.append(y_true)
    
    # 分割数据
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 定义不同复杂度的多项式模型
    class PolynomialRegression:
        def __init__(self, degree):
            self.degree = degree
            self.weights = None
        
        def create_polynomial_features(self, X):
            """创建多项式特征"""
            poly_features = []
            for sample in X:
                x = sample[0]
                features = [x ** i for i in range(1, self.degree + 1)]
                poly_features.append(features)
            return poly_features
        
        def fit(self, X, y):
            # 创建多项式特征
            X_poly = self.create_polynomial_features(X)
            
            # 简单的正规方程实现（仅用于演示）
            self.weights = [0.0] * (self.degree + 1)  # 包含bias
            
            # 使用简单的梯度下降
            learning_rate = 0.001
            for epoch in range(1000):
                for i in range(len(X_poly)):
                    # 预测
                    pred = self.weights[0] + sum(w * f for w, f in zip(self.weights[1:], X_poly[i]))
                    error = pred - y[i]
                    
                    # 更新权重
                    self.weights[0] -= learning_rate * error  # bias
                    for j in range(len(X_poly[i])):
                        self.weights[j + 1] -= learning_rate * error * X_poly[i][j]
        
        def predict(self, X):
            X_poly = self.create_polynomial_features(X)
            predictions = []
            for features in X_poly:
                pred = self.weights[0] + sum(w * f for w, f in zip(self.weights[1:], features))
                predictions.append(pred)
            return predictions
    
    # 比较不同阶数的多项式模型
    degrees = [1, 2, 3, 5]
    comparator = ModelComparator()
    
    for degree in degrees:
        print(f"\n训练{degree}阶多项式模型...")
        model = PolynomialRegression(degree)
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = model.predict(X_test)
        metrics = RegressionMetrics.evaluate_regression(y_test, y_pred, print_results=False)
        
        comparator.add_model_result(f"{degree}阶多项式", metrics)
        print(f"{degree}阶多项式 - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    # 比较所有模型
    print()
    comparator.compare_regression_models()

def evaluation_best_practices():
    """评估最佳实践"""
    print("\n=== 模型评估最佳实践 ===")
    
    print("1. 数据分割原则:")
    print("   - 训练集:验证集:测试集 = 60%:20%:20%")
    print("   - 确保各个子集的数据分布相似")
    print("   - 测试集只在最终评估时使用一次")
    
    print("\n2. 交叉验证使用:")
    print("   - 数据量小时使用K折交叉验证(K=5或10)")
    print("   - 数据量大时使用留出验证")
    print("   - 时间序列数据使用时间序列交叉验证")
    
    print("\n3. 评估指标选择:")
    print("   - 回归问题优先关注RMSE和R²")
    print("   - 分类问题需要综合考虑准确率、精确率、召回率")
    print("   - 不平衡数据集重点关注F1-Score")
    
    print("\n4. 避免常见错误:")
    print("   - 数据泄露：确保测试数据不参与训练")
    print("   - 过度调参：在验证集上过度优化会导致过拟合")
    print("   - 忽视基线：总是与简单模型比较")
    
    print("\n5. 模型选择策略:")
    print("   - 从简单模型开始")
    print("   - 逐步增加复杂度")
    print("   - 使用交叉验证选择超参数")
    print("   - 考虑模型的可解释性和计算成本")

if __name__ == "__main__":
    evaluation_theory()
    cross_validation_demo()
    model_selection_demo()
    evaluation_best_practices()
    
    print("\n=== 总结 ===")
    print("模型评估是机器学习的重要环节：")
    print("• 合理的数据分割是评估的基础")
    print("• 交叉验证能更可靠地评估模型性能")
    print("• 选择合适的评估指标很关键")
    print("• 避免过拟合和数据泄露")
    print("• 模型比较要全面客观")
    
    print("\n实践建议：")
    print("• 建立标准的评估流程")
    print("• 保存评估结果用于比较")
    print("• 关注模型的泛化能力")
    print("• 结合业务需求选择模型")