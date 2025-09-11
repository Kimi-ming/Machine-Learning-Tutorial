# 机器学习算法教程 - 支持向量机
# Support Vector Machine: 最大间隔分类器

import random
import math

def svm_theory():
    """
    支持向量机原理解释
    
    支持向量机(SVM)是一种强大的监督学习算法，适用于分类和回归。
    
    核心思想：
    - 寻找最优决策边界，使得不同类别之间的间隔最大
    - 决策边界由少数关键样本（支持向量）确定
    - 通过核函数处理非线性问题
    
    关键概念：
    1. 支持向量：距离决策边界最近的样本点
    2. 间隔(Margin)：不同类别样本到决策边界的最小距离
    3. 核技巧(Kernel Trick)：将数据映射到高维空间
    4. 软间隔：允许部分样本分类错误以提高泛化能力
    
    数学表达：
    - 决策函数：f(x) = sign(ΣαᵢyᵢK(xᵢ,x) + b)
    - 其中αᵢ是拉格朗日乘数，K(xᵢ,x)是核函数
    
    常用核函数：
    1. 线性核：K(x,z) = x·z
    2. 多项式核：K(x,z) = (γx·z + r)^d
    3. RBF核：K(x,z) = exp(-γ||x-z||²)
    
    优点：
    - 在高维空间中有效
    - 内存效率高（只需存储支持向量）
    - 多功能（通过核函数处理非线性）
    
    缺点：
    - 对特征缩放敏感
    - 不直接提供概率估计
    - 对大数据集训练时间长
    """
    print("=== 支持向量机算法原理 ===")
    print("目标：找到最优决策边界，最大化分类间隔")
    print("方法：优化问题求解 + 核函数映射")
    print("应用：文本分类、图像识别、生物信息学等")
    print()

def dot_product(x1, x2):
    """计算向量内积"""
    return sum(a * b for a, b in zip(x1, x2))

def vector_subtract(x1, x2):
    """向量减法"""
    return [a - b for a, b in zip(x1, x2)]

def euclidean_distance_squared(x1, x2):
    """计算欧几里得距离的平方"""
    return sum((a - b) ** 2 for a, b in zip(x1, x2))

class KernelFunctions:
    """核函数集合"""
    
    @staticmethod
    def linear(x1, x2):
        """线性核函数"""
        return dot_product(x1, x2)
    
    @staticmethod
    def polynomial(x1, x2, degree=3, coef0=1):
        """多项式核函数"""
        return (dot_product(x1, x2) + coef0) ** degree
    
    @staticmethod
    def rbf(x1, x2, gamma=0.1):
        """径向基函数(RBF)核"""
        return math.exp(-gamma * euclidean_distance_squared(x1, x2))

class SimplifiedSMO:
    """
    简化的序列最小优化算法(SMO)
    用于求解SVM的对偶问题
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3, coef0=1, 
                 tolerance=0.001, max_iterations=100):
        """
        初始化SVM参数
        C: 正则化参数，控制对误分类的惩罚
        kernel: 核函数类型
        gamma: RBF核的参数
        degree: 多项式核的阶数
        coef0: 多项式核的常数项
        tolerance: 收敛容忍度
        max_iterations: 最大迭代次数
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # 模型参数
        self.alphas = None
        self.b = 0
        self.support_vectors = []
        self.support_vector_labels = []
        self.support_vector_alphas = []
        
    def kernel(self, x1, x2):
        """核函数计算"""
        if self.kernel_type == 'linear':
            return KernelFunctions.linear(x1, x2)
        elif self.kernel_type == 'polynomial':
            return KernelFunctions.polynomial(x1, x2, self.degree, self.coef0)
        elif self.kernel_type == 'rbf':
            return KernelFunctions.rbf(x1, x2, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def fit(self, X, y):
        """
        训练SVM模型
        使用简化的SMO算法
        """
        n_samples = len(X)
        print(f"开始训练SVM：{n_samples}个样本，核函数={self.kernel_type}, C={self.C}")
        
        # 初始化拉格朗日乘数
        self.alphas = [0.0] * n_samples
        self.b = 0
        
        # SMO算法主循环
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iterations:
            num_changed = 0
            iteration += 1
            
            if examine_all:
                # 检查所有样本
                for i in range(n_samples):
                    num_changed += self._examine_example(i, X, y)
            else:
                # 只检查非边界样本（0 < alpha < C）
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i, X, y)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 变化的α数量 = {num_changed}")
        
        # 提取支持向量
        self._extract_support_vectors(X, y)
        
        print(f"训练完成！")
        print(f"总迭代次数: {iteration}")
        print(f"支持向量数量: {len(self.support_vectors)}")
        print(f"支持向量比例: {len(self.support_vectors)/n_samples:.2%}")
    
    def _examine_example(self, i, X, y):
        """检查第i个样本是否需要优化"""
        alpha_i = self.alphas[i]
        y_i = y[i]
        E_i = self._decision_function(X[i], X, y) - y_i
        
        # 检查KKT条件
        r_i = E_i * y_i
        if ((r_i < -self.tolerance and alpha_i < self.C) or 
            (r_i > self.tolerance and alpha_i > 0)):
            
            # 选择第二个变量
            j = self._select_second_alpha(i, E_i, X, y)
            if j >= 0:
                return self._take_step(i, j, X, y)
        
        return 0
    
    def _select_second_alpha(self, i, E_i, X, y):
        """选择第二个α变量"""
        # 简化版本：随机选择
        candidates = [j for j in range(len(X)) if j != i]
        if not candidates:
            return -1
        
        # 选择使|E_i - E_j|最大的j
        max_diff = 0
        best_j = -1
        
        for j in candidates:
            E_j = self._decision_function(X[j], X, y) - y[j]
            diff = abs(E_i - E_j)
            if diff > max_diff:
                max_diff = diff
                best_j = j
        
        return best_j if max_diff > 0 else random.choice(candidates)
    
    def _take_step(self, i, j, X, y):
        """优化第i和j个α变量"""
        if i == j:
            return 0
        
        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]
        y_i, y_j = y[i], y[j]
        
        # 计算边界
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        
        if L == H:
            return 0
        
        # 计算核函数值
        K_ii = self.kernel(X[i], X[i])
        K_jj = self.kernel(X[j], X[j])
        K_ij = self.kernel(X[i], X[j])
        
        # 计算η
        eta = K_ii + K_jj - 2 * K_ij
        
        if eta > 0:
            # 计算新的alpha_j
            E_i = self._decision_function(X[i], X, y) - y_i
            E_j = self._decision_function(X[j], X, y) - y_j
            
            alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
            
            # 裁剪到边界
            if alpha_j_new > H:
                alpha_j_new = H
            elif alpha_j_new < L:
                alpha_j_new = L
        else:
            return 0
        
        # 检查变化是否足够大
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return 0
        
        # 计算新的alpha_i
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
        
        # 更新α值
        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new
        
        # 更新偏置b
        self._update_bias(i, j, alpha_i_old, alpha_j_old, X, y)
        
        return 1
    
    def _update_bias(self, i, j, alpha_i_old, alpha_j_old, X, y):
        """更新偏置参数b"""
        E_i = self._decision_function(X[i], X, y) - y[i]
        E_j = self._decision_function(X[j], X, y) - y[j]
        
        b1 = (self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i], X[i])
              - y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(X[i], X[j]))
        
        b2 = (self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i], X[j])
              - y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(X[j], X[j]))
        
        if 0 < self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
    
    def _decision_function(self, x, X, y):
        """计算决策函数值"""
        result = 0
        for i in range(len(X)):
            result += self.alphas[i] * y[i] * self.kernel(X[i], x)
        return result + self.b
    
    def _extract_support_vectors(self, X, y):
        """提取支持向量"""
        self.support_vectors = []
        self.support_vector_labels = []
        self.support_vector_alphas = []
        
        for i in range(len(X)):
            if self.alphas[i] > 1e-6:  # 非零α对应支持向量
                self.support_vectors.append(X[i])
                self.support_vector_labels.append(y[i])
                self.support_vector_alphas.append(self.alphas[i])
    
    def predict(self, X):
        """预测新样本"""
        if not self.support_vectors:
            raise ValueError("模型未训练")
        
        if isinstance(X[0], (int, float)):  # 单个样本
            return self._predict_single(X)
        else:  # 多个样本
            return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x):
        """预测单个样本"""
        result = 0
        for i in range(len(self.support_vectors)):
            result += (self.support_vector_alphas[i] * 
                      self.support_vector_labels[i] * 
                      self.kernel(self.support_vectors[i], x))
        
        return 1 if result + self.b > 0 else -1
    
    def decision_function(self, X):
        """计算决策函数值（不进行符号判断）"""
        if isinstance(X[0], (int, float)):  # 单个样本
            return self._decision_function_single(X)
        else:  # 多个样本
            return [self._decision_function_single(x) for x in X]
    
    def _decision_function_single(self, x):
        """计算单个样本的决策函数值"""
        result = 0
        for i in range(len(self.support_vectors)):
            result += (self.support_vector_alphas[i] * 
                      self.support_vector_labels[i] * 
                      self.kernel(self.support_vectors[i], x))
        return result + self.b

def kernel_functions_demo():
    """核函数演示"""
    print("\n=== 核函数演示 ===")
    
    # 测试数据点
    x1 = [1, 2]
    x2 = [3, 4]
    x3 = [-1, -2]
    
    print(f"测试向量:")
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")  
    print(f"x3 = {x3}")
    
    print(f"\n核函数值计算:")
    print(f"{'核函数类型':<15} | {'K(x1,x2)':<10} | {'K(x1,x3)':<10} | {'K(x2,x3)':<10}")
    print("-" * 55)
    
    # 线性核
    k12_linear = KernelFunctions.linear(x1, x2)
    k13_linear = KernelFunctions.linear(x1, x3)
    k23_linear = KernelFunctions.linear(x2, x3)
    print(f"{'线性核':<15} | {k12_linear:<10.3f} | {k13_linear:<10.3f} | {k23_linear:<10.3f}")
    
    # 多项式核
    k12_poly = KernelFunctions.polynomial(x1, x2, degree=2)
    k13_poly = KernelFunctions.polynomial(x1, x3, degree=2)
    k23_poly = KernelFunctions.polynomial(x2, x3, degree=2)
    print(f"{'多项式核(2阶)':<15} | {k12_poly:<10.3f} | {k13_poly:<10.3f} | {k23_poly:<10.3f}")
    
    # RBF核
    k12_rbf = KernelFunctions.rbf(x1, x2, gamma=0.1)
    k13_rbf = KernelFunctions.rbf(x1, x3, gamma=0.1)
    k23_rbf = KernelFunctions.rbf(x2, x3, gamma=0.1)
    print(f"{'RBF核':<15} | {k12_rbf:<10.3f} | {k13_rbf:<10.3f} | {k23_rbf:<10.3f}")
    
    print(f"\n核函数特点:")
    print("• 线性核：计算简单，适合线性可分数据")
    print("• 多项式核：可以捕获特征间的交互作用")
    print("• RBF核：最常用，能处理复杂的非线性关系")

def linear_svm_example():
    """线性SVM示例"""
    print("\n=== 线性SVM示例 ===")
    print("问题：二维平面上的线性分类")
    
    # 生成线性可分的数据
    random.seed(42)
    X = []
    y = []
    
    # 正类样本：右上角
    for i in range(20):
        x1 = random.uniform(3, 6)
        x2 = random.uniform(3, 6)
        X.append([x1, x2])
        y.append(1)
    
    # 负类样本：左下角
    for i in range(20):
        x1 = random.uniform(0, 3)
        x2 = random.uniform(0, 3)
        X.append([x1, x2])
        y.append(-1)
    
    print(f"生成训练数据：")
    print(f"正类样本：{sum(1 for label in y if label == 1)}个")
    print(f"负类样本：{sum(1 for label in y if label == -1)}个")
    
    # 训练线性SVM
    svm = SimplifiedSMO(C=1.0, kernel='linear', max_iterations=100)
    svm.fit(X, y)
    
    # 在训练集上评估
    predictions = svm.predict(X)
    accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
    
    print(f"\n训练结果：")
    print(f"训练集准确率: {accuracy:.3f}")
    print(f"支持向量数量: {len(svm.support_vectors)}")
    
    # 显示支持向量
    print(f"\n支持向量:")
    for i, (sv, label, alpha) in enumerate(zip(svm.support_vectors, 
                                              svm.support_vector_labels, 
                                              svm.support_vector_alphas)):
        print(f"  SV{i+1}: {sv}, 标签={label:2d}, α={alpha:.4f}")
    
    # 测试新样本
    test_samples = [[1.5, 1.5], [4.5, 4.5], [2.5, 4], [4, 2.5]]
    print(f"\n新样本预测:")
    for i, sample in enumerate(test_samples):
        pred = svm.predict(sample)
        decision_value = svm.decision_function(sample)
        print(f"样本{i+1} {sample}: 预测={pred:2d}, 决策值={decision_value:.3f}")

def rbf_svm_example():
    """RBF-SVM处理非线性问题"""
    print("\n=== RBF-SVM处理非线性问题 ===")
    print("问题：同心圆分类（线性不可分）")
    
    # 生成同心圆数据
    random.seed(42)
    X = []
    y = []
    
    # 内圆：正类
    for i in range(30):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.5, 1.5)
        x1 = radius * math.cos(angle)
        x2 = radius * math.sin(angle)
        X.append([x1, x2])
        y.append(1)
    
    # 外环：负类
    for i in range(30):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(2.5, 3.5)
        x1 = radius * math.cos(angle)
        x2 = radius * math.sin(angle)
        X.append([x1, x2])
        y.append(-1)
    
    print(f"生成同心圆数据：")
    print(f"内圆(正类): {sum(1 for label in y if label == 1)}个点")
    print(f"外环(负类): {sum(1 for label in y if label == -1)}个点")
    
    # 比较线性SVM和RBF-SVM
    print(f"\n比较不同核函数的效果:")
    
    kernels = [
        ('linear', '线性核'),
        ('rbf', 'RBF核')
    ]
    
    for kernel_type, kernel_name in kernels:
        print(f"\n--- {kernel_name} ---")
        svm = SimplifiedSMO(C=1.0, kernel=kernel_type, gamma=0.1, 
                           max_iterations=50)
        svm.fit(X, y)
        
        # 评估性能
        predictions = svm.predict(X)
        accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
        
        print(f"训练集准确率: {accuracy:.3f}")
        print(f"支持向量数量: {len(svm.support_vectors)}")
        
        # 测试中心点和边界点
        test_points = [[0, 0], [1, 0], [2, 0], [3, 0]]
        print("测试点预测:")
        for point in test_points:
            pred = svm.predict(point)
            distance = math.sqrt(sum(x**2 for x in point))
            expected = 1 if distance < 2 else -1
            print(f"  点{point}(距离={distance:.1f}): 预测={pred:2d}, 期望={expected:2d}")

def parameter_tuning_demo():
    """参数调优演示"""
    print("\n=== SVM参数调优演示 ===")
    
    # 生成带噪声的数据
    random.seed(42)
    X = []
    y = []
    
    for i in range(100):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        
        # 基本分类规则：x1 + x2 > 10
        if x1 + x2 > 10:
            label = 1
        else:
            label = -1
        
        # 添加15%的噪声
        if random.random() < 0.15:
            label = -label
        
        X.append([x1, x2])
        y.append(label)
    
    print(f"生成带噪声数据：100个样本")
    print(f"正类：{sum(1 for label in y if label == 1)}个")
    print(f"负类：{sum(1 for label in y if label == -1)}个")
    
    # 测试不同的C值
    C_values = [0.1, 1.0, 10.0]
    
    print(f"\n测试不同正则化参数C的影响:")
    print(f"{'C值':<8} | {'准确率':<8} | {'支持向量数':<10} | {'说明'}")
    print("-" * 50)
    
    for C in C_values:
        svm = SimplifiedSMO(C=C, kernel='rbf', gamma=0.1, max_iterations=50)
        svm.fit(X, y)
        
        predictions = svm.predict(X)
        accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
        n_sv = len(svm.support_vectors)
        
        if C == 0.1:
            explanation = "软间隔大，可能欠拟合"
        elif C == 1.0:
            explanation = "平衡的正则化"
        else:
            explanation = "硬间隔，可能过拟合"
        
        print(f"{C:<8.1f} | {accuracy:<8.3f} | {n_sv:<10d} | {explanation}")

def svm_vs_other_algorithms():
    """SVM与其他算法对比"""
    print("\n=== SVM与其他算法对比 ===")
    
    comparison_table = [
        ("特性", "SVM", "逻辑回归", "决策树", "神经网络"),
        ("线性分类", "优秀", "优秀", "良好", "优秀"),
        ("非线性分类", "优秀(核函数)", "需特征工程", "优秀", "优秀"),
        ("高维数据", "优秀", "良好", "一般", "优秀"),
        ("训练速度", "中等", "快", "快", "慢"),
        ("预测速度", "快", "很快", "很快", "快"),
        ("内存占用", "小(支持向量)", "小", "中等", "大"),
        ("可解释性", "一般", "好", "优秀", "差"),
        ("参数调节", "中等", "简单", "中等", "复杂"),
        ("对噪声敏感性", "一般", "敏感", "敏感", "一般"),
        ("概率输出", "需要额外计算", "直接输出", "可以输出", "可以输出")
    ]
    
    # 打印对比表
    col_widths = [12, 15, 12, 12, 12]
    
    for i, row in enumerate(comparison_table):
        if i == 0:  # 表头
            print("=" * sum(col_widths))
        
        for j, cell in enumerate(row):
            print(f"{cell:<{col_widths[j]}}", end="")
        print()
        
        if i == 0:  # 表头后的分隔线
            print("=" * sum(col_widths))
    
    print("=" * sum(col_widths))

def svm_practical_tips():
    """SVM实用技巧"""
    print("\n=== SVM实用技巧 ===")
    
    print("1. 数据预处理:")
    print("   • 特征缩放：SVM对特征尺度敏感，建议标准化")
    print("   • 缺失值处理：SVM不能直接处理缺失值")
    print("   • 异常值处理：可能成为支持向量，影响决策边界")
    
    print("\n2. 核函数选择:")
    print("   • 线性核：数据线性可分或特征数远大于样本数")
    print("   • RBF核：通用选择，适合大多数问题")
    print("   • 多项式核：特征间存在已知的多项式关系")
    
    print("\n3. 参数调优策略:")
    print("   • C参数：从{0.1, 1, 10, 100}开始网格搜索")
    print("   • γ参数(RBF核)：从{0.001, 0.01, 0.1, 1}开始")
    print("   • 使用交叉验证选择最优参数组合")
    
    print("\n4. 性能优化:")
    print("   • 大数据集：考虑使用随机梯度下降版本")
    print("   • 不平衡数据：调整类权重或使用SMOTE")
    print("   • 多分类：使用一对一或一对多策略")
    
    print("\n5. 结果解释:")
    print("   • 支持向量：关键的训练样本")
    print("   • 决策值：距离超平面的距离，绝对值越大越自信")
    print("   • 间隔：衡量分类器的泛化能力")

if __name__ == "__main__":
    svm_theory()
    kernel_functions_demo()
    linear_svm_example()
    rbf_svm_example()
    parameter_tuning_demo()
    svm_vs_other_algorithms()
    svm_practical_tips()
    
    print("\n=== 总结 ===")
    print("支持向量机是强大的机器学习算法：")
    print("• 通过最大化间隔实现最优分类")
    print("• 核函数使其能处理非线性问题")
    print("• 只需存储支持向量，内存效率高")
    print("• 在高维数据上表现优异")
    print("• 理论基础扎实，泛化能力强")
    
    print("\n应用建议：")
    print("• 适合中小规模的分类问题")
    print("• 特征数量较多时优先考虑")
    print("• 需要可靠的泛化性能时使用")
    print("• 文本分类和图像识别的经典选择")