# 机器学习算法教程 - K-means聚类
# K-means Clustering: 代码实现 + 原理解释

import random
import math

def kmeans_theory():
    """
    K-means聚类原理解释
    
    K-means是最流行的无监督学习算法，用于将数据分成K个簇。
    
    核心思想：
    - 将数据点分配给最近的聚类中心
    - 更新聚类中心为该簇所有点的平均值
    - 重复上述过程直到收敛
    
    算法步骤：
    1. 随机初始化K个聚类中心
    2. 将每个点分配给最近的聚类中心
    3. 更新聚类中心为该簇所有点的平均位置
    4. 重复步骤2-3直到聚类中心不再变化
    
    优点：
    - 简单易懂，计算效率高
    - 适合球形分布的数据
    - 收敛速度快
    
    缺点：
    - 需要预先指定K值
    - 对初始化敏感
    - 假设簇是球形的
    - 对异常值敏感
    
    应用场景：
    - 客户群体分析
    - 图像分割
    - 数据压缩
    - 推荐系统
    """
    print("=== K-means聚类算法原理 ===")
    print("目标：将数据点分成K个相似的群组")
    print("方法：最小化簇内距离，最大化簇间距离")
    print("类型：无监督学习算法")
    print("应用：客户分析、图像分割、数据挖掘等")
    print()

def euclidean_distance(point1, point2):
    """计算欧几里得距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

class SimpleKMeans:
    """
    简单K-means聚类实现
    """
    
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = []
        self.labels = []
        self.history = []  # 记录中心点变化历史
        
    def fit(self, data):
        """训练K-means模型"""
        print(f"开始K-means聚类：K={self.k}, 数据点数={len(data)}")
        
        # 1. 随机初始化聚类中心
        self._initialize_centroids(data)
        print("初始化聚类中心完成")
        
        for iteration in range(self.max_iterations):
            print(f"\n第 {iteration + 1} 次迭代:")
            
            # 2. 分配点到最近的聚类中心
            old_labels = self.labels.copy() if self.labels else []
            self.labels = self._assign_points(data)
            
            # 3. 更新聚类中心
            old_centroids = [centroid[:] for centroid in self.centroids]
            self._update_centroids(data)
            
            # 记录历史
            self.history.append([centroid[:] for centroid in self.centroids])
            
            # 打印当前状态
            self._print_iteration_info(iteration + 1)
            
            # 4. 检查收敛
            if self._has_converged(old_centroids):
                print(f"算法在第 {iteration + 1} 次迭代后收敛")
                break
                
            # 检查标签是否变化
            if old_labels == self.labels:
                print("点的分配不再变化，算法收敛")
                break
        
        print("K-means聚类完成！")
        return self
    
    def _initialize_centroids(self, data):
        """随机初始化聚类中心"""
        # 找到数据的范围
        if not data:
            return
            
        dimensions = len(data[0])
        min_vals = [min(point[d] for point in data) for d in range(dimensions)]
        max_vals = [max(point[d] for point in data) for d in range(dimensions)]
        
        # 随机生成K个中心点
        self.centroids = []
        for _ in range(self.k):
            centroid = []
            for d in range(dimensions):
                val = random.uniform(min_vals[d], max_vals[d])
                centroid.append(val)
            self.centroids.append(centroid)
    
    def _assign_points(self, data):
        """将每个点分配给最近的聚类中心"""
        labels = []
        
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            labels.append(closest_centroid)
        
        return labels
    
    def _update_centroids(self, data):
        """更新聚类中心为该簇所有点的平均值"""
        dimensions = len(data[0])
        
        for k in range(self.k):
            # 找到属于第k个簇的所有点
            cluster_points = [data[i] for i in range(len(data)) if self.labels[i] == k]
            
            if cluster_points:  # 如果该簇不为空
                # 计算平均值作为新的中心点
                new_centroid = []
                for d in range(dimensions):
                    avg = sum(point[d] for point in cluster_points) / len(cluster_points)
                    new_centroid.append(avg)
                self.centroids[k] = new_centroid
    
    def _has_converged(self, old_centroids):
        """检查是否收敛"""
        if not old_centroids:
            return False
            
        for old_centroid, new_centroid in zip(old_centroids, self.centroids):
            distance = euclidean_distance(old_centroid, new_centroid)
            if distance > self.tolerance:
                return False
        return True
    
    def _print_iteration_info(self, iteration):
        """打印迭代信息"""
        print(f"聚类中心:")
        for i, centroid in enumerate(self.centroids):
            coords = ", ".join([f"{x:.2f}" for x in centroid])
            print(f"  簇 {i}: ({coords})")
        
        # 统计每个簇的点数
        cluster_counts = [0] * self.k
        for label in self.labels:
            cluster_counts[label] += 1
        
        print(f"各簇点数: {cluster_counts}")
    
    def predict(self, data):
        """预测新数据点的簇标签"""
        if isinstance(data[0], (int, float)):  # 单个点
            distances = [euclidean_distance(data, centroid) for centroid in self.centroids]
            return distances.index(min(distances))
        else:  # 多个点
            return [self.predict(point) for point in data]
    
    def get_cluster_info(self, data):
        """获取聚类信息"""
        info = {}
        
        for k in range(self.k):
            cluster_points = [data[i] for i in range(len(data)) if self.labels[i] == k]
            
            if cluster_points:
                # 计算簇内平均距离
                total_distance = 0
                count = 0
                for i, point1 in enumerate(cluster_points):
                    for j, point2 in enumerate(cluster_points[i+1:], i+1):
                        total_distance += euclidean_distance(point1, point2)
                        count += 1
                
                avg_distance = total_distance / count if count > 0 else 0
                
                info[k] = {
                    "center": self.centroids[k],
                    "size": len(cluster_points),
                    "avg_distance": avg_distance
                }
        
        return info

def distance_demo():
    """距离计算演示"""
    print("\n=== 距离计算演示 ===")
    
    points = [
        ([1, 2], "点A"),
        ([4, 6], "点B"),
        ([2, 3], "点C")
    ]
    
    print("示例点坐标:")
    for point, name in points:
        print(f"{name}: ({point[0]}, {point[1]})")
    
    print(f"\n距离计算（欧几里得距离）:")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            point1, name1 = points[i]
            point2, name2 = points[j]
            dist = euclidean_distance(point1, point2)
            print(f"{name1}到{name2}: {dist:.2f}")
    
    print(f"\n距离公式: d = √[(x₁-x₂)² + (y₁-y₂)²]")

def practical_example():
    """实际应用示例：客户群体分析"""
    print("\n=== 实际应用：客户群体分析 ===")
    
    # 模拟客户数据：[年收入(万元), 年消费(万元)]
    print("客户数据（年收入，年消费）:")
    
    customers = [
        [30, 15], [35, 18], [40, 20], [45, 22],  # 中等收入群体
        [60, 35], [65, 40], [70, 38], [75, 45],  # 高收入群体  
        [15, 8], [20, 10], [25, 12], [18, 9],   # 低收入群体
        [55, 25], [50, 28], [48, 24]            # 混合群体
    ]
    
    print("收入(万) | 消费(万)")
    print("-" * 20)
    for i, (income, spending) in enumerate(customers):
        print(f"客户{i+1:2d}: {income:2d}     | {spending:2d}")
    
    # 设置随机种子以获得可重现结果
    random.seed(42)
    
    # 进行K-means聚类
    kmeans = SimpleKMeans(k=3, max_iterations=10)
    kmeans.fit(customers)
    
    # 分析聚类结果
    print(f"\n聚类结果分析:")
    cluster_info = kmeans.get_cluster_info(customers)
    
    for cluster_id, info in cluster_info.items():
        center = info["center"]
        size = info["size"]
        print(f"\n群体 {cluster_id + 1}:")
        print(f"  中心点: 收入{center[0]:.1f}万, 消费{center[1]:.1f}万")
        print(f"  客户数: {size}人")
        
        # 显示该群体的客户
        cluster_customers = []
        for i, label in enumerate(kmeans.labels):
            if label == cluster_id:
                cluster_customers.append(f"客户{i+1}")
        print(f"  包含客户: {', '.join(cluster_customers)}")
    
    # 预测新客户
    new_customers = [[52, 30], [25, 14], [80, 50]]
    print(f"\n新客户分类预测:")
    
    for i, customer in enumerate(new_customers):
        cluster_id = kmeans.predict(customer)
        print(f"新客户{i+1}(收入{customer[0]}万,消费{customer[1]}万) -> 群体{cluster_id + 1}")

def elbow_method_demo():
    """肘部法则演示（选择最优K值）"""
    print("\n=== 肘部法则：选择最优K值 ===")
    
    # 生成测试数据
    test_data = [
        [2, 2], [3, 3], [2, 3], [3, 2],      # 群体1
        [8, 8], [9, 9], [8, 9], [9, 8],      # 群体2
        [15, 2], [16, 3], [15, 3], [16, 2]   # 群体3
    ]
    
    print("测试不同的K值:")
    print("K值 | 总簇内距离平方和(WCSS)")
    print("-" * 25)
    
    for k in range(1, 6):
        random.seed(42)  # 固定随机种子
        kmeans = SimpleKMeans(k=k, max_iterations=50)
        kmeans.fit(test_data)
        
        # 计算总簇内距离平方和
        wcss = 0
        for i, point in enumerate(test_data):
            cluster_center = kmeans.centroids[kmeans.labels[i]]
            distance = euclidean_distance(point, cluster_center)
            wcss += distance ** 2
        
        print(f"{k}   | {wcss:.2f}")
    
    print(f"\n肘部法则说明:")
    print("- WCSS随K值增加而减少")
    print("- 寻找WCSS下降幅度明显变缓的拐点")
    print("- 该拐点对应的K值通常是最优选择")
    print("- 在这个例子中，K=3可能是最优的")

def kmeans_limitations():
    """K-means算法限制演示"""
    print("\n=== K-means算法的限制 ===")
    
    print("1. 对初始化敏感:")
    print("   不同的初始中心点可能导致不同的结果")
    
    print("\n2. 假设簇是球形的:")
    print("   对于非球形分布的数据效果不好")
    
    print("\n3. 需要预先指定K值:")
    print("   实际应用中K值通常未知")
    
    print("\n4. 对异常值敏感:")
    print("   极端值会影响聚类中心的位置")
    
    print("\n改进方法:")
    print("- K-means++: 改进初始化方法")
    print("- Mini-batch K-means: 处理大数据集")
    print("- DBSCAN: 处理任意形状的簇")
    print("- 层次聚类: 不需要预设K值")

def algorithm_comparison():
    """聚类算法对比"""
    print("\n=== 聚类算法对比 ===")
    
    algorithms = {
        "特性": ["计算复杂度", "数据形状要求", "异常值敏感性", "参数设置", "结果稳定性"],
        "K-means": ["O(n*k*i)", "球形簇", "敏感", "需要设置K", "依赖初始化"],
        "层次聚类": ["O(n³)", "任意形状", "较敏感", "设置距离阈值", "稳定"],
        "DBSCAN": ["O(n log n)", "任意形状", "鲁棒", "设置半径和最小点数", "稳定"]
    }
    
    print(f"{'特性':<12} | {'K-means':<15} | {'层次聚类':<15} | {'DBSCAN':<15}")
    print("-" * 70)
    
    for i, feature in enumerate(algorithms["特性"]):
        kmeans_val = algorithms["K-means"][i]
        hierarchical_val = algorithms["层次聚类"][i] 
        dbscan_val = algorithms["DBSCAN"][i]
        print(f"{feature:<12} | {kmeans_val:<15} | {hierarchical_val:<15} | {dbscan_val:<15}")

if __name__ == "__main__":
    kmeans_theory()
    distance_demo()
    practical_example()
    elbow_method_demo()
    kmeans_limitations()
    algorithm_comparison()
    
    print("\n=== 总结 ===")
    print("K-means聚类算法要点:")
    print("- 无监督学习的经典算法")
    print("- 通过迭代优化聚类中心")
    print("- 适合球形分布的数据")
    print("- 需要合理选择K值")
    print("- 计算效率高，广泛应用")
    
    print("\n实际应用建议:")
    print("- 使用肘部法则选择K值")
    print("- 多次运行取最佳结果")
    print("- 考虑数据标准化")
    print("- 结合业务知识解释结果")