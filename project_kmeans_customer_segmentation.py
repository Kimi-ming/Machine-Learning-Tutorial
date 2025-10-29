"""
真实项目案例：电商客户分群系统
使用K-Means聚类实现精准营销

项目背景：
电商平台拥有大量客户，需要对客户进行智能分群，
实现差异化营销策略，提升转化率和客户价值。

数据说明（RFM模型）：
- R (Recency): 最近一次购买距今天数
- F (Frequency): 购买频率（总订单数）
- M (Monetary): 消费金额（总金额）

客户分群：
- VIP客户：高频高额消费
- 潜力客户：中频中额，有增长空间
- 沉睡客户：长时间未购买
- 新客户：首次或少量购买

学习目标：
1. 理解无监督学习的应用
2. 学会使用肘部法则选择K值
3. 掌握客户画像分析方法
4. 了解RFM模型在营销中的应用
"""

import random
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CustomerSegmentation:
    """
    客户分群系统
    基于K-Means聚类算法
    """

    def __init__(self, n_clusters=4, max_iterations=100):
        """
        初始化
        n_clusters: 聚类数量（客户群数）
        max_iterations: 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None  # 聚类中心
        self.labels = None  # 每个样本的类别
        self.feature_names = ['R', 'F', 'M']
        self.inertia_history = []  # 记录每次迭代的总距离

    def fit(self, X):
        """
        训练模型（聚类）
        X: 客户特征矩阵 [[R, F, M], ...]
        """
        if HAS_NUMPY:
            X = np.array(X)
            n_samples, n_features = X.shape

            print(f"\n开始客户分群...")
            print(f"客户数量: {n_samples}")
            print(f"分群数量: {self.n_clusters}")
            print("-" * 60)

            # 随机初始化聚类中心
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_indices]

            # K-Means迭代
            for iteration in range(self.max_iterations):
                # 分配样本到最近的聚类中心
                distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
                self.labels = np.argmin(distances, axis=1)

                # 计算总距离（inertia）
                inertia = sum(np.min(distances, axis=1) ** 2)
                self.inertia_history.append(inertia)

                # 更新聚类中心
                new_centroids = np.array([X[self.labels == k].mean(axis=0)
                                         for k in range(self.n_clusters)])

                # 检查收敛
                if np.allclose(self.centroids, new_centroids):
                    print(f"算法在第 {iteration} 次迭代后收敛")
                    break

                self.centroids = new_centroids

                if iteration % 10 == 0:
                    print(f"迭代 {iteration}: 总距离={inertia:.2f}")

            print("-" * 60)
            print("分群完成！")
            self._print_cluster_summary(X)

        else:
            # 纯Python实现
            n_samples = len(X)

            # 初始化聚类中心
            random_indices = random.sample(range(n_samples), self.n_clusters)
            self.centroids = [X[i] for i in random_indices]
            self.labels = [0] * n_samples

            print(f"\n开始分群（纯Python模式）...")

            for iteration in range(self.max_iterations):
                # 分配样本
                for i, point in enumerate(X):
                    distances = [self._euclidean_distance(point, centroid)
                               for centroid in self.centroids]
                    self.labels[i] = distances.index(min(distances))

                # 更新中心
                new_centroids = []
                for k in range(self.n_clusters):
                    cluster_points = [X[i] for i in range(n_samples) if self.labels[i] == k]
                    if cluster_points:
                        n_features = len(cluster_points[0])
                        new_center = [sum(p[j] for p in cluster_points) / len(cluster_points)
                                    for j in range(n_features)]
                        new_centroids.append(new_center)
                    else:
                        new_centroids.append(self.centroids[k])

                # 检查收敛
                if self._centroids_equal(self.centroids, new_centroids):
                    print(f"算法在第 {iteration} 次迭代后收敛")
                    break

                self.centroids = new_centroids

            print("分群完成！")

    def _euclidean_distance(self, p1, p2):
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _centroids_equal(self, c1, c2, tol=1e-4):
        """检查聚类中心是否相等"""
        for cent1, cent2 in zip(c1, c2):
            if self._euclidean_distance(cent1, cent2) > tol:
                return False
        return True

    def predict(self, X):
        """
        预测新客户属于哪个群体
        """
        if isinstance(X[0], (int, float)):
            X = [X]

        if HAS_NUMPY:
            X = np.array(X)
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
            return np.argmin(distances, axis=1).tolist()
        else:
            labels = []
            for point in X:
                distances = [self._euclidean_distance(point, centroid)
                           for centroid in self.centroids]
                labels.append(distances.index(min(distances)))
            return labels[0] if len(labels) == 1 else labels

    def _print_cluster_summary(self, X):
        """打印聚类摘要"""
        print("\n客户群体分析：")
        print("=" * 80)

        if HAS_NUMPY:
            for k in range(self.n_clusters):
                cluster_data = X[self.labels == k]
                n_customers = len(cluster_data)

                if n_customers > 0:
                    avg_r = cluster_data[:, 0].mean()
                    avg_f = cluster_data[:, 1].mean()
                    avg_m = cluster_data[:, 2].mean()

                    print(f"\n群体 {k+1}: {n_customers} 位客户")
                    print(f"  平均最近购买: {avg_r:.1f} 天前")
                    print(f"  平均购买频率: {avg_f:.1f} 次")
                    print(f"  平均消费金额: {avg_m:.0f} 元")
                    print(f"  客户占比: {n_customers/len(X)*100:.1f}%")

                    # 标签建议
                    label = self._suggest_label(avg_r, avg_f, avg_m)
                    print(f"  建议标签: {label}")
        print("=" * 80)

    def _suggest_label(self, r, f, m):
        """根据RFM值建议客户标签"""
        if f > 20 and m > 5000:
            return "💎 VIP客户 - 高价值"
        elif r < 30 and f > 10:
            return "⭐ 活跃客户 - 高频购买"
        elif r > 90:
            return "😴 沉睡客户 - 需唤醒"
        elif f < 5 and m < 1000:
            return "🌱 新客户 - 待培养"
        else:
            return "📈 潜力客户 - 可提升"


def generate_customer_data(n_customers=500):
    """
    生成模拟的客户RFM数据

    客户类型分布：
    - VIP (10%): 近期购买, 高频, 高额
    - 活跃 (30%): 近期购买, 中高频, 中额
    - 沉睡 (20%): 长期未购买, 低频, 低额
    - 新客户 (40%): 近期购买, 低频, 低额
    """
    print(f"生成 {n_customers} 位客户的RFM数据...")

    data = []

    # VIP客户 (10%)
    for _ in range(int(n_customers * 0.1)):
        r = random.uniform(1, 30)  # 最近购买
        f = random.uniform(20, 50)  # 高频
        m = random.uniform(5000, 20000)  # 高额
        data.append([r, f, m])

    # 活跃客户 (30%)
    for _ in range(int(n_customers * 0.3)):
        r = random.uniform(1, 60)
        f = random.uniform(10, 25)
        m = random.uniform(2000, 8000)
        data.append([r, f, m])

    # 沉睡客户 (20%)
    for _ in range(int(n_customers * 0.2)):
        r = random.uniform(90, 365)  # 长期未购买
        f = random.uniform(1, 10)  # 低频
        m = random.uniform(100, 2000)
        data.append([r, f, m])

    # 新客户 (40%)
    for _ in range(int(n_customers * 0.4)):
        r = random.uniform(1, 60)
        f = random.uniform(1, 5)  # 少量购买
        m = random.uniform(100, 1500)
        data.append([r, f, m])

    random.shuffle(data)
    return data


def elbow_method(X, max_k=10):
    """
    肘部法则：寻找最优K值
    """
    print("\n" + "=" * 60)
    print("肘部法则：寻找最优聚类数")
    print("=" * 60)

    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        model = CustomerSegmentation(n_clusters=k, max_iterations=50)
        model.fit(X)
        inertia = model.inertia_history[-1] if model.inertia_history else 0
        inertias.append(inertia)
        print(f"K={k}: 总距离={inertia:.2f}")

    # 简单的肘点检测
    print("\n建议分析:")
    print("选择总距离下降趋势变缓的K值（肘点）")
    print("对于客户分群，通常选择 K=3-5 比较合适")


def demo_customer_segmentation():
    """完整的客户分群演示"""
    print("\n" + "=" * 60)
    print("项目演示：电商客户智能分群系统")
    print("=" * 60)

    # 1. 生成数据
    data = generate_customer_data(n_customers=500)

    # 2. 肘部法则（可选）
    choice = input("\n是否运行肘部法则寻找最优K值？(y/n): ").strip().lower()
    if choice == 'y':
        elbow_method(data, max_k=8)

    # 3. 执行分群
    n_clusters = int(input(f"\n请输入聚类数量（推荐4）: ") or "4")
    model = CustomerSegmentation(n_clusters=n_clusters, max_iterations=100)
    model.fit(data)

    # 4. 为每个群体制定营销策略
    print("\n" + "=" * 60)
    print("精准营销策略建议")
    print("=" * 60)

    if HAS_NUMPY:
        data_array = np.array(data)
        for k in range(n_clusters):
            cluster_data = data_array[model.labels == k]
            if len(cluster_data) > 0:
                avg_r, avg_f, avg_m = cluster_data.mean(axis=0)

                print(f"\n【群体 {k+1}】")
                print(f"规模: {len(cluster_data)} 人")

                # 根据特征推荐策略
                if avg_m > 5000 and avg_f > 15:
                    print("类型: VIP客户")
                    print("策略: 专属优惠、积分加倍、新品优先体验")
                elif avg_r < 30:
                    print("类型: 活跃客户")
                    print("策略: 推荐相关商品、限时折扣、组合优惠")
                elif avg_r > 90:
                    print("类型: 沉睡客户")
                    print("策略: 发送唤醒邮件、大额优惠券、回馈活动")
                else:
                    print("类型: 普通客户")
                    print("策略: 提升频率、增加客单价、会员激励")

    # 5. 预测新客户
    print("\n" + "=" * 60)
    print("新客户分群预测")
    print("=" * 60)

    test_customers = [
        {"描述": "VIP大客户", "rfm": [5, 35, 15000]},
        {"描述": "沉睡老客户", "rfm": [180, 3, 500]},
        {"描述": "活跃中等客户", "rfm": [20, 15, 3000]},
        {"描述": "新注册客户", "rfm": [10, 1, 200]}
    ]

    for customer in test_customers:
        label = model.predict(customer["rfm"])
        print(f"\n{customer['描述']}")
        print(f"  RFM: R={customer['rfm'][0]:.0f}, F={customer['rfm'][1]:.0f}, M={customer['rfm'][2]:.0f}")
        print(f"  所属群体: 群体 {label + 1}")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        机器学习实战项目：客户智能分群系统              ║
║        基于K-Means聚类的精准营销                       ║
╚══════════════════════════════════════════════════════════╝

RFM模型说明：
- R (Recency): 最近购买时间 - 越小越好
- F (Frequency): 购买频率 - 越大越好
- M (Monetary): 消费金额 - 越大越好
    """)

    demo_customer_segmentation()

    print("\n" + "=" * 60)
    print("感谢使用客户分群系统！")
    print("=" * 60)


if __name__ == "__main__":
    main()
