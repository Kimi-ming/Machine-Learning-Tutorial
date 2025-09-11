# 机器学习数据可视化工具
# Data Visualization Tools: 纯Python实现的图表绘制

import math
import random
from collections import Counter

def visualization_theory():
    """
    数据可视化理论
    
    可视化的重要性：
    - 快速理解数据分布和模式
    - 发现异常值和离群点
    - 验证数据质量和完整性
    - 展示模型性能和结果
    - 辅助特征工程和模型选择
    
    常用图表类型：
    1. 散点图：展示两个变量间的关系
    2. 直方图：显示数据分布
    3. 线图：展示趋势变化
    4. 柱状图：比较不同类别的数值
    5. 箱线图：显示数据的五数概括
    6. 热力图：展示相关性矩阵
    
    可视化原则：
    - 选择合适的图表类型
    - 保持简洁清晰
    - 突出重要信息
    - 使用恰当的颜色和标签
    """
    print("=== 数据可视化理论 ===")
    print("目标：通过图形直观展示数据模式和分析结果")
    print("原则：简洁、准确、美观、有效")
    print("工具：纯Python实现（ASCII艺术）+ matplotlib支持")
    print()

class ASCIIPlotter:
    """纯Python实现的ASCII图表绘制器"""
    
    @staticmethod
    def scatter_plot(x_data, y_data, labels=None, width=50, height=20, title="散点图"):
        """
        绘制ASCII散点图
        """
        print(f"\n=== {title} ===")
        
        if not x_data or not y_data or len(x_data) != len(y_data):
            print("数据格式错误")
            return
        
        # 计算数据范围
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        # 避免除零错误
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        print(f"X轴范围: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y轴范围: [{y_min:.2f}, {y_max:.2f}]")
        print()
        
        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制数据点
        symbols = ['*', '+', 'o', 'x', '#'] if labels else ['*']
        label_symbols = {}
        
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            # 将数据点映射到画布坐标
            col = int((x - x_min) / x_range * (width - 1))
            row = int((y_max - y) / y_range * (height - 1))  # Y轴翻转
            
            # 确保坐标在画布范围内
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            
            # 选择符号
            if labels:
                label = labels[i]
                if label not in label_symbols:
                    label_symbols[label] = symbols[len(label_symbols) % len(symbols)]
                symbol = label_symbols[label]
            else:
                symbol = '*'
            
            canvas[row][col] = symbol
        
        # 输出图表
        # Y轴标签和图表
        y_step = y_range / (height - 1)
        for i, row in enumerate(canvas):
            y_val = y_max - i * y_step
            print(f"{y_val:6.1f} |{''.join(row)}|")
        
        # X轴
        print(" " * 7 + "-" * width)
        
        # X轴标签
        x_labels = []
        for i in range(0, width, width//5):
            x_val = x_min + (i / (width - 1)) * x_range
            x_labels.append(f"{x_val:6.1f}")
        
        label_line = " " * 7
        for i, label in enumerate(x_labels):
            pos = i * (width // 5)
            if pos < width:
                label_line += label.rjust(width//5 if i > 0 else 0)
        print(label_line)
        
        # 图例
        if labels and label_symbols:
            print("\n图例:")
            for label, symbol in label_symbols.items():
                print(f"  {symbol} = {label}")
    
    @staticmethod
    def histogram(data, bins=10, width=50, title="直方图"):
        """
        绘制ASCII直方图
        """
        print(f"\n=== {title} ===")
        
        if not data:
            print("数据为空")
            return
        
        # 计算分箱
        data_min, data_max = min(data), max(data)
        data_range = data_max - data_min if data_max != data_min else 1
        bin_width = data_range / bins
        
        # 统计每个分箱的频数
        bin_counts = [0] * bins
        for value in data:
            bin_index = int((value - data_min) / bin_width)
            bin_index = min(bin_index, bins - 1)  # 处理边界情况
            bin_counts[bin_index] += 1
        
        # 计算显示参数
        max_count = max(bin_counts) if bin_counts else 1
        
        print(f"数据范围: [{data_min:.2f}, {data_max:.2f}]")
        print(f"总样本数: {len(data)}")
        print(f"分箱数: {bins}")
        print()
        
        # 绘制直方图
        for i in range(bins):
            bin_start = data_min + i * bin_width
            bin_end = bin_start + bin_width
            count = bin_counts[i]
            
            # 计算条形长度
            bar_length = int((count / max_count) * width) if max_count > 0 else 0
            bar = '█' * bar_length
            
            print(f"[{bin_start:5.1f}-{bin_end:5.1f}) |{bar:<{width}}| {count}")
    
    @staticmethod
    def line_plot(x_data, y_data, width=60, height=15, title="线图"):
        """
        绘制ASCII线图
        """
        print(f"\n=== {title} ===")
        
        if not x_data or not y_data or len(x_data) != len(y_data):
            print("数据格式错误")
            return
        
        # 对数据按x排序
        sorted_data = sorted(zip(x_data, y_data))
        x_sorted = [x for x, y in sorted_data]
        y_sorted = [y for x, y in sorted_data]
        
        x_min, x_max = min(x_sorted), max(x_sorted)
        y_min, y_max = min(y_sorted), max(y_sorted)
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        print(f"X轴范围: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y轴范围: [{y_min:.2f}, {y_max:.2f}]")
        print()
        
        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制数据点和连线
        prev_col, prev_row = None, None
        
        for x, y in zip(x_sorted, y_sorted):
            col = int((x - x_min) / x_range * (width - 1))
            row = int((y_max - y) / y_range * (height - 1))
            
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            
            canvas[row][col] = '*'
            
            # 简单的连线（水平和垂直）
            if prev_col is not None and prev_row is not None:
                # 连接前一个点
                if abs(col - prev_col) > 1 or abs(row - prev_row) > 1:
                    # 简单的直线插值
                    steps = max(abs(col - prev_col), abs(row - prev_row))
                    for step in range(1, steps):
                        inter_col = prev_col + (col - prev_col) * step // steps
                        inter_row = prev_row + (row - prev_row) * step // steps
                        if 0 <= inter_col < width and 0 <= inter_row < height:
                            if canvas[inter_row][inter_col] == ' ':
                                canvas[inter_row][inter_col] = '-' if inter_row == prev_row else '|'
            
            prev_col, prev_row = col, row
        
        # 输出图表
        y_step = y_range / (height - 1)
        for i, row in enumerate(canvas):
            y_val = y_max - i * y_step
            print(f"{y_val:6.1f} |{''.join(row)}|")
        
        print(" " * 7 + "-" * width)
    
    @staticmethod
    def bar_chart(categories, values, width=50, title="柱状图"):
        """
        绘制ASCII柱状图
        """
        print(f"\n=== {title} ===")
        
        if not categories or not values or len(categories) != len(values):
            print("数据格式错误")
            return
        
        max_value = max(values) if values else 1
        
        print(f"类别数: {len(categories)}")
        print(f"最大值: {max_value}")
        print()
        
        for i, (category, value) in enumerate(zip(categories, values)):
            # 计算条形长度
            bar_length = int((value / max_value) * width) if max_value > 0 else 0
            bar = '█' * bar_length
            
            # 截断过长的类别名
            cat_name = category[:10] if len(category) > 10 else category
            
            print(f"{cat_name:>10} |{bar:<{width}}| {value}")

class StatisticalVisualizer:
    """统计图表绘制器"""
    
    @staticmethod
    def confusion_matrix_plot(matrix, labels, title="混淆矩阵"):
        """
        可视化混淆矩阵
        """
        print(f"\n=== {title} ===")
        
        if not matrix or not labels:
            print("数据为空")
            return
        
        n = len(labels)
        
        # 计算总数用于百分比计算
        total = sum(sum(row) for row in matrix)
        
        # 打印表头
        print("实际\\预测", end="")
        for label in labels:
            print(f"{str(label):>8}", end="")
        print(f"{'总计':>8}")
        
        print("-" * (10 + 8 * (n + 1)))
        
        # 打印每一行
        for i, label in enumerate(labels):
            print(f"{str(label):>8}", end="")
            row_sum = sum(matrix[i])
            
            for j in range(n):
                value = matrix[i][j]
                print(f"{value:>8}", end="")
            
            print(f"{row_sum:>8}")
        
        # 打印列总计
        print("总计".rjust(10), end="")
        for j in range(n):
            col_sum = sum(matrix[i][j] for i in range(n))
            print(f"{col_sum:>8}", end="")
        print(f"{total:>8}")
        
        # 计算准确率
        correct = sum(matrix[i][i] for i in range(n))
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n总准确率: {accuracy:.3f} ({correct}/{total})")
    
    @staticmethod
    def correlation_matrix_plot(data, feature_names, title="相关系数矩阵"):
        """
        可视化相关系数矩阵
        """
        print(f"\n=== {title} ===")
        
        if not data or not feature_names:
            print("数据为空")
            return
        
        n_features = len(feature_names)
        n_samples = len(data)
        
        # 计算相关系数矩阵
        correlation_matrix = []
        
        for i in range(n_features):
            row = []
            for j in range(n_features):
                if i == j:
                    row.append(1.0)
                else:
                    # 计算皮尔逊相关系数
                    x = [sample[i] for sample in data]
                    y = [sample[j] for sample in data]
                    
                    x_mean = sum(x) / len(x)
                    y_mean = sum(y) / len(y)
                    
                    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                    
                    x_var = sum((xi - x_mean) ** 2 for xi in x)
                    y_var = sum((yi - y_mean) ** 2 for yi in y)
                    
                    denominator = math.sqrt(x_var * y_var)
                    
                    correlation = numerator / denominator if denominator != 0 else 0
                    row.append(correlation)
            
            correlation_matrix.append(row)
        
        # 打印相关系数矩阵
        print(" " * 8, end="")
        for name in feature_names:
            print(f"{name[:6]:>8}", end="")
        print()
        
        print("-" * (8 + 8 * n_features))
        
        for i, name in enumerate(feature_names):
            print(f"{name[:6]:>6}", end="  ")
            for j in range(n_features):
                corr = correlation_matrix[i][j]
                
                # 使用符号表示相关强度
                if abs(corr) >= 0.7:
                    symbol = "██"
                elif abs(corr) >= 0.5:
                    symbol = "▓▓"
                elif abs(corr) >= 0.3:
                    symbol = "▒▒"
                elif abs(corr) >= 0.1:
                    symbol = "░░"
                else:
                    symbol = "  "
                
                # 负相关用不同符号
                if corr < 0:
                    symbol = symbol.replace("█", "▀").replace("▓", "▄").replace("▒", "▀").replace("░", "▄")
                
                print(f"{corr:5.2f}{symbol[0]}", end=" ")
            print()
        
        print("\n图例: ██(>0.7) ▓▓(>0.5) ▒▒(>0.3) ░░(>0.1)   (负相关用▀▄表示)")

class ModelVisualizationTools:
    """模型可视化工具"""
    
    @staticmethod
    def plot_learning_curve(train_scores, val_scores, title="学习曲线"):
        """
        绘制学习曲线
        """
        print(f"\n=== {title} ===")
        
        if not train_scores or not val_scores:
            print("数据为空")
            return
        
        epochs = list(range(1, len(train_scores) + 1))
        
        # 显示数值表格
        print("轮次  训练集得分  验证集得分  差值")
        print("-" * 35)
        
        for i, (train_score, val_score) in enumerate(zip(train_scores, val_scores)):
            diff = abs(train_score - val_score)
            print(f"{i+1:4d}  {train_score:9.4f}  {val_score:9.4f}  {diff:.4f}")
        
        # 使用线图绘制
        print(f"\n训练集得分趋势:")
        ASCIIPlotter.line_plot(epochs, train_scores, title="训练集得分")
        
        print(f"\n验证集得分趋势:")
        ASCIIPlotter.line_plot(epochs, val_scores, title="验证集得分")
        
        # 分析趋势
        if len(train_scores) > 1:
            train_trend = "上升" if train_scores[-1] > train_scores[0] else "下降"
            val_trend = "上升" if val_scores[-1] > val_scores[0] else "下降"
            
            final_diff = abs(train_scores[-1] - val_scores[-1])
            
            print(f"\n趋势分析:")
            print(f"训练集得分: {train_trend}")
            print(f"验证集得分: {val_trend}")
            print(f"最终差距: {final_diff:.4f}")
            
            if final_diff > 0.1:
                print("警告: 训练集和验证集得分差距较大，可能存在过拟合")
            elif train_scores[-1] < 0.6:
                print("提示: 整体得分较低，可能存在欠拟合")
    
    @staticmethod
    def plot_decision_boundary_1d(model, X_train, y_train, x_range=None, title="一维决策边界"):
        """
        可视化一维决策边界
        """
        print(f"\n=== {title} ===")
        
        if not X_train or not y_train:
            print("数据为空")
            return
        
        # 确定x范围
        if x_range is None:
            x_min = min(x[0] if isinstance(x, list) else x for x in X_train)
            x_max = max(x[0] if isinstance(x, list) else x for x in X_train)
            margin = (x_max - x_min) * 0.1
            x_range = (x_min - margin, x_max + margin)
        
        # 生成测试点
        n_points = 50
        test_x = []
        step = (x_range[1] - x_range[0]) / (n_points - 1)
        for i in range(n_points):
            x = x_range[0] + i * step
            test_x.append([x] if hasattr(model, 'predict') else x)
        
        # 预测
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(test_x)
            else:
                predictions = [model(x) for x in test_x]
        except Exception as e:
            print(f"预测失败: {e}")
            return
        
        # 绘制结果
        test_x_vals = [x[0] if isinstance(x, list) else x for x in test_x]
        ASCIIPlotter.scatter_plot(test_x_vals, predictions, title="决策边界")
        
        # 叠加训练数据
        train_x_vals = [x[0] if isinstance(x, list) else x for x in X_train]
        print(f"\n训练数据分布:")
        unique_labels = sorted(set(y_train))
        colors = ['*', '+', 'o', '#', '@']
        
        for i, label in enumerate(unique_labels):
            x_vals = [train_x_vals[j] for j, y in enumerate(y_train) if y == label]
            y_vals = [label] * len(x_vals)
            symbol = colors[i % len(colors)]
            
            print(f"类别 {label}: {len(x_vals)}个样本 (符号: {symbol})")
            for x_val in x_vals[:5]:  # 只显示前5个
                print(f"  x = {x_val:.3f}")
            if len(x_vals) > 5:
                print(f"  ... 还有{len(x_vals)-5}个")

def comprehensive_visualization_demo():
    """综合可视化演示"""
    print("\n=== 综合可视化演示 ===")
    
    # 生成示例数据
    random.seed(42)
    
    # 1. 二维分类数据
    print("1. 生成二维分类数据")
    X_class = []
    y_class = []
    
    for i in range(100):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        
        if x1 + x2 > 0:
            label = 1
            x1 += 1.5
            x2 += 1.5
        else:
            label = 0
            x1 -= 1.5
            x2 -= 1.5
        
        X_class.append([x1, x2])
        y_class.append(label)
    
    # 绘制散点图
    x1_vals = [x[0] for x in X_class]
    x2_vals = [x[1] for x in X_class]
    ASCIIPlotter.scatter_plot(x1_vals, x2_vals, y_class, title="二维分类数据")
    
    # 2. 回归数据
    print("\n2. 生成回归数据")
    X_reg = []
    y_reg = []
    
    for i in range(50):
        x = random.uniform(0, 10)
        y = 2 * x + 1 + random.gauss(0, 2)
        X_reg.append(x)
        y_reg.append(y)
    
    ASCIIPlotter.scatter_plot(X_reg, y_reg, title="回归数据散点图")
    
    # 3. 直方图
    print("\n3. 数据分布直方图")
    normal_data = [random.gauss(0, 1) for _ in range(200)]
    ASCIIPlotter.histogram(normal_data, bins=15, title="正态分布直方图")
    
    # 4. 学习曲线
    print("\n4. 模拟学习曲线")
    train_scores = []
    val_scores = []
    
    for epoch in range(20):
        # 模拟训练过程：训练得分逐渐提升，验证得分先升后降
        train_score = 0.3 + 0.6 * (1 - math.exp(-epoch * 0.3))
        val_score = 0.3 + 0.5 * (1 - math.exp(-epoch * 0.2)) - 0.1 * max(0, epoch - 10) * 0.02
        
        # 添加噪声
        train_score += random.gauss(0, 0.02)
        val_score += random.gauss(0, 0.03)
        
        train_scores.append(max(0, min(1, train_score)))
        val_scores.append(max(0, min(1, val_score)))
    
    ModelVisualizationTools.plot_learning_curve(train_scores, val_scores)
    
    # 5. 混淆矩阵
    print("\n5. 混淆矩阵可视化")
    confusion_matrix = [
        [45, 5, 2],
        [3, 38, 4], 
        [1, 6, 41]
    ]
    labels = ['类别A', '类别B', '类别C']
    
    StatisticalVisualizer.confusion_matrix_plot(confusion_matrix, labels)
    
    # 6. 相关系数矩阵
    print("\n6. 特征相关性分析")
    # 生成相关数据
    correlation_data = []
    for i in range(50):
        x1 = random.gauss(0, 1)
        x2 = 0.8 * x1 + random.gauss(0, 0.6)  # 与x1高度相关
        x3 = -0.5 * x1 + random.gauss(0, 0.8)  # 与x1负相关
        x4 = random.gauss(0, 1)  # 独立特征
        
        correlation_data.append([x1, x2, x3, x4])
    
    feature_names = ['特征1', '特征2', '特征3', '特征4']
    StatisticalVisualizer.correlation_matrix_plot(correlation_data, feature_names)

def visualization_best_practices():
    """可视化最佳实践"""
    print("\n=== 数据可视化最佳实践 ===")
    
    print("1. 选择合适的图表类型:")
    print("   • 散点图: 显示两个连续变量的关系")
    print("   • 线图: 展示时间序列或趋势")
    print("   • 直方图: 显示单个变量的分布")
    print("   • 柱状图: 比较不同类别的数值")
    print("   • 箱线图: 显示分布的五数概括")
    
    print("\n2. 数据预处理:")
    print("   • 处理缺失值和异常值")
    print("   • 选择合适的数据范围")
    print("   • 考虑数据的尺度和单位")
    
    print("\n3. 视觉设计原则:")
    print("   • 保持简洁，避免图表混乱")
    print("   • 使用一致的颜色和符号")
    print("   • 添加清晰的标签和图例")
    print("   • 突出重要信息")
    
    print("\n4. 机器学习可视化:")
    print("   • 数据探索: 分布图、相关性矩阵")
    print("   • 模型训练: 学习曲线、损失函数")
    print("   • 模型评估: 混淆矩阵、ROC曲线")
    print("   • 结果解释: 决策边界、特征重要性")
    
    print("\n5. 工具推荐:")
    print("   • matplotlib: Python标准绘图库")
    print("   • seaborn: 基于matplotlib的统计可视化")
    print("   • plotly: 交互式可视化")
    print("   • ASCII art: 简单的文本可视化（本工具）")

if __name__ == "__main__":
    visualization_theory()
    comprehensive_visualization_demo()
    visualization_best_practices()
    
    print("\n=== 总结 ===")
    print("数据可视化是机器学习的重要工具：")
    print("• 帮助理解数据的结构和模式")
    print("• 辅助特征工程和模型选择")
    print("• 监控模型训练过程")
    print("• 评估和解释模型结果")
    print("• 向他人展示分析成果")
    
    print("\n实践建议：")
    print("• 在每个分析阶段都进行可视化")
    print("• 选择最适合的图表类型")
    print("• 关注数据的故事和洞察")
    print("• 学习使用专业的可视化库")
    print("• 遵循良好的视觉设计原则")