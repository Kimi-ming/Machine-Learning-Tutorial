# -*- coding: utf-8 -*-
# 深度学习教程 - 循环神经网络 (RNN)
# Recurrent Neural Networks: 专门处理序列数据的深度学习架构

import random
import math

def rnn_theory():
    """
    循环神经网络原理解释
    
    RNN是专门设计用来处理序列数据的神经网络架构。
    
    核心概念：
    - 循环连接：网络具有记忆能力，当前输出依赖于历史信息
    - 隐藏状态：存储序列的历史信息
    - 权重共享：在时间步之间共享参数
    - 时间展开：将循环网络展开为前馈网络进行训练
    
    RNN的优势：
    1. 处理变长序列：可以处理任意长度的序列
    2. 记忆能力：能够记住之前的信息
    3. 参数共享：相同的参数在所有时间步使用
    4. 理论通用性：可以模拟任何递归函数
    
    RNN的挑战：
    1. 梯度消失：长序列训练困难
    2. 梯度爆炸：梯度可能指数增长
    3. 难以并行化：时间步之间存在依赖
    4. 长期依赖：难以记住很久以前的信息
    
    应用场景：
    - 自然语言处理：文本分类、机器翻译
    - 时间序列：股价预测、天气预报
    - 语音识别：语音到文字转换
    - 序列生成：文本生成、音乐创作
    """
    print("=== 循环神经网络 (RNN) 原理 ===")
    print("专门处理序列数据的深度学习架构")
    print()
    print("核心思想：")
    print("• 具有记忆：当前输出依赖历史信息")
    print("• 权重共享：时间步间共享参数")
    print("• 递归处理：逐步处理序列元素")
    print()

class SimpleRNN:
    """
    简化的RNN实现
    用于教学目的，展示RNN的基本原理
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化RNN
        input_size: 输入维度
        hidden_size: 隐藏状态维度
        output_size: 输出维度
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 权重矩阵初始化
        # Wxh: 输入到隐藏状态的权重
        self.Wxh = [[random.gauss(0, 0.1) for _ in range(hidden_size)] 
                   for _ in range(input_size)]
        
        # Whh: 隐藏状态到隐藏状态的权重（循环权重）
        self.Whh = [[random.gauss(0, 0.1) for _ in range(hidden_size)] 
                   for _ in range(hidden_size)]
        
        # Why: 隐藏状态到输出的权重
        self.Why = [[random.gauss(0, 0.1) for _ in range(output_size)] 
                   for _ in range(hidden_size)]
        
        # 偏置
        self.bh = [0.1 for _ in range(hidden_size)]
        self.by = [0.1 for _ in range(output_size)]
        
        print(f"RNN初始化完成:")
        print(f"输入维度: {input_size}")
        print(f"隐藏维度: {hidden_size}")
        print(f"输出维度: {output_size}")
        print(f"总参数量: {self.count_parameters()}")
    
    def count_parameters(self):
        """计算参数数量"""
        params = 0
        params += self.input_size * self.hidden_size  # Wxh
        params += self.hidden_size * self.hidden_size  # Whh
        params += self.hidden_size * self.output_size  # Why
        params += self.hidden_size + self.output_size  # 偏置
        return params
    
    def tanh(self, x):
        """tanh激活函数"""
        if x > 20:
            return 1.0
        elif x < -20:
            return -1.0
        exp_2x = math.exp(2 * x)
        return (exp_2x - 1) / (exp_2x + 1)
    
    def softmax(self, x):
        """softmax函数"""
        # 数值稳定的softmax
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [ei / sum_exp for ei in exp_x]
    
    def matrix_vector_multiply(self, matrix, vector):
        """矩阵向量乘法"""
        result = []
        for row in matrix:
            dot_product = sum(m * v for m, v in zip(row, vector))
            result.append(dot_product)
        return result
    
    def forward(self, inputs, initial_hidden=None):
        """
        前向传播
        inputs: 输入序列，列表的列表 [[x1], [x2], ...]
        initial_hidden: 初始隐藏状态
        """
        if initial_hidden is None:
            hidden = [0.0 for _ in range(self.hidden_size)]
        else:
            hidden = initial_hidden[:]
        
        outputs = []
        hidden_states = [hidden[:]]  # 保存所有隐藏状态
        
        for input_vec in inputs:
            # 计算隐藏状态: h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            h_input = self.matrix_vector_multiply(self.Wxh, input_vec)
            h_hidden = self.matrix_vector_multiply(self.Whh, hidden)
            
            new_hidden = []
            for i in range(self.hidden_size):
                h_val = h_input[i] + h_hidden[i] + self.bh[i]
                new_hidden.append(self.tanh(h_val))
            
            hidden = new_hidden
            hidden_states.append(hidden[:])
            
            # 计算输出: y_t = softmax(Why * h_t + by)
            y_input = self.matrix_vector_multiply(self.Why, hidden)
            output = []
            for i in range(self.output_size):
                output.append(y_input[i] + self.by[i])
            
            output = self.softmax(output)
            outputs.append(output)
        
        return outputs, hidden_states
    
    def predict(self, inputs):
        """预测"""
        outputs, _ = self.forward(inputs)
        return outputs

class SimpleLSTM:
    """
    简化的LSTM实现
    LSTM通过门控机制解决RNN的梯度消失问题
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """初始化LSTM"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # LSTM有四个门：忘记门、输入门、候选值、输出门
        # 为简化，这里只展示结构，不实现完整的训练过程
        
        print(f"LSTM初始化完成:")
        print(f"输入维度: {input_size}")
        print(f"隐藏维度: {hidden_size}")
        print(f"输出维度: {output_size}")
        print("LSTM包含：忘记门、输入门、输出门、候选值")
    
    def lstm_cell(self, x_t, h_prev, c_prev):
        """
        LSTM单元的计算步骤（概念展示）
        x_t: 当前输入
        h_prev: 前一个隐藏状态
        c_prev: 前一个细胞状态
        """
        # 忘记门：决定从细胞状态中丢弃什么信息
        # f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
        
        # 输入门：决定什么新信息被存放在细胞状态里
        # i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
        # C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
        
        # 更新细胞状态
        # C_t = f_t * C_{t-1} + i_t * C̃_t
        
        # 输出门：决定输出什么值
        # o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
        # h_t = o_t * tanh(C_t)
        
        # 这里返回概念性的输出
        return h_prev, c_prev

def sequence_prediction_demo():
    """序列预测演示"""
    print("\n=== 序列预测演示 ===")
    
    # 简单的序列预测：预测下一个数字
    print("任务：学习简单的数字序列规律")
    print("序列：0, 1, 2, 0, 1, 2, ... (重复模式)")
    print()
    
    # 准备数据
    sequence = [0, 1, 2] * 5  # [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
    
    # 创建训练样本：用前3个数字预测第4个数字
    def create_sequences(data, seq_length=3):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return X, y
    
    X, y = create_sequences(sequence, 3)
    
    print(f"生成了{len(X)}个训练样本:")
    print("输入序列 -> 目标输出")
    for i in range(min(8, len(X))):
        print(f"{X[i]} -> {y[i]}")
    
    # 将数字转换为one-hot编码
    def to_one_hot(value, num_classes=3):
        one_hot = [0] * num_classes
        one_hot[value] = 1
        return one_hot
    
    # 转换为RNN输入格式
    X_encoded = []
    for seq in X:
        encoded_seq = [to_one_hot(val) for val in seq]
        X_encoded.append(encoded_seq)
    
    y_encoded = [to_one_hot(val) for val in y]
    
    print(f"\nOne-hot编码示例:")
    print(f"数字序列: {X[0]}")
    print(f"编码后: {X_encoded[0]}")
    print(f"目标: {y[0]} -> {y_encoded[0]}")
    
    # 创建RNN模型
    rnn = SimpleRNN(input_size=3, hidden_size=5, output_size=3)
    
    # 测试前向传播
    print(f"\n测试RNN前向传播:")
    test_input = X_encoded[0]
    outputs, hidden_states = rnn.forward(test_input)
    
    print(f"输入序列长度: {len(test_input)}")
    print(f"输出序列长度: {len(outputs)}")
    print(f"最后输出: {[f'{o:.3f}' for o in outputs[-1]]}")
    
    # 预测下一个数字
    predicted_class = outputs[-1].index(max(outputs[-1]))
    print(f"预测的下一个数字: {predicted_class}")
    print(f"实际的下一个数字: {y[0]}")

def text_classification_demo():
    """文本分类演示（概念性）"""
    print("\n=== 文本情感分类演示 ===")
    
    print("任务：判断文本情感（正面/负面）")
    print()
    
    # 简化的文本数据
    texts = [
        ("我喜欢这个电影", "positive"),
        ("这个产品很糟糕", "negative"),
        ("服务态度很好", "positive"),
        ("质量不行", "negative")
    ]
    
    # 构建简单词汇表
    vocab = ["我", "喜欢", "这个", "电影", "产品", "很", "糟糕", "服务", "态度", "好", "质量", "不行"]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    print("词汇表:", vocab)
    print()
    
    # 文本转换为索引序列
    def text_to_indices(text, word_to_idx):
        # 简单的分词（实际应用中需要更复杂的处理）
        words = list(text)  # 字符级处理
        indices = []
        for word in words:
            if word in word_to_idx:
                indices.append(word_to_idx[word])
        return indices
    
    print("文本编码示例:")
    for text, label in texts:
        indices = text_to_indices(text, word_to_idx)
        print(f"'{text}' ({label}) -> {indices}")
    
    print(f"\nRNN文本分类流程:")
    print("1. 文本分词和编码")
    print("2. 词嵌入：将词索引转换为密集向量")
    print("3. RNN处理：逐词处理，更新隐藏状态")
    print("4. 分类器：最后的隐藏状态用于分类")
    print("5. 输出：情感分类结果")

def time_series_prediction_demo():
    """时间序列预测演示"""
    print("\n=== 时间序列预测演示 ===")
    
    print("任务：预测股价趋势（简化模型）")
    print()
    
    # 生成模拟的股价数据（简单趋势 + 噪声）
    def generate_stock_data(length=50):
        data = []
        base_price = 100
        trend = 0.1
        
        for i in range(length):
            # 简单趋势 + 随机波动
            price = base_price + i * trend + random.gauss(0, 0.5)
            data.append(price)
        
        return data
    
    stock_prices = generate_stock_data()
    
    print("模拟股价数据（前15天）:")
    for i in range(15):
        print(f"第{i+1:2d}天: {stock_prices[i]:.2f}")
    
    # 创建时间窗口
    def create_time_windows(data, window_size=5):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return X, y
    
    X_stock, y_stock = create_time_windows(stock_prices, window_size=5)
    
    print(f"\n时间窗口示例（用前5天预测第6天）:")
    for i in range(3):
        input_prices = [f"{p:.2f}" for p in X_stock[i]]
        target_price = y_stock[i]
        print(f"输入: [{', '.join(input_prices)}] -> 目标: {target_price:.2f}")
    
    print(f"\nRNN时间序列预测流程:")
    print("1. 数据归一化：将价格缩放到合理范围")
    print("2. 创建滑动窗口：使用历史数据预测未来")
    print("3. RNN处理：学习时间序列的模式")
    print("4. 回归输出：预测下一个时间点的值")
    print("5. 反归一化：恢复到原始价格范围")

def rnn_vs_lstm_comparison():
    """RNN vs LSTM 比较"""
    print("\n=== RNN vs LSTM 比较 ===")
    
    comparison = {
        "特性": ["记忆能力", "梯度问题", "计算复杂度", "参数数量", "训练难度"],
        "传统RNN": ["短期记忆", "梯度消失/爆炸", "简单", "少", "困难"],
        "LSTM": ["长期记忆", "缓解梯度问题", "复杂", "多", "相对容易"],
        "GRU": ["长期记忆", "缓解梯度问题", "中等", "中等", "中等"]
    }
    
    print(f"{'特性':<12} {'传统RNN':<15} {'LSTM':<15} {'GRU':<15}")
    print("-" * 60)
    
    for i, feature in enumerate(comparison["特性"]):
        rnn_val = comparison["传统RNN"][i]
        lstm_val = comparison["LSTM"][i]
        gru_val = comparison["GRU"][i]
        print(f"{feature:<12} {rnn_val:<15} {lstm_val:<15} {gru_val:<15}")
    
    print(f"\n详细对比:")
    print(f"【传统RNN】")
    print(f"优点: 结构简单，计算快速，参数少")
    print(f"缺点: 梯度消失，难以处理长序列")
    print(f"适用: 短序列，简单任务")
    print()
    
    print(f"【LSTM】")
    print(f"优点: 解决梯度消失，处理长序列，性能稳定")
    print(f"缺点: 结构复杂，计算开销大，参数多")
    print(f"适用: 长序列，复杂任务，工业应用")
    print()
    
    print(f"【GRU】")
    print(f"优点: 简化版LSTM，参数较少，训练快速")
    print(f"缺点: 性能可能略逊于LSTM")
    print(f"适用: 计算资源受限，快速原型")

def attention_mechanism_intro():
    """注意力机制简介"""
    print("\n=== 注意力机制简介 ===")
    
    print("注意力机制的动机:")
    print("• 解决RNN的信息瓶颈问题")
    print("• 允许模型关注输入序列的不同部分")
    print("• 提高长序列处理能力")
    print("• 增强模型的可解释性")
    print()
    
    print("基本思想:")
    print("1. 计算注意力权重：确定每个输入的重要性")
    print("2. 加权求和：根据注意力权重组合信息")
    print("3. 动态关注：不同时刻关注不同的输入部分")
    print()
    
    print("注意力机制的类型:")
    print("• 加性注意力：使用前馈网络计算注意力分数")
    print("• 乘性注意力：使用点积计算注意力分数")
    print("• 自注意力：序列内部元素之间的注意力")
    print("• 多头注意力：并行计算多个注意力表示")
    print()
    
    print("应用效果:")
    print("• 机器翻译：关注源语言相关词汇")
    print("• 文档分类：关注关键句子或段落") 
    print("• 图像描述：关注图像相关区域")
    print("• 问答系统：关注问题相关信息")

def rnn_applications_overview():
    """RNN应用概览"""
    print("\n=== RNN主要应用领域 ===")
    
    applications = {
        "自然语言处理": {
            "任务": ["文本分类", "情感分析", "命名实体识别", "机器翻译"],
            "特点": "处理词序列，理解语言结构",
            "挑战": "语言歧义性，长距离依赖"
        },
        
        "时间序列分析": {
            "任务": ["股价预测", "天气预报", "销量预测", "设备监控"],
            "特点": "捕获时间模式，预测未来趋势",
            "挑战": "数据稀疏性，非平稳性"
        },
        
        "语音识别": {
            "任务": ["语音转文字", "语音命令", "说话人识别"],
            "特点": "处理音频时间序列",
            "挑战": "噪声干扰，方言差异"
        },
        
        "序列生成": {
            "任务": ["文本生成", "音乐创作", "代码生成"],
            "特点": "学习序列分布，生成新序列",
            "挑战": "保持连贯性，避免重复"
        },
        
        "生物信息学": {
            "任务": ["蛋白质序列分析", "基因预测", "药物发现"],
            "特点": "处理生物序列数据",
            "挑战": "序列变异性，注释缺失"
        }
    }
    
    for domain, info in applications.items():
        print(f"【{domain}】")
        print(f"主要任务: {', '.join(info['任务'])}")
        print(f"特点: {info['特点']}")
        print(f"挑战: {info['挑战']}")
        print()

def rnn_training_tips():
    """RNN训练技巧"""
    print("=== RNN训练技巧 ===")
    
    print("解决梯度问题:")
    print("• 梯度裁剪：限制梯度的最大范数")
    print("• 使用LSTM/GRU：通过门控机制缓解梯度消失")
    print("• 合适的激活函数：避免饱和激活函数")
    print("• 权重初始化：使用合适的初始化方法")
    print()
    
    print("序列处理技巧:")
    print("• 序列填充：统一序列长度")
    print("• 序列掩码：忽略填充部分的损失")
    print("• 梯度累积：处理长序列时分批累积梯度")
    print("• 截断反向传播：限制时间步长度")
    print()
    
    print("训练策略:")
    print("• 数据增强：序列的旋转、裁剪等")
    print("• 正则化：Dropout、L2正则化")
    print("• 学习率调度：动态调整学习率")
    print("• 早停策略：防止过拟合")
    print()
    
    print("调试技巧:")
    print("• 监控梯度：检查梯度大小和分布")
    print("• 可视化注意力：理解模型关注点")
    print("• 简化数据：从简单序列开始测试")
    print("• 逐层验证：确保每层输出合理")

def main():
    """主函数"""
    print("🔄 循环神经网络 (RNN) 教程")
    print("=" * 50)
    
    rnn_theory()
    sequence_prediction_demo()
    text_classification_demo()
    time_series_prediction_demo()
    rnn_vs_lstm_comparison()
    attention_mechanism_intro()
    rnn_applications_overview()
    rnn_training_tips()
    
    print("\n" + "=" * 50)
    print("📝 RNN学习要点总结")
    print()
    print("核心概念:")
    print("• 循环连接：具有记忆能力的网络")
    print("• 权重共享：时间步间参数共享")
    print("• 序列建模：处理变长时序数据")
    print("• 门控机制：LSTM/GRU解决梯度问题")
    print()
    print("实践要点:")
    print("• 理解RNN的时间展开过程")
    print("• 掌握LSTM的门控机制")
    print("• 学会处理序列数据")
    print("• 注意梯度消失/爆炸问题")
    print()
    print("下一步学习:")
    print("• 实现文本分类项目")
    print("• 学习Transformer架构")
    print("• 了解注意力机制")
    print("• 探索预训练语言模型")

if __name__ == "__main__":
    main()