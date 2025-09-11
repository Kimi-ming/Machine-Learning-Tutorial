# -*- coding: utf-8 -*-
# 深度学习高级主题
# Advanced Deep Learning Topics: 探索前沿技术

import random
import math
import json

def advanced_topics_introduction():
    """高级主题介绍"""
    print("=== 深度学习高级主题 ===")
    print("探索深度学习的前沿技术和最新发展")
    print()
    print("涵盖内容:")
    print("- 生成对抗网络 (GANs)")
    print("- Transformer和注意力机制")
    print("- 自监督学习")
    print("- 图神经网络")
    print("- 强化学习")
    print("- 联邦学习")
    print("- 神经网络压缩")
    print("- 可解释AI")
    print()

def generative_adversarial_networks():
    """生成对抗网络 (GANs)"""
    print("\n" + "="*50)
    print("生成对抗网络 (GANs)")
    print("="*50)
    
    print("核心思想:")
    print("两个神经网络相互对抗训练：")
    print("- 生成器 (Generator): 生成假数据，试图欺骗判别器")
    print("- 判别器 (Discriminator): 区分真实数据和生成数据")
    print()
    
    print("训练过程:")
    print("1. 判别器学习区分真假数据")
    print("2. 生成器学习生成更逼真的数据")
    print("3. 两者相互竞争，共同提升")
    print("4. 最终生成器能产生高质量数据")
    print()
    
    # 简单的GAN概念演示
    class SimpleGAN:
        """简化的GAN概念演示"""
        
        def __init__(self):
            # 简化的网络参数
            self.generator_skill = 0.1    # 生成器技能水平
            self.discriminator_skill = 0.1  # 判别器技能水平
            self.training_history = []
            
        def train_discriminator(self, real_data, fake_data):
            """训练判别器"""
            # 判别器学习区分真假数据
            real_accuracy = min(0.95, self.discriminator_skill + random.uniform(0, 0.1))
            fake_accuracy = min(0.95, self.discriminator_skill + random.uniform(0, 0.1))
            
            # 更新判别器技能
            improvement = (real_accuracy + fake_accuracy) / 2 * 0.1
            self.discriminator_skill = min(0.9, self.discriminator_skill + improvement)
            
            return real_accuracy, fake_accuracy
            
        def train_generator(self, discriminator_feedback):
            """训练生成器"""
            # 生成器根据判别器反馈改进
            if discriminator_feedback < 0.5:  # 成功欺骗判别器
                improvement = 0.05
            else:
                improvement = random.uniform(0.01, 0.03)
            
            self.generator_skill = min(0.9, self.generator_skill + improvement)
            
        def train_epoch(self):
            """训练一轮"""
            # 生成假数据质量
            fake_quality = self.generator_skill + random.uniform(-0.1, 0.1)
            fake_quality = max(0, min(1, fake_quality))
            
            # 判别器评估
            real_acc, fake_acc = self.train_discriminator("real_data", fake_quality)
            discriminator_loss = 1 - (real_acc + (1 - fake_acc)) / 2
            
            # 生成器训练
            generator_feedback = fake_acc  # 被判别器识破的概率
            self.train_generator(generator_feedback)
            generator_loss = fake_acc  # 生成器希望最小化被识破的概率
            
            # 记录历史
            self.training_history.append({
                'generator_skill': self.generator_skill,
                'discriminator_skill': self.discriminator_skill,
                'fake_quality': fake_quality,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss
            })
            
            return generator_loss, discriminator_loss
    
    # 演示GAN训练
    print("GAN训练演示:")
    gan = SimpleGAN()
    
    print("轮次 | 生成器技能 | 判别器技能 | 生成质量 | 生成器损失 | 判别器损失")
    print("----|----------|----------|----------|----------|----------")
    
    for epoch in range(20):
        gen_loss, disc_loss = gan.train_epoch()
        history = gan.training_history[-1]
        
        if epoch % 4 == 0:
            print(f"{epoch:3d} | {history['generator_skill']:9.3f} | {history['discriminator_skill']:9.3f} | {history['fake_quality']:8.3f} | {gen_loss:10.3f} | {disc_loss:9.3f}")
    
    print(f"\n训练结果分析:")
    print(f"- 生成器技能提升: {gan.generator_skill:.3f}")
    print(f"- 判别器技能提升: {gan.discriminator_skill:.3f}")
    print(f"- 两者在对抗中共同成长")
    
    print(f"\nGAN的应用:")
    print("- 图像生成: 生成逼真的人脸、艺术作品")
    print("- 数据增强: 为训练生成更多样本")
    print("- 风格迁移: 改变图像的艺术风格")
    print("- 超分辨率: 提高图像分辨率")
    print("- 音乐生成: 创作新的音乐作品")
    
    print(f"\n主要挑战:")
    print("- 训练不稳定: 容易出现模式坍塌")
    print("- 平衡困难: 生成器和判别器需要平衡")
    print("- 评估困难: 缺乏客观的质量评估指标")
    print("- 计算昂贵: 需要大量计算资源")

def transformer_architecture():
    """Transformer架构详解"""
    print("\n" + "="*50)
    print("Transformer架构")
    print("="*50)
    
    print("革命性贡献:")
    print("- 完全基于注意力机制，抛弃了循环和卷积")
    print("- 可以并行化训练，大大提升效率")
    print("- 在机器翻译等任务上取得突破性进展")
    print("- 成为现代NLP的基础架构")
    print()
    
    print("核心组件:")
    print("1. 多头自注意力 (Multi-Head Self-Attention)")
    print("2. 位置编码 (Positional Encoding)")
    print("3. 前馈网络 (Feed-Forward Networks)")
    print("4. 残差连接 (Residual Connections)")
    print("5. 层归一化 (Layer Normalization)")
    print()
    
    class SimpleAttention:
        """简化的注意力机制演示"""
        
        def __init__(self):
            pass
            
        def attention_weights(self, query, keys):
            """计算注意力权重"""
            # 简化的点积注意力
            scores = []
            for key in keys:
                # 计算query和每个key的相似度
                score = sum(q * k for q, k in zip(query, key))
                scores.append(score)
            
            # Softmax归一化
            exp_scores = [math.exp(s) for s in scores]
            total = sum(exp_scores)
            weights = [s / total for s in exp_scores]
            
            return weights
            
        def apply_attention(self, query, keys, values):
            """应用注意力机制"""
            weights = self.attention_weights(query, keys)
            
            # 加权求和
            output = [0] * len(values[0])
            for i, (weight, value) in enumerate(zip(weights, values)):
                for j in range(len(output)):
                    output[j] += weight * value[j]
            
            return output, weights
    
    # 注意力机制演示
    print("注意力机制演示:")
    print("句子: ['我', '喜欢', '深度', '学习']")
    print("计算'学习'对其他词的注意力")
    
    attention = SimpleAttention()
    
    # 简化的词向量（实际中是高维向量）
    word_vectors = {
        '我': [1, 0, 0],
        '喜欢': [0, 1, 0], 
        '深度': [0, 0, 1],
        '学习': [1, 1, 0]
    }
    
    query = word_vectors['学习']
    keys = [word_vectors[word] for word in ['我', '喜欢', '深度', '学习']]
    values = keys[:]  # 在自注意力中，keys和values相同
    
    result, weights = attention.apply_attention(query, keys, values)
    
    print(f"Query (学习): {query}")
    print(f"注意力权重:")
    words = ['我', '喜欢', '深度', '学习']
    for word, weight in zip(words, weights):
        print(f"  {word}: {weight:.3f}")
    
    print(f"注意力输出: {[f'{x:.3f}' for x in result]}")
    
    print(f"\n多头注意力:")
    print("- 使用多个注意力头并行计算")
    print("- 每个头关注不同的特征子空间")
    print("- 最后将所有头的输出连接起来")
    
    print(f"\nTransformer的优势:")
    print("- 并行化: 可以同时处理整个序列")
    print("- 长距离依赖: 直接建模任意位置间的关系")
    print("- 灵活性: 适用于各种序列到序列任务")
    print("- 可解释性: 注意力权重提供可解释性")
    
    print(f"\nTransformer应用:")
    print("- BERT: 预训练语言理解模型")
    print("- GPT: 大规模语言生成模型")
    print("- Vision Transformer: 图像分类")
    print("- 机器翻译: Google翻译的核心技术")

def self_supervised_learning():
    """自监督学习"""
    print("\n" + "="*50)
    print("自监督学习")
    print("="*50)
    
    print("基本概念:")
    print("- 从未标注数据中自动生成监督信号")
    print("- 不需要人工标注，可以利用大量无标注数据")
    print("- 学习通用的数据表示")
    print("- 为下游任务提供良好的初始化")
    print()
    
    print("主要范式:")
    print()
    
    pretext_tasks = {
        "遮蔽语言建模 (MLM)": {
            "描述": "随机遮蔽部分词汇，预测被遮蔽的词",
            "例子": "输入: '我[MASK]深度学习' -> 预测: '喜欢'",
            "应用": "BERT等语言模型的预训练"
        },
        
        "下一句预测 (NSP)": {
            "描述": "判断两个句子是否连续",
            "例子": "句子A + 句子B -> 预测是否为连续句子",
            "应用": "理解句子间的关系"
        },
        
        "图像旋转预测": {
            "描述": "预测图像被旋转的角度",
            "例子": "输入旋转后的图像 -> 预测旋转角度",
            "应用": "学习图像的空间特征"
        },
        
        "图像修复": {
            "描述": "恢复被遮挡的图像区域",
            "例子": "输入部分遮挡的图像 -> 预测完整图像",
            "应用": "学习图像的结构信息"
        },
        
        "对比学习": {
            "描述": "学习相似样本靠近，不同样本远离",
            "例子": "同一图像的不同变换应该相似",
            "应用": "SimCLR, MoCo等方法"
        }
    }
    
    for task_name, info in pretext_tasks.items():
        print(f"【{task_name}】")
        print(f"描述: {info['描述']}")
        print(f"例子: {info['例子']}")
        print(f"应用: {info['应用']}")
        print()
    
    print("自监督学习的优势:")
    print("- 数据效率: 可以利用大量无标注数据")
    print("- 通用性: 学到的表示可以迁移到多个任务")
    print("- 成本低: 不需要昂贵的人工标注")
    print("- 扩展性: 容易扩展到更大的数据集")
    
    print(f"\n成功案例:")
    print("- BERT: 通过MLM学习语言表示")
    print("- SimCLR: 通过对比学习学习视觉表示")
    print("- GPT: 通过语言建模学习生成能力")
    print("- MAE: 通过图像修复学习视觉表示")

def graph_neural_networks():
    """图神经网络 (GNNs)"""
    print("\n" + "="*50)
    print("图神经网络 (GNNs)")
    print("="*50)
    
    print("应用背景:")
    print("许多现实世界的数据具有图结构:")
    print("- 社交网络: 用户和关系")
    print("- 知识图谱: 实体和关系")
    print("- 分子结构: 原子和化学键")
    print("- 交通网络: 路口和道路")
    print()
    
    print("核心思想:")
    print("- 节点特征通过边进行信息传递")
    print("- 聚合邻居信息更新节点表示")
    print("- 保持图的结构信息")
    print("- 可以处理不规则结构数据")
    print()
    
    class SimpleGCN:
        """简化的图卷积网络演示"""
        
        def __init__(self, input_dim, hidden_dim):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            # 简化的权重矩阵
            self.W = [[random.uniform(-0.5, 0.5) for _ in range(hidden_dim)] 
                     for _ in range(input_dim)]
            
        def message_passing(self, node_features, adjacency_matrix):
            """消息传递"""
            num_nodes = len(node_features)
            
            # 聚合邻居信息
            aggregated_features = []
            for i in range(num_nodes):
                # 收集邻居特征
                neighbor_features = []
                for j in range(num_nodes):
                    if adjacency_matrix[i][j] == 1:  # 如果是邻居
                        neighbor_features.append(node_features[j])
                
                # 平均聚合 (简化版)
                if neighbor_features:
                    avg_feature = []
                    for dim in range(len(node_features[0])):
                        avg = sum(feat[dim] for feat in neighbor_features) / len(neighbor_features)
                        avg_feature.append(avg)
                    aggregated_features.append(avg_feature)
                else:
                    aggregated_features.append(node_features[i][:])
            
            return aggregated_features
            
        def forward(self, node_features, adjacency_matrix):
            """前向传播"""
            # 消息传递
            aggregated = self.message_passing(node_features, adjacency_matrix)
            
            # 线性变换
            output = []
            for feat in aggregated:
                new_feat = []
                for j in range(self.hidden_dim):
                    val = sum(feat[i] * self.W[i][j] for i in range(len(feat)))
                    new_feat.append(max(0, val))  # ReLU激活
                output.append(new_feat)
            
            return output
    
    # GCN演示
    print("GCN演示 - 简单图结构:")
    print("图结构: 0-1-2")
    print("       \\ /")
    print("        3")
    
    # 邻接矩阵
    adj_matrix = [
        [0, 1, 0, 1],  # 节点0连接到节点1,3
        [1, 0, 1, 1],  # 节点1连接到节点0,2,3
        [0, 1, 0, 1],  # 节点2连接到节点1,3
        [1, 1, 1, 0]   # 节点3连接到节点0,1,2
    ]
    
    # 初始节点特征
    node_features = [
        [1, 0],  # 节点0特征
        [0, 1],  # 节点1特征
        [1, 1],  # 节点2特征
        [0, 0]   # 节点3特征
    ]
    
    gcn = SimpleGCN(input_dim=2, hidden_dim=3)
    
    print(f"\n初始节点特征:")
    for i, feat in enumerate(node_features):
        print(f"节点{i}: {feat}")
    
    output = gcn.forward(node_features, adj_matrix)
    
    print(f"\nGCN输出特征:")
    for i, feat in enumerate(output):
        print(f"节点{i}: {[f'{x:.3f}' for x in feat]}")
    
    print(f"\nGNN的优势:")
    print("- 处理非欧几里得数据: 图结构数据")
    print("- 归纳学习: 可以处理新节点")
    print("- 灵活性: 适应不同的图结构")
    print("- 可解释性: 保持结构信息")
    
    print(f"\nGNN应用:")
    print("- 社交网络分析: 用户兴趣预测")
    print("- 推荐系统: 基于图的协同过滤")
    print("- 药物发现: 分子性质预测")
    print("- 知识推理: 知识图谱补全")

def reinforcement_learning_basics():
    """强化学习基础"""
    print("\n" + "="*50)
    print("强化学习基础")
    print("="*50)
    
    print("基本概念:")
    print("- 智能体 (Agent): 学习如何行动的实体")
    print("- 环境 (Environment): 智能体所处的世界")
    print("- 状态 (State): 环境的当前情况")
    print("- 动作 (Action): 智能体可以采取的行为")
    print("- 奖励 (Reward): 环境对动作的反馈")
    print("- 策略 (Policy): 智能体的行为准则")
    print()
    
    print("学习目标:")
    print("学习最优策略，使累积奖励最大化")
    print()
    
    class SimpleQLearning:
        """简化的Q学习演示"""
        
        def __init__(self, states, actions, learning_rate=0.1, discount=0.9, epsilon=0.1):
            self.states = states
            self.actions = actions
            self.lr = learning_rate
            self.gamma = discount
            self.epsilon = epsilon
            
            # 初始化Q表
            self.q_table = {}
            for state in states:
                self.q_table[state] = {}
                for action in actions:
                    self.q_table[state][action] = 0.0
                    
        def choose_action(self, state):
            """选择动作 (ε-贪心策略)"""
            if random.random() < self.epsilon:
                return random.choice(self.actions)  # 探索
            else:
                # 选择Q值最大的动作
                return max(self.actions, key=lambda a: self.q_table[state][a])
                
        def update_q(self, state, action, reward, next_state):
            """更新Q值"""
            # Q学习更新规则
            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values()) if next_state else 0
            
            new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
        def print_q_table(self):
            """打印Q表"""
            print("Q表:")
            print("状态\\动作", end="")
            for action in self.actions:
                print(f"  {action:>6}", end="")
            print()
            
            for state in self.states:
                print(f"{state:>7}", end="")
                for action in self.actions:
                    print(f"  {self.q_table[state][action]:6.2f}", end="")
                print()
    
    # 简单的网格世界示例
    print("网格世界示例:")
    print("3x3网格，智能体从左上角到右下角")
    print("┌───┬───┬───┐")
    print("│ S │   │   │")  # S: 起始位置
    print("├───┼───┼───┤")
    print("│   │ x │   │")  # x: 障碍
    print("├───┼───┼───┤")
    print("│   │   │ G │")  # G: 目标
    print("└───┴───┴───┘")
    
    # 定义状态和动作
    states = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    actions = ['up', 'down', 'left', 'right']
    
    agent = SimpleQLearning(states, actions)
    
    # 模拟训练过程
    print(f"\n模拟Q学习训练...")
    
    # 简单的奖励函数
    def get_reward(state, action, next_state):
        if next_state == 'G':
            return 10  # 到达目标
        elif next_state == 'D':  # 撞到障碍
            return -5
        else:
            return -0.1  # 每步小惩罚
    
    # 简化的状态转移
    transitions = {
        'S': {'right': 'A', 'down': 'C'},
        'A': {'left': 'S', 'right': 'B'},
        'B': {'left': 'A', 'down': 'E'},
        'C': {'up': 'S', 'down': 'F'},
        'E': {'up': 'B', 'left': 'D', 'down': 'G'},
        'F': {'up': 'C', 'right': 'G'},
        'G': {}  # 终止状态
    }
    
    # 训练几轮
    for episode in range(100):
        state = 'S'
        while state != 'G':
            action = agent.choose_action(state)
            if state in transitions and action in transitions[state]:
                next_state = transitions[state][action]
                reward = get_reward(state, action, next_state)
                agent.update_q(state, action, reward, next_state)
                state = next_state
            else:
                # 无效动作，给予惩罚
                agent.update_q(state, action, -1, state)
                break
    
    print(f"训练完成！")
    agent.print_q_table()
    
    print(f"\n强化学习的特点:")
    print("- 试错学习: 通过与环境交互学习")
    print("- 延迟奖励: 当前行动影响未来奖励")
    print("- 探索vs利用: 平衡尝试新策略和使用已知好策略")
    print("- 序贯决策: 考虑长期影响")
    
    print(f"\n深度强化学习:")
    print("- DQN: 用神经网络近似Q函数")
    print("- Policy Gradient: 直接优化策略")
    print("- Actor-Critic: 结合价值和策略学习")
    print("- PPO: 稳定的策略优化算法")
    
    print(f"\n应用场景:")
    print("- 游戏AI: AlphaGo, OpenAI Five")
    print("- 自动驾驶: 路径规划和控制")
    print("- 机器人控制: 运动控制和任务规划")
    print("- 推荐系统: 序贯推荐")

def federated_learning():
    """联邦学习"""
    print("\n" + "="*50)
    print("联邦学习")
    print("="*50)
    
    print("核心思想:")
    print("- 分布式机器学习范式")
    print("- 数据不出本地，模型参数共享")
    print("- 保护数据隐私和安全")
    print("- 利用分布式数据训练全局模型")
    print()
    
    print("基本流程:")
    print("1. 服务器初始化全局模型")
    print("2. 分发模型到各个客户端")
    print("3. 客户端使用本地数据训练")
    print("4. 上传模型更新到服务器")
    print("5. 服务器聚合更新，更新全局模型")
    print("6. 重复步骤2-5直到收敛")
    print()
    
    class FederatedAveraging:
        """联邦平均算法演示"""
        
        def __init__(self, num_clients):
            self.num_clients = num_clients
            self.global_weights = [0.5, 0.3]  # 简化的全局模型参数
            self.client_data_sizes = [random.randint(100, 500) for _ in range(num_clients)]
            
        def client_train(self, client_id, local_epochs=5):
            """客户端本地训练"""
            # 模拟本地训练过程
            local_weights = self.global_weights[:]
            
            # 模拟梯度更新
            for epoch in range(local_epochs):
                for i in range(len(local_weights)):
                    # 简化的梯度更新
                    gradient = random.uniform(-0.1, 0.1)
                    local_weights[i] -= 0.01 * gradient
            
            return local_weights
            
        def federated_averaging(self, client_updates):
            """联邦平均聚合"""
            total_data = sum(self.client_data_sizes)
            
            # 根据数据量加权平均
            new_weights = [0.0] * len(self.global_weights)
            for client_id, update in enumerate(client_updates):
                weight = self.client_data_sizes[client_id] / total_data
                for i in range(len(new_weights)):
                    new_weights[i] += weight * update[i]
            
            return new_weights
            
        def train_round(self):
            """训练一轮"""
            # 所有客户端进行本地训练
            client_updates = []
            for client_id in range(self.num_clients):
                update = self.client_train(client_id)
                client_updates.append(update)
            
            # 服务器聚合更新
            self.global_weights = self.federated_averaging(client_updates)
            
            return client_updates
    
    print("联邦学习演示:")
    fl_system = FederatedAveraging(num_clients=5)
    
    print(f"初始全局模型参数: {fl_system.global_weights}")
    print(f"客户端数据量: {fl_system.client_data_sizes}")
    
    print(f"\n训练过程:")
    for round_num in range(5):
        client_updates = fl_system.train_round()
        print(f"轮次 {round_num + 1}:")
        print(f"  全局参数: {[f'{w:.3f}' for w in fl_system.global_weights]}")
        print(f"  客户端更新示例: {[f'{w:.3f}' for w in client_updates[0]]}")
    
    print(f"\n联邦学习的优势:")
    print("- 隐私保护: 数据不离开本地设备")
    print("- 带宽效率: 只传输模型参数")
    print("- 个性化: 可以保留本地特性")
    print("- 扩展性: 可以处理大规模分布式数据")
    
    print(f"\n挑战:")
    print("- 数据异质性: 不同客户端数据分布不同")
    print("- 通信效率: 频繁的模型传输")
    print("- 设备异质性: 不同设备的计算能力差异")
    print("- 隐私攻击: 仍可能从模型参数推断信息")
    
    print(f"\n应用场景:")
    print("- 移动设备: 手机键盘预测")
    print("- 医疗健康: 跨医院协作建模")
    print("- 金融服务: 跨机构风险评估")
    print("- 智能家居: 设备协同学习")

def model_compression():
    """神经网络压缩"""
    print("\n" + "="*50)
    print("神经网络压缩")
    print("="*50)
    
    print("压缩的必要性:")
    print("- 部署限制: 移动设备资源有限")
    print("- 推理速度: 实时应用需要快速推理")
    print("- 存储空间: 减少模型存储需求")
    print("- 能耗考虑: 降低计算能耗")
    print()
    
    compression_techniques = {
        "剪枝 (Pruning)": {
            "原理": "移除不重要的权重或神经元",
            "类型": ["结构化剪枝", "非结构化剪枝", "动态剪枝"],
            "优势": "显著减少参数量",
            "劣势": "可能需要专门的硬件支持"
        },
        
        "量化 (Quantization)": {
            "原理": "降低权重和激活的数值精度",
            "类型": ["训练后量化", "量化感知训练", "动态量化"],
            "优势": "减少存储和计算需求",
            "劣势": "精度损失"
        },
        
        "知识蒸馏 (Knowledge Distillation)": {
            "原理": "用大模型的知识训练小模型",
            "类型": ["响应蒸馏", "特征蒸馏", "注意力蒸馏"],
            "优势": "保持较高精度",
            "劣势": "需要教师模型"
        },
        
        "低秩分解 (Low-Rank Factorization)": {
            "原理": "将权重矩阵分解为低秩矩阵乘积",
            "类型": ["SVD分解", "Tucker分解", "CP分解"],
            "优势": "理论保证",
            "劣势": "压缩比有限"
        }
    }
    
    for technique, info in compression_techniques.items():
        print(f"【{technique}】")
        print(f"原理: {info['原理']}")
        print(f"类型: {', '.join(info['类型'])}")
        print(f"优势: {info['优势']}")
        print(f"劣势: {info['劣势']}")
        print()
    
    # 简单的剪枝演示
    class SimplePruning:
        """简单的权重剪枝演示"""
        
        def __init__(self, weights):
            self.original_weights = weights[:]
            self.pruned_weights = weights[:]
            
        def magnitude_pruning(self, prune_ratio=0.5):
            """基于权重大小的剪枝"""
            # 计算权重的绝对值
            abs_weights = [abs(w) for w in self.original_weights]
            
            # 找到阈值
            sorted_weights = sorted(abs_weights)
            threshold_idx = int(len(sorted_weights) * prune_ratio)
            threshold = sorted_weights[threshold_idx]
            
            # 剪枝
            self.pruned_weights = []
            pruned_count = 0
            for w in self.original_weights:
                if abs(w) <= threshold:
                    self.pruned_weights.append(0.0)
                    pruned_count += 1
                else:
                    self.pruned_weights.append(w)
            
            return pruned_count
            
        def get_compression_ratio(self):
            """计算压缩比"""
            non_zero_original = sum(1 for w in self.original_weights if w != 0)
            non_zero_pruned = sum(1 for w in self.pruned_weights if w != 0)
            
            return non_zero_pruned / non_zero_original if non_zero_original > 0 else 0
    
    print("权重剪枝演示:")
    
    # 生成示例权重
    original_weights = [random.uniform(-1, 1) for _ in range(20)]
    
    pruner = SimplePruning(original_weights)
    
    print(f"原始权重 (前10个): {[f'{w:.3f}' for w in original_weights[:10]]}")
    
    pruned_count = pruner.magnitude_pruning(prune_ratio=0.5)
    compression_ratio = pruner.get_compression_ratio()
    
    print(f"剪枝后权重 (前10个): {[f'{w:.3f}' for w in pruner.pruned_weights[:10]]}")
    print(f"剪枝了 {pruned_count} 个权重")
    print(f"压缩比: {compression_ratio:.2f} (保留了 {compression_ratio*100:.1f}% 的参数)")
    
    print(f"\n模型压缩的评估指标:")
    print("- 压缩比: 压缩后大小 / 原始大小")
    print("- 加速比: 原始推理时间 / 压缩后推理时间")
    print("- 精度损失: 原始精度 - 压缩后精度")
    print("- 能耗比: 原始能耗 / 压缩后能耗")

def explainable_ai():
    """可解释AI"""
    print("\n" + "="*50)
    print("可解释AI (XAI)")
    print("="*50)
    
    print("重要性:")
    print("- 信任建立: 用户需要理解AI的决策过程")
    print("- 法规要求: 某些领域要求算法透明")
    print("- 调试需求: 理解模型错误的原因")
    print("- 伦理责任: 确保AI决策的公平性")
    print()
    
    print("可解释性类型:")
    print("- 全局可解释性: 理解整个模型的行为")
    print("- 局部可解释性: 理解单个预测的原因")
    print("- 事后解释: 对已训练模型进行解释")
    print("- 内在可解释: 模型本身就是可解释的")
    print()
    
    explanation_methods = {
        "LIME (Local Interpretable Model-agnostic Explanations)": {
            "原理": "在预测点附近拟合简单的可解释模型",
            "适用": "任何机器学习模型",
            "输出": "特征重要性分数",
            "优势": "模型无关，易于理解"
        },
        
        "SHAP (SHapley Additive exPlanations)": {
            "原理": "基于合作博弈论的特征贡献分解",
            "适用": "各种模型类型",
            "输出": "每个特征的SHAP值",
            "优势": "理论基础扎实，公平分配"
        },
        
        "Grad-CAM": {
            "原理": "利用梯度信息生成类激活图",
            "适用": "卷积神经网络",
            "输出": "热力图显示重要区域",
            "优势": "直观的视觉解释"
        },
        
        "注意力可视化": {
            "原理": "显示注意力权重分布",
            "适用": "带注意力机制的模型",
            "输出": "注意力权重矩阵",
            "优势": "模型内置的可解释性"
        }
    }
    
    for method, info in explanation_methods.items():
        print(f"【{method}】")
        print(f"原理: {info['原理']}")
        print(f"适用: {info['适用']}")
        print(f"输出: {info['输出']}")
        print(f"优势: {info['优势']}")
        print()
    
    # 简单的特征重要性演示
    class SimpleFeatureImportance:
        """简单的特征重要性计算"""
        
        def __init__(self, model_predict):
            self.predict = model_predict
            
        def permutation_importance(self, X, y, feature_names):
            """置换特征重要性"""
            # 获取原始预测性能
            original_pred = [self.predict(x) for x in X]
            original_accuracy = sum(1 for pred, true in zip(original_pred, y) if abs(pred - true) < 0.5) / len(y)
            
            importance_scores = []
            
            for feature_idx in range(len(X[0])):
                # 置换该特征
                X_permuted = []
                for x in X:
                    x_perm = x[:]
                    # 随机置换特征值
                    random_idx = random.randint(0, len(X) - 1)
                    x_perm[feature_idx] = X[random_idx][feature_idx]
                    X_permuted.append(x_perm)
                
                # 计算置换后的性能
                permuted_pred = [self.predict(x) for x in X_permuted]
                permuted_accuracy = sum(1 for pred, true in zip(permuted_pred, y) if abs(pred - true) < 0.5) / len(y)
                
                # 重要性 = 原始性能 - 置换后性能
                importance = original_accuracy - permuted_accuracy
                importance_scores.append((feature_names[feature_idx], importance))
            
            return sorted(importance_scores, key=lambda x: x[1], reverse=True)
    
    # 模拟一个简单的模型和数据
    def simple_model(x):
        """简单的线性模型"""
        return x[0] * 0.8 + x[1] * 0.3 + x[2] * 0.1
    
    print("特征重要性演示:")
    
    # 生成模拟数据
    X_sample = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for _ in range(50)]
    y_sample = [simple_model(x) for x in X_sample]
    feature_names = ['特征1', '特征2', '特征3']
    
    explainer = SimpleFeatureImportance(simple_model)
    importance_scores = explainer.permutation_importance(X_sample, y_sample, feature_names)
    
    print("置换特征重要性结果:")
    print("特征     | 重要性分数")
    print("---------|----------")
    for feature, score in importance_scores:
        print(f"{feature:8} | {score:9.4f}")
    
    print(f"\n解释结果分析:")
    print(f"- {importance_scores[0][0]} 最重要 (分数: {importance_scores[0][1]:.4f})")
    print(f"- 这与模型设计一致 (权重: 0.8, 0.3, 0.1)")
    print(f"- 置换重要特征会显著降低模型性能")
    
    print(f"\n可解释AI的挑战:")
    print("- 解释的准确性: 解释是否真实反映模型行为")
    print("- 解释的完整性: 是否涵盖了所有重要因素")
    print("- 用户理解: 如何让非专业用户理解解释")
    print("- 计算效率: 生成解释的计算成本")
    
    print(f"\n应用场景:")
    print("- 医疗诊断: 解释诊断依据")
    print("- 金融风控: 解释信贷决策")
    print("- 司法系统: 解释量刑建议")
    print("- 自动驾驶: 解释驾驶决策")

def future_trends():
    """深度学习未来趋势"""
    print("\n" + "="*50)
    print("深度学习未来趋势")
    print("="*50)
    
    trends = {
        "大型语言模型 (LLMs)": {
            "现状": "GPT、BERT等取得突破性进展",
            "发展": "模型规模持续增长，能力不断提升",
            "应用": "对话系统、代码生成、内容创作",
            "挑战": "计算成本、数据质量、安全性"
        },
        
        "多模态学习": {
            "现状": "图像+文本、语音+文本等初步融合",
            "发展": "更多模态的深度融合和理解",
            "应用": "虚拟助手、智能搜索、内容生成",
            "挑战": "模态对齐、数据获取、计算复杂度"
        },
        
        "神经架构搜索 (NAS)": {
            "现状": "自动设计神经网络架构",
            "发展": "更高效的搜索算法和空间设计",
            "应用": "移动设备、专用硬件优化",
            "挑战": "搜索效率、泛化能力、解释性"
        },
        
        "边缘计算AI": {
            "现状": "模型压缩和移动端部署",
            "发展": "专用芯片、分布式推理",
            "应用": "物联网、智能手机、自动驾驶",
            "挑战": "资源限制、延迟要求、隐私保护"
        },
        
        "量子机器学习": {
            "现状": "理论探索和概念验证",
            "发展": "量子算法和硬件的成熟",
            "应用": "优化问题、量子化学、密码学",
            "挑战": "硬件限制、算法设计、噪声处理"
        },
        
        "可信AI": {
            "现状": "公平性、可解释性研究兴起",
            "发展": "更完善的评估和保障体系",
            "应用": "金融、医疗、司法等关键领域",
            "挑战": "标准制定、技术实现、监管平衡"
        }
    }
    
    for trend, info in trends.items():
        print(f"【{trend}】")
        print(f"现状: {info['现状']}")
        print(f"发展方向: {info['发展']}")
        print(f"应用前景: {info['应用']}")
        print(f"面临挑战: {info['挑战']}")
        print()
    
    print("技术发展方向:")
    print("- 更大规模: 模型参数和数据规模持续增长")
    print("- 更高效率: 训练和推理效率持续提升")
    print("- 更强泛化: 跨领域、跨任务的泛化能力")
    print("- 更好交互: 人机协作和人机对话")
    print("- 更安全可靠: 鲁棒性、安全性、可控性")
    print()
    
    print("社会影响趋势:")
    print("- 就业变革: 某些工作被自动化，新职业出现")
    print("- 教育改革: 个性化学习，教学方式变革")
    print("- 医疗进步: 精准医疗，药物发现加速")
    print("- 创作辅助: AI辅助艺术创作、内容生成")
    print("- 科学研究: AI加速科学发现和理论验证")

def main():
    """主函数"""
    print("深度学习高级主题")
    print("=" * 60)
    
    advanced_topics_introduction()
    generative_adversarial_networks()
    transformer_architecture()
    self_supervised_learning()
    graph_neural_networks()
    reinforcement_learning_basics()
    federated_learning()
    model_compression()
    explainable_ai()
    future_trends()
    
    print("\n" + "=" * 60)
    print("高级主题学习总结")
    print()
    print("您已经了解了深度学习的高级主题:")
    print("- GANs: 生成对抗网络的对抗训练机制")
    print("- Transformer: 基于注意力的序列建模")
    print("- 自监督学习: 从无标注数据中学习表示")
    print("- 图神经网络: 处理图结构数据")
    print("- 强化学习: 智能体与环境交互学习")
    print("- 联邦学习: 分布式隐私保护学习")
    print("- 模型压缩: 高效部署的关键技术")
    print("- 可解释AI: 理解和信任AI决策")
    print()
    print("继续学习建议:")
    print("- 选择感兴趣的方向深入研究")
    print("- 关注顶级会议和期刊的最新论文")
    print("- 参与开源项目和社区讨论")
    print("- 在实际项目中应用所学知识")
    print("- 保持对新技术的敏感度和学习热情")
    print()
    print("深度学习是一个快速发展的领域，")
    print("保持持续学习和实践是成功的关键！")

if __name__ == "__main__":
    main()