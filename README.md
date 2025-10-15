# Awesome-GRPO

Implementations and Resources for GRPO and Its Variants.



## 1. GRPO — *Group Relative Policy Optimization*

> DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

### (1) 核心思想

* 属于value-function-free 的强化学习范式。
* 每次从同一prompt下采样一组响应序列，以该组平均reward作为baseline，计算相对优势。
* 避免训练value model，大幅简化PPO流程。

### (2) 训练流程

1. 对prompt $q$ 从策略 $\pi_\theta$ 采样 $G$ 个序列 ${o_1, …, o_G}$。
2. 对每个序列计算最终奖励 $r_i \in {0,1}$（正确为1，错误为0）。
3. 用组内平均reward作为baseline。
4. 对每个序列计算相对优势 $\hat{A}_i$。
5. 应用PPO样式clipped目标函数更新策略。

### (3) 关键公式

![image-20251015112057918](C:\Users\王婧瑶\AppData\Roaming\Typora\typora-user-images\image-20251015112057918.png)

![image-20251015112139988](C:\Users\王婧瑶\AppData\Roaming\Typora\typora-user-images\image-20251015112139988.png)



## 2. DAPO — *Dynamic Sampling Policy Optimization*

> DAPO: An Open-Source LLM Reinforcement Learning System at Scale

### (1) 核心思想

相比GRPO，DAPO改进采样与归一化机制提升稳定性与探索能力，包括：

#### 1）**Decoupled Clip**

**将截断上下界分开设置，以防止熵塌缩。**

对称 clipping 意味着：

* 当 $\rho_i > 1+\epsilon$：惩罚过度更新；
* 当 $\rho_i < 1-\epsilon$：抑制过度降低概率。

这样做在小规模强化学习中没问题，但在 LLM 的序列级策略优化 中，可能导致：

- Reward 信号偏斜：大语言模型的 reward 分布极不对称（高分样本稀少，低分样本密集）。对称 clip 会导致对高 reward 样本的更新被截断太早（欠优化）。

- 梯度方向失衡：当 $A_i > 0$ 的样本更容易被 clip 到上限 $1+\epsilon$，而 $A_i < 0$ 的样本通常还没到下限 $1-\epsilon$，导致梯度整体被高 reward 样本“压扁”，收敛慢、甚至不稳定。

换句话说，对称 clip 对好样本惩罚太重，对坏样本惩罚太轻。

#### **2）Dynamic Sampling**

**跳过全对或全错的组，提高有效梯度样本比。**

当某些提示的准确度等于 1 时，现有的 RL 算法存在梯度递减问题。

例如，对于 GRPO，如果特定提示的所有输出都正确并获得相同的奖励，则该组的结果优势为零。

零优势导致零策略梯度，缩小幅度并增加批次梯度的噪声灵敏度，从而降低样本效率。

#### **3）Token-Level Averaging**

**损失在所有token上平均，而非“先序列后batch”双层平均，降低方差。**

原始的GRPO算法采用样本级损失计算，它首先对每个样本内的令牌损失进行平均，然后聚合样本之间的损失。在这种方法中，每个样本在最终损失计算中被分配相同的权重。

- reward 是 sequence-level 的，但更新是基于 token-level 的概率比率计算的；若按 sequence 平均，则短序列和长序列的梯度被等权处理；这会使长序列的每个 token 的梯度被缩小，导致长序列的优化信号被稀释。
- 长序列的梯度累积更多 token 噪声；短序列的梯度方差较小；最终 batch 平均时，长序列梯度被缩放后贡献不足 → 梯度方向偏向短序列样本。



### (2) 训练流程

1. 同GRPO，采样多条响应序列并计算reward。
2. 动态剔除无信息batch。
3. 对所有token计算同一序列的统一advantage。
4. 对所有token统一平均loss并更新。

### (3) 关键公式

![image-20251015112201428](C:\Users\王婧瑶\AppData\Roaming\Typora\typora-user-images\image-20251015112201428.png)





## 3. Dr.GRPO

> Understanding R1-Zero-Like Training: A Critical Perspective

### (1) 核心思想

* 在GRPO的基础上加入**KL散度正则项**，显式约束策略与参考策略间的偏移。
* 实质为**KL约束的GRPO**，兼顾稳定性与探索性。
* 目标函数类似GRPO + 惩罚项。

### (2) 训练流程

1. 同GRPO，采样组序列并计算reward。
2. 计算组内advantage。
3. 计算KL散度项。
4. 以加权和形式更新策略。

### (3) 关键公式








## 4. GTPO — *Group Token Policy Optimization*

> GTPO AND GRPO-S: TOKEN AND SEQUENCE-LEVELREWARD SHAPING WITH POLICY ENTROPY

### (1) 核心思想

* 解决**coarse-grained credit assignment**（粗粒度信用分配）问题。
* 引入**Dynamic Entropy Weighting**机制：用策略熵衡量“认知努力”。
* 在token级别根据熵动态调节reward，实现精细化credit assignment。

### (2) 训练流程

1. 从prompt采样 (G) 个序列，得到reward $r_i$。
2. 计算每个token的policy entropy $H_{i,t}$。
3. 成功序列中高熵token获得奖励加成，失败序列中低熵token受罚。
4. 归一化token reward形成token级advantage。
5. 应用PPO式clipped loss更新策略。

### (3) 关键公式

**成功序列token奖励：**


**失败序列token奖励：**

**token级目标函数：**






## 5. GRPO-S — *Sequence-Level Entropy-Weighted GRPO*

> GTPO AND GRPO-S: TOKEN AND SEQUENCE-LEVELREWARD SHAPING WITH POLICY ENTROPY

### (1) 核心思想

* GTPO的轻量化版本。
* 保留**熵加权奖励机制**，但在**序列级**（非token级）操作，计算开销低。
* 奖励取决于序列的平均token熵。

### (2) 训练流程

1. 计算每个序列平均熵 $\hat{H}_i$。
2. 以平均熵调节整个序列奖励（成功则加成，失败则惩罚）。
3. 用该加权reward计算组内advantage并更新。

### (3) 关键公式

**序列级奖励重塑：**

**目标函数：**

## 





