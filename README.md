# Awesome-GRPO

Implementations and Resources for GRPO and Its Variants.



## 1. GRPO — *Group Relative Policy Optimization*

> DAPO: An Open-Source LLM Reinforcement Learning System at Scale

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

优势函数：$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}\{R_i\}_{i=1}^G)}$

目标函数：$J_{\text{GRPO}}(\theta)
= \mathbb{E}_{(q,a)}\left[
\frac{1}{G} \sum_{i=1}^{G}
\min\big(
w_i(\theta)\hat{A}_i,,
\text{clip}(w_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i
\big)
\right]$

其中 $w_i(\theta) = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}$。





## 2. DAPO — *Dynamic Sampling Policy Optimization*

> DAPO: An Open-Source LLM Reinforcement Learning System at Scale

### (1) 核心思想

DAPO是GRPO的工程级强化版，通过改进采样与归一化机制提升稳定性与探索能力。

* **Decoupled Clip（解耦截断）**：将PPO的截断上下界分开设置，以防止熵塌缩。
* **Dynamic Sampling**：跳过全对或全错的组，提高有效梯度样本比。
* **Token-Level Averaging**：损失在所有token上平均，而非“先序列后batch”双层平均，降低方差。
* **Overlong Penalty**：惩罚过长输出，减少reward噪声。

### (2) 训练流程

1. 同GRPO，采样多条响应序列并计算reward。
2. 动态剔除无信息batch。
3. 对所有token计算同一序列的统一advantage。
4. 对所有token统一平均loss并更新。

### (3) 关键公式

与GRPO相似，但损失归一化不同





## 3. Dr.GRPO

> Understanding R1-Zero-Like Training: A Critical Perspective

### (1) 核心思想

* 在GRPO的基础上加入**KL散度正则项**，显式约束策略与参考策略间的偏移。
* 实质为**KL约束的GRPO**，兼顾稳定性与探索性。
* 目标函数类似GRPO + 罚项：$-\beta , D_{KL}(\pi_\theta \Vert \pi_\text{ref})$。

### (2) 训练流程

1. 同GRPO，采样组序列并计算reward。
2. 计算组内advantage。
3. 计算KL散度项。
4. 以加权和形式更新策略。

### (3) 关键公式








## 4. GTPO — *Group Token Policy Optimization*

> （Tan et al., *GTPO & GRPO-S, 2025*）

### (1) 核心思想

* 解决**coarse-grained credit assignment**（粗粒度信用分配）问题。
* 引入**Dynamic Entropy Weighting**机制：用策略熵衡量“认知努力”。
* 在token级别根据熵动态调节reward，实现精细化credit assignment。

### (2) 训练流程

1. 从prompt采样 (G) 个序列，得到reward (r_i)。
2. 计算每个token的policy entropy (H_{i,t})。
3. 成功序列中高熵token获得奖励加成，失败序列中低熵token受罚。
4. 归一化token reward形成token级advantage。
5. 应用PPO式clipped loss更新策略。

### (3) 关键公式

**成功序列token奖励：**


**失败序列token奖励：**

**token级目标函数：**






## 5. GRPO-S — *Sequence-Level Entropy-Weighted GRPO*

> （Tan et al., *same paper, 2025*）

### (1) 核心思想

* GTPO的轻量化版本。
* 保留**熵加权奖励机制**，但在**序列级**（非token级）操作，计算开销低。
* 奖励取决于序列的平均token熵。

### (2) 训练流程

1. 计算每个序列平均熵 (\hat{H}_i)。
2. 以平均熵调节整个序列奖励（成功则加成，失败则惩罚）。
3. 用该加权reward计算组内advantage并更新。

### (3) 关键公式

**序列级奖励重塑：**

**目标函数：**

## 





