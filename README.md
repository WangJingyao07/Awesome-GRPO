# Awesome-GRPO

Implementations and Resources for GRPO and Its Variants.





## 1. GRPO — *Group Relative Policy Optimization*

> （Shao et al., *DeepSeekMath, 2024*）

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

**优势函数：**
$$
\hat{A}*i = \frac{r_i - \text{mean}({r_k}*{k=1}^G)}{\text{std}({r_k}_{k=1}^G)}
\tag{1}
$$
**目标函数：**
$$
J_{\text{GRPO}}(\theta)
= \mathbb{E}\left[
\frac{1}{G} \sum_{i=1}^{G}
\min\big(
w_i(\theta)\hat{A}*i,,
\text{clip}(w_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}*i
\big)
\right]
\tag{2}
$$

其中 $w_i(\theta) = \frac{\pi*\theta(o_i|q)}{\pi*{\theta_\text{old}}(o_i|q)}$。





## 2. DAPO — *Dynamic Sampling & Decoupled Clip Policy Optimization*

> （Yu et al., *DAPO: Open-Source RL at Scale, 2025*）

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

与GRPO相似，但损失归一化不同：

$$
J_{\text{DAPO}}(\theta)
= \mathbb{E}\left[
\frac{1}{\sum_i |o_i|}\sum_{i,t}
\min\big(w_{i,t}A_i,, \text{clip}(w_{i,t},1-\epsilon,1+\epsilon)A_i\big)
\right]
\tag{3}
$$

其中 $A_i$ 仍为统一序列级优势。

DAPO改进了损失计算的**归一化层次**，但**未改变奖励内容**。





## 3. Dr.GRPO — *Divergence-Regularized Group Relative Policy Optimization*

> （用于DeepSeek-R1, 2025）

### (1) 核心思想

* 在GRPO的基础上加入**KL散度正则项**，显式约束策略与参考策略间的偏移。
* 实质为**KL约束的GRPO**，兼顾稳定性与探索性。
* 目标函数类似GRPO + 罚项：(-\beta , D_{KL}(\pi_\theta \Vert \pi_\text{ref}))。

### (2) 训练流程

1. 同GRPO，采样组序列并计算reward。
2. 计算组内advantage。
3. 计算KL散度项。
4. 以加权和形式更新策略。

### (3) 关键公式

[
J_{\text{Dr.GRPO}}(\theta)
= \mathbb{E}!\left[
\frac{1}{G}\sum_i
\min\big(w_iA_i,,\text{clip}(w_i,1-\epsilon,1+\epsilon)A_i\big)
\right]

* \beta, D_{\text{KL}}!\left[\pi_\theta(\cdot|q),|,\pi_{\text{ref}}(\cdot|q)\right]
  \tag{4}
  ]
  其中 (\beta) 控制保守性与探索性权衡。
  → 该形式后来被广泛用作**policy regularization baseline**。

---

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
[
\tilde{r}^+*{i,t} = \alpha_1 r_i + \alpha_2 \frac{H*{i,t}}{\sum_{k=1}^{n} H_{k,t}}\cdot d_t
\tag{5}
]

**失败序列token奖励：**
[
\tilde{r}^-*{j,t} = \alpha_1(-1) + \alpha_2 \frac{1/H*{j,t}}{\sum_{k=1}^{m} (1/H_{k,t})}\cdot h_t (-1)
\tag{6}
]

**token级目标函数：**
[
J_{\text{GTPO}}(\theta)
= \mathbb{E}!\left[
\frac{1}{\sum_k |o_k|}!!
\sum_{i,t}
\min(w_{i,t}\tilde{A}*{i,t},
\text{clip}(w*{i,t},1-\epsilon,1+\epsilon)\tilde{A}_{i,t})
\right]
\tag{7}
]

> ✅ 首次在GRPO框架中实现**真正的token-level信用分配**。

---

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
[
\hat{r}^+_i = \beta_1 r_i + \beta_2 \frac{\hat{H}_i}{\sum_k \hat{H}_k} \cdot n
,\quad
\hat{r}^-_j = \beta_1(-1) + \beta_2 \frac{1/\hat{H}_j}{\sum_k (1/\hat{H}_k)} \cdot m(-1)
\tag{8}
]

**目标函数：**
[
J_{\text{GRPO-S}}(\theta)
= \mathbb{E}!\left[
\frac{1}{G}\sum_i
\min(\hat{w}_i\hat{A}_i,
\text{clip}(\hat{w}_i,1-\epsilon,1+\epsilon)\hat{A}_i)
\right]
\tag{9}
]

其中：
[
\hat{w}*i = \frac{1}{|o_i|}\sum_t \frac{\pi*\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}|q,o_{i,<t})}
\tag{10}
]

> ✅ 兼顾**性能提升与计算效率**，实证上在AIME等推理基准上超越DAPO与GRPO。

## 





