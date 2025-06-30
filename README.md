### 1 | 总体思路：把 **“确定性符号推理”** 当作 RL 环境，LLM 只做 **“高层决策 + 结构整理”**

```
          ┌────────┐  (1) JSON / S-exp tactic list
LLM θ ───▶│ Lean /  │─────────────────────────────────────┐
  ▲       │Connect++│                                     │
  │(5) ∇θ │  API    │ (2) 结果：新子目标集 + 证明树 Δ + AST │
  │       └────────┘                                     ▼
  │            ↑ (3) 偏确定性状态 s′=(Goals,ProofGraph)    RL 逻辑-环境
  │            │                                          计算奖励 R(s,a)
  └────────────┴───────────────────────────────────────────┘
```

* **LLM 输出**：不是自由文本，而是 *结构化* tactic 序列 / Lean 命令 S-表达式；
  例：`["intro", "rw [mul_comm]", "apply lemma_X"]`
  ——确保可直接送进 Lean / Connect++ API， **执行结果唯一且可验证**。
* **符号环境**（Lean/Connect++）：

  * 100% 确定性：同一输入 → 同一子目标集 / 证明树；
  * 返回数据已结构化（JSON / protobuf）：
    `{"goals": [...], "proof_graph": [...], "cost": n}`
  * **LLM 不再自行“理解”逻辑**，它只需依据结构化 state 决策下一组 tactcs。
* **RL**：在 *结构化状态–动作* 空间上做策略学习；奖励 **直接由符号属性计算**，无需再对文本做概率评估。

---

### 2 | Lean / Connect++ 的 **Tool-Use 接口**（建议)

| 功能              | 接口设计                                           | 返回字段                                                 |
| --------------- | ---------------------------------------------- | ---------------------------------------------------- |
| `apply_tactics` | `POST /tactic` `{"state_id":…, "tactics":[…]}` | 新 `state_id`, `goals`, `proof_graph`, `valid` (bool) |
| `get_ast`       | `GET /node/{id}`                               | Lean AST JSON（用于后续压缩奖励）                              |
| `auto_simplify` | `POST /simplify`                               | 简化后 AST + 代价（作 reward）                               |
| `check_prereq`  | `POST /premise`                                | 是否满足前提；缺失列表                                          |

LLM 侧模板：

```json
{
  "action_type":"TacticList",
  "payload":["intro","rw [lemma1]","apply lemma2"]
}
```

该 JSON 经中间层直接喂 Lean。

---

### 3 | 用 **符号属性直接做 Reward** （不依赖概率 token）

| 奖励子项            | 纯符号定义（Lean / Connect++ 提供）                                | 公式                                                                                 |   |         |                              |                                               |    |                             |   |    |                  |    |
| --------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------- | - | ------- | ---------------------------- | --------------------------------------------- | -- | --------------------------- | - | -- | ---------------- | -- |
| **子目标减少**       | \`Δ                                                       | Goals                                                                              | = | G\_prev | -                            | G\_next                                       | \` | (\mathcal{P}= \dfrac{\Delta | G | }{ | G\_{\text{prev}} | }) |
| **依赖已满足**       | `missing_premises = ∅`                                    | $\mathcal{K}_{\text{pre}} = 1_{\{missing=∅\}}$                                     |   |         |                              |                                               |    |                             |   |    |                  |    |
| **推理成本**        | Lean cost 计数（rewrite 步数、search depth）                     | (\mathcal{U} = -\text{cost} / \Delta                                               | G | )       |                              |                                               |    |                             |   |    |                  |    |
| **图压缩**         | \`                                                        | V                                                                                  | , | E       | `of`proof\_graph\` 与 gzip 长度 | $C_\beta = 1 - L_{\text{zip}}/L_{\text{raw}}$ |    |                             |   |    |                  |    |
| **逻辑距离**        | 目标节点与当前顶点在 proof-dependency 图中的最短路径                       | $\mathcal{G} = -d_{\text{graph}}(v_t, v_{\text{goal}})$                            |   |         |                              |                                               |    |                             |   |    |                  |    |
| **不确定性下降**      | Beta 方差：$a_g,b_g$ 由 “证明/失败” 计数器维护，无文本概率                   | $\mathcal{U}_\sigma = \sum_g (\text{Var}_{\text{prior}}-\text{Var}_{\text{post}})$ |   |         |                              |                                               |    |                             |   |    |                  |    |
| **语义对齐**（跨模态可选） | 若题干为自然语言，可对 Lean AST 进行 `pretty_print` 再做 SentenceBERT 嵌入 | $A_\gamma = -\|e(v_t)-e_{\text{goal}}\|_2$                                         |   |         |                              |                                               |    |                             |   |    |                  |    |

**要点**：所有计算只依赖 *符号树/图* + 计数器，**无需** token 概率或隐式语义。

---

### 4 | 图结构如何助力

1. **状态表示**

   * 节点：子目标 / 已证定理
   * 边：战术应用、引理依赖
     用 GNN 直接汇总成 $h_v$，作为值函数输入。
2. **逻辑距离** $d_{\text{graph}}$：
   最短路径 = “剩余推理深度”估计，可直接作 shaping。
3. **中心性** $\deg(\tau)$：
   从知识图谱抽出，作为动作先验 + 奖励。
4. **图压缩 $C_\beta$**：

   * 序列化边表，gzip；
   * 重复子树越多，压缩率越低 → 惩罚冗余。

---

### 5 | LLM 的角色与约束

| 阶段        | LLM 任务                                   | 说明                       |
| --------- | ---------------------------------------- | ------------------------ |
| **解析-规划** | 根据 JSON `goals` 选择一组 Lean tactics        | 决策空间显式且离散，可直接 softmax 采样 |
| **格式化输出** | 生成结构化 JSON，保证语法可被 Lean 接口解析              | 避免自由文本幻觉                 |
| **工具调用**  | 特殊 token `<CALL:WOLFRAM f(x)>` 触发外部计算    | 结果以 JSON 回传，供下一步决策       |
| **学习方式**  | PPO 优化 **token→tactic 映射** 概率；符号奖励直接反传参数 | LLM 只学“策略”，不学底层逻辑规则      |

---

### 6 | 总结

* Lean / Connect++ **封装为确定性环境** → 奖励直接由符号属性计算 → **无语言噪声，梯度更稳定**。
* 图结构让“逻辑距离、中心性、压缩率”这些 **可解释指标** 成为 RL shaping 信号。
* LLM 只需要学习如何组织信息、选择战术、调用工具；底层逻辑验证由符号系统保证。
* 这样的 **Tool-Use + RL** 设计兼具可控性（符号确定）与可学习性（策略梯度），有效提升大型模型的可验证推理能力。

# 后面的强化学习函数设计需要与tool use找到一个平衡，目前讨论的是两个极端情况：上面的tool use符号化推导，下面的非符号化rl计算法则。实践过程中，找到借助LLM的概率的发散思维，又能受符号化语言约束，是本项目的终极目标，要在实验中找到平衡点，目前我无法给出方案。

## TPTP 文件：告诉 Connect++ “问题陈述 + 公理库”；解析后得到子句矩阵并生成 proof state。

## Connect++：验证 LLM 动作是否“一步合法”，输出局部 proof 片段与统计信息。

## 本项目的embedding space
| 层级                                | 向量是谁                                 | 生成/训练方式                                                                     | 维度 d（典型） | 主要用途                                                                                          |        |
| --------------------------------- | ------------------------------------ | --------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------- | ------ |
| **① Proof-Graph Embedding**       | 整个 **状态节点** $v$：包含未闭合子目标、已用定理、局部 AST | **Laplacian-GNN / GraphTransformer** 在 proof-graph 上自监督（邻居预测 + contrastive） | 256–512  | 1. 潜势函数 $\phi(v)$<br>2. ValueNet $V_\psi(s)$<br>3. Bayes-Surprise 中的 (p(QED                   | v)) 估计 |
| **② Formula / Tactic Embedding**  | 单条定理 / tactic $\tau$ 或 AST 片段        | token-level Transformer（BPE over Lean syntax）+ mean-pool；SFT 于 mathlib 语料   | 128–256  | 1. Dirichlet-Reuse 先验统计<br>2. Action prior $P_\theta(v,\tau)$<br>3. 搜索时做 similarity retrieval |        |
| **③ Semantic Sentence Embedding** | “自然语言/符号混合”陈述 $e(\cdot)$             | **SentenceBERT-Lean**：SciBERT 初始化→ 在 (命题, 相似命题) 对上对比学习                      | 768      | 1. 语义对齐奖励 $\mathcal{A}_\gamma$<br>2. 题目-检索最近引理<br>3. 人工评估可读性                                  |        |

## 总奖励函数 $R^{\dagger}$ —— 完整公式与各子项计算细节

$$
\boxed{
\displaystyle
R^{\dagger}
=
\underbrace{
\bigl(
w_c\mathcal{C}
+w_p\mathcal{P}
+w_e\mathcal{E}
+w_f\mathcal{F}_{\text{DR}}
+w_n\mathcal{N}_{\text{fit}}
\bigr)
}_{\text{基础五项}}
\;
+\;
\underbrace{
\lambda_1\mathcal{S}_{\text{KL}}
+\lambda_2\mathcal{R}_{\alpha}
+\lambda_3\mathcal{C}_{\beta}
+\lambda_4\mathcal{A}_{\gamma}
+\lambda_5\mathcal{U}_{\sigma}
}_{\text{贝叶斯 - 信息 - 语义五项}}
\;
+\;
\underbrace{
w_k\mathcal{K}_c
-w_h\mathcal{H}_{\text{clause}}
+w_{\text{pre}}\mathcal{K}_{\text{pre}}
+w_u\mathcal{U}
+w_{\text{nov}}\mathcal{N}_{\text{BNG}}
+w_g\mathcal{G}
}_{\text{知识图谱 - 结构六项}}
}
$$

> **说明**
>
> * 所有子项在一次完整 roll-out 结束后汇总得到单标量 $R^{\dagger}$。
> * 各权重 $w_\bullet,\lambda_\bullet$ 在验证集上网格或贝叶斯优化。
> * 若某子项在当前实验不用，将其权重置 0。

---

### Ⅰ 基础五项

| 记号                                           | 公式                                                                                  | 计算方法                             | 作用                |                    |        |
| -------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------- | ----------------- | ------------------ | ------ |
| **$\mathcal{C}$ Correct**                    | $=\mathbf 1[\text{QED}]$                                                            | 终态成功=1，失败=0                      | 保证最终正确性           |                    |        |
| **$\mathcal{P}$ Progress**                   | $\displaystyle \frac{\Delta\#\text{open-goals}}{\#\text{open-goals}_{\text{prev}}}$ | 由 Connect++ 返回未闭合子目标数            | 稠密 shaping，奖励每步收敛 |                    |        |
| **$\mathcal{E}$ Elegance**                   | 手工规则：去根号+2，分子 ± +1，AST深度\<k给 +k′                                                    | 解析 AST 即可                        | 公式美感              |                    |        |
| **$\mathcal{F}_{\text{DR}}$** Dirichlet-freq | $\log(1+\text{count}(\text{formula}))$                                              | 在数学知识库离线统计                       | 复用高频套路            |                    |        |
| **$\mathcal{N}_{\text{fit}}$** Numeric-Fit   | (-                                                                                  | f\_{\text{LLM}}-f\_{\text{true}} | )                 | Wolfram/NumPy 验证数值 | 避免数值幻觉 |

---

### Ⅱ 贝叶斯-信息-语义五项

| 记号                                           | 公式                                                                                            | 实现细节                     | 含义        |
| -------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------ | --------- |
| **$\mathcal{S}_{\text{KL}}$** Bayes-Surprise | $D_{\!KL}\!\bigl(p_{\text{post}}\|\;p_{\text{prior}}\bigr)$                                   | 见下表三种 $p$ 估计             | 信息增益      |
| **$\mathcal{R}_{\alpha}$** Reuse-Factor      | $\displaystyle\log\frac{n_i+1}{\alpha_i+n_i}$                                                 | 每调用定理 $i$ 更新 $n_i$       | 抑制机械复用    |
| **$\mathcal{C}_{\beta}$** Proof-Compression  | $1-\dfrac{L_{\text{zip}}}{L_{\text{raw}}}$                                                    | 见下方两种“压缩方案”              | 简洁优雅      |
| **$\mathcal{A}_{\gamma}$** Semantic-Align    | $-\|e(v_t)-e_{\text{goal}}\|_2$                                                               | SentenceBERT / Lean-BERT | 朝目标收敛     |
| **$\mathcal{U}_{\sigma}$** Uncertainty-Red.  | $\displaystyle\sum_g\bigl[\text{Var}_{\text{prior}}(p_g)-\text{Var}_{\text{post}}(p_g)\bigr]$ | Beta-Bernoulli 计数        | 优先消除高方差目标 |

---

### Ⅲ 知识图谱-结构六项（没有细究，列为计划）

| 记号                                               | 公式                                                  | 计算方法               | 目的     |
| ------------------------------------------------ | --------------------------------------------------- | ------------------ | ------ |
| **$\mathcal{K}_c$** Centrality                   | $\deg(\tau)/\max_j\deg(\tau_j)$                     | 预计算 KG 度中心性        | 常用引理优先 |
| **$\mathcal{H}_{\text{clause}}$** Clause-Entropy | $-\!\sum_l p(l)\log p(l)$                           | 统计当前子句文字频率         | 抑制分支爆炸 |
| **$\mathcal{K}_{\text{pre}}$** Prereq-OK         | $\mathbf 1[\text{prereq}(τ)\subseteq v]$            | KG 查询前提集合          | 合法调用定理 |
| **$\mathcal{U}$** Clause-Utility                 | $\Delta\#\text{open-goals}/\text{len}(τ)$           | Connect++ 回传       | 单位成本收益 |
| **$\mathcal{N}_{\text{BNG}}$** Novelty-Gain      | $\log\frac{\alpha_i}{\alpha_i^{(0)}}-\mathcal{K}_c$ | Dirichlet 后验 − 中心性 | 激励冷门妙招 |
| **$\mathcal{G}$** Graph-Walk Align               | $-d_{\text{KG}}(\tau,\text{goal})$                  | KG 最短路径            | 大方向对齐  |

---

## 三种 $p_{\text{prior}},p_{\text{post}}$ 估计方案（用于 $\mathcal{S}_{\text{KL}}$）

| 方案              | 计算公式                                     | 数据来源 / 训练               | 适用阶段           |
| --------------- | ---------------------------------------- | ----------------------- | -------------- |
| **M1 GNN**      | $p = f_\phi(v)$                          | 图神经网络；用 (状态,QED标签) 监督微调 | 主训练            |
| **M2 kNN-Freq** | $p = \dfrac{\#\text{近邻成功}}{\#\text{近邻}}$ | 采样历史嵌入 + FAISS          | 冷启动 / baseline |
| **M3 BayesNet** | 结构化 $P(QED \mid V,\tau,L_i)$             | EM 学习或MLE               | 分析 / 可解释对比     |


##  **公式**

$$
\mathcal{S}_{\text{KL}} = D_{\text{KL}}\left(p_{\text{post}}(\text{QED} \mid v') \,\|\, p_{\text{prior}}(\text{QED} \mid v)\right)
$$

###  **符号解释**

| 符号                                    | 含义                              |
| ------------------------------------- | ------------------------------- |
| $v$                                   | 当前证明状态（还未执行动作）                  |
| $v'$                                  | 执行动作后的状态（新状态）                   |
| $p_{\text{prior}}(\text{QED} \mid v)$ | 原状态下，成功完成整个证明的预测概率（由 GNN/贝叶斯估计） |
| $p_{\text{post}}(\text{QED} \mid v')$ | 执行该动作后，成功完成整个证明的预测概率            |
| $D_{KL}(P \,\|\, Q)$                  | KL散度，衡量两个概率分布 $P$ 与 $Q$ 的差异     |

---

##  **直觉解释**


* 在原状态 $v$ 下，我们对完成证明的信心是 $p_{\text{prior}}$；
* 你执行了一个动作（比如调用某个引理，分解一个目标）后，进入新状态 $v'$，我们再重新估算成功概率 $p_{\text{post}}$；
* 如果两者差异大（KL散度大），说明这个动作带来了**强烈的信息更新**，要么让我们更接近成功（提升信心），要么证明这条路径不通（信心下降）；
* 这本身就是“有用的尝试”，比起毫无变化、走老路的行为要更值得奖励。



##  **实现方式**

你可以用一个小的 GNN 或贝叶斯模型来拟合：

$$
f_\phi(v) \rightarrow p(\text{QED})
$$

* 输入：当前证明状态 $v$（可以用图结构编码，如 DAG / proof tree）；
* 输出：成功概率估计值 $p \in [0,1]$；
* 对比 action 前后的预测值，求 KL 或绝对差作为奖励；
* 可以微调 GNN 参数，也可以冻结只作为评估器使用。


| 设置项       | 说明                                              |
| --------- | ----------------------------------------------- |
| **任务数据集** | miniF2F / MATH / LeanDojo 子集（保证包含长推理链）          |
| **模型结构**  | 使用统一 LLM（如 fine-tuned T5 或 GPT-style）           |
| **符号环境**  | Connect++ / Lean / 自定义 Python-checker           |
| **训练方式**  | PPO 或 GRPO（同一训练策略）                              |
| **评估指标**  | ① QED 成功率<br>② 每条路径平均长度<br>③ 平均 KL 分布<br>④ 收敛速度 |



#### M1: GNN 估计器

* 输入：当前 proof state 图结构 $v$
* 输出：标量 $p \in [0,1]$
* 模型：GCN / GAT / Relational GNN
* 训练方式：

  * 采样成功/失败轨迹
  * 将中间状态打标签（是否能走到 QED）
  * 用 BCE loss 训练一个 $f_\phi(v) \approx p$

---

#### M2: 状态频率统计 + kNN

* 特征提取：对 $v$ 提取固定维度 embedding（如子目标 token embedding 均值 + depth）
* 距离函数：Cosine 或 Euclidean
* 近邻采样：查找 FAISS 中近似相似状态集合 $\{v_i\}$
* 成功率估计：

$$
p(v) = \frac{1}{k} \sum_{i=1}^k \mathbf{1}[v_i \text{ 成功}]
$$

---

####  M3: 贝叶斯网络

* 建模变量：

  * $V$：状态节点（可分解为子目标 embedding）
  * $\tau$：动作节点（可离散化为类别）
  * $L_i$：被引用引理（可嵌入）
  * $Q$：最终成功与否（布尔变量）

* 学习算法：结构化贝叶斯网络 + MLE 或 EM 学习

* 预测方法：推断 $P(Q=1 \mid V, \tau, \{L_i\})$

---

## 两种 Proof-Compression 备选（用于 $\mathcal{C}_{\beta}$）
两种可选压缩方法：
压缩比越大说明此链条越冗余，不好

| 方案                           | $L_{\text{raw}}$ 与 $L_{\text{zip}}$ 定义                | 说明          |
| ---------------------------- | ----------------------------------------------------- | ----------- |
| **Option #2 AST**            | 序列化 AST token 列表后取字节数；gzip 压缩后再计长                     | 实现简单、语言无关   |
| **Option #3 Graph** **(默认)** | 将 proof-graph 边表 `[(u,v),...]` JSON 序列化后计长；gzip 压缩后取长 | 捕获结构重复，首选方案 |
LLM → 生成推理路径 → Connect++ 验证 → 结构/文本提取 → 计算压缩率 → 反馈 reward → PPO 更新 LLM 参数

---

## 每项解释与作用（汇总）

| 子项                            | 主要衡量 | 作用          |
| ----------------------------- | ---- | ----------- |
| $\mathcal{C}$                 | 成败   | 没有 QED 不给高分 |
| $\mathcal{P}$                 | 局部进度 | 提前分摊稀疏奖励    |
| $\mathcal{E}$                 | 句法美感 | 引导简洁表达      |
| $\mathcal{F}_{\text{DR}}$     | 引理频率 | 鼓励常用套路      |
| $\mathcal{N}_{\text{fit}}$    | 数值正确 | 避免幻觉数值      |
| $\mathcal{S}_{\text{KL}}$     | 信息增益 | 奖励有价值探索     |
| $\mathcal{R}_{\alpha}$        | 创新复用 | 抑制机械复用      |
| $\mathcal{C}_{\beta}$         | 描述长  | 促成短而精炼      |
| $\mathcal{A}_{\gamma}$        | 语义对齐 | 防跑题         |
| $\mathcal{U}_{\sigma}$        | 方差下降 | 先攻难点        |
| $\mathcal{K}_c$               | 图谱中心 | 优先高价值定理     |
| $\mathcal{H}_{\text{clause}}$ | 信息熵  | 控制搜索爆炸      |
| $\mathcal{K}_{\text{pre}}$    | 前提满足 | 合法调用        |
| $\mathcal{U}$                 | 单位收益 | 高效步骤        |
| $\mathcal{N}_{\text{BNG}}$    | 冷门成功 | 激励创新        |
| $\mathcal{G}$                 | 节点距离 | 大方向收敛       |

---

# 训练流程

1. **采样**：LLM + MCTS 生成整条证明。
2. **逐步记录**：成功/失败、子目标变化、AST/Graph 结构、调用次数、Beta 计数。
3. **计算每子项奖励**：按上表公式即时或回放计算。
4. **加权求和**得 $R^{\dagger}$。
5. **PPO/GRPO** 用优势 $\hat A=R^{\dagger}-V_\psi$ 回传梯度，更新 LLM。
---

# 负责探索的函数们
##  Dirichlet-Reuse：基于贝叶斯先验的引理使用奖励（比先验多了抑制，少了促进）

###  定义公式

$$
R_\alpha(i) = \log \left( \frac{n_i + 1}{\alpha_i + n_i} \right)
$$

* $n_i$：当前定理 $i$ 被调用的次数（来自当前 episode 或所有 episode 的调用历史）
* $\alpha_i$：先验使用强度，表示我们**原本对这个定理重要性的信念**（比如设为 5 是“弱先验”，设为 100 是“强信念它重要”）

---

###  背景解释：来自贝叶斯的 MAP 原则

在贝叶斯统计中，常用 **Dirichlet 分布** 作为**多项式选择行为的先验分布**：

* 假设我们有多个选项（这里是定理 i），我们希望估计选哪个最优。
* 如果只用频率 $n_i$，容易造成过拟合，模型会过度偏好频繁出现的引理。
* 所以我们加入 Dirichlet 先验：

$$
P(i) \propto \frac{n_i + \alpha_i}{\sum_j (n_j + \alpha_j)}
$$

**本设计**等价于：对这种贝叶斯概率进行一个对数变换，用作奖励：

$$
R_\alpha(i) \sim \log P(i) = \log \left( \frac{n_i + 1}{n_i + \alpha_i} \right)
$$

这是一种 **MAP（Maximum A Posteriori）激励策略**，目标是鼓励 LLM 在调用引理时：

> 不是谁用得多我就用谁，而是谁**在当前情境中既有先验价值又真正有用**。


###  奖励设计动机

####  惩罚无脑复用

* 若某定理 $i$ 已被过度调用（$n_i \gg \alpha_i$），则：

  $$
  R_\alpha(i) \to \log \left( \frac{n_i + 1}{n_i + \alpha_i} \right) \to \log \left( \frac{1}{\alpha_i} \right)
  $$

  → 变负数，意味着重复使用会“扣分”。

####  奖励“探索性调用”

* 如果某个定理还很少被使用（$n_i \ll \alpha_i$），即使它有一定先验重要性，也会被鼓励试用一次。

  $$
  n_i = 0, \quad \Rightarrow \quad R_\alpha(i) = \log \left( \frac{1}{\alpha_i} \right)
  $$

  → 这是合理的：LLM 可以探索低频定理，形成更**多样化的推理路径**。

####  对有先验信念的重要引理容忍度更高

* 如果 $\alpha_i$ 很大（比如某些常用引理），即使 $n_i$ 高，也不会马上被惩罚（因为我们本来就信它重要）。


###  实现建议

| 模块     | 实现方式                                              |
| ------ | ------------------------------------------------- |
| 引理使用记录 | 对每个定理 i 维护一个调用次数 $n_i$，每轮 RL 后更新                  |
| 奖励计算   | 每次 LLM 选用某定理时，根据当前 $n_i$ 和预设 $\alpha_i$ 计算 log 奖励 |
| 先验设定   | 可设为：弱先验统一值 $\alpha_i = 5$，或根据图谱中心度动态设定（中心引理先验大）   |



---



## **Uncertainty Reduction（不确定性下降）**

### 公式回顾：

$$
U_{\sigma} = g \cdot \sum_{g} \left( \text{Var}_{\text{prior}}(p_g) - \text{Var}_{\text{post}}(p_g) \right)
$$

其中子目标 $g$ 的不确定性定义为：

$$
\text{Var}(p_g) = \frac{a_g b_g}{(a_g + b_g)^2 (a_g + b_g + 1)}
$$

* $a_g$：该子目标的历史**成功次数**
* $b_g$：该子目标的历史**失败次数**
* $p_g \sim \text{Beta}(a_g, b_g)$：贝塔先验下估计该子目标证明成功率

---

### **含义：**

* 若执行当前 action 后，某些子目标的贝塔不确定性显著下降，则说明这一步具有**信息探索价值**。
* 直觉上等价于主动学习中的“信息获取量”指标。
* 不确定性下降 → 模型更清楚哪些路径有望成功 → 用于引导策略的探索方向。

---

### **实现方法：**

1. 每个子目标 $g$ 初始化 Beta 分布：$a_g = 1, b_g = 1$。
2. 每次尝试该子目标：

   * 成功：$a_g \leftarrow a_g + 1$
   * 失败：$b_g \leftarrow b_g + 1$
3. 使用 Beta 分布计算尝试前后的方差变化。
4. 对每一轮 action 汇总多个子目标的不确定性下降值。
### **补充我为什么要用beta分布**                                                                                    
1. 子目标能否最终被证明——本质是 **Bernoulli 成功/失败**。Beta 分布正好是 Bernoulli 成功概率 $p_g\in[0,1]$ 的共轭先验。                               
2. 共轭性质 ⇒ 只需把 **成功次数 $a_g$** 和 **失败次数 $b_g$** 逐步加 1，就得到后验 $\text{Beta}(a_g,b_g)$。**O(1) 更新**、无梯度反向传播负担。             



# 判断与目标有多远，目前得了多少分了（完成了多少了）

## 1.1 Correct（返回0/1，判断是否结束）

$$
\mathsf{Correct} = \mathbf{1}\{\text{All goals closed by Connect++}\}
\tag{2}
$$

* 若 roll-out 终点图 `v_T` 无未证明子目标 ⇒ 返回 1
* 否则 0
## 1.2 Progress  (潜势 ϕ-shaping)（3个因素，多少子目标没有闭合，每个子目标有多复杂，GNN还有多少步收敛，判断这条路径潜在价值）

$$
\phi(v)\;=\;\text{\#open\_goals}(v)\quad(\text{越小越好})
$$

$$
\mathsf{Progress}
=\frac{\phi(v_{t})-\phi(v_{t+1})}
      {\max\bigl\{1,\;\phi(v_{t})\bigr\}}
\in[-1,1]
\tag{3}
$$

* 对整条路径取 **平均** 或 **累计求和** 皆可；
* 这就是 Ng 等人提出的 **potential-based shaping**：
  $r' = r_{\text{env}} + \gamma[\phi(v_{t+1})-\phi(v_t)]$。
## 1.3 Elegance (句法美学 ψ)

令公式的 AST 抽取下列特征（只是部分列举，需要我日后设计一个非常非常大的表格来map各种数值状态的得分，需要一些统计，可以是一个子课题）

| 特征      | 记号                | 评分函数                            | 举例 (奖励)                                           |   |   |
| ------- | ----------------- | ------------------------------- | ------------------------------------------------- | - | - |
| 去根号     | $b_{\sqrt{\;}}$   | 若本步将 $\sqrt{·}$ 完全消除 → +2       | (\sqrt{x^2}!\to!                                  | x | ) |
| ± 转移到分子 | $b_{\pm}$         | 每出现一次 → +1                      | $\frac{a\pm b}{c}$ 优于 $\frac{a}{c}\pm\frac{b}{c}$ |   |   |
| AST 深度  | $d_\text{ast}$    | 奖励 = $\max(0,\,k-d_\text{ast})$ | $k=4$                                             |   |   |
| 因式分解成功  | $b_\text{factor}$ | +δ (默认 1)                       | $x^2-1\to(x-1)(x+1)$                              |   |   |

综合得

$$
\psi(v)
= 2\,b_{\sqrt{\;}} + 1\,b_{\pm}
  + (k - d_\text{ast})_{+}
  + \delta\, b_\text{factor}
\tag{4}
$$

$$
\mathsf{Elegance}
=\sum_{t=0}^{T-1} \psi(v_{t+1})
\tag{5}
$$
## **Semantic Alignment（中间断言的语义对齐）**（我不喜欢，这个方法与符号推理无关）
### 公式回顾：
$$
A_{\gamma} = -\| e(v_t) - e_{\text{goal}} \|^2
$$

* $e(v_t)$：当前中间断言 $v_t$ 的句向量 embedding。
* $e_{\text{goal}}$：目标命题（QED）的句向量 embedding。
* $\| \cdot \|^2$：表示向量的欧式距离平方（L2 距离）。

### **含义：**

* 该公式衡量的是中间断言与最终目标的**语义相似度**。
* 奖励值越大（即距离越小，负值绝对值越小），说明当前步骤在**语义上更接近最终目标**。
* 如果 LLM 给出的中间引理（例如“设 $x \in G$，则 $x \cdot e = x$”）和目标“证明 $G$ 是群”具有相似的语义指向，则会得到正向强化。

### **实现方法可选项：**

1. 使用句向量模型，如：

   * **SciBERT**：适合数学和科学领域文本。
   * **Lean-BERT**：专门训练在 Lean 语言库的 transformer encoder。
   * **OpenAI text-embedding-ada 或 Qwen-embedding** 也可用于初步实验。
2. 将当前推理节点 $v_t$ 和目标语句文本分别转为 embedding。
3. 计算 L2 距离后取负号作为 reward。
4. 可按 step-wise 奖励方式，或累计最终路径平均语义距离。

---
## 1.4 NumericFit (数值一致性)（我不喜欢，在证明结果可以估算的情况下，我们思考的方向是否能接近要证明的结果，但是有不好的地方，容易陷入local best）

当连接步骤要求数值验证（如洛必达极限）

$$
\mathsf{NumericFit}
=-\frac1N \sum_{i=1}^{N}
     \frac{|f^{(\text{LLM})}_i - f^{(\text{true})}_i|}
          {1+|f^{(\text{true})}_i|}
\;\;\in[-1,0]
\tag{7}
$$

* **归一化**：误差越大越负；上限 0 表示完全匹配。
* 没有数值环节则该项为 0。

---

# 归一化与梯度平滑

1. **把 Progress、NumericFit 限制在 \[-1,1]**，避免梯度爆炸。
2. **Elegance、Frequency** 可能随路径长度线性增长，可在 PPO 里
   把 reward **除以路径深度 $T$** 或 **使用 reward / (1+T)**。
3. **λ-clip**：最终将 $R$ clip 在 $[-10,\,10]$。实践证明 PPO 更稳。

