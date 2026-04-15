# FedMASS 方法设计文档

> **项目目标**：基于 FedAMM 源码框架，设计一个面向 **联邦半监督医学图像分割 + 任意模态缺失** 的新方法框架。  
> **核心任务**：在每个客户端仅有少量完整模态标注数据、大量缺失模态无标注数据的情况下，实现可靠伪标签生成、跨模态语义迁移和跨客户端稳健聚合。

---

# 0. 方法暂定名称

## FedMASS

**Federated Missing-modality Anchor-guided Semi-supervised Segmentation**

中文解释：

> FedMASS 是一个面向联邦半监督任意模态缺失医学图像分割的框架。它以 FedAMM 的多模态缺失联邦分割框架为基础，引入完整模态语义锚点、缺失模式感知伪标签筛选和可靠性感知聚合机制，使少量标注数据能够有效指导大量无标注缺失模态样本的学习。

---

# 1. 任务设定

## 1.1 原始 FedAMM 任务

FedAMM 主要解决的是：

> 联邦学习场景下，客户端样本存在任意模态缺失时，如何训练一个能够适应 15 种非空模态组合的脑肿瘤分割模型。

BraTS 四个 MRI 模态包括：

- FLAIR
- T1
- T1ce
- T2

由于每个样本可能拥有任意非空模态组合，因此一共有：

$$
2^4 - 1 = 15
$$

种模态组合。

---

## 1.2 新任务设定

在 FedAMM 的基础上，进一步引入半监督设定。

每个客户端 $k$ 的数据被划分为：

$$
D_k = D_k^l \cup D_k^u
$$

其中：

$$
D_k^l = \{(x_i^{full}, y_i)\}
$$

表示 **少量有标注完整四模态数据**，例如 20%。

$$
D_k^u = \{x_j^s\}
$$

表示 **大量无标注缺失模态数据**，例如 80%。

其中：

- $x_i^{full}$：完整四模态样本；
- $y_i$：人工标注分割标签；
- $x_j^s$：无标注样本；
- $s$：当前样本的模态组合，$s \in \{1,\dots,15\}$。

---

## 1.3 关键挑战

该任务同时存在三类困难：

### 1. 标注稀缺

医学图像标注成本高，每个客户端只有少量标注样本。  
如果只用这部分数据做普通监督训练，模型容易过拟合，且无法充分利用大量无标注数据。

### 2. 任意模态缺失

无标注数据可能存在不同模态组合。  
不同缺失模式下，模型预测可靠性不同。

例如：

- 四模态样本信息最完整，预测相对可靠；
- 单模态样本信息最少，预测最不稳定；
- 双模态、三模态样本处于中间状态。

### 3. 联邦异构

不同客户端的数据分布不同，模态组合分布也不同。  
当 Dirichlet 异构参数 $\alpha$ 较小时，不同客户端看到的模态组合差异会更明显。

这会导致：

- 客户端模型更新方向不一致；
- 伪标签质量差异大；
- 聚合时容易产生 client drift；
- 全局模型对某些缺失模式泛化能力不足。

---

# 2. 整体方法思路

FedMASS 的核心思想是构建一个闭环：

```text
少量完整模态标注数据
        ↓
构建可靠 full-modal semantic anchor
        ↓
指导无标注缺失模态样本生成和筛选伪标签
        ↓
从高置信伪标签区域提取 unlabeled prototype
        ↓
服务端进行可靠性感知参数聚合和语义原型聚合
        ↓
更新全局模型与语义原型库
        ↓
反过来提升下一轮伪标签质量
```

可以概括为：

$$
\text{Reliable labeled anchors}
\rightarrow
\text{Reliable pseudo labels}
\rightarrow
\text{Reliable aggregation}
\rightarrow
\text{Better global semantics}
$$

---

# 3. 三个主要创新点概览

FedMASS 包含三个主要创新点：

## 创新点一：Full-modal Anchor Supervision

中文名称：

> 完整模态语义锚点监督

解决问题：

> 如何高效利用本地少量完整模态标注数据？

核心思想：

> 少量完整四模态标注数据不仅用于普通监督分割，还用于构建类别级 full-modal semantic anchor，并通过完整模态到单模态/缺失模态的蒸馏增强模型对缺失模态的适应能力。

---

## 创新点二：Mask-aware Distilled Pseudo-label Filtering

中文名称：

> 缺失模式感知的蒸馏式伪标签筛选

解决问题：

> 如何在不同模态缺失模式下尽可能获得可靠伪标签？

核心思想：

> 对无标注缺失模态样本，结合 EMA teacher 预测、全局 full-modal anchor 相似度以及当前缺失模式，生成 refined pseudo-label；再通过缺失模式自适应置信度、teacher-student 一致性和 anchor agreement 三重过滤，筛除噪声伪标签。

---

## 创新点三：Reliability-aware Semantic Aggregation

中文名称：

> 可靠性感知的语义聚合

解决问题：

> 如何在客户端之间进行有效聚合，实现稳健语义对齐与分割优化？

核心思想：

> 服务端不再只做普通 FedAvg，而是综合客户端有标注监督稳定性、无标注伪标签可靠性、anchor 对齐程度和模态出现比例，对模型参数、模态专属 encoder 和语义原型库进行可靠性感知聚合。

---

# 4. 创新点一：Full-modal Anchor Supervision

## 4.1 设计动机

在当前实验设定中，每个客户端只有少量完整模态标注数据。

如果只用这些数据计算：

$$
\mathcal{L}_{ce} + \mathcal{L}_{dice}
$$

那么这些标注数据的作用只停留在普通监督层面，利用效率较低。

但完整四模态样本具有一个非常重要的优势：

> 它们包含最完整的模态信息，能够提供最稳定、最可靠的语义表征。

因此，完整模态标注样本应当承担三重作用：

1. 训练基础分割能力；
2. 教会单模态/缺失模态分支学习完整模态语义；
3. 构建全局共享的类别级语义锚点。

---

## 4.2 有标注数据输入

对于客户端 $k$ 的一个有标注完整模态样本：

$$
(x_l^{full}, y_l) \in D_k^l
$$

其中：

$$
x_l^{full} =
\{x^{FLAIR}, x^{T1}, x^{T1ce}, x^{T2}\}
$$

输入 FedAMM/RFNet 主干后，得到：

$$
p_l^{full}, f_l^{full}
=
f_\theta(x_l^{full})
$$

其中：

- $p_l^{full}$：完整模态融合预测；
- $f_l^{full}$：完整模态融合特征；
- $y_l$：真实标注。

---

## 4.3 基础完整模态监督分割

首先计算普通监督分割损失：

$$
\mathcal{L}_{seg}^{full}
=
\mathcal{L}_{ce}(p_l^{full}, y_l)
+
\mathcal{L}_{dice}(p_l^{full}, y_l)
$$

该损失保证模型具备基本的分割能力。

---

## 4.4 完整模态到单模态的蒸馏

为了让模型在缺失模态条件下也能保持稳定预测，对同一个完整模态样本主动构造单模态视图：

$$
x_l^m, \quad m \in \{FLAIR,T1,T1ce,T2\}
$$

例如：

```text
完整输入：FLAIR + T1 + T1ce + T2
单模态视图 1：只保留 FLAIR，其余模态置零
单模态视图 2：只保留 T1，其余模态置零
单模态视图 3：只保留 T1ce，其余模态置零
单模态视图 4：只保留 T2，其余模态置零
```

模型分别得到单模态预测：

$$
p_l^m = f_\theta(x_l^m)
$$

然后使用完整模态预测 $p_l^{full}$ 作为 teacher，单模态预测 $p_l^m$ 作为 student：

$$
\mathcal{L}_{kd}^{uni}
=
\sum_m
D_{KL}
\left(
p_l^{full}
\| 
p_l^m
\right)
$$

也可以进一步在 feature/prototype 层面加入蒸馏：

$$
\mathcal{L}_{proto}^{uni}
=
\sum_m
\left\|
P_l^{full} - P_l^m
\right\|_2^2
$$

该设计的作用是：

> 利用完整模态标注数据，将完整模态下的稳定语义知识迁移到单模态或缺失模态路径中，从而提前增强模型对模态缺失的鲁棒性。

---

## 4.5 构建客户端 full-modal semantic anchor

从完整模态融合特征 $f_l^{full}$ 中，根据真实标签 $y_l$ 提取类别原型。

对于类别 $c$，定义：

$$
A_{k,c}^{full}
=
\frac{1}{|\Omega_c|}
\sum_{i \in \Omega_c}
f_{l,i}^{full}
$$

其中：

- $A_{k,c}^{full}$：客户端 $k$ 上类别 $c$ 的完整模态语义锚点；
- $\Omega_c$：标注中属于类别 $c$ 的 voxel 集合；
- $f_{l,i}^{full}$：第 $i$ 个 voxel 的完整模态融合特征。

该 anchor 表示：

> 在客户端 $k$ 上，类别 $c$ 在完整模态输入条件下的可靠语义中心。

---

## 4.6 服务端聚合 full-modal anchor bank

每个客户端上传：

$$
\{A_{k,c}^{full}\}_{c=1}^{C}
$$

服务端进行聚合：

$$
A_c^g
=
EMA
\left(
A_c^g,
\sum_k w_k A_{k,c}^{full}
\right)
$$

其中：

- $A_c^g$：全局类别级 full-modal anchor；
- $w_k$：客户端聚合权重；
- EMA 用于平滑更新，避免 anchor bank 每轮剧烈波动。

最终服务端维护：

$$
\mathcal{A}^{full}
=
\{A_c^g\}_{c=1}^{C}
$$

该全局 anchor bank 会在后续无标注分支中用于伪标签修正。

---

## 4.7 创新点一的数据流

```text
本地少量完整模态标注数据
        ↓
输入 FedAMM/RFNet 主干
        ↓
得到完整模态融合预测 p_full 和融合特征 f_full
        ↓
计算 CE + Dice 监督分割损失
        ↓
主动构造单模态/缺失模态视图
        ↓
完整模态预测蒸馏到单模态预测
        ↓
利用 GT 从完整模态融合特征中提取类别 anchor
        ↓
客户端上传 full-modal anchor
        ↓
服务端聚合得到 global full-modal anchor bank
        ↓
用于后续无标注缺失模态伪标签修正
```

---

## 4.8 创新点一中文总结注释

> **中文总结：**  
> 创新点一的核心是提高少量完整模态标注数据的利用效率。传统半监督方法通常只把标注数据用于 CE/Dice 监督，而在本框架中，完整四模态标注样本被进一步提升为全局语义知识源。首先，完整模态样本用于训练基础分割能力；其次，通过主动构造单模态视图，并让完整模态预测蒸馏到单模态预测，使模型提前学习在缺失模态条件下保持语义一致；最后，从完整模态融合特征中提取类别级语义 anchor，并在服务端聚合为 global full-modal anchor bank。这样，少量标注数据不仅提供像素级监督，还为后续无标注缺失模态样本的伪标签修正提供可靠的跨客户端语义先验。

---

# 5. 创新点二：Mask-aware Distilled Pseudo-label Filtering

## 5.1 设计动机

无标注数据占训练数据的大部分，但它们存在不同模态缺失模式。

不同模态组合的信息量不同：

```text
四模态：信息最完整，预测相对可靠
三模态：信息较完整，预测较可靠
双模态：信息中等，预测不确定性增加
单模态：信息最少，预测最不稳定
```

因此，不能使用统一阈值处理所有无标注样本。

普通伪标签方法通常只依赖：

$$
\max_c p_{i,c} > \tau
$$

但在模态缺失场景中，模型可能出现：

- teacher 预测置信度高但语义错误；
- 单模态样本预测偏向某些类别；
- 客户端局部模型受到本地模态组合分布影响，导致伪标签偏移；
- 极端 non-IID 下伪标签噪声被联邦聚合放大。

因此，需要设计缺失模式感知的伪标签生成与筛选机制。

---

## 5.2 无标注数据输入

对于客户端 $k$ 的无标注样本：

$$
x_u^s \in D_k^u
$$

其中：

- $s$：当前样本的模态组合；
- $|s|$：当前样本可用模态数量。

对同一个无标注样本构造两个视图：

$$
x_{u,w}^s, \quad x_{u,strong}^s
$$

其中：

- $x_{u,w}^s$：weak augmentation 视图；
- $x_{u,strong}^s$：strong augmentation 视图。

---

## 5.3 Teacher-Student 预测

EMA teacher 处理 weak view：

$$
p_t^w, f_t^w
=
f_{\theta_t}(x_{u,w}^s)
$$

Student 处理 strong view：

$$
p_s^{strong}, f_s^{strong}
=
f_{\theta_s}(x_{u,strong}^s)
$$

其中：

- $p_t^w$：teacher 预测；
- $f_t^w$：teacher 特征；
- $p_s^{strong}$：student 预测；
- $f_s^{strong}$：student 特征。

---

## 5.4 Anchor-guided pseudo-label refinement

仅依赖 teacher prediction 不够，因为 teacher 在缺失模态条件下也可能产生错误伪标签。

因此，引入创新点一得到的 global full-modal anchor bank：

$$
\mathcal{A}^{full}
=
\{A_c^g\}_{c=1}^{C}
$$

对每个 voxel 的特征 $f_{t,i}^w$，计算其与各类别 anchor 的相似度：

$$
q_{A,i,c}
=
softmax
\left(
\frac{sim(f_{t,i}^w, A_c^g)}{\tau}
\right)
$$

其中：

- $q_A$：anchor-based 类别概率分布；
- $sim(\cdot)$：余弦相似度；
- $\tau$：温度系数。

然后融合 teacher prediction 和 anchor prediction：

$$
\tilde{p}_{i,c}
=
\omega_s p_{t,i,c}^w
+
(1-\omega_s)q_{A,i,c}
$$

其中：

$$
\omega_s = \frac{|s|}{4}
$$

含义：

```text
可用模态越多，越相信 teacher prediction；
可用模态越少，越依赖 global full-modal anchor prior。
```

例如：

| 可用模态数 | $\omega_s$ | 伪标签修正策略 |
|---|---:|---|
| 4 | 1.00 | 几乎完全相信 teacher |
| 3 | 0.75 | teacher 为主，anchor 辅助 |
| 2 | 0.50 | teacher 和 anchor 均衡 |
| 1 | 0.25 | anchor prior 占主导 |

最终得到 refined pseudo-label：

$$
\hat{y}_i
=
\arg\max_c \tilde{p}_{i,c}
$$

---

## 5.5 三重伪标签筛选机制

为了尽可能获得可靠伪标签，设计三个过滤门。

### 5.5.1 过滤门一：缺失模式感知置信度过滤

定义：

$$
\mathbb{I}_{conf}(i)
=
\mathbf{1}
[
\max_c \tilde{p}_{i,c} > \tau_s
]
$$

其中阈值：

$$
\tau_s
=
\tau_0
+
\gamma
\left(
1-\frac{|s|}{4}
\right)
$$

含义：

```text
模态越完整，阈值越低；
模态越缺失，阈值越高。
```

这样可以避免单模态或严重缺失模态样本产生大量低质量伪标签。

---

### 5.5.2 过滤门二：Teacher-Student 一致性过滤

比较 teacher weak prediction 和 student strong prediction 的差异：

$$
\mathbb{I}_{cons}(i)
=
\mathbf{1}
[
D(p_t^w(i), p_s^{strong}(i)) < \epsilon_s
]
$$

如果同一样本在 weak/strong 视图下预测差异过大，说明该区域伪标签不稳定，应过滤掉。

---

### 5.5.3 过滤门三：Anchor Agreement 过滤

要求 teacher prediction 和 anchor prediction 在类别判断上保持一致：

$$
\mathbb{I}_{anchor}(i)
=
\mathbf{1}
[
\arg\max_c p_{t,i,c}^w
=
\arg\max_c q_{A,i,c}
]
$$

该门用于过滤：

> teacher 预测虽然自信，但已经偏离完整模态语义空间的错误伪标签。

---

## 5.6 最终伪标签有效区域

三个过滤门同时满足时，该 voxel 才参与无监督训练：

$$
\mathbb{I}(i)
=
\mathbb{I}_{conf}(i)
\cdot
\mathbb{I}_{cons}(i)
\cdot
\mathbb{I}_{anchor}(i)
$$

---

## 5.7 无监督伪标签损失

对筛选后的高置信区域计算无监督损失：

$$
\mathcal{L}_{pl}
=
\frac{1}{|\Omega_u|}
\sum_i
\mathbb{I}(i)
\cdot
\mathcal{L}_{ce}
(p_s^{strong}(i), \hat{y}_i)
+
\mathcal{L}_{dice}(p_s^{strong}, \hat{y})
$$

---

## 5.8 无标注 prototype alignment

从高置信伪标签区域提取无标注 prototype：

$$
P_{u,c}^{s}
=
\frac{1}{|\hat{\Omega}_c|}
\sum_{i \in \hat{\Omega}_c}
f_{s,i}^{strong}
$$

其中：

- $\hat{\Omega}_c$：伪标签中被选为类别 $c$ 的高置信区域；
- $P_{u,c}^{s}$：无标注缺失模态样本的类别 prototype。

然后与全局 full-modal anchor 对齐：

$$
\mathcal{L}_{u}^{anchor}
=
\sum_c
\left\|
P_{u,c}^{s}
-
A_c^g
\right\|_2^2
$$

该损失的作用是：

> 让缺失模态无标注样本的高层语义特征向完整模态语义空间靠近。

---

## 5.9 无标注分支总损失

$$
\mathcal{L}_{unlabeled}
=
\mathcal{L}_{pl}
+
\lambda_{anc}\mathcal{L}_{u}^{anchor}
+
\lambda_{con}\mathcal{L}_{cons}
$$

其中：

- $\mathcal{L}_{pl}$：伪标签监督损失；
- $\mathcal{L}_{u}^{anchor}$：无标注 prototype 与 full-modal anchor 对齐损失；
- $\mathcal{L}_{cons}$：teacher-student 一致性损失。

---

## 5.10 无标注数据流

```text
无标注缺失模态样本 x_u^s
        ↓
构造 weak view 和 strong view
        ↓
EMA teacher 输入 weak view
        ↓
student 输入 strong view
        ↓
teacher 得到初始预测 p_t 和特征 f_t
        ↓
计算特征与 global full-modal anchor 的类别相似度 q_A
        ↓
根据当前缺失模式 s 自适应融合 p_t 与 q_A
        ↓
得到 refined pseudo-label
        ↓
三重过滤：
  1. 缺失模式感知置信度过滤
  2. teacher-student 一致性过滤
  3. anchor agreement 过滤
        ↓
只保留高可信 voxel/region
        ↓
用于无监督 CE + Dice 训练
        ↓
从高可信区域提取 unlabeled prototype
        ↓
与 global full-modal anchor 做 feature-level 对齐
        ↓
上传高可信 prototype 用于服务端语义聚合
```

---

## 5.11 创新点二中文总结注释

> **中文总结：**  
> 创新点二的核心是提高无标注缺失模态数据的伪标签可靠性。由于不同缺失模式的信息量不同，单一置信度阈值无法适应所有无标注样本。因此，本框架首先由 EMA teacher 对 weak view 生成初始预测，再利用无标注样本特征与全局 full-modal anchor 的相似度对伪标签进行修正。修正时根据当前样本可用模态数量动态分配 teacher prediction 和 anchor prior 的权重：模态越完整越相信 teacher，模态越缺失越依赖 anchor。随后，通过缺失模式自适应置信度、teacher-student 一致性和 anchor agreement 三重过滤机制，尽可能剔除由模态缺失、本地偏差和伪标签噪声导致的不可靠区域。最终，仅高可信伪标签区域参与无监督训练，并进一步提取无标注 prototype 与全局 anchor 对齐，从而提高无标注数据利用质量。

---

# 6. 创新点三：Reliability-aware Semantic Aggregation

## 6.1 设计动机

在联邦半监督任意模态缺失场景下，客户端之间存在多重异构：

1. 数据分布异构；
2. 模态组合分布异构；
3. 标注/无标注比例带来的训练质量差异；
4. 伪标签质量差异；
5. 客户端更新稳定性差异。

如果继续使用普通 FedAvg：

$$
\theta^g = \sum_k \frac{n_k}{N}\theta_k
$$

可能出现以下问题：

- 伪标签质量差的客户端污染全局模型；
- 单一模态组合占优的客户端主导某些 encoder 更新；
- 客户端模型更新方向不一致，造成 client drift；
- 全局模型语义空间不稳定。

因此，聚合时不应只看客户端样本数，而应考虑客户端训练可靠性和语义一致性。

---

## 6.2 客户端上传信息

每轮本地训练后，客户端 $k$ 上传：

```text
1. 全模型参数 θ_k
2. 四个 modality-specific encoder 参数 θ_{k,m}
3. full-modal labeled prototypes A_{k,c}^{full}
4. high-confidence unlabeled prototypes P_{k,c,s}^{u}
5. 伪标签平均置信度 Conf_k
6. 伪标签保留比例 Sel_k
7. 本地验证 Dice 或训练稳定性指标
8. 当前客户端模态出现比例 n_{k,m}
9. 当前客户端缺失模式分布 p_k(s)
```

注意：

> 上传的是模型参数、统计信息和语义原型，不上传原始医学图像。

---

## 6.3 客户端可靠性评分

定义客户端可靠性：

$$
R_k
=
\alpha R_k^{sup}
+
\beta R_k^{pl}
+
\delta R_k^{align}
$$

其中包含三部分。

### 6.3.1 有标注监督稳定性

$$
R_k^{sup}
=
\frac{1}{Var(Dice_k^{t-w:t})+\epsilon}
$$

含义：

- 最近几轮验证 Dice 越稳定，说明客户端监督训练越可靠；
- Dice 波动越大，说明客户端更新不稳定，应降低聚合权重。

---

### 6.3.2 伪标签可靠性

$$
R_k^{pl}
=
Conf_k \cdot Sel_k
$$

其中：

$$
Conf_k =
\frac{1}{N}
\sum_i
\max_c \tilde{p}_{i,c}
$$

$$
Sel_k =
\frac{|\Omega_{selected}|}{|\Omega|}
$$

含义：

- $Conf_k$：被选中伪标签的平均置信度；
- $Sel_k$：伪标签保留比例。

两者相乘可以避免两种极端：

```text
置信度高但保留区域极少：无标注贡献不足；
保留区域多但置信度低：噪声风险较高。
```

---

### 6.3.3 Anchor 对齐可靠性

$$
R_k^{align}
=
\frac{1}{C}
\sum_c
sim(P_{k,c}^{u}, A_c^g)
$$

含义：

> 如果客户端从无标注数据中提取的 prototype 与全局 full-modal anchor 越接近，说明该客户端的无标注学习越可信。

---

## 6.4 可靠性感知全模型聚合

对客户端可靠性归一化：

$$
\bar{R}_k
=
\frac{R_k}{\sum_j R_j}
$$

全局模型聚合：

$$
\theta^g
=
\sum_k
\bar{R}_k \theta_k
$$

与普通 FedAvg 相比，该聚合方式可以降低低质量伪标签客户端对全局模型的负面影响。

---

## 6.5 可靠性感知模态 encoder 聚合

FedAMM 原本已有 modality proportion based encoder aggregation。  
在此基础上，引入客户端可靠性。

对于第 $m$ 个模态 encoder：

$$
\theta_m^g
=
\sum_k
w_{k,m}
\theta_{k,m}
$$

其中：

$$
w_{k,m}
=
\frac{
n_{k,m} R_k
}{
\sum_j n_{j,m} R_j
}
$$

其中：

- $n_{k,m}$：客户端 $k$ 中模态 $m$ 的出现比例或样本数；
- $R_k$：客户端可靠性评分。

该设计的含义是：

> 某客户端即使拥有较多某模态样本，如果它的伪标签质量差或训练不稳定，也不应该主导该模态 encoder 的全局更新。

---

## 6.6 语义原型聚合

服务端维护两个语义库。

### 6.6.1 Full-modal anchor bank

$$
\mathcal{A}^{full}
=
\{A_c^g\}_{c=1}^{C}
$$

主要由有标注完整模态样本生成的 prototype 更新。

### 6.6.2 Missing-pattern prototype bank

$$
\mathcal{A}^{miss}
=
\{A_{c,s}^g\}
$$

其中：

- $c$：类别；
- $s$：15 种模态组合。

其更新方式为：

$$
A_{c,s}^g
=
EMA
\left(
A_{c,s}^g,
\sum_k
\bar{R}_k
P_{k,c,s}^{u}
\right)
$$

其中：

- $P_{k,c,s}^{u}$：客户端 $k$ 在缺失模式 $s$ 下，从高置信无标注区域提取的类别 prototype；
- $\bar{R}_k$：客户端可靠性权重。

该 missing-pattern prototype bank 的作用是：

1. 建模不同缺失模式下的类别语义中心；
2. 为下一轮相同缺失模式样本提供更细粒度语义先验；
3. 在极端 non-IID 模态组合分布下缓解语义漂移；
4. 使跨客户端不共享原始数据的情况下仍能对齐类别语义空间。

---

## 6.7 服务端下发内容

每轮服务端下发：

```text
1. 更新后的 global model θ^g
2. 更新后的 modality-specific encoders θ_m^g
3. global full-modal anchor bank A_c^g
4. missing-pattern prototype bank A_{c,s}^g
5. 当前轮客户端可靠性统计信息
```

客户端下一轮使用这些信息进行：

- 模型初始化；
- 无标注伪标签修正；
- anchor alignment；
- 缺失模式级 prototype alignment。

---

## 6.8 服务端聚合数据流

```text
客户端本地训练结束
        ↓
上传模型参数、encoder 参数、prototype 和可靠性统计
        ↓
服务端计算每个客户端可靠性 R_k
        ↓
可靠性感知全模型聚合
        ↓
模态比例 × 可靠性感知 encoder 聚合
        ↓
更新 full-modal anchor bank
        ↓
更新 missing-pattern prototype bank
        ↓
下发模型参数和语义原型库
        ↓
下一轮客户端继续训练
```

---

## 6.9 创新点三中文总结注释

> **中文总结：**  
> 创新点三的核心是让联邦聚合从“单纯参数平均”变成“参数级与语义级联合可靠聚合”。在半监督任意模态缺失场景下，不同客户端的伪标签质量、模态组合分布和训练稳定性差异很大，普通 FedAvg 容易受到低质量客户端更新的干扰。因此，本框架为每个客户端计算可靠性评分，综合考虑有标注验证稳定性、无标注伪标签置信度、伪标签保留比例以及无标注 prototype 与全局 anchor 的对齐程度。随后，服务端利用该可靠性评分进行全模型聚合，并结合模态出现比例对 modality-specific encoders 进行加权聚合。同时，服务端维护 full-modal anchor bank 和 missing-pattern prototype bank，使客户端在参数空间和语义空间上同时对齐，从而缓解模态异构和伪标签噪声引起的 client drift。

---

# 7. 总体训练流程

## 7.1 Stage 1：监督 warm-up

训练初期只使用少量完整模态有标注数据。

目标：

1. 训练基础 segmentation model；
2. 构建初始 full-modal anchor bank；
3. 让单模态分支学习完整模态语义；
4. 避免早期伪标签质量太差导致模型崩溃。

损失函数：

$$
\mathcal{L}_{warm}
=
\mathcal{L}_{seg}^{full}
+
\lambda_{kd}\mathcal{L}_{kd}^{uni}
+
\lambda_a\mathcal{L}_{anchor}^{full}
$$

---

## 7.2 Stage 2：半监督训练

每个 iteration 同时采样：

$$
(x_l^{full}, y_l), \quad x_u^s
$$

即：

```text
一个 labeled full-modal batch
一个 unlabeled missing-modal batch
```

总损失：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{labeled}
+
\lambda_u(t)\mathcal{L}_{unlabeled}
+
\lambda_g\mathcal{L}_{global}
$$

其中 $\lambda_u(t)$ 使用 ramp-up：

$$
\lambda_u(t)
=
\lambda_{max}
\cdot
\exp
\left[
-5
\left(
1-\frac{t}{T}
\right)^2
\right]
$$

这样可以避免训练早期无标注伪标签噪声过大。

---

## 7.3 Stage 3：服务端可靠聚合

每一轮通信后，服务端更新：

$$
\theta^g,\quad
\theta_m^g,\quad
A_c^g,\quad
A_{c,s}^g,\quad
R_k
$$

然后下发到客户端，进入下一轮训练。

---

# 8. 总体损失函数

## 8.1 有标注分支损失

$$
\mathcal{L}_{labeled}
=
\mathcal{L}_{seg}^{full}
+
\lambda_{kd}\mathcal{L}_{kd}^{uni}
+
\lambda_{proto}\mathcal{L}_{proto}^{uni}
+
\lambda_a\mathcal{L}_{anchor}^{full}
$$

---

## 8.2 无标注分支损失

$$
\mathcal{L}_{unlabeled}
=
\mathcal{L}_{pl}
+
\lambda_{anc}\mathcal{L}_{u}^{anchor}
+
\lambda_{con}\mathcal{L}_{cons}
$$

---

## 8.3 全局语义对齐损失

$$
\mathcal{L}_{global}
=
\sum_c
\|P_{k,c} - A_c^g\|_2^2
+
\sum_{c,s}
\|P_{k,c,s} - A_{c,s}^g\|_2^2
$$

---

## 8.4 最终总损失

$$
\mathcal{L}_{total}
=
\mathcal{L}_{labeled}
+
\lambda_u(t)\mathcal{L}_{unlabeled}
+
\lambda_g\mathcal{L}_{global}
$$

---

# 9. 三个创新点之间的闭环关系

```text
创新点一：
少量完整模态标注数据
        ↓
构建 full-modal semantic anchor
        ↓
创新点二：
用 anchor 指导缺失模态无标注伪标签修正和筛选
        ↓
得到高质量伪标签与 high-confidence unlabeled prototype
        ↓
创新点三：
服务端根据伪标签质量和语义一致性进行可靠聚合
        ↓
更新 global model、full-modal anchor bank 和 missing-pattern prototype bank
        ↓
下一轮伪标签质量进一步提升
```

最终形成：

```text
可靠标注语义源
        ↓
可靠无标注伪监督
        ↓
可靠跨客户端聚合
        ↓
更稳健的全局模型
```

---

# 10. 实验验证建议

## 10.1 首先验证的异构程度

建议优先验证：

$$
\alpha = 0.1
$$

原因：

- $\alpha=1$：轻度异构，baseline 可能已经较强，新模块优势不明显；
- $\alpha=0.001$：极端异构，训练难度过大，不适合作为第一验证；
- $\alpha=0.1$：中度异构，既能暴露问题，又不至于训练崩溃，最适合验证三个创新点是否有效。

推荐顺序：

$$
\alpha=0.1 \rightarrow \alpha=1 \rightarrow \alpha=0.001
$$

---

## 10.2 消融实验设计

### 完整模型

```text
FedMASS full model
```

### 去掉创新点一

```text
w/o Full-modal Anchor Supervision
```

验证少量完整模态标注数据是否被高效利用。

### 去掉创新点二

```text
w/o Mask-aware Distilled Pseudo-label Filtering
```

验证缺失模式感知伪标签筛选是否有效。

### 去掉创新点三

```text
w/o Reliability-aware Semantic Aggregation
```

验证可靠性感知聚合是否优于普通 FedAvg/FedAMM 聚合。

### 去掉 anchor refinement

```text
w/o Anchor-guided Pseudo-label Refinement
```

验证 global full-modal anchor 是否真的提升伪标签质量。

### 去掉 mask-aware threshold

```text
w/o Mask-aware Threshold
```

验证不同缺失模式使用不同阈值是否必要。

### 去掉 anchor agreement

```text
w/o Anchor Agreement
```

验证 teacher prediction 与 anchor prediction 一致性过滤是否有效。

---

# 11. 代码改造 TODO

## 11.1 数据层

- [ ] 将每个客户端数据划分为 labeled / unlabeled。
- [ ] labeled 保持完整四模态。
- [ ] unlabeled 按 15 种模态组合生成缺失模式。
- [ ] 新增 labeled dataloader。
- [ ] 新增 unlabeled dataloader。
- [ ] unlabeled dataloader 返回 weak view 和 strong view。

---

## 11.2 模型训练层

- [ ] 在本地训练中加入双流 batch：
  - labeled batch
  - unlabeled batch
- [ ] 加入 EMA teacher。
- [ ] 加入 full-modal to unimodal distillation。
- [ ] 加入 anchor extraction。
- [ ] 加入 anchor-guided pseudo-label refinement。
- [ ] 加入三重伪标签过滤。
- [ ] 加入无标注 prototype alignment。

---

## 11.3 服务端聚合层

- [ ] 上传 labeled full-modal prototypes。
- [ ] 上传 high-confidence unlabeled prototypes。
- [ ] 统计 pseudo-label confidence。
- [ ] 统计 pseudo-label selected ratio。
- [ ] 统计 validation Dice fluctuation。
- [ ] 计算 client reliability score。
- [ ] 修改全模型聚合权重。
- [ ] 修改 modality encoder 聚合权重。
- [ ] 维护 full-modal anchor bank。
- [ ] 维护 missing-pattern prototype bank。

---

## 11.4 实验层

- [ ] 添加参数开关：是否启用 innovation 1。
- [ ] 添加参数开关：是否启用 innovation 2。
- [ ] 添加参数开关：是否启用 innovation 3。
- [ ] 添加伪标签质量统计日志。
- [ ] 添加每种 mask pattern 的 Dice 统计。
- [ ] 添加不同 $\alpha$ 异构程度实验。
- [ ] 添加消融实验配置。

---

# 12. 给导师汇报的一段话

本课题拟在 FedAMM 框架基础上，进一步面向“联邦半监督 + 任意模态缺失”场景进行扩展：每个客户端仅保留少量完整四模态标注数据，大量无标注数据则按照 15 种模态组合随机缺失。整体思路是构建一个 Full-modal Anchor guided Semi-supervised Federated Segmentation 框架。首先，在有标注数据流中，不仅利用完整四模态样本进行常规 Dice/CE 监督分割，还主动构造单模态或缺失模态视图，通过完整模态预测向单模态预测蒸馏，使少量标注样本同时服务于完整模态学习和缺失模态鲁棒性学习；同时，从完整模态融合特征中提取类别级语义原型，并在服务端聚合形成全局 full-modal anchor bank，作为后续无标注学习的高可信语义先验。其次，在无标注数据流中，针对不同缺失模式下伪标签可靠性不一致的问题，设计缺失模式感知的蒸馏式伪标签筛选机制：对无标注缺失模态样本构造 weak/strong 两种视图，由 EMA teacher 生成初始伪标签，再结合当前特征与全局 full-modal anchor 的类别相似度对伪标签进行修正；模态越完整，越相信 teacher prediction，模态越缺失，越依赖全局 anchor prior。随后通过三重过滤机制筛选伪标签，包括缺失模式自适应置信度阈值、teacher-student 一致性约束以及 anchor agreement 约束，从而尽可能过滤掉由模态缺失和本地模型偏差导致的噪声伪标签。最后，在服务端聚合阶段，不再仅采用普通 FedAvg 或单纯按模态比例聚合，而是设计可靠性感知的语义聚合策略：综合客户端有标注验证稳定性、无标注伪标签平均置信度、伪标签保留比例以及无标注 prototype 与全局 anchor 的对齐程度，估计每个客户端的可靠性；再结合各模态在客户端中的出现比例，对全局模型参数和 modality-specific encoders 进行加权聚合。同时，服务端维护类别级 full-modal anchor bank 和缺失模式级 prototype bank，使不同客户端在参数空间和语义空间上同时对齐。整体上，该框架形成“少量完整模态标注数据构建可靠语义锚点—锚点指导缺失模态无标注伪标签筛选—高质量伪标签与语义原型反向促进稳健聚合”的闭环，从而缓解联邦半监督场景下伪标签噪声、模态缺失异构和客户端漂移共同导致的分割性能下降问题。

---

# 13. 一句话总结

> FedMASS 通过 Full-modal Anchor Supervision 高效利用少量完整模态标注数据，通过 Mask-aware Distilled Pseudo-label Filtering 从大量缺失模态无标注数据中筛选可靠伪标签，并通过 Reliability-aware Semantic Aggregation 实现跨客户端稳健语义对齐与模型聚合，从而提升联邦半监督任意模态缺失医学图像分割性能。
