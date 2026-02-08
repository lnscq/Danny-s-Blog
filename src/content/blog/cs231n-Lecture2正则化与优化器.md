---
title: "CS231n Lecture 3: 正则化与优化"
description: "cs231n"
pubDate: "Feb 8 2026"
image: https://tc.alcy.cc/tc/20260121/50eb7681825caac08a33b160e9edc81f.webp
categories:
  - notebook
tags:
  - CV
  - Regularization
  - Optimization
---

## CS231n Lecture 3: 正则化与优化

[课程链接](https://cs231n.stanford.edu/)

### 引言

训练一个深度视觉模型本质上是在高维非凸损失景观中寻找一个泛化良好的低谷。这一过程由两个关键机制驱动：**正则化**控制模型复杂度以避免过拟合，**优化器**则决定如何高效、稳定地穿越参数空间。本讲深入剖析 L1/L2 正则化的几何直觉，并系统梳理主流优化器的设计动机与行为特性。

### 损失函数的结构

完整的训练目标由数据损失与正则化项共同构成：

$$
\mathcal{L}(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i(f(\mathbf{x}^{(i)}; \mathbf{W}), y^{(i)}) + \lambda R(\mathbf{W}),
$$

其中第一项驱使模型拟合观测数据，第二项 $R(\mathbf{W})$ 对参数 $\mathbf{W}$ 施加先验约束，超参数 $\lambda$ 平衡拟合能力与泛化能力。

### L1 与 L2 正则化的几何直觉

正则化可视为在参数空间中引入偏好区域。L1 和 L2 正则化对应不同的几何形状，从而引导优化走向不同类型的解。

L2 正则化（权重衰减）定义为 $R(\mathbf{W}) = \|\mathbf{W}\|_2^2$。其等高线是围绕原点的超球面。当优化器在数据损失的等高线与 L2 球面之间寻找首次接触点时，该点通常位于所有坐标轴方向上均有非零分量的位置。因此，L2 正则化倾向于产生**小而分散**的权重，避免任何单个参数主导预测。这种“平滑”解在图像任务中表现优异，因为它抑制了对特定像素或特征的过度依赖，提升了鲁棒性。

L1 正则化定义为 $R(\mathbf{W}) = \|\mathbf{W}\|_1$，其等高线是尖角的超立方体（如二维中的菱形）。由于尖角位于坐标轴上，优化过程更容易在这些顶点处与数据损失等高线相切，从而导致部分权重被精确压缩至零。这种**稀疏性**在特征选择场景中非常有用，但在卷积神经网络等密集连接的视觉模型中，参数稀疏性并非主要目标，且可能损害表示能力，故 L1 在实践中较少使用。

值得注意的是，L2 正则化不仅提升泛化，还能改善优化动态：通过限制权重增长，它使损失曲面更加平滑，梯度变化更稳定。

### 优化器的演进：从基础到自适应

#### 随机梯度下降（SGD）

SGD 是优化的基石。它通过小批量数据估计梯度：

$$
\mathbf{g}_t = \nabla_{\mathbf{W}} \mathcal{L}(\mathcal{B}_t), \quad \mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \mathbf{g}_t,
$$

其中 $\alpha$ 为学习率。SGD 的更新方向是局部梯度的无偏估计，但高方差使其路径充满噪声，在病态曲率区域（如“峡谷”地形）会剧烈振荡，收敛缓慢。

#### 动量法（Momentum）

动量法借鉴物理学中的惯性概念，引入速度变量 $\mathbf{v}_t$ 累积历史梯度：

$$
\mathbf{v}_t = \rho \mathbf{v}_{t-1} + \mathbf{g}_t, \quad \mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \mathbf{v}_t.
$$

其中 $\rho \in [0,1)$ 是动量系数（常取 0.9）。这相当于给参数更新增加了“惯性”：在一致方向上加速前进，在振荡方向上相互抵消。直观上，动量帮助优化器“滚下山坡”而非“来回弹跳”，显著加速收敛并提升稳定性。

#### RMSProp：自适应学习率

不同参数的梯度尺度可能差异巨大（如某些滤波器权重更新频繁，另一些则几乎静止）。固定学习率难以兼顾。RMSProp 为每个参数维护独立的学习率，通过缩放梯度来平衡更新幅度：

$$
\mathbf{r}_t = \gamma \mathbf{r}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \quad \mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\mathbf{r}_t} + \epsilon} \mathbf{g}_t.
$$

这里 $\mathbf{r}_t$ 是梯度平方的指数移动平均（EMA），$\gamma$ 常取 0.99。若某参数近期梯度较大，则 $\mathbf{r}_t$ 增大，其有效学习率自动减小；反之则增大。这使得优化器能自适应地处理不同尺度的参数。

#### Adam：融合动量与自适应

Adam（Adaptive Moment Estimation）结合了动量与 RMSProp 的思想。它同时维护梯度的一阶矩（均值）$\mathbf{m}_t$ 和二阶矩（未中心化方差）$\mathbf{r}_t$：

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
\mathbf{r}_t &= \beta_2 \mathbf{r}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}
$$

由于 $\mathbf{m}_t$ 和 $\mathbf{r}_t$ 初始为零，早期估计存在偏差。Adam 通过偏差校正得到无偏估计：

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{r}}_t = \frac{\mathbf{r}_t}{1 - \beta_2^t}.
$$

最终更新为：

$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{r}}_t} + \epsilon} \hat{\mathbf{m}}_t.
$$

Adam 的直觉是：既利用动量平滑方向，又根据历史梯度大小自适应调整步长。其默认超参数（$\beta_1=0.9, \beta_2=0.999$）在大量任务上表现稳健，使其成为深度学习的事实标准。

#### AdamW：正确的权重衰减

一个关键但常被忽视的细节是：标准 Adam 实现通常将 L2 正则化项 $\lambda \mathbf{W}$ 直接加入梯度 $\mathbf{g}_t$ 中。然而，在自适应学习率的框架下，这并不等价于在原始参数空间中执行权重衰减。具体而言，自适应缩放会扭曲正则化强度，导致不同参数受到不一致的惩罚。

AdamW 修正了这一问题，将权重衰减操作与梯度更新解耦：

$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{r}}_t} + \epsilon} + \lambda \mathbf{W}_t \right).
$$

等价地写作：

$$
\mathbf{W}_{t+1} = (1 - \alpha \lambda) \mathbf{W}_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{r}}_t} + \epsilon}.
$$

这确保了权重衰减以统一的方式作用于所有参数，无论其梯度历史如何。大量实验表明，AdamW 不仅在理论上更干净，在实践中也 consistently 提升泛化性能，尤其在大规模训练中。

### 学习率调度策略

学习率 $\alpha$ 控制每次更新的步长，是影响训练成败的最关键超参数。固定学习率往往无法兼顾初期快速下降与后期精细收敛的需求，因此常采用调度策略：

- **步进衰减（Step Decay）**：在预设轮次（如每30 epoch）将学习率乘以衰减因子（如0.1）。简单有效，但需手动设定衰减点。
- **余弦退火（Cosine Annealing）**：学习率按余弦函数从初始值平滑降至零：
  $$
  \alpha_t = \frac{\alpha_0}{2} \left(1 + \cos\left(\pi \frac{t}{T}\right)\right),
  $$
  其中 $T$ 为总训练步数。该策略提供平滑的退火过程，常与重启（restart）结合使用。
- **线性预热（Linear Warmup）**：在训练初期（如前5%的迭代），将学习率从零线性增加至目标值。这有助于稳定大批次训练的初始阶段，避免因梯度估计不准导致的发散。

### 总结

正则化与优化是深度学习训练的双引擎。L2 正则化通过鼓励小而均匀的权重提升泛化能力，其几何直觉源于超球面约束；L1 虽诱导稀疏性，但在视觉任务中应用有限。在优化器方面，从 SGD 到 AdamW 的演进反映了对损失景观复杂性的逐步应对：动量处理方向噪声，RMSProp 处理尺度差异，Adam 融合二者，而 AdamW 则修正了正则化实现的理论缺陷。对于大多数现代视觉任务，**AdamW 配合余弦退火学习率调度**已成为强健且高效的默认选择。理解这些工具背后的直觉，是进行有效模型调优的前提。
