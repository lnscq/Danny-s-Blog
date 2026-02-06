---
title: "cs224R-Lecture 3 Policy Gradients"
description: "cs224R"
pubDate: "Feb 7 2026"
image: https://tc.alcy.cc/tc/20260121/1be8ab20bdc9e1d0f7c4f8febff8bd90.webp
categories:
  - notebook
tags:
  - RL
  - Policy Gradient
---

## Lecture 3: Policy Gradients

[课程链接](https://cs224r.stanford.edu/)

### 0. 从模仿学习到强化学习：为什么需要在线学习？

在上一讲中，我们看到模仿学习（Imitation Learning）能够利用专家示范数据快速获得一个不错的策略。但它有两个根本限制：第一，它无法超越专家的表现；第二，它不能通过与环境的交互进行自我改进。

如果我们没有专家数据，或者希望策略比专家更强，就必须回到强化学习的核心范式：**让智能体通过试错，直接优化累计奖励**。这就引出了本讲的主题——策略梯度（Policy Gradient）方法。

策略梯度是第一类真正意义上的**在线强化学习算法**：它不需要专家示范，而是通过不断与环境交互、收集自己的经验，并据此更新策略，以最大化期望总回报。

### 1. 问题设定与目标函数

我们考虑一个马尔可夫决策过程（MDP），其轨迹（trajectory）定义为状态和动作的交替序列：

$$
\tau = (s_1, a_1, s_2, a_2, \dots, s_T, a_T)
$$

策略 $\pi_\theta(a|s)$ 是一个由参数 $\theta$ 决定的概率分布。轨迹 $\tau$ 在策略 $\pi_\theta$ 下出现的概率为：

$$
p_\theta(\tau) = p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)
$$

我们的目标是找到一组参数 $\theta$，使得期望总回报最大：

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]
$$

这是一个关于 $\theta$ 的函数，我们需要计算它的梯度 $\nabla_\theta J(\theta)$，并用梯度上升法更新 $\theta$。

### 2. 策略梯度的推导：对数导数技巧

直接对 $J(\theta)$ 求导很困难，因为期望的分布 $p_\theta(\tau)$ 本身依赖于 $\theta$。这里的关键技巧是“**对数导数恒等式**”（log-derivative trick）：

对于任意概率密度函数 $p_\theta(x)$，有：

$$
\nabla_\theta p_\theta(x) = p_\theta(x) \nabla_\theta \log p_\theta(x)
$$

利用这个恒等式，我们可以将梯度改写为：

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau \\
= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]
$$

其中 $R(\tau) = \sum_{t=1}^T r(s_t, a_t)$ 是整条轨迹的总回报。

接下来，我们将 $\log p_\theta(\tau)$ 展开：

$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \left( \log \pi_\theta(a_t | s_t) + \log p(s_{t+1}|s_t, a_t) \right)
$$

注意到 $\log p(s_1)$ 和 $\log p(s_{t+1}|s_t, a_t)$ 都与 $\theta$ 无关，因此它们的梯度为零。于是：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

将其代回梯度表达式，得到最终的**策略梯度公式**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \right) \left( \sum_{t'=1}^T r(s_{t'}, a_{t'}) \right) \right]
$$

### 3. 直观理解：加权的监督学习

这个公式可以这样理解：策略梯度是在做一种**加权的监督学习**。

回忆行为克隆的目标是 $\min_\theta \mathbb{E}[-\log \pi_\theta(a|s)]$，其梯度为 $-\nabla_\theta \log \pi_\theta(a|s)$，目的是增加专家动作的似然。

而策略梯度的梯度项是 $\nabla_\theta \log \pi_\theta(a_t|s_t) \times R(\tau)$。这意味着：

- 如果一条轨迹的总回报 $R(\tau)$ 很高，我们就**增加**这条轨迹上所有 $(s_t, a_t)$ 对的似然。
- 如果 $R(\tau)$ 很低（甚至是负的），我们就**减少**这些 $(s_t, a_t)$ 对的似然。

这就是“试错学习”的数学形式化：多做带来好结果的事，少做带来坏结果的事。

### 4. 算法实现：REINFORCE

基于上述梯度，我们可以写出最简单的策略梯度算法——REINFORCE：

1. 初始化策略参数 $\theta$。
2. **收集数据**：运行策略 $\pi_\theta$，得到 $N$ 条完整轨迹 $\{\tau^i\}_{i=1}^N$。
3. **估计梯度**：用蒙特卡洛采样近似期望：
   $$
   \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} | s_{i,t}) \right) R(\tau^i)
   $$
4. **更新策略**：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$。
5. 重复步骤 2-4。

这个算法完全在线，只需要智能体与环境交互产生的数据。

### 5. 问题一：高方差

策略梯度的一个致命弱点是**方差极高**。原因在于，整条轨迹的回报 $R(\tau)$ 被用来更新轨迹上的每一个时间步。即使某个早期动作与最终的高回报毫无关系，它也会被错误地加强。

更糟的是，如果回报的绝对值很大，梯度的尺度会变得极不稳定。

### 6. 改进一：因果性（Causality）

一个基本的物理事实是：**当前的动作不能影响过去已经发生的奖励**。因此，在计算时间步 $t$ 的贡献时，我们只应考虑从 $t$ 开始的未来回报，即**回报到-go**（return-to-go）：

$$
\hat{R}_t = \sum_{t'=t}^T r(s_{t'}, a_{t'})
$$

于是梯度更新变为：

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} | s_{i,t}) \hat{R}_{i,t}
$$

这显著降低了方差，因为我们剔除了与当前决策无关的历史噪声。

### 7. 改进二：引入基线（Baseline）

即使使用了回报到-go，梯度仍然可能因回报的绝对值过大而波动。我们可以减去一个与动作无关的**基线** $b$，来进一步降低方差。

关键性质是：对于任意与动作 $a_t$ 无关的基线 $b(s_t)$，以下等式成立：

$$
\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t) \right] = b(s_t) \nabla_\theta \mathbb{E}_{a_t} [1] = 0
$$

因此，减去基线不会改变梯度的期望（无偏），但能有效降低方差。

一个常用且有效的选择是**状态价值函数** $V^{\pi}(s_t)$，即在状态 $s_t$ 下遵循当前策略的期望未来回报。实践中，常使用一个简单的启发式：**整个批次的平均回报**作为全局基线。

### 8. 从 on-policy 到 off-policy

REINFORCE 是一个 **on-policy** 算法：每次更新后，策略 $\pi_\theta$ 发生了变化，旧的数据就不再反映新策略的行为，必须重新收集数据。这导致样本效率极低。

为了提高效率，我们希望实现 **off-policy** 学习：用旧策略 $\pi_{\theta_{\text{old}}}$ 采集的数据，来更新新策略 $\pi_{\theta_{\text{new}}}$。

这可以通过**重要性采样**（Importance Sampling）实现。轨迹层面的重要性权重为：

$$
\frac{p_{\theta_{\text{new}}}(\tau)}{p_{\theta_{\text{old}}}(\tau)} = \prod_{t=1}^T \frac{\pi_{\theta_{\text{new}}}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

然而，这个乘积会随着轨迹长度 $T$ 指数级地爆炸或消失，导致估计极不稳定。

一个更实用的做法是在**单个时间步**层面进行重要性采样，得到 off-policy 策略梯度：

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i,t})} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \hat{R}_{i,t}
$$

这允许我们在同一批数据上进行多次梯度更新，大大提高了样本效率。

### 9. 约束策略更新幅度

然而，off-policy 更新有一个风险：如果新旧策略相差太远，重要性权重会失真，导致梯度估计完全错误。

因此，我们需要约束策略在一次更新中不能偏离太远。一种常见的方法是加入一个**KL散度约束**：

$$
\mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}} \left[ D_{\text{KL}} ( \pi_{\theta_{\text{new}}}(\cdot|s) \| \pi_{\theta_{\text{old}}}(\cdot|s) ) \right] \leq \delta
$$

这个约束保证了新策略的行为分布不会与旧策略相差太远，从而维持了 off-policy 估计的有效性。这也是后续 PPO 等算法的核心思想之一。

### 10. 总结

策略梯度为我们提供了一个优雅而通用的框架，将强化学习问题转化为一个可微的优化问题。尽管其原始形式（REINFORCE）存在高方差和低样本效率的问题，但通过引入**因果性**、**基线**、**重要性采样**和**KL约束**等一系列技巧，我们可以逐步构建出强大而实用的在线强化学习算法。这些思想构成了现代深度强化学习，如 PPO、TRPO 等算法的基石。
