---
title: "pi_RL：基于流式VLA的在线强化学习微调"
description: "论文阅读：pi_RL 与 VLA 在线强化学习"
pubDate: "Mar 04 2026"
image: https://tc.alcy.cc/tc/20260121/c275961101a58cde6afae5f5bd03585a.webp
categories:
  - Paper Reading
tags:
  - Paper Reading
  - RL
  - VLA
---
##  $\pi_{RL}$：基于流式VLA的在线强化学习微调

### Introduction

目前主流的VLAs的训练方法主要遵循两个步骤：先对VLM进行与预训练与有监督微调（SFT）的范式。
- 1. 在预训练的VLM基础中，VLA会在大规模，异构的人类演示数据集上进行微调
- 2. 随后在目标任务上进行SFT，使其能力能够与特定的机器人形态与环境对齐。然而SFT面临一些关键的挑战。首先，大规模的专家轨迹获取成本十分昂贵且费力，齐次仅仅通过SFT微调的模型往往会过拟合专家的数据。

因此近期的工作开始探索RL扩展到VLA的后训练过程，建立了一个如下图 **预训练->SFT->RL** 的三阶段的训练范式。使 VLA 能够通过主动环境交互和制定更具泛化能力的策略，将其性能提升至超越最初专家示范的水平。

![](https://github.com/lnscq/picx-images-hosting/raw/master/image.7pu79m4ds.webp)

然而，目前主流将RL应用到VLA模型中的模型主要局限在自回归VLA模型的领域，典型代表包括OpenVLA[^openvla]、OpenVLA-OFT[^openvla-oft]。这类模型采用离散动作解码器，以自回归或并行方式生成输出。核心在于让模型直接输出Action token，将动作维度离散化为256个区间，并使用Llama分词器中使用频率最低的256个token表示。这种方式与基于扩散和流的VLA形成对比，例如 $\pi$ 系列的 $\pi_0$[^pi0]、$\pi_{0.5}$[^pi05] 等代表性工作。这类模型通过flow matching中的的迭代过程来生成动作。这类VLA范式相较于Openvla-like的模型，不近动作生成频率更高，动作更加平顺，还能完成高灵巧度的任务。

然而，由于如何针对执行动作来刻画对数似然这一根本挑战，导致目前VLA-RL算法无法直接兼容于基于流的VLA模型。为了解决上述问题，这篇论文提出了$\pi_{RL}$[^pirl]，这是首个用于微调基于流的VLA模型的开源并行在线强化学习框架。

在这篇文章中，作者提出了两种解决方案来处理上述问题：

- 1. Flow-Noise
将可学习的噪声网络集成到去噪流程中，并将该阶段建模为离散时间马尔可夫决策过程（MDP），以实现精确的对数似然估计。
- 2. Flow-SDE
将普通微分方程（ODE）去噪流程转化为随机微分方程（SDE），在保证等效边缘分布用于探索的同时，构建了将去噪流程与策略—环境交互耦合的两层 MDP，并配合混合 ODE-SDE 采样技术来加速训练。


### Related works

VLA 模型最近通过集成多模态输入，在机器人领域取得了显著进展，实现了统一的感知、推理与控制。这一进展催生了一系列架构，包括Octo[^octo]、RT-1[^rt1]、OpenVLA[^openvla]、OpenVLA-OFT[^openvla-oft]、$\pi_0$[^pi0]、$\pi_{0.5}$[^pi05] 和 GR00T[^gr00t]。

OpenVLA 作为自回归VLA 架构的代表，将动作空间离散化为符号化表示
这使得基于语言条件的控制成为可能，通过将动作作为VLM 词汇表的一部分进行处理，但该方法在本质上限制了实现精细化运动所需的分辨率。

为了实现更加灵巧且连续的物理行为，$\pi_0、\pi_{0.5}$ 作为基于流的VLA 架构的代表，引入了基于流匹配的动作分块架构。这使VLA 能够建模复杂的连续动作分布，从而实现更为灵巧的物理行为。

近期的研究越来越多的集中于利用online RL来提升VLA的性能和泛化能力。
- 1. SimpleVLA-RL[^simplevla-rl]基于OpenVLA-OFT和GRPO，展示了在数据稀缺情况下，强化学习能够提升VLA模型的长程规划能力
- 2. RL4VLA（文中对应的经验研究工作）[^rl4vla]通过阶段性稀疏奖励，实证评估了PPO、GRPO和直接偏好优化（DPO）（Rafailov等，2023），发现PPO表现最佳
- 3. VLA-RL[^vla-rl]提出了专用的机器人流程奖励模型，并优化了数据处理流程
- 4. iRe-VLA[^ire-vla]提出了在强化学习探索与SFT更新之间迭代的框架
- 5. RIPT-VLA[^ript-vla]将REINFORCE leave-one-out（RLOO）（Kool等，2018）算法应用于QueST（Mete等，2024）和OpenVLA-OFT架构
- 6. RLinf-VLA[^rlinf-vla]为大规模强化学习训练VLA模型提供了统一且高效的框架，支持多样化的VLA——OpenVLA 和 OpenVLA-OFT 等架构，以及 PPO、GRPO 等多种强化学习（RL）算法，还有包括LIBERO 和 ManiSkill 在内的多种模拟器

上述工作均体现了将online RL应用到VLA的潜力，然而由于上述提及的对数似然的挑战，使得其在基于流的VLA中的应用仍然受限。

为解决online RL在flow matching模型的应用问题，有如下研究集中在这里：

- 1. Flow-GRPO[^flow-grpo]（Liu 等，2025a）将确定性常微分方程（ODE）转化为等价的随机微分方程（SDE），以实现随机性探索。在此基础上，后续研究如 Mix-GRPO[^mix-grpo]（Li 等，2025b）和 TempFlow-GRPO[^tempflow-grpo]（He 等，2025）通过混合 ODE-SDE rollout 进一步加速训练
- 2. ReinFlow[^reinflow]（Zhang 等，2025）在流路径中注入可学习的噪声，并将其转化为具有可计算似然的离散时间马尔可夫过程，从而实现稳定的策略梯度更新
- 3. 流策略优化（FPO）[^fpo]（McAllister 等，2025）将策略优化重构为最大化条件流匹配损失的优势加权比
- 4. 策略无关强化学习（PA-RL）[^pa-rl]（Mark 等，2024）能够通过监督学习将评论家优化后的动作蒸馏到策略中，实现对各类扩散和 Transformer 架构的高效微调
- 5. 通过强化学习引导扩散（DSRL）[^dsrl]（Wagenmaker 等，2025）则在其潜在噪声空间中执行强化学习，从而优化流策略，而无需直接修改策略参数本身

[^openvla]: Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246. https://arxiv.org/abs/2406.09246
[^openvla-oft]: Kim et al., *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success* (OpenVLA-OFT). arXiv:2502.19645. https://arxiv.org/abs/2502.19645
[^pi0]: Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164. https://arxiv.org/abs/2410.24164
[^pi05]: Intelligence et al., *$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054. https://arxiv.org/abs/2504.16054
[^pirl]: Chen et al., *$\pi_\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models*. arXiv:2510.25889. https://arxiv.org/abs/2510.25889
[^octo]: Octo Model Team et al., *Octo: An Open-Source Generalist Robot Policy*. arXiv:2405.12213. https://arxiv.org/abs/2405.12213
[^rt1]: Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*. arXiv:2212.06817. https://arxiv.org/abs/2212.06817
[^gr00t]: Bjorck et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734. https://arxiv.org/abs/2503.14734
[^simplevla-rl]: Li et al., *SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning*. arXiv:2509.09674. https://arxiv.org/abs/2509.09674
[^rl4vla]: Liu et al., *What Can RL Bring to VLA Generalization? An Empirical Study*. arXiv:2505.19789. https://arxiv.org/abs/2505.19789
[^vla-rl]: Lu et al., *VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning*. arXiv:2505.18719. https://arxiv.org/abs/2505.18719
[^ire-vla]: Guo et al., *Improving Vision-Language-Action Model with Online Reinforcement Learning*. arXiv:2501.16664. https://arxiv.org/abs/2501.16664
[^ript-vla]: Tan et al., *Interactive Post-Training for Vision-Language-Action Models* (RIPT-VLA). arXiv:2505.17016. https://arxiv.org/abs/2505.17016
[^rlinf-vla]: Zang et al., *RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models*. arXiv:2510.06710. https://arxiv.org/abs/2510.06710
[^flow-grpo]: Liu et al., *Flow-GRPO: Training Flow Matching Models via Online RL*. arXiv:2505.05470. https://arxiv.org/abs/2505.05470
[^mix-grpo]: Li et al., *MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE*. arXiv:2507.21802. https://arxiv.org/abs/2507.21802
[^tempflow-grpo]: He et al., *TempFlow-GRPO: When Timing Matters for GRPO in Flow Models*. arXiv:2508.04324. https://arxiv.org/abs/2508.04324
[^reinflow]: Zhang et al., *ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning*. arXiv:2505.22094. https://arxiv.org/abs/2505.22094
[^fpo]: McAllister et al., *Flow Matching Policy Gradients*. arXiv:2507.21053. https://arxiv.org/abs/2507.21053
[^pa-rl]: Mark et al., *Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone*. arXiv:2412.06685. https://arxiv.org/abs/2412.06685
[^dsrl]: Wagenmaker et al., *Steering Your Diffusion Policy with Latent Space Reinforcement Learning*. arXiv:2506.15799. https://arxiv.org/abs/2506.15799
