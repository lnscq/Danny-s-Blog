---
title: "Paper Reading Week2"
description: "10/27-11/03"
pubDate: "Oct 29 2025"
image: https://tc.alcy.cc/q/20250908/55b5c637868669f2900ac105bcf22d7b.webp
categories:
  - Paper Reading
tags:
  - Paper
  - RL
---
# 论文信息 
>[Arxiv ID](https://arxiv.org/pdf/2505.05470)

>[幻觉翻译](https://hjfy.top/arxiv/2505.05470)
## Title:**Flow-GRPO: Training Flow Matching Models via Online RL Flow-GRPO训练流匹配模型通过在线强化学习**

---
### **1. Introduction**

介绍Flow matching的强大性能，并提出问题：
> 然而,它们在涉及多个对象、属性和关系的复杂场景组合以及文本渲染方面往往存在困难。

作者提出将在线强化学习（RL）引入到生成模型中，并列举出一些挑战：

- 流模型依赖于基于常微分方程 (ODEs)的确定性生成过程,这意味着它们在推理期间无法进行随机采样。
- 在线 RL 需要高效的采样以收集训练 数据,但流模型通常需要许多迭代步骤来生成每个样本,显著降低了采样效率。

研究提出了**Flow-GRPO**，将GRPO整合到Flow matching模型中，提出了两种关键策略：

- 采用ODE-to-SDE策略，克服了原始flow matching模型的确定性，将ODE转换为等效的随机微分方程，在保留原始边缘分布的前提下引入随机性
- 应用**降噪减少**策略，提到在线RL中的采样效率，即在训练期间减少模型的降噪步骤，在推理阶段保持完整的调度。

### **2. Background**

- LLM的强化学习：
在线强化学习已被证明在提升大型语言模型(LLMs)的推理能力方  面非常有效，如近端策略优化(PPO) 和无价值组相对策略优化(GRPO)，这篇工作采用GRPO以节省内存。

- diffusion & flow matching 模型 

- T2l对齐

### **3.Flow-GRPO**

![对比图片](https://github.com/lnscq/picx-images-hosting/raw/master/image.3uv8uuja38.webp)

受到GRPO算法的启发，通过在线强化学习来改进模型


### **4. Experiment**

# 论文信息 
>[Arxiv ID](https://arxiv.org/pdf/2506.01955)

>[幻觉翻译](https://hjfy.top/arxiv/2506.01955)
## Title:**Dual-Process Image Generation 双过程图像生成**


### **1. Introduction**

当前的大型语言模型在多个领域展现出卓越能力，并具备上下文学习新任务的能力。然而，当它们被训练为多模态模型以联合生成图像和文本时，要么无法达到仅生成图像的保真度，要么难以供学术研究实验使用。相比之下，当代图像生成模型在视觉质量上已接近照片级效果，但与之交互仍常令人沮丧。

受认知科学中双过程理论的启发，该研究提出一种双过程架构，将一个知识丰富的多模态语言模型（VLM，作为“系统2”或“审慎过程”）与一个视觉精确的图像生成器（作为“系统1”或“反射过程”）相结合。该架构通过VLM对生成图像进行评分，并将梯度反向传播以更新图像生成器的权重，从而实现对图像生成过程的精细化控制。

### **2. Background**

- 分类器引导

- 推理时搜索

- 模型微调

- VQA 评分：与 CLIP 相比, VLM 提供了更准确且可解释的评分框架。

### **3.双过程蒸馏**

![对比图片](https://github.com/lnscq/picx-images-hosting/raw/master/image.9rjt1vofme.webp)



### **4. Experiment**
