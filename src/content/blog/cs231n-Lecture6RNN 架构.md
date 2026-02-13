---
title: "CS231n Lecture 7: 循环神经网络（RNN）"
description: "cs231n"
pubDate: "Feb 14 2026"
image: https://tc.alcy.cc/tc/20260121/6b6f4cfab62eba8f064014fd83d6ddaa.webp
categories:
  - notebook
tags:
  - CV
  - RNN
  - Sequence Modeling
---

## CS231n Lecture 7: 循环神经网络（RNN）

[课程链接](https://cs231n.stanford.edu/)

### 引言

卷积神经网络擅长处理具有固定空间结构的数据，如图像，但许多重要任务涉及**变长序列**——视频帧、语音信号或文本词串。这些数据的本质在于其**时间或顺序依赖性**：当前元素的意义往往由其上下文决定。循环神经网络（Recurrent Neural Network, RNN）为此类问题提供了一个自然的建模范式，其核心是通过一个**内部状态**在时间步之间传递信息，从而将历史“记忆”融入对当前输入的理解。

### RNN 的基本结构与计算

![](https://github.com/lnscq/picx-images-hosting/raw/master/image.3rbr6d6903.webp)

RNN 的设计哲学源于一个朴素而强大的想法：用一个固定的计算单元，在每个时间步重复作用于新的输入和累积的历史。具体而言，给定一个长度为 $T$ 的输入序列 $(\mathbf{x}_1, \dots, \mathbf{x}_T)$，RNN 维护一个隐藏状态 $\mathbf{h}_t$，该状态在时间步 $t$ 被更新为：

$$
\mathbf{h}_t = \tanh\left( \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h \right).
$$

这里，$\mathbf{h}_{t-1}$ 编码了前 $t-1$ 个输入的信息，$\mathbf{x}_t$ 是当前时刻的观测，二者通过可学习的权重矩阵 $\mathbf{W}_{hh}$ 和 $\mathbf{W}_{xh}$ 投影到同一空间后相加，再经非线性激活函数 $\tanh$ 输出新的状态 $\mathbf{h}_t$。

这一公式的精妙之处在于**参数共享**：无论序列多长，所有时间步都使用同一组参数 $\{\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h\}$。这使得模型能够泛化到任意长度的序列，并极大地提升了参数效率。可以将 RNN 视作一个带有内部记忆的处理器：它在每个时刻读取新数据，并根据当前记忆和新数据共同更新其内部状态，就像人类在阅读句子时不断整合新词以形成对整体语义的理解。若需在每个时间步产生输出（如预测下一个词），只需在隐藏状态上附加一个线性层：$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$。

### 训练与梯度流：BPTT 及其挑战

![](https://github.com/lnscq/picx-images-hosting/raw/master/image.5trjuf68q0.webp)

训练 RNN 的标准方法是**随时间反向传播（Backpropagation Through Time, BPTT）**。该算法首先将 RNN 沿时间维度展开为一个深度为 $T$ 的前馈网络，然后在此展开图上执行标准的反向传播。然而，这种展开也揭示了 Vanilla RNN 的根本缺陷：**梯度消失与爆炸问题**。

在反向传播中，从时间步 $t$ 到 $t-k$ 的梯度需要连续乘以权重矩阵 $\mathbf{W}_{hh}^\top$ 共 $k$ 次。如果 $\mathbf{W}_{hh}$ 的特征值普遍小于 1，梯度会以指数速度衰减至零；反之则会爆炸。这导致模型难以学习跨越较长时间步的依赖关系——早期输入对后期输出的影响在梯度信号中几乎无法体现。实践中，梯度裁剪（Gradient Clipping）可缓解爆炸问题，但对于更普遍的梯度消失，必须从架构层面寻求解决方案。

### LSTM：通过门控机制解决长期依赖

![](https://github.com/lnscq/picx-images-hosting/raw/master/image.39lphs7lot.webp)

长短期记忆网络（LSTM）通过引入一套精巧的**门控机制**，从根本上改善了信息在长时间尺度上的流动。其核心创新是分离了两个状态：一个用于快速传递信息的**细胞状态（cell state）** $\mathbf{c}_t$，和一个用于与外界交互的**隐藏状态** $\mathbf{h}_t$。

在每个时间步，LSTM 首先计算三个门：

- **遗忘门** $\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t])$ 决定从细胞状态中丢弃哪些旧信息；
- **输入门** $\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t])$ 控制哪些新信息被写入；
- **候选细胞状态** $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t])$ 提供待写入的新内容。

细胞状态的更新公式为：

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
$$

其中 $\odot$ 表示逐元素相乘。由于此操作主要是元素级的加法与乘法，而非矩阵乘法，梯度在 $\mathbf{c}_t$ 到 $\mathbf{c}_{t-1}$ 的路径上得以近乎无损地回传，形成了一个稳定的“信息高速公路”。最后，**输出门** $\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t])$ 决定基于当前细胞状态输出多少信息作为隐藏状态：$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$。通过这种方式，LSTM 能够自主地选择记住或遗忘信息，从而有效捕获长期依赖。

### RNN 的应用、权衡与现代发展

RNN 及其变体曾是序列建模的主导力量，在机器翻译、语音识别和图像描述生成等任务中取得了里程碑式的成果。其优势在于能优雅地处理变长输入，且模型复杂度与序列长度无关。然而，其固有的**顺序计算**特性使其难以并行化，训练速度慢；即使有 LSTM，极长距离的依赖在实践中依然难以可靠建模。

这些局限性催生了**Transformer**架构，后者完全摒弃循环结构，转而依赖自注意力机制，实现了高度并行化和更强的全局建模能力。近年来，**状态空间模型（SSMs）** 如 Mamba 试图融合 RNN 的线性扩展效率（$O(N)$）与 Transformer 的全局感受野，代表了序列建模范式的新探索方向。

### 总结

RNN 通过其循环连接和共享参数，为序列数据建模提供了一个概念清晰且功能强大的框架。Vanilla RNN 虽然简洁，但受困于梯度流问题；LSTM 通过门控和细胞状态的设计，巧妙地解决了长期依赖学习的难题，成为深度学习发展史上的关键一环。尽管其地位已被 Transformer 等架构部分取代，但 RNN 所蕴含的“状态演化”思想及其对序列动态的直观刻画，依然是理解现代序列模型不可或缺的基础。
