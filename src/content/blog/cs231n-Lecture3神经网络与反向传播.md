---
title: "CS231n Lecture 4: 神经网络与反向传播"
description: "cs231n"
pubDate: "Feb 8 2026"
image: https://tc.alcy.cc/tc/20260121/e0f632a423d5c328b898630e1ea58f48.webp
categories:
  - notebook
tags:
  - CV
  - Neural Networks
  - Backpropagation
---

## CS231n Lecture 4: 神经网络与反向传播

[课程链接](https://cs231n.stanford.edu/)

### 引言

线性分类器虽然简洁高效，但其固有的局限性——只能学习线性决策边界——使其无法应对现实世界中普遍存在的复杂、非线性模式。为突破这一限制，我们引入**神经网络**，一种通过堆叠多个线性变换与非线性激活函数构成的通用函数逼近器。本讲将阐述神经网络的基本架构，并重点剖析其训练的核心：**反向传播（Backpropagation）** 算法。

### 从线性分类器到神经网络

一个标准的线性分类器可表示为 $f(\mathbf{x}) = \mathbf{W} \mathbf{x} + \mathbf{b}$。为了增加模型的表达能力，我们可以在其前插入一个或多个中间计算层。一个两层神经网络（或称单隐藏层网络）的形式如下：

$$
\begin{aligned}
\mathbf{h} &= \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \\
\mathbf{s} &= \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2,
\end{aligned}
$$

其中 $\mathbf{h}$ 是隐藏层的激活值，$\text{ReLU}(z) = \max(0, z)$ 是**激活函数**。这种结构被称为**全连接网络（Fully-Connected Network）** 或**多层感知机（MLP）**。

关键在于，若没有非线性激活函数（即 $\mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$），整个网络将退化为一个简单的线性映射：$\mathbf{s} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)$。无论堆叠多少层，其表达能力都等同于单层线性分类器。因此，**非线性激活函数是神经网络具备强大表示能力的根本原因**，它使得网络能够学习复杂的特征层次和决策边界。

### 激活函数的选择

激活函数负责在网络中引入非线性。历史上曾使用 Sigmoid ($\sigma(z) = 1/(1+e^{-z})$) 和 Tanh 函数，但它们在深度网络中易导致梯度消失问题。现代神经网络几乎普遍采用 **ReLU（Rectified Linear Unit）** 作为默认激活函数：

$$
\text{ReLU}(z) = \max(0, z).
$$

ReLU 计算简单，且在正区间的梯度恒为 1，有效缓解了梯度消失问题，极大地促进了深度网络的训练。尽管存在“神经元死亡”（输出恒为零）的潜在风险，但其在实践中的优异表现使其成为首选。

### 网络架构与容量

神经网络的架构由其层数和每层的神经元数量（即宽度）决定。增加层数（深度）或每层的神经元数（宽度）会提升模型的**容量（Capacity）**，即拟合复杂函数的能力。然而，更大的容量也意味着更高的过拟合风险。因此，网络规模不应作为主要的正则化手段；应优先使用如 L2 正则化、Dropout 等显式正则化技术来控制泛化误差。

### 反向传播：高效梯度计算

训练神经网络需要计算损失函数 $\mathcal{L}$ 关于所有参数（如 $\mathbf{W}_1, \mathbf{W}_2$）的梯度。对于包含成千上万个参数的复杂网络，手动推导解析梯度不切实际。**反向传播**提供了一种高效、系统化的解决方案。

反向传播的本质是**链式法则（Chain Rule）** 在计算图上的递归应用。任何复杂的函数都可以分解为一系列基本运算（如加法、乘法、函数应用），这些运算构成一个**计算图（Computational Graph）**。反向传播通过一次从输出到输入的反向遍历，高效地计算出损失对所有中间变量和参数的梯度。

考虑一个简单函数 $f(x, y, z) = (x + y) z$。其计算图包含两个节点：$q = x + y$ 和 $f = q z$。假设我们已知上游梯度 $\partial \mathcal{L} / \partial f$，反向传播首先计算局部梯度 $\partial f / \partial q = z$ 和 $\partial f / \partial z = q$，然后利用链式法则得到：

$$
\frac{\partial \mathcal{L}}{\partial q} = \frac{\partial \ \mathcal{L}}{\partial f} \frac{\partial f}{\partial q}, \quad \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial f} \frac{\partial f}{\partial z}.
$$

接着，将 $\partial \mathcal{L} / \partial q$ 作为上游梯度传递给 $q = x + y$ 节点，计算 $\partial \mathcal{L} / \partial x$ 和 $\partial \mathcal{L} / \partial y$。这个过程可以推广到任意复杂的计算图。

### 向量与矩阵的反向传播

在神经网络中，变量通常是向量或矩阵。例如，对于矩阵乘法 $\mathbf{Y} = \mathbf{X} \mathbf{W}$，我们需要计算 $\partial \mathcal{L} / \partial \mathbf{X}$ 和 $\partial \mathcal{L} / \partial \mathbf{W}$。直接构造雅可比矩阵（Jacobian）在高维下内存消耗巨大（例如，对于 $64 \times 4096$ 的矩阵，雅可比矩阵可能占用数百GB内存）。反向传播的精妙之处在于，它通过**隐式**的矩阵运算直接计算梯度，而无需显式构造雅可比矩阵。

对于 $\mathbf{Y} = \mathbf{X} \mathbf{W}$，给定上游梯度 $\partial \mathcal{L} / \partial \mathbf{Y}$，其梯度公式为：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{W}^\top, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}.
$$

这些公式的正确性可以通过维度匹配直观验证，它们是深度学习框架（如 PyTorch、TensorFlow）自动微分系统的核心。

### 实现模式：模块化与自动微分

现代深度学习库采用**模块化**设计。每个基本操作（如线性层、ReLU、Sigmoid）都被实现为一个具有 `forward` 和 `backward` 方法的模块。`forward` 方法执行前向计算并缓存必要的中间变量；`backward` 方法接收上游梯度，并利用缓存的变量计算并返回下游梯度。通过组合这些模块，可以构建任意复杂的网络，并由框架自动完成整个反向传播过程。这种设计将复杂的全局梯度计算分解为一系列简单的局部梯度计算，极大地简化了模型开发。

### 总结

神经网络通过堆叠线性层与非线性激活函数，构建了强大的非线性函数逼近能力。ReLU 激活函数因其简单性和有效性成为标准选择。而反向传播算法，则是训练这些复杂模型的基石，它通过在计算图上高效地应用链式法则，使得计算数百万甚至数十亿参数的梯度成为可能。理解反向传播不仅是掌握深度学习的关键，也为后续学习卷积神经网络等更高级架构奠定了坚实基础。
