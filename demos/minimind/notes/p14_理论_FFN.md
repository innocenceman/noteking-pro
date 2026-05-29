# 第14集: 理论：FFN

## 课程概览

本集是 MiniMind 第 14/26 集，主题是 **理论：FFN**，重点讲解 Transformer 中的 **前馈网络 Feed Forward Network, FFN**，以及现代大模型中常用的 **SwiGLU 激活函数**。

{IMAGE:1}

在 Transformer Block 中，除了注意力机制外，FFN 是另一个核心计算模块。注意力机制主要负责 **token 之间的信息交互**，而 FFN 更多负责对每个 token 的表示进行 **逐位置的非线性变换与特征加工**。

{IMPORTANT}  
FFN 可以理解为 Transformer 中对每个 token 独立执行的“小型多层感知机”。它不混合不同 token 的信息，而是在隐藏维度上扩展、激活、压缩，从而增强模型表达能力。  
{/IMPORTANT}

本节笔记围绕以下内容展开：

- FFN 在 Transformer Block 中的位置
- 标准 FFN 的结构
- 为什么 FFN 通常先升维再降维
- 激活函数在 FFN 中的作用
- SwiGLU 的结构与公式
- MiniMind 中 FFN 的 PyTorch 实现思路

本节小结：FFN 是 Transformer 中与 Attention 并列的重要模块，负责对 token 表示做非线性加工，是大模型能力的重要来源之一。

---

## FFN 在 Transformer 中的位置

{IMAGE:2}

一个典型的 Transformer Block 通常包含两个主要子模块：

1. Multi-Head Self-Attention
2. Feed Forward Network

常见结构可以写成：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

$$
x = x + \text{FFN}(\text{Norm}(x))
$$

其中：

- $x$ 表示输入的 token hidden states
- Norm 通常是 LayerNorm 或 RMSNorm
- Attention 负责 token 间交互
- FFN 负责每个 token 内部表示的非线性变换
- 残差连接用于稳定训练和保留原始信息

{IMAGE:3}

从计算角度看，如果输入张量形状是：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$ 是 batch size
- $T$ 是序列长度
- $C$ 是 hidden size，也叫 embedding dimension

FFN 会对每个位置上的 $C$ 维向量独立进行相同的变换：

$$
\text{FFN}(x_{b,t}) = f(x_{b,t})
$$

这里的 $f$ 是一个共享的前馈网络。

{KNOWLEDGE}  
FFN 不会直接在序列维度 $T$ 上做混合。也就是说，第 1 个 token 的 FFN 输出只依赖第 1 个 token 在 Attention 后的表示，不直接依赖其他 token。token 间的信息已经由 Attention 完成传递。  
{/KNOWLEDGE}

本节小结：Attention 解决 token 间的信息流动，FFN 解决每个 token 表示本身的特征变换，两者共同构成 Transformer Block 的主体。

---

## 标准 FFN 结构

{IMAGE:5}

最经典的 Transformer FFN 通常由两层线性层和一个非线性激活函数组成：

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

其中：

- $W_1$：第一层线性映射
- $W_2$：第二层线性映射
- $\sigma$：激活函数，例如 ReLU、GELU、SiLU
- $b_1, b_2$：偏置项，部分现代大模型会省略 bias

如果 hidden size 为 $d$，中间层维度为 $d_{ff}$，则：

$$
W_1: d \rightarrow d_{ff}
$$

$$
W_2: d_{ff} \rightarrow d
$$

因此 FFN 的整体结构是：

$$
d \rightarrow d_{ff} \rightarrow d
$$

{IMAGE:6}

在很多 Transformer 模型中，$d_{ff}$ 通常设置为 $4d$。例如：

- hidden size $d = 512$
- FFN 中间维度 $d_{ff} = 2048$

这意味着模型会先把 token 表示从 512 维扩展到 2048 维，在更高维空间中进行非线性加工，再压缩回 512 维。

{IMPORTANT}  
FFN 的“升维-激活-降维”结构，是 Transformer 增强表达能力的重要设计。升维提供更大的特征空间，激活函数引入非线性，降维保证输出形状与残差连接兼容。  
{/IMPORTANT}

本节小结：标准 FFN 是两层 MLP，典型结构为 hidden size 升维到 intermediate size，再降回 hidden size。

---

## 为什么 FFN 要先升维再降维

{IMAGE:7}

如果 FFN 只是一个线性层：

$$
y = Wx
$$

那么无论堆叠多少线性层，本质上仍然可以合并成一个线性变换：

$$
W_2(W_1x) = (W_2W_1)x
$$

这会严重限制模型的表达能力。

所以 FFN 中必须加入激活函数：

$$
y = W_2 \sigma(W_1x)
$$

激活函数让模型能够表达复杂的非线性映射。

{IMAGE:8}

升维的意义可以从特征空间角度理解。原始 token 表示可能是 $d$ 维，在更高的 $d_{ff}$ 维空间中，模型可以学习到更多中间特征：

$$
h = W_1x
$$

$$
h \in \mathbb{R}^{d_{ff}}
$$

然后通过激活函数筛选、调制这些特征：

$$
a = \sigma(h)
$$

最后再映射回原始 hidden size：

$$
y = W_2a
$$

{WARNING}  
不要把 FFN 理解成“改变序列长度”的模块。FFN 改变的是每个 token 的隐藏特征维度，中间会升维，但最终输出维度仍然回到 hidden size。  
{/WARNING}

本节小结：FFN 先升维是为了获得更丰富的中间特征空间，激活函数负责引入非线性，最后降维是为了匹配 Transformer Block 的残差结构。

---

## 激活函数的作用

{IMAGE:9}

激活函数是 FFN 中的关键组成部分。没有激活函数，FFN 就只是线性变换，表达能力不足。

常见激活函数包括：

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

特点：

- 计算简单
- 正数保留，负数置零
- 曾经广泛用于早期神经网络

缺点：

- 负半轴梯度为 0
- 可能出现神经元“死亡”问题

### GELU

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

GELU 在 BERT、GPT 等模型中非常常见。它不是简单地截断负数，而是根据输入大小进行平滑加权。

### SiLU / Swish

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

其中：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

SiLU 的特点是平滑、可导，并且在很多深度模型中表现良好。

{IMAGE:10}

{KNOWLEDGE}  
现代大模型中，激活函数不仅仅是“加一个非线性”。它还会影响梯度传播、训练稳定性、特征选择方式和最终模型性能。  
{/KNOWLEDGE}

本节小结：激活函数让 FFN 具备非线性表达能力，现代语言模型通常使用 GELU、SiLU 或 GLU 变体。

---

## GLU：门控线性单元

{IMAGE:11}

在理解 SwiGLU 前，需要先理解 GLU，Gate Linear Unit，门控线性单元。

普通 FFN 可以写成：

$$
\text{FFN}(x) = W_2 \sigma(W_1x)
$$

而 GLU 引入了一个“门控分支”：

$$
\text{GLU}(x) = (xW_a) \otimes \sigma(xW_b)
$$

其中：

- $W_a$ 是值分支
- $W_b$ 是门控分支
- $\sigma$ 是 sigmoid 激活
- $\otimes$ 表示逐元素相乘

直观理解：

- 一个分支生成候选特征
- 另一个分支生成门控权重
- 门控权重决定哪些特征应该被保留或抑制

{IMAGE:12}

这类似于给特征加了一个“开关”：

$$
\text{output feature} = \text{candidate feature} \times \text{gate value}
$$

如果 gate 接近 1，该特征被保留；如果 gate 接近 0，该特征被压制。

{IMPORTANT}  
GLU 类激活的核心思想是：不是简单对特征做激活，而是用一个分支控制另一个分支的信息流动。  
{/IMPORTANT}

本节小结：GLU 通过门控机制增强了 FFN 的表达能力，为 SwiGLU 奠定了基础。

---

## SwiGLU 激活函数

{IMAGE:13}

SwiGLU 是 GLU 的一种变体，它把 GLU 中的 sigmoid 门控替换为 Swish/SiLU 风格的激活。

常见形式如下：

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_1) \otimes (xW_3)
$$

然后再经过输出投影：

$$
\text{FFN}(x) = W_2 \left( \text{SiLU}(xW_1) \otimes xW_3 \right)
$$

也可以写作：

$$
\text{FFN}(x) = W_{down} \left( \text{SiLU}(W_{gate}x) \otimes W_{up}x \right)
$$

其中：

- $W_{gate}$：门控投影
- $W_{up}$：升维投影
- $W_{down}$：降维投影
- $\otimes$：逐元素乘法
- SiLU 是激活函数

{IMAGE:14}

在很多 LLaMA 类模型中，FFN 通常使用类似结构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # gate_proj: 生成门控分支
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)

        # up_proj: 生成候选特征分支
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        # down_proj: 将中间维度映射回 hidden size
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # gate 分支经过 SiLU 激活
        gate = F.silu(self.gate_proj(x))

        # up 分支提供被调制的候选特征
        up = self.up_proj(x)

        # 逐元素相乘完成门控
        hidden = gate * up

        # 降维回原始 hidden size
        return self.down_proj(hidden)
```

这段代码对应公式：

$$
y = W_{down}(\text{SiLU}(W_{gate}x) \otimes W_{up}x)
$$

{WARNING}  
SwiGLU 中通常有三组线性权重：gate projection、up projection、down projection。不要误以为它仍然只有传统 FFN 的两层线性层。  
{/WARNING}

本节小结：SwiGLU 是一种带门控的 FFN 结构，通过 SiLU 激活分支与 up 分支逐元素相乘，再降维输出。

---

## SwiGLU 与普通 FFN 的区别

{IMAGE:15}

普通 FFN：

$$
y = W_2 \sigma(W_1x)
$$

SwiGLU FFN：

$$
y = W_{down}(\text{SiLU}(W_{gate}x) \otimes W_{up}x)
$$

主要区别如下：

| 对比项 | 普通 FFN | SwiGLU FFN |
|---|---|---|
| 线性层数量 | 2 个 | 3 个 |
| 中间结构 | 单分支 | 双分支门控 |
| 激活方式 | 对单分支激活 | gate 分支激活后调制 up 分支 |
| 表达能力 | 较强 | 通常更强 |
| 常见模型 | Transformer 原始结构、BERT/GPT 早期模型 | LLaMA、PaLM 等现代大模型 |

{IMAGE:16}

从直觉上看，普通 FFN 是：

$$
\text{生成特征} \rightarrow \text{激活} \rightarrow \text{输出}
$$

而 SwiGLU 是：

$$
\text{生成候选特征} + \text{生成门控信号} \rightarrow \text{逐元素调制} \rightarrow \text{输出}
$$

这让模型不仅能生成特征，还能学习“哪些特征在当前上下文中更重要”。

{KNOWLEDGE}  
虽然 FFN 本身不直接混合 token，但它处理的是 Attention 后的 token 表示。因此每个 token 的向量中已经包含上下文信息，SwiGLU 实际是在对上下文融合后的表示进行更精细的非线性加工。  
{/KNOWLEDGE}

本节小结：SwiGLU 相比普通 FFN 多了门控机制，能更灵活地控制中间特征的信息流。

---

## FFN 的维度变化

{IMAGE:17}

假设输入张量形状为：

$$
x \in \mathbb{R}^{B \times T \times d}
$$

对于 SwiGLU FFN：

### gate projection

$$
g = xW_{gate}
$$

形状变化：

$$
[B, T, d] \rightarrow [B, T, d_{ff}]
$$

### up projection

$$
u = xW_{up}
$$

形状变化：

$$
[B, T, d] \rightarrow [B, T, d_{ff}]
$$

### 激活与门控

$$
h = \text{SiLU}(g) \otimes u
$$

形状保持：

$$
[B, T, d_{ff}]
$$

### down projection

$$
y = hW_{down}
$$

形状变化：

$$
[B, T, d_{ff}] \rightarrow [B, T, d]
$$

{IMAGE:18}

所以完整过程是：

$$
[B,T,d]
\rightarrow [B,T,d_{ff}]
\rightarrow [B,T,d_{ff}]
\rightarrow [B,T,d]
$$

在 Transformer Block 中，FFN 输出必须和输入 $x$ 形状一致，才能进行残差连接：

$$
x_{\text{out}} = x + \text{FFN}(\text{Norm}(x))
$$

{WARNING}  
残差连接要求输入和输出形状相同。如果 FFN 最后一层没有降回 hidden size，就无法直接与原输入相加。  
{/WARNING}

本节小结：SwiGLU FFN 中 gate 和 up 分支都会升维到中间维度，逐元素相乘后再通过 down projection 降回 hidden size。

---

## MiniMind 中的 FFN 实现思路

{IMAGE:19}

在 MiniMind 这类从零实现大模型的项目中，FFN 通常会被封装成一个独立模块，并在 Transformer Block 中调用。

一个简化版本如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMindFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        # 门控分支：决定哪些中间特征更重要
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        # 升维分支：生成候选中间特征
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        # 降维分支：映射回 hidden_size
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        # SwiGLU: gate 分支调制 up 分支
        x = gate * up

        # 输出形状恢复为 [batch_size, seq_len, hidden_size]
        x = self.down_proj(x)

        return x
```

在 Transformer Block 中，它可能这样使用：

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = MiniMindFFN(hidden_size, intermediate_size)

    def forward(self, x):
        # Pre-Norm 结构：先归一化，再进入 FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)

        # 残差连接
        x = residual + x

        return x
```

{IMAGE:20}

实际大模型中常使用 RMSNorm 替代 LayerNorm，也可能会配合 dropout、模型并行、张量并行等优化。但核心 FFN 逻辑仍然是：

$$
\text{Norm} \rightarrow \text{SwiGLU FFN} \rightarrow \text{Residual}
$$

本节小结：MiniMind 的 FFN 实现可以采用 LLaMA 风格的 SwiGLU 结构，代码上表现为 gate_proj、up_proj、down_proj 三个线性层。

---

## 参数量与计算量理解

{IMAGE:21}

假设 hidden size 为 $d$，intermediate size 为 $d_{ff}$。

标准 FFN 参数量约为：

$$
d \times d_{ff} + d_{ff} \times d = 2dd_{ff}
$$

SwiGLU FFN 参数量约为：

$$
d \times d_{ff} + d \times d_{ff} + d_{ff} \times d = 3dd_{ff}
$$

因为 SwiGLU 有三个投影矩阵：

- gate projection
- up projection
- down projection

如果直接把 $d_{ff}$ 设为 $4d$，SwiGLU 的参数量会比普通 FFN 更大。因此有些模型会把 SwiGLU 的中间维度调整为约 $\frac{2}{3} \times 4d$，使总参数量接近传统 FFN。

例如：

$$
d_{ff} \approx \frac{8}{3}d
$$

这样 SwiGLU 的参数量：

$$
3d \cdot \frac{8}{3}d = 8d^2
$$

普通 $4d$ FFN 参数量：

$$
2d \cdot 4d = 8d^2
$$

二者大致相当。

{IMPORTANT}  
SwiGLU 虽然多一个线性分支，但可以通过调整 intermediate size 控制总参数量，使其与传统 FFN 接近，同时获得更好的表达能力。  
{/IMPORTANT}

本节小结：SwiGLU 的参数量通常是 $3dd_{ff}$，实际模型会通过设置合适的中间维度来平衡性能和计算成本。

---

## 易错点总结

### 1. FFN 不是卷积，也不是注意力

FFN 不负责 token 之间的信息交换。它对每个 token 位置独立应用同一组 MLP 参数。

### 2. FFN 输出维度必须等于 hidden size

因为 Transformer Block 中需要做残差连接：

$$
x + \text{FFN}(x)
$$

所以 FFN 的输出必须和输入形状一致。

### 3. SwiGLU 不是单纯的 SiLU

SwiGLU 的核心不是只把激活函数换成 SiLU，而是引入了双分支门控：

$$
\text{SiLU}(W_{gate}x) \otimes W_{up}x
$$

### 4. gate 和 up 的输出形状必须一致

因为二者需要逐元素相乘：

$$
[B,T,d_{ff}] \otimes [B,T,d_{ff}]
$$

如果维度不一致，会直接报 shape mismatch。

本节小结：理解 FFN 时要抓住三个关键词：逐位置、升维降维、非线性门控。

---

## 关键收获

1. FFN 是 Transformer Block 的核心模块之一，负责对每个 token 的隐藏表示进行非线性加工。
2. 标准 FFN 通常是 $d \rightarrow d_{ff} \rightarrow d$ 的两层 MLP。
3. FFN 不混合 token 维度，token 间信息交互主要由 Attention 完成。
4. 激活函数使 FFN 具备非线性表达能力。
5. SwiGLU 使用 gate_proj 和 up_proj 两个分支，通过 $\text{SiLU}(gate) \times up$ 实现门控。
6. SwiGLU FFN 通常包含三个线性层：gate_proj、up_proj、down_proj。
7. FFN 最终输出必须回到 hidden size，才能与残差连接相加。

{IMAGE:4}

---

## 思考题

1. 为什么 Transformer 中已经有 Attention 了，还需要 FFN？
2. 如果去掉 FFN 中的激活函数，只保留两层线性层，会发生什么？
3. SwiGLU 相比普通 FFN 多了一个 gate 分支，它可能给模型表达能力带来哪些优势？