# 第15集: 代码：FFN

## 课程定位：FFN 在 MiniMind 中负责什么

{IMAGE:1}

本集对应 MiniMind 课程第 15/26 集，主题是“代码：FFN”，核心目标是手写 Transformer 中的 `FeedForward` 类，也就是前馈神经网络层。

在 Transformer Block 里，通常有两类核心子结构：

1. 多头注意力层：负责 token 之间的信息交互。
2. 前馈网络层 FFN：负责对每个 token 的隐藏状态做非线性变换和特征增强。

如果把注意力机制理解为“让每个 token 看见上下文”，那么 FFN 可以理解为“让每个 token 独立思考并加工自己的表示”。

{IMPORTANT}FFN 不负责 token 之间的信息交换，它通常逐位置作用在每个 token 的隐藏向量上。上下文交互主要由 Attention 完成，FFN 主要提升模型的表达能力。{/IMPORTANT}

### 本节小结

本集重点不是重新讲 Transformer 原理，而是把 FFN 的数学结构落实为 PyTorch 代码，并理解每一层线性映射、激活函数、门控机制和维度变化。

## FFN 的基本结构

{IMAGE:2}

传统 Transformer 的 FFN 通常写成：

$$
\mathrm{FFN}(x)=W_2 \cdot \sigma(W_1x+b_1)+b_2
$$

其中：

- $x$ 是输入隐藏状态，形状一般为 $(B, T, C)$。
- $B$ 表示 batch size。
- $T$ 表示序列长度。
- $C$ 表示隐藏维度，也就是 `dim`。
- $W_1$ 将隐藏维度从 $C$ 扩展到更大的中间维度。
- $\sigma$ 是非线性激活函数。
- $W_2$ 再把中间维度投影回 $C$。

在大语言模型中，FFN 的中间维度通常比隐藏维度大很多，例如：

$$
d_{ffn} \approx 4d_{model}
$$

也就是说，如果模型隐藏维度是 512，那么 FFN 中间层可能是 2048 左右。

{KNOWLEDGE}为什么要扩展维度？因为先升维再降维可以让模型在更高维空间中进行非线性特征组合，提升表达能力。类似于先把信息展开，再筛选压缩回原始维度。{/KNOWLEDGE}

### 本节小结

FFN 的经典结构是“线性升维 -> 激活函数 -> 线性降维”。它对每个 token 的向量独立处理，但共享同一组参数。

## MiniMind 中的 FeedForward 结构

{IMAGE:6}

MiniMind 里的 `FeedForward` 类通常不是最朴素的两层 MLP，而是采用了更接近现代 LLM 的门控前馈网络结构，例如 SwiGLU 风格。

常见代码结构类似：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()

        # 计算 FFN 中间层维度
        hidden_dim = int(2 * hidden_dim / 3)

        # 将 hidden_dim 调整为 multiple_of 的倍数，便于硬件高效计算
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 第一条分支：用于生成候选特征
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        # 第二条分支：用于生成门控权重
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # 输出投影：把 hidden_dim 映射回 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

这段代码里最关键的是：

```python
F.silu(self.w1(x)) * self.w3(x)
```

它不是简单地对 `w1(x)` 做激活，而是引入了一个额外的门控分支 `w3(x)`。

### 本节小结

MiniMind 的 FFN 采用了门控结构，核心形式是 `SiLU(w1(x)) * w3(x)`，再经过 `w2` 投影回原隐藏维度。

## 输入输出维度分析

{IMAGE:7}

假设输入张量：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$：批大小。
- $T$：序列长度。
- $C$：隐藏维度，即 `dim`。

经过第一层线性变换：

```python
self.w1(x)
```

输出形状为：

$$
(B, T, hidden\_dim)
$$

经过第三层线性变换：

```python
self.w3(x)
```

输出形状同样是：

$$
(B, T, hidden\_dim)
$$

因此二者可以逐元素相乘：

$$
\mathrm{SiLU}(W_1x) \odot W_3x
$$

其中 $\odot$ 表示逐元素乘法。

最后经过：

```python
self.w2(...)
```

形状变回：

$$
(B, T, C)
$$

也就是说 FFN 的输入输出维度保持一致：

$$
\mathrm{FeedForward}: \mathbb{R}^{B \times T \times C} \rightarrow \mathbb{R}^{B \times T \times C}
$$

{WARNING}FFN 内部可以改变中间维度，但最终输出必须回到 `dim`，否则无法和残差连接相加。{/WARNING}

### 本节小结

FFN 的输入输出形状相同，中间会升维到 `hidden_dim`。这一点保证它可以嵌入 Transformer Block 的残差结构中。

## 为什么需要 w1、w2、w3 三个线性层

{IMAGE:8}

朴素 FFN 只需要两个线性层：

```python
x = activation(w1(x))
x = w2(x)
```

但 MiniMind 使用三个线性层：

```python
x = w2(silu(w1(x)) * w3(x))
```

其中：

- `w1`：生成被激活的候选特征。
- `w3`：生成门控分支。
- `w2`：把门控后的中间特征投影回原维度。

数学形式可以写为：

$$
\mathrm{FFN}(x)=W_2(\mathrm{SiLU}(W_1x) \odot W_3x)
$$

这类结构常见于 LLaMA 等现代大模型，被称为 SwiGLU 或类似门控 FFN 的变体。

{KNOWLEDGE}GLU，全称 Gated Linear Unit，即门控线性单元。它的思想是用一条分支生成内容，用另一条分支控制哪些内容通过。{/KNOWLEDGE}

### 本节小结

三个线性层中的 `w1` 和 `w3` 共同完成门控特征变换，`w2` 负责降维输出。门控 FFN 比普通 MLP 具备更强的表达能力。

## SiLU 激活函数

{IMAGE:9}

代码中使用的是：

```python
F.silu(self.w1(x))
```

SiLU 的数学形式为：

$$
\mathrm{SiLU}(x)=x \cdot \sigma(x)
$$

其中：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

因此：

$$
\mathrm{SiLU}(x)=\frac{x}{1+e^{-x}}
$$

SiLU 也常被称为 Swish。它相比 ReLU 更平滑：

- ReLU 在 0 点不可导。
- SiLU 是平滑连续函数。
- SiLU 对负值不是直接截断为 0，而是保留一部分负区间信息。

示例：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = F.silu(x)

print(y)
```

大致趋势是：负数区域被压缩，正数区域近似线性通过。

### 本节小结

SiLU 是现代 LLM 中常见的激活函数，形式为 $x \cdot \sigma(x)$，比 ReLU 更平滑，适合大模型训练。

## hidden_dim 的计算逻辑

{IMAGE:10}

MiniMind 中的 `hidden_dim` 并不是直接使用传入值，而是可能经过两步处理：

```python
hidden_dim = int(2 * hidden_dim / 3)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

第一步：

```python
hidden_dim = int(2 * hidden_dim / 3)
```

这通常和 SwiGLU 的参数量控制有关。

因为门控 FFN 有 `w1` 和 `w3` 两条升维分支，如果仍然使用传统 FFN 的 $4d$ 中间维度，参数量会明显增加。为了保持参数量接近普通 FFN，需要适当缩小中间维度。

普通 FFN 参数量大约为：

$$
d \times 4d + 4d \times d = 8d^2
$$

SwiGLU 风格 FFN 参数量大约为：

$$
d \times h + d \times h + h \times d = 3dh
$$

如果希望：

$$
3dh \approx 8d^2
$$

则：

$$
h \approx \frac{8}{3}d
$$

所以现代 LLM 中经常能看到中间维度被设为 $\frac{2}{3}$ 的某个基础扩展维度。

### 本节小结

`hidden_dim = int(2 * hidden_dim / 3)` 不是随意写的，而是为了在引入门控分支后控制参数量，使其与传统 FFN 大致可比。

## multiple_of 对齐的作用

{IMAGE:11}

第二步：

```python
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

作用是把 `hidden_dim` 向上取整到 `multiple_of` 的倍数。

例如：

```python
hidden_dim = 170
multiple_of = 64
```

计算过程：

$$
\left\lceil \frac{170}{64} \right\rceil = 3
$$

所以：

$$
hidden\_dim = 3 \times 64 = 192
$$

代码中的表达式：

```python
(hidden_dim + multiple_of - 1) // multiple_of
```

是整数向上取整的常见写法。

{IMPORTANT}把隐藏维度对齐到某个倍数，通常是为了更好地适配 GPU 矩阵乘法、Tensor Core 或底层 kernel 的计算效率。{/IMPORTANT}

### 本节小结

`multiple_of` 的作用是维度对齐。它不会改变 FFN 的逻辑功能，但会影响计算效率和模型参数规模。

## bias=False 的含义

{IMAGE:12}

代码里线性层通常写作：

```python
nn.Linear(dim, hidden_dim, bias=False)
```

这表示线性变换只有权重矩阵，没有偏置项。

普通线性层数学形式是：

$$
y = xW^T + b
$$

当 `bias=False` 时，变为：

$$
y = xW^T
$$

为什么大模型里经常不使用 bias？

常见原因包括：

1. 参数量更少。
2. 结构更简洁。
3. 搭配归一化层时，bias 的收益可能有限。
4. 一些现代 LLM 架构本身就采用无 bias 设计。

不过这不是绝对规则。不同模型架构可能有不同选择。

{WARNING}`bias=False` 不代表线性层不是线性变换。它仍然有权重矩阵，只是没有额外偏置向量。{/WARNING}

### 本节小结

MiniMind 的 FFN 使用无偏置线性层，符合很多现代大模型的简洁设计习惯。

## Dropout 的位置

{IMAGE:13}

FeedForward 最后使用：

```python
self.dropout(self.w2(...))
```

也就是在输出投影后加 Dropout。

Dropout 的作用是在训练时随机置零部分神经元输出，降低过拟合风险：

$$
\tilde{x}_i =
\begin{cases}
0, & \text{with probability } p \\
\frac{x_i}{1-p}, & \text{with probability } 1-p
\end{cases}
$$

其中 $p$ 是 dropout 概率。

在 PyTorch 中：

```python
dropout = nn.Dropout(p=0.1)

model.train()
y_train = dropout(x)

model.eval()
y_eval = dropout(x)
```

训练模式下 Dropout 生效，推理模式下 Dropout 不再随机置零。

{WARNING}如果模型处于 `eval()` 模式，Dropout 不会像训练时那样随机丢弃元素。因此调试时要注意 `model.train()` 和 `model.eval()` 的区别。{/WARNING}

### 本节小结

Dropout 用于正则化，MiniMind 中它作用在 FFN 的输出位置。训练时生效，推理时关闭。

## forward 函数逐行拆解

{IMAGE:14}

核心前向传播代码：

```python
def forward(self, x):
    return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

可以拆成更容易理解的版本：

```python
def forward(self, x):
    # 分支 1：线性升维，生成候选特征
    x1 = self.w1(x)

    # 对候选特征使用 SiLU 激活
    x1 = F.silu(x1)

    # 分支 2：线性升维，生成门控信号
    x3 = self.w3(x)

    # 门控融合：逐元素相乘
    x = x1 * x3

    # 输出投影：从 hidden_dim 降回 dim
    x = self.w2(x)

    # Dropout 正则化
    x = self.dropout(x)

    return x
```

这一版更适合初学者阅读，而一行写法更紧凑。

门控部分是核心：

$$
g = \mathrm{SiLU}(W_1x)
$$

$$
u = W_3x
$$

$$
z = g \odot u
$$

$$
y = W_2z
$$

### 本节小结

`forward` 函数可以理解为四步：生成候选特征、生成门控信号、逐元素相乘、投影回原维度。

## FFN 与残差连接的关系

{IMAGE:15}

在完整 Transformer Block 中，FFN 通常不是单独使用，而是放在残差结构中：

```python
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

也就是说 FFN 的输出会和原输入相加：

$$
x_{out}=x+\mathrm{FFN}(\mathrm{Norm}(x))
$$

因此要求：

$$
\mathrm{shape}(x)=\mathrm{shape}(\mathrm{FFN}(x))
$$

这也是为什么 `w2` 必须把 `hidden_dim` 映射回 `dim`。

{IMPORTANT}Transformer Block 中的 Attention 和 FFN 都通常配合残差连接使用。残差连接要求子层输入输出维度一致。{/IMPORTANT}

### 本节小结

FFN 输出维度必须回到原始隐藏维度，因为它要参与残差相加。这是代码维度设计的关键约束。

## FFN 的计算成本

{IMAGE:16}

FFN 在大模型中非常耗参数和计算量。对于隐藏维度 $d$，中间维度 $h$，SwiGLU 风格 FFN 参数量近似为：

$$
d \times h + d \times h + h \times d = 3dh
$$

分别对应：

- `w1`: $d \times h$
- `w3`: $d \times h$
- `w2`: $h \times d$

如果 `bias=False`，就没有额外偏置参数。

例如：

```python
dim = 512
hidden_dim = 1408
```

参数量约为：

$$
3 \times 512 \times 1408 = 2162688
$$

也就是约 216 万参数。

这说明 FFN 并不是 Transformer 中的小配角。在很多模型里，FFN 参数量甚至占据整体参数的大头。

### 本节小结

FFN 的参数量约为 $3dh$，是大模型参数和计算开销的重要来源。

## 与普通 MLP 的对比

{IMAGE:17}

普通 MLP：

```python
y = w2(F.relu(w1(x)))
```

MiniMind 门控 FFN：

```python
y = w2(F.silu(w1(x)) * w3(x))
```

两者区别：

| 对比项 | 普通 FFN | 门控 FFN |
|---|---|---|
| 升维分支 | 1 条 | 2 条 |
| 激活函数 | ReLU/GELU 常见 | SiLU 常见 |
| 特征控制 | 只靠激活函数 | 激活 + 门控 |
| 参数量 | 较少 | 同 hidden_dim 下更多 |
| 表达能力 | 标准 | 更强 |

门控 FFN 的核心优势在于：模型不只是生成特征，还能动态控制特征通过程度。

### 本节小结

MiniMind 使用的门控 FFN 比普通 MLP 多一个分支，能提升表达能力，但也需要通过调整 `hidden_dim` 控制参数量。

## PyTorch 实现细节

{IMAGE:18}

完整实现可以写成：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout):
        super().__init__()

        # SwiGLU 结构中，为了控制参数量，对中间维度做缩放
        hidden_dim = int(2 * hidden_dim / 3)

        # 将 hidden_dim 向上对齐到 multiple_of 的倍数
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        x = F.silu(self.w1(x)) * self.w3(x)

        # x: [batch_size, seq_len, hidden_dim]
        x = self.w2(x)

        # x: [batch_size, seq_len, dim]
        x = self.dropout(x)

        return x
```

需要注意 `nn.Linear` 可以直接处理三维张量。它会默认作用在最后一个维度上。

例如输入：

```python
x.shape == (batch_size, seq_len, dim)
```

经过：

```python
nn.Linear(dim, hidden_dim)(x)
```

输出为：

```python
(batch_size, seq_len, hidden_dim)
```

不需要手动 reshape。

{KNOWLEDGE}PyTorch 的 `nn.Linear(in_features, out_features)` 会对输入张量的最后一维做线性变换，前面的维度会被保留。{/KNOWLEDGE}

### 本节小结

PyTorch 实现 FFN 很简洁，关键是理解 `nn.Linear` 对最后一维生效，以及门控分支的维度必须一致。

## 常见错误与排查

{IMAGE:19}

### 错误一：w2 输入维度写错

错误示例：

```python
self.w2 = nn.Linear(dim, hidden_dim, bias=False)
```

正确写法：

```python
self.w2 = nn.Linear(hidden_dim, dim, bias=False)
```

因为 `w2` 接收的是中间层特征，输入维度应该是 `hidden_dim`。

### 错误二：w1 和 w3 输出维度不一致

错误示例：

```python
self.w1 = nn.Linear(dim, hidden_dim, bias=False)
self.w3 = nn.Linear(dim, dim, bias=False)
```

这样会导致：

```python
F.silu(self.w1(x)) * self.w3(x)
```

维度不匹配。

### 错误三：忘记导入 F

如果使用：

```python
F.silu(...)
```

需要：

```python
import torch.nn.functional as F
```

### 错误四：不理解 Dropout 模式

训练和推理结果不同可能来自 Dropout：

```python
model.train()  # Dropout 生效
model.eval()   # Dropout 关闭
```

### 本节小结

FFN 的常见错误主要集中在维度、导入和训练模式上。调试时优先打印张量形状。

## 用一个例子验证维度

{IMAGE:20}

可以用如下代码验证：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

batch_size = 2
seq_len = 4
dim = 8
hidden_dim = 32
multiple_of = 8
dropout = 0.1

ffn = FeedForward(dim, hidden_dim, multiple_of, dropout)

x = torch.randn(batch_size, seq_len, dim)
y = ffn(x)

print(x.shape)  # torch.Size([2, 4, 8])
print(y.shape)  # torch.Size([2, 4, 8])
```

这说明 FFN 前后形状保持一致。

### 本节小结

通过简单的随机输入可以验证 FFN 是否满足输入输出维度一致的要求，这是写 Transformer 模块时的基本检查。

## 在 Transformer Block 中的位置

{IMAGE:21}

一个典型 Block 可以抽象成：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, attention, feed_forward, norm1, norm2):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm1 = norm1
        self.norm2 = norm2

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
```

这里使用的是 Pre-Norm 结构：

$$
x = x + \mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x = x + \mathrm{FFN}(\mathrm{Norm}(x))
$$

Pre-Norm 的优点是训练深层 Transformer 时通常更稳定。

FFN 在 Attention 之后继续处理每个 token 的表示，使模型能够进一步组合非线性特征。

### 本节小结

FFN 是 Transformer Block 的第二个核心子层，通常放在 Attention 之后，并配合 Norm 和残差连接使用。

## 本集代码的核心流程

{IMAGE:22}

整体流程可以总结为：

```python
输入 x
  |
  |-- w1(x) --> SiLU ----|
  |                      |--> 逐元素相乘 --> w2 --> Dropout --> 输出
  |-- w3(x) -------------|
```

对应数学形式：

$$
y = \mathrm{Dropout}\left(W_2\left(\mathrm{SiLU}(W_1x) \odot W_3x\right)\right)
$$

其中：

- $W_1$：候选特征投影。
- $W_3$：门控特征投影。
- $\mathrm{SiLU}$：非线性激活。
- $\odot$：逐元素乘法。
- $W_2$：输出投影。
- $\mathrm{Dropout}$：训练正则化。

{IMPORTANT}记住这一行代码即可抓住 MiniMind FFN 的核心：`self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))`。{/IMPORTANT}

### 本节小结

MiniMind 的 FFN 是一个 SwiGLU 风格的门控前馈网络，代码虽短，但包含了升维、激活、门控、降维和正则化多个关键设计。

## 关键收获

{IMAGE:3}

1. FFN 是 Transformer Block 中和 Attention 并列的重要模块。
2. Attention 负责 token 间交互，FFN 负责每个 token 内部的非线性特征加工。
3. MiniMind 的 `FeedForward` 使用门控结构，而不是最朴素的两层 MLP。
4. 核心公式是：

$$
\mathrm{FFN}(x)=W_2(\mathrm{SiLU}(W_1x)\odot W_3x)
$$

5. `w1` 和 `w3` 输出维度必须一致，才能逐元素相乘。
6. `w2` 必须把中间维度映射回 `dim`，以支持残差连接。
7. `multiple_of` 用于把中间维度对齐到指定倍数，提高计算友好性。
8. `bias=False` 是现代大模型中常见的简化设计。
9. Dropout 训练时生效，推理时关闭。
10. `nn.Linear` 会默认作用在输入张量最后一维上。

{IMAGE:4}

### 思考题

1. 为什么门控 FFN 中需要把 `hidden_dim` 调整为原来的 $\frac{2}{3}$ 左右，而不是直接使用传统 FFN 的中间维度？
2. 如果把 `F.silu(self.w1(x)) * self.w3(x)` 改成 `F.relu(self.w1(x))`，模型结构和参数量会发生什么变化？
3. 在完整 Transformer Block 中，为什么 FFN 的输出维度必须和输入维度完全一致？

{IMAGE:5}