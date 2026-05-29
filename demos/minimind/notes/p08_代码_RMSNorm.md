# 第8集: 代码：RMSNorm

## 课程导言：本节要实现什么

本集是 MiniMind 课程第 8 集，主题是 **代码：RMSNorm**，目标是在 PyTorch 中从零实现大模型常用的归一化层 **RMSNorm**。RMSNorm 是很多现代 Transformer 架构中的基础组件，例如 LLaMA 系列常用的就是 RMSNorm，而不是传统的 LayerNorm。

{IMAGE:4}

在大模型结构中，归一化层通常出现在注意力层、前馈网络层的前后，用来稳定激活值分布，改善训练稳定性。MiniMind 作为一个从零手敲的大模型项目，需要先把这些基础模块逐个实现出来，后续才能组合成完整的 Transformer Block。

{IMPORTANT}本节核心任务：理解 RMSNorm 的数学含义，并用 PyTorch 写出一个可复用的 `RMSNorm` 模块。{/IMPORTANT}

### 本节小结

本节不是单纯写几行代码，而是要理解 RMSNorm 为什么这样写、每个张量维度代表什么、以及它和 LayerNorm 的区别。

---

## RMSNorm 的背景知识

### 为什么大模型需要归一化

神经网络训练时，每一层的输入分布会随着参数更新而变化。如果激活值尺度过大或过小，可能导致梯度不稳定、训练速度变慢，甚至出现 loss 不收敛的问题。归一化层的作用，就是把输入张量调整到相对稳定的数值范围。

{IMAGE:5}

常见的归一化方法包括：

- BatchNorm：常用于 CNN，对 batch 维度做统计。
- LayerNorm：常用于 Transformer，对特征维度做均值和方差归一化。
- RMSNorm：LayerNorm 的简化变体，只使用均方根，不减均值。

{KNOWLEDGE}Transformer 中通常不使用 BatchNorm，因为序列长度、batch 构造和自回归生成场景会让 batch 统计不稳定。LayerNorm 和 RMSNorm 更适合语言模型。{/KNOWLEDGE}

### LayerNorm 的基本形式

LayerNorm 通常会对最后一个维度做归一化。假设输入为 $x$，其最后一维长度为 $d$，则 LayerNorm 大致形式为：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

LayerNorm 会先减去均值，再除以标准差，最后乘以可学习缩放参数 $\gamma$，加上可学习偏置 $\beta$。

### RMSNorm 的简化思路

RMSNorm 的核心思想是：**不减均值，只根据均方根 Root Mean Square 调整输入尺度**。

其公式为：

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

其中：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

也可以写成：

$$
\text{RMSNorm}(x) = x \cdot \frac{1}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

{IMPORTANT}RMSNorm 只关心向量的整体尺度，不强制把均值移到 0。它比 LayerNorm 更简单，计算量也更低。{/IMPORTANT}

### 本节小结

RMSNorm 可以看作 LayerNorm 的轻量化版本：去掉均值中心化和偏置项，只保留基于均方根的尺度归一化。

---

## RMSNorm 在大模型中的位置

在 MiniMind 这样的 Transformer 模型中，RMSNorm 通常用于每个子层之前，也就是常见的 **Pre-Norm** 结构。

{IMAGE:6}

一个典型的 Transformer Block 可能类似：

```python
x = x + self.attention(self.attention_norm(x))
x = x + self.feed_forward(self.ffn_norm(x))
```

这里的 `attention_norm` 和 `ffn_norm` 就可以是 RMSNorm。它们分别在注意力模块和前馈网络模块之前对输入做归一化。

### 为什么很多大模型选择 RMSNorm

RMSNorm 的优点包括：

- 计算更简单，不需要计算均值。
- 参数更少，通常只有一个缩放参数 `weight`。
- 在大语言模型中效果稳定。
- 适合 Pre-Norm Transformer 架构。
- 推理阶段计算开销较低。

{IMAGE:7}

{KNOWLEDGE}LLaMA、Qwen 等现代大模型结构中都大量使用 RMSNorm。理解它的代码实现，是继续理解 Transformer Block 的前提。{/KNOWLEDGE}

### 本节小结

RMSNorm 是大模型基础模块之一，虽然代码很短，但它会在模型中被反复调用，直接影响训练和推理稳定性。

---

## 数学公式拆解

### 输入张量的形状

在语言模型中，输入张量通常是三维：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$：batch size，批大小。
- $T$：sequence length，序列长度。
- $C$：hidden size，也叫 embedding dimension。

RMSNorm 一般对最后一维 $C$ 做归一化，也就是对每个 token 的 hidden vector 单独处理。

{IMAGE:1}

如果某个 token 的 hidden vector 是：

$$
x = [x_1, x_2, ..., x_C]
$$

则 RMSNorm 会计算：

$$
\text{mean}(x^2) = \frac{x_1^2 + x_2^2 + ... + x_C^2}{C}
$$

然后得到：

$$
r = \frac{1}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

最终输出：

$$
y_i = x_i \cdot r \cdot w_i
$$

其中 $w_i$ 是可学习参数。

### 为什么要加 $\epsilon$

实际计算时，如果 $\text{mean}(x^2)$ 非常接近 0，直接开方再求倒数可能导致数值不稳定。因此会加一个很小的常数 $\epsilon$：

$$
\frac{1}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

常见取值是：

```python
eps = 1e-6
```

{WARNING}不要省略 `eps`。在深度学习代码中，`eps` 通常是防止除零、NaN、Inf 的关键保护。{/WARNING}

### 本节小结

RMSNorm 的数学过程可以概括为：平方、求均值、加 eps、开根号、取倒数、乘回原输入、再乘可学习权重。

---

## PyTorch 实现思路

### 模块结构

在 PyTorch 中，自定义网络层通常继承 `nn.Module`：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        pass
```

这里有两个重要成员：

- `self.eps`：数值稳定项。
- `self.weight`：可学习缩放参数，形状为 `[dim]`。

{IMAGE:2}

### 为什么 `weight` 初始化为 1

RMSNorm 的输出中会乘上 `weight`：

$$
y = \hat{x} \cdot w
$$

如果一开始把 `weight` 初始化为 1，就表示刚开始不额外改变归一化后的尺度，让模型在初始阶段更稳定。

```python
self.weight = nn.Parameter(torch.ones(dim))
```

`nn.Parameter` 的作用是告诉 PyTorch：这是模型参数，需要参与梯度更新。

{IMPORTANT}`torch.ones(dim)` 只是普通张量，包上 `nn.Parameter` 后才会被 `model.parameters()` 收集并由优化器更新。{/IMPORTANT}

### 本节小结

RMSNorm 的类结构很简单：初始化保存 `eps` 和可学习权重，前向传播中完成归一化计算。

---

## 核心代码实现

### 标准实现

一个常见 RMSNorm 实现如下：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, dim]
        # 对最后一个维度 dim 计算均方值
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 将输入按 RMS 缩放，再乘以可学习权重
        return self.weight * x * norm
```

{IMAGE:8}

这里最核心的一行是：

```python
norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

逐步拆解：

```python
x.pow(2)
```

表示对输入每个元素平方。

```python
.mean(dim=-1, keepdim=True)
```

表示沿最后一个维度求均值，并保留维度。

```python
+ self.eps
```

用于数值稳定。

```python
torch.rsqrt(...)
```

表示计算 reciprocal square root，也就是：

$$
\text{rsqrt}(z) = \frac{1}{\sqrt{z}}
$$

所以这行代码等价于：

```python
norm = 1.0 / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

但 `torch.rsqrt` 更直接，也可能更高效。

### 为什么 `keepdim=True`

假设输入形状是：

```python
x.shape == [B, T, C]
```

计算：

```python
x.pow(2).mean(dim=-1)
```

得到形状：

```python
[B, T]
```

如果要和原始的 `x` 相乘，会缺少最后一个维度。使用：

```python
x.pow(2).mean(dim=-1, keepdim=True)
```

得到形状：

```python
[B, T, 1]
```

这样就可以通过广播机制和 `[B, T, C]` 相乘。

{WARNING}`keepdim=True` 是 RMSNorm 实现中的常见细节。如果省略，某些场景下广播会失败，或者得到不符合预期的形状。{/WARNING}

### 本节小结

RMSNorm 的核心代码就是沿最后一维计算 RMS 缩放因子，再利用广播机制乘回输入。

---

## 类型精度与稳定性

### float16 / bfloat16 场景

大模型训练和推理中经常使用半精度，例如 `float16` 或 `bfloat16`。半精度可以节省显存、提升速度，但数值范围和精度更有限。

一种更稳妥的实现会先把输入转成 `float32` 计算归一化，再转回原来的 dtype：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # 使用 float32 计算归一化，提升数值稳定性
        x = x.float()

        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = x * norm

        # 转回输入原本的数据类型，再乘可学习权重
        return self.weight * output.to(input_dtype)
```

{IMAGE:9}

这种写法在大模型实现中非常常见，因为归一化层对数值稳定性比较敏感。

### 权重 dtype 的广播

`self.weight` 的形状是 `[dim]`，输入 `x` 的形状一般是 `[B, T, dim]`。PyTorch 会自动把 `[dim]` 广播到 `[B, T, dim]`。

如果 `x` 是半精度，而 `weight` 是 float32，PyTorch 会根据类型提升规则处理计算。实际工程中也可以更显式地写成：

```python
return self.weight.to(output.dtype) * output
```

不过在许多教学实现中，直接写：

```python
return self.weight * output
```

已经足够清晰。

{WARNING}半精度训练时，归一化层里的平方、均值、开方操作更容易暴露数值误差。将中间计算转为 float32 是常见稳健做法。{/WARNING}

### 本节小结

为了适配大模型训练，RMSNorm 实现不仅要写对公式，还要注意 dtype，尤其是半精度场景下的数值稳定性。

---

## 与 LayerNorm 的对比

### 计算步骤对比

LayerNorm：

$$
y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \cdot \gamma + \beta
$$

RMSNorm：

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma
$$

{IMAGE:10}

二者主要区别：

| 对比项 | LayerNorm | RMSNorm |
|---|---|---|
| 是否减均值 | 是 | 否 |
| 是否使用方差 | 是 | 否，使用均方根 |
| 参数 | `weight` 和 `bias` | 通常只有 `weight` |
| 计算量 | 略高 | 略低 |
| 常见用途 | Transformer 广泛使用 | 大语言模型中非常常见 |

### 直观理解

LayerNorm 关心的是：

- 数据中心在哪里。
- 数据分散程度多大。

RMSNorm 更关心：

- 这个向量整体尺度有多大。

如果向量整体数值很大，就缩小；如果整体数值很小，就放大。它不要求向量均值为 0。

{KNOWLEDGE}RMSNorm 的假设是：对很多深度模型来说，控制激活尺度已经足够重要，不一定每次都需要中心化。{/KNOWLEDGE}

### 本节小结

RMSNorm 相比 LayerNorm 更轻量，少了均值中心化和偏置项，但依然能有效稳定大模型训练。

---

## 代码中的维度推演

### 示例输入

假设：

```python
B = 2
T = 4
C = 8
x = torch.randn(B, T, C)
norm = RMSNorm(C)
y = norm(x)
```

输入形状：

```python
x.shape == torch.Size([2, 4, 8])
```

在前向传播中：

```python
x.pow(2).shape
# [2, 4, 8]
```

```python
x.pow(2).mean(dim=-1, keepdim=True).shape
# [2, 4, 1]
```

```python
norm.shape
# [2, 4, 1]
```

```python
x * norm
# [2, 4, 8]
```

```python
self.weight.shape
# [8]
```

最终：

```python
y.shape
# [2, 4, 8]
```

{IMAGE:11}

### 广播机制说明

PyTorch 广播时会从右往左对齐维度：

```text
x:          [2, 4, 8]
norm:       [2, 4, 1]
weight:           [8]
```

其中：

- `norm` 的最后一维是 1，可以广播到 8。
- `weight` 只有一维 `[8]`，可以对齐到最后一维。

所以：

```python
self.weight * x * norm
```

能够正常得到 `[2, 4, 8]` 的输出。

{WARNING}RMSNorm 的 `dim` 必须和输入最后一维一致。如果模型 hidden size 是 512，那么 `RMSNorm(512)` 才能正确匹配。{/WARNING}

### 本节小结

RMSNorm 的维度逻辑依赖 PyTorch 广播机制。只要 `weight` 的长度等于输入最后一维，就可以自然扩展到 batch 和序列维度。

---

## 在 MiniMind 中的典型使用方式

### 定义模型配置

在大模型项目中，通常会有一个配置对象保存隐藏层维度：

```python
class ModelConfig:
    dim = 512
    n_layers = 8
    n_heads = 8
    vocab_size = 32000
```

然后在模块里使用：

```python
self.norm = RMSNorm(config.dim)
```

{IMAGE:12}

### Transformer Block 中使用 RMSNorm

一个简化版 Transformer Block 可能写成：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, attention, feed_forward):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = attention
        self.feed_forward = feed_forward

    def forward(self, x):
        # 注意力子层：先归一化，再进入 attention，最后残差连接
        x = x + self.attention(self.attention_norm(x))

        # 前馈子层：先归一化，再进入 FFN，最后残差连接
        x = x + self.feed_forward(self.ffn_norm(x))

        return x
```

这里体现的是 Pre-Norm 结构：

```text
Norm -> SubLayer -> Residual
```

而不是：

```text
SubLayer -> Residual -> Norm
```

### 为什么 Pre-Norm 更常见

Pre-Norm 在深层 Transformer 中通常更稳定，因为归一化发生在子层计算之前，可以让每个子模块接收更稳定的输入。

{IMPORTANT}现代大语言模型通常采用 Pre-Norm 结构，RMSNorm 经常直接放在 Attention 和 FFN 之前。{/IMPORTANT}

### 本节小结

RMSNorm 不是孤立模块，它会作为 Transformer Block 的基础组件，服务于注意力层和前馈网络层。

---

## 从零实现时的常见错误

### 错误一：忘记继承 `nn.Module`

错误写法：

```python
class RMSNorm:
    pass
```

正确写法：

```python
class RMSNorm(nn.Module):
    pass
```

如果不继承 `nn.Module`，PyTorch 无法正确管理参数、子模块和设备迁移。

{IMAGE:13}

### 错误二：忘记 `super().__init__()`

错误写法：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        self.weight = nn.Parameter(torch.ones(dim))
```

正确写法：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
```

`super().__init__()` 会初始化 PyTorch 模块内部状态。缺少它可能导致参数注册异常。

### 错误三：`weight` 没有用 `nn.Parameter`

错误写法：

```python
self.weight = torch.ones(dim)
```

正确写法：

```python
self.weight = nn.Parameter(torch.ones(dim))
```

如果不是 `nn.Parameter`，优化器不会更新它。

### 错误四：沿错误维度归一化

错误写法：

```python
x.pow(2).mean(dim=0, keepdim=True)
```

正确写法：

```python
x.pow(2).mean(dim=-1, keepdim=True)
```

对于 `[B, T, C]` 的语言模型张量，应该对 hidden dimension 归一化，也就是最后一维。

{WARNING}不要对 batch 维度或 sequence 维度做 RMSNorm。RMSNorm 的目标是归一化每个 token 的 hidden vector。{/WARNING}

### 本节小结

RMSNorm 代码短，但容易在模块注册、参数定义、归一化维度和广播细节上出错。

---

## 一份更完整的可测试代码

下面是一份更完整的教学版代码，包括简单测试：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # 可学习缩放参数，形状与 hidden size 一致
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入原始 dtype，方便半精度场景下转回去
        input_dtype = x.dtype

        # 使用 float32 做归一化计算，提高数值稳定性
        x_float = x.float()

        # 计算 RMS 的倒数：1 / sqrt(mean(x^2) + eps)
        rms_inv = torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        # 归一化后转回原 dtype
        x_normed = (x_float * rms_inv).to(input_dtype)

        # 乘以可学习缩放参数
        return self.weight * x_normed


if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    x = torch.randn(batch_size, seq_len, hidden_dim)
    norm = RMSNorm(hidden_dim)

    y = norm(x)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("weight shape:", norm.weight.shape)
```

{IMAGE:14}

理论上输出形状应保持不变：

```text
input shape: torch.Size([2, 4, 8])
output shape: torch.Size([2, 4, 8])
weight shape: torch.Size([8])
```

归一化层通常不会改变张量形状，它只改变数值尺度。

### 本节小结

完整实现中要兼顾公式正确性、参数注册、dtype 稳定性和输出形状保持不变。

---

## RMSNorm 的计算直觉

### 从向量长度角度理解

假设某个 token 的隐藏向量为：

$$
x = [3, 4]
$$

则：

$$
\text{mean}(x^2)=\frac{3^2+4^2}{2}=\frac{25}{2}=12.5
$$

$$
\text{RMS}(x)=\sqrt{12.5}
$$

归一化后：

$$
\hat{x} = \frac{x}{\sqrt{12.5+\epsilon}}
$$

这会把原本尺度较大的向量压到更稳定的范围。

{IMAGE:15}

### 为什么叫 RMS

RMS 是 Root Mean Square 的缩写：

- Square：先平方。
- Mean：再求均值。
- Root：最后开平方。

也就是：

$$
\sqrt{\text{mean}(x^2)}
$$

RMS 经常用于衡量信号强度或向量整体幅度。RMSNorm 就是用这个量来规范化隐藏状态。

{KNOWLEDGE}RMS 可以看作一种“尺度估计”。它不关心正负抵消，而是通过平方保留数值幅度。{/KNOWLEDGE}

### 本节小结

RMSNorm 的直觉是：估计每个 hidden vector 的整体尺度，然后把它缩放到稳定范围。

---

## 代码风格与工程实践

### 简洁实现版本

在教学或轻量模型中，可以写得非常简洁：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
```

{IMAGE:16}

### 稳健实现版本

在更接近大模型工程的代码中，可以使用 float32 中间计算：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)
```

这两种写法本质相同，区别在于数值稳定性和工程严谨度。

### 是否需要 bias

RMSNorm 通常不使用 bias。因为它的设计目标是保持简单，只通过 `weight` 控制每个维度的缩放。

如果加上 bias，就会更像 LayerNorm 的参数形式，但这不是经典 RMSNorm 的常见写法。

{WARNING}实现 MiniMind 时应尽量与目标模型结构一致。如果课程设计使用 RMSNorm，就不要随意添加 bias，否则会改变参数量和模型行为。{/WARNING}

### 本节小结

RMSNorm 可以写得很短，但工程中应根据训练精度和模型目标决定是否加入 float32 中间计算。

---

## 与后续模块的关系

### RMSNorm 与 Attention

在注意力层之前使用 RMSNorm，可以让 Query、Key、Value 的输入尺度更稳定：

```python
x_norm = self.attention_norm(x)
attn_out = self.attention(x_norm)
x = x + attn_out
```

{IMAGE:17}

这对注意力分数尤其重要。注意力中通常有：

$$
QK^T
$$

如果输入尺度过大，点积结果也可能变大，导致 softmax 过于尖锐，影响训练稳定。

### RMSNorm 与 Feed Forward

前馈网络通常包含线性层和激活函数：

```python
ffn_out = self.feed_forward(self.ffn_norm(x))
x = x + ffn_out
```

归一化可以让 FFN 接收到尺度稳定的输入，避免激活值过大。

{IMAGE:18}

### RMSNorm 与残差连接

Transformer 中大量使用残差连接：

$$
x_{l+1} = x_l + F(\text{Norm}(x_l))
$$

归一化层和残差连接配合使用，可以让深层网络更容易训练。

{IMPORTANT}RMSNorm、Attention、FFN、Residual Connection 共同组成了现代大模型 Transformer Block 的基础骨架。{/IMPORTANT}

### 本节小结

RMSNorm 是后续 Attention 和 FFN 模块稳定工作的前置条件，是完整大模型结构中的基础零件。

---

## 课堂代码复盘

### 最核心的三行

本集代码中，最值得记住的是：

```python
self.weight = nn.Parameter(torch.ones(dim))
```

```python
x.pow(2).mean(dim=-1, keepdim=True)
```

```python
torch.rsqrt(... + self.eps)
```

{IMAGE:19}

它们分别对应：

- 定义可学习缩放参数。
- 对 hidden dimension 计算平方均值。
- 得到 RMS 的倒数。

### 完整前向传播逻辑

可以把 forward 理解成四步：

```text
输入 x
  -> 计算 x 的平方均值
  -> 计算 RMS 倒数
  -> x 乘 RMS 倒数
  -> 再乘可学习 weight
  -> 输出 y
```

对应公式：

$$
y = x \cdot \frac{1}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}} \cdot w
$$

{IMAGE:20}

### 本节小结

RMSNorm 的实现短小，但它背后对应清晰的数学公式和 Transformer 工程实践。

---

## 关键总结

{IMAGE:3}

{IMPORTANT}RMSNorm 是大语言模型中常见的归一化层，它通过均方根控制 hidden vector 的整体尺度，通常只包含一个可学习缩放参数。{/IMPORTANT}

本集关键点：

- RMSNorm 全称是 Root Mean Square Layer Normalization。
- 它和 LayerNorm 的主要区别是：不减均值，不使用方差中心化，通常没有 bias。
- 对语言模型输入 `[B, T, C]`，RMSNorm 沿最后一维 `C` 做归一化。
- 核心公式是：

$$
\text{RMSNorm}(x)=x \cdot \frac{1}{\sqrt{\text{mean}(x^2)+\epsilon}} \cdot w
$$

- PyTorch 中使用 `nn.Parameter(torch.ones(dim))` 定义可学习缩放参数。
- `keepdim=True` 能保证归一化因子形状适合广播。
- `torch.rsqrt` 表示计算 $1 / \sqrt{x}$。
- 半精度训练时，常见做法是先转成 float32 计算，再转回原 dtype。
- RMSNorm 常用于 Pre-Norm Transformer Block，放在 Attention 和 FFN 之前。

## 思考题

1. RMSNorm 为什么可以不减均值？它牺牲了什么，又换来了什么？
2. 如果输入张量形状是 `[batch, seq_len, hidden_dim]`，为什么 RMSNorm 应该沿 `dim=-1` 计算？
3. 在半精度训练中，如果不把 RMSNorm 的中间计算转为 float32，可能会出现哪些数值问题？