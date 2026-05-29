# 第7集: 理论：RMSNorm

## 课程定位：为什么要讲 RMSNorm

{IMAGE:6}

本节是 MiniMind 课程第 7 集，主题是 **RMSNorm 原理与数学推导**。在大模型结构中，归一化层是非常关键的稳定训练组件。Transformer 中常见的归一化方法包括：

- LayerNorm
- RMSNorm
- BatchNorm
- GroupNorm

其中，大语言模型里非常常见的是 **RMSNorm**。例如 LLaMA、Qwen 等模型都大量使用 RMSNorm，而不是传统的 LayerNorm。

{IMPORTANT}核心概念{/IMPORTANT}

RMSNorm 的目标是：  
在不改变向量方向主要语义信息的前提下，对每个 token 的隐藏状态向量做尺度归一化，使模型训练更稳定、梯度传播更平滑。

简单说，RMSNorm 做的事情是：

$$
x \rightarrow \frac{x}{\mathrm{RMS}(x)} \cdot \gamma
$$

其中：

- $x$ 是输入隐藏向量
- $\mathrm{RMS}(x)$ 是均方根
- $\gamma$ 是可学习缩放参数

本节会从直觉、数学公式、与 LayerNorm 的区别、PyTorch 实现几个角度理解 RMSNorm。

本节小结：RMSNorm 是大模型中常用的归一化层，它用均方根控制向量尺度，是 LayerNorm 的轻量替代方案。

---

## 一、从归一化的目的开始理解

{IMAGE:1}

神经网络在训练过程中，每一层的输出分布会不断变化。如果某一层输出的数值尺度过大或过小，后续层的计算就会变得不稳定。

例如一个隐藏向量：

$$
x = [10, 20, 30, 40]
$$

它的数值明显偏大。如果后续进入线性层、注意力层或激活函数，可能导致：

- 激活值过大
- softmax 饱和
- 梯度不稳定
- 训练震荡
- 收敛速度变慢

归一化的作用就是把输入向量调整到一个相对稳定的尺度范围内。

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在 Transformer 中，归一化通常不是对整个 batch 做，而是对每个 token 的 hidden dimension 做。假设输入张量形状为：

$$
[B, T, C]
$$

其中：

- $B$：batch size
- $T$：序列长度
- $C$：hidden size，也就是 embedding 维度

RMSNorm 通常沿着最后一维 $C$ 做归一化。

也就是说，对每一个 token 的隐藏向量单独计算 RMS：

$$
x_{b,t} \in \mathbb{R}^{C}
$$

本节小结：RMSNorm 不是为了改变 token 的语义，而是为了控制隐藏向量的数值尺度，让训练更加稳定。

---

## 二、LayerNorm 的回顾

{IMAGE:2}

要理解 RMSNorm，最好先回顾 LayerNorm。LayerNorm 的标准公式是：

$$
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中：

$$
\mu = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2
$$

各符号含义：

- $x$：输入向量
- $n$：向量维度
- $\mu$：均值
- $\sigma^2$：方差
- $\epsilon$：防止除零的小常数
- $\gamma$：可学习缩放参数
- $\beta$：可学习偏移参数

LayerNorm 做了两件事：

1. 去均值：$x - \mu$
2. 除以标准差：$\sqrt{\sigma^2 + \epsilon}$

因此 LayerNorm 归一化后的向量大致具有：

$$
\text{mean} \approx 0
$$

$$
\text{variance} \approx 1
$$

{WARNING}易错点{/WARNING}

LayerNorm 的核心不是简单地把数值变小，而是同时做了 **中心化** 和 **尺度归一化**。

中心化指的是减去均值，让向量整体围绕 0 分布。

本节小结：LayerNorm 会计算均值和方差，既调整中心位置，也调整尺度。

---

## 三、RMSNorm 的核心思想

{IMAGE:7}

RMSNorm 的全称是 **Root Mean Square Layer Normalization**，即均方根归一化。

它的核心公式是：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

其中分母部分：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}
$$

称为 **均方根**。

和 LayerNorm 相比，RMSNorm 最大的区别是：

- LayerNorm 会减去均值
- RMSNorm 不减均值
- LayerNorm 使用标准差
- RMSNorm 使用均方根
- LayerNorm 通常有 $\gamma$ 和 $\beta$
- RMSNorm 通常只有 $\gamma$

{IMPORTANT}核心概念{/IMPORTANT}

RMSNorm 只关心向量的整体尺度，不关心向量是否以 0 为中心。

它认为对大模型来说，很多情况下只要控制隐藏状态的模长或能量大小，就已经足够稳定训练。

本节小结：RMSNorm 是一种更简单的归一化方式，它保留输入均值信息，只根据均方根调整尺度。

---

## 四、均方根 RMS 的数学意义

{IMAGE:8}

给定一个向量：

$$
x = [x_1, x_2, \dots, x_n]
$$

RMS 定义为：

$$
\mathrm{RMS}(x) = \sqrt{\frac{x_1^2 + x_2^2 + \cdots + x_n^2}{n}}
$$

它可以理解为向量元素的“平均能量强度”。

例如：

$$
x = [3, 4]
$$

则：

$$
\mathrm{RMS}(x) = \sqrt{\frac{3^2 + 4^2}{2}}
$$

$$
= \sqrt{\frac{9 + 16}{2}}
$$

$$
= \sqrt{12.5}
$$

$$
\approx 3.535
$$

RMSNorm 会将原向量除以这个值：

$$
\hat{x} = \frac{x}{\mathrm{RMS}(x)}
$$

因此：

$$
\hat{x} = \left[\frac{3}{3.535}, \frac{4}{3.535}\right]
$$

$$
\approx [0.849, 1.131]
$$

归一化后，向量整体尺度被调整到稳定范围。

{KNOWLEDGE}背景知识{/KNOWLEDGE}

RMS 与 L2 范数有密切关系。向量 $x$ 的 L2 范数为：

$$
\|x\|_2 = \sqrt{\sum_{i=1}^{n}x_i^2}
$$

而 RMS 是：

$$
\mathrm{RMS}(x) = \frac{\|x\|_2}{\sqrt{n}}
$$

因此 RMSNorm 本质上是在根据向量的 L2 范数进行尺度归一化。

本节小结：RMS 衡量的是向量元素的整体能量，RMSNorm 通过 RMS 控制隐藏向量的尺度。

---

## 五、RMSNorm 的推导过程

{IMAGE:3}

假设输入向量为：

$$
x = [x_1, x_2, \dots, x_n]
$$

我们希望得到归一化后的向量：

$$
y = [y_1, y_2, \dots, y_n]
$$

第一步，计算平方：

$$
x_i^2
$$

第二步，计算平方均值：

$$
\frac{1}{n}\sum_{i=1}^{n}x_i^2
$$

第三步，开平方得到 RMS：

$$
\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}
$$

第四步，为了数值稳定加入 $\epsilon$：

$$
\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}
$$

第五步，将每个元素除以 RMS：

$$
\hat{x_i} = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2 + \epsilon}}
$$

第六步，乘上可学习参数 $\gamma_i$：

$$
y_i = \gamma_i \cdot \hat{x_i}
$$

最终公式为：

$$
y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2 + \epsilon}}
$$

向量形式为：

$$
y = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \odot \gamma
$$

其中 $\odot$ 表示逐元素相乘。

本节小结：RMSNorm 的推导非常直接，核心就是平方、求均值、开根号、相除、乘可学习权重。

---

## 六、为什么 RMSNorm 不减均值

{IMAGE:9}

LayerNorm 会做：

$$
x - \mu
$$

RMSNorm 不做这一步。

这背后的设计思想是：  
在 Transformer 大模型中，隐藏状态的均值信息可能本身也包含有用信号。如果强制减去均值，可能会丢掉一部分表达能力。

RMSNorm 选择只控制尺度：

$$
x \rightarrow \frac{x}{\mathrm{RMS}(x)}
$$

这样做有几个好处：

1. 计算更简单
2. 参数更少
3. 速度更快
4. 保留均值信息
5. 在大模型中经验效果很好

{IMPORTANT}核心概念{/IMPORTANT}

RMSNorm 可以看作是 LayerNorm 的简化版本：  
它去掉了均值中心化，只保留尺度归一化。

两者对比如下：

| 项目 | LayerNorm | RMSNorm |
|---|---|---|
| 是否减均值 | 是 | 否 |
| 是否计算方差 | 是 | 否 |
| 归一化依据 | 标准差 | 均方根 |
| 可学习参数 | $\gamma, \beta$ | 通常只有 $\gamma$ |
| 计算复杂度 | 较高 | 较低 |
| 大模型使用 | 常见 | 非常常见 |

本节小结：RMSNorm 不减均值，是为了简化计算并保留隐藏状态的原始偏移信息。

---

## 七、RMSNorm 中的可学习参数

{IMAGE:10}

RMSNorm 的输出公式通常写成：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)} \cdot w
$$

这里的 $w$ 或 $\gamma$ 是一个可学习参数，形状通常为：

$$
[C]
$$

也就是和 hidden size 一样长。

如果 hidden size 是 512，那么：

$$
\gamma \in \mathbb{R}^{512}
$$

它的作用是：  
归一化之后，模型仍然可以学习每个通道应该放大或缩小多少。

如果没有 $\gamma$，归一化会把所有通道强行压到统一尺度，表达能力会受限。

例如：

$$
y_i = \gamma_i \cdot \frac{x_i}{\mathrm{RMS}(x)}
$$

不同维度有不同的 $\gamma_i$，模型可以自动学习哪些维度更重要。

{WARNING}易错点{/WARNING}

$\gamma$ 不是一个标量，而通常是一个向量。  
它会对 hidden dimension 上的每个通道分别缩放。

本节小结：RMSNorm 的可学习参数 $\gamma$ 用来恢复和调节模型表达能力。

---

## 八、epsilon 的作用

{IMAGE:11}

RMSNorm 中通常会看到一个很小的数 $\epsilon$：

$$
\frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}
$$

它的作用是防止分母为 0。

例如，如果输入向量全是 0：

$$
x = [0, 0, 0, 0]
$$

那么：

$$
\mathrm{mean}(x^2) = 0
$$

如果没有 $\epsilon$：

$$
\sqrt{0} = 0
$$

此时：

$$
\frac{x}{0}
$$

会产生数值错误。

加入 $\epsilon$ 后：

$$
\sqrt{0 + \epsilon}
$$

分母不再为 0。

常见取值包括：

$$
\epsilon = 10^{-5}
$$

或：

$$
\epsilon = 10^{-6}
$$

在很多大模型实现里，RMSNorm 的 $\epsilon$ 会设置得比较小，例如：

$$
1e-6
$$

本节小结：$\epsilon$ 是数值稳定项，主要用于避免除零和浮点不稳定。

---

## 九、从张量维度看 RMSNorm

{IMAGE:12}

在 PyTorch 中，输入通常不是单个向量，而是一个三维张量：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

例如：

$$
x.shape = [2, 4, 8]
$$

表示：

- batch size = 2
- sequence length = 4
- hidden size = 8

RMSNorm 对最后一维做归一化：

$$
\mathrm{mean}(x^2, \text{dim}=-1, \text{keepdim}=True)
$$

得到的 RMS 形状是：

$$
[B, T, 1]
$$

然后通过广播机制作用到：

$$
[B, T, C]
$$

计算过程为：

$$
r_{b,t} = \sqrt{\frac{1}{C}\sum_{i=1}^{C}x_{b,t,i}^{2} + \epsilon}
$$

$$
y_{b,t,i} = \gamma_i \cdot \frac{x_{b,t,i}}{r_{b,t}}
$$

{KNOWLEDGE}背景知识{/KNOWLEDGE}

`keepdim=True` 非常重要。它让计算得到的 RMS 保留最后一维，便于后续广播相除。

如果没有 `keepdim=True`，张量形状可能从：

```python
[B, T, C]
```

变成：

```python
[B, T]
```

后续广播可能不符合预期。

本节小结：RMSNorm 通常沿 hidden dimension 做归一化，每个 token 单独计算自己的 RMS。

---

## 十、PyTorch 手写 RMSNorm

{IMAGE:13}

下面是一个简洁的 RMSNorm 实现：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 每个 hidden channel 一个可学习缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]

        # 1. 计算最后一维的平方均值
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # 2. 计算 RMS，并进行归一化
        x_norm = x * torch.rsqrt(mean_square + self.eps)

        # 3. 乘以可学习参数
        return self.weight * x_norm
```

这里有一个关键函数：

```python
torch.rsqrt(t)
```

它表示：

$$
\frac{1}{\sqrt{t}}
$$

因此：

```python
x * torch.rsqrt(mean_square + self.eps)
```

等价于：

$$
\frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}
$$

这种写法比先 `sqrt` 再除法更常见，也更贴近高性能实现。

本节小结：RMSNorm 的代码实现非常短，核心就是 `x.pow(2).mean()` 和 `torch.rsqrt()`。

---

## 十一、与 MiniMind 模型结构的关系

{IMAGE:14}

在 MiniMind 这类从零实现大模型的课程中，RMSNorm 通常会出现在 Transformer Block 的关键位置。

常见结构是 Pre-Norm Transformer：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

也就是：

1. 先对输入做 RMSNorm
2. 再送入 Attention
3. 使用残差连接
4. 再做 RMSNorm
5. 送入 MLP
6. 再使用残差连接

对应公式可以写为：

$$
h = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
y = h + \mathrm{MLP}(\mathrm{RMSNorm}(h))
$$

这种结构有利于训练深层 Transformer。

{IMPORTANT}核心概念{/IMPORTANT}

RMSNorm 常用于 Pre-Norm 架构中。  
它不是单独决定模型能力的模块，但对训练稳定性非常重要。

本节小结：在 MiniMind 的 Transformer Block 中，RMSNorm 通常位于 Attention 和 MLP 之前。

---

## 十二、RMSNorm 与残差连接的配合

{IMAGE:15}

Transformer 中大量使用残差连接：

$$
x_{out} = x + F(x)
$$

如果没有归一化，随着层数加深，隐藏状态的尺度可能不断累积，导致数值越来越大。

RMSNorm 可以在进入子模块前控制输入尺度：

$$
F(\mathrm{RMSNorm}(x))
$$

于是残差结构变成：

$$
x_{out} = x + F(\mathrm{RMSNorm}(x))
$$

这样做的好处是：

- 子模块输入更稳定
- Attention 分数更稳定
- MLP 激活更稳定
- 梯度传播更顺畅

不过要注意，RMSNorm 并不会直接归一化残差相加后的结果，除非下一层再次使用 RMSNorm。

本节小结：RMSNorm 与残差连接配合，能缓解深层网络中数值尺度不断漂移的问题。

---

## 十三、RMSNorm 的数值例子

{IMAGE:16}

假设输入为：

$$
x = [1, 2, 3, 4]
$$

第一步，平方：

$$
x^2 = [1, 4, 9, 16]
$$

第二步，求平方均值：

$$
\mathrm{mean}(x^2) = \frac{1 + 4 + 9 + 16}{4}
$$

$$
= \frac{30}{4}
$$

$$
= 7.5
$$

第三步，开平方：

$$
\mathrm{RMS}(x) = \sqrt{7.5}
$$

$$
\approx 2.7386
$$

第四步，归一化：

$$
\hat{x} = \frac{x}{2.7386}
$$

$$
\approx [0.365, 0.730, 1.095, 1.461]
$$

如果 $\gamma = [1, 1, 1, 1]$，输出就是：

$$
y \approx [0.365, 0.730, 1.095, 1.461]
$$

如果：

$$
\gamma = [1, 2, 1, 0.5]
$$

则：

$$
y \approx [0.365, 1.460, 1.095, 0.730]
$$

本节小结：RMSNorm 先统一尺度，再由 $\gamma$ 学习每个维度的缩放强度。

---

## 十四、RMSNorm 和向量方向

{IMAGE:17}

RMSNorm 的一个重要直觉是：  
它主要改变向量长度，而不是随意改变向量方向。

忽略 $\gamma$ 时：

$$
\hat{x} = \frac{x}{\mathrm{RMS}(x)}
$$

这是用一个标量除以整个向量。所有维度都除以同一个数。

因此，向量内部各维度之间的比例保持不变：

$$
\frac{\hat{x_i}}{\hat{x_j}} = \frac{x_i}{x_j}
$$

这说明 RMSNorm 在未乘 $\gamma$ 之前不会破坏向量方向，只是调整整体尺度。

乘上 $\gamma$ 后，每个维度会被不同权重缩放，模型获得了进一步调整表达的能力。

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在表示学习中，向量方向往往与语义相关，向量长度则可能影响数值稳定性。RMSNorm 的设计可以理解为优先控制长度，同时尽量保留方向信息。

本节小结：RMSNorm 通过统一标量缩放隐藏向量，主要控制尺度，较少干扰向量内部比例。

---

## 十五、为什么大模型偏爱 RMSNorm

{IMAGE:18}

很多现代大语言模型采用 RMSNorm，主要因为它有较好的工程折中。

相比 LayerNorm，RMSNorm 少了均值计算和中心化步骤：

LayerNorm：

$$
\mu = \mathrm{mean}(x)
$$

$$
\sigma^2 = \mathrm{mean}((x-\mu)^2)
$$

RMSNorm：

$$
r = \sqrt{\mathrm{mean}(x^2) + \epsilon}
$$

因此 RMSNorm：

- 更简单
- 更快
- 更省计算
- 参数更少
- 对大模型训练足够有效

尤其当模型规模很大时，每个小的计算节省都会被放大。

{WARNING}易错点{/WARNING}

RMSNorm 并不是“永远比 LayerNorm 更好”。  
它是在大模型实践中表现非常好的一种选择，但具体任务和架构仍然可能影响最终效果。

本节小结：RMSNorm 的优势来自简化计算和良好经验效果，因此被广泛用于现代大语言模型。

---

## 十六、RMSNorm 的常见实现细节

{IMAGE:19}

在实际代码中，RMSNorm 还会涉及一些细节。

### 1. 数据类型处理

在混合精度训练中，输入可能是 `float16` 或 `bfloat16`。为了数值稳定，有些实现会先转成 `float32` 计算，再转回原类型：

```python
def forward(self, x):
    input_dtype = x.dtype

    # 用 float32 计算 RMS，减少半精度下的数值误差
    x_float = x.float()

    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + self.eps)

    # 转回原始 dtype，并乘以可学习权重
    return (self.weight * x_norm).to(input_dtype)
```

这里变量名有时叫 `variance`，但严格来说它不是方差，因为没有减均值。它实际表示：

$$
\mathrm{mean}(x^2)
$$

### 2. 参数初始化

RMSNorm 的 `weight` 通常初始化为全 1：

```python
self.weight = nn.Parameter(torch.ones(dim))
```

这样模型一开始不会额外改变归一化后的尺度。

### 3. 是否需要 bias

RMSNorm 通常不使用 bias：

```python
self.bias = None
```

因为它不做中心化，也通常不需要额外平移参数。

本节小结：工程实现中要注意 dtype、参数初始化和变量命名，尤其不要把 RMSNorm 的平方均值误解成标准方差。

---

## 十七、RMSNorm 在前向传播中的完整流程

{IMAGE:20}

假设输入张量：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

完整计算如下：

1. 输入隐藏状态：

$$
x_{b,t,:}
$$

2. 对最后一维平方：

$$
x_{b,t,i}^{2}
$$

3. 对 hidden dimension 求平均：

$$
m_{b,t} = \frac{1}{C}\sum_{i=1}^{C}x_{b,t,i}^{2}
$$

4. 加上数值稳定项：

$$
m_{b,t} + \epsilon
$$

5. 计算倒平方根：

$$
s_{b,t} = \frac{1}{\sqrt{m_{b,t} + \epsilon}}
$$

6. 缩放输入：

$$
\hat{x}_{b,t,i} = x_{b,t,i} \cdot s_{b,t}
$$

7. 乘以可学习权重：

$$
y_{b,t,i} = \gamma_i \cdot \hat{x}_{b,t,i}
$$

输出形状仍然是：

$$
y \in \mathbb{R}^{B \times T \times C}
$$

这保证 RMSNorm 可以无缝插入 Transformer Block 中，不改变张量形状。

本节小结：RMSNorm 只改变数值，不改变输入输出形状。

---

## 十八、和代码中的 MiniMind RMSNorm 对齐

{IMAGE:21}

在从零实现 MiniMind 时，RMSNorm 类通常可以写成：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

逐行解释：

```python
self.weight = nn.Parameter(torch.ones(dim))
```

创建可训练参数 $\gamma$。

```python
x.pow(2)
```

计算每个元素的平方。

```python
.mean(-1, keepdim=True)
```

沿最后一维计算平方均值。

```python
torch.rsqrt(...)
```

计算倒平方根：

$$
\frac{1}{\sqrt{\cdot}}
$$

```python
x * torch.rsqrt(...)
```

完成归一化。

```python
.type_as(x)
```

把结果转换回输入张量原来的数据类型。

{IMPORTANT}核心概念{/IMPORTANT}

如果你能看懂这一行，就基本掌握了 RMSNorm：

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
```

它对应的数学公式就是：

$$
\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}}
$$

本节小结：MiniMind 中的 RMSNorm 实现高度贴近公式，代码和数学表达可以一一对应。

---

## 十九、常见问题与易错理解

{IMAGE:22}

### 1. RMSNorm 是不是让均值变成 0？

不是。RMSNorm 不减均值，因此输出均值不一定为 0。

例如：

$$
x = [1, 2, 3]
$$

归一化后仍可能整体偏正。

### 2. RMSNorm 是不是让方差变成 1？

也不是严格意义上的方差归一化。它控制的是：

$$
\mathrm{mean}(x^2)
$$

而不是：

$$
\mathrm{mean}((x-\mu)^2)
$$

### 3. RMSNorm 是否改变张量形状？

不会。输入和输出形状一致：

$$
[B, T, C] \rightarrow [B, T, C]
$$

### 4. 为什么不用 bias？

RMSNorm 通常只做尺度归一化，加上 $\gamma$ 就够了。bias 不是必须组件。

### 5. `variance` 变量名是否准确？

很多代码里会写：

```python
variance = x.pow(2).mean(-1, keepdim=True)
```

但它严格来说不是统计学意义上的方差，因为没有减均值。更准确的名字应该是：

```python
mean_square
```

{WARNING}易错点{/WARNING}

RMSNorm 的分母不是标准差，而是均方根。  
不要把：

$$
\sqrt{\mathrm{mean}(x^2)}
$$

和：

$$
\sqrt{\mathrm{mean}((x-\mu)^2)}
$$

混为一谈。

本节小结：RMSNorm 控制的是均方根尺度，不保证零均值，也不等价于标准方差归一化。

---

## 二十、结尾画面与课程收束

{IMAGE:4}

{IMAGE:5}

本节从 LayerNorm 引入，推导了 RMSNorm 的数学形式，并结合 PyTorch 代码解释了它在 MiniMind 模型中的实现方式。

RMSNorm 的核心代码虽然只有几行，但背后对应的是大模型训练中非常重要的数值稳定思想。理解它之后，再看 Transformer Block、Attention 前归一化、MLP 前归一化，就会更清楚模型为什么这样搭建。

本节小结：RMSNorm 是一个短小但关键的模块，是理解现代大模型结构的基础组件之一。

---

## Key Takeaways

1. RMSNorm 的核心公式是：

$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\cdot\gamma
$$

2. RMSNorm 与 LayerNorm 的主要区别是：RMSNorm 不减均值，只根据均方根做尺度归一化。

3. RMSNorm 通常沿 hidden dimension 计算，不改变输入输出形状。

4. $\gamma$ 是可学习缩放参数，用于恢复和增强模型表达能力。

5. $\epsilon$ 用于数值稳定，防止除零。

6. 在大模型中，RMSNorm 常用于 Pre-Norm Transformer Block：

$$
x = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
x = x + \mathrm{MLP}(\mathrm{RMSNorm}(x))
$$

7. RMSNorm 的 PyTorch 核心实现可以概括为：

```python
x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
out = weight * x_norm
```

---

## 思考题

1. RMSNorm 不减均值，这会带来哪些潜在好处？又可能带来哪些风险？

2. 如果把 RMSNorm 中的 `keepdim=True` 去掉，代码在广播时可能出现什么问题？

3. 在 Transformer Block 中，为什么很多大模型选择在 Attention 和 MLP 之前使用 RMSNorm，而不是之后？