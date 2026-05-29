# 第16集: 拼接：Block

## 课程定位：从组件到 TransformerBlock

本集主题是“拼接：Block”，也就是把前面已经手写过的几个核心模块组装成一个完整的 `TransformerBlock`。在大模型结构中，单个 Block 是最重要的重复单元：模型不是靠一个复杂的大模块完成全部能力，而是靠许多个结构相同的 Block 堆叠起来，逐层抽象上下文信息。

{IMAGE:1}

{IMPORTANT}核心概念{/IMPORTANT}

`TransformerBlock` 通常由三类结构组成：

1. 归一化层：`RMSNorm` 或 `LayerNorm`
2. 自注意力层：`Attention`
3. 前馈网络层：`FeedForward` 或 `MLP`

它们通过残差连接拼接在一起，形成如下结构：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

$$
x = x + \text{FeedForward}(\text{Norm}(x))
$$

这就是常见的 **Pre-Norm Transformer Block** 写法。

本节小结：本集不是重新发明新模块，而是把注意力、归一化、前馈网络和残差连接组装成可重复堆叠的 Transformer 基本单元。

## Block 的整体结构

{IMAGE:5}

在 MiniMind 这类从零实现的大模型课程中，`TransformerBlock` 一般对应模型中的一层。假设输入张量为：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$ 表示 batch size
- $T$ 表示序列长度
- $C$ 表示隐藏层维度，也就是 embedding 维度

一个 Block 接收形状为 `[batch, seq_len, hidden_size]` 的输入，并输出同样形状的张量。这个形状不变非常关键，因为多个 Block 才能顺序堆叠：

```python
x = block1(x)
x = block2(x)
x = block3(x)
```

每个 Block 的输入输出维度一致，模型深度就可以通过简单循环扩展。

{IMAGE:6}

典型代码结构如下：

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 注意力模块前的归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 自注意力模块
        self.attention = Attention(args)

        # 前馈网络模块前的归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 前馈网络模块
        self.feed_forward = FeedForward(args)

    def forward(self, x, freqs_cis=None, mask=None):
        # 注意力子层：Norm -> Attention -> Residual
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # 前馈子层：Norm -> FFN -> Residual
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

本节小结：TransformerBlock 的核心任务是保持输入输出形状一致，并在内部完成“归一化、自注意力、残差、前馈网络、残差”的组合。

## 为什么要使用残差连接

{IMAGE:7}

残差连接是 Transformer 能够堆叠很多层的重要原因。没有残差连接时，输入经过多层复杂非线性变换后，梯度容易消失或不稳定。残差连接让每一层学习的是“增量变化”，而不是从零生成一个全新的表示。

公式如下：

$$
y = x + F(x)
$$

其中：

- $x$ 是子层输入
- $F(x)$ 是注意力或前馈网络计算出的变化量
- $y$ 是残差输出

在注意力子层中：

$$
h = x + \text{Attention}(\text{Norm}(x))
$$

在前馈子层中：

$$
out = h + \text{FFN}(\text{Norm}(h))
$$

{KNOWLEDGE}背景知识{/KNOWLEDGE}

残差连接最早在 ResNet 中被广泛使用，它解决了深层神经网络训练困难的问题。Transformer 借鉴了这个思想，使得模型可以稳定堆叠几十层、上百层甚至更多层。

从直觉上看，残差连接相当于给信息保留了一条“直通道路”。如果某个子模块暂时没有学到有用变换，模型至少还能保留原始输入，而不会完全破坏表示。

本节小结：残差连接使 Block 学习输入的修正量，既保留原始信息，又改善深层模型训练稳定性。

## Pre-Norm 与 Post-Norm

{IMAGE:8}

TransformerBlock 中一个非常重要的设计点是归一化层的位置。常见有两种写法。

### Post-Norm 写法

原始 Transformer 更接近下面这种形式：

$$
x = \text{Norm}(x + \text{Attention}(x))
$$

$$
x = \text{Norm}(x + \text{FFN}(x))
$$

也就是先执行子层计算和残差相加，再做归一化。

### Pre-Norm 写法

现代大语言模型更常用 Pre-Norm：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

$$
x = x + \text{FFN}(\text{Norm}(x))
$$

也就是先归一化，再送入子层，最后做残差相加。

{IMAGE:9}

MiniMind 中的 Block 更符合 Pre-Norm 结构。它的优点是训练更稳定，尤其是在模型层数较深时，梯度传播更加顺畅。

{WARNING}易错点{/WARNING}

不要把下面两种写法混淆：

```python
# Pre-Norm
x = x + attention(norm(x))

# Post-Norm
x = norm(x + attention(x))
```

两者的计算顺序不同，训练稳定性和实现风格也不同。现代 LLaMA 系列、MiniMind 这类实现通常更偏向 Pre-Norm。

本节小结：Pre-Norm 是现代大模型常用结构，归一化发生在注意力和前馈网络之前，有利于深层网络稳定训练。

## RMSNorm 在 Block 中的作用

{IMAGE:10}

Block 中通常会出现两个归一化层：

```python
self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
```

第一个用于注意力模块之前，第二个用于前馈网络之前。它们虽然结构相同，但参数是独立的。

RMSNorm 的核心思想是按均方根对向量做缩放：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot w
$$

其中：

- $d$ 是隐藏维度
- $\epsilon$ 用于避免除零
- $w$ 是可学习缩放参数
- $\odot$ 表示逐元素乘法

相比 LayerNorm，RMSNorm 不减去均值，只根据均方根缩放，计算更简单。

{IMAGE:11}

代码示例：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch, seq_len, dim]
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x
```

本节小结：Block 中两个 RMSNorm 分别服务于注意力子层和前馈子层，它们负责稳定输入尺度，让后续模块更容易训练。

## Attention 子层的拼接方式

{IMAGE:12}

自注意力模块负责让每个 token 聚合上下文信息。在 Block 中，它不是直接处理原始输入，而是处理归一化后的输入：

```python
h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
```

这行代码包含三个动作：

1. `self.attention_norm(x)`：先归一化
2. `self.attention(...)`：执行自注意力计算
3. `x + ...`：残差相加

{IMAGE:13}

注意力内部通常会完成以下步骤：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\text{Attention}(Q,K,V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V
$$

对于自回归语言模型，mask 很关键。它保证当前位置只能看到当前位置及以前的 token，不能偷看未来信息。

{WARNING}易错点{/WARNING}

在 Block 层调用 Attention 时，通常需要传入 `mask` 和位置编码相关参数，例如 `freqs_cis`。如果忘记传 mask，训练语言模型时可能会出现信息泄漏，模型在训练时“看见未来”，推理时表现异常。

本节小结：Attention 子层通过归一化输入、计算上下文交互、再残差相加，为每个 token 注入上下文信息。

## FeedForward 子层的拼接方式

{IMAGE:14}

注意力子层之后，Block 会继续经过前馈网络：

```python
out = h + self.feed_forward(self.ffn_norm(h))
```

这里的结构与注意力子层类似：

1. 对 `h` 做归一化
2. 输入前馈网络
3. 与 `h` 做残差相加

前馈网络通常逐 token 独立处理，也就是说它不会在序列维度上混合 token，而是在每个 token 的隐藏维度上做非线性变换。

{IMAGE:15}

常见 FFN 公式可以写成：

$$
\text{FFN}(x) = W_2 \sigma(W_1x)
$$

在 LLaMA 风格模型中，更常见的是 SwiGLU 结构：

$$
\text{FFN}(x) = W_2(\text{SiLU}(W_1x) \odot W_3x)
$$

其中：

- $W_1$ 和 $W_3$ 通常把维度从 `dim` 扩展到 `hidden_dim`
- $\text{SiLU}$ 是激活函数
- $\odot$ 是逐元素乘法
- $W_2$ 再把维度投影回 `dim`

示例代码：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: silu(w1(x)) * w3(x)，再投影回原维度
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

本节小结：FeedForward 子层负责对每个 token 的隐藏表示做非线性变换，增强模型表达能力，并通过残差保持形状不变。

## Block 的完整前向传播过程

{IMAGE:16}

把所有步骤串起来，一个 Block 的前向传播可以分解为：

$$
x_1 = \text{RMSNorm}_1(x)
$$

$$
a = \text{Attention}(x_1)
$$

$$
h = x + a
$$

$$
h_1 = \text{RMSNorm}_2(h)
$$

$$
f = \text{FeedForward}(h_1)
$$

$$
out = h + f
$$

对应代码：

```python
def forward(self, x, freqs_cis=None, mask=None):
    # 1. 注意力前归一化
    norm_x = self.attention_norm(x)

    # 2. 自注意力计算
    attn_out = self.attention(norm_x, freqs_cis, mask)

    # 3. 第一次残差连接
    h = x + attn_out

    # 4. 前馈网络前归一化
    norm_h = self.ffn_norm(h)

    # 5. 前馈网络计算
    ffn_out = self.feed_forward(norm_h)

    # 6. 第二次残差连接
    out = h + ffn_out

    return out
```

{IMAGE:17}

更紧凑的写法如下：

```python
def forward(self, x, freqs_cis, mask):
    h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```

这也是实际工程中常见的写法。虽然代码只有两行核心逻辑，但背后包含了完整的 Transformer 层结构。

本节小结：Block 的前向传播可以概括为两次“Norm -> 子模块 -> Residual”，第一次用于 Attention，第二次用于 FFN。

## 多个 Block 的堆叠

{IMAGE:18}

单个 Block 只是模型的一层。真正的大语言模型会堆叠多个 Block：

```python
self.layers = nn.ModuleList([
    TransformerBlock(args) for _ in range(args.n_layers)
])
```

前向传播时依次通过每一层：

```python
for layer in self.layers:
    x = layer(x, freqs_cis, mask)
```

如果有 $N$ 层 TransformerBlock，可以写成：

$$
x^{(0)} = \text{Embedding}(tokens)
$$

$$
x^{(l+1)} = \text{Block}_l(x^{(l)})
$$

$$
l = 0,1,\dots,N-1
$$

最终得到：

$$
x^{(N)}
$$

再接输出归一化和语言模型头：

$$
logits = x^{(N)}W_{vocab}
$$

{KNOWLEDGE}背景知识{/KNOWLEDGE}

模型参数量和能力很大程度上来自 Block 的重复堆叠。层数越多，模型越能形成高层语义抽象；隐藏维度越大，每个 token 的表示容量越强；注意力头数越多，上下文交互的子空间越丰富。

本节小结：TransformerBlock 是可堆叠单元，整个大模型就是 embedding、多个 Block、最终 norm 和输出头的组合。

## 维度检查与实现细节

{IMAGE:19}

写 Block 时最容易出错的是维度不匹配。为了能做残差连接，子模块输出必须和输入形状一致。

对于输入：

```python
x.shape == [batch_size, seq_len, dim]
```

注意力输出应为：

```python
attn_out.shape == [batch_size, seq_len, dim]
```

前馈网络输出应为：

```python
ffn_out.shape == [batch_size, seq_len, dim]
```

这样下面两行才合法：

```python
h = x + attn_out
out = h + ffn_out
```

{WARNING}易错点{/WARNING}

如果 FFN 中间层扩展到 `hidden_dim`，最后一定要投影回 `dim`：

```python
self.w2 = nn.Linear(hidden_dim, dim, bias=False)
```

否则 `h + ffn_out` 会因为最后一维不同而报错。

另一个常见错误是忘记用 `nn.ModuleList` 保存多层 Block：

```python
# 正确
self.layers = nn.ModuleList([...])

# 错误风险：普通 list 中的模块不会被 PyTorch 正确注册参数
self.layers = [...]
```

本节小结：Block 的输入输出维度必须一致，残差连接要求 Attention 和 FFN 最终都回到原始隐藏维度。

## 从工程视角理解 Block

{IMAGE:20}

在代码组织上，`TransformerBlock` 的价值是把模型结构模块化。每个组件各司其职：

- `RMSNorm`：控制数值尺度
- `Attention`：建模 token 之间的上下文关系
- `FeedForward`：增强每个 token 表示的非线性表达
- `Residual`：保证信息和梯度稳定流动
- `TransformerBlock`：把这些模块组装为可复用层

这种组织方式让模型主体代码非常清晰：

```python
class MiniMindModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens, freqs_cis, mask):
        x = self.tok_embeddings(tokens)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.output(x)

        return logits
```

{IMAGE:21}

这样，模型整体结构就从“很多散乱函数”变成了清晰的层级：

```text
MiniMindModel
├── Token Embedding
├── TransformerBlock × N
│   ├── RMSNorm
│   ├── Attention
│   ├── RMSNorm
│   └── FeedForward
├── Final RMSNorm
└── LM Head
```

本节小结：TransformerBlock 是工程组织上的关键抽象，它让复杂模型可以被拆成清晰、可测试、可复用的模块。

## 片尾回顾

{IMAGE:3}

本集完成的是从单个组件到完整 Block 的拼接。前面课程中已经分别实现过归一化、注意力、前馈网络等模块，而本集把它们组合成 Transformer 的基本层。

一个标准 Pre-Norm Block 可以概括为：

```python
h = x + attention(norm1(x))
out = h + ffn(norm2(h))
```

虽然代码很短，但包含了现代大语言模型的关键结构设计。

{IMAGE:4}

本节小结：掌握 TransformerBlock 后，就可以进一步理解完整大模型的主体结构，因为大模型本质上就是多个 Block 的重复堆叠。

## Key Takeaways

1. `TransformerBlock` 是大模型中反复堆叠的基本单元。
2. MiniMind 中的 Block 通常采用 Pre-Norm 结构：先归一化，再进入子模块，最后残差相加。
3. 一个 Block 包含两个核心子层：Attention 子层和 FeedForward 子层。
4. 残差连接要求输入输出形状一致，因此 Attention 和 FFN 最终都必须输出 `dim` 维。
5. RMSNorm 用于稳定数值尺度，比 LayerNorm 更轻量，常见于 LLaMA 风格模型。
6. 多个 `TransformerBlock` 可以通过 `nn.ModuleList` 注册并循环执行。
7. Block 的核心代码虽然简洁，但它连接了上下文建模、非线性表达和稳定训练三个关键目标。

## 思考题

1. 为什么现代大语言模型更常使用 Pre-Norm，而不是原始 Transformer 的 Post-Norm？
2. 如果 `FeedForward` 的输出维度不是 `dim`，残差连接会出现什么问题？
3. Attention 子层和 FeedForward 子层分别解决了 token 表示中的哪类问题？