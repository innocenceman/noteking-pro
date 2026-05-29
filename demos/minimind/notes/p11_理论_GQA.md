# 第11集: 理论：GQA

## 课程定位：为什么要讲 GQA

本集 MiniMind 第 11/26 讲的是 Transformer 注意力机制中的一个重要工程优化：**分组查询注意力 GQA，Grouped Query Attention**。它位于标准多头注意力 MHA 和多查询注意力 MQA 之间，是现代大模型中非常常见的折中方案。

{IMAGE:5}

在大模型推理阶段，尤其是自回归生成时，模型每生成一个 token 都需要读取之前所有 token 的 Key 和 Value 缓存，也就是常说的 **KV Cache**。随着上下文长度、层数、注意力头数增大，KV Cache 会占据大量显存，并直接影响推理速度。

{IMPORTANT}核心概念：GQA 的目标不是改变注意力公式本身，而是减少 Key/Value 头的数量，让多个 Query 头共享一组 Key/Value，从而降低 KV Cache 显存和推理带宽压力。{/IMPORTANT}

本节的核心对比对象是：

- **MHA**：Multi-Head Attention，多头注意力
- **MQA**：Multi-Query Attention，多查询注意力
- **GQA**：Grouped Query Attention，分组查询注意力

本节小结：GQA 是 MHA 和 MQA 之间的工程折中，主要解决大模型推理时 KV Cache 显存和带宽开销过大的问题。

---

## 标准多头注意力 MHA

### MHA 的基本结构

{IMAGE:1}

标准 Transformer 使用的是 **多头注意力 MHA**。假设隐藏维度为 $d_{model}$，注意力头数为 $h$，每个头的维度为 $d_h$，则通常有：

$$
d_{model} = h \times d_h
$$

输入 hidden states 经过三个线性层，分别得到 Query、Key、Value：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

然后将 $Q,K,V$ 都切分成 $h$ 个头，每个 Query 头都有自己对应的 Key 头和 Value 头。

{IMAGE:2}

单个注意力头的计算公式是：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

其中：

- $Q$ 用来表示当前位置想要查询什么信息
- $K$ 用来表示历史 token 提供什么索引
- $V$ 用来表示历史 token 实际携带什么内容
- $\sqrt{d_h}$ 用于缩放点积，避免数值过大导致 softmax 梯度不稳定

### MHA 的张量形状

假设 batch size 为 $B$，序列长度为 $T$，注意力头数为 $n\_heads$，每头维度为 $head\_dim$，则 MHA 中常见形状是：

$$
Q,K,V \in \mathbb{R}^{B \times T \times n\_heads \times head\_dim}
$$

在计算注意力时，通常会转置为：

$$
Q,K,V \in \mathbb{R}^{B \times n\_heads \times T \times head\_dim}
$$

{IMAGE:3}

MHA 的特点是每个 Query 头都有独立的 Key 和 Value 头：

$$
n\_{q\_heads} = n\_{k\_heads} = n\_{v\_heads}
$$

例如，如果有 8 个 Query 头，那么也有 8 个 Key 头和 8 个 Value 头。

{KNOWLEDGE}背景知识：多头注意力的价值在于，不同注意力头可以学习不同的关系模式。例如有的头关注局部邻近 token，有的头关注语法结构，有的头关注长距离依赖。{/KNOWLEDGE}

本节小结：MHA 表达能力强，但每个注意力头都保存独立的 K/V，在推理时 KV Cache 开销较大。

---

## KV Cache：为什么注意力会吃显存

### 自回归生成中的重复计算问题

在大语言模型生成文本时，模型不是一次性生成完整句子，而是一个 token 一个 token 地生成。

假设已经生成了：

```text
我 今天 想
```

下一步要预测新 token 时，当前位置的 Query 只来自最新 token，但它需要和之前所有 token 的 Key/Value 做注意力计算。

如果每一步都重新计算历史 token 的 K/V，会造成大量重复计算。因此推理时通常会缓存历史 token 的 K/V，这就是 **KV Cache**。

{IMAGE:6}

### KV Cache 的存储量

对于每一层 Transformer，KV Cache 大致需要保存：

$$
K,V \in \mathbb{R}^{B \times n\_{kv\_heads} \times T \times head\_dim}
$$

因为 K 和 V 都要保存，所以总元素数近似为：

$$
2 \times B \times n\_{kv\_heads} \times T \times head\_dim
$$

如果模型有 $L$ 层，则总 KV Cache 近似为：

$$
2 \times L \times B \times n\_{kv\_heads} \times T \times head\_dim
$$

这里最关键的变量是：

$$
n\_{kv\_heads}
$$

也就是 Key/Value 的头数。

{IMPORTANT}核心概念：GQA 优化的重点就是减少 $n\_{kv\_heads}$，而不是减少 Query 头数。Query 头数通常仍然保持较多，以维持模型表达能力。{/IMPORTANT}

本节小结：KV Cache 的显存与层数、上下文长度、head_dim 和 KV 头数成正比，其中 GQA 主要压缩的是 KV 头数。

---

## MQA：把所有 Query 头共享同一组 K/V

### MQA 的结构

{IMAGE:7}

MQA，即 Multi-Query Attention，做法非常激进：

$$
n\_{q\_heads} > 1,\quad n\_{kv\_heads} = 1
$$

也就是说，模型仍然有多个 Query 头，但所有 Query 头共享同一个 Key 头和 Value 头。

例如：

```text
MHA:
Q: 8 heads
K: 8 heads
V: 8 heads

MQA:
Q: 8 heads
K: 1 head
V: 1 head
```

这会显著降低 KV Cache：

$$
\frac{KVCache\_{MQA}}{KVCache\_{MHA}} = \frac{1}{n\_{heads}}
$$

如果原来 MHA 有 8 个 KV 头，MQA 只有 1 个 KV 头，那么 KV Cache 理论上可以减少到原来的 $1/8$。

{IMAGE:8}

### MQA 的优缺点

MQA 的优点非常明确：

- KV Cache 极小
- 推理速度更快
- 显存带宽压力更低
- 长上下文生成更友好

但缺点也明显：

- 所有 Query 头共享同一套 K/V
- K/V 表达能力被压缩得太强
- 可能影响模型质量
- 对大模型训练和收敛质量有一定挑战

{WARNING}易错点：MQA 不是只有一个注意力头。MQA 仍然可以有很多 Query 头，只是 Key 和 Value 只有一组。不要把 MQA 理解成单头注意力。{/WARNING}

本节小结：MQA 极大压缩 KV Cache，但共享程度太高，可能损失注意力表达能力。

---

## GQA：MHA 与 MQA 的折中方案

### GQA 的核心思想

{IMAGE:9}

GQA，即 Grouped Query Attention，介于 MHA 和 MQA 之间。它把多个 Query 头分成若干组，每一组共享一个 Key 头和 Value 头。

假设：

$$
n\_{q\_heads} = 8
$$

如果：

$$
n\_{kv\_heads} = 2
$$

那么每 4 个 Query 头共享一组 K/V：

$$
group\_size = \frac{n\_{q\_heads}}{n\_{kv\_heads}} = \frac{8}{2} = 4
$$

结构可以理解为：

```text
Q heads:  Q0 Q1 Q2 Q3 | Q4 Q5 Q6 Q7
KV heads: K0 V0       | K1 V1
```

每组 Query 头使用同一个 Key/Value 头。

{IMAGE:10}

因此三者关系可以概括为：

$$
MHA: n\_{kv\_heads} = n\_{q\_heads}
$$

$$
GQA: 1 < n\_{kv\_heads} < n\_{q\_heads}
$$

$$
MQA: n\_{kv\_heads} = 1
$$

{IMPORTANT}核心概念：GQA 不是一种新的注意力公式，而是对 K/V 头数和 Q 头数关系的重新设计。它让 Query 保持多头表达，同时让 Key/Value 以组为单位共享。{/IMPORTANT}

本节小结：GQA 通过分组共享 K/V，在 MHA 的质量和 MQA 的效率之间取得平衡。

---

## MHA、MQA、GQA 的直观对比

### 头数关系对比

{IMAGE:11}

可以用一个统一变量来描述三者：

- $n\_{heads}$：Query 头数
- $n\_{kv\_heads}$：Key/Value 头数
- $n\_{rep}$：每个 KV 头要复制给多少个 Query 头

其中：

$$
n\_{rep} = \frac{n\_{heads}}{n\_{kv\_heads}}
$$

三种注意力分别对应：

| 类型 | Query 头数 | KV 头数 | 每个 KV 服务的 Q 头数 |
|---|---:|---:|---:|
| MHA | $h$ | $h$ | 1 |
| GQA | $h$ | $g$ | $h/g$ |
| MQA | $h$ | 1 | $h$ |

当 $g = h$ 时，GQA 退化为 MHA。

当 $g = 1$ 时，GQA 退化为 MQA。

{IMAGE:12}

### 显存开销对比

KV Cache 大小与 $n\_{kv\_heads}$ 成正比：

$$
KVCache \propto n\_{kv\_heads}
$$

因此：

$$
\frac{KVCache\_{GQA}}{KVCache\_{MHA}} =
\frac{n\_{kv\_heads}}{n\_{heads}}
$$

例如：

```text
n_heads = 8
n_kv_heads = 2
```

那么：

$$
\frac{KVCache\_{GQA}}{KVCache\_{MHA}} = \frac{2}{8} = \frac{1}{4}
$$

也就是说，GQA 的 KV Cache 约为 MHA 的 25%。

{IMAGE:13}

本节小结：MHA、MQA、GQA 可以放在同一条连续谱上理解，区别只在于 KV 头数相对于 Query 头数的比例。

---

## MiniMind 中的 GQA 实现思路

### 注意力配置参数

在代码实现中，通常会有如下几个参数：

```python
n_heads = 8       # Query 注意力头数量
n_kv_heads = 2    # Key/Value 注意力头数量
head_dim = dim // n_heads
n_rep = n_heads // n_kv_heads
```

其中：

```python
n_rep = n_heads // n_kv_heads
```

表示每个 KV 头需要被多少个 Query 头共享。

{IMAGE:14}

例如：

```python
n_heads = 8
n_kv_heads = 2
n_rep = 4
```

含义是：

```text
8 个 Q 头
2 个 K/V 头
每个 K/V 头复制 4 次，对齐到 8 个 Q 头
```

### Q/K/V 投影层的区别

在 MHA 中，Q、K、V 的输出维度通常都是：

$$
n\_{heads} \times head\_dim
$$

而在 GQA 中：

$$
Q: n\_{heads} \times head\_dim
$$

$$
K: n\_{kv\_heads} \times head\_dim
$$

$$
V: n\_{kv\_heads} \times head\_dim
$$

对应 PyTorch 代码大致是：

```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        # Q 仍然投影出所有 Query 头
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)

        # K/V 只投影出较少的 KV 头
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)

        # 输出投影回 dim
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
```

{WARNING}易错点：GQA 中 `head_dim` 通常仍然由 `dim // n_heads` 决定，而不是 `dim // n_kv_heads`。K/V 只是头数减少，每个头的维度仍要和 Query head_dim 对齐。{/WARNING}

本节小结：GQA 的代码变化主要体现在 K/V 投影输出维度变小，并额外引入 `n_rep` 来对齐 Query 头数。

---

## repeat_kv：如何让较少的 K/V 对齐多个 Q

### 为什么需要 repeat_kv

{IMAGE:15}

注意力计算时，Query 和 Key 的头维度需要对齐：

$$
Q \in \mathbb{R}^{B \times n\_{heads} \times T \times head\_dim}
$$

但 GQA 中原始 Key/Value 是：

$$
K,V \in \mathbb{R}^{B \times n\_{kv\_heads} \times T \times head\_dim}
$$

如果直接做矩阵乘法：

$$
QK^T
$$

头数维度对不上。因此需要把 K/V 在头维度上扩展为和 Q 一样的头数。

### repeat_kv 示例代码

```python
def repeat_kv(x, n_rep):
    """
    将较少的 KV 头复制到 Query 头数量。

    x shape:
        [batch, seq_len, n_kv_heads, head_dim]

    return shape:
        [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    batch, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    # 增加一个分组内复制维度
    x = x[:, :, :, None, :]

    # 在复制维度上扩展 n_rep 次
    x = x.expand(batch, seq_len, n_kv_heads, n_rep, head_dim)

    # 合并 n_kv_heads 和 n_rep，得到 n_heads
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
```

{IMAGE:16}

这里要注意，`expand` 通常不会真实复制底层数据，而是通过 stride 视图进行广播；后续 `reshape` 会得到适合计算的形状。

在 MiniMind 这类教学实现中，`repeat_kv` 的目的是让逻辑更清晰：

```text
原始 KV:
[B, T, 2, D]

repeat 之后:
[B, T, 8, D]
```

其中每个 KV 头对应 4 个 Query 头。

{KNOWLEDGE}背景知识：虽然代码里看起来把 K/V 重复到了和 Q 一样的头数，但真正节省显存的地方在于 KV Cache 保存的是重复之前的 K/V，也就是较少的 `n_kv_heads`。{/KNOWLEDGE}

本节小结：`repeat_kv` 负责把少量 K/V 头扩展到 Query 头数，用于完成标准注意力矩阵计算。

---

## GQA 的完整前向传播流程

### 张量变换过程

{IMAGE:17}

一个简化的 GQA forward 流程如下：

```python
def forward(self, x):
    batch, seq_len, dim = x.shape

    # 1. 线性投影
    q = self.wq(x)
    k = self.wk(x)
    v = self.wv(x)

    # 2. reshape 成多头格式
    q = q.view(batch, seq_len, self.n_heads, self.head_dim)
    k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
    v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

    # 3. 对 K/V 做分组复制，使其头数与 Q 对齐
    k = repeat_kv(k, self.n_rep)
    v = repeat_kv(v, self.n_rep)

    # 4. 转置为注意力计算常用格式
    q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
    k = k.transpose(1, 2)  # [B, n_heads, T, head_dim]
    v = v.transpose(1, 2)  # [B, n_heads, T, head_dim]

    # 5. scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    # 6. 合并多头
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch, seq_len, self.n_heads * self.head_dim)

    # 7. 输出投影
    return self.wo(out)
```

### 注意力矩阵形状

其中注意力分数矩阵为：

$$
scores \in \mathbb{R}^{B \times n\_{heads} \times T \times T}
$$

如果是带因果 mask 的自回归语言模型，还需要屏蔽未来 token：

$$
scores = scores + mask
$$

mask 通常是上三角矩阵，未来位置填入 $-\infty$，使 softmax 后概率接近 0。

{IMAGE:18}

{WARNING}易错点：GQA 减少的是 K/V 投影和 KV Cache，不是 attention score 的头数。最终参与注意力计算的 Query 头数仍然是 `n_heads`。{/WARNING}

本节小结：GQA 的 forward 与 MHA 非常相似，关键差异是 K/V 的头数更少，并在计算前通过 `repeat_kv` 对齐。

---

## 从效率角度理解 GQA

### 训练与推理的收益不同

{IMAGE:19}

GQA 对推理阶段的收益尤其明显，因为推理时 KV Cache 会随着生成长度不断增长。

KV Cache 存储量：

$$
2 \times L \times B \times T \times n\_{kv\_heads} \times head\_dim
$$

如果把 $n\_{kv\_heads}$ 从 32 降到 8，那么 KV Cache 约减少到原来的：

$$
\frac{8}{32} = \frac{1}{4}
$$

这意味着：

- 更低显存占用
- 更长上下文容量
- 更高 batch 并发
- 更低显存带宽读取压力
- 更快的 token 解码速度

### 为什么不用全部 MQA

如果 MQA 更省显存，为什么不全部使用 MQA？

原因是 MQA 的共享程度过高：

$$
n\_{kv\_heads} = 1
$$

所有 Query 头只能依赖同一组 K/V 表示，可能限制模型对不同语义关系的建模能力。

GQA 则保留了多组 K/V：

$$
1 < n\_{kv\_heads} < n\_{heads}
$$

因此它保留了一部分多头表达能力，同时仍显著降低 KV Cache。

{IMAGE:20}

本节小结：GQA 的价值主要体现在推理效率上，它比 MHA 更省资源，又通常比 MQA 更稳妥。

---

## 设计 GQA 参数时要注意什么

### n_heads 与 n_kv_heads 的整除关系

在实现中通常要求：

$$
n\_{heads} \mod n\_{kv\_heads} = 0
$$

这样才能得到整数：

$$
n\_{rep} = \frac{n\_{heads}}{n\_{kv\_heads}}
$$

如果不能整除，分组复制就会变得不规则，代码和计算都会更麻烦。

```python
assert n_heads % n_kv_heads == 0
n_rep = n_heads // n_kv_heads
```

### 常见配置理解

假设模型有 16 个 Query 头：

```text
MHA: n_heads=16, n_kv_heads=16, n_rep=1
GQA: n_heads=16, n_kv_heads=4,  n_rep=4
MQA: n_heads=16, n_kv_heads=1,  n_rep=16
```

其中 GQA 的 `n_kv_heads=4` 表示每 4 个 Query 头共享一组 K/V。

{IMAGE:21}

### 参数量也会减少吗

GQA 不仅减少 KV Cache，也会减少 K/V 投影层参数量。

MHA 中：

$$
W_K, W_V: d_{model} \times d_{model}
$$

GQA 中：

$$
W_K, W_V: d_{model} \times (n\_{kv\_heads} \times head\_dim)
$$

因为：

$$
n\_{kv\_heads} \times head\_dim < n\_{heads} \times head\_dim = d_{model}
$$

所以 K/V 投影的参数量也减少了。

但注意，Q 投影和输出投影通常不变，因此总参数量下降幅度不如 KV Cache 那么显著。

{WARNING}易错点：GQA 的主要收益不是训练参数量减少，而是推理时 KV Cache 和显存带宽下降。参数量减少只是附带收益。{/WARNING}

本节小结：GQA 参数设计的关键是 `n_heads` 与 `n_kv_heads` 的比例，它决定了显存压缩率和表达能力折中。

---

## 三种注意力机制的总结对照

{IMAGE:4}

### 从结构上看

```text
MHA:
每个 Q 头都有独立 K/V
表达能力强，KV Cache 大

MQA:
所有 Q 头共享一个 K/V
KV Cache 最小，但表达能力压缩明显

GQA:
一组 Q 头共享一个 K/V
在质量和效率之间折中
```

### 从公式上看

统一公式仍然是：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

区别是：

$$
Q \in \mathbb{R}^{B \times T \times n\_{heads} \times d_h}
$$

$$
K,V \in \mathbb{R}^{B \times T \times n\_{kv\_heads} \times d_h}
$$

然后通过复制或广播将 K/V 对齐到 Query 头数。

### 从工程上看

| 机制 | KV Cache | 表达能力 | 推理效率 | 常见用途 |
|---|---:|---:|---:|---|
| MHA | 最大 | 强 | 较低 | 早期 Transformer、较小模型 |
| MQA | 最小 | 压缩较强 | 最高 | 极致推理优化 |
| GQA | 中等偏小 | 较强 | 高 | 现代大语言模型常用 |

本节小结：GQA 是现代大模型中非常实用的注意力结构，它保留多个 Query 头，同时减少 KV 头数，是效率和质量之间的实用折中。

---

## 关键收获

1. MHA、MQA、GQA 的注意力公式相同，区别主要在 Query 头数和 KV 头数的关系。
2. MHA 中 $n\_{kv\_heads}=n\_{heads}$，每个 Query 头有独立 K/V。
3. MQA 中 $n\_{kv\_heads}=1$，所有 Query 头共享一组 K/V。
4. GQA 中 $1<n\_{kv\_heads}<n\_{heads}$，多个 Query 头分组共享 K/V。
5. GQA 的核心收益是降低 KV Cache 显存占用和推理带宽压力。
6. `repeat_kv` 的作用是把较少的 K/V 头扩展到 Query 头数，以便进行标准注意力计算。
7. 设计 GQA 时通常要求 `n_heads % n_kv_heads == 0`。
8. GQA 是 MHA 和 MQA 的中间形态：比 MHA 更省资源，比 MQA 更保留表达能力。

## 思考题

1. 如果 `n_heads=32, n_kv_heads=8`，那么 `n_rep` 是多少？KV Cache 相比 MHA 理论上减少到原来的多少？
2. 为什么 GQA 中 `head_dim` 通常使用 `dim // n_heads`，而不是 `dim // n_kv_heads`？
3. 如果一个模型追求极限推理速度，选择 MQA 还是 GQA？如果追求更稳妥的模型质量，又该如何选择？