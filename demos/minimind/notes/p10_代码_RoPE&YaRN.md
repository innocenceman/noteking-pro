# 第10集: 代码：RoPE&YaRN

## 课程定位：从位置编码到 RoPE 与 YaRN

{IMAGE:15}

本集对应 MiniMind 第 10/26 集，主题是“代码：RoPE & YaRN”。这一讲的重点不是单纯介绍公式，而是把大模型中常见的旋转位置编码 RoPE，以及用于长上下文外推的 YaRN 缩放策略，落到 PyTorch 代码实现上。

RoPE 的核心作用是：让 Transformer 的注意力机制在计算 Query 和 Key 的相似度时，天然感知 token 的相对位置信息。传统绝对位置编码通常是把位置向量加到词向量上，而 RoPE 是直接对 $q$ 和 $k$ 做旋转变换。

{IMPORTANT}RoPE 不是给输入 embedding 简单“加位置”，而是在注意力计算前，对 Query 和 Key 的特征维度按二维成对旋转，从而把位置信息编码进点积注意力里。{/IMPORTANT}

### 本节小结

本集的目标是理解 RoPE 的数学直觉、旋转矩阵的高效实现方式，以及 YaRN 如何通过缩放频率帮助模型支持更长上下文。

---

## 一、为什么需要 RoPE

{IMAGE:1}

Transformer 本身的自注意力机制对序列顺序并不敏感。也就是说，如果不提供位置编码，模型只知道有哪些 token，却不知道它们出现的顺序。

设输入序列为：

$$
x_1, x_2, \dots, x_n
$$

普通 self-attention 会计算：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $QK^T$ 只反映 token 之间的内容相似度。如果没有位置编码，两个 token 交换位置后，注意力结构很难区分这种顺序变化。

{KNOWLEDGE}位置编码的常见类型包括：绝对位置编码、相对位置编码、ALiBi、RoPE 等。RoPE 的优势是形式优雅、实现高效，并且可以天然表达相对位置信息。{/KNOWLEDGE}

{IMAGE:2}

RoPE 解决的问题可以概括为：

1. 将位置 $m$ 编码进 Query。
2. 将位置 $n$ 编码进 Key。
3. 让 $q_m$ 和 $k_n$ 的点积结果与相对距离 $m-n$ 有关。

理想效果是：

$$
\langle f(q,m), f(k,n) \rangle = g(q,k,m-n)
$$

这意味着注意力分数不仅依赖 token 内容，也依赖它们之间的相对位置。

### 本节小结

RoPE 的设计目标是让注意力分数显式携带相对位置信息，同时避免额外维护复杂的位置偏置表。

---

## 二、RoPE 的数学直觉：二维旋转

{IMAGE:16}

RoPE 的基本操作是把隐藏维度两两分组，例如：

$$
(x_0,x_1), (x_2,x_3), (x_4,x_5), \dots
$$

每一组二维向量都可以看成平面上的一个点。对二维向量做旋转，可以写成：

$$
\begin{bmatrix}
x'_0 \\
x'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
$$

展开后：

$$
x'_0 = x_0 \cos\theta - x_1 \sin\theta
$$

$$
x'_1 = x_0 \sin\theta + x_1 \cos\theta
$$

在 RoPE 中，角度 $\theta$ 与 token 位置和维度频率有关。对第 $i$ 对维度，位置 $m$ 的旋转角度可以写为：

$$
\theta_{m,i}=m \cdot \omega_i
$$

其中：

$$
\omega_i = \frac{1}{\theta_{\text{base}}^{2i/d}}
$$

通常 $\theta_{\text{base}}=10000$，$d$ 是 head dimension。

{IMPORTANT}RoPE 的旋转角度由“位置编号 × 维度频率”决定。低维频率较高，变化快；高维频率较低，变化慢。{/IMPORTANT}

### 本节小结

RoPE 把每两个隐藏维度视为一个二维平面，并按位置对其做旋转。不同维度使用不同频率，从而让模型在多尺度上感知位置信息。

---

## 三、从旋转矩阵到代码实现

{IMAGE:17}

直接构造完整旋转矩阵并做矩阵乘法是低效的。实际实现中通常不会创建形如 $d \times d$ 的大矩阵，而是预先计算每个位置、每个维度对应的 $\cos$ 和 $\sin$。

假设 head dimension 为 `dim`，序列长度为 `seq_len`。首先构造频率向量：

```python
import torch

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # dim 通常是 attention head 的维度
    # 这里只取偶数维，因为 RoPE 是两两一组旋转
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 位置编号：0, 1, 2, ..., seq_len - 1
    t = torch.arange(seq_len, dtype=torch.float)

    # 外积得到每个位置、每个频率对应的旋转角度
    # shape: [seq_len, dim // 2]
    freqs = torch.outer(t, freqs)

    # 返回 cos 和 sin，后续直接用于旋转
    return torch.cos(freqs), torch.sin(freqs)
```

这里的关键是：

$$
\text{freqs}[m,i] = m \cdot \omega_i
$$

然后分别计算：

$$
\cos(m\omega_i), \quad \sin(m\omega_i)
$$

{IMAGE:18}

真正应用 RoPE 时，需要把向量拆成偶数维和奇数维：

```python
def apply_rope(x, cos, sin):
    # x shape: [batch, seq_len, n_heads, head_dim]
    # cos/sin shape: [seq_len, head_dim // 2]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 调整 cos/sin 形状以便广播
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # 二维旋转
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos

    # 交错还原到原始维度顺序
    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_out = x_out.flatten(-2)

    return x_out
```

旋转后的向量保持原始 shape 不变：

$$
[B, T, H, D] \rightarrow [B, T, H, D]
$$

其中：

- $B$ 是 batch size
- $T$ 是 sequence length
- $H$ 是 attention heads 数量
- $D$ 是 head dimension

{WARNING}RoPE 要求旋转维度通常是偶数，因为它需要把维度两两配对。如果 head_dim 是奇数，就无法完整组成二维旋转对。{/WARNING}

### 本节小结

RoPE 实现的核心不是显式构造旋转矩阵，而是预计算 `cos` 和 `sin`，再对偶数维、奇数维做成对变换。

---

## 四、MiniMind 中 RoPE 的典型实现结构

{IMAGE:19}

在小型大模型实现中，RoPE 通常分为两部分：

1. 初始化或前向时预计算频率。
2. 在 attention 中对 $q$ 和 $k$ 应用旋转。

伪代码结构如下：

```python
class RotaryEmbedding:
    def __init__(self, dim, max_seq_len, base=10000):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        self.cos_cached, self.sin_cached = self._build_cache()

    def _build_cache(self):
        freqs = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        positions = torch.arange(self.max_seq_len).float()
        angles = torch.outer(positions, freqs)

        return torch.cos(angles), torch.sin(angles)

    def apply(self, q, k):
        seq_len = q.shape[1]

        cos = self.cos_cached[:seq_len].to(q.device)
        sin = self.sin_cached[:seq_len].to(q.device)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        return q, k
```

在 Attention 模块中，流程通常是：

```python
def attention_forward(x):
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    q = q.view(batch, seq_len, n_heads, head_dim)
    k = k.view(batch, seq_len, n_kv_heads, head_dim)
    v = v.view(batch, seq_len, n_kv_heads, head_dim)

    q, k = rotary_emb.apply(q, k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    return out
```

{IMAGE:20}

注意，RoPE 只作用于 $q$ 和 $k$，一般不作用于 $v$。原因是注意力权重由 $q$ 和 $k$ 的点积决定，位置信息应该影响“关注谁”，而不是直接旋转被聚合的 value 内容。

{IMPORTANT}RoPE 作用于 Query 和 Key，不作用于 Value。它改变的是注意力分数的计算方式，而不是最终被加权求和的语义向量本身。{/IMPORTANT}

### 本节小结

MiniMind 这类从零实现的大模型中，RoPE 通常作为 Attention 的一部分，在计算注意力分数前对 q/k 做位置旋转。

---

## 五、旋转矩阵的等价写法：rotate_half

{IMAGE:3}

很多代码库不会显式写偶数维和奇数维拆分，而会使用 `rotate_half` 函数。它的思想是把向量拆成两半或者按奇偶维拆分，然后构造旋转后的另一个方向。

常见实现如下：

```python
def rotate_half(x):
    # 按最后一维切成两半
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # 相当于二维旋转中的 [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)
```

此时 RoPE 可以写成：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # q/k shape: [batch, heads, seq_len, head_dim]
    # cos/sin shape 需要能广播到 q/k

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

这个形式来自复数乘法或二维旋转的等价表达：

$$
x' = x\cos\theta + \text{rotate\_half}(x)\sin\theta
$$

{WARNING}`rotate_half` 有两种常见约定：一种是前半维与后半维配对，另一种是偶数维与奇数维配对。实现时必须保证 `cos/sin` 的排布方式与 `rotate_half` 的配对方式一致。{/WARNING}

{IMAGE:21}

如果使用奇偶维配对，可以写成：

```python
def rotate_half_interleaved(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)
```

对应：

```python
x_rope = x * cos + rotate_half_interleaved(x) * sin
```

这里 `cos` 和 `sin` 通常需要被重复到完整维度：

```python
cos_full = torch.repeat_interleave(cos, repeats=2, dim=-1)
sin_full = torch.repeat_interleave(sin, repeats=2, dim=-1)
```

### 本节小结

`rotate_half` 是 RoPE 中常见的代码技巧，本质上是把二维旋转中的垂直方向分量提前构造出来，从而让公式变成逐元素乘加。

---

## 六、RoPE 为什么能表达相对位置

{IMAGE:4}

RoPE 最重要的性质是：两个经过旋转的位置向量做点积时，结果与相对位置有关。

设位置 $m$ 的旋转矩阵为 $R_m$，位置 $n$ 的旋转矩阵为 $R_n$。则：

$$
\langle R_m q, R_n k \rangle
= q^T R_m^T R_n k
$$

旋转矩阵满足：

$$
R_m^T R_n = R_{n-m}
$$

所以：

$$
\langle R_m q, R_n k \rangle
= q^T R_{n-m} k
$$

这说明注意力分数中自然出现了相对距离 $n-m$。

{IMAGE:5}

也就是说，RoPE 并不是简单让模型知道“当前位置是第几个 token”，而是让 $q$ 和 $k$ 的相似度随着相对距离变化。

{KNOWLEDGE}旋转矩阵有一个关键性质：多个旋转可以相加角度，逆旋转等价于负角度旋转。因此 $R_m^T R_n$ 可以化简为与 $n-m$ 有关的旋转。{/KNOWLEDGE}

### 本节小结

RoPE 的相对位置能力来自旋转矩阵的群结构：两个位置旋转后的点积会自动变成相对位移的函数。

---

## 七、缓存 cos/sin 的工程细节

{IMAGE:22}

为了避免每次 forward 都重复计算三角函数，通常会提前缓存 `cos` 和 `sin`。这样可以减少计算开销，尤其是在推理阶段。

```python
class RoPECache:
    def __init__(self, dim, max_seq_len, base=10000):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.cos, self.sin = self.build()

    def build(self):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        pos = torch.arange(self.max_seq_len).float()
        angles = torch.outer(pos, inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin
```

实际训练时还需要考虑 dtype 和 device：

```python
cos = cos.to(device=x.device, dtype=x.dtype)
sin = sin.to(device=x.device, dtype=x.dtype)
```

但有时为了数值稳定，也会先用 `float32` 计算三角函数，再转换到 `bf16` 或 `fp16`。

{WARNING}不要在低精度下直接计算很长序列的三角函数，尤其是长上下文外推时，角度可能较大，低精度会放大误差。常见做法是先用 float32 生成频率和三角值。{/WARNING}

{IMAGE:23}

在推理缓存 KV cache 时，RoPE 的位置也必须和当前 token 的真实位置一致。例如生成第 $t$ 个 token 时，需要使用 position id 为 $t$ 的 cos/sin，而不是每次都从 0 开始。

### 本节小结

RoPE 的缓存实现要注意三点：缓存长度、device/dtype 转换，以及推理时 position id 与 KV cache 的对齐。

---

## 八、YaRN 的背景：为什么要缩放 RoPE

{IMAGE:6}

RoPE 在训练时通常只见过有限长度的上下文，比如 2K、4K 或 8K。如果直接把上下文扩展到 32K、64K，模型会遇到位置外推问题。

原因在于：

$$
\theta_{m,i}=m \cdot \omega_i
$$

当 $m$ 远大于训练长度时，旋转角度分布会进入模型没有充分学习过的范围。注意力模式可能失真，表现为困惑度上升、长文理解能力变差。

YaRN 的目标是让模型在扩展上下文时，对 RoPE 的频率进行合理缩放，让长位置映射到更平滑、更可控的旋转范围。

{IMPORTANT}YaRN 是一种 RoPE 长上下文扩展方法。它不是替换 RoPE，而是在 RoPE 的频率或位置尺度上做调整，使模型更好适应超过训练长度的上下文。{/IMPORTANT}

### 本节小结

RoPE 本身具备一定外推能力，但当上下文显著超过训练长度时，频率分布会失配。YaRN 通过缩放缓解这种失配。

---

## 九、YaRN 的基本思想

{IMAGE:7}

最简单的位置缩放思想是把位置 $m$ 除以一个缩放因子 $s$：

$$
m' = \frac{m}{s}
$$

然后使用：

$$
\theta'_{m,i}=m' \cdot \omega_i
$$

也就是：

$$
\theta'_{m,i}=\frac{m}{s}\omega_i
$$

这相当于让位置增长变慢，从而把更长的上下文压缩到模型熟悉的旋转范围内。

但是，简单线性缩放并不总是最优。高频维度和低频维度对上下文长度的敏感程度不同。YaRN 的思路更细致：对不同频率区域使用不同的缩放策略。

{IMAGE:8}

可以把 RoPE 维度频率分成几类：

1. 高频区域：对局部位置变化敏感。
2. 中频区域：兼顾局部和中距离。
3. 低频区域：负责更长距离的信息。

YaRN 会对这些频段做插值式调整，使模型在短距离上尽量保持原有能力，在长距离上获得更好的外推。

{KNOWLEDGE}长上下文扩展的难点在于：如果缩放太强，短距离位置信息会被压扁；如果缩放太弱，长距离位置又会超出训练分布。YaRN 的价值就在于平衡这两者。{/KNOWLEDGE}

### 本节小结

YaRN 不是简单粗暴地整体拉伸位置，而是按频率区域进行更合理的 RoPE 缩放，以兼顾短上下文质量和长上下文能力。

---

## 十、YaRN 相关公式与实现轮廓

{IMAGE:9}

在工程实现中，YaRN 通常会修改 RoPE 的 `inv_freq`。原始 RoPE 的频率为：

$$
\omega_i=\frac{1}{\theta_{\text{base}}^{2i/d}}
$$

若上下文扩展倍数为：

$$
s = \frac{L_{\text{target}}}{L_{\text{train}}}
$$

其中：

- $L_{\text{train}}$ 是训练上下文长度
- $L_{\text{target}}$ 是目标上下文长度

简单缩放可以写成：

$$
\omega'_i = \frac{\omega_i}{s}
$$

但 YaRN 会引入一个随维度变化的混合系数 $\alpha_i$：

$$
\omega'_i = (1-\alpha_i)\omega_i + \alpha_i \frac{\omega_i}{s}
$$

其中：

$$
0 \leq \alpha_i \leq 1
$$

高频维度可以更接近原始频率，低频维度可以更多使用缩放频率。

伪代码如下：

```python
def yarn_scaled_inv_freq(dim, base, scale, beta_fast=32, beta_slow=1):
    # 原始 RoPE 频率
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2).float() / dim)
    )

    # 简化示意：构造一个从 0 到 1 的 ramp
    # 实际 YaRN 会根据旋转周期和 beta_fast/beta_slow 计算边界
    ramp = torch.linspace(0, 1, inv_freq.shape[0])

    # 缩放后的频率
    scaled_inv_freq = inv_freq / scale

    # 混合原始频率与缩放频率
    inv_freq_yarn = (1 - ramp) * inv_freq + ramp * scaled_inv_freq

    return inv_freq_yarn
```

{WARNING}上面代码是讲解用的简化版本。真实 YaRN 实现通常会根据每个维度对应的旋转周期、目标长度、`beta_fast`、`beta_slow` 等参数计算 ramp 区间。{/WARNING}

{IMAGE:24}

实际构造 cos/sin 的方式仍然类似：

```python
def precompute_yarn_rope(dim, seq_len, base=10000, scale=4.0):
    inv_freq = yarn_scaled_inv_freq(
        dim=dim,
        base=base,
        scale=scale,
    )

    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, inv_freq)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin
```

### 本节小结

YaRN 的工程入口通常是修改 RoPE 的频率向量 `inv_freq`。后续的 `cos/sin` 缓存和 `apply_rope` 逻辑可以基本保持不变。

---

## 十一、Attention 中接入 RoPE 与 YaRN

{IMAGE:10}

在完整 Attention 中，接入 RoPE/YaRN 的位置很固定：完成线性投影并 reshape 成多头之后，在计算 attention scores 之前应用。

```python
class MiniMindAttention(torch.nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_len, rope_theta=10000):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        cos, sin = precompute_freqs_cis(
            dim=self.head_dim,
            seq_len=max_seq_len,
            theta=rope_theta,
        )

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous()
        out = out.view(bsz, seq_len, self.hidden_size)

        return self.o_proj(out)
```

{IMAGE:11}

如果接入 YaRN，只需要在构造缓存时替换频率生成逻辑：

```python
cos, sin = precompute_yarn_rope(
    dim=self.head_dim,
    seq_len=max_seq_len,
    base=rope_theta,
    scale=scale,
)
```

后续 attention forward 不需要大改。

{IMPORTANT}RoPE/YaRN 的优雅之处在于：它们主要影响 q/k 的位置旋转，不改变 Transformer block 的整体结构。{/IMPORTANT}

### 本节小结

在 Attention 中接入 RoPE 很直接：q/k 投影后旋转，旋转后再计算注意力。YaRN 则主要替换 RoPE 的频率缓存生成方式。

---

## 十二、形状变换与广播细节

{IMAGE:12}

RoPE 代码最容易出错的地方往往不是公式，而是 tensor shape。

常见形状有两种约定：

第一种：

$$
[B, T, H, D]
$$

第二种：

$$
[B, H, T, D]
$$

如果 `q` 是 `[B, T, H, D]`，那么 `cos` 的 shape `[T, D/2]` 需要扩展成：

```python
cos = cos[None, :, None, :]
sin = sin[None, :, None, :]
```

如果 `q` 是 `[B, H, T, D]`，则需要扩展成：

```python
cos = cos[None, None, :, :]
sin = sin[None, None, :, :]
```

{IMAGE:13}

在使用完整维度版本时，`cos/sin` 可能是 `[T, D]`：

```python
cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
```

然后才能和 `x` 的最后一维逐元素相乘。

{WARNING}必须明确当前代码采用 `[B,T,H,D]` 还是 `[B,H,T,D]`。很多 RoPE bug 都来自 `cos/sin` 广播维度写错，代码不报错但旋转到了错误位置。{/WARNING}

### 本节小结

RoPE 实现必须严格对齐 tensor shape。位置维、head 维、hidden 维一旦混淆，模型可能可以运行，但学习效果会明显异常。

---

## 十三、长上下文推理中的 position ids

{IMAGE:25}

在训练时，输入通常是一整段序列，position ids 可以简单从 0 到 $T-1$：

$$
0,1,2,\dots,T-1
$$

但在自回归推理时，模型会逐 token 生成，并使用 KV cache。假设 prompt 长度为 100，接下来生成第一个新 token 时，它的位置不是 0，而是 100。

因此推理时需要根据当前缓存长度选取对应位置的 cos/sin：

```python
def get_rope_slice(cos_cached, sin_cached, start_pos, seq_len):
    end_pos = start_pos + seq_len
    cos = cos_cached[start_pos:end_pos]
    sin = sin_cached[start_pos:end_pos]
    return cos, sin
```

在增量推理中：

```python
cos, sin = get_rope_slice(
    cos_cached,
    sin_cached,
    start_pos=past_kv_len,
    seq_len=current_seq_len,
)
```

{WARNING}使用 KV cache 时，如果每一步都把 position 从 0 开始算，会导致新 token 的位置编码错误，注意力与历史 key 的相对位置关系也会错乱。{/WARNING}

### 本节小结

长上下文推理中，RoPE 必须与真实 position id 对齐。KV cache 越长，这个细节越重要。

---

## 十四、数值稳定性与性能考虑

{IMAGE:26}

RoPE/YaRN 的代码虽然短，但涉及几个工程取舍：

1. 是否提前缓存完整 `cos/sin`。
2. 缓存使用 `fp32`、`fp16` 还是 `bf16`。
3. 是否支持动态扩展缓存。
4. 是否在训练和推理共用同一套逻辑。
5. YaRN 缩放参数是否与模型训练配置一致。

对于小模型教学实现，通常可以预先缓存：

```python
self.register_buffer("cos_cached", cos, persistent=False)
self.register_buffer("sin_cached", sin, persistent=False)
```

`persistent=False` 表示这些 buffer 不一定保存进 checkpoint，因为它们可以根据配置重新生成。

{KNOWLEDGE}`register_buffer` 适合保存不是模型参数、但需要跟随模型移动 device 的张量，例如 mask、position cache、RoPE cos/sin cache。{/KNOWLEDGE}

{IMAGE:27}

如果支持超过原缓存长度的输入，可以动态扩容：

```python
def maybe_update_rope_cache(self, seq_len, device):
    if seq_len <= self.max_seq_len_cached:
        return

    self.max_seq_len_cached = seq_len
    cos, sin = precompute_freqs_cis(
        dim=self.head_dim,
        seq_len=seq_len,
        theta=self.rope_theta,
    )

    self.cos_cached = cos.to(device)
    self.sin_cached = sin.to(device)
```

不过动态扩容时要注意训练配置和推理配置是否一致，尤其是使用 YaRN 时，目标长度和 scale 不应随意变化。

### 本节小结

RoPE 的工程实现需要兼顾性能和稳定性。缓存能减少重复计算，但要处理好 dtype、device、长度扩展和 checkpoint 行为。

---

## 十五、代码阅读时的关键检查点

{IMAGE:28}

阅读 MiniMind 或其他 LLM 项目中的 RoPE/YaRN 代码时，可以按以下顺序检查：

1. `head_dim` 是否为偶数。
2. `inv_freq` 是否按 `torch.arange(0, dim, 2)` 构造。
3. `position` 是否正确生成。
4. `torch.outer(position, inv_freq)` 的 shape 是否为 `[seq_len, dim/2]`。
5. `cos/sin` 是否正确广播到 q/k。
6. `rotate_half` 的配对方式是否与 `cos/sin` 排布一致。
7. q/k 是否在 attention score 计算前完成旋转。
8. v 是否没有被 RoPE 旋转。
9. 推理时 position 是否考虑 KV cache 长度。
10. YaRN 的 scale 是否与目标上下文一致。

{IMAGE:29}

典型错误示例：

```python
# 错误示例：直接把 cos shape [T, D/2] 乘到 x shape [B,T,H,D]
# 最后一维 D 和 D/2 对不上，或者错误广播
x = x * cos
```

正确做法之一：

```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]

cos = cos[None, :, None, :]
sin = sin[None, :, None, :]

x_even_new = x_even * cos - x_odd * sin
x_odd_new = x_even * sin + x_odd * cos
```

### 本节小结

RoPE 代码审查的重点是频率生成、维度配对、广播方式、q/k 应用位置，以及推理 position 对齐。

---

## 十六、从 RoPE 到 YaRN 的整体理解

{IMAGE:30}

可以把 RoPE 和 YaRN 的关系理解为：

- RoPE：定义位置如何进入 q/k。
- YaRN：定义长上下文下 RoPE 的频率如何调整。

RoPE 的基础流程：

$$
\text{position} \rightarrow \text{angle} \rightarrow \cos/\sin \rightarrow q/k \text{ rotation}
$$

YaRN 修改的是中间的频率或角度：

$$
\omega_i \rightarrow \omega'_i
$$

然后继续走同样流程：

$$
\text{position} \rightarrow \text{scaled angle} \rightarrow \cos/\sin \rightarrow q/k \text{ rotation}
$$

{IMAGE:31}

这也是为什么很多代码实现里，YaRN 看起来只是 RoPE 初始化的一小段变化，但它对长上下文能力影响很大。

{IMPORTANT}理解 YaRN 时不要只看 `scale` 参数。更重要的是理解不同频率维度对短距离和长距离建模的作用不同，因此需要分段或插值式缩放。{/IMPORTANT}

### 本节小结

RoPE 是位置编码机制，YaRN 是 RoPE 的长上下文缩放策略。两者在代码上高度耦合，但概念层次不同。

---

## 十七、本集关键收获

{IMAGE:14}

1. RoPE 通过旋转 Query 和 Key，把位置信息编码进注意力分数。
2. RoPE 的数学基础是二维旋转矩阵。
3. 每两个 hidden dimension 组成一组旋转平面。
4. 旋转角度由位置和维度频率共同决定。
5. RoPE 的点积结果天然依赖相对位置。
6. 工程实现中通常预计算 `cos` 和 `sin`，避免显式构造大旋转矩阵。
7. `rotate_half` 是实现 RoPE 的常见技巧，但必须注意维度配对约定。
8. RoPE 通常只作用于 q/k，不作用于 v。
9. 推理时必须让 RoPE position 与 KV cache 长度对齐。
10. YaRN 通过缩放 RoPE 频率改善长上下文外推。
11. YaRN 的重点是兼顾短距离建模质量和长距离位置扩展能力。
12. 实现 RoPE/YaRN 时，最常见 bug 来自 shape、broadcast、dtype、position id。

{IMPORTANT}一句话总结：RoPE 用旋转让注意力知道相对位置，YaRN 用频率缩放让 RoPE 更适应长上下文。{/IMPORTANT}

### 思考题

1. 为什么 RoPE 只旋转 Query 和 Key，而通常不旋转 Value？
2. 如果 `rotate_half` 的维度配对方式和 `cos/sin` 的排列方式不一致，会对模型训练造成什么影响？
3. YaRN 为什么不能简单理解为“把所有位置都除以同一个 scale”？