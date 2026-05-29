# 第9集: 理论：RoPE&YaRN

## 第9讲：理论：RoPE & YaRN

### 本集主题

本节课围绕大模型中的位置编码展开，重点讲解：

- 为什么 Transformer 需要位置编码
- RoPE，Rotary Position Embedding，旋转位置编码的核心思想
- RoPE 如何把绝对位置编码转化为相对位置信息
- RoPE 在注意力计算中的数学形式
- 长上下文外推时 RoPE 的问题
- YaRN 如何改造 RoPE，使模型支持更长上下文

{IMAGE:1}

{IMPORTANT}核心概念：Transformer 的自注意力机制本身对 token 顺序不敏感，因此必须引入位置编码，让模型知道“谁在前、谁在后、相隔多远”。RoPE 的关键做法不是把位置向量直接加到 embedding 上，而是对 Query 和 Key 做与位置相关的旋转变换。{/IMPORTANT}

本节小结：本讲从位置编码的必要性出发，逐步过渡到 RoPE 的旋转建模方式，并进一步讨论长上下文外推中的 YaRN 方法。

---

## 一、为什么需要位置编码

Transformer 的核心结构是自注意力机制。对于输入序列：

$$
x_1, x_2, \dots, x_n
$$

自注意力主要通过 Query、Key、Value 计算：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

注意力分数为：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

从公式上看，$QK^T$ 只关心 token 表示之间的相似度，并不天然包含顺序信息。

例如：

- “我 喜欢 你”
- “你 喜欢 我”

如果没有位置编码，模型看到的只是三个 token 的集合，很难区分语义差异。

{IMAGE:2}

{KNOWLEDGE}背景知识：自注意力机制具有置换不变性。也就是说，如果不加入位置信息，调换输入 token 的顺序，模型内部的注意力计算结构并不会自动知道顺序发生了变化。{/KNOWLEDGE}

常见位置编码方式包括：

1. 绝对位置编码
2. 相对位置编码
3. 旋转位置编码 RoPE
4. 长上下文扩展方法，如 NTK scaling、YaRN 等

本节小结：Transformer 必须引入位置信息，否则它只能建模 token 之间的内容相关性，无法可靠理解顺序结构。

---

## 二、绝对位置编码与相对位置编码

### 1. 绝对位置编码

早期 Transformer 使用正弦余弦位置编码：

$$
PE_{pos,2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{pos,2i+1} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

其中：

- $pos$ 表示 token 的位置
- $i$ 表示维度索引
- $d$ 表示 hidden size

然后将位置编码加到 token embedding 上：

$$
h_{pos}=x_{pos}+PE_{pos}
$$

这种方法简单直接，但存在一个问题：位置编码是“加进去”的，模型需要自己从表示中解读位置信息。

### 2. 相对位置编码

相对位置编码更关心两个 token 之间的距离：

$$
i-j
$$

在语言建模中，相对距离往往比绝对位置更重要。

比如在句子中，“主语”和“谓语”的距离可能比“它们分别处在第几个 token”更有意义。

{IMAGE:4}

{IMPORTANT}核心概念：RoPE 的优势在于，它使用绝对位置构造旋转矩阵，但在 Query 和 Key 的点积中自然体现相对位置差。也就是说，RoPE 形式上使用绝对位置，效果上具备相对位置编码能力。{/IMPORTANT}

本节小结：绝对位置编码容易实现，但泛化到长序列时不够自然；相对位置编码更符合语言结构。RoPE 的价值在于把二者结合起来。

---

## 三、RoPE 的直观理解：向量旋转

### 1. 二维旋转矩阵

二维平面中，一个向量 $(x, y)$ 旋转角度 $\theta$ 后得到：

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

记为：

$$
R_\theta x
$$

RoPE 的基本思想就是：把 embedding 的相邻两个维度看成一个二维平面，然后按照 token 的位置，对它做旋转。

{IMAGE:5}

对于第 $m$ 个位置，其旋转角度不是固定的，而是：

$$
m\theta_i
$$

其中 $\theta_i$ 是第 $i$ 个二维子空间对应的频率。

### 2. 多维向量中的旋转

假设 hidden dimension 为 $d$，RoPE 会把向量分成 $d/2$ 对：

$$
(x_0,x_1), (x_2,x_3), \dots, (x_{d-2},x_{d-1})
$$

每一对使用不同频率旋转：

$$
\theta_i = 10000^{-2i/d}
$$

第 $m$ 个位置上的旋转为：

$$
R_{\theta,m}
$$

也就是每一组二维向量旋转 $m\theta_i$。

{IMAGE:6}

{KNOWLEDGE}背景知识：低维频率变化慢，适合表示长距离关系；高维频率变化快，适合表示短距离、局部位置信息。这与正弦位置编码的设计思想一致。{/KNOWLEDGE}

本节小结：RoPE 将向量的相邻维度两两配对，在每个二维平面内根据位置进行旋转，从而把位置信息编码进 Query 和 Key。

---

## 四、RoPE 的数学推导

### 1. 对 Query 和 Key 加入旋转

设第 $m$ 个位置的 Query 为 $q_m$，第 $n$ 个位置的 Key 为 $k_n$。

RoPE 不直接改 token embedding，而是对 Query 和 Key 做旋转：

$$
\tilde q_m = R_m q_m
$$

$$
\tilde k_n = R_n k_n
$$

然后注意力分数变成：

$$
\tilde q_m^T \tilde k_n
=
(R_m q_m)^T(R_n k_n)
$$

展开：

$$
= q_m^T R_m^T R_n k_n
$$

由于旋转矩阵满足：

$$
R_m^T = R_{-m}
$$

所以：

$$
R_m^T R_n = R_{n-m}
$$

因此：

$$
\tilde q_m^T \tilde k_n = q_m^T R_{n-m} k_n
$$

{IMAGE:7}

{IMPORTANT}核心概念：RoPE 的注意力分数只依赖于位置差 $n-m$，因此天然具备相对位置编码的性质。{/IMPORTANT}

### 2. 为什么只旋转 Q 和 K，不旋转 V

注意力权重由 $QK^T$ 决定：

$$
\alpha_{mn}=\text{softmax}\left(\frac{q_m^T k_n}{\sqrt d}\right)
$$

位置关系主要影响“关注谁”，也就是注意力分数。

Value 主要承载被聚合的信息内容，因此通常不需要加入旋转位置编码。

{IMAGE:8}

{WARNING}易错点：RoPE 不是给输入 embedding 加一个位置向量，而是在注意力层中对 Query 和 Key 做旋转。它影响的是注意力分数，而不是直接改写 token 的原始语义表示。{/WARNING}

本节小结：RoPE 的关键数学性质是旋转矩阵相乘后会变成相对角度差，因此点积中自然包含相对位置信息。

---

## 五、RoPE 的 PyTorch 实现思路

### 1. 构造频率

常见实现中会先构造 inverse frequency：

$$
\text{inv\_freq}_i = \frac{1}{\theta^{2i/d}}
$$

通常 $\theta=10000$。

```python
import torch

def build_inv_freq(dim, base=10000.0):
    """
    dim: head_dim
    base: RoPE 频率基数，常用 10000
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2).float() / dim)
    )
    return inv_freq
```

如果 `head_dim = 8`，则会产生 4 个频率，对应 4 组二维旋转平面。

### 2. 根据位置生成 cos 和 sin

```python
def build_rope_cache(seq_len, dim, base=10000.0, device="cpu"):
    """
    预先生成 RoPE 所需的 cos/sin cache
    """
    inv_freq = build_inv_freq(dim, base).to(device)

    # 位置索引：[0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device).float()

    # 外积得到每个位置、每个频率的角度
    freqs = torch.einsum("i,j->ij", positions, inv_freq)

    # 每个二维对共享一个角度，因此复制到偶数/奇数维
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()
    sin = emb.sin()
    return cos, sin
```

### 3. rotate_half 操作

对于向量：

$$
[x_0,x_1,x_2,x_3]
$$

旋转时需要构造：

$$
[-x_1,x_0,-x_3,x_2]
$$

```python
def rotate_half(x):
    """
    将最后一维按两半或偶奇配对旋转。
    这里给出常见简化写法，具体实现需与模型维度布局一致。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

### 4. 应用 RoPE

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

{IMAGE:9}

{WARNING}易错点：不同项目中 RoPE 的维度排列方式可能不同。有的实现按前后两半切分，有的实现按偶数维和奇数维交错配对。只要数学上对应同一组二维旋转即可，但代码实现必须保持一致。{/WARNING}

本节小结：RoPE 实现通常包括三步：构造频率、缓存 cos/sin、对 Q/K 应用旋转。工程中要注意维度布局与 broadcasting。

---

## 六、RoPE 与注意力机制的结合

在标准多头注意力中：

```python
def attention(q, k, v, mask=None):
    """
    q, k, v: [batch, heads, seq_len, head_dim]
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (q.shape[-1] ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    return output
```

加入 RoPE 后，流程变为：

```python
def attention_with_rope(q, k, v, cos, sin, mask=None):
    """
    在计算注意力分数前，对 q/k 注入旋转位置编码。
    """
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (q.shape[-1] ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    return output
```

{IMAGE:10}

RoPE 的位置编码不是一个独立模块，而是嵌入到 attention 计算路径中。模型的每一层 attention 都可以使用 RoPE，使不同层都能感知相对位置关系。

本节小结：RoPE 在工程上通常位于 attention 层内部，作用于 Q/K，随后再进入正常的 scaled dot-product attention。

---

## 七、RoPE 的长上下文外推问题

### 1. 训练长度与推理长度不一致

假设模型训练时最大长度为：

$$
L_{\text{train}}=2048
$$

推理时希望扩展到：

$$
L_{\text{infer}}=8192
$$

这就是上下文外推问题。

RoPE 虽然有一定外推能力，但直接把位置从 2048 推到 8192，可能会出现明显性能下降。

{IMAGE:11}

原因包括：

1. 训练中没有见过过大的位置索引
2. 高频维度旋转过快，远距离位置角度变化剧烈
3. 注意力分布在长上下文中可能失衡
4. 位置插值会影响局部位置分辨率

### 2. RoPE 中的位置频率

RoPE 的角度为：

$$
m \cdot \theta_i
$$

当位置 $m$ 变大时，角度会不断增大。由于三角函数周期性：

$$
\sin(x),\cos(x)
$$

会产生周期重叠问题。

高频维度尤其容易在长位置处发生快速震荡，使模型难以稳定理解长距离关系。

{IMAGE:12}

{WARNING}易错点：RoPE 可以外推，不等于可以无限外推。超过训练长度太多时，模型可能会出现困惑度上升、长距离检索失败、生成质量下降等问题。{/WARNING}

本节小结：RoPE 的外推能力来自频率结构，但过长上下文会让位置角度进入训练外分布，导致注意力行为不稳定。

---

## 八、位置插值：Position Interpolation

一种朴素思路是：把长序列位置压缩回训练长度范围。

如果训练长度是 2048，推理长度是 8192，可以令：

$$
m' = \frac{m}{s}
$$

其中：

$$
s = \frac{L_{\text{infer}}}{L_{\text{train}}}
$$

也就是：

$$
s = \frac{8192}{2048}=4
$$

那么位置 8192 会被映射到 2048 附近。

{IMAGE:13}

对应到 RoPE 角度：

$$
m\theta_i \rightarrow \frac{m}{s}\theta_i
$$

这相当于降低了所有频率的旋转速度。

### 优点

- 简单
- 能减少训练外位置带来的角度爆炸
- 对长上下文有帮助

### 缺点

- 所有维度统一缩放
- 局部位置分辨率下降
- 短距离关系可能受损

{KNOWLEDGE}背景知识：长上下文任务既需要远距离能力，也需要短距离精度。简单位置插值会把所有位置都压缩，导致相邻 token 的角度差变小，局部顺序感可能变弱。{/KNOWLEDGE}

本节小结：位置插值通过压缩位置索引来扩展上下文，但它对所有频率一视同仁，容易牺牲局部精度。

---

## 九、YaRN 的核心思想

YaRN，全称 Yet another RoPE extensioN，是一种改进 RoPE 长上下文外推的方法。

它的目标是：

1. 保留短上下文内的局部分辨率
2. 改善长上下文外推能力
3. 避免简单位置插值对所有频率统一缩放
4. 让不同频率维度采用不同的缩放策略

{IMAGE:14}

{IMPORTANT}核心概念：YaRN 的本质是对 RoPE 的频率缩放做更精细的控制。它不是简单地把所有位置除以同一个比例，而是区分高频和低频，让不同频率承担不同的上下文建模职责。{/IMPORTANT}

### 1. 高频与低频的不同作用

在 RoPE 中：

- 高频维度：变化快，更适合局部位置关系
- 低频维度：变化慢，更适合长距离关系

如果统一缩放所有频率：

$$
\theta_i \rightarrow \frac{\theta_i}{s}
$$

那么高频维度也会变慢，局部分辨率受损。

YaRN 希望：

- 高频部分尽量保留原始 RoPE
- 低频部分进行更强的缩放，支持更远距离
- 中间频率平滑过渡

{IMAGE:15}

本节小结：YaRN 的核心不是“让位置变小”这么简单，而是按频率分区处理，使短程和长程能力兼顾。

---

## 十、YaRN 的频率缩放与插值机制

### 1. 缩放因子

设目标扩展倍数为：

$$
s = \frac{L_{\text{target}}}{L_{\text{original}}}
$$

例如从 2048 扩展到 8192：

$$
s=4
$$

YaRN 会设计一个随维度变化的缩放函数，让不同频率使用不同程度的缩放。

可以抽象写成：

$$
\theta'_i = \frac{\theta_i}{g(i)}
$$

其中：

- $i$ 是频率维度索引
- $g(i)$ 是与维度相关的缩放因子
- 对某些维度，$g(i) \approx 1$
- 对另一些维度，$g(i) \approx s$

{IMAGE:16}

### 2. 平滑过渡

YaRN 通常不会让频率突然从“不缩放”跳到“完全缩放”，而是使用 ramp 函数做平滑过渡。

可抽象表示为：

$$
\lambda_i = \text{ramp}(i)
$$

$$
\theta'_i
=
(1-\lambda_i)\theta_i
+
\lambda_i \frac{\theta_i}{s}
$$

其中：

- $\lambda_i=0$ 表示保持原始频率
- $\lambda_i=1$ 表示完全按比例缩放
- $0<\lambda_i<1$ 表示插值过渡

```python
def linear_ramp_mask(dim, low, high):
    """
    构造一个从 0 到 1 的线性过渡 mask。
    low/high 控制哪些维度开始、结束缩放。
    """
    idx = torch.arange(dim, dtype=torch.float32)
    mask = (idx - low) / (high - low)
    mask = torch.clamp(mask, 0.0, 1.0)
    return mask
```

### 3. 简化版 YaRN 频率构造

```python
def build_yarn_inv_freq(dim, base=10000.0, scale=4.0, low=8, high=32):
    """
    简化版 YaRN 思路：
    对不同维度的 inv_freq 进行平滑缩放。
    这不是完整官方实现，只用于理解原理。
    """
    inv_freq = build_inv_freq(dim, base)

    # 每个频率对应一个维度索引
    freq_dim = inv_freq.shape[0]

    # 生成 0 到 1 的平滑 mask
    ramp = linear_ramp_mask(freq_dim, low, high)

    # 缩放后的频率
    scaled_inv_freq = inv_freq / scale

    # 高频保留原频率，低频使用缩放频率，中间平滑过渡
    yarn_inv_freq = (1 - ramp) * inv_freq + ramp * scaled_inv_freq

    return yarn_inv_freq
```

{WARNING}易错点：YaRN 不是简单把 RoPE 的 base 改大，也不是简单把 position 除以 scale。它通常包含频率分段、平滑插值以及注意力缩放等细节。{/WARNING}

本节小结：YaRN 通过对不同频率维度设置不同缩放强度，在保留短距离分辨率的同时增强长距离外推。

---

## 十一、YaRN 中的 attention scaling

除了频率缩放，YaRN 还可能引入 attention scaling，用于补偿长上下文下注意力分布的变化。

在标准注意力中：

$$
\text{score}=\frac{qk^T}{\sqrt d}
$$

当上下文长度变长时，softmax 的竞争范围扩大，注意力分布可能发生变化。

因此可以引入一个额外缩放系数：

$$
\text{score}=\frac{qk^T}{\sqrt d}\cdot a
$$

其中 $a$ 与扩展倍率有关。

{IMAGE:17}

直观理解：

- 长上下文中 token 数量更多
- softmax 归一化范围更大
- 注意力峰值和分布形态可能改变
- scaling 有助于稳定外推后的注意力行为

本节小结：YaRN 不只关注 RoPE 角度，还可能调整注意力分数尺度，以适应更长上下文带来的分布变化。

---

## 十二、MiniMind 中理解 RoPE/YaRN 的实现重点

在 MiniMind 这类从零手写大模型项目中，重点不是一开始追求最复杂的优化，而是理解结构：

1. RoPE 作用在 attention 的 Q/K 上
2. cos/sin 通常提前缓存
3. position index 控制旋转角度
4. 长上下文扩展主要改的是频率或位置映射
5. YaRN 是更精细的 RoPE scaling 方案

{IMAGE:18}

一个简化的模块结构可以写成：

```python
class RotaryEmbedding:
    def __init__(self, dim, max_seq_len, base=10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        self.cos_cached, self.sin_cached = build_rope_cache(
            seq_len=max_seq_len,
            dim=dim,
            base=base,
        )

    def apply(self, q, k, position_ids=None):
        """
        q, k: [batch, heads, seq_len, head_dim]
        position_ids: 可选，用于支持 KV cache 或非连续位置
        """
        seq_len = q.shape[-2]

        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]

        return apply_rotary_pos_emb(q, k, cos, sin)
```

在真实训练和推理中，还需要考虑：

- batch 维度
- 多头维度
- KV cache 时的位置偏移
- mixed precision
- device 一致性
- max sequence length 动态扩展

{IMAGE:19}

{IMPORTANT}核心概念：理解 RoPE 的最好切入点是 attention score。只要明白旋转后的 $q_m^T k_n$ 变成与 $n-m$ 有关，就能抓住它的本质。{/IMPORTANT}

本节小结：在 MiniMind 中实现 RoPE 时，应重点关注 Q/K 旋转、cos/sin 缓存和 position_ids 对齐；YaRN 则主要改造频率生成逻辑。

---

## 十三、RoPE 与 YaRN 的对比

| 方法 | 核心做法 | 优点 | 缺点 |
|---|---|---|---|
| 绝对位置编码 | 位置向量加到 embedding | 简单直观 | 长度外推较弱 |
| 相对位置编码 | 显式建模位置差 | 相对关系清晰 | 实现可能更复杂 |
| RoPE | 对 Q/K 做位置旋转 | 相对位置自然进入点积 | 超长外推仍会退化 |
| 位置插值 | 压缩 position index | 简单有效 | 局部分辨率下降 |
| YaRN | 分频率缩放 RoPE | 长短距离兼顾 | 实现细节更多 |

{IMAGE:20}

### RoPE 适合解决什么问题

RoPE 主要解决：

- attention 无位置信息
- 绝对位置编码不够自然
- 需要在点积中体现相对距离
- 希望位置编码与多头注意力深度结合

### YaRN 适合解决什么问题

YaRN 主要解决：

- RoPE 原始上下文长度不够
- 直接外推效果下降
- 简单插值损伤短距离能力
- 希望以较低成本扩展模型上下文窗口

本节小结：RoPE 是现代大模型中非常常用的位置编码方式，YaRN 则是围绕 RoPE 的长上下文扩展技巧。

---

## 十四、常见误区总结

{WARNING}易错点：误以为 RoPE 是 embedding 加法。  
RoPE 实际作用于 attention 中的 Query 和 Key，而不是直接加到输入 embedding 上。{/WARNING}

{WARNING}易错点：误以为 RoPE 只能表示绝对位置。  
RoPE 使用绝对位置角度进行旋转，但点积后体现的是相对位置差。{/WARNING}

{WARNING}易错点：误以为长上下文只需要把 max_seq_len 改大。  
如果不处理 RoPE 外推问题，模型可能虽然能跑更长输入，但效果明显下降。{/WARNING}

{WARNING}易错点：误以为 YaRN 等于简单 position interpolation。  
YaRN 更强调不同频率维度的差异化缩放，以及必要的注意力尺度调整。{/WARNING}

本节小结：RoPE 和 YaRN 都不只是工程技巧，它们背后有明确的数学动机和建模目标。

---

## 十五、关键公式回顾

### RoPE 旋转

$$
\tilde q_m = R_m q_m
$$

$$
\tilde k_n = R_n k_n
$$

### 点积中的相对位置

$$
\tilde q_m^T \tilde k_n
=
q_m^T R_m^T R_n k_n
=
q_m^T R_{n-m}k_n
$$

### RoPE 频率

$$
\theta_i = 10000^{-2i/d}
$$

### 位置插值

$$
m' = \frac{m}{s}
$$

### YaRN 抽象缩放

$$
\theta'_i = \frac{\theta_i}{g(i)}
$$

或用插值形式表达：

$$
\theta'_i
=
(1-\lambda_i)\theta_i
+
\lambda_i \frac{\theta_i}{s}
$$

本节小结：RoPE 的数学核心是旋转矩阵的相对角度性质；YaRN 的数学核心是对不同频率进行分层缩放。

---

## Key Takeaways

1. Transformer 自注意力本身不包含顺序信息，因此需要位置编码。
2. RoPE 通过对 Query 和 Key 做旋转，把位置信息注入注意力分数。
3. RoPE 形式上使用绝对位置，点积后自然得到相对位置关系。
4. RoPE 的不同维度对应不同频率，高频关注局部，低频关注长距离。
5. 长上下文外推时，原始 RoPE 可能出现性能退化。
6. YaRN 通过频率分段缩放和平滑过渡，在保留局部精度的同时扩展上下文。
7. 实现 RoPE 时最重要的是 cos/sin cache、维度配对、Q/K 对齐和 position_ids 处理。

## 思考题

1. 为什么 RoPE 只作用在 Query 和 Key 上，而通常不作用在 Value 上？
2. 如果把所有 RoPE 频率都统一除以同一个 scale，会对短距离建模产生什么影响？
3. YaRN 为什么要区分高频和低频，而不是简单增加最大位置长度？