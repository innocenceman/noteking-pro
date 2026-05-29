# 第13集: 代码：GQA 下

## 第 13 讲：代码：GQA 下

### 本讲主题

本集继续实现 MiniMind 中的注意力模块，重点落在 **注意力计算、因果掩码、输出投影** 三个环节。前一部分已经完成了 Q、K、V 的线性映射、维度变换，以及 GQA 中 `repeat_kv` 的准备工作；这一讲则进入真正的 attention forward 逻辑。

{IMAGE:9}

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在 Transformer Decoder 中，每一层的自注意力需要满足两个核心要求：

1. 当前 token 只能看到自己以及之前的 token，不能偷看未来。
2. Query 头数可以多于 Key/Value 头数，这就是 GQA 的核心动机，用更少的 KV head 降低推理缓存和计算成本。

本节总结：本讲从代码层面串起 GQA 注意力的后半部分：点积注意力、因果 mask、softmax、加权求和以及最终输出映射。

---

## 一、GQA 中 Q、K、V 的形状回顾

### 1. 输入与线性映射

假设输入隐藏状态为：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$：batch size
- $T$：序列长度
- $C$：模型隐藏维度，即 `dim`

经过三个线性层后：

```python
xq = self.wq(x)
xk = self.wk(x)
xv = self.wv(x)
```

在 GQA 中，Q 的头数通常大于 K/V 的头数：

```python
self.n_heads      # Query heads
self.n_kv_heads   # Key/Value heads
self.head_dim     # 每个 head 的维度
```

因此形状通常是：

$$
Q: [B, T, n\_heads \times head\_dim]
$$

$$
K,V: [B, T, n\_kv\_heads \times head\_dim]
$$

{IMAGE:10}

### 2. reshape 与 transpose

线性层输出之后，需要拆分 head 维度：

```python
xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
```

随后为了方便矩阵乘法，通常转成：

```python
xq = xq.transpose(1, 2)
xk = xk.transpose(1, 2)
xv = xv.transpose(1, 2)
```

此时：

$$
Q: [B, n\_heads, T, head\_dim]
$$

$$
K,V: [B, n\_kv\_heads, T, head\_dim]
$$

{IMAGE:1}

{IMPORTANT}核心概念{/IMPORTANT}

注意力计算要求 Q 和 K 的 head 数一致。如果使用 GQA，那么 K/V 的 head 数较少，需要通过 `repeat_kv` 将 K/V 在 head 维度上重复到和 Q 一致。

本节总结：GQA 的前提是 Q head 多、KV head 少；在真正计算 attention 前，要保证 Q、K、V 的 batch、head、sequence、head_dim 维度可以对齐。

---

## 二、repeat_kv：让 K/V 适配 Q 的头数

### 1. 为什么需要 repeat

在标准 MHA 中：

$$
n\_heads = n\_kv\_heads
$$

每一个 Query head 都有自己独立的 Key head 和 Value head。

但在 GQA 中：

$$
n\_heads > n\_kv\_heads
$$

例如：

```python
n_heads = 8
n_kv_heads = 2
```

这表示 8 个 Query head 共享 2 组 Key/Value head，每组 KV head 服务多个 Q head。

共享比例为：

$$
n\_rep = \frac{n\_heads}{n\_kv\_heads}
$$

例如：

$$
n\_rep = \frac{8}{2} = 4
$$

即每个 KV head 重复 4 次。

{IMAGE:11}

### 2. repeat_kv 的典型实现

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [B, n_kv_heads, T, head_dim]
    返回: [B, n_kv_heads * n_rep, T, head_dim]
    """
    bs, n_kv_heads, slen, head_dim = x.shape

    if n_rep == 1:
        return x

    x = x[:, :, None, :, :].expand(
        bs,
        n_kv_heads,
        n_rep,
        slen,
        head_dim
    )

    return x.reshape(bs, n_kv_heads * n_rep, slen, head_dim)
```

核心是先插入一个维度：

$$
[B, n\_{kv}, T, D] \rightarrow [B, n\_{kv}, 1, T, D]
$$

然后 expand：

$$
[B, n\_{kv}, n\_{rep}, T, D]
$$

最后 reshape：

$$
[B, n\_{kv} \times n\_{rep}, T, D]
$$

{IMAGE:2}

{WARNING}易错点{/WARNING}

`expand` 不是真正复制数据，而是通过 stride 共享底层存储，效率更高。但后续如果要做需要连续内存的操作，可能需要注意 `.contiguous()` 或 reshape 的行为。

本节总结：`repeat_kv` 的目标是让较少的 KV head 在逻辑上扩展到与 Q head 数量一致，从而完成 GQA 的注意力计算。

---

## 三、注意力分数计算

### 1. 点积注意力公式

注意力的核心公式是：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：

- $Q$：查询向量
- $K$：键向量
- $V$：值向量
- $d_k$：每个 head 的维度，即 `head_dim`
- $\sqrt{d_k}$：缩放因子，防止点积值过大导致 softmax 饱和

{IMAGE:3}

### 2. 代码实现

在代码中，Q 和 K 的形状分别为：

```python
xq.shape  # [B, n_heads, T, head_dim]
xk.shape  # [B, n_heads, T, head_dim]
```

计算注意力分数：

```python
scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
```

其中：

```python
xk.transpose(2, 3)
```

会将 K 从：

$$
[B, H, T, D]
$$

变成：

$$
[B, H, D, T]
$$

于是矩阵乘法为：

$$
[B,H,T,D] \times [B,H,D,T] = [B,H,T,T]
$$

即：

```python
scores.shape  # [B, n_heads, T, T]
```

每个位置对序列中所有位置都有一个注意力分数。

{IMAGE:12}

### 3. 为什么要除以 $\sqrt{head\_dim}$

如果不缩放，随着 $head\_dim$ 增大，点积结果的方差也会增大，softmax 输入变得非常大或非常小，导致输出接近 one-hot，梯度变差。

缩放后：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

可以让数值更稳定。

{IMPORTANT}核心概念{/IMPORTANT}

注意力分数本质上是每个 token 的 Query 与所有 token 的 Key 做相似度计算。分数矩阵的最后两个维度 $T \times T$ 表示“每个位置看向每个位置”。

本节总结：注意力分数通过 $QK^T$ 得到，形状为 `[B, H, T, T]`，表示每个 head 中所有 token 之间的两两注意力关系。

---

## 四、因果掩码：禁止看到未来

### 1. Decoder 为什么需要 mask

MiniMind 是自回归语言模型。训练时虽然整段文本一次性输入模型，但预测第 $t$ 个 token 时，只允许依赖 $0 \sim t$ 的信息，不能看到 $t+1$ 之后的 token。

如果不加因果掩码，模型会在训练时“作弊”，直接利用未来 token，导致推理阶段性能崩溃。

{IMAGE:13}

### 2. 因果 mask 的矩阵形式

假设序列长度为 5，未 mask 前每个位置都能看所有位置：

$$
\begin{bmatrix}
s_{00} & s_{01} & s_{02} & s_{03} & s_{04} \\
s_{10} & s_{11} & s_{12} & s_{13} & s_{14} \\
s_{20} & s_{21} & s_{22} & s_{23} & s_{24} \\
s_{30} & s_{31} & s_{32} & s_{33} & s_{34} \\
s_{40} & s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix}
$$

因果 mask 要屏蔽上三角，也就是当前位置右侧的未来 token：

$$
\begin{bmatrix}
s_{00} & -\infty & -\infty & -\infty & -\infty \\
s_{10} & s_{11} & -\infty & -\infty & -\infty \\
s_{20} & s_{21} & s_{22} & -\infty & -\infty \\
s_{30} & s_{31} & s_{32} & s_{33} & -\infty \\
s_{40} & s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix}
$$

经过 softmax 后，$-\infty$ 对应的位置概率变成 0。

{IMAGE:14}

### 3. 代码中的 mask

常见代码形式：

```python
mask = torch.full(
    (1, 1, seqlen, seqlen),
    float("-inf"),
    device=x.device
)

mask = torch.triu(mask, diagonal=1)
```

解释：

```python
torch.full(...)
```

创建一个全是 `-inf` 的矩阵。

```python
torch.triu(mask, diagonal=1)
```

保留主对角线右上方的部分，即未来 token 的位置。

形状设计为：

```python
[1, 1, T, T]
```

这样可以广播到：

```python
[B, H, T, T]
```

然后加到 scores 上：

```python
scores = scores + mask
```

{IMAGE:15}

### 4. 为什么是加 mask 而不是乘 mask

注意力分数后面要进入 softmax。为了让某些位置概率变成 0，最自然的做法是把它们设成极小值：

$$
softmax(-\infty)=0
$$

如果用乘法把未来位置乘成 0，softmax 后它们仍然可能获得非零概率，因为 0 不代表“不可能”，只是一个普通分数。

{WARNING}易错点{/WARNING}

mask 应该加在 softmax 之前，而不是 softmax 之后。softmax 之后再处理概率，容易破坏归一化性质，也可能引入数值问题。

本节总结：因果 mask 通过给未来位置加 `-inf`，让这些位置在 softmax 后概率为 0，从而保证自回归模型不能看到未来。

---

## 五、softmax 得到注意力权重

### 1. 从 scores 到权重

完成 mask 后：

```python
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

其中 `dim=-1` 表示对最后一个维度做 softmax，也就是对每个 query token 能看的所有 key token 做归一化。

数学形式：

$$
a_{ij} = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
$$

其中：

- $i$：当前 query 位置
- $j$：被关注的 key 位置
- $a_{ij}$：位置 $i$ 对位置 $j$ 的注意力权重

{IMAGE:16}

### 2. 为什么常用 `scores.float()`

在混合精度训练中，Q、K、V 可能是 `float16` 或 `bfloat16`。softmax 对数值稳定性较敏感，因此常见写法是先转成 float32 计算 softmax，再转回原 dtype：

```python
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

这样可以减少溢出、下溢或 NaN 风险。

{KNOWLEDGE}背景知识{/KNOWLEDGE}

softmax 会放大最大值的影响。如果输入分数过大，指数运算容易造成数值不稳定。因此注意力中既要除以 $\sqrt{d_k}$，也常常在 softmax 时使用更高精度。

本节总结：softmax 将注意力分数转成概率分布，最后一维每一行的和为 1，表示当前 token 对历史 token 的关注比例。

---

## 六、与 V 相乘得到上下文向量

### 1. 注意力加权求和

注意力权重形状为：

```python
scores.shape  # [B, H, T, T]
```

Value 形状为：

```python
xv.shape      # [B, H, T, D]
```

两者相乘：

```python
output = torch.matmul(scores, xv)
```

形状变化：

$$
[B,H,T,T] \times [B,H,T,D] = [B,H,T,D]
$$

每个位置得到一个新的向量，它是所有可见 Value 向量的加权和。

{IMAGE:17}

### 2. 直观理解

对第 $i$ 个 token：

$$
o_i = \sum_{j \le i} a_{ij}v_j
$$

其中：

- $a_{ij}$：第 $i$ 个 token 对第 $j$ 个 token 的注意力权重
- $v_j$：第 $j$ 个 token 的 Value 向量
- $j \le i$：来自因果 mask 的限制，只允许看当前位置及之前的位置

{IMAGE:18}

本节总结：softmax 后的权重与 V 相乘，得到每个 token 聚合上下文后的表示，形状仍为 `[B, H, T, head_dim]`。

---

## 七、拼回多头并做输出投影

### 1. transpose 回 token 主维度

注意力输出当前形状是：

```python
output.shape  # [B, H, T, D]
```

但 Transformer 后续模块通常需要：

```python
[B, T, C]
```

所以先转置：

```python
output = output.transpose(1, 2)
```

得到：

$$
[B,T,H,D]
$$

然后合并 head 维度：

```python
output = output.contiguous().view(bsz, seqlen, -1)
```

因为：

$$
H \times D = C
$$

最终得到：

```python
output.shape  # [B, T, C]
```

{IMAGE:19}

### 2. 为什么要 contiguous

`transpose` 只改变张量的视图，不一定让内存连续。后续使用 `view` 时通常要求内存布局连续，因此常见写法是：

```python
output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
```

如果不加 `.contiguous()`，可能报错，或者在某些情况下产生非预期行为。

{WARNING}易错点{/WARNING}

`view` 依赖内存连续性。经过 `transpose`、`permute` 等操作后，如果要 `view`，通常应先 `.contiguous()`；或者使用更灵活的 `.reshape()`。

### 3. 输出投影

多头拼接后还需要经过一个输出线性层：

```python
output = self.wo(output)
```

这里的 `wo` 通常定义为：

```python
self.wo = nn.Linear(
    args.n_heads * self.head_dim,
    args.dim,
    bias=False
)
```

输出投影的作用是将多个 head 的信息重新混合，回到模型主干维度 `dim`。

{IMAGE:20}

本节总结：注意力结果先从 `[B,H,T,D]` 转回 `[B,T,H,D]`，再合并为 `[B,T,C]`，最后经过 `wo` 输出投影，供残差连接和后续 FFN 使用。

---

## 八、完整注意力 forward 代码串讲

### 1. 代码骨架

下面是一个典型 GQA Attention 的 forward 逻辑：

```python
def forward(self, x):
    """
    x: [B, T, dim]
    """
    bsz, seqlen, _ = x.shape

    # 1. 线性映射得到 Q、K、V
    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)

    # 2. 拆分多头
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    # 3. 调整维度，方便注意力矩阵乘法
    xq = xq.transpose(1, 2)  # [B, n_heads, T, D]
    xk = xk.transpose(1, 2)  # [B, n_kv_heads, T, D]
    xv = xv.transpose(1, 2)  # [B, n_kv_heads, T, D]

    # 4. GQA: 重复 K/V head，让它们与 Q head 对齐
    xk = repeat_kv(xk, self.n_rep)
    xv = repeat_kv(xv, self.n_rep)

    # 5. 计算注意力分数
    scores = torch.matmul(xq, xk.transpose(2, 3))
    scores = scores / math.sqrt(self.head_dim)

    # 6. 构造并添加因果 mask
    mask = torch.full(
        (1, 1, seqlen, seqlen),
        float("-inf"),
        device=x.device
    )
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask

    # 7. softmax 得到注意力权重
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    # 8. 加权求和 Value
    output = torch.matmul(scores, xv)

    # 9. 拼回多头
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    # 10. 输出投影
    output = self.wo(output)

    return output
```

{IMAGE:21}

### 2. 维度流转总表

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `x` | `[B, T, dim]` |
| Q 映射 | `xq` | `[B, T, n_heads * D]` |
| K/V 映射 | `xk/xv` | `[B, T, n_kv_heads * D]` |
| 拆 head | `xq` | `[B, T, n_heads, D]` |
| 拆 head | `xk/xv` | `[B, T, n_kv_heads, D]` |
| 转置 | `xq` | `[B, n_heads, T, D]` |
| 转置 | `xk/xv` | `[B, n_kv_heads, T, D]` |
| repeat KV | `xk/xv` | `[B, n_heads, T, D]` |
| attention scores | `scores` | `[B, n_heads, T, T]` |
| attention output | `output` | `[B, n_heads, T, D]` |
| 拼 head | `output` | `[B, T, dim]` |
| 输出投影 | `output` | `[B, T, dim]` |

{IMAGE:22}

{IMPORTANT}核心概念{/IMPORTANT}

写注意力代码时最重要的是盯住形状。只要 Q、K、V 的维度流转清楚，GQA、MHA、MQA 的差异就只是 head 数量和 KV 复用方式的差异。

本节总结：完整 forward 可以拆成十步：映射、拆头、转置、repeat KV、算分数、加 mask、softmax、乘 V、拼头、输出投影。

---

## 九、因果掩码与训练 / 推理的关系

### 1. 训练阶段

训练时通常一次输入完整序列：

```python
input_ids = [x0, x1, x2, x3, x4]
```

模型并行计算每个位置的输出，但第 $i$ 个位置只能看 $0 \sim i$：

$$
P(x_{i+1}|x_0,x_1,\dots,x_i)
$$

这就是自回归建模目标。

{IMAGE:23}

### 2. 推理阶段

推理时通常逐 token 生成：

```python
x0 -> x1 -> x2 -> x3 -> ...
```

由于未来 token 还不存在，本身不会看到未来。但如果使用 KV cache，每一步仍然要保证当前位置只与历史 KV 计算注意力。

GQA 的优势在推理阶段尤其明显，因为 KV cache 的大小与 `n_kv_heads` 相关，而不是与 `n_heads` 完全相同。

KV cache 大小近似与下面成正比：

$$
B \times T \times n\_{kv\_heads} \times head\_dim
$$

所以减少 `n_kv_heads` 可以显著降低显存占用。

{IMAGE:24}

本节总结：训练阶段依赖因果 mask 阻止信息泄漏；推理阶段 GQA 可以减少 KV cache 压力，提高部署效率。

---

## 十、常见问题与调试方法

### 1. scores 维度不对

如果报错类似矩阵乘法维度不匹配，优先检查：

```python
print(xq.shape)
print(xk.shape)
print(xv.shape)
```

正确情况下应为：

```python
xq: [B, n_heads, T, D]
xk: [B, n_heads, T, D]
xv: [B, n_heads, T, D]
```

如果 `xk`、`xv` 还是 `[B, n_kv_heads, T, D]`，说明忘了 `repeat_kv`。

### 2. mask 设备不一致

如果输入在 GPU 上，但 mask 在 CPU 上，会报 device mismatch。应使用：

```python
device=x.device
```

或者：

```python
mask = mask.to(x.device)
```

### 3. dtype 问题

混合精度时要注意：

```python
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

这能提升 softmax 数值稳定性。

{IMAGE:25}

### 4. transpose 后 view 报错

典型原因是张量非连续：

```python
output = output.transpose(1, 2).view(bsz, seqlen, -1)
```

应改为：

```python
output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
```

或者：

```python
output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
```

{WARNING}易错点{/WARNING}

很多 attention bug 不是公式错，而是维度顺序错、mask 广播错、设备不一致、dtype 不稳定或 transpose 后直接 view。

本节总结：调试注意力代码时，优先打印形状、设备、dtype，并确认 mask 的位置、方向和广播方式。

---

## 十一、从代码理解 GQA、MHA、MQA 的关系

### 1. MHA

Multi-Head Attention：

$$
n\_heads = n\_kv\_heads
$$

每个 Q head 都有独立 K/V。

### 2. MQA

Multi-Query Attention：

$$
n\_kv\_heads = 1
$$

所有 Q head 共享同一组 K/V。

### 3. GQA

Grouped-Query Attention：

$$
1 < n\_kv\_heads < n\_heads
$$

多个 Q head 分组共享 K/V。

{IMAGE:4}

可以把三者看作同一套代码的不同参数配置：

```python
# MHA
n_heads = 8
n_kv_heads = 8

# GQA
n_heads = 8
n_kv_heads = 2

# MQA
n_heads = 8
n_kv_heads = 1
```

共享比例：

$$
n\_{rep} = \frac{n\_{heads}}{n\_{kv\_heads}}
$$

{IMAGE:5}

本节总结：MHA、GQA、MQA 的本质差异是 KV head 的数量。GQA 在表达能力和推理效率之间取得折中。

---

## 十二、本讲关键收获

{IMAGE:6}

1. 注意力分数由 $QK^T / \sqrt{d_k}$ 得到，形状是 `[B, H, T, T]`。
2. 因果 mask 用于屏蔽未来 token，通常通过给上三角位置加 `-inf` 实现。
3. softmax 应在最后一维进行，表示每个 query token 对所有可见 key token 的概率分布。
4. 注意力权重乘以 V 后得到上下文向量，形状回到 `[B, H, T, D]`。
5. 多头输出需要转置、合并，再经过 `wo` 输出投影回到 `[B, T, dim]`。
6. GQA 通过减少 `n_kv_heads` 降低 KV cache 开销，同时通过 `repeat_kv` 适配 Q head。
7. 写 attention 代码时，维度顺序比公式本身更容易出错。

{IMAGE:7}

## 思考题

1. 为什么因果 mask 要加在 softmax 之前，而不是 softmax 之后？
2. 如果 `n_heads=16`，`n_kv_heads=4`，那么 `repeat_kv` 中的 `n_rep` 等于多少？K/V 最终会变成多少个 head？
3. GQA 相比 MHA 为什么更适合大模型推理阶段？主要节省的是哪一部分显存？

{IMAGE:8}