# 第12集: 代码：GQA 上

## 课程定位与本集目标

本集是 MiniMind 第 12/26 集，主题是“代码：GQA 上”，重点围绕注意力模块中的 **Q/K/V 投影** 与 **repeat_kv** 展开。它处在从 Transformer 理论进入 PyTorch 实现的关键阶段：前面我们已经知道注意力机制需要 Query、Key、Value 三组向量，本集开始真正把这些张量如何从隐藏状态中投影出来、如何拆分多头、以及如何支持 GQA 写进代码。

{IMAGE:1}

{IMPORTANT}本集核心目标：理解 MiniMind 中注意力层如何把输入 hidden states 变成 Q/K/V，并解释为什么 GQA 需要 `repeat_kv` 将较少的 K/V 头扩展到和 Q 头数量一致。{/IMPORTANT}

### 本节小结

本集不是只讲公式，而是把公式落到 PyTorch 张量形状和代码实现上。理解本集的关键是始终跟踪张量维度变化。

---

## 从输入 hidden states 到 Q/K/V

在 Transformer 的每一层中，注意力模块接收的输入通常是：

$$
X \in \mathbb{R}^{B \times T \times C}
$$

其中：

- $B$：batch size
- $T$：序列长度
- $C$：hidden size，也就是模型隐藏维度

注意力机制不会直接拿 $X$ 做计算，而是先通过三个线性层投影得到：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

{IMAGE:12}

在代码中通常对应：

```python
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)
```

如果输入 `x` 的形状是：

```python
(batch_size, seq_len, hidden_size)
```

那么投影后的形状大致是：

```python
q: (batch_size, seq_len, num_attention_heads * head_dim)
k: (batch_size, seq_len, num_key_value_heads * head_dim)
v: (batch_size, seq_len, num_key_value_heads * head_dim)
```

这里要注意，GQA 中 `q` 的头数和 `k/v` 的头数可以不同。

{IMAGE:2}

{KNOWLEDGE}传统 MHA 中，Q、K、V 的头数通常相同；而 GQA 中，Q 的头数更多，K/V 的头数更少。这样可以减少 KV Cache 的显存占用，同时保持较好的表达能力。{/KNOWLEDGE}

### 本节小结

Q/K/V 都来自同一个输入 `x`，但通过不同线性层投影得到。GQA 的特殊之处在于：Q 的头数通常大于 K/V 的头数。

---

## 多头注意力中的维度拆分

投影后的 Q/K/V 还不能直接进入注意力计算，因为多头注意力需要把 hidden dimension 拆成多个 head。

设：

$$
C = H \times D
$$

其中：

- $H$：注意力头数
- $D$：每个 head 的维度，即 `head_dim`

对于 Query：

$$
Q \in \mathbb{R}^{B \times T \times H_qD}
$$

需要 reshape 成：

$$
Q \in \mathbb{R}^{B \times T \times H_q \times D}
$$

再 transpose 成：

$$
Q \in \mathbb{R}^{B \times H_q \times T \times D}
$$

代码形式一般是：

```python
q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
q = q.transpose(1, 2)
```

{IMAGE:3}

对 Key 和 Value 也类似，但头数是 `num_key_value_heads`：

```python
k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

此时形状为：

```python
q: (B, num_heads, T, head_dim)
k: (B, num_key_value_heads, T, head_dim)
v: (B, num_key_value_heads, T, head_dim)
```

{IMAGE:4}

{WARNING}易错点：`view` 之前必须确保最后一维大小能整除 head 数。也就是说 `hidden_size = num_heads * head_dim`。如果配置不一致，会直接形状错误。{/WARNING}

### 本节小结

多头注意力的本质是把一个大的 hidden vector 拆成多个小 head，并让每个 head 独立计算注意力。GQA 中 Q 和 K/V 拆分出来的 head 数不一样。

---

## MHA、MQA 与 GQA 的区别

为了理解 `repeat_kv`，必须先理解三种注意力结构的区别。

### MHA：Multi-Head Attention

普通多头注意力中：

$$
H_q = H_k = H_v
$$

例如：

```python
num_heads = 8
num_key_value_heads = 8
```

每个 Query head 都有自己对应的 Key head 和 Value head。

### MQA：Multi-Query Attention

MQA 中所有 Query head 共享同一组 K/V：

$$
H_q > 1,\quad H_k = H_v = 1
$$

例如：

```python
num_heads = 8
num_key_value_heads = 1
```

它显著减少 KV Cache，但可能损失一部分表达能力。

### GQA：Grouped-Query Attention

GQA 是 MHA 和 MQA 的折中：

$$
1 < H_{kv} < H_q
$$

例如：

```python
num_heads = 8
num_key_value_heads = 2
```

这意味着 8 个 Query head 被分成若干组，每组共享一个 K/V head。

{IMAGE:13}

如果：

```python
num_heads = 8
num_key_value_heads = 2
```

那么每个 K/V head 被 4 个 Query head 共享：

$$
\text{num\_key\_value\_groups} = \frac{H_q}{H_{kv}} = \frac{8}{2} = 4
$$

{IMPORTANT}GQA 的核心思想：Query 保持较多头数以维持表达能力，Key/Value 使用较少头数以减少计算和缓存开销。{/IMPORTANT}

### 本节小结

`repeat_kv` 存在的根本原因是：注意力计算要求 Q/K/V 的 head 数在矩阵乘法时对齐，而 GQA 中 K/V head 数比 Q 少。

---

## 注意力计算为什么要求 head 数对齐

标准 scaled dot-product attention 公式为：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

在多头场景下，通常对每个 head 分别计算：

```python
attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
attn_output = torch.matmul(attn_weights, v)
```

此时张量形状为：

```python
q: (B, Hq, T, D)
k: (B, Hkv, T, D)
```

执行：

```python
q @ k.transpose(-2, -1)
```

需要 head 维度能够广播或一致。最清晰的实现方式是把 K/V 扩展成和 Q 相同的 head 数：

```python
k: (B, Hq, T, D)
v: (B, Hq, T, D)
```

于是就引出了：

```python
k = repeat_kv(k, num_key_value_groups)
v = repeat_kv(v, num_key_value_groups)
```

{IMAGE:14}

### 本节小结

注意力计算时，Q 和 K/V 要在 head 维度上对应。GQA 通过重复 K/V head，让较少的 K/V head 匹配较多的 Q head。

---

## repeat_kv 的作用

`repeat_kv` 的功能可以概括为：

> 将 K/V 的 head 数从 `num_key_value_heads` 扩展到 `num_attention_heads`。

假设：

```python
x.shape = (B, num_key_value_heads, T, head_dim)
```

希望重复后变成：

```python
(B, num_attention_heads, T, head_dim)
```

其中：

```python
num_attention_heads = num_key_value_heads * num_key_value_groups
```

{IMAGE:15}

典型实现如下：

```python
def repeat_kv(hidden_states, n_rep):
    """
    hidden_states: (batch, num_key_value_heads, seq_len, head_dim)
    n_rep: 每个 KV head 需要重复的次数
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )

    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        seq_len,
        head_dim,
    )
```

这个函数最关键的几步是：

1. 在 KV head 后面插入一个新维度。
2. 用 `expand` 在新维度上重复。
3. 用 `reshape` 把两个 head 相关维度合并。

{IMAGE:16}

### 本节小结

`repeat_kv` 不是改变 K/V 的内容，而是把每个 K/V head 复制给多个 Query head 使用。

---

## repeat_kv 的形状推导

以一个具体例子说明。

假设：

```python
batch_size = 2
seq_len = 4
num_heads = 8
num_key_value_heads = 2
head_dim = 16
```

那么：

```python
num_key_value_groups = num_heads // num_key_value_heads
# 8 // 2 = 4
```

K/V 初始形状：

```python
k: (2, 2, 4, 16)
v: (2, 2, 4, 16)
```

调用：

```python
k = repeat_kv(k, 4)
v = repeat_kv(v, 4)
```

中间过程：

```python
# 原始
(2, 2, 4, 16)

# 插入维度
(2, 2, 1, 4, 16)

# expand
(2, 2, 4, 4, 16)

# reshape
(2, 8, 4, 16)
```

最终：

```python
q: (2, 8, 4, 16)
k: (2, 8, 4, 16)
v: (2, 8, 4, 16)
```

{IMAGE:17}

然后就可以进行注意力分数计算：

$$
S = \frac{QK^T}{\sqrt{D}}
$$

对应形状：

```python
q:             (B, H, T, D)
k.transpose:   (B, H, D, T)
attn_weights:  (B, H, T, T)
```

### 本节小结

`repeat_kv` 的核心不是复杂计算，而是维度重排。只要能跟住 `(B, H, T, D)`，GQA 的代码就容易理解。

---

## 为什么使用 expand 而不是直接 repeat

在 PyTorch 中，`repeat` 和 `expand` 都能产生“重复”的效果，但语义和内存行为不同。

```python
x_repeat = x.repeat(...)
x_expand = x.expand(...)
```

通常：

- `repeat` 会真实复制数据，占用更多内存。
- `expand` 尽量通过 stride 视图实现，不立即复制底层数据。
- 后续 `reshape` 可能触发连续化或拷贝，但整体写法仍然是常见高效实现。

{IMAGE:18}

`repeat_kv` 的经典实现借鉴了 LLaMA 等模型中的写法：

```python
hidden_states = hidden_states[:, :, None, :, :].expand(...)
return hidden_states.reshape(...)
```

这样写的目的主要是清楚表达：

> 每个 KV head 在 group 维度上被共享给多个 Q head。

{WARNING}易错点：`expand` 得到的张量可能不是 contiguous 的。如果后续代码要求连续内存，需要留意 `.contiguous()` 或 reshape 的实际行为。{/WARNING}

### 本节小结

`expand` 更符合“共享视图”的语义，避免无意义的立即复制，是实现 `repeat_kv` 的常见方式。

---

## Q/K/V 投影层的配置关系

在 MiniMind 或类似 LLaMA 架构中，注意力层初始化时通常会定义：

```python
self.num_heads = config.num_attention_heads
self.num_key_value_heads = config.num_key_value_heads
self.num_key_value_groups = self.num_heads // self.num_key_value_heads
self.head_dim = config.hidden_size // self.num_heads
```

然后定义投影层：

```python
self.q_proj = nn.Linear(
    config.hidden_size,
    self.num_heads * self.head_dim,
    bias=False,
)

self.k_proj = nn.Linear(
    config.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=False,
)

self.v_proj = nn.Linear(
    config.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=False,
)

self.o_proj = nn.Linear(
    self.num_heads * self.head_dim,
    config.hidden_size,
    bias=False,
)
```

{IMAGE:19}

这里有几个关键点：

- `q_proj` 输出维度与 Q head 数相关。
- `k_proj` 和 `v_proj` 输出维度与 KV head 数相关。
- `o_proj` 接收拼接后的所有 attention heads，再投影回 hidden size。

数学上可以理解为：

$$
W_Q \in \mathbb{R}^{C \times H_qD}
$$

$$
W_K, W_V \in \mathbb{R}^{C \times H_{kv}D}
$$

$$
W_O \in \mathbb{R}^{H_qD \times C}
$$

### 本节小结

GQA 的参数节省主要体现在 K/V 投影层和 KV Cache 上，因为 K/V 的 head 数减少了。

---

## MiniMind 注意力前向过程梳理

一个典型的 GQA attention forward 流程可以写成：

```python
def forward(self, x, attention_mask=None):
    batch_size, seq_len, hidden_size = x.shape

    # 1. Q/K/V 投影
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 2. 拆分多头
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

    # 3. 调整维度到 attention 习惯格式
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 4. GQA: 将 K/V 重复到与 Q head 数一致
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    # 5. 注意力分数
    attn_scores = torch.matmul(q, k.transpose(2, 3))
    attn_scores = attn_scores / math.sqrt(self.head_dim)

    # 6. 加 mask
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # 7. softmax 得到权重
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # 8. 加权求和
    attn_output = torch.matmul(attn_weights, v)

    # 9. 合并多头
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)

    # 10. 输出投影
    output = self.o_proj(attn_output)

    return output
```

{IMAGE:20}

这个过程对应的张量变化可以总结为：

```python
x:
(B, T, C)

q projection:
(B, T, Hq * D)

k/v projection:
(B, T, Hkv * D)

after view + transpose:
q: (B, Hq, T, D)
k: (B, Hkv, T, D)
v: (B, Hkv, T, D)

after repeat_kv:
q: (B, Hq, T, D)
k: (B, Hq, T, D)
v: (B, Hq, T, D)

attention weights:
(B, Hq, T, T)

attention output:
(B, Hq, T, D)

merged output:
(B, T, C)
```

### 本节小结

前向传播可以分成五个阶段：投影、拆头、重复 K/V、注意力计算、合并输出。每一步都围绕形状变换展开。

---

## attention mask 与 causal attention 的关系

虽然本集主题集中在 Q/K/V 和 `repeat_kv`，但实际 attention 计算中通常还会加入 mask。

自回归语言模型不能看到未来 token，因此需要 causal mask：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

加入注意力分数：

$$
S' = S + M
$$

再做 softmax：

$$
A = \text{softmax}(S')
$$

这样当前位置只能关注自己和之前的位置。

{IMAGE:21}

代码中可能类似：

```python
attn_scores = attn_scores + attention_mask
attn_weights = torch.softmax(attn_scores, dim=-1)
```

{WARNING}易错点：mask 的形状通常需要能广播到 `(B, H, T, T)`。如果 mask 维度不匹配，可能出现广播错误或错误掩码。{/WARNING}

### 本节小结

`repeat_kv` 解决的是 head 对齐问题，mask 解决的是可见性问题。两者作用不同，但都发生在注意力计算过程中。

---

## GQA 对 KV Cache 的意义

在推理阶段，自回归模型每生成一个 token，都会保存历史 token 的 K/V，这就是 KV Cache。

如果使用普通 MHA：

```python
k_cache: (B, num_heads, past_len, head_dim)
v_cache: (B, num_heads, past_len, head_dim)
```

如果使用 GQA：

```python
k_cache: (B, num_key_value_heads, past_len, head_dim)
v_cache: (B, num_key_value_heads, past_len, head_dim)
```

因为：

$$
H_{kv} < H_q
$$

所以 KV Cache 显著变小。

{IMAGE:22}

例如：

```python
num_heads = 32
num_key_value_heads = 8
```

KV Cache 大小约减少到原来的：

$$
\frac{8}{32} = \frac{1}{4}
$$

也就是 25%。

{IMPORTANT}GQA 对大模型推理尤其重要，因为长上下文推理时 KV Cache 往往是显存占用的主要来源之一。{/IMPORTANT}

### 本节小结

GQA 不只是训练时的结构变化，更重要的是推理时减少 KV Cache，降低显存压力，提高长文本生成效率。

---

## 代码实现中的常见易错点

### 1. `num_heads` 必须能整除 `num_key_value_heads`

通常需要满足：

$$
H_q \bmod H_{kv} = 0
$$

否则无法得到整数的 `num_key_value_groups`。

```python
assert self.num_heads % self.num_key_value_heads == 0
```

{IMAGE:23}

### 2. `hidden_size` 必须能整除 `num_heads`

因为：

$$
D = \frac{C}{H_q}
$$

所以：

```python
assert hidden_size % num_heads == 0
```

### 3. transpose 后使用 view 前要 contiguous

例如：

```python
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(batch_size, seq_len, hidden_size)
```

如果没有 `.contiguous()`，某些情况下 `view` 会报错。

{IMAGE:24}

### 4. 不要把 repeat_kv 放错位置

正确流程是先把 K/V 变成：

```python
(B, Hkv, T, D)
```

再 repeat 成：

```python
(B, Hq, T, D)
```

如果在 `(B, T, Hkv, D)` 时重复，后续维度很容易混乱。

### 本节小结

GQA 代码最常见的问题不是公式错，而是维度顺序错、整除关系错、以及 `transpose` 后直接 `view`。

---

## 从概念到 MiniMind 实现的理解路径

学习本集时，可以按以下顺序建立心智模型：

1. 输入 `x` 是所有 token 的隐藏状态。
2. `q_proj/k_proj/v_proj` 分别得到 Q/K/V。
3. Q 使用 `num_heads`，K/V 使用 `num_key_value_heads`。
4. reshape + transpose 后进入 `(B, H, T, D)` 格式。
5. `repeat_kv` 把 K/V head 数扩展到 Q head 数。
6. 计算 scaled dot-product attention。
7. 合并多头并通过 `o_proj` 回到 hidden size。

{IMAGE:25}

这条路径背后的数学关系是：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)
$$

$$
O = AV
$$

$$
Y = OW_O
$$

其中 GQA 只改变了 K/V 的 head 组织方式，不改变注意力计算的基本公式。

### 本节小结

GQA 并不是新发明一套注意力公式，而是在多头组织方式上做优化，使 K/V 能被多个 Q head 共享。

---

## 图示串讲：从投影到 repeat_kv

{IMAGE:5}

这一部分通常展示代码中注意力层的结构。可以重点观察：

- 初始化中有哪些线性层。
- `q_proj/k_proj/v_proj` 的输出维度是否不同。
- 是否计算了 `num_key_value_groups`。

{IMAGE:6}

在进入 forward 后，重点看输入形状如何被拆开：

```python
bsz, q_len, _ = hidden_states.size()
```

然后进行投影：

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)
```

{IMAGE:7}

这里要把代码和公式对应起来：

$$
\text{query\_states} = XW_Q
$$

$$
\text{key\_states} = XW_K
$$

$$
\text{value\_states} = XW_V
$$

{IMAGE:8}

接下来是 reshape：

```python
query_states = query_states.view(
    bsz, q_len, self.num_heads, self.head_dim
).transpose(1, 2)

key_states = key_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
).transpose(1, 2)

value_states = value_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
).transpose(1, 2)
```

{IMAGE:9}

这一步之后，Query 和 Key/Value 的 head 数不同：

```python
query_states: (B, num_heads, T, D)
key_states:   (B, num_key_value_heads, T, D)
value_states: (B, num_key_value_heads, T, D)
```

{IMAGE:10}

然后执行：

```python
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)
```

{IMAGE:26}

重复完成后：

```python
key_states:   (B, num_heads, T, D)
value_states: (B, num_heads, T, D)
```

最后才能和 Query 共同进入矩阵乘法：

```python
attn_weights = torch.matmul(
    query_states,
    key_states.transpose(2, 3),
) / math.sqrt(self.head_dim)
```

{IMAGE:27}

### 本节小结

图示部分要抓住一条主线：Q/K/V 投影后先变成多头格式，然后 K/V 通过 `repeat_kv` 补齐 head 数，最后进入注意力计算。

---

## 完整知识点总结

{IMAGE:11}

{IMPORTANT}本集最重要的知识点有三个：第一，Q/K/V 是由输入 hidden states 经过不同线性层投影得到的；第二，GQA 中 Q head 数大于 K/V head 数；第三，`repeat_kv` 用于把 K/V head 扩展到与 Q head 对齐。{/IMPORTANT}

从工程实现看，注意力模块不是一行公式，而是一连串严密的张量变换：

```python
x
-> q_proj / k_proj / v_proj
-> view
-> transpose
-> repeat_kv
-> q @ k.transpose
-> softmax
-> attn @ v
-> transpose
-> view
-> o_proj
```

从数学理解看，核心公式仍然是：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
$$

GQA 改变的是 $K,V$ 的 head 数量与共享方式，而不是注意力的基本计算逻辑。

### 本节小结

掌握本集后，应能独立读懂 MiniMind 中 attention forward 的主要代码，并能解释每次 reshape、transpose、repeat 的目的。

---

## Key Takeaways

1. Q/K/V 都由输入 hidden states 线性投影得到，但 GQA 中 Q 和 K/V 的输出头数不同。
2. 多头注意力代码的核心难点是张量形状管理，尤其是 `(B, T, C)` 到 `(B, H, T, D)` 的转换。
3. `repeat_kv` 的作用是把较少的 K/V head 扩展到与 Q head 数一致。
4. GQA 在 MHA 和 MQA 之间折中，减少 KV Cache 的同时保留较强表达能力。
5. 写 attention 代码时要特别检查整除关系、transpose 后的 contiguous、mask 广播形状，以及 repeat 的维度位置。

## 思考题

1. 如果 `num_heads = 16`，`num_key_value_heads = 4`，那么 `num_key_value_groups` 等于多少？K/V 每个 head 会被几个 Query head 共享？
2. 为什么 GQA 能减少推理阶段的 KV Cache 显存占用？
3. 如果忘记对 `key_states` 和 `value_states` 调用 `repeat_kv`，在注意力矩阵乘法时可能出现什么问题？