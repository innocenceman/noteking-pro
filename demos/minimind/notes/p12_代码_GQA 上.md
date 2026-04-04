# 第12集: 代码：GQA 上

# MiniMind Episode 12/26 讲义：代码：GQA 上

## 课程概述

**课程名称**：MiniMind - PyTorch从零手敲大模型  
**本集时长**：13分8秒  
**核心主题**：Q/K/V投影、repeat_kv  
**代码实现**：Grouped Query Attention (GQA) 的核心组件

{IMAGE:1}

---

## 1. 引言：为什么需要 GQA？

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在传统的 **Multi-Head Attention (MHA)** 机制中，每个注意力头都有独立的 Query (Q)、Key (K) 和 Value (V) 投影。对于一个拥有 $n_{heads}$ 个注意力头的大型语言模型，这会消耗大量的计算资源和显存。

**GQA (Grouped Query Attention)** 是 LLaMA 等模型采用的一种优化策略，其核心思想是：
- 使用较少的 **Key 头** 和 **Value 头**（$n_{kv\_heads}$）
- 使用较多的 **Query 头**（$n_{q\_heads}$）
- 通常 $n_{kv\_heads} < n_{q\_heads}$

{IMPORTANT}核心概念{/IMPORTANT}

$$n_{q\_heads} = n_{heads} \quad \text{（通常等于隐藏层维度）}$$
$$n_{kv\_heads} \ll n_{q\_heads} \quad \text{（如 LLaMA-7B 中 } n_{q}=32, n_{kv}=8\text{）}$$

{IMAGE:2}

---

## 2. Q/K/V 投影详解

### 2.1 投影的本质

{Q/K/V 投影}是 Attention 机制中将输入隐藏状态 $x$ 线性变换为 Query、Key、Value 向量的过程。

对于输入 $x \in \mathbb{R}^{B \times L \times H}$（其中 $B$ 为批次大小，$L$ 为序列长度，$H$ 为隐藏维度）：

$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$

其中：
- $W_Q \in \mathbb{R}^{H \times H}$（或 $H \times (H / n_{q\_heads} \times n_{q\_heads})$）
- $W_K \in \mathbb{R}^{H \times K}$，这里 $K$ 是 Key 向量的总维度
- $W_V \in \mathbb{R}^{H \times K}$，这里 $K$ 是 Value 向量的总维度

{IMAGE:3}

### 2.2 代码实现

让我们来看 PyTorch 中 Q/K/V 投影的完整实现：

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads  # Key/Value 头数量
        self.n_q_heads = args.n_q_heads    # Query 头数量
        self.n_heads = args.n_heads        # 总头数
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        
        # Q 投影：需要为每个 Query 头创建投影
        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # KV 投影：只需要较少的头
        self.wk = Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)
```

{IMAGE:4}

{IMPORTANT}核心概念{/IMPORTANT}

注意 Q 投影的输出维度是 `n_q_heads * head_dim`，而 K/V 投影的输出维度是 `n_kv_heads * head_dim`。这正是 GQA 节省参数的关键！

### 2.3 投影维度的数学推导

假设：
- 隐藏维度 $H_{dim} = 4096$
- Query 头数 $n_q = 32$
- Key/Value 头数 $n_{kv} = 8$
- 头维度 $d_{head} = H_{dim} / n_q = 4096 / 32 = 128$

则投影矩阵的参数量：

$$W_Q: H_{dim} \times (n_q \times d_{head}) = 4096 \times 4096$$

$$W_K: H_{dim} \times (n_{kv} \times d_{head}) = 4096 \times 1024$$

$$W_V: H_{dim} \times (n_{kv} \times d_{head}) = 4096 \times 1024$$

通过减少 $n_{kv}$，K/V 投影参数量减少至 MHA 的 $\frac{n_{kv}}{n_q}$ 倍！

{IMAGE:5}

---

## 3. repeat_kv 机制

### 3.1 为什么需要 repeat_kv？

{IMAGE:6}

当我们计算注意力分数时：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

这个计算要求 $Q$ 和 $K$ 的头数匹配。但在 GQA 中：
- $Q$ 有 $n_q$ 个头
- $K$ 只有 $n_{kv}$ 个头

**repeat_kv** 就是为了解决这个维度不匹配问题！

### 3.2 repeat_kv 的工作原理

{WARNING}易错点{/WARNING}

`repeat_kv` 的核心思想是：**将每个 Key/Value 头复制多次**，以匹配 Query 头的数量。

数学上，如果 $n_q = 32$，$n_{kv} = 8$，则每个 KV 头需要重复：

$$\text{repeat\_factor} = \frac{n_q}{n_{kv}} = \frac{32}{8} = 4 \text{ 次}$$

{IMAGE:7}

### 3.3 代码实现

```python
def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    将 Key/Value 张量在头维度上重复 n_rep 次
    
    参数:
        x: 张量形状 [batch, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数
    
    返回:
        张量形状 [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    
    if n_rep == 1:
        return x  # 无需重复（MHA 情况）
    
    # reshape: [B, L, n_kv_heads, 1, head_dim]
    x = x[:, :, :, None, :]
    # expand: [B, L, n_kv_heads, n_rep, head_dim]
    x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    # reshape: [B, L, n_kv_heads * n_rep, head_dim]
    x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    
    return x
```

{IMAGE:8}

### 3.4 图解 repeat_kv 操作

{IMAGE:9}

{KNOWLEDGE}内存效率{/KNOWLEDGE}

虽然 repeat_kv 在计算时扩展了 KV 矩阵，但实际存储的 KV 缓存（用于自回归生成）仍然只保存 $n_{kv}$ 个头的权重。这大大减少了推理时的显存占用！

---

## 4. 完整的 Attention 前向传播

### 4.1 forward 方法实现

```python
def forward(
    self,
    x: Tensor,
    start_pos: int = 0,
    freqs_cis: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
):
    """
    Attention 前向传播
    
    参数:
        x: 输入张量 [batch, seq_len, hidden_dim]
        start_pos: 当前位置（用于 KV 缓存）
        freqs_cis: 旋转位置编码的复数形式
        mask: 注意力掩码
    """
    bsz, seqlen, _ = x.shape
    
    # 1. Q/K/V 投影
    xq = self.wq(x)  # [B, L, n_q_heads * head_dim]
    xk = self.wk(x)  # [B, L, n_kv_heads * head_dim]
    xv = self.wv(x)  # [B, L, n_kv_heads * head_dim]
    
    # 2. Reshape 为多头格式
    xq = xq.view(bsz, seqlen, self.n_q_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    
    # 3. 应用旋转位置编码
    if freqs_cis is not None:
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    
    # 4. 处理 KV 缓存（推理时使用）
    if self.cache is not None:
        keys = self.cache.update(xk, start_pos, self.n_kv_heads)
        values = self.cache.update(xv, start_pos, self.n_kv_heads)
    else:
        keys = xk
        values = xv
    
    # 5. repeat_kv：将 KV 头扩展到与 Q 头匹配
    keys = repeat_kv(keys, self.n_rep)  # n_rep = n_q_heads / n_kv_heads
    values = repeat_kv(values, self.n_rep)
    
    # 6. 计算注意力
    # 形状调整以进行批量矩阵乘法
    xq = xq.transpose(1, 2)  # [B, n_q_heads, L, head_dim]
    keys = keys.transpose(1, 2)  # [B, n_q_heads, L, head_dim]
    values = values.transpose(1, 2)  # [B, n_q_heads, L, head_dim]
    
    # 计算注意力分数
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    # 应用掩码
    if mask is not None:
        scores = scores + mask
    
    # softmax + 加权求和
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)
    
    # 7. 合并多头输出
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    
    # 8. 输出投影
    return self.wo(output)
```

{IMAGE:10}

{IMAGE:11}

### 4.2 关键参数 n_rep

```python
# 在 __init__ 中计算重复因子
self.n_rep = self.n_q_heads // self.n_kv_heads
```

这个比值决定了每个 KV 头需要复制多少次。典型配置：
- LLaMA-7B: $n_q=32, n_{kv}=8 \Rightarrow n_{rep}=4$
- LLaMA-70B: $n_q=64, n_{kv}=8 \Rightarrow n_{rep}=8$

{IMAGE:12}

---

## 5. GQA vs MHA vs MQA 对比

| 特性 | MHA | MQA | GQA |
|------|-----|-----|-----|
| Q 头数 | $n_h$ | $n_h$ | $n_h$ |
| K 头数 | $n_h$ | 1 | $n_{kv}$ |
| V 头数 | $n_h$ | 1 | $n_{kv}$ |
| KV 缓存 | 大 | 最小 | 中等 |
| 参数量 | 最大 | 最小 | 中等 |
| 推理速度 | 慢 | 最快 | 快 |
| 表达能力 | 强 | 弱 | 中等 |

{IMPORTANT}核心概念{/IMPORTANT}

GQA 在**推理效率**和**表达能力**之间取得了良好的平衡，是现代大模型的主流选择（如 LLaMA、Mistral、Qwen 等）。

---

## 6. 本章小结

### 核心知识点

1. **Q/K/V 投影**：将输入状态线性变换为 Query、Key、Value 向量，是 Attention 机制的基础操作。

2. **repeat_kv 机制**：在 GQA 中，通过将较少的 KV 头重复扩展，与较多的 Q 头进行匹配，解决维度不匹配问题。

3. **参数量节省**：减少 $n_{kv}$ 可以显著降低 K/V 投影和 KV 缓存的参数量。

4. **计算效率**：虽然 repeat_kv 扩展了计算时的矩阵大小，但不影响 KV 缓存的实际存储大小。

### 关键公式

$$n_{rep} = \frac{n_{q\_heads}}{n_{kv\_heads}}$$

$$W_K \in \mathbb{R}^{H \times (n_{kv} \cdot d_{head})}$$

---

## 7. 思考题

### 思考题 1

> 在 LLaMA 模型中，如果隐藏维度为 4096，n_heads=32，n_kv_heads=8，请计算：
> - 每个头的维度 `head_dim`
> - Q 投影的输出维度
> - K/V 投影的输出维度
> - `repeat_kv` 的重复因子 `n_rep`

### 思考题 2

> GQA 通过 repeat_kv 机制解决了 Q/K 头数不匹配的问题。请分析：如果我们将 Q 头压缩到与 KV 头数量相同（而不是将 KV 头扩展到与 Q 头数量相同），会有什么不同？这种方法可行吗？

---

*下一集预告：在《代码：GQA 下》中，我们将继续深入探讨 KV 缓存的实现细节，以及 Flash Attention 与 GQA 的结合使用。*