# 第13集: 代码：GQA 下

# MiniMind Episode 13/26 讲义
## 代码：GQA 下 (19m22s)
### 注意力计算、因果掩码、输出投影

---

## 本节概述

{IMAGE:1}

本节课我们将深入探讨 **Grouped Query Attention (GQA)** 的核心计算逻辑，包括：

1. **注意力分数的计算过程**
2. **因果掩码（Causal Mask）的作用与实现**
3. **输出投影层的设计与实现**

{IMPORTANT}本节是 GQA 实现的重点内容，建议在观看视频的同时配合代码实践。{/IMPORTANT}

---

## 第一部分：GQA 注意力计算详解

### 1.1 从多头注意力到分组查询注意力

{KNOWLEDGE}背景知识{/KNOWLEDGE}

回顾 MHA（Multi-Head Attention）的结构：
- 每个头有独立的 $W_Q, W_K, W_V$ 投影矩阵
- 计算成本：$O(n \times d_{model}^2)$

GQA 的核心改进：
- **多个查询头（Query Heads）** 共享 **更少的键值头（Key-Value Heads）**
- 减少键值对的计算和存储开销

$$N_{total} = N_{query} \times d_{head}$$
$$N_{kv} = N_{kv\_heads} \times d_{head}$$

{IMAGE:2}

### 1.2 注意力分数计算公式

{IMAGE:3}

标准的缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于 GQA，我们需要：
1. 将 $N_q$ 个查询头映射到 $N_{kv}$ 个键值头
2. 每个键值头被多个查询头共享

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads          # 查询头数量
        self.num_kv_heads = num_kv_heads    # 键值头数量
        self.head_dim = d_model // num_heads
        
        # 确保 num_heads 是 num_kv_heads 的整数倍
        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads
```

### 1.3 分数扩展机制（Score Extension）

{IMAGE:4}

核心问题：**如何将 $N_q$ 个查询头的注意力分数映射到 $N_{kv}$ 个键值头？**

解决方案：复制扩展

$$Q_{expanded} = Q \cdot W_{Q\_expand}$$

```python
def forward(self, x):
    batch_size, seq_len, _ = x.shape
    
    # 计算 Q, K, V
    q = self.W_q(x)  # [batch, seq, num_heads * head_dim]
    k = self.W_k(x)
    v = self.W_v(x)
    
    # 重塑为多头格式
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    
    # 将查询头扩展以匹配键值头数量
    # [batch, num_heads, seq, head_dim] -> [batch, num_kv_heads, num_queries_per_kv, seq, head_dim]
    q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # 扩展查询：每个 KV 头对应多个 Q 头
    q = q.view(batch_size, self.num_kv_heads, 
               self.num_queries_per_kv, seq_len, self.head_dim)
```

{IMAGE:5}

---

## 第二部分：因果掩码（Causal Masking）

### 2.1 因果掩码的必要性

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在语言模型中，我们需要确保**当前位置只能看到之前的位置**，不能"看到未来"。这称为**因果性（Causality）**。

$$P(w_t | w_1, w_2, ..., w_{t-1})$$

{IMAGE:6}

### 2.2 因果掩码的实现

{IMAGE:7}

方法：创建一个下三角矩阵，将未来位置设为 $-\infty$

$$\text{Mask}_{i,j} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

```python
def _apply_causal_mask(self, scores, seq_len):
    """
    应用因果掩码，确保位置 i 只能attend到位置 j <= i
    
    Args:
        scores: 注意力分数 [batch, num_heads, seq_len, seq_len]
    
    Returns:
        masked_scores: 带掩码的注意力分数
    """
    # 创建因果掩码矩阵
    # [seq_len, seq_len] -> 下三角为0，上三角为-inf
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1
    )
    
    # 将掩码应用到分数上
    # scores[masked] = -inf，这样 softmax 后变为 0
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    return scores
```

{IMAGE:8}

### 2.3 掩码后的 Softmax

{IMAGE:9}

掩码后的 softmax 计算：

$$\text{Attention}_{masked} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$$

其中 $M$ 是掩码矩阵。

```python
# 完整注意力计算流程
def compute_attention(self, q, k, v):
    # 1. 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(self.head_dim)
    
    # 2. 应用因果掩码
    seq_len = q.size(-2)
    scores = self._apply_causal_mask(scores, seq_len)
    
    # 3. Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    attn_output = torch.matmul(attn_weights, v)
    
    return attn_output
```

{WARNING}易错点{/WARNING}

- **掩码位置错误**：确保下三角为 0（可见），上三角为 $-\infty$（不可见）
- **数值稳定性**：softmax 前将 $-\infty$ 加入分数，而不是之后
- **对角线**：对角线位置（自己 attend 自己是允许的）应设为 0

---

## 第三部分：输出投影层（Output Projection）

### 3.1 输出投影的作用

{IMAGE:10}

注意力机制的输出需要通过线性投影变回原始维度：

$$O = \text{Attention}(Q, K, V) \cdot W_O$$

其中 $W_O \in \mathbb{R}^{(N_{kv} \times d_{head}) \times d_{model}}$

{IMAGE:11}

### 3.2 完整的 GQA 实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(num_kv_heads * self.head_dim, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 转置以便计算
        q = q.transpose(1, 2)  # [B, num_heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 扩展查询头以匹配键值头
        # 每个 KV 头对应多个 Q 头
        q = q.reshape(batch_size, self.num_kv_heads, 
                      self.num_queries_per_kv, seq_len, self.head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 应用因果掩码
        scores = self._apply_causal_mask(scores, seq_len)
        
        # 外部掩码（如果提供）
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        # [B, num_kv_heads, num_queries_per_kv, seq, head_dim] 
        # -> [B, seq, num_kv_heads * head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, 
                                          self.num_kv_heads * self.head_dim)
        
        # 最终输出投影
        output = self.W_o(attn_output)
        
        return output
```

{IMAGE:12}

---

## 第四部分：GQA vs MHA vs MQA 对比

| 特性 | MHA | MQA | GQA |
|------|-----|-----|-----|
| Query 头数 | N | N | N |
| KV 头数 | N | 1 | G (1 < G < N) |
| 计算量 | 高 | 最低 | 中等 |
| 显存占用 | 高 | 最低 | 中等 |
| 模型质量 | 最高 | 较低 | 接近 MHA |

{IMPORTANT}GQA 通过设置 $G$ 个键值头，在保持接近 MHA 性能的同时，大幅降低计算和显存开销。{/IMPORTANT}

---

## 本节总结

1. **注意力计算**：GQA 通过分数扩展机制，让多个查询头共享较少的键值头
2. **因果掩码**：使用下三角掩码确保语言模型的因果性，防止信息泄露
3. **输出投影**：将多头注意力的输出投影回原始维度 $d_{model}$

---

## 关键要点（Key Takeaways）

{IMPORTANT}
- GQA 的核心是在 $N_q$ 个查询头和 $N_{kv}$ 个键值头之间建立分组关系
- 因果掩码通过将未来位置设为 $-\infty$ 来实现 autoregressive 生成
- 输出投影层 $W_O$ 将所有注意力头的输出合并回原始维度
- 实现时注意张量形状变换的正确性
{/IMPORTANT}

---

## 思考题

1. **思考**：如果设置 `num_kv_heads = num_heads`（即 GQA 退化为 MHA），代码需要做哪些修改？

2. **思考**：在推理时使用 KV Cache，GQA 如何从减少的键值头数量中获益？

---

*下一节我们将学习 Transformer 层的完整实现，包括前馈神经网络（FFN）和残差连接。*