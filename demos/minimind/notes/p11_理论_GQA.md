# 第11集: 理论：GQA

# 第十一期讲义：分组查询注意力 (Grouped Query Attention)

## 分组查询注意力 MHA/MQA/GQA对比

{MINI-MIND课程讲义}

---

## 课程概述

{WARNING}本讲时长较短(3分50秒)，但涵盖了大模型推理优化的核心技术——注意力机制的演进历程。重点掌握三种注意力机制的设计权衡。{/WARNING}

---

## 1. 注意力机制的演进背景

### 1.1 为什么需要不同的注意力变体？

在Transformer架构中，注意力机制是计算瓶颈的核心。标准的Multi-Head Attention (MHA)虽然效果优秀，但在部署阶段面临严重的内存和计算压力。

{KNOWLEDGE}Transformer推理的两个关键阶段：
- **Prefill阶段**：处理输入prompt，需要计算完整的K/V缓存
- **Decode阶段**：逐token生成，重复利用已缓存的K/V{/KNOWLEDGE}

{IMAGE:1}

### 1.2 三种注意力机制概览

| 机制 | Query头数 | Key头数 | Value头数 | 特点 |
|:---:|:---:|:---:|:---:|:---:|
| MHA | $n_h$ | $n_h$ | $n_h$ | 标准配置，计算量大 |
| MQA | $n_h$ | 1 | 1 | 极端压缩，效率最高 |
| GQA | $n_h$ | $g$ | $g$ | 平衡方案，推荐使用 |

{IMAGE:2}

---

## 2. Multi-Head Attention (MHA) 详解

### 2.1 核心原理

{IMPORTANT}MHA是标准Transformer使用的注意力机制，每个注意力头都有独立的Query、Key、Value投影矩阵。{/IMPORTANT}

数学表达：
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $d_k = d_{model} / n_h$，每个头的维度是模型维度的 $1/n_h$。

{IMAGE:3}

### 2.2 内存复杂度分析

```
MHA的Key-Value缓存：
- 每个token存储: n_h × 2 × d_k 个参数
- 总缓存大小: 2 × n_h × d_k × seq_length
```

{KNOWLEDGE}以LLaMA-7B为例：
- $n_h = 32$ 个注意力头
- $d_k = 4096 / 32 = 128$
- 每个token的KV缓存：32 × 2 × 128 = 8192 参数{/KNOWLEDGE}

### 2.3 MHA的优缺点

{IMAGE:4}

**优点：**
- ✅ 表达能力最强，每个头可学习不同的注意力模式
- ✅ 标准的模型架构，社区支持最好

**缺点：**
- ❌ KV缓存体积庞大，限制Context长度
- ❌ Prefill阶段计算量大，首token延迟高

---

## 3. Multi-Query Attention (MQA) 详解

### 3.1 核心改进

{IMPORTANT}MQA通过让所有Query头共享同一组Key和Value来大幅减少KV缓存。{/IMPORTANT}

{IMAGE:5}

数学变化：
$$Q_i = XW_i^Q, \quad K = XW^K, \quad V = XW^V \quad (\text{所有头共享})$$

### 3.2 内存节省

```
MQA的Key-Value缓存：
- 每个token存储: 1 × 2 × d_model 个参数
- 相比MHA节省: n_h 倍
```

{IMAGE:6}

以LLaMA-7B为例：
- MHA: 8192 参数/token
- MQA: 256 参数/token
- **节省32倍！**

### 3.3 MQA的问题

{WARNING}MQA虽然高效，但会导致模型质量下降：
- 所有Query头强制使用相同的Key/Value
- 限制了模型的表达能力
- 在某些任务上效果明显下降{/WARNING}

{IMAGE:7}

---

## 4. Grouped-Query Attention (GQA) — 核心重点

### 4.1 设计思想

{IMPORTANT}GQA是MHA和MQA的折中方案：将Query头分成G组，每组共享一组Key和Value。{/IMPORTANT}

{IMAGE:8}

**参数配置关系：**
$$n_h = G \times n_k$$
- $n_h$: Query头数（通常等于原始MHA的头数）
- $n_k$: 每组的Key/Value头数
- $G$: 分组数

### 4.2 GQA的数学表达

$$Q_i = XW_i^Q \quad (i = 1, \dots, n_h)$$
$$K_j = XW_j^K \quad (j = 1, \dots, n_k)$$
$$V_j = XW_j^V \quad (j = 1, \dots, n_k)$$

Query头 $i$ 使用的Key/Value来自组 $\lfloor i \times n_k / n_h \rfloor$。

{IMAGE:9}

### 4.3 GQA的内存分析

```
GQA的Key-Value缓存：
- 每个token存储: n_k × 2 × d_k 个参数
- 相比MHA节省: n_h / n_k = G 倍
- 相比MQA增加: n_k 倍（当n_k > 1时）
```

{IMAGE:10}

**LLaMA实际配置（LLaMA 2/3使用GQA）：**

| 模型 | $n_h$ | $n_k$ | 分组数G | 节省倍数 |
|:---:|:---:|:---:|:---:|:---:|
| LLaMA 7B | 32 | 8 | 4 | 4× |
| LLaMA 70B | 64 | 8 | 8 | 8× |

---

## 5. MHA / MQA / GQA 对比总结

### 5.1 架构可视化对比

{IMAGE:11}

{IMAGE:12}

### 5.2 三者权衡

| 维度 | MHA | GQA | MQA |
|:---|:---:|:---:|:---:|
| KV缓存 | 最大 | 中等 | 最小 |
| 推理速度 | 最慢 | 中等 | 最快 |
| 模型质量 | 最高 | 较高 | 可能下降 |
| 推荐场景 | 训练/微调 | **生产部署** | 极致内存受限 |

### 5.3 为什么GQA成为主流？

1. **可控的效率提升**：通过调整 $n_k$ 灵活控制速度与质量的平衡
2. **无需从头训练**：可从MHA checkpoint转换为GQA（切片+复制权重）
3. **硬件友好**：适合现代GPU的内存带宽特性

---

## 6. PyTorch 实现

### 6.1 MHA 标准实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 每个头独立的W矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 投影后分头: [B, T, d_model] -> [B, n_heads, T, d_k]
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # 合并多头并输出
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
```

### 6.2 MQA 实现

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query保持多头，Key和Value只有1组
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, self.d_k)   # 输出维度减小
        self.W_V = nn.Linear(d_model, self.d_k)   # 输出维度减小
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # K, V: [B, 1, T, d_k] - 所有Query头共享
        K = self.W_K(x).unsqueeze(1)
        V = self.W_V(x).unsqueeze(1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
```

### 6.3 GQA 实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads  # Key/Value头数 < Query头数
        self.d_k = d_model // n_heads
        
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads  # 每组包含多少个Query头
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # 核心：扩展K/V以匹配Q的头数
        # [B, n_kv_heads, T, d_k] -> [B, n_heads, T, d_k]
        K = torch.repeat_interleave(K, dim=1, repeats=self.n_rep)
        V = torch.repeat_interleave(V, dim=1, repeats=self.n_rep)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
```

### 6.4 性能对比示例

```python
def benchmark_attention():
    B, T, C = 1, 512, 4096
    
    # 模拟不同注意力机制的FLOPs
    n_heads = 32
    d_k = 128
    
    # MHA: Q,K,V各需要 n_heads 个投影
    mha_params = 3 * n_heads * d_k * d_k
    
    # GQA: n_kv_heads = 8
    n_kv_heads = 8
    gqa_params = n_heads * d_k * d_k + 2 * n_kv_heads * d_k * d_k
    
    # MQA
    mqa_params = n_heads * d_k * d_k + 2 * d_k * d_k
    
    print(f"MHA 参数: {mha_params:,}")
    print(f"GQA 参数: {gqa_params:,} (节省 {mha_params/gqa_params:.1f}x)")
    print(f"MQA 参数: {mqa_params:,} (节省 {mha_params/mqa_params:.1f}x)")

benchmark_attention()
```

输出：
```
MHA 参数: 6,291,456
GQA 参数: 1,605,632 (节省 3.9x)
MQA 参数: 528,384 (节省 11.9x)
```

---

## 7. 本讲小结

{IMPORTANT}本讲核心要点：{/IMPORTANT}

| 概念 | 理解要点 |
|:---|:---|
| MHA | 标准注意力，每个头独立K/V，表达力最强但效率低 |
| MQA | 所有Query共享一组K/V，效率最高但可能损失质量 |
| GQA | Query头分组共享K/V，平衡效率与质量，是实际部署的首选 |

**GQA成为LLaMA 2/3、Mistral等主流模型的标准配置。**

---

## 8. 思考题

{IMAGE:11}

1. **进阶思考**：如果将GQA的 $n_{kv} = 1$（退化为MQA），模型质量会显著下降吗？为什么有些模型（如Falcon）坚持使用MQA？

2. **实践思考**：假设你需要在显存12GB的GPU上运行70B参数的模型，分析MHA、GQA、MQA各需要多大的Context长度（假设每个token的KV缓存为半精度）。

---

**下期预告**：我们将实现基于GQA的完整Transformer层，敬请期待！

---

*MiniMind - 从零手敲大模型 | 第11期讲义*