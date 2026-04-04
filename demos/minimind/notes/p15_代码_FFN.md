# 第15集: 代码：FFN

# 课程讲义：代码实现 - FeedForward 前馈网络

{IMAGE:1}

---

## 课程概述

本节课我们将深入剖析 Transformer 架构中 **FeedForward Network（FFN）** 的 PyTorch 实现。FFN 是 Transformer 编码器和解码器中每个注意力层之后的关键组件，尽管结构看似简单，却在模型的表现能力中扮演着不可或缺的角色。

{IMPORTANT}核心概念{/IMPORTANT}
- FeedForward Network = 前馈神经网络
- 在 Transformer 中位于 Multi-Head Attention 之后
- 本质是一个两层的全连接网络，中间层维度通常会扩展

{KNOWLEDGE}背景知识{/KNOWLEDGE}
FFN 首次在《Attention Is All You Need》论文中被提出，其数学形式为：
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

这等价于两个线性变换之间夹着一个 ReLU 激活函数。

---

## 1. FFN 在 Transformer 中的位置

{IMAGE:2}

让我们首先明确 FFN 在整体架构中的位置：

```
┌─────────────────────────────────────────────────────┐
│                    Transformer Layer                 │
│  ┌─────────────────────────────────────────────┐    │
│  │           Multi-Head Self-Attention          │    │
│  └─────────────────────────────────────────────┘    │
│                        ↓                             │
│  ┌─────────────────────────────────────────────┐    │
│  │         Add & LayerNorm                      │    │
│  └─────────────────────────────────────────────┘    │
│                        ↓                             │
│  ┌─────────────────────────────────────────────┐    │
│  │           FeedForward Network               │    │
│  └─────────────────────────────────────────────┘    │
│                        ↓                             │
│  ┌─────────────────────────────────────────────┐    │
│  │         Add & LayerNorm                      │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

{IMAGE:3}

---

## 2. FeedForward 网络结构详解

{IMAGE:4}

FFN 的核心结构可以用下面的公式描述：

$$\text{FFN}(x) = \text{Dropout}(\text{GELU}(x \cdot W_1 + b_1)) \cdot W_2 + b_2$$

### 2.1 网络结构图示

{IMAGE:5}

```
输入 x (batch, seq_len, d_model)
    │
    │  Linear(d_model → d_ff)
    ▼
中间表示 (batch, seq_len, d_ff)
    │
    │  GELU 激活函数
    ▼
激活后 (batch, seq_len, d_ff)
    │
    │  Dropout (可选)
    ▼
    │
    │  Linear(d_ff → d_model)
    ▼
输出 (batch, seq_len, d_model)
```

### 2.2 维度扩展机制

{IMAGE:6}

{KNOWLEDGE}关键参数说明{/KNOWLEDGE}
- **d_model**: 输入/输出的隐藏层维度（通常为 512、768、1024 等）
- **d_ff**: FFN 中间层维度（通常为 d_model 的 2-4 倍）
- **expand_ratio**: 扩展比率，默认通常为 4

$$
d_{ff} = d_{model} \times \text{expand\_ratio}
$$

例如，当 d_model = 768，expand_ratio = 4 时：
$$d_{ff} = 768 \times 4 = 3072$$

{IMAGE:7}

---

## 3. PyTorch 代码实现

### 3.1 完整代码展示

{IMAGE:8}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    FeedForward Network for Transformer
    实现位置：每个 Transformer Block 的后半部分
    
    结构：Linear -> Activation -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化 FFN 模块
        
        Args:
            d_model: 输入输出的隐藏层维度
            d_ff: FFN 中间层的维度（通常为 d_model 的 2-4 倍）
            dropout: Dropout 概率，用于正则化
        """
        super().__init__()
        
        # 第一层线性变换：扩展维度
        self.w1 = nn.Linear(d_model, d_ff)
        
        # 第二层线性变换：压缩回原维度
        self.w2 = nn.Linear(d_ff, d_model)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数（可在初始化时选择）
        # 常见的激活函数选择：
        # - GELU (GPT, BERT 等常用)
        # - ReLU (经典 Transformer)
        # - SwiGLU (LLaMA 等新型模型)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # x.shape = (batch, seq_len, d_model)
        x = self.w1(x)           # (batch, seq_len, d_ff)
        x = F.gelu(x)            # 激活函数
        x = self.dropout(x)      # Dropout
        x = self.w2(x)           # (batch, seq_len, d_model)
        
        return x
```

{IMAGE:9}

### 3.2 代码逐行解析

{IMAGE:10}

#### 3.2.1 `__init__` 方法详解

```python
def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    super().__init__()
    
    # 第一层：从 d_model 扩展到 d_ff
    self.w1 = nn.Linear(d_model, d_ff)
    
    # 第二层：从 d_ff 压缩回 d_model
    self.w2 = nn.Linear(d_ff, d_model)
    
    # Dropout 层用于正则化
    self.dropout = nn.Dropout(dropout)
```

{WARNING}易错点{/WARNING}
1. **维度顺序**：PyTorch 的 `nn.Linear` 期望输入形状为 `(..., in_features)`，最后一维是特征维度
2. **偏置项**：默认 `nn.Linear` 包含偏置，如果不需要可以设置 `bias=False`
3. **d_ff 选择**：d_ff 太小会导致模型容量不足，太大会增加计算量

#### 3.2.2 `forward` 方法详解

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: 维度扩展
    x = self.w1(x)  # (B, S, D) -> (B, S, D_ff)
    
    # Step 2: 非线性激活
    x = F.gelu(x)   # Gaussian Error Linear Unit
    
    # Step 3: 正则化
    x = self.dropout(x)
    
    # Step 4: 维度压缩回原大小
    x = self.w2(x)  # (B, S, D_ff) -> (B, S, D)
    
    return x
```

{IMAGE:11}

---

## 4. 激活函数的选择

### 4.1 常见激活函数对比

{IMAGE:12}

| 激活函数 | 公式 | 使用模型 | 特点 |
|---------|------|---------|------|
| **ReLU** | $\max(0, x)$ | 原始 Transformer | 简单高效 |
| **GELU** | $0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ | GPT, BERT, LLaMA | 平滑近似 |
| **SwiGLU** | $\text{Swish}(x) \cdot \sigma(x)$ | LLaMA-2, PaLM | 性能更优 |

### 4.2 GELU 激活函数可视化

GELU（Gaussian Error Linear Unit）的数学表达式：

$$\text{GELU}(x) = x \cdot \Phi(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 100)

# ReLU
relu = np.maximum(0, x)

# GELU 近似
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# 绘制对比图...
```

---

## 5. 维度变换详解

假设输入张量的形状为：
$$x \in \mathbb{R}^{B \times S \times D_{model}}$$

其中：
- $B$ = batch_size（批量大小）
- $S$ = seq_len（序列长度）
- $D_{model}$ = 模型隐藏层维度

经过 FFN 的维度变换过程：

| 层级 | 操作 | 形状变化 |
|-----|------|---------|
| 1 | 输入 | $(B, S, D_{model})$ |
| 2 | $W_1$ 线性变换 | $(B, S, D_{ff})$ |
| 3 | GELU 激活 | $(B, S, D_{ff})$ |
| 4 | Dropout | $(B, S, D_{ff})$ |
| 5 | $W_2$ 线性变换 | $(B, S, D_{model})$ |
| 6 | 输出 | $(B, S, D_{model})$ |

---

## 6. 完整使用示例

```python
import torch

# 参数设置
batch_size = 2
seq_len = 10
d_model = 512
d_ff = 2048  # d_model * 4
dropout = 0.1

# 初始化 FFN
ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

# 模拟输入
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output = ffn(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"FFN 参数总量: {sum(p.numel() for p in ffn.parameters())}")

# 输出:
# 输入形状: torch.Size([2, 10, 512])
# 输出形状: torch.Size([2, 10, 512])
# FFN 参数总量: 1050624
```

---

## 7. 参数数量计算

FFN 的可学习参数主要来自两个线性层：

$$\text{参数总量} = \underbrace{D_{model} \times D_{ff}}_{\text{w1 权重}} + \underbrace{D_{ff}}_{\text{w1 偏置}} + \underbrace{D_{ff} \times D_{model}}_{\text{w2 权重}} + \underbrace{D_{model}}_{\text{w2 偏置}}$$

以 d_model=512, d_ff=2048 为例：
$$\text{参数总量} = 512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2,098,304$$

{WARNING}计算复杂度{/WARNING}
FFN 的时间复杂度为 $O(n \cdot d_{model} \cdot d_{ff})$，其中 $n$ 是序列长度。由于 $d_{ff}$ 通常是 $d_{model}$ 的 2-4 倍，FFN 占据了 Transformer 大约 2/3 的参数量。

---

## 本章小结

{IMPORTANT}本章核心要点{/IMPORTANT}

1. **FFN 位置**：位于 Transformer 每个 Block 的 Multi-Head Attention 之后
2. **网络结构**：两层全连接网络 + 激活函数 + Dropout
3. **维度变化**：先扩展（d_model → d_ff），再压缩（d_ff → d_model）
4. **激活函数**：GELU 是现代 LLM 的主流选择
5. **参数量**：FFN 占据 Transformer 大部分参数，是模型容量的主要来源

---

## 思考题

1. **维度扩展的意义**：为什么 FFN 需要先将维度扩展到 d_ff 再压缩回 d_model，而不是直接使用单层线性变换？请从模型容量和非线性表达能力角度分析。

2. **SwiGLU 变体**：查阅资料了解 SwiGLU 激活函数的实现原理，并尝试修改代码实现 SwiGLU 版本的 FFN，分析其相比 GELU 的优势。

---

**下节预告**：在下一节课中，我们将学习如何将 Multi-Head Attention 和 FeedForward 组合成一个完整的 Transformer Block，实现完整的自注意力机制层。

{IMAGE:10} {IMAGE:11} {IMAGE:12}