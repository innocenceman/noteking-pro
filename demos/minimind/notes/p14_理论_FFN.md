# 第14集: 理论：FFN

# 第十四讲：理论 · 前馈网络（FFN）与 SwiGLU 激活函数

**课程**：MiniMind - PyTorch从零手敲大模型  
**课时**：第14/26讲 · 时长 6分16秒  
**关键词**：前馈神经网络、SwiGLU、Swish、GELU、门控线性单元

---

## 1. 课程概述与学习目标

{IMAGE:1}

本讲将深入剖析 **Transformer 架构中的前馈神经网络（Feed-Forward Network, FFN）** 模块，这是现代大语言模型的核心组件之一。我们将重点学习：

- FFN 在 Transformer 中的定位与作用
- 从传统 GELU 激活到 SwiGLU 的演进历程
- SwiGLU 激活函数的数学原理与实现
- PyTorch 代码层面的完整实现

{KNOWLEDGE}学习背景{/KNOWLEDGE}
本讲承接前几讲对 Transformer 架构的整体介绍，要求读者具备：
- 基本的矩阵运算与线性代数知识
- Python/PyTorch 编程基础
- 对深度学习中"激活函数"概念的初步理解

---

## 2. FFN 在 Transformer 中的角色

{IMAGE:2}

### 2.1 Transformer 架构回顾

在标准的 Transformer 编码器或解码器层中，每个注意力子层之后都会接一个 **前馈网络子层**。这一设计最早出现在 Google 的经典论文 *"Attention Is All You Need"* 中。

{WARNING}常见误区{/WARNING}
许多初学者认为注意力机制（Attention）是最重要的部分，而将 FFN 视为"配角"。实际上，在 Transformer 的参数分布中，**FFN 通常占据了整个模型约 2/3 的参数**，是名副其实的"参数量担当"。

### 2.2 FFN 的数学定义

标准 FFN 的计算可以表示为：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

其中：
- $x \in \mathbb{R}^{d_{\text{model}}}$ 是输入向量
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ 是权重矩阵
- $d_{\text{ff}}$ 是 FFN 隐藏层的维度，通常设为 $d_{\text{model}} \times 4$

{IMAGE:3}

### 2.3 FFN 的结构图示

{IMAGE:4}

如图所示，经典 FFN 由两个线性层组成，中间夹着一个非线性激活函数（传统上使用 ReLU 或 GELU）。这种"先扩展后压缩"的设计让模型能够在更高维度的空间中进行非线性变换。

{IMPORTANT}核心概念{/IMPORTANT}
FFN 的本质是一个两层感知机（MLP），但它与注意力机制是**并行工作**的——每个位置的 token 独立地通过相同的 FFN 进行变换，这也是 FFN 区别于卷积层和循环层的重要特征。

---

## 3. 从 ReLU 到 GELU：激活函数的演进

{IMAGE:5}

### 3.1 ReLU 的局限性

早期 Transformer（如原始的 BERT、GPT-2）使用 ReLU 作为 FFN 的激活函数：

$$\text{ReLU}(x) = \max(0, x)$$

ReLU 的问题在于：
- **梯度稀疏**：负半轴梯度为 0，可能导致"死神经元"
- **无负值输出**：无法处理负相关的特征
- **非平滑**：在 0 处不可导

### 3.2 GELU：高斯误差线性单元

{IMAGE:6}

2016年，Hendrycks 和 Gimpel 提出了 **GELU（Gaussian Error Linear Unit）**，它结合了 ReLU 的线性特性和 Dropout 的随机正则化：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）。

{IMAGE:7}

GELU 可以近似为：

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

{KNOWLEDGE}为什么 GELU 更优？{/KNOWLEDGE}
- **自适应门控**：输入乘以 0~1 之间的概率权重，实现软性的"开/关"
- **平滑连续**：处处可导，便于梯度流动
- **与 Dropout 兼容**：两者的随机性有协同作用

BERT、GPT-2、RoBERTa 等主流模型都采用了 GELU 激活。

---

## 4. SwiGLU 激活函数详解

{IMAGE:8}

### 4.1 Swish 激活函数

Swish 由 Prajit Ramachandran 等人在 2017 年提出：

$$\text{Swish}(x) = x \cdot \sigma(x)$$

其中 $\sigma(x)$ 是 Sigmoid 函数。

{IMAGE:9}

Swish 的特点：
- **自门控**：输入本身作为门控信号
- **负值保留**：可以输出负值，允许特征抑制
- **可学习**：门控权重随输入动态调整

### 4.2 GLU：门控线性单元

{IMAGE:10}

**GLU（Gated Linear Unit）** 源自 2017 年的论文 *"Language Modeling with Gated Convolutional Networks"*，其核心思想是引入一个"门"来控制信息流动：

$$\text{GLU}(x) = \sigma(xW + b) \otimes (xV + c)$$

其中：
- $\otimes$ 表示逐元素乘法（Element-wise Product）
- $\sigma$ 是 Sigmoid 门控函数
- $W, V$ 是独立的线性变换矩阵

{IMPORTANT}核心概念{/IMPORTANT}
GLU 的关键在于：门控信号 $\sigma(xW + b)$ 控制了多少信息可以通过 $(xV + c)$ 传递。这类似于 LSTM 中的遗忘门和输入门，但实现更加简洁高效。

### 4.3 SwiGLU：Swish 与 GLU 的融合

{IMAGE:11}

**SwiGLU** 将 Swish 作为 GLU 的门控函数：

$$\text{SwiGLU}(x) = \text{Swish}_\beta(xW_1) \otimes (xW_2)$$

其中：

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

当 $\beta = 1$ 时，就是标准的 Swish。

标准 SwiGLU 实现（Shengding Hu 等人在 LLaMA 中使用）：

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2 + b_2)$$

### 4.4 SwiGLU 的优势

{IMAGE:12}

| 特性 | ReLU | GELU | SwiGLU |
|------|------|------|--------|
| 平滑性 | ❌ | ✅ | ✅ |
| 可学习门控 | ❌ | ❌ | ✅ |
| 负值处理 | 截断 | 保留 | 保留 |
| 计算复杂度 | 低 | 中 | 中高 |
| 主流模型采用 | 早期 | BERT, GPT-2 | LLaMA, LLaMA-2, Vicuna |

{KNOWLEDGE}为什么 SwiGLU 表现更好？{/KNOWLEDGE}
1. **双重非线性**：Swish 本身是非线性的，再加上逐元素乘法的交互作用
2. **动态路由**：门控机制让模型自适应决定保留或抑制每个特征
3. **参数效率**：三个线性层（W1, W2, W3）比单一的大 FFN 更高效

---

## 5. PyTorch 实现

### 5.1 标准 FFN 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    """
    标准 FFN 实现（ReLU/GELU）
    对应原始 Transformer 架构
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 第一个线性层：扩展维度
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        # 第二个线性层：压缩回原维度
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        Returns:
            [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        """
        # 扩展 -> 激活 -> Dropout -> 压缩
        return self.w2(self.dropout(F.gelu(self.w1(x))))
```

### 5.2 SwiGLU FFN 实现

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN 实现
    
    SwiGLU(x) = Swish(xW1) ⊗ (xW2)
    
    相比标准 FFN，多了一个 W3 用于门控计算，
    但可以在相同参数量下获得更好性能
    
    参考文献：Shengding Hu et al., "LLaMA: Open and Efficient Foundation Language Models"
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # SwiGLU 需要三个线性层
        # W1, W2: 标准 FFN 的两层
        # W3: 额外的门控层（输出维度与 W1 相同）
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Swish 门控路径
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # 主信号路径
        self.w3 = nn.Linear(d_ff, d_model, bias=True)   # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的 beta 参数（Swishβ = x * σ(βx)）
        # 设为可训练参数，初始化为 1.0
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def swish(self, x: torch.Tensor) -> torch.Tensor:
        """Swish 激活函数：x * sigmoid(x)"""
        return x * torch.sigmoid(self.beta * x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(xW1) ⊗ (xW2) @ W3
        return self.dropout(self.w3(self.swish(self.w1(x)) * self.w2(x)))
```

### 5.3 精简版 SwiGLU（推荐实现）

```python
class SwiGLUFFNV2(nn.Module):
    """
    精简版 SwiGLU FFN
    
    这是 LLaMA、Mistral 等模型实际使用的版本
    特点：使用 SiLU（Swish-1 的别名）作为激活函数
    """
    def __init__(self, d_model: int, intermediate_size: int = None, dropout: float = 0.0):
        super().__init__()
        # LLaMA 的标准配置：intermediate_size = d_model * 4 / 3
        # 然后向上取整到 128 的倍数
        if intermediate_size is None:
            intermediate_size = int((d_model * 4 + 2) / 3)
            intermediate_size = ((intermediate_size + 127) // 128) * 128
        
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)  # W1
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)    # W2
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)   # W3
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # SiLU = Swish = x * sigmoid(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x) @ down_proj
        """
        return self.down_proj(
            self.dropout(
                self.activation(self.gate_proj(x)) * self.up_proj(x)
            )
        )
```

### 5.4 三种 FFN 的对比测试

```python
def test_ffn_comparison():
    """测试三种 FFN 实现的输出维度一致性"""
    batch_size, seq_len, d_model = 2, 10, 512
    d_ff = 2048  # intermediate_size
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化三种 FFN
    ffn_standard = StandardFFN(d_model, d_ff)
    ffn_swiglu = SwiGLUFFN(d_model, d_ff)
    ffn_swiglu_v2 = SwiGLUFFNV2(d_model, d_ff)
    
    # 前向传播
    out1 = ffn_standard(x)
    out2 = ffn_swiglu(x)
    out3 = ffn_swiglu_v2(x)
    
    print(f"输入形状: {x.shape}")
    print(f"Standard FFN 输出: {out1.shape}")
    print(f"SwiGLU FFN 输出: {out2.shape}")
    print(f"SwiGLU V2 输出: {out3.shape}")
    
    # 参数数量对比
    print("\n参数量对比:")
    print(f"Standard FFN: {sum(p.numel() for p in ffn_standard.parameters()):,}")
    print(f"SwiGLU FFN: {sum(p.numel() for p in ffn_swiglu.parameters()):,}")
    print(f"SwiGLU V2: {sum(p.numel() for p in ffn_swiglu_v2.parameters()):,}")

# test_ffn_comparison()
```

---

## 6. 参数效率分析

$$参数总量 = 2 \times (d_{model} \times d_{ff}) + d_{model} + d_{ff} + d_{ff} + d_{model}$$

对于 $d_{model} = 512$, $d_{ff} = 2048$：

| FFN 类型 | 参数量 | 相对比例 |
|---------|--------|----------|
| Standard | ~2.1M | 100% |
| SwiGLU | ~2.1M | ~100% |

{KNOWLEDGE}关键发现{/KNOWLEDGE}
虽然 SwiGLU 多了 W3 层，但通过调整隐藏层维度 $d_{ff}$（如 LLaMA 的 $\frac{4}{3}d_{model}$），可以在**相同参数量**下获得更好的性能。

---

## 7. 本讲小结

{IMAGE:12}

| 知识点 | 要点回顾 |
|--------|----------|
| FFN 定位 | Transformer 中每个注意力层后的 MLP，占模型 2/3 参数 |
| GELU 激活 | $x \cdot \Phi(x)$，平滑非饱和，适合大模型 |
| Swish 激活 | $x \cdot \sigma(x)$，自门控可学习 |
| GLU 机制 | 门控信号控制信息流动，类似 LSTM 但更简洁 |
| SwiGLU | $\text{SiLU}(xW_1) \otimes (xW_2) @ W_3$，现代大模型标配 |

{IMPORTANT}核心公式{/IMPORTANT}
$$\text{SwiGLU}(x) = \underbrace{\text{SiLU}(xW_1)}_{\text{门控}} \otimes \underbrace{(xW_2)}_{\text{信号}} \cdot W_3$$

---

## 8. 关键要点与思考题

### 关键要点

1. **FFN 是 Transformer 的参数量担当**：理解 FFN 的设计对优化模型至关重要

2. **SwiGLU 优于传统激活**：自门控机制让模型能动态控制每个特征的信息流动

3. **PyTorch 实现简洁**：使用 `nn.SiLU()` + 逐元素乘法即可实现 SwiGLU

4. **工程实践**：LLaMA、Mistral、Qwen 等主流开源模型均采用 SwiGLU

### 思考题

**思考题 1**：
> SwiGLU 中的门控机制与 LSTM/GRU 中的门控有何异同？为什么说 SwiGLU 更适合 Transformer 架构？

**提示**：考虑并行计算效率、信息流动方向、以及与自注意力机制的协同。

**思考题 2**：
> 如果将 SwiGLU 中的 SiLU 替换为 ReLU 或 GELU，模型性能可能会如何变化？请从门控机制的角度分析。

**提示**：考虑门控信号的值域范围对信息流动的影响。

---

*下一讲预告*：我们将实现完整的 Transformer Encoder 层，将 FFN 与 Multi-Head Attention 进行整合。