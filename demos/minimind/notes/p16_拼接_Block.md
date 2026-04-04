# 第16集: 拼接：Block

# 第十六讲：TransformerBlock 组件详解

**课程**：MiniMind - PyTorch从零手敲大模型  
**时长**：5分47秒  
**本讲重点**：掌握 TransformerBlock 的完整结构，理解各组件之间的连接关系

---

## 一、课程导入

{IMAGE:1}

在上一讲中，我们已经完成了 GPT 模型的宏观架构设计。本讲我们将深入到 **TransformerBlock（Transformer 块）** 的内部实现，这是 GPT 模型的核心组成单元。整个 GPT 模型正是由 N 个 TransformerBlock 堆叠而成，因此深入理解单个 Block 的构造至关重要。

{KNOWLEDGE}Transformer 架构最初由 Vaswani 等人在 2017 年的论文《Attention Is All You Need》中提出，是现代大语言模型的基础。本讲内容将聚焦于 GPT 风格（仅使用解码器部分）的 TransformerBlock 实现。{/KNOWLEDGE}

---

## 二、TransformerBlock 核心组件概述

{IMAGE:2}

一个标准的 TransformerBlock 主要由以下几部分组成：

1. **多头自注意力层（Multi-Head Self-Attention）**
2. **残差连接（Residual Connection）**
3. **层归一化（Layer Normalization）**
4. **前馈神经网络（Feed-Forward Network）**
5. **再次残差连接与层归一化**

{IMPORTANT}核心概念：TransformerBlock 采用" sandwich "结构，即 Attention + Add & Norm + FFN + Add & Norm 的层层堆叠，这种设计已成为现代 Transformer 模型的标配。{/IMPORTANT}

---

## 三、多头自注意力层详解

{IMAGE:3}

{IMAGE:4}

### 3.1 自注意力机制原理

自注意力机制允许输入序列中的每个位置关注序列中的所有其他位置，其数学表达式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$（Query）：查询向量，表示当前位置"想要查找什么"
- $K$（Key）：键向量，表示每个位置"包含什么信息"
- $V$（Value）：值向量，表示实际要传递的内容
- $\sqrt{d_k}$：缩放因子，防止点积值过大导致梯度消失

### 3.2 多头注意力实现

{IMAGE:5}

多头注意力将注意力机制并行运行多次，每次使用不同的 Query、Key、Value 投影：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

其中每个 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        # 将 embedding 分成多个头
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, d_k)
```

---

## 四、残差连接与层归一化

{IMAGE:6}

{IMAGE:7}

### 4.1 残差连接（Residual Connection）

残差连接，又称 Skip Connection，最早出现在 ResNet 中。其核心思想是**让网络学习输入与输出之间的残差**，而不是直接学习完整的映射：

$$\text{Output} = \mathcal{F}(x) + x$$

{IMPORTANT}核心概念：残差连接使得梯度能够直接反向传播到较浅的层，有效缓解了深层网络的梯度消失问题，使得构建深层 Transformer 成为可能。{/IMPORTANT}

### 4.2 层归一化（Layer Normalization）

层归一化对每一层的输出进行归一化处理：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu$ 和 $\sigma$ 是该层所有样本的均值和标准差，$\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数。

{KNOWLEDGE}与 Batch Normalization 不同，Layer Normalization 不依赖 batch 维度，在序列建模任务中更为稳定。GPT 统一使用 Pre-LayerNorm（即在注意力/FFN之前进行归一化），研究表明这种设计训练更稳定。{/KNOWLEDGE}

```python
# Pre-LayerNorm 结构的 Add & Norm
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
```

---

## 五、前馈神经网络

{IMAGE:8}

{IMAGE:9}

### 5.1 FFN 结构

前馈神经网络位于每个 TransformerBlock 的后半部分，其标准配置为两层全连接网络，中间带有激活函数：

$$\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2$$

其中 $\sigma$ 通常为 GELU 激活函数（GPT-2 及之后版本常用）。

### 5.2 FFN 的作用

{IMPORTANT}核心概念：FFN 层为每个位置提供非线性变换能力，是 Transformer 学习复杂模式的关键组件。尽管 FFN 的参数量占整个模型的大部分，但它独立处理每个位置，不捕捉序列中的依赖关系。{/IMPORTANT}

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # 扩展维度
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

---

## 六、完整 TransformerBlock 组装

{IMAGE:10}

{IMAGE:11}

{IMAGE:12}

现在我们将所有组件组装成完整的 TransformerBlock：

```python
class TransformerBlock(nn.Module):
    """
    单个 Transformer Block
    
    采用 Pre-LayerNorm 结构：
    x -> LayerNorm -> Attention -> Add -> LayerNorm -> FFN -> Add -> output
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        # 多头自注意力层
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前馈神经网络
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LayerNorm: 先归一化再做注意力
        attn_input = self.norm1(x)
        # 自注意力计算
        attn_output = self.attention(attn_input, mask=mask)
        # 残差连接
        x = x + self.dropout(attn_output)
        
        # Pre-LayerNorm: 先归一化再做 FFN
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        # 残差连接
        x = x + self.dropout(ffn_output)
        
        return x
```

{WARNING}易错点：在实现时容易忽略残差连接的 Dropout，但这在训练时是重要的正则化手段。同时确保 mask 被正确传递到注意力层，否则模型会看到不该看到的位置。{/WARNING}

---

## 七、本讲小结

本讲我们详细介绍了 TransformerBlock 的完整结构，包括：

| 组件 | 作用 | 位置 |
|------|------|------|
| Multi-Head Attention | 捕捉序列内长距离依赖 | 第一层 |
| Add & Norm (1) | 残差连接 + 归一化 | 注意力后 |
| Feed-Forward Network | 非线性变换 | 第二层 |
| Add & Norm (2) | 残差连接 + 归一化 | FFN后 |

{IMAGE:12} 完整的 TransformerBlock 遵循 **Pre-LayerNorm** 设计范式，这种结构已被 GPT、BERT、T5 等主流模型广泛采用。

---

## 关键要点（Key Takeaways）

1. **TransformerBlock 是 GPT 模型的基本计算单元**，N 个 Block 堆叠构成完整的 GPT 架构

2. **残差连接是深层网络的关键**，确保梯度有效流动，支持 12-96 层甚至更深的网络

3. **Pre-LayerNorm 比 Post-LayerNorm 更稳定**，已成为现代 Transformer 的默认选择

4. **FFN 占模型参数的主体**（约 2/3），但独立处理每个位置，不引入序列依赖

5. **代码实现要关注组件间的连接顺序**，确保 mask 正确传递到注意力层

---

## 思考题

1. **为什么 Transformer 中需要 FFN 层？能否只用注意力机制完成所有计算？**

2. **如果将残差连接移除，深层 Transformer 会面临什么问题？请从梯度和信息流动的角度分析。**