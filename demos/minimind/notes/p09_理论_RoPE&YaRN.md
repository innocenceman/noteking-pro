# 第9集: 理论：RoPE&YaRN

# 第九讲：旋转位置编码 RoPE 与外推技术 YaRN

## 讲义概述

本讲将深入探讨现代大语言模型中两项关键技术：**旋转位置编码（Rotary Position Embedding, RoPE）** 和 **YaRN（Yet another RoPE extensioN）**。我们将从头推导 RoPE 的数学原理，解释为何它能有效解决传统位置编码的问题，并详细讲解如何通过 YaRN 技术实现上下文窗口的扩展外推。

---

## 1. 位置编码的背景与动机

### 1.1 Transformer 中的位置问题

{KNOWLEDGE}背景知识{/KNOWLEDGE}

在原始的 Transformer 架构中，注意力机制（Self-Attention）具有**置换不变性**——这意味着输入序列 `[A, B, C]` 和 `[C, B, A]` 经过注意力计算后会产生相同的输出，因为模型无法区分 token 之间的相对位置关系。

{IMAGE:1}

为了解决这个问题，Vaswani 等人在《Attention Is All You Need》中提出了**绝对位置编码（Absolute Position Encoding, APE）**。

### 1.2 传统位置编码的局限性

{IMAGE:2}

原始 Transformer 使用正弦/余弦函数生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

{WARNING}易错点{/WARNING}

这种编码方式存在几个关键问题：

1. **长度外推性差**：训练时序列长度固定，推理时若超出训练长度，位置编码无法泛化
2. **计算效率低**：需要将位置编码与 token embedding 相加，增加参数量
3. **相对位置信息缺失**：绝对位置编码难以直接表达 token 之间的相对距离

{KNOWLEDGE}背景知识{/KNOWLEDGE}

后续研究提出了**相对位置编码（Relative Position Encoding, RPE）**，如 Shaw 等人的工作，通过引入相对位置偏移量来改善这一问题。然而，这些方法仍然存在计算复杂度和表达能力上的局限。

---

## 2. 旋转位置编码 RoPE

### 2.1 核心思想

{IMPORTANT}核心概念{/IMPORTANT}

**RoPE 的核心思想**：将位置信息编码为旋转矩阵，通过**旋转 query 和 key 向量**来实现位置感知，而不改变注意力分数的直接计算方式。

{IMAGE:3}

设 query 向量 $\mathbf{q}_m$ 和 key 向量 $\mathbf{k}_n$ 分别位于位置 $m$ 和 $n$，RoPE 的目标是设计一个旋转函数 $R(\cdot)$，使得：

$$\text{Attention}(\mathbf{q}_m, \mathbf{k}_n) = \langle R(\mathbf{q}_m, m), R(\mathbf{k}_n, n) \rangle = \langle \mathbf{q}_m, \mathbf{k}_n \rangle_{\text{relative}}$$

即注意力分数只依赖于 **相对位置** $m-n$。

### 2.2 二维旋转的数学推导

{IMAGE:4}

让我们从最简单的二维情况开始推导。

对于二维向量 $\mathbf{q} = [q_0, q_1]^T$，我们希望应用旋转矩阵：

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

应用旋转后：
$$\mathbf{q}' = R(\theta) \cdot \mathbf{q}$$

两个旋转后向量的内积为：

$$\langle R(\theta_m)\mathbf{q}, R(\theta_n)\mathbf{k} \rangle = \langle \mathbf{q}, \mathbf{k} \rangle \cos(\theta_m - \theta_n) + \underbrace{(\cdots)}_{\text{正交分量}}$$

{IMAGE:5}

{IMPORTANT}核心概念{/IMPORTANT}

**关键发现**：旋转后的内积结果**只依赖于相对角度差** $\theta_m - \theta_n$！这正是我们需要的性质。

### 2.3 多维扩展

{IMAGE:6}

对于 $d$ 维向量，我们将其分成 $d/2$ 个**二维子平面**，每个子平面应用不同的旋转角度：

$$\mathbf{q}'_m = \begin{bmatrix} q_0 \cos(m\theta_0) - q_1 \sin(m\theta_0) \\ q_0 \sin(m\theta_0) + q_1 \cos(m\theta_0) \\ q_2 \cos(m\theta_1) - q_3 \sin(m\theta_1) \\ q_2 \sin(m\theta_1) + q_3 \cos(m\theta_1) \\ \vdots \end{bmatrix}$$

其中旋转频率为：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, 2, \ldots, d/2 - 1$$

### 2.4 RoPE 的矩阵形式

{IMAGE:7}

RoPE 可以等效地表示为与位置编码的**逐元素乘法**：

$$\mathbf{q}'_m = \mathbf{q}_m \odot \begin{bmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \end{bmatrix} + \mathbf{q}_m^{\text{rotated}} \odot \begin{bmatrix} -\sin(m\theta_0) \\ \sin(m\theta_0) \\ -\sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \end{bmatrix}$$

{IMAGE:8}

{KNOWLEDGE}背景知识{/KNOWLEDGE}

这种表示形式更加高效，因为：
- 不需要显式构造大型旋转矩阵
- 可以通过简单的逐元素运算实现
- 与 Flash Attention 等高效注意力机制兼容

### 2.5 RoPE 与其他位置编码的对比

{IMAGE:9}

| 特性 | 绝对位置编码 | 相对位置编码 | RoPE |
|------|------------|------------|------|
| 位置信息编码方式 | 与 embedding 相加 | 融入注意力计算 | 旋转 query/key |
| 相对位置表达 | 隐式 | 显式 | 显式（旋转不变性） |
| 计算效率 | 较低 | 中等 | 高（线性运算） |
| 外推能力 | 差 | 一般 | 较好 |
| 实现复杂度 | 低 | 高 | 中等 |

---

## 3. RoPE 的实现

### 3.1 基础 RoPE 实现

{IMAGE:10}

```python
import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    预计算旋转位置编码的频率矩阵
    
    Args:
        dim: 嵌入维度
        seq_len: 序列长度
        theta: 基础频率参数
    
    Returns:
        freqs_cis: 复数形式的频率矩阵, shape = [seq_len, dim//2]
    """
    # 计算各层的旋转频率
    # theta_i = theta ** (-2i/dim) for i in [0, 1, ..., dim//2-1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # 生成位置索引
    positions = torch.arange(seq_len)
    
    # 计算每个位置的相位角: theta_i * pos
    # 外积得到 [seq_len, dim//2] 的角度矩阵
    angles = positions[:, None] * freqs[None, :]
    
    # 转换为复数形式: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    
    return freqs_cis

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, 
                     freqs_cis: torch.Tensor):
    """
    应用旋转位置编码到 query 和 key
    
    Args:
        q: query 向量, shape = [batch, seq_len, n_heads, head_dim]
        k: key 向量, shape = [batch, seq_len, n_heads, head_dim]
        freqs_cis: 预计算的频率矩阵, shape = [seq_len, head_dim//2]
    
    Returns:
        q_rot, k_rot: 应用 RoPE 后的 query 和 key
    """
    # 将实数张量转换为复数形式
    # q.shape = [..., seq_len, head_dim], 拆分奇偶位置
    q_real = q.float().reshape(*q.shape[:-1], -1, 2)  # [..., seq_len, head_dim/2, 2]
    k_real = k.float().reshape(*k.shape[:-1], -1, 2)
    
    # 转换为复数
    q_complex = torch.view_as_complex(q_real)
    k_complex = torch.view_as_complex(k_real)
    
    # 调整 freq_cis 维度以便广播
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    
    # 复数乘法实现旋转
    q_rot = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rot = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
    
    return q_rot, k_rot
```

### 3.2 融合版本的优化实现

{IMAGE:11}

```python
def apply_rotary_emb_fused(q: torch.Tensor, k: torch.Tensor,
                           freqs_cis: torch.Tensor):
    """
    更高效的融合实现，直接使用实数运算
    
    这种方式避免了复数转换的开销，在实际框架中被广泛采用
    """
    batch_size, seq_len, n_heads, head_dim = q.shape
    head_dim_half = head_dim // 2
    
    # 分离奇偶维度
    q1 = q[..., :head_dim_half]  # 前半部分
    q2 = q[..., head_dim_half:]  # 后半部分
    k1 = k[..., :head_dim_half]
    k2 = k[..., head_dim_half:]
    
    # 展开 freq_cis 的实部和虚部
    cos = freqs_cis.real  # [seq_len, head_dim_half]
    sin = freqs_cis.imag  # [seq_len, head_dim_half]
    
    # 广播到正确维度
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim_half]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 应用旋转公式
    # q' = q1 * cos + rotate_half(q2) * sin
    # 其中 rotate_half 将后半部分取负并交换顺序
    q_rot = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin
    ], dim=-1)
    
    k_rot = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin
    ], dim=-1)
    
    return q_rot, k_rot
```

{IMAGE:12}

{WARNING}易错点{/WARNING}

**实现注意事项**：
1. `freqs_cis` 必须预计算并存储，**不要在每次前向传播时重新计算**
2. 确保 `head_dim` 为偶数，否则无法正确拆分
3. 使用半精度（fp16/bf16）时，注意数值精度问题
4. 在推理时，`freqs_cis` 的长度应覆盖最大序列长度

---

## 4. 位置编码的长度外推问题

### 4.1 什么是长度外推

{IMPORTANT}核心概念{/IMPORTANT}

**长度外推（Length Extrapolation）** 是指模型在训练时使用固定最大长度（如 2048 tokens），但推理时可以处理更长序列（如 8192+ tokens）的能力。

{KNOWLEDGE}背景知识{/KNOWLEDGE}

这个问题在大语言模型中尤为重要：
- 训练成本高昂，无法每次都重新训练
- 实际应用场景需要处理长文档、长对话
- 上下文窗口直接影响模型可用性

### 4.2 RoPE 外推的挑战

{IMAGE:1}

即使 RoPE 相对位置编码的特性使其理论上具有更好的外推能力，但在实践中仍面临挑战：

1. **分布偏移**：训练时见过的大位置编码与测试时的位置编码分布不同
2. **低频信息丢失**：高频维度（$i$ 较大）的旋转周期短，在短序列中就能观察到多周期；而低频维度的周期可能超过训练长度
3. **注意力机制的脆弱性**：远程位置的注意力分数可能变得不稳定

---

## 5. YaRN：RoPE 外推技术

### 5.1 YaRN 概述

{IMPORTANT}核心概念{/IMPORTANT}

**YaRN（Yet another RoPE extensioN）** 是 2023 年提出的 RoPE 扩展方法，通过**三项关键改进**显著提升了 RoPE 的长度外推能力：

1. **温度缩放（Temperature Scaling）**
2. **注意力缩放（Attention Scaling）**
3. **位置插值（Position Interpolation）**

### 5.2 理论基础：频率与外推性

{IMAGE:2}

YaRN 的核心洞察在于：RoPE 中不同频率维度的行为差异导致了外推困难。

设旋转角度为 $\theta_i = \theta_0^{-2i/d}$，对应的**旋转周期**为：

$$T_i = \frac{2\pi}{\theta_i}$$

{IMAGE:3}

- **高频维度**（$i$ 大）：$\theta_i$ 大，周期 $T_i$ 小，容易在短序列中充分采样
- **低频维度**（$i$ 小）：$\theta_i$ 小，周期 $T_i$ 大，可能超过训练长度

当测试序列超过训练长度时，低频维度面临**从未见过的角度范围**，导致泛化失败。

### 5.3 温度缩放

{IMAGE:4}

{IMPORTANT}核心概念{/IMPORTANT}

**温度缩放**通过引入参数 $\tau$ 来调整所有旋转频率：

$$\theta_i' = \frac{\theta_i}{\tau}$$

{IMAGE:5}

物理意义：
- 增大 $\tau$ 会减小旋转角度，使每个位置的旋转更"小"
- 这使得在相同序列长度内能覆盖更多周期
- 相当于重新调整了频率分布，使低频维度更不易于"超调"

```python
def precompute_freqs_cis_with_temperature(
    dim: int, 
    seq_len: int, 
    theta: float = 10000.0,
    tau: float = 1.0
):
    """
    带温度缩放的 RoPE 频率预计算
    
    Args:
        tau: 温度参数，通常设置为略大于 1.0（如 1.0-2.0）
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # 温度缩放
    freqs = freqs / tau
    
    positions = torch.arange(seq_len)
    angles = positions[:, None] * freqs[None, :]
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    
    return freqs_cis
```

### 5.4 位置插值

{IMAGE:6}

{IMAGE:7}

**位置插值（Position Interpolation, PI）** 由苏剑林等人提出，核心思想是**线性压缩位置索引**：

对于原始位置 $x \in [0, L_{max})$，通过缩放因子 $s$ 映射到训练范围：

$$x' = \frac{x}{s}, \quad s = \frac{L_{new}}{L_{max}}$$

{IMAGE:8}

{KNOWLEDGE}背景知识{/KNOWLEDGE}

这确保了任意新位置 $x$ 都被映射到训练时见过的位置范围内：
- $x \in [0, L_{new}) \Rightarrow x' \in [0, L_{max})$
- 模型不需要外推到未见过的位置

但直接插值会损失高频分辨率，YaRN 通过组合多种策略来解决这个问题。

### 5.5 注意力缩放因子

{IMAGE:9}

{IMPORTANT}核心概念{/IMPORTANT}

**注意力缩放**是 YaRN 的关键创新之一。通过在注意力计算中引入缩放因子 $\sqrt{t}$，补偿因位置压缩导致的信息密度变化：

$$\text{Attention}_{\text{YaRN}}(q, k, v) = \text{Softmax}\left(\frac{q^T k}{\sqrt{t} \cdot s}\right) v$$

其中 $t = \frac{\theta_i}{\beta}$ 是与频率相关的参数，$\beta$ 是边界阈值。

{IMAGE:10}

{IMAGE:11}

```python
def compute_attention_with_yarn(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: float = 1.0,
    yarn_enabled: bool = True
):
    """
    带 YaRN 缩放的注意力计算
    
    Args:
        scale: 额外的注意力缩放因子
        yarn_enabled: 是否启用 YaRN
    """
    # 应用 RoPE
    q_rot, k_rot = apply_rotary_emb_fused(q, k, freqs_cis)
    
    # 计算注意力分数
    # [batch, n_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
    
    if yarn_enabled:
        # YaRN 注意力缩放
        # 缩放因子帮助模型更好地处理外推位置
        scores = scores / (scale * math.sqrt(q.shape[-1]))
    else:
        scores = scores / math.sqrt(q.shape[-1])
    
    # Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    attn_output = torch.matmul(attn_weights, v)
    
    return attn_output, attn_weights
```

### 5.6 完整 YaRN 实现

{IMAGE:12}

```python
class YaRNRoPE:
    """
    完整的 YaRN RoPE 实现，支持长度外推
    """
    def __init__(
        self,
        dim: int,
        max_position: int = 2048,      # 训练时的最大长度
        base: float = 10000.0,
        scaling_factor: float = 1.0,   # 位置缩放因子
        beta: float = 32.0,            # YaRN 频率边界
        mscale: float = 1.0,           # 注意力缩放因子
    ):
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scaling_factor = scaling_factor
        self.beta = beta
        self.mscale = mscale
        
        # 预计算频率
        self._compute_freqs()
    
    def _compute_freqs(self):
        """计算旋转频率"""
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2).float() / self.dim
        ))
        self.inv_freq = inv_freq
        
        # YaRN: 应用温度缩放
        # 频率调整使外推更稳定
        self.scaled_inv_freq = inv_freq / self.scaling_factor
    
    def _get_mscale(self, scale_factor: float = 1.0) -> float:
        """
        计算 YaRN 的注意力缩放因子
        
        基于原始论文的推荐值
        """
        if self.mscale != 1.0:
            return self.mscale
        
        # 当缩放因子变化时动态计算 mscale
        # 经验公式，帮助平衡信息密度
        return 0.1 * scale_factor * math.log(scale_factor) + 1.0
    
    def forward(
        self, 
        seq_len: int,
        device: torch.device
    ):
        """
        生成指定长度的 RoPE 频率
        
        支持动态扩展到任意长度，实现长度外推
        """
        # 生成位置序列（可超出 max_position）
        positions = torch.arange(seq_len, device=device)
        
        # 计算角度
        angles = positions[:, None] * self.scaled_inv_freq[None, :]
        
        # 转换为复数形式
        freqs_cis = torch.polar(
            torch.ones_like(angles), 
            angles
        )
        
        return freqs_cis
```

---

## 6. 总结与实践建议

### 6.1 本讲核心要点

{IMPORTANT}核心概念{/IMPORTANT}

1. **RoPE 原理**：通过将位置编码为旋转矩阵，使 query 和 key 旋转后内积只依赖相对位置
2. **高效实现**：RoPE 可通过逐元素乘法实现，无需构造大型稀疏矩阵
3. **外推挑战**：不同频率维度的周期差异导致长序列外推困难
4. **YaRN 改进**：温度缩放 + 注意力缩放 + 位置插值组合提升外推能力

### 6.2 实践建议

| 场景 | 推荐配置 |
|------|---------|
| 短上下文（≤2K） | 原始 RoPE，无需 YaRN |
| 中等上下文（2K-8K） | RoPE + 轻微温度缩放（τ≈1.1） |
| 长上下文（>8K） | YaRN + 位置插值 + 注意力缩放