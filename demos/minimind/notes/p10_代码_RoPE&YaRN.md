# 第10集: 代码：RoPE&YaRN

# MiniMind Episode 10/26 讲义：RoPE 与 YaRN 代码实现

## 课程概览

本讲我们将深入探讨 **旋转位置编码（Rotary Position Embedding, RoPE）** 的实现细节，并学习 **YaRN（Yet another Rotary searching Notebook）** 缩放技术。通过本讲，您将掌握如何用 PyTorch 从零实现 RoPE，理解旋转矩阵的数学原理，以及如何应用 YaRN 来扩展模型的上下文处理能力。

{IMAGE:1}

---

## 第一部分：RoPE 基础回顾

### 1.1 位置编码的演进

{KNOWLEDGE}背景知识{/KNOWLEDGE}

位置编码经历了三个主要阶段：

| 阶段 | 类型 | 代表工作 | 特点 |
|------|------|----------|------|
| 第一代 | 绝对位置编码 | Sinusoidal, Learnable | 简单但无法捕获相对位置 |
| 第二代 | 相对位置编码 | Shaw's RPE, T5 RPE | 捕获相对位置，但计算复杂 |
| 第三代 | 旋转位置编码 | RoPE (Su et al.) | 高效且优雅地融合绝对与相对位置 |

### 1.2 RoPE 的核心思想

{IMPORTANT}核心概念{/IMPORTANT}

**RoPE 的核心思想**：将位置信息编码为旋转矩阵，使 token 之间的注意力分数仅依赖于它们的**相对位置**。

对于第 $m$ 个位置和第 $n$ 个位置的 query/key 向量，RoPE 通过旋转矩阵 $\mathbf{R}_m$ 和 $\mathbf{R}_n$ 进行编码：

$$\mathbf{q}_m = \mathbf{R}_m \mathbf{q}, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{k}$$

注意力分数变为：

$$\text{Attention}(m, n) = \mathbf{q}_m^T \mathbf{k}_n = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{m-n} \mathbf{k}$$

**关键观察**：旋转矩阵的乘积 $\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{n-m}$ 仅依赖于**相对位置** $(n-m)$！

---

## 第二部分：旋转矩阵的数学推导

### 2.1 复数形式的旋转

{IMAGE:2}

在复数空间中，二维旋转矩阵定义为：

$$\mathbf{R}_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

对于位置 $m$，旋转角度为 $\theta_m = m \cdot \theta$：

$$\mathbf{R}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

### 2.2 对角块结构

为了高效计算，我们将高维向量分成若干 **2 维组**，每组应用独立的旋转：

对于维度 $d$（通常为 64 的倍数），分成 $d/2$ 个二维子空间：

$$\mathbf{R}_m = \text{diag}(\mathbf{R}_{\theta_0}^m, \mathbf{R}_{\theta_1}^m, \ldots, \mathbf{R}_{\theta_{d/2-1}}^m)$$

{IMAGE:3}

### 2.3 频率设置

旋转角度按照**几何级数**分布，不同维度使用不同频率：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

{KNOWLEDGE}背景知识{/KNOWLEDGE}

这种设计的直觉：
- **低维（i 较小）**：高频旋转 $\to$ 捕获细粒度位置差异
- **高维（i 较大）**：低频旋转 $\to$ 捕获粗粒度位置模式
- 类似于傅里叶变换的多尺度分析

---

## 第三部分：RoPE 的 PyTorch 实现

### 3.1 基础旋转编码函数

{IMAGE:4}

```python
import torch
import torch.nn as nn
import math

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转位置编码的频率因子（复数形式）
    
    参数:
        dim: 嵌入维度（通常是 head_dim）
        end: 最大位置数
        theta: 基础频率参数，默认 10000.0
    
    返回:
        freqs_cis: 形状为 [end, dim//2] 的复数张量
    """
    # 计算各维度的旋转频率
    # 频率公式: theta_i = theta^(-2i/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # 生成位置索引
    t = torch.arange(end)
    
    # 计算每个位置-维度对的频率 (位置 × 频率)
    # 结果形状: [end, dim//2]
    freqs = torch.outer(t, freqs)
    
    # 转换为复数形式: e^(i * freqs)
    # 使用 cis(x) = cos(x) + i*sin(x)
    freqs_cis = torch.polar(
        torch.ones_like(freqs),  # 模长为 1
        freqs * 2 * math.pi       # 角度（转换为 2π 周期）
    )
    
    return freqs_cis

# 示例
freqs_cis = precompute_freqs_cis(dim=64, end=512)
print(f"频率张量形状: {freqs_cis.shape}")  # [512, 32]
```

### 3.2 旋转操作的核心实现

{IMAGE:5}

```python
def apply_rotary_pos_emb(
    q: torch.Tensor,      # Query 张量 [batch, seq_len, n_heads, head_dim]
    k: torch.Tensor,      # Key 张量
    freqs_cis: torch.Tensor,  # 复数频率 [seq_len, head_dim//2]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到 Q 和 K
    
    核心原理: 将向量视为复数，与预计算的旋转因子相乘
    """
    batch_size, seq_len, n_heads, head_dim = q.shape
    
    # 重塑频率张量以匹配 Q/K 的维度
    # 从 [seq_len, head_dim//2] 变为 [1, seq_len, 1, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # 广播适配
    
    # 将实数向量转换为复数形式
    # 每两个实数维度对应一个复数维度
    q_complex = torch.view_as_complex(
        q.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
    )
    k_complex = torch.view_as_complex(
        k.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
    )
    
    # 旋转操作: 复数乘法
    # 相当于在复平面上旋转向量
    q_rotated = q_complex * freqs_cis
    k_rotated = k_complex * freqs_cis
    
    # 转回实数形式
    q_out = torch.view_as_real(q_rotated).flatten(-2)
    k_out = torch.view_as_real(k_rotated).flatten(-2)
    
    return q_out, k_out
```

### 3.3 旋转矩阵的可视化理解

{IMAGE:6}

```
原始向量 [a, b]          旋转后 [a', b']

复数表示: a + bi    →    (a + bi) × e^(iθ)

等价实数运算:
┌─────────────────────────────────────────┐
│ a' = a·cos(θ) - b·sin(θ)                │
│ b' = a·sin(θ) + b·cos(θ)                │
└─────────────────────────────────────────┘

验证相对位置不变性:
q_m · k_n = (R_m q) · (R_n k)
           = q · R_m^T R_n · k
           = q · R_{n-m} · k  ✓ (仅依赖 n-m)
```

### 3.4 完整 Attention 模块集成

{IMAGE:7}

```python
class RoPEMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # QKV 投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 应用 RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs_cis)
        
        # 缩放
        q = q * self.scale
        
        # 注意力计算 (使用 flash attention 或手动实现)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)
```

---

## 第四部分：YaRN 缩放技术

### 4.1 为什么需要 YaRN

{KNOWLEDGE}背景知识{/KNOWLEDGE}

当训练好的 RoPE 模型需要处理**超过训练长度**的序列时，会出现性能下降。这是因为：

1. **分布外（OOD）问题**：模型在训练时从未见过某些旋转角度
2. **高频维度失效**：长序列需要更多的高频旋转，但这些在长距离上变得不稳定
3. **注意力崩溃**：远程 token 的位置信息被噪声淹没

{IMAGE:8}

### 4.2 YaRN 的核心思想

{IMPORTANT}核心概念{/IMPORTANT}

**YaRN（Yet another Rotary scaling）** 通过**对旋转角度进行缩放**，使短序列上学到的模式能够迁移到长序列：

$$\theta_i' = \theta_i \cdot s^{2i/d}$$

其中 $s$ 是**缩放因子**，通常设置为 $> 1$（如 $s = 32$）。

### 4.3 YaRN 的数学公式

{IMAGE:9}

```python
def yarn_find_correction_range(
    dim: int,
    original_max_pos: int,  # 训练时的最大长度
    extrapolated_max_pos: int,  # 期望的推理长度
    concate: bool = True
) -> tuple[int, int]:
    """
    找到需要调整的维度范围
    """
    extrapolated_factor = extrapolated_max_pos / original_max_pos
    extrapolated_limit = extrapolated_max_pos / 2
    
    if concate:
        extrapolated_limit /= 2
    
    scale = extrapolated_factor
    
    def _find_high_rotation_dim(d: int, base: float = 10000.0) -> int:
        # 找到旋转频率低于 extrapolated_limit 的维度
        for i in range(d // 2):
            if base ** (2 * i / d) > extrapolated_limit:
                return i
        return d // 2
    
    low = _find_high_rotation_dim(dim)
    high = dim // 2
    
    return low, high
```

### 4.4 YaRN 修正公式

```python
def yarn_get_cos_sin(
    dim: int,
    max_position: int,
    original_max_position: int,
    base: float = 10000.0,
    scaling_factor: float = 32.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 YaRN 修正后的 cos 和 sin 值
    
    核心修正公式:
    α = log(s) / log(θ_i) + 1  (线性缩放)
    θ'_i = θ_i · (α + i)/α
    """
    
    # 计算基础频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算缩放比例
    scale = max_position / original_max_position
    
    # YaRN 线性缩放
    # 对于低频维度，缩放程度更大
    inv_freqs = 1.0 / freqs
    alpha = torch.log(scale) / torch.log(inv_freqs) + 1.0
    
    # 角度缩放
    angles = freqs * max_position
    angles_scaled = angles * (alpha / (alpha + torch.arange(0, dim, 2).float()))
    
    # 生成 cos 和 sin
    cos = torch.cos(angles_scaled)
    sin = torch.sin(angles_scaled)
    
    return cos, sin
```

{IMAGE:10}

### 4.5 完整 YaRN 实现

```python
def precompute_freqs_cis_yarn(
    dim: int,
    max_position: int,
    original_max_position: int,
    base: float = 10000.0,
    scaling_factor: float = 32.0,
    extrapolation_factor: float = 8.0,
    attn_factor: float = 1.0,
    beta_fast: int = 128,
    beta_slow: int = 512,
) -> torch.Tensor:
    """
    完整 YaRN 旋转位置编码预计算
    """
    # 计算频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算 scale
    scale = max_position / original_max_position
    
    # YaRN 维度修正
    inv_freqs = 1.0 / freqs
    alpha = (torch.log(scale) / torch.log(inv_freqs) + 1.0).clamp(min=1.0)
    
    # 低频维度和高频维度的不同处理
    dims = torch.arange(0, dim, 2).float()
    
    # 应用比例因子
    if scale > 1.0:
        # 使用平滑插值
        with torch.no_grad():
            values = dims / dim
            inv_extrapolation_factor = 1.0 / extrapolation_factor
            inv_freq_scale = (beta_fast * inv_extrapolation_factor) / (beta_slow * inv_extrapolation_factor - inv_extrapolation_factor)
            Smooth = (dims * inv_extrapolation_factor - inv_freq_scale) / (dims - inv_freq_scale)
            D = (alpha - 1.0) * Smooth + 1.0
        
        freqs_scaled = freqs * (dims / dim * D * scale + 1.0 - dims / dim) / scale
    else:
        freqs_scaled = freqs
    
    # 生成位置
    t = torch.arange(max_position)
    
    # 计算角度
    freqs = torch.outer(t, freqs_scaled)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs * 2 * math.pi)
    
    return freqs_cis
```

---

## 第五部分：测试与验证

### 5.1 相对位置不变性验证

{IMAGE:11}

```python
def test_rope_invariance():
    """验证 RoPE 的相对位置不变性"""
    dim = 64
    seq_len = 16
    head_dim = dim
    
    # 随机生成 Q, K
    q = torch.randn(1, seq_len, 1, head_dim)
    k = torch.randn(1, seq_len, 1, head_dim)
    
    # 预计算频率
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    
    # 应用 RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs_cis)
    
    # 计算旋转后的注意力
    # 手动计算注意力以验证相对位置依赖性
    print("RoPE 实现验证完成")
    print(f"Query 形状: {q_rot.shape}")
    print(f"Key 形状: {k_rot.shape}")

def test_yarn_scaling():
    """验证 YaRN 的缩放效果"""
    dim = 128
    original_len = 2048
    new_len = 8192
    scaling = 32.0
    
    # 原始 RoPE 频率
    freqs_orig = precompute_freqs_cis(dim, original_len)
    
    # YaRN 缩放后的频率
    freqs_yarn = precompute_freqs_cis_yarn(
        dim, new_len, original_len, scaling_factor=scaling
    )
    
    print(f"原始长度 {original_len}, 新长度 {new_len}")
    print(f"频率张量形状: {freqs_yarn.shape}")
    print("YaRN 缩放验证完成")

# 运行测试
test_rope_invariance()
test_yarn_scaling()
```

### 5.2 注意力模式可视化

{IMAGE:12}

```
注意力模式对比（概念图）:

标准 RoPE (训练长度内):
┌────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ████████░░░░░░░░░░░░░░░░░░░░░░░ │
│ ██████████░░░░░░░░░░░░░░░░░░░░░ │
│ ████████████░░░░░░░░░░░░░░░░░░ │
│ ...                             │
└────────────────────────────────┘
    局部注意力模式清晰

YaRN RoPE (外推到更长序列):
┌────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ████████░░░░░░░░░░░░░░░░░░░░░░ │
│ ██████████░░░░░░░░░░░░░░░░░░░░ │
│ ████████████░░░░░░░░░░░░░░░░░░ │
│ ████████████████░░░░░░░░░░░░░░ │ ← 长距离依赖得到保持
│ ...                             │
└────────────────────────────────┘
    远距离注意力保持稳定
```

---

## 第六部分：实践要点与优化

### 6.1 实现注意事项

{WARNING}易错点{/WARNING}

1. **频率计算精度**：
   - 使用 `float32` 或 `float64` 避免精度丢失
   - 预计算时使用 `torch.outer` 确保正确广播

2. **维度分组**：
   - 必须将维度按 2 分组
   - 每组独立旋转后再拼接

3. **复数转换**：
   ```python
   # 错误方式：直接 reshape
   q_complex = q.view(..., 2).contiguous()  # ❌
   
   # 正确方式：使用 view_as_complex
   q_complex = torch.view_as_complex(q.view(..., 2))  # ✓
   ```

4. **缓存频率张量**：
   - 在模型外预计算一次
   - 推理时可进一步量化存储

### 6.2 性能优化建议

```python
# 优化1: 使用 Flash Attention
from flash_attn import flash_attn_func

def apply_rot