# 第17集: 组装：Model

## 第17集：组装：Model

### 课程定位

本集进入 MiniMind Transformer 实现中非常关键的一步：把前面已经手写完成的模块组装成一个完整的语言模型 `Model`。

在前面的课程中，我们已经分别实现过：

- `RMSNorm`
- `Attention`
- `FeedForward`
- `TransformerBlock`
- `RoPE` 旋转位置编码
- `KV Cache`
- Token Embedding
- 输出层 Linear Head

这一集的目标不是重新解释某个单独模块，而是把这些模块按照大模型的标准结构串联起来，形成一个可以训练、可以推理、可以计算 loss 的完整 Transformer Decoder-only 模型。

{IMAGE:5}

{IMPORTANT}本集核心目标：理解一个大语言模型类 `Model` 的完整组成，包括输入 token 如何变成 logits，训练时如何计算 loss，推理时如何只取最后一个 token 的输出。{/IMPORTANT}

本节小结：本集是从“零件实现”走向“完整模型”的关键阶段，重点在于整体结构和数据流。

---

## 一、Transformer Model 的整体结构

### 1. Decoder-only 架构

MiniMind 使用的是典型的 Decoder-only Transformer 结构，和 GPT、LLaMA 一类自回归语言模型类似。

整体流程可以概括为：

$$
\text{tokens} \rightarrow \text{Embedding} \rightarrow N \times \text{TransformerBlock} \rightarrow \text{Norm} \rightarrow \text{LM Head} \rightarrow \text{logits}
$$

也就是：

1. 输入 token id
2. 经过词嵌入层变成向量
3. 依次通过多层 Transformer Block
4. 做最终归一化
5. 映射到词表大小
6. 得到每个位置预测下一个 token 的概率分布

{IMAGE:1}

### 2. 模型类的职责

`Model` 类通常负责：

- 定义词嵌入层
- 定义多层 Transformer Block
- 定义最终 RMSNorm
- 定义输出线性层
- 预计算 RoPE 频率
- 管理 forward 训练逻辑
- 管理推理时的 KV Cache
- 在有标签时计算交叉熵损失

{KNOWLEDGE}在大语言模型中，`Model` 类不是某一个模块，而是“总装车间”。它负责把所有子模块按照数据流顺序组织起来，并处理训练和推理之间的差异。{/KNOWLEDGE}

本节小结：完整模型的本质是多个模块的有序组合，Decoder-only 模型的输出目标是预测下一个 token。

---

## 二、模型配置参数

### 1. 常见配置项

完整模型通常依赖一个配置类，例如：

```python
@dataclass
class LMConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 6400
    max_seq_len: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5
    hidden_dim: int = None
```

这些字段决定了模型规模：

- `dim`：隐藏层维度，也就是 token embedding 的维度
- `n_layers`：Transformer Block 层数
- `n_heads`：注意力头数量
- `n_kv_heads`：KV 头数量，用于 GQA/MQA
- `vocab_size`：词表大小
- `max_seq_len`：最大上下文长度
- `dropout`：训练时随机失活概率
- `norm_eps`：归一化中的数值稳定项
- `hidden_dim`：FFN 中间层维度

{IMAGE:6}

### 2. 参数规模与模型能力

模型参数量大致来自四个部分：

1. 词嵌入矩阵
2. Attention 中的 Q/K/V/O 投影
3. FeedForward 中的线性层
4. 输出层 LM Head

词嵌入参数量为：

$$
\text{Embedding Params} = \text{vocab\_size} \times \text{dim}
$$

如果词表大小是 $6400$，隐藏维度是 $512$，那么 embedding 层参数约为：

$$
6400 \times 512 = 3,276,800
$$

{WARNING}模型越大不一定越好。参数规模、数据规模、训练算力、上下文长度需要配合，否则容易出现训练不足、过拟合或推理成本过高。{/WARNING}

本节小结：配置类是模型结构的蓝图，控制层数、维度、头数、词表和上下文长度。

---

## 三、词嵌入层：从 token id 到向量

### 1. 输入张量形状

语言模型输入通常是一个二维整数张量：

$$
x \in \mathbb{Z}^{B \times T}
$$

其中：

- $B$：batch size
- $T$：sequence length
- 每个元素是一个 token id

例如：

```python
# batch_size = 2, seq_len = 5
x = torch.tensor([
    [12, 45, 87, 91, 3],
    [8, 19, 20, 64, 7],
])
```

### 2. Embedding 映射

Embedding 层将 token id 映射为连续向量：

$$
h = \text{Embedding}(x)
$$

输出形状变为：

$$
h \in \mathbb{R}^{B \times T \times C}
$$

其中 $C = \text{dim}$。

```python
self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

h = self.tok_embeddings(tokens)
```

{IMAGE:7}

{KNOWLEDGE}Embedding 可以理解为一个可训练查表操作。输入 token id 是离散编号，输出 embedding 是模型可以处理的连续向量。{/KNOWLEDGE}

本节小结：词嵌入层负责把离散 token 转换为连续向量，是 Transformer 的输入入口。

---

## 四、RoPE 位置编码的预计算

### 1. 为什么需要位置编码

自注意力机制本身对输入顺序不敏感。也就是说，如果没有位置编码，模型很难区分：

- “我 爱 你”
- “你 爱 我”

这两个序列中的 token 集合类似，但语义完全不同。

因此，Transformer 必须加入位置信息。

MiniMind 中通常使用 RoPE，即旋转位置编码。

{IMAGE:8}

### 2. RoPE 的基本思想

RoPE 不直接把位置向量加到 token embedding 上，而是在计算 Q、K 时对向量做旋转变换。

其核心思想是：让位置关系通过向量旋转角度体现出来。

对于某个维度对，可以写成：

$$
\begin{bmatrix}
x'_1 \\
x'_2
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

其中 $\theta$ 与 token 位置有关。

### 3. 预计算频率

为了提高效率，模型初始化时通常会预先计算 RoPE 所需的复数频率或正余弦值：

```python
self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
    dim=config.dim // config.n_heads,
    end=config.max_seq_len
)
```

forward 时只需要根据当前序列位置切片即可：

```python
freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]
```

{WARNING}RoPE 的位置切片必须和当前 token 的真实位置对应。训练时通常从 0 开始，推理增量生成时则需要使用 `start_pos` 接续历史位置。{/WARNING}

本节小结：RoPE 通过旋转 Q/K 向量注入位置信息，预计算可以减少 forward 阶段的重复开销。

---

## 五、Transformer Block 堆叠

### 1. 多层结构

完整模型中会有多个 Transformer Block：

```python
self.layers = nn.ModuleList([
    TransformerBlock(layer_id, config)
    for layer_id in range(config.n_layers)
])
```

每一层通常包含：

- Attention
- FeedForward
- RMSNorm
- 残差连接

{IMAGE:9}

每层的计算可以简化为：

$$
h = h + \text{Attention}(\text{Norm}(h))
$$

$$
h = h + \text{FeedForward}(\text{Norm}(h))
$$

这是 Pre-Norm Transformer 的典型写法。

### 2. 为什么要堆很多层

单层 Transformer 只能做一次特征交互，多层堆叠可以让模型逐步抽象：

- 底层捕捉局部 token 关系
- 中层学习短语、句法关系
- 高层形成语义和任务相关表示

虽然小模型层数有限，但结构上与大模型一致。

{IMAGE:10}

{IMPORTANT}Transformer 的能力不是来自某个神奇模块，而是来自 Attention、FFN、残差、归一化在多层堆叠中的协同作用。{/IMPORTANT}

本节小结：`ModuleList` 用于保存多层 Transformer Block，forward 时按顺序逐层处理 hidden states。

---

## 六、完整 forward 训练流程

### 1. forward 的输入

训练时，forward 通常接收：

```python
def forward(self, tokens, targets=None, start_pos=0):
    ...
```

其中：

- `tokens`：输入 token id
- `targets`：监督标签，通常是下一个 token
- `start_pos`：推理缓存位置，训练时一般为 0

输入形状：

$$
tokens \in \mathbb{Z}^{B \times T}
$$

标签形状：

$$
targets \in \mathbb{Z}^{B \times T}
$$

{IMAGE:11}

### 2. 训练阶段完整代码示例

```python
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # dropout 只在训练阶段生效
        self.dropout = nn.Dropout(config.dropout)

        # 多层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, config)
            for layer_id in range(config.n_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 输出到词表维度
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 预计算 RoPE
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len
        )

    def forward(self, tokens, targets=None, start_pos=0):
        batch_size, seq_len = tokens.shape

        # 1. token id -> embedding
        h = self.tok_embeddings(tokens)

        # 2. dropout
        h = self.dropout(h)

        # 3. 取出当前位置对应的 RoPE 参数
        freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]

        # 4. 逐层通过 Transformer Block
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, start_pos)

        # 5. 最终归一化
        h = self.norm(h)

        # 6. 输出 logits
        logits = self.output(h)

        # 7. 如果提供 targets，则计算训练 loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
```

本节小结：训练 forward 是完整序列并行计算，每个位置都会输出一个词表分布，并与 targets 计算交叉熵损失。

---

## 七、Logits 与语言模型目标

### 1. logits 的含义

模型输出：

$$
logits \in \mathbb{R}^{B \times T \times V}
$$

其中：

- $B$：batch size
- $T$：序列长度
- $V$：词表大小

对于每个位置 $t$，模型都会输出一个长度为 $V$ 的向量，表示下一个 token 的未归一化分数。

{IMAGE:12}

### 2. Softmax 概率

logits 经过 softmax 得到概率：

$$
P(x_{t+1}=i \mid x_{\le t}) =
\frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

其中：

- $z_i$ 是第 $i$ 个词的 logit
- 分母对整个词表求和
- 输出表示下一个 token 是某个词的概率

### 3. 自回归训练目标

语言模型训练目标是预测下一个 token：

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(x_{t+1} \mid x_{\le t})
$$

在 PyTorch 中通常用交叉熵：

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1)
)
```

{IMPORTANT}语言模型不是直接“理解句子”，而是在训练中不断学习：给定前文，预测下一个 token。复杂能力来自这个目标在大规模数据上的涌现。{/IMPORTANT}

本节小结：logits 是每个位置对词表的预测分数，训练目标是最大化正确下一个 token 的概率。

---

## 八、训练与推理的 forward 差异

### 1. 训练：整段并行

训练时，输入通常是一整段 token：

```python
tokens.shape = [batch_size, seq_len]
```

模型一次性输出所有位置的 logits：

```python
logits.shape = [batch_size, seq_len, vocab_size]
```

这样可以并行计算 loss，效率高。

{IMAGE:13}

### 2. 推理：只关心最后一个位置

自回归生成时，每一步只需要预测下一个 token，因此只需要最后一个位置的 logits：

```python
logits = self.output(h[:, [-1], :])
```

这样输出形状是：

$$
[B, 1, V]
$$

而不是：

$$
[B, T, V]
$$

这样可以减少不必要计算和显存占用。

```python
if targets is None:
    # 推理时只取最后一个 token 的 hidden state
    logits = self.output(h[:, [-1], :])
else:
    # 训练时需要所有位置的 logits 来计算 loss
    logits = self.output(h)
```

{WARNING}训练和推理的输出范围不同。训练需要所有位置的 logits，推理通常只需要最后一个位置的 logits。这个区别会影响显存、速度和后处理逻辑。{/WARNING}

本节小结：训练阶段关注整个序列的监督信号，推理阶段关注最后一个 token 的下一个词预测。

---

## 九、KV Cache 与 start_pos

### 1. 为什么需要 KV Cache

在自回归生成中，模型每次生成一个 token。如果每一步都重新计算整段上下文的 K/V，会非常浪费。

例如已经有：

```text
我 爱 自然
```

要生成下一个 token 时，如果重新计算“我”“爱”“自然”的 K/V，就会重复做大量工作。

KV Cache 的思想是：

- 历史 token 的 K/V 只计算一次
- 后续步骤复用缓存
- 新 token 只计算自己的 Q/K/V

{IMAGE:14}

### 2. start_pos 的作用

`start_pos` 表示当前 token 在整段上下文中的起始位置。

训练时：

```python
start_pos = 0
```

增量推理时：

```python
start_pos = 已经缓存的历史长度
```

例如：

- 第一次输入 prompt 长度为 10，位置是 0 到 9
- 下一步生成 token，`start_pos = 10`
- 再下一步，`start_pos = 11`

RoPE 和 KV Cache 都依赖这个位置。

### 3. Attention 中的缓存更新

简化逻辑如下：

```python
# xk, xv 是当前 step 新计算出来的 key/value
self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

# 取出从开头到当前位置的所有 key/value
keys = self.cache_k[:batch_size, :start_pos + seq_len]
values = self.cache_v[:batch_size, :start_pos + seq_len]
```

{KNOWLEDGE}KV Cache 是推理优化，不是训练必需品。训练时整段序列并行计算，一般不需要缓存历史 K/V。{/KNOWLEDGE}

本节小结：`start_pos` 是连接 RoPE 位置编码和 KV Cache 的关键变量，保证增量生成时位置连续、缓存正确。

---

## 十、最终 Norm 与输出层

### 1. Final RMSNorm

经过多层 Transformer Block 后，hidden state 会进入最终归一化层：

```python
h = self.norm(h)
```

RMSNorm 的核心公式：

$$
\text{RMSNorm}(x) =
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
\odot w
$$

其中：

- $d$ 是隐藏维度
- $\epsilon$ 防止除零
- $w$ 是可训练缩放参数

{IMAGE:15}

### 2. LM Head

输出层将隐藏向量映射到词表大小：

```python
logits = self.output(h)
```

其数学形式是：

$$
z = hW^T
$$

其中：

- $h \in \mathbb{R}^{B \times T \times C}$
- $W \in \mathbb{R}^{V \times C}$
- $z \in \mathbb{R}^{B \times T \times V}$

### 3. 权重共享

有些语言模型会让输入 embedding 和输出 head 共享权重：

```python
self.output.weight = self.tok_embeddings.weight
```

这样可以减少参数量，也常常有助于训练稳定。

{WARNING}如果使用权重共享，必须保证 embedding 维度和 output 输入维度一致，否则无法共享同一个权重矩阵。{/WARNING}

本节小结：最终 Norm 稳定输出分布，LM Head 把隐藏状态转换成词表预测。

---

## 十一、Loss 计算细节

### 1. 为什么要 reshape

`F.cross_entropy` 期望输入形状通常是：

$$
[N, C]
$$

标签形状是：

$$
[N]
$$

而语言模型 logits 是：

$$
[B, T, V]
$$

因此需要 reshape：

```python
logits = logits.view(-1, logits.size(-1))
targets = targets.view(-1)
```

变成：

$$
[B \times T, V]
$$

和：

$$
[B \times T]
$$

{IMAGE:16}

### 2. ignore_index

训练数据中可能有 padding 或不参与 loss 的位置，可以用 `ignore_index` 忽略：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1
)
```

如果某个 target 是 `-1`，该位置不会参与 loss 计算。

{IMPORTANT}`ignore_index` 对小模型训练很实用。它允许我们在 batch 内对齐不同长度样本，同时不让 padding 位置污染训练目标。{/IMPORTANT}

本节小结：loss 计算需要把 batch 和 sequence 维度展平，并通过 `ignore_index` 忽略无效标签。

---

## 十二、完整模型数据流回顾

### 1. 从输入到输出

完整 forward 可以按以下步骤理解：

```text
tokens
  ↓
Embedding
  ↓
Dropout
  ↓
TransformerBlock × n_layers
  ↓
RMSNorm
  ↓
Linear Head
  ↓
logits
  ↓
CrossEntropy Loss
```

{IMAGE:17}

对应张量形状：

```text
tokens: [B, T]
embedding: [B, T, C]
hidden: [B, T, C]
logits: [B, T, V]
loss: scalar
```

### 2. 关键状态变量

在完整模型中，几个变量尤其重要：

- `tokens`：输入 token
- `targets`：训练标签
- `h`：hidden states
- `freqs_cos/freqs_sin`：RoPE 位置参数
- `start_pos`：当前序列起始位置
- `logits`：词表预测分数
- `loss`：训练损失

{IMAGE:18}

{KNOWLEDGE}阅读 Transformer 代码时，最重要的是跟踪张量形状。只要每一步的 shape 是合理的，模型结构通常就容易理解。{/KNOWLEDGE}

本节小结：完整模型 forward 是一条清晰的数据流水线，理解 shape 变化是掌握实现的关键。

---

## 十三、训练样本的输入与标签构造

### 1. 语言模型的 shift 关系

训练语言模型时，输入和标签通常错开一位：

```text
原始文本 token:
[我, 爱, PyTorch, 从, 零, 实现, 大模型]

输入 tokens:
[我, 爱, PyTorch, 从, 零, 实现]

目标 targets:
[爱, PyTorch, 从, 零, 实现, 大模型]
```

也就是：

$$
\text{targets}_t = \text{tokens}_{t+1}
$$

{IMAGE:19}

### 2. 为什么这样训练

因为自回归模型的任务是：

$$
P(x_1, x_2, ..., x_T)
=
\prod_{t=1}^{T}
P(x_t \mid x_{<t})
$$

模型每个位置只能看到当前位置及之前的 token，然后预测下一个 token。

### 3. Causal Mask

为了防止模型偷看未来 token，Attention 中需要 causal mask。

对于长度为 4 的序列，mask 结构类似：

$$
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

这样第 $t$ 个位置只能关注 $0$ 到 $t$ 的历史信息。

{WARNING}如果 causal mask 写错，模型可能在训练中看到答案，loss 会异常好看，但推理能力会很差。{/WARNING}

本节小结：语言模型训练依赖输入和标签错位，并通过 causal mask 保证只能利用历史上下文。

---

## 十四、模型初始化与工程细节

### 1. 参数初始化

模型初始化会影响训练稳定性。常见做法包括：

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

然后在模型中调用：

```python
self.apply(self._init_weights)
```

{IMAGE:20}

### 2. Dropout 的训练与推理差异

Dropout 只在训练模式下生效：

```python
model.train()  # dropout 生效
model.eval()   # dropout 关闭
```

这意味着同一个输入在训练时可能有随机扰动，在推理时则保持确定性。

### 3. device 与 dtype

模型实际训练时还需要注意：

- 参数是否在 GPU 上
- 输入 tokens 是否在同一 device
- 混合精度训练时 dtype 是否一致
- KV Cache 是否提前分配到正确设备

```python
tokens = tokens.to(device)
model = model.to(device)
```

{WARNING}很多运行错误不是模型结构错，而是 device 或 dtype 不一致。例如参数在 GPU，输入还在 CPU，会直接报错。{/WARNING}

本节小结：工程细节决定模型能否稳定训练，初始化、dropout、device、dtype 都需要认真处理。

---

## 十五、完整 Model 的典型实现骨架

下面是一个更完整的结构化示例，展示 `Model` 如何组装各个模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # 输入 token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # 多层 Decoder Block
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        # 最终 RMSNorm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 语言模型输出头
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # RoPE 预计算
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len
        )

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 线性层和 embedding 使用正态分布初始化
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, start_pos=0):
        batch_size, seq_len = tokens.shape

        # token id -> hidden states
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 当前序列对应的 RoPE 参数
        freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]

        # 逐层通过 Transformer
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, start_pos)

        # 最终归一化
        h = self.norm(h)

        # 训练时输出所有位置；推理时只输出最后一个位置
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.output(h[:, [-1], :])
            loss = None

        return logits, loss
```

{IMAGE:21}

本节小结：完整 Model 类把 embedding、blocks、norm、head、loss、RoPE、推理优化统一封装起来，是训练脚本和推理脚本调用的核心对象。

---

## 十六、从 Model 到训练闭环

### 1. 一次训练 step

有了完整模型后，训练循环就可以写成：

```python
model.train()

logits, loss = model(tokens, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

完整训练过程包括：

1. 前向传播得到 loss
2. 反向传播计算梯度
3. 优化器更新参数
4. 重复大量 batch

### 2. 一次推理 step

推理时通常：

```python
model.eval()

with torch.no_grad():
    logits, _ = model(tokens, start_pos=start_pos)
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
```

如果使用采样，还可以加入：

- temperature
- top-k
- top-p
- repetition penalty

{IMAGE:22}

### 3. 训练和推理的统一接口

一个设计良好的 `Model` 类通常可以同时服务训练和推理：

- 有 `targets`：计算训练 loss
- 无 `targets`：返回推理 logits
- 有 `start_pos`：支持 KV Cache 增量生成

{IMPORTANT}好的模型实现应该让训练脚本和推理脚本都能自然调用，而不是为训练和推理写两套完全割裂的模型代码。{/IMPORTANT}

本节小结：Model 是训练闭环和推理闭环的共同核心，forward 的设计直接影响后续工程复杂度。

---

## 十七、常见易错点总结

### 1. shape 错误

最常见的问题是张量维度不匹配，例如：

- logits 没有 reshape 就传给 cross entropy
- hidden dim 和 head dim 不匹配
- RoPE 维度和 Q/K 维度不一致
- 推理时只取最后 token 后 shape 处理错误

### 2. start_pos 错误

增量推理中，如果 `start_pos` 没有正确递增，会导致：

- RoPE 位置错乱
- KV Cache 覆盖错误
- 输出质量明显下降

### 3. mask 错误

如果 causal mask 失效，模型训练 loss 可能下降很快，但推理表现异常。

### 4. device 错误

例如：

```text
Expected all tensors to be on the same device
```

通常说明输入、模型参数、RoPE 缓存或 KV Cache 不在同一个设备上。

{WARNING}Transformer 实现调试时，不要只看 loss。还要检查 shape、mask、位置编码、缓存和推理输出是否符合预期。{/WARNING}

{IMAGE:4}

本节小结：完整模型的问题往往不是单个模块错，而是多个模块之间接口没有对齐。

---

## 十八、本集关键收获

### Key Takeaways

1. MiniMind 的 `Model` 是一个 Decoder-only Transformer，由 embedding、多层 block、final norm 和 lm head 组成。
2. 输入 token 形状是 $[B,T]$，经过 embedding 后变为 $[B,T,C]$。
3. 模型输出 logits 形状是 $[B,T,V]$，其中 $V$ 是词表大小。
4. 训练时需要所有位置的 logits 来计算交叉熵 loss。
5. 推理时通常只需要最后一个位置的 logits，用于生成下一个 token。
6. RoPE 通过旋转 Q/K 注入位置信息，`start_pos` 对增量推理非常关键。
7. KV Cache 可以避免重复计算历史 token 的 K/V，是自回归推理加速的核心。
8. 完整模型实现的重点不是某一个模块，而是模块之间的 shape、位置、缓存、loss 逻辑都要对齐。

### 思考题

1. 为什么训练时可以一次性并行计算整个序列，而推理时通常要一个 token 一个 token 地生成？
2. 如果推理时 `start_pos` 始终写成 0，会对 RoPE 和 KV Cache 造成什么影响？
3. 训练 loss 很低但推理结果很差时，应该优先检查 causal mask、数据构造还是模型参数量？为什么？