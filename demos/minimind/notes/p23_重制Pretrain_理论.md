# 第23集: 重制Pretrain：理论

## 课程定位：为什么重制 Pretrain 理论

本集是 MiniMind 第 23/26 集，主题是“重制 Pretrain：理论”，重点围绕两个问题展开：

1. 预训练到底让模型学什么？
2. 训练时的损失函数如何把“预测下一个 token”变成可优化目标？

{IMAGE:1}

在大语言模型训练流程中，Pretrain 是最基础、最耗资源、也是最决定模型“语言能力底座”的阶段。它通常发生在 SFT、RLHF、DPO 等对齐阶段之前。预训练阶段并不要求模型学会“听话”，而是要求模型从海量文本中学习语言规律、知识分布、语法结构、上下文依赖和简单推理模式。

{IMPORTANT}预训练的核心目标不是直接训练一个聊天助手，而是训练一个能够根据上下文预测后续文本的通用语言模型。{/IMPORTANT}

本节小结：Pretrain 是大模型能力的底座，主要目标是通过大量文本训练模型的 next token prediction 能力。

---

## 预训练目标：Next Token Prediction

### 自回归语言模型的基本任务

MiniMind 属于典型的 decoder-only 自回归语言模型。所谓自回归，就是模型在生成第 $t$ 个 token 时，只能看到它之前的 token：

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

也就是说，一整句话的概率可以拆解成每一步“根据前文预测当前 token”的条件概率乘积。

{IMAGE:5}

例如文本序列为：

```text
我 喜欢 学习 人工 智能
```

训练时模型并不是一次性预测整句话，而是构造如下监督信号：

```text
输入: 我
目标: 喜欢

输入: 我 喜欢
目标: 学习

输入: 我 喜欢 学习
目标: 人工

输入: 我 喜欢 学习 人工
目标: 智能
```

在实际代码中，为了提升效率，通常会把整个 token 序列并行送入模型，通过 causal mask 保证每个位置只能看到它左侧的信息。

{KNOWLEDGE}虽然 Transformer 可以并行计算所有位置的 hidden states，但 decoder-only 模型会使用因果注意力掩码，禁止当前位置关注未来 token。{/KNOWLEDGE}

本节小结：Pretrain 的训练目标是让模型在每个位置根据历史上下文预测下一个 token。

---

## 输入与标签的错位关系

### shift by one

预训练数据通常是一段连续 token 序列：

$$
x = [x_0, x_1, x_2, ..., x_{T-1}]
$$

模型输入通常是：

$$
[x_0, x_1, ..., x_{T-2}]
$$

标签则是：

$$
[x_1, x_2, ..., x_{T-1}]
$$

也就是输入和目标相差一个位置。

{IMAGE:6}

这就是语言模型训练中常见的 shift 操作。模型在位置 $i$ 的输出 logits，要用来预测真实标签 $x_{i+1}$。

```python
# 假设 input_ids 是一个 batch 的 token 序列
# shape: [batch_size, seq_len]

inputs = input_ids[:, :-1]   # 去掉最后一个 token，作为输入
targets = input_ids[:, 1:]   # 去掉第一个 token，作为预测目标

# 模型输出 logits
# logits shape: [batch_size, seq_len - 1, vocab_size]
logits = model(inputs)

# 计算交叉熵损失
loss = cross_entropy(logits, targets)
```

{WARNING}易错点：不是用当前位置预测当前位置，而是用当前位置的模型输出预测下一个 token。输入和标签必须正确错位。{/WARNING}

本节小结：预训练标签来自原文本本身，通过输入与标签错位一位构造监督信号。

---

## 模型输出：Logits 与概率分布

### vocab_size 维分类问题

在每个位置，模型最终会输出一个长度为 `vocab_size` 的向量。这个向量称为 logits：

$$
z_t \in \mathbb{R}^{V}
$$

其中 $V$ 是词表大小。每一维对应一个 token 的未归一化分数。

{IMAGE:7}

例如词表中有：

```text
[我, 喜欢, 学习, 人工, 智能, 的, 是, ...]
```

当模型看到“我 喜欢”时，它可能对不同 token 给出不同分数：

```text
学习: 8.2
吃饭: 4.1
的: 1.7
是: 0.5
```

logits 本身还不是概率，需要经过 softmax 转换。

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

softmax 会把所有 token 的分数变成概率分布，并保证：

$$
\sum_{i=1}^{V} p_i = 1
$$

{IMAGE:8}

本节小结：模型每个位置输出 vocab_size 维 logits，经过 softmax 后表示下一个 token 的概率分布。

---

## 损失函数：交叉熵 Cross Entropy

### 从概率最大化到损失最小化

预训练希望模型给真实 token 分配尽可能高的概率。如果真实下一个 token 是“学习”，模型预测概率为：

```text
P(学习 | 我 喜欢) = 0.8
```

这说明模型预测较好；如果概率只有 0.01，则说明预测很差。

训练目标可以写成最大化真实序列概率：

$$
\max_\theta \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})
$$

为了方便优化，通常取 log：

$$
\max_\theta \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

深度学习训练一般最小化 loss，所以变成负对数似然：

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

如果对 token 数取平均，就是常见的语言模型交叉熵损失：

$$
\mathcal{L} = - \frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

{IMAGE:9}

{IMPORTANT}交叉熵损失本质上是在惩罚模型对正确 token 分配的概率太低。正确 token 的概率越高，loss 越小。{/IMPORTANT}

本节小结：Pretrain 的 loss 通常是 next token prediction 上的交叉熵，也等价于负对数似然。

---

## 单个位置的交叉熵计算

### one-hot 标签与预测分布

假设词表大小为 5，真实 token 是第 2 类，标签可以写成 one-hot：

$$
y = [0, 0, 1, 0, 0]
$$

模型预测概率为：

$$
p = [0.05, 0.10, 0.80, 0.03, 0.02]
$$

交叉熵为：

$$
H(y, p) = - \sum_i y_i \log p_i
$$

由于 one-hot 中只有真实类别对应位置为 1，所以公式简化为：

$$
H(y, p) = -\log p_{\text{target}}
$$

也就是：

$$
-\log(0.80)
$$

如果模型只给真实 token 0.01 的概率：

$$
-\log(0.01)
$$

损失会明显变大。

{IMAGE:10}

```python
import torch
import torch.nn.functional as F

# 单个位置的 logits，假设 vocab_size = 5
logits = torch.tensor([[1.0, 2.0, 4.0, 0.5, 0.1]])

# 真实类别索引，表示第 2 号 token 是正确答案
target = torch.tensor([2])

# PyTorch 的 cross_entropy 内部会自动做 log_softmax + NLLLoss
loss = F.cross_entropy(logits, target)

print(loss)
```

{WARNING}易错点：`F.cross_entropy` 的输入应该是 logits，而不是 softmax 之后的概率。PyTorch 会在内部处理 softmax 相关计算。{/WARNING}

本节小结：单个 token 的交叉熵就是正确 token 预测概率的负对数。

---

## Batch 与序列维度上的损失计算

### 从二维分类扩展到三维语言模型输出

语言模型输出通常是三维张量：

$$
\text{logits shape} = [B, T, V]
$$

其中：

- $B$：batch size
- $T$：sequence length
- $V$：vocab size

标签通常是：

$$
\text{targets shape} = [B, T]
$$

每个 batch、每个时间位置都有一个正确 token id。

{IMAGE:11}

PyTorch 的 `cross_entropy` 通常要求分类维度在第二维，因此语言模型训练中常见两种写法。

写法一：展平 batch 和 seq 维度：

```python
import torch.nn.functional as F

# logits: [batch_size, seq_len, vocab_size]
# targets: [batch_size, seq_len]

B, T, V = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, V),     # [B*T, V]
    targets.reshape(B * T)        # [B*T]
)
```

写法二：调换维度：

```python
# logits: [B, T, V] -> [B, V, T]
loss = F.cross_entropy(
    logits.transpose(1, 2),
    targets
)
```

在 MiniMind 这类从零实现项目中，第一种展平写法更直观，方便理解每个 token 位置都是一个分类任务。

本节小结：语言模型损失是在 batch 内所有 token 位置上计算交叉熵并取平均。

---

## Causal Mask 与“不能偷看答案”

### 为什么需要 mask

Transformer 的 self-attention 默认每个 token 都可以看到序列中所有 token。如果不加限制，模型在训练时可能直接看到未来 token，相当于考试时偷看答案。

{IMAGE:12}

例如输入：

```text
我 喜欢 学习 人工 智能
```

如果位置“喜欢”能够看到后面的“学习”，那么它预测“学习”就没有意义了。训练出来的模型在推理时没有未来信息，效果会崩掉。

因此 decoder-only 模型必须使用 causal mask：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

含义是：

- 当前位置可以看自己和过去位置
- 当前位置不能看未来位置
- 被 mask 的位置在 softmax 后概率接近 0

{IMAGE:13}

```python
import torch

seq_len = 5

# 上三角为 True，表示未来位置需要被屏蔽
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

print(mask)
```

本节小结：Causal mask 保证 next token prediction 的训练条件与推理条件一致。

---

## 预训练数据：文本如何变成 token

### tokenizer 的作用

模型不能直接处理汉字、英文单词或字符串。文本需要先经过 tokenizer 转换成 token id。

{IMAGE:14}

流程如下：

```text
原始文本 -> tokenizer -> token ids -> embedding -> Transformer -> logits
```

例如：

```python
text = "我喜欢学习人工智能"

# 示例：实际 token id 由具体 tokenizer 决定
input_ids = tokenizer.encode(text)

print(input_ids)
```

在预训练阶段，数据通常不是一问一答格式，而是大量自然文本、代码、百科、网页、书籍等内容。模型通过这些文本学习语言统计规律。

{KNOWLEDGE}预训练数据质量会显著影响模型能力。高质量、去重、低噪声、多样化的数据通常比单纯堆数量更重要。{/KNOWLEDGE}

本节小结：文本需要先被 tokenizer 编码成 token id，预训练数据一般是大规模连续文本。

---

## Perplexity：语言模型损失的直观指标

### 从 loss 到困惑度

交叉熵 loss 虽然可优化，但不够直观。语言模型中常用 perplexity，中文常译为“困惑度”。

如果平均交叉熵为：

$$
\mathcal{L}
$$

则困惑度为：

$$
\text{PPL} = e^{\mathcal{L}}
$$

直观理解：PPL 越低，说明模型对下一个 token 越不困惑，预测越确定。

{IMAGE:15}

例如：

$$
\mathcal{L} = 2.0
$$

则：

$$
\text{PPL} = e^2 \approx 7.39
$$

可以粗略理解为模型平均每一步像是在约 7 个候选 token 中犹豫。

{WARNING}PPL 只能在相同 tokenizer、相似数据分布、相同评估方式下比较。不同词表或不同数据集上的 PPL 不宜直接比较。{/WARNING}

本节小结：PPL 是交叉熵 loss 的指数形式，常用于评估语言模型预测能力。

---

## Pretrain 与 SFT 的区别

### 数据形式不同

预训练数据通常是普通文本：

```text
大语言模型是一种基于深度学习的生成式模型……
```

SFT 数据通常是指令格式：

```text
用户：请解释什么是交叉熵。
助手：交叉熵是一种衡量两个概率分布差异的损失函数……
```

{IMAGE:16}

### 训练目标相似但侧重点不同

Pretrain 和 SFT 本质上都可以使用 next token prediction 和交叉熵损失，但它们的目的不同：

- Pretrain：学习通用语言建模能力和世界知识
- SFT：学习按照指令、对话格式和人类偏好输出
- Pretrain 数据规模通常更大
- SFT 数据质量和格式更关键

有些 SFT 训练会只对 assistant 部分计算 loss，而不对 user prompt 部分计算 loss。这可以通过 label mask 实现，把不需要计算 loss 的位置标记为 `ignore_index`。

```python
# targets 中不参与 loss 的位置设为 -100
# PyTorch cross_entropy 默认 ignore_index=-100
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=-100
)
```

本节小结：Pretrain 和 SFT 都可用交叉熵，但 Pretrain 学基础能力，SFT 学指令跟随与对话风格。

---

## MiniMind 中的训练逻辑抽象

### 一个最小训练步骤

{IMAGE:17}

一个典型的预训练 step 可以抽象为：

```python
import torch
import torch.nn.functional as F

def pretrain_step(model, input_ids, optimizer):
    """
    input_ids: [batch_size, seq_len]
    """
    # 1. 构造输入和标签
    x = input_ids[:, :-1]      # [B, T-1]
    y = input_ids[:, 1:]       # [B, T-1]

    # 2. 前向传播
    logits = model(x)          # [B, T-1, vocab_size]

    # 3. 计算交叉熵损失
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    )

    # 4. 反向传播与参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

这个训练步骤背后的数学含义是：

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

其中：

- $\theta$ 是模型参数
- $\eta$ 是学习率
- $\mathcal{L}$ 是交叉熵损失
- $\nabla_\theta \mathcal{L}$ 是损失对参数的梯度

{IMAGE:18}

本节小结：MiniMind 的预训练实现可以理解为“错位构造标签、前向算 logits、交叉熵算 loss、反向更新参数”。

---

## 为什么 loss 会下降

### 梯度优化的直观理解

训练初期，模型参数随机初始化，对下一个 token 的预测接近随机。若词表大小为 $V$，随机预测时真实 token 的概率大约是：

$$
\frac{1}{V}
$$

此时 loss 大约为：

$$
-\log \frac{1}{V} = \log V
$$

随着训练进行，模型逐渐学习到上下文规律。例如看到“人工”后，预测“智能”的概率会提高；看到“机器”后，预测“学习”的概率也会提高。

{IMAGE:19}

当真实 token 概率提高时：

$$
P_\theta(x_t \mid x_{<t}) \uparrow
$$

对应损失下降：

$$
-\log P_\theta(x_t \mid x_{<t}) \downarrow
$$

这就是预训练 loss 下降的本质。

本节小结：loss 下降意味着模型给真实下一个 token 分配了更高概率。

---

## 常见实现细节与坑点

### logits、labels 的 shape

{IMAGE:20}

语言模型训练中最常见错误之一是 shape 不匹配。应确认：

```text
logits: [B, T, V]
labels: [B, T]
```

展平后：

```text
logits: [B*T, V]
labels: [B*T]
```

### labels 的 dtype

`cross_entropy` 要求 labels 是类别索引，通常为 `torch.long`，不是 one-hot，也不是 float。

```python
targets = targets.long()
```

### 不要重复 softmax

错误写法：

```python
probs = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, targets)
```

正确写法：

```python
loss = F.cross_entropy(logits, targets)
```

### 注意 pad token

如果 batch 中存在 padding，需要避免 pad 位置参与 loss：

```python
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=pad_token_id
)
```

或者将 pad 标签替换成 `-100`：

```python
targets[targets == pad_token_id] = -100
```

{WARNING}如果 padding token 参与 loss，模型可能会错误地学习预测大量 padding，从而污染训练信号。{/WARNING}

本节小结：预训练实现中最重要的工程细节是 shift、shape、dtype、ignore_index 和不要手动 softmax 后再交叉熵。

---

## 从理论到本集结尾

{IMAGE:21}

本集最后回到重制 Pretrain 的理论主线：我们并不是神秘地“喂数据让模型变聪明”，而是在做一个极其明确的监督学习任务：

$$
\text{给定前文，预测下一个 token}
$$

模型输出的是整个词表上的概率分布；真实答案来自原始文本向右平移一位；损失函数是交叉熵；训练过程通过反向传播不断提高真实 token 的预测概率。

{IMAGE:3}

{IMAGE:4}

本节小结：Pretrain 理论可以被压缩为一个闭环：文本 token 化、错位构造标签、自回归预测、交叉熵优化、梯度下降更新。

---

## Key Takeaways

1. 预训练的核心任务是 next token prediction，即根据历史上下文预测下一个 token。
2. Decoder-only 模型必须使用 causal mask，避免训练时看到未来信息。
3. 模型每个位置输出 vocab_size 维 logits，表示对所有 token 的打分。
4. 交叉熵损失等价于真实 token 概率的负对数，真实 token 概率越高，loss 越低。
5. PyTorch 中 `F.cross_entropy` 输入应为 logits，不需要提前 softmax。
6. 语言模型训练时需要特别注意输入和标签 shift、张量 shape、padding mask 与 `ignore_index`。
7. Pretrain 和 SFT 都可能使用交叉熵，但 Pretrain 学语言基础能力，SFT 学指令跟随和对话格式。

---

## 思考题

1. 如果训练 decoder-only 模型时不加 causal mask，会导致什么训练-推理不一致问题？
2. 为什么 `F.cross_entropy` 要输入 logits，而不是 softmax 后的概率？
3. 在 SFT 中，为什么有时只对 assistant 回复部分计算 loss，而不对 user prompt 计算 loss？