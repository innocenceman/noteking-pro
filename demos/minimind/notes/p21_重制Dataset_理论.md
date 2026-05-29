# 第21集: 重制Dataset：理论

## 课程定位与本集目标

{IMAGE:11}

本集是 MiniMind 课程第 21 集，主题是“重制 Dataset：理论”，重点讨论大模型预训练阶段的数据格式设计，以及 tokenizer 在整个训练链路中的作用。

前面课程已经搭建了模型结构、训练流程和若干工程组件。本集开始进入一个非常关键但容易被低估的部分：**数据如何进入模型**。

对于自回归语言模型来说，训练目标看似简单：给定前面的 token，预测下一个 token。但在工程实现中，原始文本必须经过一系列转换：

1. 原始语料收集与清洗
2. 文本切分与格式组织
3. tokenizer 编码为 token id
4. 拼接、截断、补齐为固定长度序列
5. 构造输入 `input_ids` 与训练标签 `labels`
6. 送入模型计算 loss

{IMPORTANT}预训练 Dataset 的核心任务不是“读文件”，而是把大量文本稳定、连续、高效地转换成语言模型可以学习的 token 序列。{/IMPORTANT}

**本节小结：**  
本集的核心是理解预训练数据格式和 tokenizer 的角色，为后续手写 Dataset 代码打基础。

---

## 预训练数据的基本形式

{IMAGE:1}

大语言模型的预训练通常使用大规模无监督文本数据。所谓“无监督”，并不是没有训练目标，而是不需要人工标注的问答、分类或标签数据。

典型原始语料可能长这样：

```text
今天天气很好，我们一起去公园散步。
MiniMind 是一个从零实现的小型语言模型项目。
语言模型通过预测下一个 token 来学习文本规律。
```

这些文本不能直接输入神经网络。模型只能处理数字张量，因此必须先经过 tokenizer 转换为整数 id。

```python
text = "语言模型通过预测下一个 token 来学习文本规律。"

# tokenizer.encode 会把字符串转换成 token id 序列
ids = tokenizer.encode(text)

print(ids)
# 示例输出：[123, 456, 789, 1024, ...]
```

对于 GPT 类自回归模型，训练样本通常不是“输入一句话，输出一个标签”，而是：

```text
输入:  [t0, t1, t2, t3, ... , tn-1]
目标:  [t1, t2, t3, t4, ... , tn]
```

也就是每个位置都在预测下一个 token。

数学上可以写成：

$$
P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, ..., x_{i-1})
$$

其中：

- $x_i$ 表示第 $i$ 个 token
- $P(x_i \mid x_{<i})$ 表示模型根据前文预测当前 token 的概率
- 训练目标是最大化真实文本序列的似然，等价于最小化交叉熵 loss

{KNOWLEDGE}自回归语言模型的训练样本天然来自文本本身：前面的 token 是输入，后面的 token 是监督信号。{/KNOWLEDGE}

**本节小结：**  
预训练数据的核心格式是 token 序列，模型通过“当前位置预测下一个 token”完成无监督学习。

---

## Dataset 为什么需要重制

{IMAGE:12}

在小规模实验中，可以直接逐行读取文本，然后 tokenize。但是当训练数据变大后，这种方式会出现几个问题：

1. 每次训练都重复 tokenize，效率低
2. 文本长度差异很大，batch 内 padding 浪费严重
3. 单条文本太短，无法充分利用上下文长度
4. 数据格式不统一，后续扩展 SFT、RLHF、评测数据会混乱
5. 随机读取、断点恢复和多进程加载不够稳定

因此需要重新设计 Dataset，使它更适合预训练任务。

常见思路是：  
先把大量文本 tokenize 成连续 token 流，然后按照固定长度 `max_seq_len` 切块。

例如：

```text
原始 token 流:
[12, 35, 98, 77, 41, 62, 19, 23, 88, 90, 51, 14, ...]

如果 max_seq_len = 6

样本 1:
input_ids = [12, 35, 98, 77, 41, 62]
labels    = [35, 98, 77, 41, 62, 19]

样本 2:
input_ids = [19, 23, 88, 90, 51, 14]
labels    = [23, 88, 90, 51, 14, ...]
```

这里有一个关键点：为了构造 labels，通常需要读取 `max_seq_len + 1` 个 token，然后错位一位。

```python
def build_sample(token_ids, start, max_seq_len):
    # 取 max_seq_len + 1 个 token，用于构造输入和标签
    chunk = token_ids[start: start + max_seq_len + 1]

    input_ids = chunk[:-1]  # 前 max_seq_len 个 token
    labels = chunk[1:]      # 后 max_seq_len 个 token

    return input_ids, labels
```

{WARNING}很多初学者会把 input_ids 和 labels 设成完全一样，这会破坏 next-token prediction 的训练目标。labels 应该相对 input_ids 左移一位。{/WARNING}

**本节小结：**  
重制 Dataset 的主要目的是提升数据效率、统一格式，并正确构造自回归训练样本。

---

## tokenizer 的作用

{IMAGE:13}

tokenizer 是文本和模型之间的桥梁。模型并不理解“字”“词”或“句子”，它只处理 token id。

tokenizer 主要完成两个方向的转换：

```text
文本 -> token id 序列
token id 序列 -> 文本
```

对应接口通常是：

```python
ids = tokenizer.encode("你好，MiniMind")
text = tokenizer.decode(ids)
```

在大模型中，tokenizer 的设计会直接影响：

1. 词表大小 `vocab_size`
2. 序列长度
3. 训练效率
4. 中文、英文、代码、符号的压缩率
5. 模型生成文本的质量
6. OOV 问题，即遇到未知字符时如何处理

{IMAGE:14}

如果 tokenizer 粒度太细，例如按字符切分，中文可能还好，但英文和代码会变长：

```text
"tokenizer" -> ["t", "o", "k", "e", "n", "i", "z", "e", "r"]
```

如果粒度太粗，例如按词切分，又会遇到大量未登录词：

```text
"MiniMindTokenizerV2" -> 词表中可能不存在
```

现代大模型常用 BPE、WordPiece、SentencePiece 等子词算法，在字符和词之间取得平衡：

```text
"tokenizer" -> ["token", "izer"]
"unbelievable" -> ["un", "believ", "able"]
```

{KNOWLEDGE}子词 tokenizer 的优势是：既能复用高频片段，又能拆解低频词，减少未知词问题。{/KNOWLEDGE}

**本节小结：**  
tokenizer 决定了文本如何被数字化，也决定了模型看到的“语言基本单位”。

---

## token、词表与 embedding 的关系

{IMAGE:2}

tokenizer 输出的是整数 id，例如：

```python
text = "我喜欢机器学习"
ids = tokenizer.encode(text)
# 假设输出: [101, 524, 873, 902, 135]
```

这些 id 本身没有语义。模型首先会通过 embedding 层把 id 映射成向量：

```python
import torch
import torch.nn as nn

vocab_size = 6400
hidden_size = 512

embedding = nn.Embedding(vocab_size, hidden_size)

input_ids = torch.tensor([[101, 524, 873, 902, 135]])
x = embedding(input_ids)

print(x.shape)
# torch.Size([1, 5, 512])
```

这里：

- `vocab_size` 是 tokenizer 词表大小
- `hidden_size` 是每个 token 的向量维度
- 输入 shape 是 `[batch_size, seq_len]`
- embedding 输出 shape 是 `[batch_size, seq_len, hidden_size]`

从数学上看，embedding 相当于一个矩阵：

$$
E \in \mathbb{R}^{V \times d}
$$

其中：

- $V$ 是词表大小
- $d$ 是隐藏层维度
- 每个 token id 都对应矩阵中的一行向量

如果 token id 为 $i$，则其 embedding 为：

$$
x_i = E[i]
$$

{IMPORTANT}tokenizer 的词表大小必须和模型配置中的 vocab_size 对齐，否则 embedding 层会出现越界或权重维度不匹配。{/IMPORTANT}

**本节小结：**  
tokenizer 负责产生 token id，embedding 层负责把 token id 变成可训练的稠密向量。

---

## 预训练数据格式设计

{IMAGE:3}

一个预训练 Dataset 通常需要返回字典格式，方便训练循环统一处理：

```python
{
    "input_ids": Tensor[int64],
    "labels": Tensor[int64]
}
```

在 PyTorch 中，一个简化的 Dataset 可以这样设计：

```python
from torch.utils.data import Dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, token_ids, max_seq_len):
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len

        # 每个样本需要 max_seq_len + 1 个 token
        self.num_samples = (len(token_ids) - 1) // max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1

        chunk = self.token_ids[start:end]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
```

注意 `__getitem__` 中读取的是 `max_seq_len + 1` 个 token，因为要构造错位标签。

如果 `max_seq_len = 512`，则：

```text
chunk 长度: 513
input_ids 长度: 512
labels 长度: 512
```

{WARNING}如果最后剩余 token 不足 `max_seq_len + 1`，通常应丢弃或特殊处理，避免 batch 内 shape 不一致。{/WARNING}

**本节小结：**  
预训练 Dataset 的输出应直接服务于训练循环，最常见字段是 `input_ids` 和 `labels`。

---

## 文本拼接与 EOS 标记

{IMAGE:4}

在预训练中，经常会把多篇文本拼接成一个长 token 流。为了让模型知道文本边界，通常会插入特殊 token，例如 EOS。

```text
文本 A + EOS + 文本 B + EOS + 文本 C + EOS
```

EOS 是 End Of Sequence 的缩写，表示一段文本结束。它的作用包括：

1. 告诉模型一段文本结束
2. 帮助模型学习停止生成
3. 避免不同文档之间完全无边界地混合
4. 在生成阶段可作为停止条件

示例：

```python
texts = [
    "今天天气很好。",
    "我们正在学习 MiniMind。",
    "预训练数据需要先经过 tokenizer。"
]

all_ids = []

for text in texts:
    ids = tokenizer.encode(text)
    ids.append(tokenizer.eos_token_id)
    all_ids.extend(ids)
```

如果没有 EOS，模型可能会把两篇无关文本当成连续上下文：

```text
文章1最后一句。文章2第一句。
```

这会引入一些噪声。虽然模型可以容忍大量噪声，但清晰的数据边界通常更有利。

{IMPORTANT}EOS 不只是生成时的停止符，也是预训练数据组织中的文档边界信号。{/IMPORTANT}

**本节小结：**  
拼接文本时建议插入 EOS，让模型学习文本结束和文档边界。

---

## max_seq_len 与上下文窗口

{IMAGE:5}

`max_seq_len` 表示单个训练样本的最大 token 长度，也对应模型能处理的上下文窗口长度。

例如：

```python
max_seq_len = 512
```

表示每个样本包含 512 个输入 token，模型在训练时最多基于前 511 个 token 预测后续 token。

对于 Transformer 来说，注意力复杂度约为：

$$
O(n^2)
$$

其中 $n$ 是序列长度。也就是说，序列长度翻倍，注意力计算量大约变成四倍。

因此，`max_seq_len` 的选择需要权衡：

| max_seq_len | 优点 | 缺点 |
|---|---|---|
| 较短 | 训练快，显存占用低 | 难学习长上下文 |
| 较长 | 能学习长距离依赖 | 显存和计算开销高 |
| 适中 | 工程上更稳定 | 需要结合模型规模调参 |

对于 MiniMind 这类小模型，常见做法是先使用相对较短的上下文长度，例如 256、512 或 1024，保证训练稳定和实验速度。

{WARNING}更长的 max_seq_len 不一定带来更好的效果。如果模型容量、数据质量和训练步数不足，长上下文可能只是增加计算成本。{/WARNING}

**本节小结：**  
`max_seq_len` 决定了训练样本长度，也直接影响显存、速度和模型上下文能力。

---

## input_ids 与 labels 的错位关系

{IMAGE:6}

自回归语言模型训练的核心是错位预测。

假设 token 序列为：

```text
[我, 喜欢, 学习, 大模型, EOS]
```

对应 token id：

```text
[10, 20, 30, 40, 2]
```

训练样本应构造为：

```text
input_ids = [10, 20, 30, 40]
labels    = [20, 30, 40, 2]
```

也就是说：

- 输入位置 0：看到“我”，预测“喜欢”
- 输入位置 1：看到“我 喜欢”，预测“学习”
- 输入位置 2：看到“我 喜欢 学习”，预测“大模型”
- 输入位置 3：看到“我 喜欢 学习 大模型”，预测“EOS”

模型输出 logits 的 shape 通常为：

```text
[batch_size, seq_len, vocab_size]
```

labels 的 shape 为：

```text
[batch_size, seq_len]
```

交叉熵会对每个位置计算预测误差：

$$
\mathcal{L}
= - \frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_{\le i})
$$

其中：

- $N$ 是参与 loss 计算的 token 数
- $y_i$ 是当前位置要预测的目标 token
- $P(y_i)$ 是模型给目标 token 分配的概率

代码示意：

```python
import torch
import torch.nn.functional as F

# logits: [batch_size, seq_len, vocab_size]
# labels: [batch_size, seq_len]

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
    labels.view(-1)                    # [batch_size * seq_len]
)
```

{IMPORTANT}预训练不是只预测最后一个 token，而是序列中每个位置都参与 next-token prediction。{/IMPORTANT}

**本节小结：**  
`input_ids` 和 `labels` 必须错位一格，模型在每个位置预测下一个 token。

---

## padding 与 ignore_index

{IMAGE:7}

如果不同样本长度不一致，通常需要 padding 到同样长度，才能组成 batch。

例如：

```text
样本 A: [10, 20, 30, 2]
样本 B: [11, 21, 2]
```

padding 后：

```text
样本 A: [10, 20, 30, 2]
样本 B: [11, 21, 2, 0]
```

其中 `0` 可能是 `pad_token_id`。

但是 padding token 不应该参与 loss，否则模型会被迫学习预测无意义的 PAD。常见做法是把 labels 中 padding 位置设为 `-100`，因为 PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`。

```python
pad_token_id = 0
ignore_index = -100

input_ids = torch.tensor([11, 21, 2, pad_token_id])
labels = torch.tensor([21, 2, pad_token_id, ignore_index])

# 更常见的处理：凡是 padding 位置，label 设置为 -100
labels[input_ids == pad_token_id] = ignore_index
```

不过，对于预训练的连续 token block，如果每个样本都固定长度，通常可以减少 padding 的使用。这样训练效率更高。

{WARNING}padding token 可以出现在 input_ids 中，但 padding 对应的 labels 通常必须设为 ignore_index，否则 loss 会被污染。{/WARNING}

**本节小结：**  
padding 是 batch 对齐手段，但 padding 位置不应参与训练损失。

---

## 预训练 Dataset 与 SFT Dataset 的区别

{IMAGE:8}

本集重点是预训练数据，但理解它和 SFT 数据的区别很重要。

预训练数据通常是普通文本：

```json
{"text": "语言模型通过预测下一个 token 来学习文本规律。"}
```

SFT 数据通常是指令问答格式：

```json
{
  "instruction": "解释什么是 tokenizer",
  "input": "",
  "output": "tokenizer 是把文本转换成 token id 的工具。"
}
```

二者训练目标也不同：

| 类型 | 数据形态 | 目标 |
|---|---|---|
| 预训练 | 大规模连续文本 | 学习语言规律、知识和通用表示 |
| SFT | 指令-回答数据 | 学习按照指令输出 |
| 偏好训练 | chosen/rejected | 学习人类偏好 |

对于预训练，通常所有有效 token 都参与 loss。  
对于 SFT，很多时候只希望回答部分参与 loss，指令部分作为上下文，不计算监督损失。

```text
SFT:
[用户问题 tokens] [助手回答 tokens]
 labels:
[-100, -100, -100] [回答 token labels]
```

{KNOWLEDGE}预训练让模型“会说话、懂语言”，SFT 让模型“按指令说话”。{/KNOWLEDGE}

**本节小结：**  
预训练 Dataset 关注连续文本建模，SFT Dataset 更关注对话格式和 loss mask。

---

## 数据文件格式：jsonl 与 bin

{IMAGE:9}

预训练语料常见保存格式包括：

1. `.txt`
2. `.json`
3. `.jsonl`
4. `.bin`
5. `.npy`
6. memory-mapped 文件

对于教学项目，`jsonl` 很常见：

```json
{"text": "第一条训练文本。"}
{"text": "第二条训练文本。"}
{"text": "第三条训练文本。"}
```

读取示例：

```python
import json

texts = []

with open("pretrain.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])
```

如果数据量较小，可以在初始化 Dataset 时全部 tokenize。  
如果数据量较大，更推荐预处理成 token id 文件，例如 `.bin`，训练时直接读取整数数组。

```python
import numpy as np

# 保存 token id
arr = np.array(all_token_ids, dtype=np.uint16)
arr.tofile("pretrain_ids.bin")

# 读取 token id
ids = np.fromfile("pretrain_ids.bin", dtype=np.uint16)
```

选择 `uint16` 还是 `uint32` 取决于词表大小：

$$
2^{16} = 65536
$$

如果 `vocab_size <= 65536`，可以用 `uint16` 节省空间。  
如果词表更大，则需要 `uint32`。

{WARNING}保存 token id 时，数据类型必须能覆盖最大 token id。否则会发生溢出，导致训练数据损坏。{/WARNING}

**本节小结：**  
`jsonl` 适合原始文本存储，二进制 token 文件适合高效训练读取。

---

## tokenizer 训练与词表规模

{IMAGE:15}

如果项目使用自定义 tokenizer，就需要先在语料上训练 tokenizer。训练 tokenizer 的关键参数之一是词表大小 `vocab_size`。

词表太小：

- 序列变长
- 一个词被拆得很碎
- 训练和推理 token 数增加

词表太大：

- embedding 参数增多
- 输出层 softmax 更大
- 小模型可能浪费容量
- 低频 token 学不充分

embedding 参数量约为：

$$
\text{Params}_{emb} = V \times d
$$

如果：

- $V = 6400$
- $d = 512$

则：

$$
6400 \times 512 = 3,276,800
$$

仅 embedding 就有约 327 万参数。

如果词表扩大到 32000：

$$
32000 \times 512 = 16,384,000
$$

embedding 参数会明显增加。

{IMAGE:16}

对于 MiniMind 这类小模型，选择较小词表是合理的，因为模型容量有限，需要控制参数规模与训练成本。

{IMPORTANT}词表大小不是越大越好，而要和模型规模、语料语言、训练成本匹配。{/IMPORTANT}

**本节小结：**  
tokenizer 的词表大小会影响序列长度、参数量、训练速度和模型效果。

---

## 特殊 token 的设计

{IMAGE:17}

tokenizer 通常需要若干特殊 token，例如：

| token | 作用 |
|---|---|
| `<pad>` | padding 补齐 |
| `<unk>` | 未知字符 |
| `<bos>` | 序列开始 |
| `<eos>` | 序列结束 |
| `<s>` / `</s>` | 一些模型中的开始与结束标记 |

在预训练中，最常用的是 EOS。BOS 是否使用取决于模型设计和数据格式。

示例配置：

```python
special_tokens = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
}
```

特殊 token 需要被加入词表，并有稳定的 id：

```python
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
```

{WARNING}不要在不同阶段随意更换 tokenizer 或特殊 token id。模型 embedding 已经和 token id 绑定，更换后语义会错位。{/WARNING}

**本节小结：**  
特殊 token 是数据格式协议的一部分，必须在 tokenizer、Dataset 和模型配置之间保持一致。

---

## Dataset 的工程效率问题

{IMAGE:18}

一个好的 Dataset 不仅要逻辑正确，还要训练时足够快。

常见优化方向包括：

1. 预先 tokenize，而不是训练时重复 tokenize
2. 使用连续 token 数组，减少 Python 对象开销
3. 固定长度切块，减少 padding
4. 使用 `DataLoader` 多进程加载
5. 使用二进制格式减少解析成本
6. 避免在 `__getitem__` 中做复杂字符串处理

低效写法：

```python
def __getitem__(self, idx):
    # 每次取样本时才读取文本和 tokenize，训练会很慢
    text = self.texts[idx]
    ids = tokenizer.encode(text)
    return ids
```

更高效的思路：

```python
def __getitem__(self, idx):
    # 直接从 token id 数组切片
    start = idx * self.max_seq_len
    chunk = self.token_ids[start:start + self.max_seq_len + 1]
    return build_sample(chunk)
```

如果数据量非常大，可以考虑 `np.memmap`：

```python
import numpy as np

ids = np.memmap(
    "pretrain_ids.bin",
    dtype=np.uint16,
    mode="r"
)
```

这样可以不用一次性把全部数据读入内存。

{KNOWLEDGE}Dataset 的性能瓶颈常常不是模型本身，而是 CPU 侧数据读取、解析和 tokenize。{/KNOWLEDGE}

**本节小结：**  
高效 Dataset 应尽量让训练阶段只做轻量切片和张量转换。

---

## attention mask 是否必须

{IMAGE:19}

在很多 Transformer 代码中，输入除了 `input_ids` 和 `labels`，还会包含 `attention_mask`。

```python
{
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": attention_mask
}
```

`attention_mask` 通常用于区分真实 token 和 padding token：

```text
input_ids:      [11, 21, 2, 0, 0]
attention_mask: [ 1,  1, 1, 0, 0]
```

但是在 GPT 类自回归模型中，还存在 causal mask，即下三角注意力 mask，用来防止当前位置看到未来 token。

causal mask 的形式大致是：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

含义是：

- 当前位置可以看自己和之前的 token
- 不能看未来 token
- softmax 前加上 $-\infty$，未来位置概率变为 0

如果预训练样本都是固定长度且没有 padding，则 Dataset 可以不返回 `attention_mask`，模型内部只需要 causal mask。

{WARNING}padding mask 和 causal mask 不是同一个东西。padding mask 屏蔽无效 token，causal mask 屏蔽未来 token。{/WARNING}

**本节小结：**  
预训练固定长度样本通常不强依赖 attention_mask，但自回归模型必须使用 causal mask。

---

## 数据质量与清洗

{IMAGE:20}

数据格式正确不等于数据质量好。预训练语料中可能存在：

1. 重复文本
2. 乱码
3. HTML 标签
4. 广告内容
5. 低质量灌水文本
6. 隐私信息
7. 过短或无意义样本
8. 语言混杂比例失衡

清洗策略包括：

```python
def clean_text(text):
    text = text.strip()

    # 去掉过短文本
    if len(text) < 10:
        return None

    # 简单示例：过滤明显无效内容
    bad_keywords = ["点击下载", "广告合作", "备用网址"]

    for word in bad_keywords:
        if word in text:
            return None

    return text
```

对于小模型来说，数据质量尤其重要。因为模型容量有限，低质量数据会更明显地占用学习能力。

{IMPORTANT}小模型更依赖高质量数据。对于 MiniMind 这类项目，干净、结构稳定的数据通常比盲目扩大数据量更重要。{/IMPORTANT}

**本节小结：**  
预训练数据不仅要能被读取，还要尽量干净、去重、格式统一。

---

## 从文本到训练 batch 的完整流程

{IMAGE:21}

整个数据链路可以概括为：

```text
原始文本
  -> 清洗
  -> tokenizer.encode
  -> 插入 eos
  -> 拼接为连续 token 流
  -> 按 max_seq_len + 1 切块
  -> 构造 input_ids 和 labels
  -> DataLoader 组成 batch
  -> 模型 forward
  -> 计算 cross entropy loss
```

完整示意代码：

```python
import json
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len
        self.token_ids = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["text"].strip()

                if not text:
                    continue

                ids = tokenizer.encode(text)
                ids.append(tokenizer.eos_token_id)

                self.token_ids.extend(ids)

        self.num_samples = (len(self.token_ids) - 1) // max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        chunk = self.token_ids[start:start + self.max_seq_len + 1]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
```

这个版本适合教学理解，但如果数据很大，不建议每次启动训练都重新 tokenize。更好的方式是先离线预处理，再训练读取。

**本节小结：**  
预训练 Dataset 的完整流程是文本到 token，再到固定长度 next-token prediction 样本。

---

## DataLoader 与 batch 组织

{IMAGE:22}

Dataset 返回单条样本，DataLoader 负责把多条样本组成 batch。

```python
from torch.utils.data import DataLoader

dataset = PretrainDataset(
    jsonl_path="pretrain.jsonl",
    tokenizer=tokenizer,
    max_seq_len=512
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

batch = next(iter(loader))

print(batch["input_ids"].shape)
print(batch["labels"].shape)
```

输出通常是：

```text
torch.Size([16, 512])
torch.Size([16, 512])
```

模型 forward：

```python
logits = model(batch["input_ids"])
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch["labels"].view(-1)
)
```

这里 batch 的 token 数为：

$$
\text{tokens per batch} = batch\_size \times max\_seq\_len
$$

如果 `batch_size = 16`，`max_seq_len = 512`：

$$
16 \times 512 = 8192
$$

也就是每个 step 训练 8192 个 token。

{KNOWLEDGE}语言模型训练中，很多时候比起“多少条样本”，更重要的是“训练了多少 token”。{/KNOWLEDGE}

**本节小结：**  
DataLoader 将固定长度样本堆叠成 batch，训练规模通常按 token 数衡量。

---

## 常见错误与排查方式

{IMAGE:23}

### 错误一：token id 超出词表范围

现象：

```text
IndexError: index out of range in self
```

原因通常是：

- tokenizer 词表大小和模型 `vocab_size` 不一致
- 加了特殊 token 但没有 resize embedding
- 读取 token 文件时 dtype 溢出或损坏

排查：

```python
max_id = max(token_ids)
print(max_id, model.config.vocab_size)

assert max_id < model.config.vocab_size
```

### 错误二：input_ids 和 labels 长度不一致

```python
assert input_ids.shape == labels.shape
```

如果不一致，通常是切片范围错了。

### 错误三：loss 不下降

可能原因：

1. labels 没有错位
2. 数据全是 padding 或无效 token
3. 学习率不合适
4. tokenizer 和模型词表不匹配
5. 数据质量太差
6. causal mask 错误，模型偷看未来或完全看不到上下文

### 错误四：训练很慢

可能原因：

1. 每个 batch 临时 tokenizer
2. json 解析过慢
3. `num_workers` 设置不合理
4. CPU 数据处理跟不上 GPU
5. 小 batch 导致 GPU 利用率低

{WARNING}Dataset bug 往往不会直接报错，而是表现为 loss 异常、生成质量很差或训练效率极低。{/WARNING}

**本节小结：**  
排查 Dataset 问题要重点检查 token 范围、shape、labels 错位和数据读取效率。

---

## 理论视角：为什么这样训练有效

{IMAGE:24}

自回归语言模型的本质是估计文本序列的联合概率。根据链式法则：

$$
P(x_1, x_2, ..., x_T)
= P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)\cdots P(x_T|x_1,...,x_{T-1})
$$

训练时，我们给模型真实前文，让它预测真实下一个 token。交叉熵损失为：

$$
\mathcal{L}
= -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

其中：

- $\theta$ 是模型参数
- $x_{<t}$ 是当前位置之前的上下文
- 目标是让真实 token 的概率尽可能高

当模型在大量文本上反复训练时，它会学习：

1. 字词搭配
2. 语法结构
3. 常识知识
4. 推理模式
5. 代码结构
6. 对话风格
7. 文档格式

不过模型学习到的并不是显式规则，而是分布规律。

{IMPORTANT}预训练的目标非常简单：预测下一个 token。但在大规模数据和大模型容量下，这个目标可以逼出复杂语言能力。{/IMPORTANT}

**本节小结：**  
next-token prediction 通过最大化文本概率，让模型从海量语料中学习语言分布。

---

## MiniMind 场景下的实践建议

{IMAGE:25}

针对 MiniMind 这种从零手写的大模型项目，可以采用以下实践策略：

1. tokenizer 词表不要过大
2. 数据先小规模跑通，再扩大
3. 先用 jsonl 理解流程，再转 bin 提升效率
4. 每条文本后加入 EOS
5. Dataset 返回 `input_ids` 和 `labels`
6. labels 相对 input_ids 左移一位
7. 固定 `max_seq_len`，减少 padding
8. 检查最大 token id 是否小于 vocab_size
9. 训练前打印一个 batch 的 shape
10. 训练初期观察 loss 是否正常下降

调试代码示例：

```python
batch = next(iter(loader))

print("input_ids:", batch["input_ids"].shape)
print("labels:", batch["labels"].shape)
print("max token id:", batch["input_ids"].max().item())
print("min token id:", batch["input_ids"].min().item())

assert batch["input_ids"].shape == batch["labels"].shape
assert batch["input_ids"].max().item() < vocab_size
```

还可以 decode 一小段检查数据是否正常：

```python
sample_ids = batch["input_ids"][0].tolist()
print(tokenizer.decode(sample_ids[:100]))
```

{KNOWLEDGE}手写大模型时，先验证数据，再怀疑模型。很多训练异常最早都来自 Dataset。{/KNOWLEDGE}

**本节小结：**  
MiniMind 的 Dataset 应以正确、简单、可调试为优先，再逐步优化性能。

---

## 本集结尾回顾

{IMAGE:26}

本集从理论上解释了预训练 Dataset 和 tokenizer 的关键关系。整体可以浓缩为一句话：

{IMPORTANT}把清洗后的文本通过 tokenizer 转成连续 token 流，再按固定长度切成 input_ids 和错位 labels，用 next-token prediction 训练模型。{/IMPORTANT}

完整数据路径如下：

```text
text
 -> tokenizer.encode
 -> token ids
 -> add eos
 -> concatenate
 -> block by max_seq_len + 1
 -> input_ids = chunk[:-1]
 -> labels = chunk[1:]
 -> model
 -> cross entropy loss
```

最后一帧通常对应课程过渡，说明后续将从理论进入 Dataset 的具体代码重制。

{IMAGE:10}

**本节小结：**  
本集完成了 Dataset 重制前的理论铺垫，后续可以据此实现更清晰、更高效的预训练数据加载逻辑。

---

## Key Takeaways

1. 预训练数据的本质是连续 token 序列，不是传统监督学习中的“样本-标签”表格。
2. tokenizer 决定文本如何映射成 token id，必须和模型 `vocab_size`、特殊 token 保持一致。
3. 自回归训练中，`input_ids = chunk[:-1]`，`labels = chunk[1:]`。
4. 每条文本后加入 EOS，有助于模型学习文本边界和停止生成。
5. `max_seq_len` 影响上下文能力、显存占用和训练速度。
6. padding 位置不应参与 loss，通常用 `ignore_index=-100`。
7. 小模型更需要高质量、格式稳定的数据。
8. Dataset 的效率会直接影响训练吞吐，预先 tokenize 通常优于训练时 tokenize。

## 思考题

1. 如果 `input_ids` 和 `labels` 完全相同，模型训练目标会发生什么问题？
2. 为什么预训练数据中通常要在不同文本之间插入 EOS？
3. 对于 MiniMind 这样的小模型，词表过大可能带来哪些负面影响？