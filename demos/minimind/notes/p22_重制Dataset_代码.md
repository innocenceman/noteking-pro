# 第22集: 重制Dataset：代码

## 课程定位：为什么要重制 Dataset

本集是 MiniMind 预训练代码部分的关键环节，主题是实现 `PretrainDataset` 类。它负责把已经准备好的文本数据转成模型训练所需的张量样本，是连接“原始语料”和“Transformer 预训练循环”的桥梁。

{IMAGE:1}

在大模型预训练中，模型并不是直接读取一句话、一篇文章或一个 JSON 文件，而是读取固定长度的 token 序列。`Dataset` 的职责可以概括为：

{IMPORTANT}  
`PretrainDataset` 的核心任务：  
把文本样本编码成 token id，并切分成长度固定的输入 `X` 和预测目标 `Y`。  
{/IMPORTANT}

对于自回归语言模型，训练目标是“根据前面的 token 预测下一个 token”。因此同一段 token 序列通常会被错位切分：

$$
X = [t_0, t_1, ..., t_{n-1}]
$$

$$
Y = [t_1, t_2, ..., t_n]
$$

其中 $X$ 是模型输入，$Y$ 是标签。模型在位置 $i$ 看到 $t_i$，学习预测 $t_{i+1}$。

{IMAGE:2}

### 本节小结

本节说明了 `PretrainDataset` 的作用：它不是简单读取文件，而是负责完成文本到 token、token 到训练样本的转换。

## Dataset 在 PyTorch 训练流程中的位置

PyTorch 中通常使用 `Dataset` 和 `DataLoader` 配合完成数据读取：

```python
from torch.utils.data import Dataset, DataLoader
```

`Dataset` 负责定义“数据是什么”，核心接口包括：

```python
class MyDataset(Dataset):
    def __len__(self):
        return 数据集样本数量

    def __getitem__(self, index):
        return 第 index 条训练样本
```

`DataLoader` 负责定义“如何批量加载数据”，例如：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

在 MiniMind 的预训练流程中，`PretrainDataset` 通常输出两个张量：

```python
X, Y = dataset[index]
```

其中：

- `X`：输入 token 序列，形状通常是 `[max_length]`
- `Y`：标签 token 序列，形状通常是 `[max_length]`
- `DataLoader` 拼接后得到 `[batch_size, max_length]`

{IMAGE:3}

{KNOWLEDGE}  
`Dataset` 本身不负责训练，也不负责反向传播。  
它只关心如何把某个索引 `index` 对应的数据样本变成模型可用的张量。  
{/KNOWLEDGE}

### 本节小结

`PretrainDataset` 是 PyTorch 数据管线中的底层组件，定义每条样本如何构造；`DataLoader` 在其基础上完成批处理、打乱和多进程读取。

## 预训练数据的基本形态

预训练数据通常来自大量文本，例如：

```text
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。
```

或者 JSONL 格式：

```json
{"text": "春眠不觉晓，处处闻啼鸟。"}
{"text": "人工智能正在改变软件开发方式。"}
```

在进入模型前，需要先经过 tokenizer 编码：

```python
ids = tokenizer.encode(text)
```

编码结果可能类似：

```python
[101, 2345, 6789, 3456, 102]
```

这些整数就是 token id。模型的 Embedding 层只能处理整数 id，不能直接处理字符串。

{IMAGE:9}

常见的数据处理流程是：

1. 读取原始文本
2. 使用 tokenizer 编码为 token id
3. 添加特殊 token，例如 BOS、EOS
4. 截断或填充到固定长度
5. 构造输入 `X` 和标签 `Y`
6. 返回 PyTorch 张量

### 本节小结

预训练 Dataset 的核心不是保存文本，而是保存“可被模型直接消费”的 token 序列。

## PretrainDataset 的类结构

一个典型的 `PretrainDataset` 实现如下：

```python
import json
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                text = obj["text"]
                samples.append(text)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.samples[index]
        token_ids = self.tokenizer.encode(text)

        token_ids = token_ids[: self.max_length + 1]

        if len(token_ids) < self.max_length + 1:
            pad_len = self.max_length + 1 - len(token_ids)
            token_ids += [self.tokenizer.pad_token_id] * pad_len

        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)

        return x, y
```

{IMAGE:10}

这个类中最重要的是三个函数：

```python
__init__()
__len__()
__getitem__()
```

它们分别负责初始化、返回数据长度、返回单条训练样本。

### 本节小结

`PretrainDataset` 的结构并不复杂，但每一步都直接影响模型输入质量和训练稳定性。

## 初始化函数：保存配置与加载数据

`__init__` 通常接收以下参数：

```python
def __init__(self, data_path, tokenizer, max_length=512):
    self.data_path = data_path
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.samples = self.load_data(data_path)
```

各参数含义如下：

- `data_path`：训练数据路径
- `tokenizer`：分词器，用于把文本转成 token id
- `max_length`：模型单次训练接收的最大序列长度
- `samples`：加载后的文本样本列表

{IMAGE:11}

`max_length` 非常关键。Transformer 的计算复杂度与序列长度有关，注意力矩阵大小约为：

$$
O(n^2)
$$

其中 $n$ 是序列长度。也就是说，序列长度翻倍，注意力计算量大约变为原来的四倍。

{IMPORTANT}  
`max_length` 不只是数据参数，也是显存、速度和模型上下文能力之间的权衡。  
{/IMPORTANT}

### 本节小结

初始化阶段完成数据路径、tokenizer 和长度配置的保存，同时把原始数据加载到内存或建立索引。

## 数据加载：从 JSONL 中读取文本

在预训练项目中，数据常用 JSONL 格式，即每一行都是一个 JSON 对象：

```json
{"text": "今天我们来实现 PretrainDataset。"}
{"text": "语言模型通过预测下一个 token 来学习语言规律。"}
```

读取代码示例：

```python
def load_data(self, data_path):
    samples = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "")

            if text:
                samples.append(text)

    return samples
```

{IMAGE:12}

这里有几个细节：

- 使用 `encoding="utf-8"`，避免中文乱码
- 使用 `strip()` 去掉换行符
- 跳过空行
- 使用 `obj.get("text", "")` 可以避免某些样本缺字段时报错
- 空文本不加入样本列表

{WARNING}  
如果数据字段名不一致，例如有的叫 `content`，有的叫 `text`，Dataset 会读不到有效文本。  
训练前应统一数据格式。  
{/WARNING}

### 本节小结

数据加载阶段需要保证格式稳定、文本有效、编码正确。预训练的质量很大程度取决于这一步的数据清洗。

## Tokenizer 编码：文本如何进入模型

模型无法直接理解中文字符或英文单词。Tokenizer 的任务是把文本映射到整数序列：

```python
text = "你好，MiniMind"
ids = tokenizer.encode(text)
```

结果可能是：

```python
[15496, 11, 32000, 32001]
```

{IMAGE:13}

一般来说，tokenizer 还可能提供特殊 token：

```python
bos_token_id  # 句子开始
eos_token_id  # 句子结束
pad_token_id  # 填充
unk_token_id  # 未知字符
```

在预训练中，常见做法是添加 EOS：

```python
token_ids = tokenizer.encode(text)
token_ids.append(tokenizer.eos_token_id)
```

EOS 的作用是告诉模型“这段文本结束了”。如果没有 EOS，模型可能难以学习句子或文档边界。

{KNOWLEDGE}  
BOS 表示 beginning of sequence，EOS 表示 end of sequence。  
在自回归模型中，EOS 很重要，因为它让模型学习何时停止生成。  
{/KNOWLEDGE}

### 本节小结

Tokenizer 是文本和模型之间的转换器。`PretrainDataset` 必须正确调用 tokenizer，并处理特殊 token。

## 截断与填充

模型通常要求 batch 中所有样本长度一致。如果一条文本过长，需要截断；如果太短，需要填充。

```python
token_ids = token_ids[: self.max_length + 1]
```

为什么是 `max_length + 1`？

因为要构造：

```python
x = token_ids[:-1]
y = token_ids[1:]
```

如果最终希望 `x` 和 `y` 的长度都是 `max_length`，原始 token 序列就需要长度 `max_length + 1`。

{IMAGE:14}

例如：

```python
token_ids = [10, 20, 30, 40, 50]
max_length = 4
```

则：

```python
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

模型看到 `10` 预测 `20`，看到 `20` 预测 `30`，以此类推。

如果长度不足，则填充：

```python
if len(token_ids) < self.max_length + 1:
    pad_len = self.max_length + 1 - len(token_ids)
    token_ids += [self.tokenizer.pad_token_id] * pad_len
```

{WARNING}  
构造自回归训练样本时，必须先保证 token 序列长度是 `max_length + 1`，再切分 `X` 和 `Y`。  
如果只准备 `max_length` 个 token，切分后长度会变成 `max_length - 1`。  
{/WARNING}

### 本节小结

截断和填充保证了每条样本长度一致；`max_length + 1` 是为了错位切分后仍保留 `max_length` 长度。

## 输入 X 与标签 Y 的构造

预训练语言模型的核心目标是 next-token prediction。给定输入序列：

$$
X = [x_1, x_2, ..., x_T]
$$

模型输出每个位置的预测分布：

$$
P(x_{t+1} \mid x_1, x_2, ..., x_t)
$$

训练目标是最大化真实下一个 token 的概率，等价于最小化交叉熵损失：

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P(y_t \mid x_{\le t})
$$

其中：

- $x_t$ 是当前位置输入 token
- $y_t$ 是当前位置标签 token，也就是下一个 token
- $T$ 是序列长度

{IMAGE:15}

代码实现：

```python
x = torch.tensor(token_ids[:-1], dtype=torch.long)
y = torch.tensor(token_ids[1:], dtype=torch.long)
```

这里的 `dtype=torch.long` 很重要，因为 Embedding 层和交叉熵损失都要求 token id 是整数类型。

{IMPORTANT}  
语言模型不是一次只预测最后一个 token，而是在序列的每个位置都预测下一个 token。  
所以一个长度为 $T$ 的样本，会产生 $T$ 个监督信号。  
{/IMPORTANT}

### 本节小结

`X` 和 `Y` 的错位构造是自回归语言模型训练的核心。理解这一步，就理解了预训练 Dataset 的本质。

## Padding 对损失计算的影响

如果短文本被填充为固定长度，`Y` 中会包含 padding token。例如：

```python
token_ids = [10, 20, 30, 0, 0]
x = [10, 20, 30, 0]
y = [20, 30, 0, 0]
```

其中 `0` 可能是 `pad_token_id`。通常不希望模型在 padding 位置上计算损失，否则模型会被迫学习大量“预测 PAD”的无意义任务。

解决方式是在损失函数中使用 `ignore_index`：

```python
loss_fn = torch.nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id
)
```

或者在 Dataset 中把 `y` 中的 padding 改成 `-100`：

```python
labels = token_ids[1:]

labels = [
    token if token != self.tokenizer.pad_token_id else -100
    for token in labels
]

y = torch.tensor(labels, dtype=torch.long)
```

然后训练时：

```python
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
```

{IMAGE:16}

{WARNING}  
如果没有忽略 padding 位置的 loss，训练指标可能看似下降，但模型会学到很多无效模式。  
{/WARNING}

### 本节小结

Padding 是为了 batch 对齐，但 padding 位置不应该参与有效语言建模损失。

## 一个更完整的 PretrainDataset 实现

下面是一个更完整、更稳健的版本：

```python
import json
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_jsonl(data_path)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

    def _load_jsonl(self, data_path):
        samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                obj = json.loads(line)
                text = obj.get("text", "")

                if isinstance(text, str) and text.strip():
                    samples.append(text.strip())

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.samples[index]

        token_ids = self.tokenizer.encode(text)

        if self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)

        token_ids = token_ids[: self.max_length + 1]

        pad_len = self.max_length + 1 - len(token_ids)
        if pad_len > 0:
            token_ids += [self.pad_token_id] * pad_len

        input_ids = token_ids[:-1]
        labels = token_ids[1:]

        labels = [
            token if token != self.pad_token_id else -100
            for token in labels
        ]

        x = torch.tensor(input_ids, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)

        return x, y
```

{IMAGE:17}

这个版本处理了几个实际问题：

- 自动加载 JSONL
- 跳过空文本
- 添加 EOS
- 处理没有 PAD token 的 tokenizer
- 保证输入长度固定
- 把 padding label 替换成 `-100`

### 本节小结

完整 Dataset 实现不仅要能跑，还要处理边界情况，例如空文本、特殊 token 缺失和 padding loss。

## 与 DataLoader 的配合

定义好 Dataset 后，可以这样使用：

```python
dataset = PretrainDataset(
    data_path="data/pretrain.jsonl",
    tokenizer=tokenizer,
    max_length=512
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

训练循环中：

```python
for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

{IMAGE:18}

这里涉及一个形状转换。模型输出通常是：

$$
logits \in \mathbb{R}^{B \times T \times V}
$$

其中：

- $B$ 是 batch size
- $T$ 是序列长度
- $V$ 是词表大小

交叉熵通常要求输入形状是：

$$
(B \times T, V)
$$

标签形状是：

$$
(B \times T)
$$

所以需要：

```python
logits.view(-1, logits.size(-1))
y.view(-1)
```

### 本节小结

Dataset 输出单条样本，DataLoader 组织 batch，训练循环计算 loss。三者形状必须严格匹配。

## 为什么不能直接一行文本一个样本

如果每一行文本长度差异很大，会出现两个问题：

1. 短文本 padding 过多，浪费计算
2. 长文本被频繁截断，丢失上下文

例如：

```text
你好。
这是一段非常非常长的文章，包含大量上下文信息……
```

如果统一填充到 512，第一个样本只有几个有效 token，大部分计算都浪费在 PAD 上。

{IMAGE:19}

更高效的做法是把多个文本拼接成连续 token 流，然后按固定长度切块：

```python
all_tokens = []

for text in texts:
    ids = tokenizer.encode(text)
    ids.append(tokenizer.eos_token_id)
    all_tokens.extend(ids)

chunks = []
for i in range(0, len(all_tokens), max_length + 1):
    chunk = all_tokens[i : i + max_length + 1]
    if len(chunk) == max_length + 1:
        chunks.append(chunk)
```

这种方式能提高 token 利用率，减少 padding。

{KNOWLEDGE}  
大规模预训练通常更接近“token 流切块”，而不是“每行文本单独作为一个样本”。  
不过教学项目中，一行一个样本更直观，便于理解。  
{/KNOWLEDGE}

### 本节小结

一行一个样本适合教学和小规模数据；大规模预训练更常使用拼接后的 token 流切块，提高训练效率。

## Dataset 性能优化思路

当数据量较大时，Dataset 的性能会直接影响训练速度。常见优化方向包括：

### 预先 tokenization

每次 `__getitem__` 都调用 tokenizer 会有额外开销：

```python
token_ids = self.tokenizer.encode(text)
```

可以在初始化阶段预处理：

```python
self.samples = []

for text in texts:
    token_ids = tokenizer.encode(text)
    self.samples.append(token_ids)
```

这样训练时只需要切片和转 tensor。

{IMAGE:20}

### 使用二进制缓存

对于更大的数据集，可以把 token id 保存成二进制文件，例如 `.bin` 或 `.npy`，训练时直接读取整数数组。

### 懒加载

如果数据太大，不能一次性读入内存，可以只保存文件偏移量：

```python
self.offsets = []
```

然后在 `__getitem__` 中按偏移读取对应行。

{WARNING}  
教学代码优先清晰；工程代码优先吞吐。  
不要在还没跑通训练前过早引入复杂的数据缓存系统。  
{/WARNING}

### 本节小结

Dataset 的性能优化包括预先编码、缓存 token、懒加载和减少 padding。应根据数据规模选择复杂度。

## 常见错误排查

### 错误一：输入和标签长度不一致

错误代码：

```python
token_ids = token_ids[: self.max_length]
x = token_ids[:-1]
y = token_ids[1:]
```

此时 `x` 和 `y` 长度是 `max_length - 1`，可能与模型配置不一致。

正确做法：

```python
token_ids = token_ids[: self.max_length + 1]
x = token_ids[:-1]
y = token_ids[1:]
```

{IMAGE:21}

### 错误二：label 没有忽略 PAD

如果 `pad_token_id` 参与 loss，模型会过度学习 padding。

正确做法：

```python
labels = [
    token if token != pad_token_id else -100
    for token in labels
]
```

### 错误三：dtype 错误

Embedding 层需要 `torch.long` 类型：

```python
x = torch.tensor(input_ids, dtype=torch.long)
```

不能使用 float：

```python
x = torch.tensor(input_ids, dtype=torch.float)
```

### 错误四：tokenizer 缺少 pad_token_id

一些自回归模型 tokenizer 可能没有 PAD。可以临时使用 EOS 作为 PAD：

```python
if tokenizer.pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id
```

{IMAGE:22}

### 本节小结

Dataset 中最常见的问题集中在长度、padding、dtype 和特殊 token 上。调试时应优先检查这些位置。

## 与 MiniMind 预训练目标的关系

MiniMind 的目标是从零实现一个小型大语言模型。`PretrainDataset` 虽然代码量不大，但它直接决定模型看到什么样的数据。

{IMAGE:4}

预训练阶段的核心链路是：

```text
文本语料
  -> tokenizer 编码
  -> Dataset 构造 X/Y
  -> DataLoader 组成 batch
  -> Transformer 前向传播
  -> CrossEntropyLoss
  -> 反向传播更新参数
```

{IMAGE:5}

如果 Dataset 构造错误，即使模型结构完全正确，也可能训练不出合理结果。例如：

- 标签没有错位，模型变成复制当前 token
- padding 参与训练，loss 被无效位置污染
- 没有 EOS，模型边界意识较弱
- 截断逻辑错误，样本长度不稳定

{IMAGE:6}

### 本节小结

`PretrainDataset` 是预训练质量的入口。数据处理错误往往比模型代码错误更隐蔽，但影响同样严重。

## 推荐的调试方式

实现 Dataset 后，不要立刻开始完整训练。应先人工检查几个样本。

```python
x, y = dataset[0]

print(x.shape)
print(y.shape)
print(x[:20])
print(y[:20])
```

期望看到：

```python
torch.Size([512])
torch.Size([512])
```

还可以 decode 回文本：

```python
print(tokenizer.decode(x.tolist()))
print(tokenizer.decode([
    token for token in y.tolist()
    if token != -100
]))
```

{IMAGE:7}

重点检查：

- `x` 和 `y` 是否等长
- `y` 是否是 `x` 向后错一位
- 是否有大量 padding
- padding label 是否为 `-100`
- EOS 是否出现在合理位置

{IMAGE:8}

{IMPORTANT}  
Dataset 调试的最佳方法不是只看 loss，而是直接打印 token、decode 文本、检查错位关系。  
{/IMPORTANT}

### 本节小结

在正式训练前，应先抽样检查 Dataset 输出。数据正确，后续训练问题才有定位基础。

## 关键代码回顾

下面用简化版代码回顾本集重点：

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        ids = self.tokenizer.encode(text)
        ids.append(self.tokenizer.eos_token_id)

        ids = ids[: self.max_length + 1]

        if len(ids) < self.max_length + 1:
            ids += [self.tokenizer.pad_token_id] * (
                self.max_length + 1 - len(ids)
            )

        x = ids[:-1]
        y = ids[1:]

        y = [
            token if token != self.tokenizer.pad_token_id else -100
            for token in y
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
```

{IMAGE:23}

### 本节小结

这段代码包含了预训练 Dataset 的完整核心逻辑：编码、加 EOS、截断、填充、错位、转 tensor。

## Key Takeaways

1. `PretrainDataset` 的核心职责是把文本变成自回归语言模型训练所需的 `X` 和 `Y`。
2. 自回归训练使用错位标签：`X = token_ids[:-1]`，`Y = token_ids[1:]`。
3. 为了得到长度为 `max_length` 的输入和标签，原始 token 序列应处理成 `max_length + 1`。
4. Padding 位置通常不应参与 loss，可用 `ignore_index=-100` 忽略。
5. `torch.long` 是 token id 的正确 dtype。
6. EOS 有助于模型学习文本边界。
7. Dataset 的错误会直接污染训练目标，必须先抽样检查再开始长时间训练。

## 思考题

1. 如果 `Y` 没有相对 `X` 向后移动一位，而是直接等于 `X`，模型会学到什么？
2. 为什么 padding token 不应该参与语言模型的交叉熵损失？
3. 对于大规模预训练，为什么“拼接 token 流后切块”通常比“一行文本一个样本”更高效？