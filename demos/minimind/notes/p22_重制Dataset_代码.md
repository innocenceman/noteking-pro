# 第22集: 重制Dataset：代码

# MiniMind 第22集讲义：PretrainDataset类实现

## 课程概述

**Episode 22/26：重制Dataset：代码**  
**时长**：8分4秒  
**主题**：PretrainDataset 预训练数据集类的完整实现  
**课程**：MiniMind - PyTorch从零手敲大模型

本集将带领大家从零实现一个完整的预训练数据集处理类，这是大模型训练流程中最基础也是最关键的数据处理环节。一个设计良好的Dataset类不仅能保证数据的高效加载，还能为后续的模型训练提供稳定的数据流支持。

{KNOWLEDGE}背景知识{/KNOWLEDGE}
在实现PretrainDataset之前，我们需要理解预训练模型数据处理的核心需求：
- **大规模文本处理**：预训练通常需要处理GB甚至TB级别的文本数据
- **高效tokenization**：将原始文本转换为模型可处理的token序列
- **动态批处理**：根据序列长度动态调整batch size
- **流式加载**：避免一次性将所有数据加载到内存中

---

## 1. PretrainDataset类的整体架构

{PretrainDataset}是MiniMind项目中数据处理的核心类，它继承自PyTorch的`Dataset`基类，负责将原始文本数据转换为模型训练所需的tensor格式。{IMAGE:1}

### 1.1 类设计原则

```python
class PretrainDataset(Dataset):
    """
    预训练数据集类
    
    设计原则：
    1. 惰性加载：不在初始化时读取所有数据，节省内存
    2. 高效tokenization：使用流式处理减少预处理时间
    3. 动态采样：根据序列长度动态构建样本
    """
```

{KNOWLEDGE}为什么要继承Dataset？{/KNOWLEDGE}
PyTorch的`Dataset`是数据加载的核心抽象，它强制实现`__len__`和`__getitem__`两个方法。这使得我们的数据集可以无缝接入`DataLoader`，享受多进程加载、shuffle、batch collation等高级功能。

---

## 2. 初始化方法详解

{IMAGE:2}展示的是PretrainDataset的初始化方法，这是整个数据处理流水线的起点。

### 2.1 构造函数实现

```python
def __init__(
    self, 
    data_path: str, 
    tokenizer,  # tokenizer对象
    max_length: int = 512,
    stride: int = 128  # 滑动窗口步长
):
    super().__init__()
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.stride = stride
    
    # 读取原始文本数据
    with open(data_path, 'r', encoding='utf-8') as f:
        self.texts = f.readlines()
    
    # 数据索引构建
    self.indices = self._build_indices()
```

{IMPORTANT}核心概念：滑动窗口策略{/IMPORTANT}
`stride`参数控制滑动窗口的步长。当`stride < max_length`时，相邻样本之间会有重叠，这可以：
- 增加训练样本数量
- 允许模型学习更多的上下文边界关系
- 提高数据利用率

### 2.2 索引构建算法

{IMAGE:3}详细展示了索引构建的逻辑，这是PretrainDataset中最核心的预处理步骤。

```python
def _build_indices(self):
    """
    构建样本索引列表
    
    每个索引项记录：(文本索引, 起始位置, 结束位置)
    用于后续的快速随机访问
    """
    indices = []
    
    for text_idx, text in enumerate(self.texts):
        # 计算该文本可以切分出多少个样本
        text_len = len(text)
        
        # 滑动窗口切分
        start = 0
        while start < text_len:
            end = min(start + self.max_length, text_len)
            
            # 记录索引
            indices.append((text_idx, start, end))
            
            # 滑动
            if end == text_len:
                break
            start += self.stride
    
    return indices
```

---

## 3. Tokenization处理流程

{IMAGE:4}展示了tokenization的核心流程，这是将人类可读文本转换为模型可处理数值的关键步骤。

### 3.1 Tokenizer的作用

$$text \xrightarrow{\text{Tokenizer}} [token_1, token_2, ..., token_n] \xrightarrow{\text{编码}} [id_1, id_2, ..., id_n]$$

{KNOWLEDGE}Tokenizer的分类{/KNOWLEDGE}
常见的tokenizer包括：
- **WordPiece**：BERT使用的子词分词器
- **BPE (Byte Pair Encoding)**：GPT系列使用的分词方法
- **SentencePiece**：Google开发的统一分词框架

### 3.2 数据编码实现

{IMAGE:5}展示了实际的编码代码实现。

```python
def encode_text(self, text: str) -> List[int]:
    """
    将文本编码为token id序列
    
    Args:
        text: 原始文本字符串
    
    Returns:
        token_ids: token id列表
    """
    # 调用tokenizer进行编码
    encoding = self.tokenizer.encode(
        text,
        add_special_tokens=True,  # 添加[CLS]和[SEP]
        max_length=self.max_length,
        truncation=True,
        padding='max_length'
    )
    
    return encoding['input_ids']
```

---

## 4. 数据集接口实现

{IMAGE:6}展示了PyTorch Dataset接口的核心实现，这是连接数据存储和训练循环的桥梁。

### 4.1 `__len__`方法

```python
def __len__(self):
    """
    返回数据集中样本的总数
    
    这个方法对于DataLoader正确分批至关重要
    """
    return len(self.indices)
```

{IMPORTANT}核心概念：索引与样本的区别{/IMPORTANT}
- `len(self.texts)` = 原始文本的数量
- `len(self)` = 可用训练样本的数量（经过滑动窗口切分后）

由于滑动窗口的存在，样本数通常远大于文本数。

### 4.2 `__getitem__`方法

{IMAGE:7}和{IMAGE:8}详细展示了`__getitem__`的实现，这是Dataset类最关键的方法。

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    根据索引获取单个训练样本
    
    Args:
        idx: 样本索引
    
    Returns:
        sample: 包含input_ids和labels的字典
    """
    # 获取索引信息
    text_idx, start, end = self.indices[idx]
    
    # 提取文本片段
    text = self.texts[text_idx][start:end]
    
    # 编码
    token_ids = self.encode_text(text)
    
    # 构建样本
    # 对于GPT-style模型，labels就是input_ids的shifted版本
    return {
        'input_ids': torch.tensor(token_ids, dtype=torch.long),
        'labels': torch.tensor(token_ids, dtype=torch.long)
    }
```

{IMAGE:9}展示了训练样本的最终格式，其中input和label的设计是因果语言模型的标准做法。

---

## 5. 自定义Collate函数

{IMAGE:10}展示了自定义collate函数的实现，这对于处理可变长度序列至关重要。

### 5.1 为什么需要自定义Collate？

{PretrainDataset}处理的是变长序列，但PyTorch要求同一个batch中的tensor具有相同的形状。自定义collate函数可以实现动态padding。

### 5.2 Collate函数实现

```python
def pretrain_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数
    
    功能：
    1. 将多个样本堆叠成batch
    2. 处理序列长度不一致的问题
    
    Args:
        batch: 单个batch的样本列表
    
    Returns:
        batched: 批处理后的张量字典
    """
    # 提取input_ids和labels
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 方法1：固定长度padding（简单但可能有padding浪费）
    # input_ids = torch.stack(input_ids)
    # labels = torch.stack(labels)
    
    # 方法2：动态padding到batch内最大长度
    max_len = max(len(ids) for ids in input_ids)
    
    # padding操作
    padded_input_ids = []
    padded_labels = []
    
    for ids in input_ids:
        pad_len = max_len - len(ids)
        padded_ids = F.pad(ids, (0, pad_len), value=0)  # padding token id = 0
        padded_input_ids.append(padded_ids)
    
    for lbl in labels:
        pad_len = max_len - len(lbl)
        padded_lbl = F.pad(lbl, (0, pad_len), value=-100)  # 忽略padding位置的loss
        padded_labels.append(padded_lbl)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels)
    }
```

{IMAGE:11}和{IMAGE:12}展示了collate函数在实际使用中的效果对比。

{WARNING}易错点：Padding Token的Loss处理{/WARNING}
在计算loss时，padding位置的token不应该对梯度产生贡献。标准做法是将这些位置的label设置为-100，因为在PyTorch的CrossEntropyLoss中，默认忽略target为-100的样本：

```python
loss_fct = CrossEntropyLoss(ignore_index=-100)
```

---

## 6. DataLoader集成

### 6.1 创建DataLoader

```python
from torch.utils.data import DataLoader

# 创建数据集实例
dataset = PretrainDataset(
    data_path='./data/pretrain.txt',
    tokenizer=tokenizer,
    max_length=512,
    stride=128
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=pretrain_collate_fn,
    pin_memory=True  # 加速CPU到GPU的数据传输
)
```

### 6.2 训练循环中的使用

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')
        
        # 前向传播
        outputs = model(input_ids=input_ids)
        loss = loss_fct(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1)
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

---

## 7. 完整代码汇总

以下是PretrainDataset的完整实现：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

class PretrainDataset(Dataset):
    """预训练数据集类"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 512,
        stride: int = 128
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 加载原始文本
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        
        # 构建索引
        self.indices = self._build_indices()
    
    def _build_indices(self) -> List:
        """构建样本索引"""
        indices = []
        for text_idx, text in enumerate(self.texts):
            text_len = len(text)
            start = 0
            while start < text_len:
                end = min(start + self.max_length, text_len)
                indices.append((text_idx, start, end))
                if end == text_len:
                    break
                start += self.stride
        return indices
    
    def encode_text(self, text: str) -> List[int]:
        """文本编码"""
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        return encoding['input_ids']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text_idx, start, end = self.indices[idx]
        text = self.texts[text_idx][start:end]
        token_ids = self.encode_text(text)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long)
        }


def pretrain_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理collate函数"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    
    for ids in input_ids:
        pad_len = max_len - len(ids)
        padded_ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
        padded_input_ids.append(padded_ids)
    
    for lbl in labels:
        pad_len = max_len - len(lbl)
        padded_lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
        padded_labels.append(padded_lbl)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels)
    }
```

---

## 8. 性能优化建议

### 8.1 数据加载优化

| 优化策略 | 效果 |
|---------|------|
| `num_workers > 0` | 启用多进程加载，避免I/O阻塞 |
| `pin_memory=True` | 加速CPU-GPU数据传输 |
| `prefetch_factor` | 预加载下一个batch |
| 流式读取 | 减少内存占用 |

### 8.2 Tokenization优化

```python
# 使用fast tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=True  # 使用Rust实现的快速tokenizer
)

# 批量编码
encoded = tokenizer.batch_encode_plus(
    texts,
    padding=True,
    truncation=True,
    max_length=512
)
```

---

## 本集总结

{IMAGE:12}是本集的最后一帧，总结了PretrainDataset的核心设计要点。

本集学习了：

1. **PretrainDataset类的整体架构**：继承PyTorch Dataset，实现`__len__`和`__getitem__`
2. **索引构建机制**：使用滑动窗口策略增加训练样本数量
3. **Tokenization处理**：将原始文本转换为token id序列
4. **自定义Collate函数**：处理变长序列的动态padding
5. **DataLoader集成**：实现高效的数据加载流水线

{IMPORTANT}核心要点{/IMPORTANT}
- Dataset类的设计遵循"惰性加载"原则，避免一次性占用过多内存
- 滑动窗口步长`stride`的选择影响训练样本数量和上下文覆盖度
- Padding位置的loss需要被忽略（设置为-100）
- 自定义collate函数是处理变长序列的关键

---

## 思考题

**思考题1**：当`stride = max_length`时，相邻样本之间没有重叠。这种设置有什么优缺点？

**提示**：考虑训练效率、内存占用、以及模型能否学习到相邻文本段之间的关系。

**思考题2**：在计算loss时，我们设置padding位置的label为-100。如果不这样处理，会有什么问题？

**提示**：考虑模型是否会在padding位置学习到有意义的表示，以及这对训练稳定性的影响。

---

*下一集预告*：我们将学习如何将PretrainDataset与模型训练循环集成，实现完整的预训练流程。