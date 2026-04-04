# 第21集: 重制Dataset：理论

# MiniMind Episode 21 讲义

## 重制Dataset：理论

**课程**: MiniMind - PyTorch从零手敲大模型  
**时长**: 5分32秒  
**主题**: 预训练数据格式、Tokenizer 原理与实现

---

## 1. 课程概述

本节课我们将深入探讨大语言模型（LLM）预训练阶段的数据处理理论。在训练一个强大的语言模型之前，高质量的数据准备是至关重要的基础工作。本节将重点讲解：

- 预训练数据的标准格式
- Tokenizer（分词器）的核心概念与原理
- 数据预处理管线的设计思路
- 代码实现细节

{IMAGE:1}

{KNOWLEDGE}大语言模型的预训练数据通常来源于互联网文本、书籍、代码库等多种来源。在喂入模型之前，这些原始数据必须经过严格的清洗、分词、格式化等处理流程。{/KNOWLEDGE}

---

## 2. 预训练数据格式规范

### 2.1 标准数据格式

在MiniMind项目中，我们采用JSONL格式存储预处理后的数据。JSONL（JSON Lines）是一种每行包含一个完整JSON对象的文本格式，非常适合大规模数据集的处理。

{IMAGE:2}

```json
{"text": "今天天气真好，我们一起去公园散步吧。"}
{"text": "人工智能技术正在深刻改变我们的生活方式。"}
{"text": "Python是一门易学易用的编程语言。"}
```

{IMPORTANT}核心概念：为什么选择JSONL格式？{/IMPORTANT}

1. **流式处理友好**：可以逐行读取，无需将整个文件加载到内存
2. **并行处理方便**：天然支持多进程/多线程并行读取
3. **扩展性强**：可轻松添加新字段（如`label`、`source`等）
4. **兼容性高**：几乎所有编程语言都原生支持JSON解析

### 2.2 数据字段设计

{IMAGE:3}

一个标准的预训练数据样本通常包含以下字段：

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `text` | string | 原始文本内容 | "今天天气很好" |
| `id` | int | 样本唯一标识 | 10001 |
| `source` | string | 数据来源 | "web_text" |
| `length` | int | 文本长度（字符数） | 6 |

```python
# 定义数据样本的Python表示
class DataSample:
    """预训练数据样本结构"""
    def __init__(self, text: str, sample_id: int = 0, source: str = "unknown"):
        self.text = text
        self.id = sample_id
        self.source = source
        self.length = len(text)
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "text": self.text,
            "id": self.id,
            "source": self.source,
            "length": self.length
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataSample':
        """从字典反序列化"""
        return cls(
            text=data["text"],
            sample_id=data.get("id", 0),
            source=data.get("source", "unknown")
        )
```

### 2.3 训练样本的组织方式

{IMAGE:4}

{IMAGE:5}

对于自回归语言模型，训练时我们将文本按照固定长度（context_length）进行切分，形成多个训练样本。

$$N_{\text{samples}} = \sum_{i=1}^{M} \left\lceil \frac{L_i - 1}{L_{\text{context}}} \right\rceil$$

其中：
- $M$ 是原始文档数量
- $L_i$ 是第 $i$ 个文档的长度
- $L_{\text{context}}$ 是上下文窗口长度

```python
def create_training_samples(
    text: str, 
    context_length: int = 512,
    stride: int = 512
) -> list[str]:
    """
    将长文本切分为训练样本
    
    Args:
        text: 输入文本
        context_length: 上下文窗口长度
        stride: 滑动步长（控制样本间重叠程度）
    
    Returns:
        训练样本列表
    """
    # 将文本编码为token序列
    tokens = tokenize(text)
    
    samples = []
    start = 0
    while start < len(tokens):
        end = start + context_length
        sample = tokens[start:end]
        samples.append(sample)
        start += stride  # 非重叠切分
    
    return samples
```

---

## 3. Tokenizer（分词器）详解

### 3.1 什么是Tokenizer

{IMAGE:6}

{IMPORTANT}核心概念：Tokenizer的作用{/IMPORTANT}

Tokenizer是将原始文本转换为模型可处理的数字序列（token IDs）的关键组件。它是连接人类语言与机器语言的桥梁。

```
原始文本: "今天天气很好"
    ↓ Tokenize
Token序列: [101, 1956, 1857, 3300, 3173, 102]
    ↓
模型输入: tensor([101, 1956, 1857, 3300, 3173, 102])
```

{IMAGE:7}

### 3.2 Tokenization策略

{IMAGE:8}

主流的Tokenization策略有以下几种：

#### 3.2.1 词级别（Word-level）

将文本按空格和标点分割为单词：

```
"Machine learning is amazing"
→ ["Machine", "learning", "is", "amazing"]
```

**优点**: 语义清晰  
**缺点**: 词汇表庞大，未登录词（OOV）问题严重

#### 3.2.2 字符级别（Character-level）

将文本按字符分割：

```
"Machine" → ["M", "a", "c", "h", "i", "n", "e"]
```

**优点**: 词汇表极小，无OOV问题  
**缺点**: 序列长度爆炸，语义信息丢失

#### 3.2.3 子词级别（Subword-level）⭐

{B WARNING}易错点：子词tokenization{/WARNING}

子词tokenization是目前LLM的主流选择，它在词级别和字符级别之间取得平衡：

| 方法 | 代表模型 | 特点 |
|------|---------|------|
| BPE | GPT-2, RoBERTa | 字节级编码，OOV友好 |
| WordPiece | BERT, DistilBERT | 基于频率的词汇合并 |
| SentencePiece | LLaMA, ChatGLM | 统一框架，支持多语言 |

```
原始文本: "unbelievably"
BPE分割: ["un", "##believ", "##ably"]
     → [1057, 3987, 12842]
```

### 3.3 BPE算法原理

{IMAGE:9}

{IMAGE:10}

Byte Pair Encoding（BPE）是一种基于频率的子词压缩算法。其核心思想是迭代地合并出现频率最高的字符对。

**算法步骤：**

1. 将文本切分为字符序列
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对
4. 重复步骤2-3直到达到预设词汇表大小

$$V = V_0 + N_{\text{merge}}$$

其中 $V_0$ 是初始字符词汇表大小，$N_{\text{merge}}$ 是合并操作次数。

```python
from collections import Counter, defaultdict

class SimpleBPE:
    """简化的BPE实现"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        
    def get_stats(self, words: list[list[str]]) -> Counter:
        """统计字符对的频率"""
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[word[i], word[i+1]] += 1
        return pairs
    
    def merge_pair(self, words: list[list[str]], pair: tuple) -> list[list[str]]:
        """合并指定的字符对"""
        first, second = pair
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words
    
    def train(self, corpus: list[str]):
        """训练BPE模型"""
        # 初始化：将文本切分为字符
        words = [list(word) + ['</w>'] for word in corpus]
        
        # 迭代合并
        num_merges = self.vocab_size - 256  # 保留256个基础字符
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            words = self.merge_pair(words, best_pair)
            self.merges.append(best_pair)
            print(f"合并 {i+1}: {best_pair} -> {best_pair[0]+best_pair[1]}")
```

---

## 4. 数据预处理完整流程

### 4.1 流水线架构

{IMAGE:11}

一个完整的数据预处理流水线通常包含以下步骤：

```
原始文本 → 清洗 → 分句 → Tokenize → 格式化 → 存储
   ↓        ↓       ↓        ↓          ↓         ↓
 Crawler   Rules  SentencePiece   BPE     JSONL    Dataset
```

### 4.2 数据清洗策略

{IMAGE:12}

| 清洗步骤 | 方法 | 目的 |
|---------|------|------|
| HTML标签移除 | 正则匹配 | 去除网页噪音 |
| 重复内容过滤 | SimHash/Near-dedup | 保证数据多样性 |
| 长度过滤 | 阈值判断 | 去除过短/过长文本 |
| 质量评分 | 语言模型困惑度 | 筛选高质量内容 |
| 隐私信息脱敏 | NER识别 | 去除敏感信息 |

```python
import re
from typing import Optional

class TextCleaner:
    """文本清洗工具"""
    
    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length
    
    def clean(self, text: str) -> Optional[str]:
        """执行完整的清洗流程"""
        # 1. 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 3. 移除特殊控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # 4. 长度过滤
        if not (self.min_length <= len(text) <= self.max_length):
            return None
        
        return text.strip()
    
    def remove_duplicates(self, texts: list[str]) -> list[str]:
        """简单去重（基于精确匹配）"""
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        return unique_texts
```

---

## 5. 代码实现：构建Dataset类

```python
import json
from torch.utils.data import Dataset
from typing import Callable, Optional
import torch

class PretrainDataset(Dataset):
    """
    预训练数据集类
    支持从JSONL文件加载数据并进行动态tokenization
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: Callable,
        context_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.samples = []
        
        # 加载数据
        print(f"正在加载数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text:
                        self.samples.append(text)
        
        print(f"加载完成，共 {len(self.samples)} 个样本")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个训练样本
        
        Returns:
            dict: 包含input_ids和labels的字典
        """
        text = self.samples[idx]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.context_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        
        # 对于语言模型，input和label相同（预测下一个token）
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


# 使用示例
def demo():
    """演示Dataset的使用"""
    
    # 假设我们使用一个简单的字符级tokenizer
    def simple_tokenizer(text: str, **kwargs) -> dict:
        """简化版tokenizer"""
        chars = list(text)
        return {'input_ids': torch.tensor([ord(c) % 50000 for c in chars])}
    
    # 创建数据集
    dataset = PretrainDataset(
        file_path='data/train.jsonl',
        tokenizer=simple_tokenizer,
        context_length=128
    )
    
    # 测试获取样本
    sample = dataset[0]
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")


if __name__ == '__main__':
    demo()
```

---

## 6. 本章小结

{IMPORTANT}本章核心要点{/IMPORTANT}

1. **数据格式**: JSONL格式是预训练数据的标准存储方式，支持流式读取和并行处理

2. **Tokenization**: 子词级别分词（如BPE、WordPiece）是现代LLM的主流选择，能有效平衡词汇表大小与语义表达

3. **数据清洗**: 完整的清洗流程包括去噪、去重、长度过滤和质量评分

4. **训练样本构建**: 通过滑动窗口将长文本切分为固定长度的训练样本

$$N_{\text{total}} = N_{\text{samples}} \times N_{\text{epochs}}$$

训练时总样本数 = 切分后样本数 × 训练轮数

---

## 7. 关键要点与思考题

### 关键要点总结

| 概念 | 关键理解 |
|------|---------|
| JSONL格式 | 流式处理友好，适合大规模数据 |
| BPE算法 | 基于频率的子词合并，迭代优化词汇表 |
| 上下文窗口 | 决定单次前向传播处理的token数量 |
| 数据清洗 | 保证训练数据质量和多样性 |

### 思考题

{QUESTION}**思考题1**: 在实际项目中，如果发现模型在某些特定词汇上表现较差（如专业术语、网络新词），可能是什么原因？应该如何优化Tokenizer？

{QUESTION}**思考题2**: 比较BPE和WordPiece两种分词算法，它们的合并策略有何不同？哪种更适合处理多语言场景？

---

**下节预告**: 下一节课我们将进入"重制Dataset：实践"环节，从理论走向代码实现，完成整个数据处理流水线的搭建。敬请期待！

---

*讲义制作: MiniMind Course*  
*版本: Episode 21 - 理论篇*