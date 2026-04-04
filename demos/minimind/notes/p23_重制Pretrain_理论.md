# 第23集: 重制Pretrain：理论

# Lecture Notes: 重制Pretrain：理论

## 课程信息

| 项目 | 内容 |
|------|------|
| **课程** | MiniMind - PyTorch从零手敲大模型 |
| **集数** | Episode 23/26 |
| **时长** | 5分44秒 |
| **主题** | 预训练目标、损失函数 |
| **Episode Title** | Retraining Pretraining: Theory |

{IMAGE:1}

---

## 课程概述

本节课将深入探讨大语言模型预训练（Pre-training）阶段的核心理论与实现细节。我们将学习：

1. **预训练的核心目标与任务设计**
2. **损失函数的设计原理**
3. **从理论到代码的完整实现**

{IMPORTANT}预训练是大型语言模型成功的基础，它使模型能够学习通用的语言表示，为后续的微调阶段奠定坚实的基础。{/IMPORTANT}

---

## 第一节：预训练核心概念

### 1.1 什么是预训练？

预训练（Pre-training）是指在大规模无标注语料库上进行的自监督学习过程。模型通过预测被掩码或破坏的部分来学习语言规律。

{KNOWLEDGE}预训练的核心思想源自迁移学习（Transfer Learning）：首先在通用任务上学习通用特征，然后在特定任务上进行微调。这一思想最早在计算机视觉领域提出，后来被自然语言处理领域广泛采用。{/KNOWLEDGE}

### 1.2 预训练 vs 微调

| 特征 | 预训练（Pre-training） | 微调（Fine-tuning） |
|------|------------------------|---------------------|
| **数据规模** | 数十亿至万亿词元 | 数千至数万样本 |
| **数据标注** | 无需人工标注（自监督） | 通常需要人工标注 |
| **目标** | 学习通用语言表示 | 适配特定任务 |
| **计算资源** | 巨大（需数百至数千GPU） | 较少（单卡即可） |
| **训练时间** | 数天至数周 | 数分钟至数小时 |

{IMAGE:2}

---

## 第二节：预训练目标详解

### 2.1 语言建模任务分类

{LANGUAGE_MODELING_TAXONOMY}

{IMAGE:3}

预训练目标主要分为两大类：

1. **自回归语言建模（Autoregressive LM / CLM）**
2. **掩码语言建模（Masked Language Modeling / MLM）**

### 2.2 自回归语言建模 (Causal LM)

{IMAGE:4}

自回归语言建模是GPT系列模型采用的核心训练目标。

**核心思想**：给定前文上下文，预测下一个词元

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, ..., x_{i-1})$$

**训练方式**：
- 输入：词元序列 $[x_1, x_2, ..., x_{n-1}]$
- 目标：预测 $[x_2, x_3, ..., x_n]$
- 使用因果掩码（Causal Mask）确保只看前文

{IMAGE:5}

**因果掩码示意图**：

```
位置:     0    1    2    3    4
输入:    [CLS]  我   爱   深   度
注意力:  [1]   [1,1] [1,1,1] [1,1,1,1] [1,1,1,1,1]
         ↑    ↑↑   ↑↑↑  ↑↑↑↑   ↑↑↑↑↑
       只看  只看  只看  只看   全部
       自己  之前  之前  之前   
```

{IMPORTANT}因果掩码确保模型在预测第$i$个词时，只能看到位置$0$到$i-1$的信息，不能"看到未来"。{/IMPORTANT}

### 2.3 掩码语言建模 (MLM)

{IMAGE:6}

掩码语言建模是BERT系列模型采用的核心训练目标。

**核心思想**：随机掩码部分词元，模型需要预测被掩码的词元

**掩码策略**（以BERT为例）：
- 80% 替换为 `[MASK]` 标记
- 10% 替换为随机词元
- 10% 保持不变

$$L_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$

其中 $M$ 是被掩码的位置集合。

{IMAGE:7}

**MLM vs CLM 对比**：

| 特性 | CLM（GPT） | MLM（BERT） |
|------|------------|-------------|
| **训练方式** | 单向 | 双向 |
| **注意力** | 因果注意力 | 全注意力 |
| **适合任务** | 生成任务 | 理解任务 |
| **模型结构** | Decoder-only | Encoder-only |
| **代表模型** | GPT-2, GPT-3 | BERT, RoBERTa |

{IMAGE:8}

### 2.4 Next Sentence Prediction (NSP)

{IMAGE:9}

NSP是BERT引入的辅助训练目标，用于学习句子级别的关系。

**任务定义**：
- 输入：句子A + [SEP] + 句子B
- 目标：判断句子B是否是句子A的下一句

**标签定义**：
- `IsNext`：B确实是A的下一句
- `NotNext`：B是随机选取的句子

**损失函数**：
$$L_{NSP} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$$

{KNOWLEDGE}后续研究发现，NSP任务对模型性能提升有限，甚至可能带来负面影响。RoBERTa等模型在去掉NSP后取得了更好的效果。这可能是因为NSP任务过于简单，模型容易通过表层特征（如主题匹配）判断，而非真正理解句子关系。{/KNOWLEDGE}

{IMAGE:10}

---

## 第三节：损失函数详解

### 3.1 交叉熵损失函数

{IMAGE:11}

对于语言建模任务，最常用的损失函数是**交叉熵损失（Cross-Entropy Loss）**。

**数学定义**：

对于单个样本：
$$L_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

其中：
- $C$ 是类别总数（词表大小）
- $y_c$ 是真实标签（one-hot编码）
- $\hat{y}_c$ 是预测概率

**对于语言模型**：
$$L_{LM} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

{IMAGE:12}

### 3.2 损失函数的实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelLoss(nn.Module):
    """
    语言模型损失计算器
    支持 CLM 和 MLM 两种训练目标
    """
    
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward_clm(self, logits, labels):
        """
        自回归语言建模损失 (用于GPT类模型)
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] - 目标词元ID
        
        Returns:
            loss: 标量张量
        """
        # 位移处理：labels是下一个词元的ID
        # logits[:, :-1] 预测第1到T-1个位置的下一个词
        # labels[:, 1:] 是第2到T个词元
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # 计算交叉熵损失
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def forward_mlm(self, logits, labels):
        """
        掩码语言建模损失 (用于BERT类模型)
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] - 目标词元ID
                   被掩码位置为真实词元ID，未掩码位置为ignore_index
        
        Returns:
            loss: 标量张量
        """
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        return loss
    
    def forward_combined(self, lm_logits, nsp_logits, lm_labels, nsp_labels):
        """
        组合损失 (BERT原始训练)
        
        Args:
            lm_logits: 语言模型logits
            nsp_logits: NSP预测logits
            lm_labels: 语言模型标签
            nsp_labels: NSP标签
        
        Returns:
            total_loss: 加权组合损失
        """
        # 语言模型损失
        lm_loss = self.forward_mlm(lm_logits, lm_labels)
        
        # NSP损失
        nsp_loss = F.cross_entropy(
            nsp_logits, 
            nsp_labels
        )
        
        # 组合权重 (BERT原论文: λ=1)
        total_loss = lm_loss + nsp_loss
        
        return total_loss
```

{WARNING}在计算CLM损失时，logits和labels必须进行位移（shift）处理。labels的第0个位置在损失计算中不使用，因为我们没有"之前的词"来预测它。{/WARNING}

### 3.3 训练过程中的损失监控

```python
class LossMonitor:
    """训练损失监控器"""
    
    def __init__(self):
        self.history = {
            'total_loss': [],
            'lm_loss': [],
            'ppl': [],  # Perplexity 困惑度
        }
    
    @staticmethod
    def compute_perplexity(loss):
        """
        计算困惑度 (Perplexity)
        困惑度越低，模型预测越准确
        
        PPL = exp(loss)
        """
        return torch.exp(loss).item()
    
    def update(self, total_loss, lm_loss=None):
        """更新损失记录"""
        self.history['total_loss'].append(total_loss.item())
        if lm_loss is not None:
            self.history['lm_loss'].append(lm_loss.item())
        self.history['ppl'].append(self.compute_perplexity(total_loss))
```

---

## 第四节：完整训练循环实现

### 4.1 数据准备

{IMAGE:13}

```python
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class PretrainDataset(Dataset):
    """
    预训练数据集
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载原始文本数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # 对文本进行分词
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 对于CLM，输入和目标相同（位移后）
        # 目标 = 输入位移一位
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # CLM使用相同序列作为labels
        }
```

### 4.2 训练步骤

```python
def training_step(model, batch, criterion, optimizer, device):
    """
    单步训练
    
    Args:
        model: 语言模型
        batch: 包含input_ids, attention_mask, labels的字典
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    Returns:
        loss: 本步损失值
    """
    # 将数据移动到设备
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    # 计算损失
    loss = criterion.forward_clm(outputs.logits, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    optimizer.step()
    
    return loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        loss = training_step(model, batch, criterion, optimizer, device)
        total_loss += loss.item()
        
        # 定期打印进度
        if batch_idx % 100 == 0:
            perplexity = LossMonitor.compute_perplexity(loss)
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | PPL: {perplexity:.2f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
```

### 4.3 完整训练脚本

```python
def main_train(
    model,
    train_dataset,
    output_dir='./checkpoint',
    epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    warmup_steps=1000,
    save_steps=5000,
):
    """完整的预训练流程"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 损失函数和优化器
    criterion = LanguageModelLoss(ignore_index=0)  # 0通常是padding
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1  # 10% warmup
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            # 训练步骤
            loss = training_step(model, batch, criterion, optimizer, device)
            
            # 更新学习率
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 定期保存检查点
            if global_step % save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, 
                               global_step, output_dir)
        
        # Epoch结束，打印统计信息
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs} 完成:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  困惑度: {LossMonitor.compute_perplexity(torch.tensor(avg_loss)):.2f}")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.2e}\n")
    
    # 保存最终模型
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir, 
                   final=True)
    print("训练完成!")

def save_checkpoint(model, optimizer, scheduler, step, output_dir, final=False):
    """保存模型检查点"""
    suffix = 'final' if final else f'step_{step}'
    save_path = f"{output_dir}/{suffix}"
    
    os.makedirs(save_path, exist_ok=True)
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"{save_path}/checkpoint.pt")
    
    # 保存模型配置
    model.config.to_json_file(f"{save_path}/config.json")
    
    print(f"检查点已保存: {save_path}")
```

---

## 第五节：训练策略与技巧

### 5.1 学习率调度

{IMAGE:14}

**常见的学习率调度策略**：

1. **Linear Warmup + Cosine Decay**
2. **OneCycleLR**
3. **Constant with Warmup**

```python
# 推荐配置
config = {
    'learning_rate': 1e-4,      # 基础学习率
    'warmup_ratio': 0.1,         # 10%的步数用于warmup
    'min_lr_ratio': 0.1,         # 最大学习率的10%
    'weight_decay': 0.01,        # 权重衰减
}
```

### 5.2 梯度处理

{WARNING}梯度问题是预训练中的常见陷阱：**
1. **梯度消失**：模型无法学习深层表示
2. **梯度爆炸**：训练不稳定，loss发散
3. **解决方案**：梯度裁剪（clip_grad_norm）+ 合适的权重初始化
{/WARNING}

```python
# 梯度裁剪示例
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,  # 梯度范数上限
    norm_type=2    # L2范数
)
```

### 5.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def training_step_fp16(model, batch, criterion, optimizer, device):
    """使用混合精度加速训练"""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    
    # 前向传播使用FP16
    with autocast():
        outputs = model(input_ids)
        loss = criterion.forward_clm(outputs.logits, labels)
    
    # 反向传播使用FP32
    scaler.scale(loss).backward()
    
    # 梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 参数更新
    scaler.step(optimizer)
    scaler.update()
    
    return loss
```

---

## 章节小结

{IMAGE:15}

### 本节核心要点

1. **预训练目标**：
   - 自回归语言建模（CLM）：GPT系列，适合生成任务
   - 掩码语言建模（MLM）：BERT系列，适合理解任务
   - NSP辅助任务：学习句子间关系

2. **损失函数**：
   - 交叉熵损失是语言建模的标准损失
   - 困惑度（Perplexity）是评估语言模型的核心指标
   - $PPL = e^{L}$，越低越好

3. **实现要点**：
   - CLM需要对logits和labels进行位移处理
   - MLM需要正确设置mask位置
   - 梯度裁剪和学习率调度对训练稳定性至关重要

4. **训练技巧**：
   - 混合精度训练可大幅加速
   - 检查点保存防止训练中断
   - 损失监控便于调试

---

## 思考题

### 思考题 1

> **为什么GPT系列模型使用因果掩码（Causal Mask）而不是双向注意力？**

*提示：考虑生成任务的特性和训练目标*

### 思考题 2

> **如果训练过程中发现困惑度（PPL）突然飙升，可能的原因有哪些？如何排查和解决？**

*提示：考虑学习率、梯度、数据处理等方面*

---

## 参考资料

1. Radford, A. et al. "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
2. Radford, A. et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
3. Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (BERT, 2018)
4. Liu, Y. et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)

---

*Notes generated for MiniMind Course - Episode 23*