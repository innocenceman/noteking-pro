# 第24集: 重制Pretrain：代码

# 第24讲：重制Pretrain：代码

## 训练循环、优化器与学习率调度

{IMAGE:1}

---

## 课程概述

{IMPORTANT}本讲目标{/IMPORTANT}

- 掌握 PyTorch 训练循环的完整实现
- 理解主流优化器的原理与选择策略
- 学会设计合适的学习率调度器
- 从零实现一个完整的预训练代码框架

本讲是 MiniMind 项目预训练代码的重制版本，我们将从底层构建训练流程，深入理解每个组件的设计动机。

---

## 一、训练循环基础架构

### 1.1 训练循环的三大阶段

一个完整的训练循环包含三个核心阶段：**前向传播**、**反向传播**和**参数更新**。

$$ \text{训练循环} = \text{Forward} \rightarrow \text{Loss Computation} \rightarrow \text{Backward} \rightarrow \text{Optimizer Step} $$

{IMAGE:2}

{KNOWLEDGE}训练循环的本质{/KNOWLEDGE}

训练循环的本质是**梯度下降**的工程实现。每一次迭代，模型参数 $\theta$ 沿损失函数梯度的负方向更新：

$$ \theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t) $$

其中 $\eta$ 是学习率，$\nabla_\theta \mathcal{L}$ 是损失函数对参数的梯度。

### 1.2 基础训练循环实现

```python
# 基础训练循环框架
def train_step(model, batch, optimizer, criterion, device):
    model.train()
    
    # 1. 将数据迁移到设备
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 2. 前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 3. 计算损失
    # 对于语言模型，outputs.logits 形状为 [batch, seq_len, vocab_size]
    loss = criterion(
        outputs.logits.view(-1, outputs.logits.size(-1)),  # [B*L, V]
        labels.view(-1)                                      # [B*L]
    )
    
    # 4. 反向传播（梯度清零是必要步骤！）
    optimizer.zero_grad()  # 避免梯度累积
    loss.backward()
    
    # 5. 参数更新
    optimizer.step()
    
    return loss.item()
```

{WARNING}常见错误：梯度累积{/WARNING}

```python
# ❌ 错误写法：忘记梯度清零
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # 梯度会累积！
    optimizer.step()

# ✅ 正确写法：每步清零梯度
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 二、优化器详解

{IMAGE:3}

### 2.1 SGD（随机梯度下降）

SGD 是最基础的优化器，其更新规则为：

$$ \theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) + \lambda \theta_t $$

其中 $\lambda \theta_t$ 是 L2 正则化项（权重衰减）。

```python
import torch.optim as optim

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,        # 动量因子，加速收敛
    weight_decay=1e-4    # L2 正则化强度
)
```

### 2.2 Adam（自适应矩估计）

Adam 结合了动量法和 RMSProp 的优点，是深度学习中最常用的优化器：

$$ \begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩，即动量）} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩，即方差）} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \quad \text{（偏差校正）} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \quad \text{（偏差校正）} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{aligned} $$

{IMAGE:4}

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,             # 默认学习率
    betas=(0.9, 0.999),  # 动量和二阶矩估计的衰减率
    eps=1e-8,            # 数值稳定性常数
    weight_decay=0       # Adam 自带的 weight_decay 有别于 L2
)
```

### 2.3 AdamW（Adam with Weight Decay）

AdamW 是 LLM 训练的标准选择，它将权重衰减与自适应学习率解耦：

$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t $$

{IMAGE:5}

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1      # 独立的权重衰减
)
```

{KNOWLEDGE}Adam vs AdamW{/KNOWLEDGE}

| 特性 | Adam | AdamW |
|------|------|-------|
| 权重衰减 | 与梯度耦合 | 独立控制 |
| L2 正则化 | 等价于 weight_decay | 真正的权重衰减 |
| 收敛稳定性 | 较好 | 更好 |
| 大模型训练 | 可用 | **首选** |

---

## 三、学习率调度策略

{IMAGE:6}

学习率是训练中最重要的超参数之一。合理的学习率调度能显著提升训练稳定性和模型性能。

### 3.1 常见调度策略

#### 3.1.1 余弦退火（Cosine Annealing）

$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right) $$

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps,
    eta_min=1e-5
)
```

{IMAGE:7}

#### 3.1.2 预热+余弦退火（Warmup + Cosine）

大模型训练通常需要学习率预热（Warmup），以避免早期训练不稳定：

```python
# 分阶段学习率调度
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        # Warmup 阶段：线性增长
        optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps
        ),
        # 主阶段：余弦退火
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-5
        )
    ],
    milestones=[warmup_steps]
)
```

{IMAGE:8}

#### 3.1.3 常数值（Constant）

适用于某些特定场景：

```python
scheduler = optim.lr_scheduler.ConstantLR(
    optimizer,
    factor=1.0,
    total_iters=0
)
```

### 3.2 学习率曲线可视化

{IMAGE:9}

{IMAGE:10}

---

## 四、完整训练循环实现

{IMAGE:11}

### 4.1 训练函数框架

```python
def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    log_interval=10,
    save_interval=1000,
    save_dir="./checkpoints"
):
    """完整训练循环实现"""
    
    model.train()
    global_step = 0
    total_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # 数据迁移
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算损失（支持梯度累积）
            loss = criterion(outputs.logits, labels)
            loss = loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积逻辑
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_grad_norm
                )
                
                # 参数更新
                optimizer.step()
                scheduler.step()  # 学习率调度器更新
                optimizer.zero_grad()
                
                global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # 日志输出
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        # Epoch 结束后的验证
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
```

### 4.2 梯度累积技术

{WARNING}梯度累积的原理{/WARNING}

当 GPU 显存受限时，可以通过梯度累积模拟大 batch size 训练：

```python
# 梯度累积示例
# 目标 batch_size = 2048，但 GPU 只能容纳 batch_size = 512

effective_batch_size = 2048
micro_batch_size = 512
gradient_accumulation_steps = effective_batch_size // micro_batch_size  # = 4

# 每个 micro batch 独立前向传播，梯度累积
for micro_batch in dataloader:
    loss = model(micro_batch)
    loss.backward()  # 梯度自动累积
```

---

## 五、MiniMind 训练配置

{IMAGE:12}

### 5.1 配置文件示例

```python
# config.py
TrainingConfig = {
    # 优化器配置
    "optimizer": {
        "type": "AdamW",
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "eps": 1e-8
    },
    
    # 学习率调度配置
    "scheduler": {
        "type": "cosine",
        "warmup_ratio": 0.01,      # 预热比例
        "min_lr": 3e-5             # 最小学习率
    },
    
    # 训练配置
    "training": {
        "num_epochs": 20,
        "batch_size": 32,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "mixed_precision": True     # 混合精度训练
    }
}
```

### 5.2 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 前向传播使用 FP16
    with autocast():
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)
    
    # 反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪和更新
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scala.update()
```

---

## 六、调试与监控技巧

### 6.1 梯度监控

```python
def check_gradients(model):
    """监控梯度状态，帮助诊断训练问题"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10:
                print(f"⚠️  {name}: grad_norm = {grad_norm:.2f} (可能过大)")
            elif grad_norm < 1e-7:
                print(f"⚠️  {name}: grad_norm = {grad_norm:.2e} (可能梯度消失)")
```

### 6.2 训练异常检测

```python
# 检测 NaN/Inf 损失
if torch.isnan(loss) or torch.isinf(loss):
    print("❌ 检测到异常损失值！")
    raise ValueError("Loss is NaN or Inf")

# 检测梯度爆炸
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
if total_norm > 10:
    print(f"⚠️  梯度爆炸警告: norm = {total_norm:.2f}")
```

---

## 本讲小结

| 核心组件 | 要点 |
|---------|------|
| **训练循环** | Forward → Loss → Backward → Step 的标准流程 |
| **优化器** | AdamW 是 LLM 训练的首选 |
| **学习率调度** | Warmup + Cosine 是黄金组合 |
| **梯度累积** | 突破显存限制的有效手段 |
| **混合精度** | 加速训练 + 节省显存 |

---

## 关键收获

1. **训练循环的三阶段**：前向传播、反向传播、参数更新，缺一不可
2. **优化器选择**：AdamW 结合了动量法和自适应学习率，是大模型训练的标准选择
3. **学习率调度**：预热阶段避免早期训练不稳定，余弦退火保证收敛质量
4. **梯度累积**：通过累积多个小 batch 的梯度，模拟大 batch size 训练

---

## 思考题

1. **为什么 LLM 训练推荐使用 AdamW 而不是 Adam？权重衰减的实现方式有何本质区别？**

2. **在梯度累积场景下，学习率是否需要相应调整？如果 batch size 翻倍，训练效果会有何变化？**