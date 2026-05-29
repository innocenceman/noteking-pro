# 第24集: 重制Pretrain：代码

## 课程概览：重制 Pretrain 代码在做什么

{IMAGE:1}

本集是 MiniMind 预训练代码重制部分的核心实战课，重点围绕三个问题展开：

1. 如何写一个稳定的训练循环；
2. 如何配置优化器，尤其是 AdamW；
3. 如何设计学习率调度，让模型在预训练阶段更稳定地收敛。

预训练不是简单地“把数据喂给模型然后反向传播”。对于大模型而言，训练循环需要同时处理数据加载、前向传播、损失计算、梯度累积、混合精度、梯度裁剪、优化器更新、学习率调度、日志记录、保存 checkpoint 等多个环节。

{IMPORTANT}预训练代码的核心目标不是“能跑”，而是“稳定、可恢复、可扩展地跑”。{/IMPORTANT}

本节可以理解为 MiniMind 从模型结构走向真实训练工程的关键一步。

**本节小结：**  
预训练代码的主体由训练循环、优化器、学习率调度和工程化辅助逻辑组成，目标是让模型稳定学习语言建模任务。

---

## 一、预训练任务的基本形式

{IMAGE:6}

MiniMind 的预训练一般采用自回归语言模型目标，也就是让模型根据前面的 token 预测下一个 token。

给定输入序列：

$$
x = [x_1, x_2, x_3, ..., x_T]
$$

模型要学习：

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

训练时通常会构造：

```python
input_ids = batch[:, :-1]
labels = batch[:, 1:]
```

也就是说，输入是前 $T-1$ 个 token，标签是后移一位的 token。

例如：

```text
原始序列:   我  爱  自  然  语  言
输入:       我  爱  自  然  语
标签:       爱  自  然  语  言
```

模型输出 logits 后，与 labels 做交叉熵损失：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

在代码中通常对应：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)
```

{KNOWLEDGE}语言模型预训练本质是一个大规模分类任务。每个位置都要从词表大小 $V$ 中预测下一个 token，因此 logits 的形状通常是 `[batch_size, seq_len, vocab_size]`。{/KNOWLEDGE}

**本节小结：**  
预训练数据通过“输入右移一位、标签左移一位”的方式构造，模型目标是最小化下一个 token 的交叉熵损失。

---

## 二、训练循环的整体结构

{IMAGE:7}

一个标准的 PyTorch 训练循环通常包含以下步骤：

1. 设置模型为训练模式；
2. 从 DataLoader 中取出 batch；
3. 将数据移动到 GPU；
4. 前向传播得到 logits 和 loss；
5. 反向传播计算梯度；
6. 梯度裁剪；
7. 优化器更新参数；
8. 学习率调度器更新；
9. 清空梯度；
10. 记录日志与保存模型。

示例代码结构如下：

```python
model.train()

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

其中每一步都很重要。

`model.train()` 会启用 Dropout 等训练行为；`loss.backward()` 会根据损失计算梯度；`optimizer.step()` 根据梯度更新模型参数；`optimizer.zero_grad()` 清空上一轮梯度，避免梯度错误累加。

{WARNING}如果忘记调用 `optimizer.zero_grad()`，梯度会不断累积，训练会变得异常不稳定。除非你是在有意做梯度累积，否则每次参数更新后都应该清空梯度。{/WARNING}

**本节小结：**  
训练循环的基本流程是前向传播、损失计算、反向传播、参数更新、学习率更新和梯度清零。

---

## 三、梯度累积：小显存训练大 batch

{IMAGE:8}

大模型训练通常希望使用较大的 batch size，因为更大的 batch 可以让梯度估计更稳定。但显存有限时，无法一次放入很大的 batch，于是可以使用梯度累积。

假设真实想要的 batch size 是：

$$
B_{\text{global}} = B_{\text{micro}} \times N_{\text{accum}}
$$

其中：

- $B_{\text{micro}}$ 是每次实际送入 GPU 的小 batch；
- $N_{\text{accum}}$ 是梯度累积步数；
- $B_{\text{global}}$ 是等效全局 batch size。

代码示例：

```python
gradient_accumulation_steps = 4

for step, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss

    # 平均到每个累积步，避免梯度放大
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

这里最关键的是：

```python
loss = loss / gradient_accumulation_steps
```

如果不除以累积步数，等效梯度会被放大 $N_{\text{accum}}$ 倍，学习率相当于被隐式放大，容易导致 loss 爆炸。

{IMPORTANT}梯度累积的本质是多次 forward/backward 后再进行一次 optimizer.step()，从而模拟更大的 batch size。{/IMPORTANT}

**本节小结：**  
梯度累积可以在显存有限的情况下模拟大 batch 训练，但必须正确缩放 loss，并且只在累积完成后更新优化器和学习率。

---

## 四、优化器：为什么常用 AdamW

{IMAGE:9}

在大模型训练中，最常用的优化器是 AdamW，而不是普通 SGD。

Adam 的核心思想是为每个参数维护一阶矩估计和二阶矩估计：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

其中：

- $g_t$ 是当前梯度；
- $m_t$ 是梯度的一阶动量；
- $v_t$ 是梯度平方的二阶动量；
- $\beta_1$ 通常取 0.9；
- $\beta_2$ 通常取 0.95 或 0.999。

AdamW 相比 Adam 的关键区别在于权重衰减方式。AdamW 将 weight decay 从梯度更新中解耦出来，使正则化行为更加稳定。

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

{KNOWLEDGE}AdamW 中的 W 表示 decoupled weight decay，即解耦权重衰减。它不是简单地把 $L_2$ 正则项加到 loss 上，而是在参数更新时单独处理权重衰减。{/KNOWLEDGE}

**本节小结：**  
AdamW 是大模型预训练中的常用优化器，具有自适应学习率和更合理的权重衰减机制。

---

## 五、参数分组：哪些参数不应该 weight decay

{IMAGE:10}

虽然 AdamW 通常会设置 weight decay，但不是所有参数都适合衰减。

一般来说，需要 weight decay 的参数：

- Linear 层的权重；
- Embedding 权重，有时也会 decay，具体取决于实现策略。

不适合 weight decay 的参数：

- bias；
- LayerNorm / RMSNorm 的权重；
- 其他归一化层参数。

常见写法是将参数分成两组：

```python
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if name.endswith("bias") or "norm" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95)
)
```

这样做可以避免对归一化参数施加不必要的收缩。

{WARNING}对 LayerNorm 或 RMSNorm 参数使用 weight decay，可能会影响归一化层的尺度学习能力，从而降低训练稳定性。{/WARNING}

**本节小结：**  
优化器参数分组是预训练代码中的重要细节，通常对权重使用 weight decay，而对 bias 和 norm 参数禁用 weight decay。

---

## 六、学习率调度：为什么不能固定学习率

{IMAGE:11}

预训练中很少全程使用固定学习率。原因是：

1. 训练初期模型参数随机，过大的学习率容易导致不稳定；
2. 训练中期需要较大学习率提升收敛速度；
3. 训练后期需要逐步降低学习率，让模型细致收敛。

因此常见策略是：

- 前期 warmup；
- 后期 cosine decay。

学习率变化可以表示为：

$$
lr(t) =
\begin{cases}
lr_{\max} \cdot \frac{t}{T_{\text{warmup}}}, & t < T_{\text{warmup}} \\
lr_{\min} + \frac{1}{2}(lr_{\max} - lr_{\min}) \left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right), & t \geq T_{\text{warmup}}
\end{cases}
$$

其中：

- $t$ 是当前 step；
- $T_{\text{warmup}}$ 是 warmup 步数；
- $T_{\text{total}}$ 是总训练步数；
- $lr_{\max}$ 是峰值学习率；
- $lr_{\min}$ 是最小学习率。

{IMPORTANT}学习率调度决定了模型“什么时候大胆学，什么时候谨慎收敛”。对于预训练稳定性非常关键。{/IMPORTANT}

**本节小结：**  
预训练通常使用 warmup + cosine decay 学习率策略，避免初期震荡并提升后期收敛质量。

---

## 七、手写 Cosine Learning Rate Scheduler

{IMAGE:12}

如果不使用现成的 scheduler，也可以手写学习率函数：

```python
import math

def get_lr(iter_num, warmup_iters, max_iters, learning_rate, min_lr):
    # 1. warmup 阶段：线性升高
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters

    # 2. 超过总步数后：保持最小学习率
    if iter_num > max_iters:
        return min_lr

    # 3. cosine decay 阶段
    decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)
```

在训练循环中使用：

```python
lr = get_lr(
    iter_num=global_step,
    warmup_iters=warmup_iters,
    max_iters=max_iters,
    learning_rate=learning_rate,
    min_lr=min_lr
)

for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

这种写法非常直观，也便于理解学习率在每个 step 的变化。

{KNOWLEDGE}PyTorch 的 scheduler 本质上也是在每个训练 step 或 epoch 修改 optimizer.param_groups 中的 lr。{/KNOWLEDGE}

**本节小结：**  
手写学习率调度函数可以帮助理解 warmup 和 cosine decay 的具体计算过程，也方便根据项目需求灵活调整。

---

## 八、混合精度训练：提升速度与节省显存

{IMAGE:13}

大模型训练常使用混合精度，例如 FP16 或 BF16。它可以降低显存占用并提升计算速度。

在 PyTorch 中可以使用 `torch.cuda.amp.autocast`：

```python
scaler = torch.cuda.amp.GradScaler(enabled=True)

for step, batch in enumerate(train_loader):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

如果使用 BF16，很多新显卡上可以不使用 GradScaler：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
```

{WARNING}使用 AMP 时，如果要进行梯度裁剪，需要先 `scaler.unscale_(optimizer)`，否则裁剪的是缩放后的梯度，结果不正确。{/WARNING}

**本节小结：**  
混合精度可以显著节省显存和提升训练速度，但要注意 GradScaler、反向传播和梯度裁剪的顺序。

---

## 九、梯度裁剪：防止梯度爆炸

{IMAGE:14}

预训练过程中，某些 batch 可能导致梯度非常大，从而让参数更新异常剧烈，出现 loss 突然爆炸。

梯度裁剪常用写法：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

它的作用是限制所有参数梯度的整体范数：

$$
\|g\|_2 \leq C
$$

如果当前梯度范数超过 $C$，则按比例缩小：

$$
g' = g \cdot \frac{C}{\|g\|_2}
$$

其中 $C$ 就是 `max_norm`。

{IMPORTANT}梯度裁剪不会改变梯度方向，只会在梯度过大时缩小梯度长度。{/IMPORTANT}

在大模型训练中，`max_norm=1.0` 是非常常见的设置。

**本节小结：**  
梯度裁剪是训练稳定性的重要保护机制，尤其适用于 Transformer 这类深层模型。

---

## 十、global step 与训练进度管理

{IMAGE:15}

训练代码中通常会维护一个 `global_step`，它表示已经完成了多少次参数更新，而不是读了多少个 batch。

尤其在使用梯度累积时，`step` 和 `global_step` 不一定相同：

```python
global_step = 0

for epoch in range(num_epochs):
    for micro_step, batch in enumerate(train_loader):
        loss = model(**batch).loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (micro_step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
```

学习率调度通常应该基于 `global_step`，因为学习率应当随着真实参数更新次数变化，而不是随着 micro batch 次数变化。

{WARNING}使用梯度累积时，如果每个 micro step 都调用 scheduler.step()，学习率会下降得过快。正确做法通常是在 optimizer.step() 后调用 scheduler.step()。{/WARNING}

**本节小结：**  
`global_step` 应该对应真实参数更新次数，是学习率调度、日志记录和 checkpoint 命名的重要依据。

---

## 十一、日志记录：观察 loss、lr 与训练速度

{IMAGE:16}

训练时需要定期打印或记录关键指标：

- 当前 step；
- 当前 loss；
- 当前 learning rate；
- 每秒处理 token 数；
- 梯度范数；
- epoch 或数据进度；
- 预计剩余时间。

示例：

```python
if global_step % log_interval == 0:
    print(
        f"step {global_step}: "
        f"loss {loss.item():.4f}, "
        f"lr {lr:.6e}"
    )
```

更完整的训练速度统计可以使用：

```python
tokens_per_iter = batch_size * seq_len * gradient_accumulation_steps
tokens_per_second = tokens_per_iter / elapsed_time
```

训练速度可以帮助判断数据加载、模型计算或通信是否存在瓶颈。

{KNOWLEDGE}loss 是训练质量指标，lr 是训练策略指标，tokens/s 是训练效率指标。三者一起观察，才能判断训练是否正常。{/KNOWLEDGE}

**本节小结：**  
日志不是装饰，而是训练监控系统。良好的日志可以帮助快速发现 loss 异常、学习率错误和性能瓶颈。

---

## 十二、保存与恢复 Checkpoint

{IMAGE:17}

长时间预训练必须支持 checkpoint 保存和恢复，否则一次中断就可能损失大量训练进度。

保存 checkpoint 时通常包含：

- model state；
- optimizer state；
- scheduler state；
- global step；
- epoch；
- config；
- random seed 状态，视项目需要而定。

示例：

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "config": config,
}

torch.save(ckpt, "pretrain_ckpt.pt")
```

恢复：

```python
ckpt = torch.load("pretrain_ckpt.pt", map_location=device)

model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])

global_step = ckpt["global_step"]
start_epoch = ckpt["epoch"]
```

{IMPORTANT}只保存模型参数不够。预训练恢复时，如果不恢复 optimizer 状态，AdamW 的动量信息会丢失，训练轨迹会发生明显变化。{/IMPORTANT}

**本节小结：**  
checkpoint 是预训练工程的基本能力，应同时保存模型、优化器和训练进度，保证训练可恢复。

---

## 十三、完整训练循环示例

{IMAGE:18}

下面是一个整合了梯度累积、学习率调度、梯度裁剪和日志记录的简化版训练循环：

```python
import math
import torch

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step >= total_steps:
        return min_lr

    ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


model.train()
global_step = 0
optimizer.zero_grad(set_to_none=True)

for epoch in range(num_epochs):
    for micro_step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 根据真实更新步数设置学习率
        lr = get_lr(
            global_step,
            warmup_steps,
            total_steps,
            max_lr,
            min_lr
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # 前向传播
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # 梯度累积时缩放 loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # 到达累积步数后才更新参数
        if (micro_step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % log_interval == 0:
                print(
                    f"step={global_step}, "
                    f"loss={loss.item() * gradient_accumulation_steps:.4f}, "
                    f"lr={lr:.6e}"
                )

            if global_step % save_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    f"ckpt_{global_step}.pt"
                )
```

这里需要注意，日志中打印 loss 时乘回了 `gradient_accumulation_steps`，是为了显示原始 loss 尺度。

**本节小结：**  
完整训练循环要把多个细节正确串起来，尤其是 loss 缩放、学习率更新时机、梯度裁剪顺序和 checkpoint 保存。

---

## 十四、训练循环中的常见错误

{IMAGE:19}

### 1. 忘记 `model.train()`

如果模型中有 Dropout，不调用 `model.train()` 会导致训练行为不正确。

```python
model.train()
```

### 2. scheduler 调用频率错误

使用梯度累积时，scheduler 应该跟随 optimizer 更新，而不是每个 micro batch 更新。

```python
if should_update:
    optimizer.step()
    scheduler.step()
```

### 3. 忘记缩放 loss

```python
loss = loss / gradient_accumulation_steps
```

否则等效梯度会变大。

### 4. 梯度裁剪位置错误

如果使用 AMP，需要先 unscale 再裁剪。

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. 没有保存 optimizer 状态

只保存模型会导致恢复训练不连续。

{WARNING}训练代码最容易出问题的地方不是模型结构，而是这些看似普通的工程细节。{/WARNING}

**本节小结：**  
预训练代码的稳定性来自细节：训练模式、梯度累积、学习率调度、AMP 顺序和 checkpoint 都不能写错。

---

## 十五、从代码角度理解训练稳定性

{IMAGE:20}

训练稳定性可以从以下几个维度理解：

### 损失函数稳定

交叉熵应平稳下降，短期有波动是正常的，但长期不应爆炸。

### 梯度稳定

梯度范数不应长期为 0，也不应频繁极大。

### 学习率稳定

学习率应符合预设曲线，warmup 后逐渐进入 decay。

### 数据稳定

数据 batch 应 shape 正确、token 合法、label 对齐。

### checkpoint 稳定

中断后能够恢复到相近训练状态。

如果训练 loss 出现 NaN，通常要检查：

```python
if torch.isnan(loss):
    print("loss is NaN")
    break
```

也可以检查梯度：

```python
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient found in {name}")
```

{KNOWLEDGE}训练稳定性不是单个技巧带来的，而是优化器、学习率、梯度处理、数据质量和数值精度共同作用的结果。{/KNOWLEDGE}

**本节小结：**  
调试预训练代码要同时关注 loss、梯度、学习率、数据和 checkpoint，不能只盯着模型结构。

---

## 十六、MiniMind 预训练代码的工程视角

{IMAGE:21}

对于 MiniMind 这样的教学型大模型项目，预训练代码通常追求两点平衡：

1. 足够简单，方便理解；
2. 足够完整，接近真实训练流程。

因此代码中通常不会一开始就引入过多复杂分布式框架，而是先用 PyTorch 原生能力把核心训练逻辑讲清楚：

```python
model = MiniMindLM(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=config.weight_decay
)

for epoch in range(config.epochs):
    train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        config=config
    )
```

这种结构适合教学，因为它把主线保持得很清楚：

- 模型负责计算 logits；
- DataLoader 负责提供 batch；
- loss 负责衡量预测错误；
- optimizer 负责更新参数；
- scheduler 负责控制学习率；
- checkpoint 负责保存训练状态。

{IMPORTANT}理解单卡训练循环，是理解分布式训练、LoRA 微调、SFT、RLHF 等后续训练范式的基础。{/IMPORTANT}

**本节小结：**  
MiniMind 的预训练代码虽然简化，但已经包含真实大模型训练中最关键的工程思想。

---

## 十七、结尾画面与课程串联

{IMAGE:3}

{IMAGE:22}

{IMAGE:4}

{IMAGE:5}

本集把“预训练代码”从抽象概念落到了具体实现。前面课程中已经完成了模型结构、Tokenizer、数据处理等模块，而这一集将它们串联成真正可以训练的流程。

可以把整个流程理解为：

```text
原始文本
  -> Tokenizer 编码
  -> Dataset / DataLoader
  -> MiniMind 模型前向传播
  -> CrossEntropy Loss
  -> backward 计算梯度
  -> AdamW 更新参数
  -> 学习率调度控制步长
  -> checkpoint 保存训练状态
```

预训练代码并不是单一函数，而是多个模块之间的协作系统。

**本节小结：**  
本集完成了从数据到模型更新的闭环，是 MiniMind 从“模型定义”进入“模型训练”的关键一步。

---

## Key Takeaways

1. 预训练的核心目标是下一个 token 预测，本质是自回归语言建模。
2. 训练循环的标准顺序是前向传播、loss、反向传播、梯度裁剪、优化器更新、学习率更新、清空梯度。
3. 梯度累积可以模拟大 batch，但必须对 loss 除以累积步数。
4. AdamW 是大模型训练常用优化器，通常要对 bias 和 norm 参数关闭 weight decay。
5. 学习率调度通常采用 warmup + cosine decay，学习率应跟随真实参数更新步数变化。
6. AMP 可以提升速度和节省显存，但梯度裁剪前要注意 unscale。
7. checkpoint 应保存 model、optimizer、global_step 等状态，保证训练可恢复。
8. 稳定训练依赖多个细节共同正确，而不是某一个单独技巧。

## 思考题

1. 如果使用梯度累积但忘记把 loss 除以 `gradient_accumulation_steps`，训练会发生什么变化？
2. 为什么 AdamW 中的 LayerNorm / RMSNorm 参数通常不做 weight decay？
3. 学习率 warmup 为什么对大模型预训练尤其重要？