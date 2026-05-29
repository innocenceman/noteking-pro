# 第18集: 封装：CausalLM

## 课程概览：第 18/26 集「封装：CausalLM」

{IMAGE:1}

本集主题是把前面已经实现的 MiniMind 基础 Transformer 模型进一步封装成 **Causal Language Model，因果语言模型**，并补齐推理阶段最核心的 `generate` 方法。前面课程更多关注模型结构本身，例如 Embedding、Attention、MLP、Decoder Layer、RMSNorm 等；这一集则进入“大模型真正可用”的接口层：如何让模型接收 token 序列，输出下一个 token 的概率分布，并通过自回归方式不断生成文本。

{IMPORTANT}本集的核心不是重新设计 Transformer，而是把已有模型封装成符合语言建模任务的形式：输入一段 token，预测下一个 token，并在推理时循环调用模型完成文本生成。{/IMPORTANT}

### 本节小结

本集解决的是“模型能算”到“模型能生成”的过渡：CausalLM 封装让 MiniMind 从一个 Transformer Backbone 变成可训练、可推理的语言模型。

---

## 一、什么是 CausalLM

{IMAGE:2}

CausalLM 全称是 **Causal Language Model**，中文常译为“因果语言模型”或“自回归语言模型”。它的训练目标是：给定前面的 token，预测当前位置或下一个位置的 token。

例如输入：

```text
我 爱 自然 语
```

模型学习预测：

```text
爱 自然 语 言
```

也就是每个位置只能看到它左边和当前位置之前的信息，不能偷看未来 token。

### 因果性的含义

CausalLM 中的 “Causal” 主要体现在 Attention Mask 上。对于长度为 $T$ 的序列，模型第 $i$ 个位置只能关注 $j \le i$ 的位置：

$$
\text{AttentionMask}_{i,j} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

这样在计算 softmax 时，未来位置的注意力分数会被压到接近 0：

$$
\text{softmax}(q_i k_j^T + \text{mask}_{i,j})
$$

如果 $j > i$，mask 是 $-\infty$，则该位置不会被关注。

{KNOWLEDGE}CausalLM 和 BERT 类 MaskedLM 的区别在于：CausalLM 是从左到右生成，适合文本续写、聊天、代码补全；MaskedLM 是随机遮盖词再预测，适合理解类任务。{/KNOWLEDGE}

### 本节小结

CausalLM 的本质是“只能看过去，预测未来”。它通过因果 Mask 保证训练和推理逻辑一致，是 GPT 类模型的标准建模方式。

---

## 二、MiniMind 的封装目标

{IMAGE:6}

在 MiniMind 项目中，基础模型通常只负责把 token 序列转换成隐藏状态：

$$
\text{hidden\_states} = \text{MiniMindModel}(\text{input\_ids})
$$

但语言模型还需要把隐藏状态映射到词表维度，得到每个 token 的 logits：

$$
\text{logits} = \text{hidden\_states} W_{\text{lm\_head}}^T
$$

其中：

- `hidden_states` 形状通常是 `[batch_size, seq_len, hidden_size]`
- `lm_head.weight` 形状通常是 `[vocab_size, hidden_size]`
- `logits` 形状通常是 `[batch_size, seq_len, vocab_size]`

这一步就是 CausalLM 封装的核心职责之一。

### 典型封装结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMindForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Transformer 主体，负责把 token 映射为上下文隐藏状态
        self.model = MiniMindModel(config)

        # 语言模型头，将 hidden_size 映射到 vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)

        # logits: [batch, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 训练时计算 next-token prediction loss
            loss = self.compute_loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }
```

{WARNING}不要把 CausalLM 理解成一个全新的模型结构。它更多是对 Backbone 的任务封装：加上 lm_head、loss 计算、generate 推理接口。{/WARNING}

### 本节小结

MiniMindForCausalLM 的作用是把 Transformer Backbone 包装成语言模型：输入 token，输出词表 logits，并在训练时计算自回归损失。

---

## 三、语言模型头 lm_head

{IMAGE:7}

`lm_head` 是 CausalLM 中非常关键但结构上很简单的一层线性映射。它把模型内部的隐藏向量转换成对词表中每个 token 的打分。

假设某个位置的隐藏状态为：

$$
h_t \in \mathbb{R}^{d}
$$

词表大小为 $V$，语言模型头参数为：

$$
W \in \mathbb{R}^{V \times d}
$$

则该位置对所有 token 的 logits 为：

$$
z_t = W h_t
$$

其中：

$$
z_t \in \mathbb{R}^{V}
$$

每个维度对应一个 token 的未归一化分数。

### logits 到概率

模型输出 logits 后，可以通过 softmax 得到概率分布：

$$
p(x_{t+1}=v \mid x_{\le t}) =
\frac{\exp(z_v)}{\sum_{i=1}^{V} \exp(z_i)}
$$

但在训练中，一般不会手动先 softmax 再算交叉熵，而是直接使用 `CrossEntropyLoss`，因为 PyTorch 内部会更稳定地处理 `log_softmax`。

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)
```

这里的形状变化是：

- 原始 logits：`[batch, seq_len, vocab_size]`
- reshape 后：`[batch * seq_len, vocab_size]`
- labels：`[batch, seq_len]`
- reshape 后：`[batch * seq_len]`

### 本节小结

`lm_head` 把隐藏状态投影到词表空间，是模型从“理解上下文”到“预测 token”的最后一步。

---

## 四、训练目标：Next Token Prediction

{IMAGE:8}

CausalLM 的训练目标是下一个 token 预测。给定 token 序列：

$$
x = [x_1, x_2, x_3, ..., x_T]
$$

模型学习：

$$
P(x_2 \mid x_1), P(x_3 \mid x_1,x_2), ..., P(x_T \mid x_1,...,x_{T-1})
$$

整体语言模型概率可写为：

$$
P(x_1, x_2, ..., x_T)
= \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

训练时最大化这个概率，等价于最小化负对数似然：

$$
\mathcal{L}
= - \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

在实际实现中，通常会对 logits 和 labels 做错位对齐。

### shift logits 与 shift labels

{IMAGE:9}

模型在位置 $t$ 输出的是对下一个 token 的预测，所以训练时常见写法是：

```python
def compute_loss(self, logits, labels):
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]

    # 第 0 到倒数第 2 个位置的输出，用来预测第 1 到最后一个 token
    shift_logits = logits[:, :-1, :].contiguous()

    # label 从第 1 个 token 开始
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    return loss
```

举例：

```text
input_ids:  [BOS, 我, 爱, NLP, EOS]
logits位置:   0   1   2    3    4
labels:       我  爱  NLP  EOS
```

也就是说：

- `logits[0]` 预测 `我`
- `logits[1]` 预测 `爱`
- `logits[2]` 预测 `NLP`
- `logits[3]` 预测 `EOS`

最后一个位置没有下一个 token 可预测，因此通常丢弃。

{WARNING}训练 CausalLM 时最常见的错误之一是 logits 和 labels 没有正确 shift，导致模型在预测当前 token，而不是预测下一个 token。{/WARNING}

### 本节小结

CausalLM 的 loss 来自 next-token prediction。实现时要特别注意 logits 和 labels 的错位，否则训练目标会发生偏移。

---

## 五、forward 接口设计

{IMAGE:10}

一个完整的 CausalLM `forward` 通常需要支持：

- `input_ids`
- `labels`
- `attention_mask`
- `position_ids`
- `past_key_values`
- `use_cache`
- 返回 `loss` 和 `logits`

在 MiniMind 的从零实现中，可以先保留最核心的输入输出，后续再扩展 KV Cache 等推理优化。

### 简化版 forward

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits if loss is None else (loss, logits)
```

### 更规范的返回形式

实际工程中更推荐返回字典或 dataclass，因为后续字段会越来越多：

```python
return {
    "loss": loss,
    "logits": logits,
    "hidden_states": hidden_states
}
```

这样调用端不容易因为返回值顺序变化而出错。

{KNOWLEDGE}Hugging Face Transformers 中的 `CausalLMOutputWithPast` 就是类似思想：不仅返回 logits，还可以返回 loss、past_key_values、hidden_states、attentions 等。{/KNOWLEDGE}

### 本节小结

`forward` 是训练和推理共用的主入口。简化实现可以只返回 loss 和 logits，但良好的封装应为后续扩展保留空间。

---

## 六、generate：自回归生成流程

{IMAGE:11}

`generate` 是本集最重要的推理接口。它的基本思想非常直接：模型每次预测一个 token，把这个 token 拼回输入序列，再继续预测下一个 token。

### 自回归生成公式

初始输入为：

$$
x_{1:n}
$$

第 $t$ 步生成：

$$
x_{n+t} \sim P(x \mid x_{1:n+t-1})
$$

不断重复，直到达到最大长度或生成结束符 `eos_token_id`。

### 基础 generate 实现

```python
@torch.no_grad()
def generate(
    self,
    input_ids,
    max_new_tokens=100,
    eos_token_id=None
):
    self.eval()

    for _ in range(max_new_tokens):
        # 前向计算，得到所有位置的 logits
        outputs = self.forward(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        # 只取最后一个位置的 logits
        next_token_logits = logits[:, -1, :]

        # 贪心解码：选择概率最高的 token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # 拼接到输入序列末尾
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 如果生成了 eos，则提前停止
        if eos_token_id is not None:
            if torch.all(next_token == eos_token_id):
                break

    return input_ids
```

这里关键点有三个：

- 每次只根据最后一个位置的 logits 选下一个 token
- 新 token 会追加到 `input_ids` 后面
- 下一轮模型会看到更长的上下文

{IMPORTANT}generate 的本质是循环调用 forward。训练时一次性并行预测所有位置；推理时必须按 token 顺序逐步生成。{/IMPORTANT}

### 本节小结

`generate` 让模型具备真正的文本生成能力。它通过“预测一个、追加一个、继续预测”的循环完成自回归推理。

---

## 七、为什么推理只取最后一个 logits

{IMAGE:12}

模型前向输出的 logits 形状是：

```text
[batch_size, seq_len, vocab_size]
```

其中每个位置都有一个词表分布。但在生成阶段，已有输入序列中的前面 token 已经确定，不需要重新采样。我们只关心最后一个位置对下一个 token 的预测：

```python
next_token_logits = logits[:, -1, :]
```

假设输入是：

```text
我 喜欢 学习
```

模型输出：

```text
位置0 logits: 预测“喜欢”
位置1 logits: 预测“学习”
位置2 logits: 预测下一个 token
```

前两个位置对应的预测在当前生成步骤中已经没有意义，只有最后位置用于决定新 token。

### 计算上的低效

基础版 `generate` 每一步都会把完整序列重新送进模型：

```text
第1步：输入长度 n
第2步：输入长度 n+1
第3步：输入长度 n+2
...
```

这会重复计算前面 token 的 hidden states。更高效的做法是 KV Cache，也就是缓存历史 Key 和 Value，使每一步只计算新 token。

{KNOWLEDGE}本集重点是实现 generate 的正确逻辑。KV Cache 属于后续性能优化主题，不影响自回归生成的基本原理。{/KNOWLEDGE}

### 本节小结

推理时只需要最后一个位置的 logits，因为它代表“基于当前完整上下文，下一个 token 应该是什么”。

---

## 八、解码策略：Greedy、Sampling 与 Top-k

{IMAGE:13}

最简单的生成方式是贪心解码：

$$
x_{t+1} = \arg\max_v P(v \mid x_{\le t})
$$

也就是每次选择概率最高的 token。

```python
next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
```

贪心解码稳定、可复现，但缺点是容易生成单调、重复、缺乏多样性的文本。

### 温度采样 Temperature

可以使用 temperature 调整分布尖锐程度：

$$
z'_i = \frac{z_i}{T}
$$

其中 $T$ 是 temperature：

- $T < 1$：分布更尖锐，生成更保守
- $T = 1$：保持原始分布
- $T > 1$：分布更平滑，生成更多样

```python
probs = F.softmax(next_token_logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

### Top-k 采样

Top-k 只保留概率最高的 k 个 token，再从其中采样。

```python
def top_k_sample(logits, k=50, temperature=1.0):
    logits = logits / temperature

    # 取前 k 大的 logits
    values, indices = torch.topk(logits, k, dim=-1)

    # 只在 top-k 范围内计算概率
    probs = F.softmax(values, dim=-1)

    # 从 top-k 候选中采样一个位置
    sampled_pos = torch.multinomial(probs, num_samples=1)

    # 映射回原始词表 token id
    next_token = torch.gather(indices, -1, sampled_pos)
    return next_token
```

{WARNING}temperature 太高会导致输出发散；top-k 太小会导致表达受限。推理参数需要根据模型大小、训练质量和任务类型调整。{/WARNING}

### 本节小结

生成质量不仅取决于模型，也取决于解码策略。贪心解码简单稳定，采样方法则能提升多样性。

---

## 九、改进版 generate 示例

{IMAGE:14}

下面给出一个更接近实用版本的 `generate`，支持 greedy 和 top-k sampling。

```python
@torch.no_grad()
def generate(
    self,
    input_ids,
    max_new_tokens=100,
    eos_token_id=None,
    temperature=1.0,
    top_k=None,
    do_sample=False
):
    self.eval()

    for _ in range(max_new_tokens):
        outputs = self.forward(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        # 只取最后位置
        next_token_logits = logits[:, -1, :]

        if do_sample:
            # temperature 调整
            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                # 将非 top-k token 的 logits 置为 -inf
                values, _ = torch.topk(next_token_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_values,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits
                )

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # greedy decoding
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None:
            if torch.all(next_token == eos_token_id):
                break

    return input_ids
```

### 关键细节

- `@torch.no_grad()`：推理时不记录梯度，节省显存
- `self.eval()`：关闭 dropout 等训练行为
- `logits[:, -1, :]`：只使用最后位置预测
- `torch.cat`：把新 token 拼回序列
- `eos_token_id`：提前结束生成

{IMPORTANT}推理函数必须同时考虑正确性和资源消耗。即使是最简单的 generate，也应使用 `torch.no_grad()` 和 `eval()`。{/IMPORTANT}

### 本节小结

一个可用的 generate 至少应包含推理模式、最后 logits 选择、解码策略、token 拼接和结束条件。

---

## 十、输入输出形状跟踪

{IMAGE:15}

理解 CausalLM 时，最重要的调试方式之一就是跟踪 tensor shape。

假设：

```python
batch_size = 2
seq_len = 8
hidden_size = 512
vocab_size = 6400
```

则：

```text
input_ids:      [2, 8]
hidden_states:  [2, 8, 512]
logits:         [2, 8, 6400]
next_logits:    [2, 6400]
next_token:     [2, 1]
new_input_ids:  [2, 9]
```

### 训练阶段形状

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

对应：

```text
shift_logits: [2, 7, 6400]
shift_labels: [2, 7]
```

再展平：

```text
shift_logits.view(-1, 6400): [14, 6400]
shift_labels.view(-1):       [14]
```

### 推理阶段形状

每次循环：

```python
next_token_logits = logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
input_ids = torch.cat([input_ids, next_token], dim=1)
```

对应：

```text
next_token_logits: [2, 6400]
next_token:        [2, 1]
input_ids:         [2, seq_len + 1]
```

{WARNING}如果 `next_token` 不是 `[batch, 1]`，而是 `[batch]`，`torch.cat` 时经常会因为维度不一致报错。因此 `keepdim=True` 很重要。{/WARNING}

### 本节小结

CausalLM 的很多 bug 都来自 shape 不匹配。训练看 `shift`，推理看 `last logits` 和 `[batch, 1]` 的新 token。

---

## 十一、封装后的训练调用方式

{IMAGE:16}

封装为 CausalLM 后，训练代码会更清晰。数据加载器只需要提供 `input_ids` 和 `labels`，模型内部负责计算 loss。

```python
model = MiniMindForCausalLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()

for batch in dataloader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### labels 是否等于 input_ids

在标准语言模型预训练中，通常：

```python
labels = input_ids.clone()
```

因为模型学习的是同一段文本中的下一个 token。真正的错位由模型内部的 `shift_logits` 和 `shift_labels` 完成。

对于 padding token，可以把 labels 中对应位置设为 `-100`，让 loss 忽略它：

```python
labels[input_ids == pad_token_id] = -100
```

这样：

```python
F.cross_entropy(..., ignore_index=-100)
```

就不会把 padding 部分算进训练损失。

{KNOWLEDGE}`ignore_index=-100` 是 PyTorch 中语言模型训练的常见约定，也被 Hugging Face Transformers 广泛使用。{/KNOWLEDGE}

### 本节小结

CausalLM 封装让训练循环变得简单：外部只管喂数据，模型内部负责 logits 和 loss 的细节。

---

## 十二、封装后的推理调用方式

{IMAGE:17}

推理时，用户通常先把 prompt 编码成 token id，然后调用 `generate`，最后再 decode 回文本。

```python
prompt = "人工智能的未来是"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

text = tokenizer.decode(output_ids[0].tolist())
print(text)
```

### 生成过程示意

```text
初始输入：
人工智能 的 未来 是

第1步：
人工智能 的 未来 是 光明

第2步：
人工智能 的 未来 是 光明 的

第3步：
人工智能 的 未来 是 光明 的 ，

...
```

每一步都把刚生成的 token 当作下一步输入的一部分。

### 输出是否包含 prompt

基础版 `generate` 返回的是完整序列：

```text
[prompt tokens + generated tokens]
```

如果只想拿新生成部分，可以切片：

```python
new_tokens = output_ids[:, input_ids.shape[1]:]
```

{WARNING}很多初学者会误以为 generate 返回的只有新增 token。实际常见实现返回的是完整上下文加生成结果。{/WARNING}

### 本节小结

推理流程是 encode、generate、decode。生成结果通常包含原始 prompt，需要根据业务需求决定是否切片。

---

## 十三、CausalLM 与模型保存加载

{IMAGE:18}

完成 CausalLM 封装后，保存和加载模型也要以封装后的模型为单位。因为语言模型头 `lm_head` 是最终预测所必需的参数。

```python
# 保存
torch.save(model.state_dict(), "minimind_causallm.pt")

# 加载
model = MiniMindForCausalLM(config)
state_dict = torch.load("minimind_causallm.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

如果只保存 Backbone：

```python
model.model.state_dict()
```

则加载后会缺少 `lm_head` 权重，模型无法正确输出词表 logits。

### 权重绑定 Weight Tying

一些语言模型会把输入 Embedding 和输出 lm_head 的权重绑定：

$$
W_{\text{lm\_head}} = W_{\text{embed}}
$$

这样可以减少参数量，并让输入 token 表示和输出 token 分类共享语义空间。

示例：

```python
self.lm_head.weight = self.model.embed_tokens.weight
```

但权重绑定要求：

```text
hidden_size == embedding_size
```

并且实现时要确保参数共享而不是复制。

{KNOWLEDGE}GPT 类模型常使用 embedding 与 lm_head 权重绑定。MiniMind 是否使用该技巧，取决于课程实现目标和代码结构。{/KNOWLEDGE}

### 本节小结

保存 CausalLM 时必须包含 `lm_head`。权重绑定是常见优化，但需要模型维度和实现方式匹配。

---

## 十四、常见错误与调试方法

{IMAGE:19}

### 错误一：忘记 shift labels

错误写法：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1)
)
```

这会让位置 $t$ 的输出直接预测位置 $t$ 的 token，不符合 next-token prediction。

正确写法：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

### 错误二：推理时没有关闭梯度

错误写法：

```python
def generate(self, input_ids):
    outputs = self(input_ids)
```

正确写法：

```python
@torch.no_grad()
def generate(self, input_ids):
    self.eval()
    outputs = self(input_ids)
```

### 错误三：使用所有 logits 采样

错误写法：

```python
next_token = torch.argmax(logits, dim=-1)
```

这样会对每个位置都取一个 token，得到 `[batch, seq_len]`，不是下一 token。

正确写法：

```python
next_token_logits = logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
```

### 错误四：生成没有停止条件

如果不设置 `max_new_tokens` 或 `eos_token_id`，生成循环可能无法合理结束。

```python
for _ in range(max_new_tokens):
    ...
```

{IMPORTANT}调试 CausalLM 时优先检查三件事：loss shift 是否正确、logits shape 是否正确、generate 是否只取最后位置。{/IMPORTANT}

### 本节小结

CausalLM 的实现并不复杂，但细节错误很多。最重要的检查点是 shift、shape、no_grad、last logits 和停止条件。

---

## 十五、从训练到生成的整体链路

{IMAGE:20}

MiniMind CausalLM 的完整工作链路可以总结如下：

### 训练阶段

```text
文本
 -> tokenizer
 -> input_ids
 -> MiniMindForCausalLM
 -> logits
 -> shift logits / labels
 -> cross entropy loss
 -> backward
 -> update parameters
```

### 推理阶段

```text
prompt
 -> tokenizer
 -> input_ids
 -> forward
 -> last token logits
 -> decode strategy
 -> next token
 -> append
 -> repeat
 -> tokenizer.decode
 -> generated text
```

### 二者的统一性

训练和推理看似不同，但本质共享同一个条件概率模型：

$$
P(x_{t+1} \mid x_{\le t})
$$

训练时所有位置并行计算：

$$
\{P(x_2|x_1), P(x_3|x_{\le2}), ..., P(x_T|x_{<T})\}
$$

推理时一步一步串行采样：

$$
x_{t+1} \sim P(\cdot \mid x_{\le t})
$$

{KNOWLEDGE}Transformer 训练可以并行，是因为真实序列已知；推理必须串行，是因为下一个 token 依赖上一个新生成 token。{/KNOWLEDGE}

### 本节小结

CausalLM 将训练和推理统一到 next-token prediction 上。训练是并行监督学习，推理是串行自回归生成。

---

## 十六、本集结尾回顾

{IMAGE:21}

本集完成了 MiniMind 从 Backbone 到 CausalLM 的关键封装。通过添加 `lm_head`、实现 loss 计算和 `generate` 方法，模型已经具备了语言模型的基本能力。

{IMAGE:3}

{IMAGE:4}

{IMAGE:5}

### 核心代码骨架

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "loss": loss,
            "logits": logits
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, eos_token_id=None):
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            logits = outputs["logits"]

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None:
                if torch.all(next_token == eos_token_id):
                    break

        return input_ids
```

### 本节小结

第 18 集是 MiniMind 走向可用大模型的重要节点：有了 CausalLM 封装，模型才能以标准语言模型方式训练、保存、加载和生成文本。

---

## 关键 Takeaways

1. CausalLM 是 GPT 类模型的标准任务封装，核心目标是 next-token prediction。
2. `lm_head` 将隐藏状态映射到词表 logits，形状从 `[B, T, H]` 变为 `[B, T, V]`。
3. 训练 loss 必须使用 `shift_logits` 和 `shift_labels`，让当前位置预测下一个 token。
4. `generate` 是自回归循环：取最后 logits，选出 next token，拼接回输入，再继续生成。
5. 推理时应使用 `@torch.no_grad()` 和 `model.eval()`。
6. Greedy 解码稳定但单调，temperature、top-k 等采样方法能增加生成多样性。
7. 基础 generate 实现会重复计算历史 token，后续可通过 KV Cache 优化性能。

## 思考题

1. 为什么 CausalLM 训练时可以并行预测所有位置，而推理时必须一个 token 一个 token 地生成？
2. 如果忘记对 logits 和 labels 做 shift，模型实际学到的任务会发生什么变化？
3. 在 MiniMind 中加入 KV Cache 后，`generate` 的输入输出接口需要如何调整？