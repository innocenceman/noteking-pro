# 第26集: Eval：完结！

## 课程定位：最后一集在讲什么

{IMAGE:11}

本集是 MiniMind 系列第 26/26 集，标题为 **“Eval：完结！”**。从课程结构上看，这一集不是继续扩展模型结构，而是把前面训练出的 MiniMind 模型放到 **评估与推理测试** 的场景中，回答几个关键问题：

1. 模型到底能不能用？
2. 它在推理时表现如何？
3. 如何观察模型输出质量？
4. 小模型和大模型之间的差距在哪里？
5. 后续如果继续优化，应该从哪些方向入手？

{IMPORTANT}本集的核心不是“再训练一个模型”，而是学习如何判断一个模型训练完成后是否具备可用性，以及如何通过推理测试发现模型能力边界。{/IMPORTANT}

简而言之，Eval 阶段是整个大模型开发流程中的收尾环节，也是下一轮改进的起点。

**本节小结：**  
本集将前面训练、微调、对齐等内容连接到最终效果验证，重点关注模型评估、推理测试和未来优化方向。

---

## 一、Eval 的意义：为什么训练完还必须评估

{IMAGE:1}

在机器学习和大模型开发中，训练损失下降并不等价于模型好用。尤其是语言模型，训练过程中的 loss 只能说明模型在给定数据分布上预测下一个 token 的能力有所提升，但不能直接说明它具备良好的问答、推理、遵循指令或生成能力。

模型评估的目的包括：

1. **验证训练是否成功**
2. **观察模型是否学会基础语言规律**
3. **检查指令微调是否生效**
4. **测试推理输出是否稳定**
5. **发现幻觉、重复、答非所问等问题**
6. **为后续数据、结构、训练策略调整提供依据**

{KNOWLEDGE}语言模型训练目标通常是自回归 next-token prediction，即给定前文 $x_1, x_2, ..., x_{t-1}$，预测下一个 token $x_t$。训练 loss 低，只说明模型在概率分布上更接近训练数据，但不保证它在开放式问题上表现优秀。{/KNOWLEDGE}

语言模型的基本训练目标可以写为：

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

其中：

- $x_t$ 表示第 $t$ 个 token；
- $x_{<t}$ 表示当前位置之前的上下文；
- $P(x_t \mid x_{<t})$ 是模型预测真实 token 的概率；
- loss 越低，说明模型对训练数据的拟合越好。

但 Eval 阶段关注的不只是这个数值，而是更接近人类使用体验的结果。

**本节小结：**  
训练 loss 是重要指标，但 Eval 才能真正暴露模型在实际推理、问答和文本生成中的表现。

---

## 二、推理测试：从模型参数到可读输出

{IMAGE:2}

评估 MiniMind 时，通常会加载已经训练好的模型权重，然后输入 prompt，让模型自回归生成文本。推理流程可以概括为：

1. 加载 tokenizer；
2. 构造输入 prompt；
3. 将文本编码为 token ids；
4. 输入模型；
5. 模型预测下一个 token；
6. 按采样策略选择 token；
7. 拼接到上下文中；
8. 重复生成直到达到长度上限或结束符。

一个简化版推理逻辑如下：

```python
import torch

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.8):
    # 将输入文本编码成 token id
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)

    for _ in range(max_new_tokens):
        # 前向推理，得到 logits
        logits = model(input_ids)

        # 只取最后一个位置的预测结果
        next_token_logits = logits[:, -1, :]

        # temperature 控制分布平滑程度
        next_token_logits = next_token_logits / temperature

        # softmax 得到概率分布
        probs = torch.softmax(next_token_logits, dim=-1)

        # 按概率采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼接到输入序列后面
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 如果遇到结束符，可提前停止
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist())
```

{WARNING}推理测试时常见误区是只看模型是否“能说话”。真正的评估还要看内容是否相关、逻辑是否连贯、是否遵循指令、是否重复、是否胡编。{/WARNING}

**本节小结：**  
推理测试把训练好的参数转化为真实文本输出，是观察模型能力最直接的方式。

---

## 三、采样策略对输出质量的影响

{IMAGE:3}

语言模型不是直接“写答案”，而是在每一步预测下一个 token 的概率分布。不同采样策略会显著影响最终文本。

常见策略包括：

### 1. Greedy Search

每一步都选择概率最高的 token：

$$
x_t = \arg\max_x P(x \mid x_{<t})
$$

优点是稳定、确定；缺点是容易生成死板、重复、缺少多样性的文本。

### 2. Temperature Sampling

temperature 用于调节概率分布：

$$
P_i' = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中：

- $z_i$ 是第 $i$ 个 token 的 logit；
- $T$ 是 temperature；
- $T < 1$ 时输出更保守；
- $T > 1$ 时输出更多样，但更容易跑偏。

### 3. Top-k Sampling

只在概率最高的 $k$ 个 token 中采样。这样可以过滤掉低概率噪声 token。

```python
def top_k_filtering(logits, top_k=50):
    # 取出 top_k 中最小的 logit 作为阈值
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)

    # 小于阈值的位置设为负无穷，softmax 后概率接近 0
    logits = torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits
    )
    return logits
```

### 4. Top-p / Nucleus Sampling

选择累计概率达到 $p$ 的最小 token 集合，再从中采样。它比 top-k 更自适应。

{IMPORTANT}推理效果差不一定完全是模型能力差，也可能是采样参数不合适。评估时需要固定参数，避免将采样随机性误判为模型本身问题。{/IMPORTANT}

**本节小结：**  
采样策略直接影响生成文本的稳定性、多样性和准确性，是 Eval 阶段必须记录和控制的变量。

---

## 四、模型评估维度：不能只看一个指标

{IMAGE:12}

MiniMind 作为从零实现的小型语言模型，评估时可以从多个维度观察：

### 1. 语言流畅性

看模型输出是否符合基本语法，是否存在明显乱码、断句混乱、重复 token 等问题。

### 2. 指令遵循能力

例如输入：

```text
请用三句话解释什么是 Transformer。
```

观察模型是否真的输出三句话，是否围绕 Transformer 回答。

### 3. 知识问答能力

测试常识性、课程相关、编程相关问题。不过小模型参数量有限，知识覆盖通常较弱。

### 4. 推理能力

例如：

```text
小明有 3 个苹果，又买了 2 个，一共几个？
```

看模型是否能完成简单算术和逻辑推理。

### 5. 代码能力

由于课程基于 PyTorch，可以测试模型是否能生成简单代码片段。

```python
# 示例 prompt
prompt = "用 PyTorch 写一个简单的线性层前向传播示例。"
```

### 6. 稳定性

同一个问题多次采样，观察结果是否大幅波动。

{KNOWLEDGE}大模型评估通常包含自动指标和人工评估。自动指标如 perplexity、accuracy、BLEU、ROUGE 等；人工评估关注可读性、事实性、帮助性、安全性和指令遵循。{/KNOWLEDGE}

**本节小结：**  
模型评估应覆盖语言、知识、推理、代码、稳定性等多个方面，单一指标无法完整代表模型质量。

---

## 五、Perplexity：语言模型常用指标

{IMAGE:13}

困惑度 Perplexity，简称 PPL，是语言模型中非常常见的指标。它衡量模型对测试文本的“不确定程度”。

如果平均交叉熵 loss 为 $\mathcal{L}$，则：

$$
\text{PPL} = e^{\mathcal{L}}
$$

如果使用以 2 为底的对数，也可写为：

$$
\text{PPL} = 2^{\mathcal{L}}
$$

PPL 越低，说明模型越能预测测试集中的文本。但需要注意：

1. PPL 依赖测试集分布；
2. 不同 tokenizer 下 PPL 不一定可直接比较；
3. PPL 低不等价于聊天体验好；
4. 指令模型、对话模型还需要额外评估。

一个简化的 PPL 计算示例：

```python
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_ppl(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        logits = model(input_ids)

        # 将 logits 和 labels 对齐为 next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl
```

{WARNING}PPL 更适合衡量语言建模能力，不适合作为唯一的聊天模型质量指标。一个 PPL 较低的模型仍然可能答非所问或无法遵循复杂指令。{/WARNING}

**本节小结：**  
PPL 是重要基础指标，但 Eval 最终仍要结合真实生成效果进行判断。

---

## 六、MiniMind 推理效果观察

{IMAGE:14}

在课程最后阶段，重点通常会回到 MiniMind 这个小模型本身。它的意义不是和 ChatGPT、Claude、Qwen、Llama 等大型模型直接竞争，而是帮助学习者完整理解大模型从数据、tokenizer、Transformer、预训练、SFT 到推理评估的全链路。

小模型推理时可能出现的问题包括：

1. 回答短；
2. 逻辑断裂；
3. 重复句子；
4. 知识不足；
5. 格式遵循不稳定；
6. 对复杂问题容易胡编；
7. 长上下文能力有限。

这些现象并不意外，因为模型能力受到多个因素限制：

$$
\text{Model Capability} \approx f(\text{参数量}, \text{数据质量}, \text{训练 token 数}, \text{结构设计}, \text{对齐方法})
$$

MiniMind 的主要价值在于教学：

- 它足够小，便于理解和运行；
- 它覆盖了大模型核心技术路径；
- 它能让学习者看到训练和推理之间的真实联系；
- 它适合作为后续改造、扩展、优化的实验平台。

**本节小结：**  
MiniMind 的目标是教学和实验，不是追求商业级大模型效果。评估时应结合模型规模和训练资源合理判断。

---

## 七、评估 Prompt 的设计

{IMAGE:15}

好的 Eval 需要设计一组覆盖不同能力的 prompt。可以按任务类型构造测试集。

### 1. 基础问答

```text
中国的首都是哪里？
```

观察模型是否具备基本常识。

### 2. 课程理解

```text
请解释 Transformer 中 self-attention 的作用。
```

观察模型是否掌握课程相关内容。

### 3. 数学推理

```text
一个班有 24 人，其中一半是女生，女生有多少人？
```

观察简单算术能力。

### 4. 指令遵循

```text
请用列表形式给出训练语言模型的三个步骤。
```

观察输出格式是否符合要求。

### 5. 代码生成

```text
请用 PyTorch 写一个两层 MLP。
```

观察代码结构和可执行性。

### 6. 安全与边界

```text
如果模型不知道答案，应该如何回答？
```

观察模型是否倾向于胡编。

{IMPORTANT}评估 prompt 应该固定、可复现，并尽量覆盖模型预期使用场景。否则每次测试都换问题，很难判断模型是否真的变好了。{/IMPORTANT}

**本节小结：**  
Prompt 设计决定了 Eval 的有效性。好的评估集应该覆盖常识、课程知识、推理、代码、格式和安全边界。

---

## 八、自动评估与人工评估

{IMAGE:16}

模型评估可以分为自动评估和人工评估。

### 自动评估

自动评估适合大规模、可重复运行。常见方式包括：

- loss / perplexity；
- 多选题准确率；
- 判断题准确率；
- 代码单元测试通过率；
- 与参考答案的相似度；
- 使用更强模型作为 judge。

例如，对于选择题，可以计算模型输出是否匹配标准答案：

```python
def eval_choice_answer(pred, answer):
    # 简化处理：去掉空格并统一大写
    pred = pred.strip().upper()
    answer = answer.strip().upper()

    # 判断模型输出是否包含正确选项
    return answer in pred[:5]
```

### 人工评估

人工评估更接近真实用户体验，尤其适合开放式生成任务。可以从以下维度打分：

| 维度 | 说明 |
|---|---|
| 相关性 | 是否回答了问题 |
| 正确性 | 事实和逻辑是否正确 |
| 完整性 | 是否覆盖关键点 |
| 清晰度 | 表达是否易懂 |
| 指令遵循 | 是否按要求格式回答 |
| 安全性 | 是否避免危险或虚假内容 |

{WARNING}使用大模型自动打分时要谨慎。Judge 模型本身也可能有偏差，最好配合规则、标准答案和人工抽查。{/WARNING}

**本节小结：**  
自动评估适合规模化回归测试，人工评估适合判断开放式输出质量，两者应该结合使用。

---

## 九、从 Eval 结果反推优化方向

{IMAGE:17}

Eval 的真正价值在于指导下一步改进。不同问题对应不同优化方向。

| 发现的问题 | 可能原因 | 改进方向 |
|---|---|---|
| 输出乱码 | tokenizer 或训练不充分 | 检查数据、词表、训练稳定性 |
| 重复严重 | 采样策略或数据问题 | 调整 repetition penalty、清洗数据 |
| 答非所问 | SFT 数据不足 | 增加指令微调数据 |
| 常识薄弱 | 预训练数据少 | 扩大预训练语料 |
| 推理差 | 模型规模/数据不足 | 增加推理数据、做 CoT 训练 |
| 代码差 | 代码数据不足 | 加入高质量代码语料 |
| 格式不稳定 | 指令模板不统一 | 统一 prompt 模板与训练格式 |

{KNOWLEDGE}大模型能力通常不是单一因素决定的。模型结构、参数规模、训练 token 数、数据质量、对齐方式和推理策略都会共同影响最终效果。{/KNOWLEDGE}

一个常见的改进闭环是：

$$
\text{Train} \rightarrow \text{Eval} \rightarrow \text{Diagnose} \rightarrow \text{Data/Model Update} \rightarrow \text{Retrain}
$$

也就是：

1. 训练模型；
2. 构造评估集；
3. 运行推理测试；
4. 记录失败案例；
5. 分析失败原因；
6. 修改数据或训练策略；
7. 再次训练与评估。

**本节小结：**  
Eval 不是终点，而是优化循环中的诊断环节。评估结果应该转化为具体的数据、训练和推理改进动作。

---

## 十、推理性能：速度、显存与上下文长度

{IMAGE:18}

除了生成质量，推理性能也是评估的重要部分。尤其是在本地小模型或教学模型中，需要关注：

1. 单 token 延迟；
2. 每秒生成 token 数；
3. 显存占用；
4. batch size 支持；
5. 最大上下文长度；
6. CPU / GPU 推理差异。

自回归生成的时间复杂度与序列长度密切相关。标准注意力机制复杂度约为：

$$
O(n^2 d)
$$

其中：

- $n$ 是序列长度；
- $d$ 是隐藏维度；
- $n^2$ 来自 attention 中每个 token 与其他 token 的两两交互。

在推理时，如果没有 KV Cache，每生成一个新 token 都要重复计算全部历史上下文，效率很低。使用 KV Cache 后，可以缓存过去 token 的 key 和 value，只计算新 token 的增量部分。

{IMPORTANT}KV Cache 是现代大模型推理加速的关键技术之一。它不改变模型输出逻辑，但显著减少重复计算。{/IMPORTANT}

**本节小结：**  
Eval 不只评估回答质量，也要评估推理速度、显存和上下文能力。KV Cache 是推理优化的重要方向。

---

## 十一、模型能力边界：小模型为什么会“不会”

{IMAGE:19}

MiniMind 作为教学型模型，参数量和训练数据都有限，因此在复杂任务上表现不足是正常现象。理解这种边界非常重要。

小模型常见能力边界包括：

### 1. 知识容量有限

参数量较小，无法记住大量事实知识。

### 2. 泛化能力有限

遇到训练中较少出现的任务形式，容易失败。

### 3. 长文本建模能力有限

上下文长度和训练样本长度限制了长文理解。

### 4. 指令对齐不足

如果 SFT 数据量较小，模型可能知道一些语言模式，但不稳定地遵循指令。

### 5. 推理链条较弱

多步数学、逻辑、代码推理通常需要更强的数据和模型容量支持。

可以粗略理解为：

$$
\text{能力上限} \leq \min(\text{模型容量}, \text{数据质量}, \text{训练充分度})
$$

{WARNING}不要把小模型的失败简单归因于“代码写错”。如果 loss 正常下降、推理流程正确，但开放式能力有限，往往是规模和数据带来的自然限制。{/WARNING}

**本节小结：**  
MiniMind 的不足本身也是学习重点，它帮助我们理解为什么现代大模型需要大规模参数、高质量数据和复杂训练流程。

---

## 十二、课程完整链路回顾

{IMAGE:20}

到本集为止，MiniMind 课程已经从零走完了一个大模型项目的核心流程：

1. 环境准备；
2. 数据处理；
3. tokenizer；
4. Transformer 结构；
5. attention；
6. MLP；
7. RMSNorm；
8. RoPE；
9. 预训练；
10. SFT；
11. DPO / 对齐；
12. LoRA；
13. 推理；
14. Eval。

这个流程可以抽象为：

$$
\text{Data} + \text{Model} + \text{Training} + \text{Alignment} + \text{Inference} + \text{Eval}
$$

课程的重点不是只背 API，而是理解每个模块为什么存在：

- tokenizer 负责把文本变成模型可处理的离散 token；
- embedding 把 token 映射到连续向量；
- attention 建模上下文关系；
- MLP 提供非线性变换能力；
- norm 保持训练稳定；
- RoPE 注入位置信息；
- pretrain 学语言分布；
- SFT 学指令格式；
- DPO 学偏好对齐；
- Eval 判断最终效果。

**本节小结：**  
MiniMind 的价值在于完整性。通过小规模实现，可以理解大模型工程中的主要组成部分和相互关系。

---

## 十三、从 MiniMind 走向更强模型

{IMAGE:21}

如果想在 MiniMind 基础上继续提升，可以从以下方向展开。

### 1. 数据质量提升

高质量数据通常比盲目堆数据更重要。可以考虑：

- 去重；
- 清洗乱码；
- 过滤低质量网页；
- 增加高质量问答；
- 加入代码数据；
- 构造推理数据；
- 保持 prompt 模板一致。

### 2. 模型规模提升

增大参数量通常可以提升表达能力，但也会增加训练成本。

关键参数包括：

- hidden size；
- layer 数；
- attention head 数；
- vocabulary size；
- context length。

### 3. 训练策略优化

包括：

- learning rate schedule；
- warmup；
- gradient clipping；
- mixed precision；
- batch size；
- checkpoint；
- distributed training。

### 4. 对齐数据优化

SFT 和偏好数据会直接影响聊天体验。对于对话模型，优质指令数据非常关键。

### 5. 推理工程优化

包括：

- KV Cache；
- 量化；
- batch 推理；
- speculative decoding；
- FlashAttention；
- serving 框架。

{KNOWLEDGE}现代大模型能力提升通常来自“模型规模、数据规模、训练算力、对齐技术、推理工程”的共同进步，而不是某一个单点技巧。{/KNOWLEDGE}

**本节小结：**  
MiniMind 是起点。如果继续扩展，可以围绕数据、模型、训练、对齐和推理工程五条主线推进。

---

## 十四、评估记录模板：让实验可复现

{IMAGE:22}

为了让 Eval 有意义，每次评估都应该记录实验条件。推荐记录如下信息：

| 项目 | 示例 |
|---|---|
| 模型版本 | minimind-sft-v1 |
| 权重路径 | checkpoints/sft_xxx.pt |
| tokenizer | tokenizer.model |
| prompt 模板 | chatml / alpaca / custom |
| max_new_tokens | 128 |
| temperature | 0.7 |
| top_k | 50 |
| top_p | 0.9 |
| repetition_penalty | 1.1 |
| 测试集 | eval_prompts_v1.json |
| 评估时间 | yyyy-mm-dd |
| 主要问题 | 重复、知识不足、格式不稳 |

一个简单的评估样本可以设计成 JSON 格式：

```python
eval_samples = [
    {
        "category": "instruction",
        "prompt": "请用三点说明什么是大语言模型。",
        "expected": "能够按三点解释 LLM 的定义、训练方式和应用。"
    },
    {
        "category": "reasoning",
        "prompt": "如果一本书 20 元，买 3 本多少钱？",
        "expected": "60 元。"
    },
    {
        "category": "code",
        "prompt": "用 PyTorch 写一个线性回归模型。",
        "expected": "包含 nn.Linear、forward、loss 和 optimizer。"
    }
]
```

{IMPORTANT}没有记录的 Eval 很难复现，也很难比较不同模型版本之间的进步。实验记录是模型迭代的基础设施。{/IMPORTANT}

**本节小结：**  
评估要可复现，必须记录模型版本、推理参数、prompt 集和主要失败案例。

---

## 十五、常见失败案例分析

{IMAGE:23}

Eval 阶段最有价值的材料往往不是成功样例，而是失败样例。失败样例可以揭示模型真正的短板。

### 失败 1：重复输出

示例：

```text
大语言模型是人工智能模型。大语言模型是人工智能模型。大语言模型是……
```

可能原因：

- temperature 太低；
- repetition penalty 不足；
- 训练数据中重复内容多；
- 模型容量不足导致陷入局部模式。

### 失败 2：答非所问

```text
用户：请解释 attention。
模型：今天天气很好……
```

可能原因：

- SFT 数据不足；
- prompt 模板不匹配；
- 模型没有学会指令响应格式。

### 失败 3：事实错误

```text
用户：中国的首都是哪里？
模型：上海。
```

可能原因：

- 预训练知识不足；
- 数据质量差；
- 小模型记忆能力有限。

### 失败 4：格式不稳定

```text
用户：请用 JSON 输出。
模型：好的，下面是答案：名字是小明……
```

可能原因：

- 训练集中结构化输出样本不足；
- 解码随机性过强；
- 指令遵循能力弱。

{WARNING}失败案例不要只保存一句“模型不好”。应记录 prompt、输出、推理参数、模型版本和初步原因，否则无法指导下一轮改进。{/WARNING}

**本节小结：**  
失败案例是 Eval 的关键产物。系统性记录失败，才能系统性提升模型。

---

## 十六、完结视角：从“手敲”到“理解”

{IMAGE:24}

MiniMind 课程强调“PyTorch 从零手敲大模型”，重点是把大模型从抽象概念拆成可实现、可运行、可观察的模块。通过 Eval 这一集，学习者应该能够形成完整认知：

1. 大模型不是黑盒魔法；
2. Transformer 可以逐层实现；
3. 训练目标可以用数学公式表达；
4. 推理过程就是不断预测下一个 token；
5. 模型效果需要通过 Eval 验证；
6. 失败输出能反推训练和数据问题；
7. 小模型是理解大模型的实验室。

{IMPORTANT}真正掌握课程的标志，不是跑通一次代码，而是能解释每个模块为什么存在、哪里会出问题、如何通过实验验证自己的判断。{/IMPORTANT}

**本节小结：**  
本课程的完结不是学习结束，而是从“能运行”走向“能诊断、能改进、能扩展”的开始。

---

## 十七、结合关键画面理解本集节奏

{IMAGE:25}

从时间线看，本集前半段主要围绕 Eval 与推理测试展开，中后段更偏向总结和展望。可按以下节奏理解：

- 开头：引入最终评估与课程收尾；
- 中段：展示模型推理、回答效果和测试方式；
- 后段：回顾 MiniMind 项目，讨论未来优化方向；
- 结尾：课程完结，总结学习价值。

{IMAGE:4}

这里适合重点关注模型输出是否符合预期，尤其是输入 prompt 后，模型生成内容的连贯性和相关性。

{IMAGE:26}

接近结尾时，应从单次输出转向整体工程视角：一个模型项目不是只写模型类，还包括数据、训练、推理、评估和迭代。

{IMAGE:5}

{IMAGE:6}

{IMAGE:7}

{IMAGE:8}

{IMAGE:9}

{IMAGE:10}

最后几帧可以理解为课程收束：MiniMind 的完整链路已经跑通，后续学习者可以基于这个框架继续尝试更大模型、更好数据和更完善评估体系。

**本节小结：**  
本集时间虽短，但承担了课程总结功能：既验证模型效果，也把整个大模型开发闭环串起来。

---

## 十八、实践建议：如何自己复现实验

如果你已经跟完课程，可以按以下步骤做一次完整 Eval：

1. 固定一个模型 checkpoint；
2. 固定 tokenizer；
3. 准备 20 到 100 条评估 prompt；
4. 按类别划分：常识、指令、推理、代码、课程知识；
5. 固定推理参数；
6. 保存每条输出；
7. 给每条输出打标签；
8. 汇总失败类型；
9. 根据失败类型决定下一轮改进。

推荐的输出记录结构：

```python
result = {
    "model": "minimind-sft",
    "prompt": "请解释什么是 self-attention。",
    "output": "self-attention 是一种用于计算序列中不同位置关系的机制……",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "score": {
        "relevance": 4,
        "correctness": 3,
        "clarity": 4,
        "instruction_following": 4
    },
    "notes": "基本正确，但缺少 Q/K/V 解释。"
}
```

这样做的好处是，后续模型升级后可以直接比较：

- 是否更少重复；
- 是否回答更完整；
- 是否更遵循格式；
- 是否代码更正确；
- 是否推理更稳定。

**本节小结：**  
复现实验时，关键是固定变量、记录输出、分析失败，而不是随便问几个问题后凭感觉判断模型好坏。

---

## 关键结论

1. Eval 是大模型开发闭环中不可缺少的一环。
2. 训练 loss 下降不代表模型在真实场景中好用。
3. 推理测试要关注采样参数、prompt 模板和输出稳定性。
4. PPL 是基础指标，但不能替代开放式生成评估。
5. MiniMind 的核心价值是教学和实验，不是追求商业模型效果。
6. 失败案例比成功案例更能指导下一轮优化。
7. 后续提升应从数据质量、模型规模、训练策略、对齐数据和推理工程共同入手。

## 思考题

1. 如果 MiniMind 在 Eval 中频繁答非所问，你会优先检查 prompt 模板、SFT 数据，还是模型结构？为什么？
2. 为什么 perplexity 较低的模型仍然可能在聊天任务中表现不好？
3. 如果你要为 MiniMind 设计一个 50 条样本的评估集，会如何划分任务类别？