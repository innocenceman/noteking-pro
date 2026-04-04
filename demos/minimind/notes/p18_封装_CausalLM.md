# 第18集: 封装：CausalLM

# 第18讲：封装：CausalLM（因果语言模型封装与Generate推理）

**课程**：MiniMind - PyTorch从零手敲大模型  
**集数**：第18/26讲  
**时长**：10分钟  
**关键词**：CausalLM、因果语言模型、自回归生成、generate推理、模型封装

---

## 课程导入

在前面的课程中，我们已经实现了Transformer架构的各个组件：多头注意力机制、前馈网络、旋转位置编码等。现在，我们需要将这些组件封装成一个完整的语言模型类——**CausalLM**（Causal Language Model，因果语言模型）。

{IMAGE:1}

本讲将学习：
1. 为什么要进行模型封装
2. CausalLM类的设计与实现
3. generate方法的原理与实现
4. 自回归生成的多种策略

---

## 一、从神经网络到语言模型

### 1.1 什么是因果语言模型（Causal LM）

{IMPORTANT}核心概念{/IMPORTANT}
**因果语言模型（Causal Language Model）** 是一种基于Transformer架构的自回归模型，其核心特点是：**在预测当前位置的词时，只能看到当前位置之前的所有词（因果注意力）**。

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, ..., x_{i-1})$$

这个公式表示：整个句子的概率等于每个词在给定前文条件下概率的乘积。这就是语言模型的核心——**自左向右的因果关系**。

{IMAGE:2}

### 1.2 为什么要封装成CausalLM类

{KNOWLEDGE}背景知识{/KNOWLEDGE}
封装的意义在于：

1. **简化接口**：隐藏底层实现细节，提供简洁的API
2. **统一规范**：无论是小型模型还是大型模型，都使用相同的接口
3. **可复用性**：封装后的模型可以方便地用于训练、推理、微调
4. **代码组织**：让主程序更加清晰易读

```
# 未封装：每次都要手动处理嵌入、位置编码、解码循环
logits = model.transformer(input_ids)
predictions = model.lm_head(logits)

# 封装后：一行代码即可
output = model(input_ids)
```

---

## 二、CausalLM类的核心设计

### 2.1 类结构概览

{IMAGE:3}

CausalLM类通常包含以下核心组件：

```python
class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 2. 位置编码（RoPE）
        self.rotary_emb = RotaryEmbedding(config)
        
        # 3. Transformer解码层堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # 4. 层归一化
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # 5. 语言模型头部（词汇投影层）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

### 2.2 前向传播（Forward）实现

{IMAGE:4}

```python
def forward(self, input_ids: torch.Tensor):
    """
    前向传播
    
    参数:
        input_ids: 输入token序列，shape [batch_size, seq_len]
    
    返回:
        logits: 预测的logits，shape [batch_size, seq_len, vocab_size]
    """
    # 1. 词嵌入 + 位置信息
    hidden_states = self.embed_tokens(input_ids)
    
    # 2. 获取位置编码
    position_ids = torch.arange(
        input_ids.size(1), 
        device=input_ids.device
    ).unsqueeze(0)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    
    # 3. 通过每一层Transformer块
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings
        )
    
    # 4. 最终归一化
    hidden_states = self.norm(hidden_states)
    
    # 5. 投影到词汇表
    logits = self.lm_head(hidden_states)
    
    return logits
```

{IMAGE:5}

---

## 三、Generate推理详解

### 3.1 自回归生成的原理

{IMPORTANT}核心概念{/IMPORTANT}
**自回归生成（Autoregressive Generation）** 是语言模型生成文本的核心机制：

1. 给定起始token（如 `<sos>` 或 `BOS`）
2. 模型预测下一个token的概率分布
3. 根据策略选择一个token
4. 将新token加入序列，重复步骤2-3
5. 直到生成终止token（如 `<eos>` 或 `EOS`）或达到最大长度

{IMAGE:6}

```python
def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
    """
    自回归文本生成
    
    参数:
        input_ids: 输入token序列
        max_new_tokens: 最多生成的新token数
        temperature: 温度参数，控制随机性
    
    返回:
        generated_ids: 生成的完整序列
    """
    self.eval()  # 切换到评估模式
    
    for _ in range(max_new_tokens):
        # 限制输入长度（避免超过模型最大上下文）
        input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings \
                         else input_ids[:, -self.config.max_position_embeddings:]
        
        # 前向传播获取logits
        logits = self.forward(input_ids_cond)
        
        # 只取最后一个位置的logits
        logits = logits[:, -1, :] / temperature
        
        # 获取词汇表大小的概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 采样下一个token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 追加到序列
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 如果生成了终止token，提前结束
        if next_token.item() == self.config.eos_token_id:
            break
    
    return input_ids
```

{IMAGE:7}

### 3.2 采样策略详解

{IMAGE:8}

{KNOWLEDGE}背景知识{/KNOWLEDGE}
不同的采样策略会影响生成结果的多样性和质量：

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| **Greedy Search** | 始终选择概率最高的词 | 需要确定性的结果 |
| **Temperature Sampling** | 调整概率分布的锐度 | 平衡多样性与质量 |
| **Top-K Sampling** | 只从top-k个最高概率词中采样 | 避免低概率词被选中 |
| **Top-P (Nucleus) Sampling** | 选取累积概率达p的最小词集合 | 更动态的采样范围 |

#### Greedy Search

```python
def greedy_search(logits):
    """贪婪搜索：始终选择概率最高的token"""
    return torch.argmax(logits, dim=-1, keepdim=True)
```

#### Temperature Sampling

{IMAGE:9}

```python
def temperature_sample(logits, temperature=1.0):
    """
    温度采样
    
    温度 > 1：分布更平滑，增加随机性
    温度 < 1：分布更尖锐，减少随机性
    温度 = 1：保持原始概率分布
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

#### Top-K Sampling

```python
def top_k_sample(logits, k=50):
    """Top-K采样：只从概率最高的k个词中采样"""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # 将不在top-k中的logits设为-inf
    logits = torch.full_like(logits, float('-inf'))
    logits.scatter_(1, top_k_indices, top_k_logits)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

#### Top-P (Nucleus) Sampling

{IMAGE:10}

```python
def top_p_sample(logits, p=0.9):
    """
    Top-P（核）采样：
    选取累积概率刚好超过p的最小词集合进行采样
    """
    # 按概率降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # 找到累积概率刚超过p的位置
    nucleus_mask = cumulative_probs > p
    # 保证至少包含一个token
    nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
    nucleus_mask[..., 0] = True
    
    # 过滤logits
    filtered_logits = torch.where(nucleus_mask, sorted_logits, float('-inf'))
    filtered_probs = F.softmax(filtered_logits, dim=-1)
    
    return sorted_indices.gather(1, torch.multinomial(filtered_probs, 1))
```

### 3.3 完整的Generate方法

{IMAGE:11}

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    repetition_penalty: float = 1.0
):
    """
    完整的生成方法
    
    参数:
        input_ids: 输入token序列 [batch_size, seq_len]
        max_new_tokens: 最大生成token数
        temperature: 温度参数 (0.0, 1.0]
        top_k: Top-K采样参数，None则不使用
        top_p: Top-P采样参数，None则不使用
        repetition_penalty: 重复惩罚 >1.0 抑制重复
    
    返回:
        generated_ids: 生成的完整序列
    """
    self.eval()
    batch_size = input_ids.size(0)
    
    # 记录已生成的token用于去重惩罚
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # 1. 上下文长度限制
        model_input = self._trim_context(generated)
        
        # 2. 前向传播
        logits = self.forward(model_input)
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 3. 应用重复惩罚
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                for token_id in set(generated[i].tolist()):
                    next_token_logits[i, token_id] /= repetition_penalty
        
        # 4. 应用温度
        if temperature != 1.0:
            next_token_logits /= temperature
        
        # 5. 应用采样策略
        if top_k is not None:
            next_token_logits = self._apply_top_k(next_token_logits, top_k)
        
        if top_p is not None:
            next_token_logits = self._apply_top_p(next_token_logits, top_p)
        
        # 6. 转换为概率并采样
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 7. 追加到序列
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 8. 检查是否所有序列都已生成终止符
        if (next_token == self.config.eos_token_id).all():
            break
    
    return generated
```

---

## 四、实战：完整的CausalLM实现

### 4.1 配置类定义

```python
@dataclass
class ModelConfig:
    """模型配置类"""
    vocab_size: int = 32000          # 词汇表大小
    hidden_size: int = 512           # 隐藏层维度
    intermediate_size: int = 1376    # FFN中间层维度
    num_layers: int = 8              # Transformer层数
    num_heads: int = 8               # 注意力头数
    max_position_embeddings: int = 512  # 最大位置长度
    rope_theta: float = 10000.0       # RoPE基础频率
    dropout: float = 0.0              # Dropout比例
    eps: float = 1e-6                # LayerNorm epsilon
```

### 4.2 完整模型代码

```python
class CausalLM(nn.Module):
    """
    因果语言模型
    
    将Transformer组件封装为完整的语言模型，
    支持训练和推理两种模式。
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config.num_layers)
        ])
        
        # 最终归一化
        self.norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.eps
        )
        
        # 语言模型头部
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # 权重绑定（可选）：词嵌入和LM头部共享权重
        self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 归一化
        hidden_states = self.norm(hidden_states)
        
        # 投影到词汇表
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, input_ids, **kwargs):
        """文本生成接口"""
        # 简化实现，实际使用上面完整的generate方法
        # ...
```

---

## 五、使用示例与注意事项

### 5.1 基本使用

```python
# 初始化模型
config = ModelConfig()
model = CausalLM(config)
model.eval()

# 准备输入
input_text = "今天天气"
input_ids = tokenizer.encode(input_text)
input_tensor = torch.tensor([input_ids])

# 生成
with torch.no_grad():
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )

# 解码输出
output_text = tokenizer.decode(output_ids[0])
print(output_text)
```

{WARNING}易错点{/WARNING}
1. **模型模式切换**：推理时务必调用 `model.eval()`，确保BatchNorm和Dropout行为正确
2. **上下文长度**：避免输入超过模型支持的最大长度
3. **设备一致性**：输入tensor和模型必须在同一设备上（CPU/GPU）
4. **重复惩罚**：过高的repetition_penalty可能导致生成质量下降

---

## 六、章节小结

{IMPORTANT}核心概念{/IMPORTANT}

| 概念 | 说明 |
|------|------|
| **CausalLM** | 因果语言模型，只能看到当前位置之前的信息 |
| **自回归生成** | 逐token生成，每个token依赖前文 |
| **温度采样** | 通过调整温度控制随机性 |
| **Top-K/P采样** | 截断低概率token，提高生成质量 |
| **模型封装** | 隐藏实现细节，提供统一接口 |

本讲的核心要点：
1. CausalLM封装了Transformer架构，提供简洁的forward和generate接口
2. generate方法是自回归的，需要循环调用forward
3. 不同的采样策略（greedy/temperature/top-k/top-p）适用于不同场景
4. 实际应用中需要注意设备、上下文长度、模式切换等问题

---

## 思考题

1. **为什么因果语言模型只能使用因果注意力，而不能使用双向注意力？**
   > 提示：考虑语言模型的核心任务是**预测下一个词**，双向注意力会导致信息泄露。

2. **在generate过程中，为什么要限制输入序列的长度？有什么方法可以在不损失信息的情况下处理超长文本？**
   > 提示：考虑滑动窗口、稀疏注意力、RWKV架构等方案。

---

**下一讲预告**：我们将学习**模型训练流程**，包括损失函数计算、反向传播优化等核心内容。

---

*注：本讲义中的代码示例基于MiniMind课程实现，具体细节可能与实际代码有所出入。*