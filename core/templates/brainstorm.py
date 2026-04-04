"""Brainstorm template: idea capture with mind map and action suggestions."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class BrainstormTemplate(BaseTemplate):
    name = "brainstorm"
    display_name = "灵感记录"
    description = "Idea capture with mind map structure and action suggestions"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n描述: {user_context}" if user_context else ""

        return f"""请根据以下录音转录稿，提炼其中的灵感和想法，生成结构化的灵感记录。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

录音转录稿:
{transcript}

请按以下格式输出:

# 灵感记录: [主题]

## 核心想法
[用1-2句话概括最核心的想法]

## 想法清单

### 想法 1: [想法名称]
- **描述**: [详细描述]
- **价值**: [为什么重要]
- **可行性**: ⭐⭐⭐⭐⭐ (1-5星评估)
- **下一步**: [具体行动]

### 想法 2: [想法名称]
...

## 思维导图

```
[主题]
├── [分支1]
│   ├── [子想法]
│   └── [子想法]
├── [分支2]
│   ├── [子想法]
│   └── [子想法]
└── [分支3]
    └── [子想法]
```

## 想法关联
[描述各想法之间的联系和组合可能]

## 行动建议
1. **立即执行**: [最紧急的行动]
2. **本周内**: [短期行动]
3. **后续跟进**: [长期规划]

## 灵感原话
> "[值得记录的原话]"

要求:
1. 提炼核心想法，不是逐字翻译
2. 评估每个想法的可行性
3. 给出具体可执行的行动建议
4. 用中文输出"""
