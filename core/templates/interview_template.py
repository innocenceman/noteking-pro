"""Interview template: Q&A structure with key insights and positions."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class InterviewTemplate(BaseTemplate):
    name = "interview"
    display_name = "访谈记录"
    description = "Structured interview notes with Q&A, key insights, and speaker positions"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        speakers = ctx.extra.get("speakers", [])

        context_info = f"\n访谈描述: {user_context}" if user_context else ""
        speaker_info = f"\n参与者: {', '.join(speakers)}" if speakers else ""

        return f"""请根据以下访谈/对话录音转录稿，生成一份结构化的访谈记录。

访谈标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}{speaker_info}

访谈录音转录稿:
{transcript}

请按以下格式输出:

# 访谈记录: [标题]

## 访谈概要
- **主题**: [访谈主题]
- **参与者**: [采访者和受访者的身份]
- **核心观点**: [一句话概括]

## 人物介绍
[每位参与者的简要背景介绍]

## Q&A 结构化整理

### Q1: [问题概括]
**[受访者]**: [完整回答要点]
> "[原话引用]"

### Q2: [问题概括]
...

## 关键观点提取
1. **[观点1]**: [详细说明] —— [发言人]
2. **[观点2]**: [详细说明] —— [发言人]

## 金句收集
- > "[金句1]" —— [发言人]
- > "[金句2]" —— [发言人]

## 人物立场分析
| 参与者 | 核心立场 | 支撑论据 |
|--------|---------|---------|

## 访谈总结
[总结访谈的核心收获和启发]

要求:
1. 按Q&A逻辑重组，不必严格按时间顺序
2. 保留重要原话引用
3. 分析各方立场和观点
4. 用中文输出"""
