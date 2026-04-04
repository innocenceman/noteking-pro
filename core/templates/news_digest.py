"""News digest template: 5W1H structured summary for news/podcast content."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class NewsDigestTemplate(BaseTemplate):
    name = "news_digest"
    display_name = "新闻/播客摘要"
    description = "5W1H structured news digest with quotes and background"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n描述: {user_context}" if user_context else ""

        return f"""请根据以下新闻/播客录音转录稿，生成一份结构化的新闻摘要。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

录音转录稿:
{transcript}

请按以下格式输出:

# 新闻摘要: [标题]

## 一句话速览
[一句话概括核心新闻]

## 5W1H 结构分析
- **Who (谁)**: [涉及的人物/组织]
- **What (什么)**: [发生了什么]
- **When (何时)**: [时间]
- **Where (何地)**: [地点/场景]
- **Why (为何)**: [原因/动机]
- **How (如何)**: [过程/方式]

## 关键事实
1. [事实1]
2. [事实2]
3. [事实3]

## 各方观点
| 发言人/立场 | 核心观点 |
|------------|---------|

## 关键引用
> "[引用1]" —— [发言人]
> "[引用2]" —— [发言人]

## 背景补充
[相关背景知识和上下文]

## 影响分析
[对相关领域的潜在影响]

要求:
1. 区分事实和观点
2. 保留关键引用原话
3. 提供必要的背景信息
4. 用中文输出"""
