"""Smart summary template: adaptive-length summary based on content complexity."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class SmartSummaryTemplate(BaseTemplate):
    name = "smart_summary"
    display_name = "智能摘要"
    description = "Adaptive-length summary that adjusts based on content complexity"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        num_speakers = ctx.extra.get("num_speakers", 0)

        context_info = f"\n描述: {user_context}" if user_context else ""
        speaker_info = f"\n说话人数量: {num_speakers}" if num_speakers else ""

        return f"""请根据以下录音转录稿，生成一份智能摘要。根据内容的复杂度和信息密度自动调节摘要的长度和深度。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}{speaker_info}

录音转录稿:
{transcript}

请按以下三个层次输出摘要:

# 智能摘要: [标题]

## 一句话概括
[用一句精炼的话概括全部内容]

## 要点速览 (30秒阅读)
- [要点1]
- [要点2]
- [要点3]
- [要点4]
- [要点5]

## 详细摘要 (3分钟阅读)

### 背景
[内容的背景和上下文]

### 核心内容
[详细的内容摘要，按逻辑组织]

### 关键观点
1. **[观点1]**: [阐述]
2. **[观点2]**: [阐述]
3. **[观点3]**: [阐述]

### 结论/成果
[最终结论或成果]

## 关键数据/事实
- [提取的关键数据或事实]

## 值得记住的话
> "[引用]" —— [发言人]

要求:
1. 三个层次从简到详，满足不同阅读需求
2. 不遗漏重要信息
3. 区分事实和观点
4. 用中文输出"""
