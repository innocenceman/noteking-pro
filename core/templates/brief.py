"""Brief summary template: 3-5 sentence overview."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class BriefTemplate(BaseTemplate):
    name = "brief"
    display_name = "简要总结"
    description = "3-5 sentence quick overview of the video content"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请对以下视频内容生成一个简要总结。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒

视频内容转录:
{transcript}

要求:
1. 用 3-5 句话概括视频的核心内容
2. 突出最重要的要点和结论
3. 语言简洁精炼，适合快速浏览
4. 格式为 Markdown

请直接输出总结内容:"""
