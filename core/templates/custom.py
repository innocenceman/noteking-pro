"""Custom template: user-defined prompt."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class CustomTemplate(BaseTemplate):
    name = "custom"
    display_name = "自定义模板"
    description = "User-defined prompt template for any output format"

    def __init__(self, user_prompt: str = ""):
        self.user_prompt = user_prompt

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        custom = self.user_prompt or ctx.extra.get("custom_prompt", "")

        if not custom:
            custom = "请对这个视频内容进行全面分析和总结。"

        return f"""视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒

视频内容转录:
{transcript}

用户自定义要求:
{custom}

请根据上述要求处理视频内容:"""
