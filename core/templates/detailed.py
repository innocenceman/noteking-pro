"""Detailed learning notes template with chapters and key points."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class DetailedTemplate(BaseTemplate):
    name = "detailed"
    display_name = "详细学习笔记"
    description = "Structured learning notes with chapters, key points, and summaries"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        chapters_info = ""
        if ctx.has_chapters:
            chapters_info = "\n视频章节:\n"
            for ch in ctx.meta.chapters:
                t = ch.get("title", "")
                s = ch.get("start_time", 0)
                chapters_info += f"- [{s:.0f}s] {t}\n"

        return f"""请根据以下视频内容，生成一份详细的结构化学习笔记。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒
{chapters_info}

视频内容转录:
{transcript}

要求:
1. 使用 Markdown 格式，带清晰的层级结构 (## / ### / ####)
2. 按教学逻辑组织内容，而非简单按时间顺序排列
3. 每个主要章节包含:
   - 核心概念解释
   - 关键要点 (用列表形式)
   - 重要公式/代码 (如果有)
   - 实例说明
4. 在笔记开头添加「📋 内容概览」列出所有章节
5. 在笔记结尾添加「💡 核心收获」总结 3-5 个最重要的收获
6. 在笔记结尾添加「🔗 延伸思考」提出 2-3 个值得深入探讨的问题
7. 用 > 引用格式标注原始视频中的关键原话
8. 标注重要概念为 **粗体**

请直接输出完整的学习笔记:"""
