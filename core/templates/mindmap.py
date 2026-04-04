"""Mind map template: Markmap-compatible markdown."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class MindMapTemplate(BaseTemplate):
    name = "mindmap"
    display_name = "思维导图"
    description = "Markmap-compatible hierarchical mind map in Markdown"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成一份思维导图格式的笔记。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 使用 Markdown 标题层级表示思维导图结构 (# -> ## -> ### -> ####)
2. 第一级 (#) 是视频主题
3. 第二级 (##) 是主要分支 (3-7 个)
4. 第三级 (###) 是子主题
5. 第四级 (####) 是具体要点
6. 每个叶子节点用简短的一句话或关键词
7. 确保结构清晰、层次分明
8. 适合用 Markmap 工具渲染为可视化思维导图

请直接输出思维导图内容 (纯 Markdown 标题层级格式):"""
