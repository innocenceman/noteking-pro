"""Timeline template: timestamped notes with jump links."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class TimelineTemplate(BaseTemplate):
    name = "timeline"
    display_name = "时间线笔记"
    description = "Timestamped notes aligned to video timeline"

    def build_prompt(self, ctx: TemplateContext) -> str:
        segments_text = ""
        for seg in ctx.subtitles.segments[:500]:
            segments_text += f"[{seg.start_ts}] {seg.text}\n"

        if not segments_text:
            segments_text = _truncate_transcript(ctx.transcript)

        url = ctx.meta.webpage_url
        return f"""请根据以下带时间戳的视频内容，生成时间线笔记。

视频标题: {ctx.meta.title}
视频链接: {url}
作者: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒

带时间戳的内容:
{segments_text}

要求:
1. 按时间顺序组织，每个重要节点标注时间戳
2. 格式为:

## 🎬 视频时间线

### [00:00:00] 开场/引言
- 要点 1
- 要点 2

### [00:05:30] 第一个主题
- 核心观点
- 关键细节

3. 合并相近的内容段落，不需要逐句翻译
4. 每个时间节点提炼 2-5 个要点
5. 标注转折点、高潮点、重要结论
6. 在文件末尾生成一个快速跳转索引

请直接输出时间线笔记:"""
