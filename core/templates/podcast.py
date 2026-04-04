"""Podcast/interview summary template."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class PodcastTemplate(BaseTemplate):
    name = "podcast"
    display_name = "播客/访谈摘要"
    description = "Structured summary for podcasts, interviews, and discussions"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下播客/访谈内容，生成一份结构化摘要。

标题: {ctx.meta.title}
主持/嘉宾: {ctx.meta.uploader}
时长: {ctx.meta.duration:.0f} 秒

内容转录:
{transcript}

要求:
1. 格式为:

## 🎙️ 播客摘要: {ctx.meta.title}

### 参与者
- [识别并列出所有参与者及其身份/背景]

### 主题概述
[1-2 句话概括本期主题]

### 核心观点
1. **[观点 1 标题]**: 详细说明 (标注来自哪位嘉宾)
2. **[观点 2 标题]**: 详细说明
3. ...

### 精彩语录
> "[原话摘录 1]" —— [发言者]
> "[原话摘录 2]" —— [发言者]

### 讨论亮点
- [值得关注的讨论点或交锋]

### 提到的资源/推荐
- [提到的书籍/工具/网站等]

### 要点总结
- 3-5 条核心收获

2. 区分不同发言者的观点
3. 保留有价值的原始语录
4. 标注有争议或有趣的讨论点

请直接输出播客摘要:"""
