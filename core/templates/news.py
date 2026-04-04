"""News/information summary template."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class NewsTemplate(BaseTemplate):
    name = "news"
    display_name = "新闻速览"
    description = "News-style quick summary for information and current events"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成新闻速览格式的摘要。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}
发布日期: {ctx.meta.upload_date}

视频内容转录:
{transcript}

要求:
1. 格式为:

## 📰 新闻速览

**标题:** [一句话标题]
**来源:** {ctx.meta.uploader} | {ctx.meta.upload_date}

### 核心要点 (TL;DR)
- 要点 1
- 要点 2
- 要点 3

### 详细内容
[分段展开核心内容，每段 2-3 句话]

### 相关背景
[补充理解所需的背景信息]

### 影响分析
[这条信息的意义和可能的影响]

2. 语言风格: 客观、简洁、信息密度高
3. 避免主观评价，侧重事实陈述

请直接输出新闻速览:"""
