"""Podcast show notes template: guest info, chapter timestamps, and key takeaways."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class PodcastShowNotesTemplate(BaseTemplate):
    name = "podcast_shownotes"
    display_name = "播客节目笔记"
    description = "Podcast show notes with guest info, chapter timestamps, and takeaways"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        speakers = ctx.extra.get("speakers", [])

        context_info = f"\n描述: {user_context}" if user_context else ""
        speaker_info = f"\n嘉宾/主持人: {', '.join(speakers)}" if speakers else ""

        return f"""请根据以下播客/对话节目录音转录稿，生成一份节目笔记。

节目标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}{speaker_info}

录音转录稿:
{transcript}

请按以下格式输出:

# 节目笔记: [标题]

## 节目信息
- **主题**: [节目主题]
- **主持人**: [主持人]
- **嘉宾**: [嘉宾及身份]
- **时长**: {ctx.meta.duration:.0f} 秒

## 内容摘要
[3-5句话概括节目核心内容]

## 章节时间线
| 时间 | 章节 | 要点 |
|------|------|------|
| [00:00] | [开场/引入] | [简述] |
| [MM:SS] | [话题1] | [简述] |
| [MM:SS] | [话题2] | [简述] |

## 各章节详情

### [话题1] (MM:SS - MM:SS)
[详细内容]

**嘉宾观点**:
- [观点1]
- [观点2]

### [话题2]
...

## 关键收获 (Key Takeaways)
1. [收获1]
2. [收获2]
3. [收获3]

## 金句
- > "[金句1]" —— [说话人]
- > "[金句2]" —— [说话人]

## 提到的资源/推荐
- [书籍/工具/网站等]

要求:
1. 时间戳尽量准确
2. 按话题组织章节
3. 保留重要金句原话
4. 列出提到的资源推荐
5. 用中文输出"""
