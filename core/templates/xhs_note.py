"""Xiaohongshu (Little Red Book) note format template."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class XHSNoteTemplate(BaseTemplate):
    name = "xhs_note"
    display_name = "小红书笔记"
    description = "Xiaohongshu-style note with emojis, tags, and engagement hooks"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成一篇小红书风格的笔记。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 标题: 20 字以内，使用「痛点标题法」，加 2-4 个相关 emoji
2. 正文: 600-800 字
   - 开头用一句话引起共鸣
   - 用 emoji 引导每个段落
   - 内容分 3-5 个要点展开
   - 加入 2-3 个互动提示 (如 "你觉得呢?" "评论区聊聊")
   - 语气亲切自然，像朋友分享
3. 结尾:
   - 总结一句话收获
   - 引导收藏/点赞
4. 标签: 生成 5-10 个相关标签，格式为 #标签名
5. 整体风格: 实用干货 + 亲和力 + 高信息密度

示例格式:
---
[emoji] [标题]

[emoji] 第一个要点
内容...

[emoji] 第二个要点
内容...

💬 你觉得呢？评论区聊聊~

#标签1 #标签2 #标签3
---

请直接输出小红书笔记:"""
