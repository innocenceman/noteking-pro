"""Flashcard template: Anki-compatible Q&A pairs."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class FlashCardTemplate(BaseTemplate):
    name = "flashcard"
    display_name = "闪卡 (Anki)"
    description = "Anki-compatible flashcards with Q&A pairs for spaced repetition"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成一组学习闪卡 (Flashcards)。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 生成 15-30 张闪卡
2. 每张闪卡格式:

### 卡片 N
**Q:** [问题]
**A:** [答案]

3. 问题类型多样化:
   - 概念定义题: "什么是 X?"
   - 对比区分题: "X 和 Y 的区别是什么?"
   - 应用场景题: "什么时候应该使用 X?"
   - 原理机制题: "X 是如何工作的?"
   - 关键数据题: "X 的关键参数/指标是什么?"
4. 答案简洁但完整，控制在 1-3 句话
5. 遵循 Anki 闪卡的原子性原则: 每张卡只测试一个知识点
6. 在文件开头注明视频来源信息

请直接输出闪卡内容:"""
