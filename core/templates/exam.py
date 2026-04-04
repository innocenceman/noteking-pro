"""Exam review notes template: optimized for test preparation."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class ExamTemplate(BaseTemplate):
    name = "exam"
    display_name = "考试复习笔记"
    description = "Exam-oriented review notes with key concepts and formulas"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成一份考试复习笔记。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 按考试复习的逻辑组织内容:

## 📚 考试复习笔记: {ctx.meta.title}

### 一、核心概念清单
(列出所有需要掌握的核心概念，每个概念用一句话定义)

### 二、重点公式/定理
(列出所有重要公式，标注使用场景和注意事项)

### 三、知识点详解
(分主题详细展开，每个知识点标注重要程度 ⭐⭐⭐)

### 四、易错点/易混淆点
(列出常见的错误理解和混淆概念)

### 五、速记口诀/记忆技巧
(帮助快速记忆的口诀或联想方法)

### 六、模拟考题
(3-5 道典型考题及解答思路)

2. 使用 ⭐ 标注重要程度 (⭐ 到 ⭐⭐⭐⭐⭐)
3. 用高亮标注必考点
4. 适合打印后快速复习

请直接输出考试复习笔记:"""
