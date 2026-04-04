"""Exam preparation template: flashcards, mock questions, and study checklist."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class ExamPrepTemplate(BaseTemplate):
    name = "exam_prep"
    display_name = "考试复习"
    description = "Exam prep with flashcards, mock questions, key concepts, and study checklist"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n描述: {user_context}" if user_context else ""

        return f"""请根据以下课程/讲座录音转录稿，生成考试复习材料。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

录音转录稿:
{transcript}

请按以下格式输出:

# 考试复习: [课程主题]

## 核心知识点清单
- [ ] [知识点1]
- [ ] [知识点2]
- [ ] [知识点3]

## 知识卡片

### 卡片 1
- **问题**: [问题]
- **答案**: [答案]
- **难度**: ⭐⭐ (1-5)

### 卡片 2
...

## 模拟选择题

**1. [题目]**
- A. [选项]
- B. [选项]
- C. [选项]
- D. [选项]

**答案**: [正确答案及解析]

**2. [题目]**
...

## 模拟简答题

**1. [题目]**
**参考答案**: [答案要点]

## 重点公式/概念
| 概念 | 定义/公式 | 备注 |
|------|----------|------|

## 易错点提醒
1. ⚠️ [易错点1]
2. ⚠️ [易错点2]

## 复习建议
[学习策略和重点复习方向]

要求:
1. 知识点要全面覆盖
2. 题目难度适中，涵盖选择题和简答题
3. 公式用 LaTeX 格式
4. 标注易错点和重难点
5. 用中文输出"""
