"""Quiz template: multiple-choice and short-answer questions."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class QuizTemplate(BaseTemplate):
    name = "quiz"
    display_name = "测验题"
    description = "Multiple-choice and short-answer quiz questions"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，生成一套测验题。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 生成 10-15 道题目，包含:
   - 5-8 道选择题 (单选，4 个选项)
   - 3-5 道简答题
   - 2-3 道判断题 (对/错)
2. 每道选择题格式:

#### 第 N 题 (选择题)
[题目内容]
- A. [选项]
- B. [选项]
- C. [选项]
- D. [选项]

**答案:** [正确选项]
**解析:** [为什么选这个答案]

3. 每道简答题格式:

#### 第 N 题 (简答题)
[题目内容]

**参考答案:**
[详细解答]

4. 题目难度分布: 简单 40%、中等 40%、困难 20%
5. 覆盖视频的主要知识点

请直接输出测验题:"""
