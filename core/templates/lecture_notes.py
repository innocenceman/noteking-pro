"""Lecture notes template: structured knowledge with formulas and exercises."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class LectureNotesTemplate(BaseTemplate):
    name = "lecture_notes"
    display_name = "课堂笔记"
    description = "Structured lecture notes with knowledge hierarchy, formulas, examples, and study tips"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n课程描述: {user_context}" if user_context else ""

        return f"""请根据以下课程/讲座录音转录稿，生成一份专业的课堂学习笔记。

课程标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

课程录音转录稿:
{transcript}

请按以下格式输出课堂笔记:

# 课堂笔记: [课程标题]

## 课程概览
[简要概括本节课的主题和学习目标]

## 知识点详解

### 1. [知识点名称]

**核心概念**: [详细解释]

**重要公式/定义**:
$$[LaTeX 公式，如果有]$$

**示例**: [举例说明]

**易错点**: [常见误区和注意事项]

### 2. [知识点名称]
[继续...]

## 关键公式汇总
| 公式 | 含义 | 适用场景 |
|------|------|---------|

## 课堂练习
1. [从内容中提取或推导的练习题]
2. [练习题]

## 学习要点总结
- [要点1]
- [要点2]
- [要点3]

## 延伸学习建议
- [推荐阅读/学习资源]

要求:
1. 按知识点逻辑组织，不按时间顺序
2. 重要公式用 LaTeX 格式 ($..$ 或 $$..$$)
3. 代码示例用代码块
4. 标注难点和易错点
5. 用中文输出"""
