"""Cornell notes template: cue column, notes area, and summary section."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class CornellNotesTemplate(BaseTemplate):
    name = "cornell_notes"
    display_name = "康奈尔笔记"
    description = "Cornell note-taking method with cue column, notes area, and summary"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n描述: {user_context}" if user_context else ""

        return f"""请根据以下录音转录稿，使用康奈尔笔记法生成结构化笔记。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

录音转录稿:
{transcript}

请按以下康奈尔笔记格式输出:

# 康奈尔笔记: [标题]

---

## 📝 笔记区

### [主题1]

| 线索栏 (关键词/问题) | 笔记栏 (详细内容) |
|---------------------|------------------|
| [关键词1] | [对应的详细笔记内容] |
| [关键问题1] | [对应的回答/内容] |
| [关键概念1] | [解释和细节] |

### [主题2]

| 线索栏 | 笔记栏 |
|--------|--------|
| [关键词] | [详细内容] |

### [主题3]
...

---

## 📌 总结区
[用自己的话概括全部内容的精华，3-5句话]

---

## 🔑 关键术语
| 术语 | 定义 |
|------|------|

## ❓ 待解决问题
1. [需要进一步了解的问题]
2. [问题]

要求:
1. 严格遵循康奈尔笔记的三栏结构
2. 线索栏用关键词和问题，便于复习时自测
3. 笔记栏详细但精炼
4. 总结区用自己的话概括，不是简单复制
5. 用中文输出"""
