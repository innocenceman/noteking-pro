"""Meeting minutes template: structured meeting notes with action items."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class MeetingMinutesTemplate(BaseTemplate):
    name = "meeting_minutes"
    display_name = "会议纪要"
    description = "Structured meeting minutes with attendees, topics, decisions, and action items"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        num_speakers = ctx.extra.get("num_speakers", 0)
        speakers = ctx.extra.get("speakers", [])

        context_info = f"\n内容描述: {user_context}" if user_context else ""
        speaker_info = ""
        if speakers:
            speaker_info = f"\n检测到 {num_speakers} 位说话人: {', '.join(speakers)}"

        return f"""请根据以下会议录音转录稿，生成一份专业的结构化会议纪要。

会议标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}{speaker_info}

会议录音转录稿（带说话人标签）:
{transcript}

请按以下格式输出会议纪要:

# 会议纪要: [会议标题]

## 会议基本信息
- **日期**: [从内容推断或标注"待确认"]
- **时长**: {ctx.meta.duration:.0f} 秒
- **参会人**: [从说话人标签和内容推断参会人身份及角色]

## 会议摘要
[2-3句话概括会议核心内容和成果]

## 议题与讨论

### 议题 1: [议题名称]
- **讨论要点**:
  - [要点1，标注发言人]
  - [要点2]
- **结论/决议**: [达成的共识或决定]

### 议题 2: ...
[继续列出所有议题]

## 关键决议
- [决议1]
- [决议2]

## 行动项
| 序号 | 行动事项 | 负责人 | 截止日期 |
|------|---------|--------|---------|
| 1 | [事项描述] | [负责人] | [日期/待定] |

## 重要引用
> [会议中的关键原话]
> —— [发言人]

## 后续跟进
- [需要跟进的事项]

要求:
1. 从对话内容中智能识别议题、决议和行动项
2. 保持客观准确，不添加未讨论的内容
3. 用中文输出
4. 重要观点标注发言人
5. 行动项要具体、可执行"""
