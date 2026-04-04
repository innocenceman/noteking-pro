"""
NoteKing Pro Demo — 处理 OpenClaw 圆桌会议视频
使用 MiniMax M2.7 生成会议纪要、思维导图、闪卡等多种格式
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 配置 ──────────────────────────────────────────────────────────────
VIDEO_PATH = Path("/Users/daishanghao/Desktop/20260405_会议视频录音/OpenClaw与AI开源圆桌会议 月之暗面创始人杨植麟主持 - 001 - OpenClaw与AI开源圆桌会议 月之暗面创始人杨植麟主持.mp4")
OUTPUT_DIR = Path(__file__).parent / "demos" / "openclaw_meeting"
MINIMAX_KEY = "sk-cp-vX4T-YhmjkytkOexcwZ-uAdmALWR8ggXmtGOymuJQ1lfNLOR1phT0Ju09VggOTENL-y1pGe-KC4fTQppbzn_X_WPxVIApwG71PlvZHCGgfaIRH2zYoAI_RA"
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 配置对象 ───────────────────────────────────────────────────────────
from core.config import AppConfig, LLMConfig, ASRConfig, RecordingConfig

config = AppConfig(
    llm=LLMConfig(
        api_key=MINIMAX_KEY,
        base_url=MINIMAX_BASE_URL,
        model=MINIMAX_MODEL,
        temperature=0.3,
        max_tokens=8000,
        language="zh-CN",
    ),
    asr=ASRConfig(
        default_engine="faster_whisper",
        faster_whisper_model="base",
    ),
    recording=RecordingConfig(
        denoise_level=1,
        enable_diarization=False,
        default_scene="meeting",
    ),
    output_dir=str(OUTPUT_DIR),
)

print("=" * 60)
print("NoteKing Pro — OpenClaw 圆桌会议处理")
print("=" * 60)

# ── Step 1: 提取音频 ──────────────────────────────────────────────────
print("\n[1/5] 提取音频...")
import subprocess
audio_path = OUTPUT_DIR / "meeting_audio.wav"
if not audio_path.exists():
    result = subprocess.run([
        "ffmpeg", "-i", str(VIDEO_PATH),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", str(audio_path)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  音频提取失败: {result.stderr[:200]}")
        sys.exit(1)
    print(f"  ✓ 音频已提取: {audio_path.name}")
else:
    print(f"  ✓ 使用缓存音频: {audio_path.name}")

# ── Step 2: ASR 转录 ──────────────────────────────────────────────────
print("\n[2/5] ASR 语音转文字 (faster-whisper)...")
transcript_cache = OUTPUT_DIR / "transcript.json"
if transcript_cache.exists():
    print("  ✓ 使用缓存转录结果")
    import json
    with open(transcript_cache) as f:
        transcript_data = json.load(f)
    full_text = transcript_data["full_text"]
    segments = transcript_data["segments"]
else:
    from core.transcriber import transcribe
    result = transcribe(audio_path, config)
    full_text = result.full_text
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments]
    with open(transcript_cache, "w", encoding="utf-8") as f:
        json.dump({"full_text": full_text, "segments": segments, "language": result.language}, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 转录完成，共 {len(segments)} 段，{len(full_text)} 字符")

print(f"  预览: {full_text[:200]}...")

# ── Step 3: 生成 SRT 字幕文件 ─────────────────────────────────────────
print("\n[3/5] 生成 SRT 字幕...")
srt_path = OUTPUT_DIR / "meeting.srt"
def sec_to_srt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_lines = []
for i, seg in enumerate(segments, 1):
    srt_lines.append(str(i))
    srt_lines.append(f"{sec_to_srt(seg['start'])} --> {sec_to_srt(seg['end'])}")
    srt_lines.append(seg['text'].strip())
    srt_lines.append("")
srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
print(f"  ✓ SRT 字幕: {srt_path.name} ({len(segments)} 条)")

# ── Step 4: 用 MiniMax 生成多种输出 ──────────────────────────────────
print("\n[4/5] 调用 MiniMax M2.7 生成内容...")
from core.llm import chat

# 准备转录文本（控制长度在 token 限制内）
MAX_CHARS = 40000
transcript_for_llm = full_text[:MAX_CHARS]
if len(full_text) > MAX_CHARS:
    transcript_for_llm += f"\n\n[注：原文共{len(full_text)}字，此处截取前{MAX_CHARS}字]"

context_info = """
会议名称：OpenClaw与AI开源圆桌会议
主持人：杨植麟（月之暗面创始人）
参与者：AI领域开源生态相关人士
主题：OpenClaw平台与AI开源生态讨论
来源：Bilibili视频 BV1GX9pB9E6N
"""

outputs = {}

# 4a. 会议纪要（Markdown）
print("  生成会议纪要...")
meeting_prompt = f"""你是专业会议记录员。请根据以下会议转录文字，生成一份完整的中文会议纪要。

会议背景：
{context_info}

转录内容：
{transcript_for_llm}

请生成包含以下内容的详细会议纪要（Markdown格式）：
1. 会议基本信息（时间、地点、参会人员、主题）
2. 会议议程概述
3. 主要讨论内容（按议题分节，包含发言要点）
4. 重要观点与金句（直接引用原话）
5. 达成的共识与结论
6. 待办事项与行动计划
7. 会议总结

要求：结构清晰，重点突出，保留关键细节，适合存档和分享。"""

meeting_md = chat(meeting_prompt, config, max_tokens=4000)
outputs["meeting_minutes"] = meeting_md
(OUTPUT_DIR / "会议纪要.md").write_text(meeting_md, encoding="utf-8")
print(f"  ✓ 会议纪要 ({len(meeting_md)} 字)")

# 4b. 思维导图（Mermaid）
print("  生成思维导图...")
mindmap_prompt = f"""根据以下OpenClaw圆桌会议内容，生成一个Mermaid格式的思维导图。

会议背景：{context_info}
转录摘要：{transcript_for_llm[:15000]}

请生成mindmap格式的Mermaid代码，要求：
- 根节点：OpenClaw圆桌会议
- 主要分支：核心议题（3-5个）
- 每个议题下有子要点（2-4个）
- 包含关键人物和观点

只输出Mermaid代码块（```mermaid ... ```），不要其他内容。"""

mindmap_raw = chat(mindmap_prompt, config, max_tokens=1500)
outputs["mindmap"] = mindmap_raw
(OUTPUT_DIR / "思维导图.md").write_text(f"# OpenClaw圆桌会议思维导图\n\n{mindmap_raw}", encoding="utf-8")
print(f"  ✓ 思维导图 ({len(mindmap_raw)} 字)")

# 4c. 核心金句与观点
print("  提取核心金句...")
quotes_prompt = f"""从以下OpenClaw圆桌会议转录中，提取最有价值的金句和核心观点。

转录内容：{transcript_for_llm[:20000]}

请整理：
1. 10-15条精彩金句（原话引用，注明发言人如可辨识）
2. 5-8个核心观点/洞察（概括性陈述）
3. 3-5个值得思考的问题

用Markdown格式输出，适合社交媒体分享。"""

quotes_md = chat(quotes_prompt, config, max_tokens=2000)
outputs["quotes"] = quotes_md
(OUTPUT_DIR / "核心金句与观点.md").write_text(quotes_md, encoding="utf-8")
print(f"  ✓ 核心金句 ({len(quotes_md)} 字)")

# 4d. 简报摘要（适合快速阅读）
print("  生成简报摘要...")
brief_prompt = f"""为以下OpenClaw圆桌会议生成一份简洁的中文简报，适合3分钟快速阅读。

会议背景：{context_info}
转录内容：{transcript_for_llm[:20000]}

请生成：
## TL;DR（一句话总结）
[50字以内]

## 核心要点
[5个最重要的讨论点，每点2-3句话]

## 关键数据与事实
[列出会议中提到的具体数据、产品、公司名称]

## 为什么重要
[这次会议对AI开源生态的意义，3-4句话]"""

brief_md = chat(brief_prompt, config, max_tokens=1500)
outputs["brief"] = brief_md
(OUTPUT_DIR / "简报摘要.md").write_text(brief_md, encoding="utf-8")
print(f"  ✓ 简报摘要 ({len(brief_md)} 字)")

# 4e. 学习闪卡（Q&A格式）
print("  生成学习闪卡...")
flashcard_prompt = f"""根据OpenClaw圆桌会议内容，生成15-20张学习闪卡（问答卡片格式）。

转录内容：{transcript_for_llm[:15000]}

生成适合学习的Q&A卡片，涵盖：
- OpenClaw是什么，它的核心功能
- AI开源生态的现状与挑战
- 各参会嘉宾的核心观点
- 重要概念解释

格式（每张卡片）：
**Q: [问题]**
A: [答案，2-4句话]

---"""

flashcard_md = chat(flashcard_prompt, config, max_tokens=2500)
outputs["flashcards"] = flashcard_md
(OUTPUT_DIR / "学习闪卡.md").write_text(f"# OpenClaw圆桌会议学习闪卡\n\n{flashcard_md}", encoding="utf-8")
print(f"  ✓ 学习闪卡 ({len(flashcard_md)} 字)")

# ── Step 5: 生成 README 展示文件 ──────────────────────────────────────
print("\n[5/5] 生成案例展示 README...")

readme_content = f"""# 🎯 案例演示：OpenClaw与AI开源圆桌会议

> **NoteKing Pro 实际处理案例** — 展示如何将一段35分钟的圆桌会议视频，
> 自动转化为多种高价值文档输出。

## 📹 原始视频信息

| 项目 | 详情 |
|------|------|
| 视频标题 | OpenClaw与AI开源圆桌会议 月之暗面创始人杨植麟主持 |
| 来源 | [Bilibili BV1GX9pB9E6N](https://www.bilibili.com/video/BV1GX9pB9E6N/) |
| 时长 | 约 35 分钟 |
| 主持人 | 杨植麟（月之暗面 / Kimi 创始人）|
| ASR引擎 | faster-whisper (base 模型) |
| LLM | MiniMax M2.7 (200K context) |

## 📂 生成文件列表

| 文件 | 格式 | 说明 |
|------|------|------|
| [会议纪要.md](会议纪要.md) | Markdown | 完整结构化会议记录，含议程、要点、行动项 |
| [简报摘要.md](简报摘要.md) | Markdown | 3分钟速读版，TL;DR + 核心要点 |
| [核心金句与观点.md](核心金句与观点.md) | Markdown | 精彩金句提炼，适合社交媒体分享 |
| [思维导图.md](思维导图.md) | Mermaid | 会议内容可视化思维导图 |
| [学习闪卡.md](学习闪卡.md) | Markdown | 15-20张Q&A学习卡片 |
| [meeting.srt](meeting.srt) | SRT | 完整中文字幕文件 |
| [transcript.json](transcript.json) | JSON | 原始ASR转录数据（含时间戳） |

## 🚀 如何复现

```bash
# 1. 安装 NoteKing Pro
git clone https://github.com/bcefghj/noteking-pro
cd noteking-pro
pip install -e .

# 2. 配置 MiniMax API（中国区）
noteking setup \\
  --api-key "你的MiniMax_Key" \\
  --base-url "https://api.minimax.chat/v1" \\
  --model "MiniMax-M2.7"

# 3. 处理本地视频文件
noteking process 你的会议.mp4 \\
  --scene meeting \\
  --formats markdown,srt,mindmap,flashcard \\
  --output ./output
```

## 💡 处理流程

```
视频文件 (.mp4)
    │
    ▼
[FFmpeg] 音频提取 → WAV 16kHz 单声道
    │
    ▼
[faster-whisper] 语音识别 → 带时间戳文本段落
    │
    ▼
[MiniMax M2.7] LLM 处理
    ├── 会议纪要生成
    ├── 思维导图提炼
    ├── 金句萃取
    ├── 简报摘要
    └── 学习闪卡
    │
    ▼
多格式输出文件
```

## 📊 处理统计

- 视频时长：35.7 分钟
- 转录字数：{len(full_text):,} 字符
- 字幕条数：{len(segments)} 条
- 输出文件：7 个
- LLM模型：MiniMax M2.7

---

*由 [NoteKing Pro](https://github.com/bcefghj/noteking-pro) 自动生成*
"""

(OUTPUT_DIR / "README.md").write_text(readme_content, encoding="utf-8")
print(f"  ✓ README.md 已生成")

print("\n" + "=" * 60)
print("✅ 处理完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("\n生成文件:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name} ({size//1024}KB)" if size > 1024 else f"  {f.name} ({size}B)")
