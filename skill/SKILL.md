---
name: noteking-pro
description: |
  全网最强视频/录音处理工具。将视频、录音、会议转化为精美图文 PDF 讲义、结构化会议纪要、思维导图、学习闪卡等多种格式。支持说话人分离、降噪增强、中英混合、30+ 平台视频链接。涵盖会议/课堂/访谈/灵感/娱乐全场景，23 种输出模板，本地优先+云端可选。
version: 2.0.0
metadata:
  openclaw:
    requires:
      env:
        - NOTEKING_LLM_API_KEY
      bins:
        - ffmpeg
        - python3
    primaryEnv: NOTEKING_LLM_API_KEY
    emoji: "👑"
    homepage: https://github.com/bcefghj/noteking
    os: ["macos", "linux", "windows"]
    install:
      - kind: brew
        formula: ffmpeg
        bins: [ffmpeg, ffprobe]
---

# NoteKing Pro - 全网最强视频/录音处理工具

Use this skill when the user wants to:
- Convert a video or audio recording into structured notes
- Generate meeting minutes from a recording
- Transcribe audio/video with speaker identification
- Create study materials from lectures
- Process interviews, brainstorm sessions, or podcasts

## Core Capabilities

### 1. Online Video Processing (30+ platforms)
Convert any online video URL into structured notes.

### 2. Local Recording Processing (NEW)
Process local audio/video files with advanced features:
- **Speaker diarization**: Identify who said what in multi-speaker recordings
- **Noise reduction**: Clean up noisy recordings (3 levels)
- **Multi-file merge**: Combine multiple recording segments
- **Language detection**: Auto-detect language including Chinese-English mix

### 3. 23 Output Templates

| Template | Name | Best For |
|----------|------|----------|
| `brief` | 简要总结 | Quick overview |
| `detailed` | 详细学习笔记 | Systematic study |
| `mindmap` | 思维导图 | Visual knowledge structure |
| `flashcard` | 闪卡 | Spaced repetition |
| `quiz` | 测验题 | Self-testing |
| `timeline` | 时间线笔记 | Timestamped notes |
| `exam` | 考试复习 | Exam preparation |
| `tutorial` | 教程步骤 | Step-by-step extraction |
| `news` | 新闻速览 | News summary |
| `podcast` | 播客摘要 | Discussion summary |
| `xhs_note` | 小红书笔记 | Social media note |
| `latex_pdf` | LaTeX PDF | Professional lecture notes |
| `custom` | 自定义 | User-defined prompt |
| `meeting_minutes` | 会议纪要 | Meeting notes with action items |
| `lecture_notes` | 课堂笔记 | Knowledge hierarchy |
| `interview` | 访谈记录 | Q&A with insights |
| `brainstorm` | 灵感记录 | Idea capture |
| `news_digest` | 新闻摘要 | 5W1H structured digest |
| `exam_prep` | 考试复习 | Flashcards + mock questions |
| `cornell_notes` | 康奈尔笔记 | Cornell note method |
| `podcast_shownotes` | 播客笔记 | Show notes with chapters |
| `entertainment` | 娱乐内容 | Highlights and quotes |
| `smart_summary` | 智能摘要 | Adaptive-length summary |

## Supported Platforms

- **Bilibili** (B站), **YouTube**, **Douyin** (抖音), **Xiaohongshu** (小红书)
- **TikTok**, **Twitter/X**, **Instagram**, **Twitch**, **Vimeo**
- **1800+ additional sites** via yt-dlp
- **Local files**: MP4, MKV, AVI, MOV, MP3, WAV, FLAC, M4A, OGG, AAC

## Dependencies

Required:
- `ffmpeg` - Audio/video processing
- `python3` >= 3.11

Python packages (auto-installed):
- `openai` - LLM API client
- `yt-dlp` - Video downloading
- `rich` - Terminal UI

Optional (for best experience):
- `funasr` - Best Chinese ASR (WER 8.4%)
- `faster-whisper` - Local multilingual ASR
- `pyannote.audio` - Speaker diarization
- `noisereduce` - Noise reduction

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTEKING_LLM_API_KEY` | Yes | API key for LLM (MiniMax/OpenAI/DeepSeek) |
| `NOTEKING_LLM_BASE_URL` | No | Custom API base URL |
| `NOTEKING_LLM_MODEL` | No | Model name (default: gpt-4o-mini) |
| `HF_TOKEN` | No | HuggingFace token for pyannote speaker diarization |
| `NOTEKING_PROXY` | No | Proxy URL for YouTube access from China |
| `BILIBILI_SESSDATA` | No | Bilibili login cookie |

## Usage

### Process Online Video
```bash
cd /path/to/noteking
python -m cli run "<VIDEO_URL>" -t <TEMPLATE>
```

### Process Local Recording (NEW)
```bash
# Meeting recording with speaker diarization
python -m cli process meeting.mp4 -t meeting_minutes -c "产品周会" --speakers 4

# Lecture with noise reduction
python -m cli process lecture.mp3 -t lecture_notes --denoise 2

# Multiple files merged
python -m cli process part1.mp4 part2.mp4 -t meeting_minutes

# Interview with multiple output formats
python -m cli process interview.wav -t interview --format markdown,srt,json
```

### Transcribe Only
```bash
python -m cli transcribe recording.mp4
```

### Denoise Audio
```bash
python -m cli denoise noisy.wav --level 2
```

### Merge Files
```bash
python -m cli merge segment1.mp4 segment2.mp4 -o merged.mp4
```

## Scene Types

| Scene | Template | Description |
|-------|----------|-------------|
| meeting | meeting_minutes | 会议纪要: 参会人/议题/决议/行动项 |
| lecture | lecture_notes | 课堂笔记: 知识点/公式/习题 |
| interview | interview | 访谈: Q&A/观点/立场 |
| brainstorm | brainstorm | 灵感: 想法/思维导图/行动 |
| news | news_digest | 新闻: 5W1H/引用/背景 |
| exam | exam_prep | 考试: 闪卡/模拟题/要点 |
| entertainment | entertainment | 娱乐: 高光/金句/推荐 |

## Workflow

### For Online Videos:
1. **Parse URL** → Detect platform
2. **Extract subtitles** → CC → ASR → Visual fallback
3. **Generate notes** → Apply template via LLM
4. **Save output** → Markdown + SRT

### For Local Recordings:
1. **Preprocess** → Validate, merge files, extract audio, denoise
2. **Transcribe** → Auto-detect language, FunASR(Chinese)/Whisper(multilingual)
3. **Diarize** → Identify speakers with pyannote-audio
4. **Generate** → LLM with scene context and speaker labels
5. **Output** → Multi-format (Markdown/PDF/SRT/VTT/JSON)

## Notes for AI Agents

- For meetings: use `process` command with `meeting_minutes` template and specify `--speakers`
- For lectures: use `lecture_notes` template, optionally with `--denoise`
- For noisy recordings: increase `--denoise` level (1=light, 2=medium, 3=heavy)
- For long recordings split into parts: pass all files to `process` command
- Default to `meeting_minutes` for recordings unless user specifies otherwise
- For quick overview, use `smart_summary` template
- Check `ffmpeg` and `python3` are installed before processing
