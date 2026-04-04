# 🎯 案例演示：OpenClaw与AI开源圆桌会议

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
noteking setup \
  --api-key "你的MiniMax_Key" \
  --base-url "https://api.minimax.chat/v1" \
  --model "MiniMax-M2.7"

# 3. 处理本地视频文件
noteking process 你的会议.mp4 \
  --scene meeting \
  --formats markdown,srt,mindmap,flashcard \
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
- 转录字数：12,430 字符
- 字幕条数：1405 条
- 输出文件：7 个
- LLM模型：MiniMax M2.7

---

*由 [NoteKing Pro](https://github.com/bcefghj/noteking-pro) 自动生成*
