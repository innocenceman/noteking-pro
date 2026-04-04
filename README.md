# 👑 NoteKing Pro — 全网最强视频/录音处理工具

一键把视频、录音、会议转化为精美图文 PDF 讲义、结构化会议纪要、思维导图、学习闪卡等多种格式。

**支持 30+ 平台视频 | 本地录音/视频上传 | 说话人分离 | 降噪增强 | 23 种输出模板 | 中英混合**

[![GitHub stars](https://img.shields.io/github/stars/bcefghj/noteking?style=social)](https://github.com/bcefghj/noteking)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 核心亮点

| 功能 | 说明 |
|------|------|
| 🌐 **30+ 平台** | B站、YouTube、抖音、小红书、TikTok 等 1800+ 站点 |
| 🎙️ **本地录音/视频** | 直接上传 MP4/MP3/WAV 等文件处理 |
| 🗣️ **说话人分离** | 自动识别多人对话，标注谁说了什么 (pyannote-audio) |
| 🔇 **降噪增强** | 三级降噪，嘈杂环境也能清晰转录 |
| 📄 **LaTeX PDF 讲义** | ctex + tcolorbox + xelatex 专业排版 |
| 📋 **23 种模板** | 会议纪要、课堂笔记、访谈、闪卡、思维导图等 |
| 🌍 **多语言** | 中文最强 (FunASR WER 8.4%)、英文、50+ 语言 |
| 📦 **多文件合并** | 连续录制的多段文件自动拼接 |
| 🔌 **全形态部署** | CLI / Web / MCP Server / OpenClaw Skill / 桌面端 |
| 🆓 **开源免费** | 本地 ASR 免费，只有 LLM 按需付费 |

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/bcefghj/noteking.git
cd noteking

# 安装 (基础)
pip install -e .

# 安装 (带录音处理)
pip install -e ".[meeting,asr]"

# 安装 (全部功能)
pip install -e ".[all]"

# 首次配置
noteking setup --api-key "你的API_KEY" --base-url "https://api.minimax.chat/v1" --model "MiniMax-M2.7"
```

**前置依赖**: Python 3.11+, FFmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

### 处理在线视频

```bash
# B站视频 -> 详细学习笔记
noteking run "https://www.bilibili.com/video/BV1xx" -t detailed

# YouTube -> 思维导图
noteking run "https://youtu.be/xxx" -t mindmap

# 本地视频 -> LaTeX PDF
noteking run "./lecture.mp4" -t latex_pdf
```

### 处理本地录音/会议 (NEW)

```bash
# 会议录音 -> 会议纪要（带说话人分离）
noteking process meeting.mp4 -t meeting_minutes -c "产品周会" --speakers 4

# 课堂录音 -> 课堂笔记（降噪处理）
noteking process lecture.mp3 -t lecture_notes --denoise 2

# 多段录音合并处理
noteking process part1.mp4 part2.mp4 -t meeting_minutes

# 访谈 -> 多格式输出
noteking process interview.wav -t interview --format markdown,srt,json

# 仅转录
noteking transcribe recording.mp4

# 仅降噪
noteking denoise noisy.wav --level 2

# 合并文件
noteking merge seg1.mp4 seg2.mp4 -o merged.mp4
```

---

## 📋 23 种输出模板

### 视频模板
| 模板 | 名称 | 适用场景 |
|------|------|---------|
| `brief` | 简要总结 | 快速概览 |
| `detailed` | 详细学习笔记 | 系统学习 |
| `mindmap` | 思维导图 | 知识结构 |
| `flashcard` | 闪卡 (Anki) | 间隔重复 |
| `quiz` | 测验题 | 自测 |
| `timeline` | 时间线笔记 | 带时间戳 |
| `exam` | 考试复习 | 备考 |
| `tutorial` | 教程步骤 | 操作指南 |
| `news` | 新闻速览 | 快讯 |
| `podcast` | 播客摘要 | 对话 |
| `xhs_note` | 小红书笔记 | 社交分享 |
| `latex_pdf` | LaTeX PDF | 专业讲义 |
| `custom` | 自定义 | 自由定义 |

### 录音/会议模板 (NEW)
| 模板 | 名称 | 适用场景 |
|------|------|---------|
| `meeting_minutes` | 会议纪要 | 会议记录，含行动项 |
| `lecture_notes` | 课堂笔记 | 知识点/公式/习题 |
| `interview` | 访谈记录 | Q&A/观点/立场 |
| `brainstorm` | 灵感记录 | 想法/思维导图 |
| `news_digest` | 新闻摘要 | 5W1H 结构 |
| `exam_prep` | 考试复习 | 闪卡+模拟题 |
| `cornell_notes` | 康奈尔笔记 | 经典学习法 |
| `podcast_shownotes` | 播客节目笔记 | 章节+要点 |
| `entertainment` | 娱乐内容 | 高光/金句 |
| `smart_summary` | 智能摘要 | 自适应长度 |

---

## 🏗️ 技术架构

```
输入 → 预处理 → ASR转录 → 说话人分离 → LLM生成 → 多格式输出

输入层:
├── 在线视频 (30+ 平台, yt-dlp)
├── 本地视频 (MP4/MKV/AVI/MOV)
├── 本地录音 (MP3/WAV/FLAC/M4A)
└── 多文件合并 (FFmpeg concat)

预处理:
├── 音频提取 (FFmpeg 16kHz mono WAV)
├── 降噪增强 (noisereduce / DeepFilterNet)
└── VAD 分片 (超长音频切分)

ASR 引擎 (自动选择最佳):
├── FunASR Paraformer-zh (中文最强, WER 8.4%)
├── faster-whisper (英文/多语言)
├── SenseVoice (50+ 语言)
└── Groq/OpenAI API (云端回退)

说话人分离:
├── pyannote-audio 3.x
└── WhisperX 风格对齐

输出:
├── Markdown / LaTeX PDF / SRT / VTT
├── 思维导图 / 闪卡 / JSON
└── 带说话人标签的完整转录
```

---

## 🌐 多形态部署

### Web 网站
```bash
# Docker 一键部署
docker compose up -d
# 访问 http://localhost:8000
```

### CLI 命令行
```bash
pip install -e ".[meeting]"
noteking process meeting.mp4 -t meeting_minutes
```

### MCP Server (Cursor / Claude Desktop / OpenClaw)
```json
{
  "mcpServers": {
    "noteking": {
      "command": "npx",
      "args": ["tsx", "mcp/src/index.ts"],
      "env": { "NOTEKING_DIR": "/path/to/noteking" }
    }
  }
}
```

### OpenClaw Skill
已发布到 ClawHub，直接安装使用。

### API
```bash
# 启动 API 服务
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 处理视频
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bilibili.com/video/BV1xx", "template": "detailed"}'

# 上传录音处理
curl -X POST http://localhost:8000/api/v1/recording/upload -F "file=@meeting.mp4"
curl -X POST http://localhost:8000/api/v1/recording/process \
  -F "file_id=xxx" -F "template=meeting_minutes" -F "context=产品周会"
```

---

## 📖 部署教程

详见 [docs/deploy-guide.md](docs/deploy-guide.md)

- **方案 A**: Railway 一键部署（免费，最简单）
- **方案 B**: Docker + 云服务器（推荐正式运营）
- **方案 C**: 带 GPU 部署（最佳 ASR 质量）
- **个人本地**: pip install 即用

---

## 🔧 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `NOTEKING_LLM_API_KEY` | 是 | LLM API Key (MiniMax/OpenAI/DeepSeek) |
| `NOTEKING_LLM_BASE_URL` | 否 | 自定义 API 地址 |
| `NOTEKING_LLM_MODEL` | 否 | 模型名称 (默认 gpt-4o-mini) |
| `HF_TOKEN` | 否 | HuggingFace Token (pyannote 说话人分离) |
| `NOTEKING_PROXY` | 否 | 代理 (YouTube 访问) |
| `BILIBILI_SESSDATA` | 否 | B站登录 Cookie |

---

## 📄 License

MIT License - 开源免费，随意使用

---

## 🙏 致谢

核心技术栈: OpenAI Whisper, FunASR, pyannote-audio, faster-whisper, FFmpeg, yt-dlp, OpenAI SDK

灵感来源: 钉钉AI听记, Otter.ai, NotebookLM, Meetily, Open Notebook, BibiGPT
