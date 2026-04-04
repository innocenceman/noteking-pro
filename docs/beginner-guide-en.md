# NoteKing Beginner Guide (English)

> A step-by-step guide to turn any video into beautiful illustrated PDF lecture notes.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Get an LLM API Key](#2-get-an-llm-api-key)
3. [Method 1: Docker (Easiest)](#3-method-1-docker-easiest)
4. [Method 2: pip Install (Flexible)](#4-method-2-pip-install-flexible)
5. [Method 3: OpenClaw Agent](#5-method-3-openclaw-agent)
6. [Generate Your First Notes](#6-generate-your-first-notes)
7. [Generate LaTeX PDF Lecture Notes](#7-generate-latex-pdf-lecture-notes)
8. [Batch Processing a Course](#8-batch-processing-a-course)
9. [Template Guide](#9-template-guide)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Required | Description |
|----------|-------------|
| A computer | Windows / Mac / Linux |
| An LLM API Key | Powers the AI note generation |

| Optional | Description |
|----------|-------------|
| Docker Desktop | For the easiest one-click setup |
| Python 3.11+ | For pip-based installation |
| TinyTeX | For generating professional LaTeX PDF lecture notes |

---

## 2. Get an LLM API Key

NoteKing works with any OpenAI-compatible LLM API. Choose one provider:

### Option A: DeepSeek (Affordable)

1. Visit https://platform.deepseek.com/
2. Create an account
3. Go to "API Keys" → Create a new key
4. Top up $5 (lasts a long time)
5. Note down:
   - API Key: `sk-xxx...`
   - Base URL: `https://api.deepseek.com/v1`
   - Model: `deepseek-chat`

### Option B: OpenAI (Best quality)

1. Visit https://platform.openai.com/
2. Create an account and add credits
3. Create an API Key
4. Note down:
   - API Key: `sk-xxx...`
   - Base URL: `https://api.openai.com/v1`
   - Model: `gpt-4o` or `gpt-4o-mini`

### Option C: MiniMax (Chinese provider, good value)

1. Visit https://www.minimaxi.com/
2. Create an account
3. Go to API Keys → Create a key
4. Purchase a plan
5. Note down:
   - API Key: `sk-cp-xxx...`
   - Base URL: `https://api.minimax.chat/v1`
   - Model: `MiniMax-M2.7`

---

## 3. Method 1: Docker (Easiest)

### Step 1: Install Docker Desktop

- **Mac**: Download from https://www.docker.com/products/docker-desktop/ → drag to Applications → launch
- **Windows**: Download from https://www.docker.com/products/docker-desktop/ → run installer → restart if prompted for WSL 2

### Step 2: Open a Terminal

- **Mac**: Press `Cmd + Space`, type "Terminal", hit Enter
- **Windows**: Press `Win + X`, select "Windows PowerShell"

### Step 3: Run These Commands

```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking

# Configure your API key
echo "NOTEKING_LLM_API_KEY=your-api-key-here" > .env
echo "NOTEKING_LLM_BASE_URL=https://api.deepseek.com/v1" >> .env
echo "NOTEKING_LLM_MODEL=deepseek-chat" >> .env

# Start
docker compose up -d
```

### Step 4: Use It

Open your browser and go to `http://localhost:3000`. Paste a video URL, select a template, and click Generate!

---

## 4. Method 2: pip Install (Flexible)

### Step 1: Install Python 3.11+

- **Mac**: `brew install python@3.11`
- **Windows**: Download from https://www.python.org/downloads/ — make sure to check "Add Python to PATH"

### Step 2: Install ffmpeg

- **Mac**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html and add to PATH

### Step 3: Install Dependencies

```bash
pip install yt-dlp openai httpx pillow imagehash scenedetect opencv-python-headless
```

### Step 4: Clone and Configure

```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking

export NOTEKING_LLM_API_KEY="your-api-key"
export NOTEKING_LLM_BASE_URL="https://api.deepseek.com/v1"
export NOTEKING_LLM_MODEL="deepseek-chat"
```

### Step 5: Generate Notes

```bash
python -m noteking.cli run "https://www.youtube.com/watch?v=xxxxx" --template detailed
```

---

## 5. Method 3: OpenClaw Agent

If you're using OpenClaw, simply say:

> Please install the NoteKing video notes skill

Then:

> Summarize this video: https://www.youtube.com/watch?v=xxxxx

---

## 6. Generate Your First Notes

```bash
# Bilibili video
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template detailed

# YouTube video
python -m noteking.cli run "https://www.youtube.com/watch?v=xxxxx" --template brief
```

Output will be saved to the current directory. The terminal will display the output file path.

---

## 7. Generate LaTeX PDF Lecture Notes

### Step 1: Install TinyTeX

**Mac / Linux:**
```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
export PATH=$PATH:~/Library/TinyTeX/bin/universal-darwin
tlmgr install ctex tcolorbox listings booktabs float fancyhdr xcolor enumitem etoolbox environ trimspaces adjustbox collectbox caption hyperref geometry graphicx fontspec xunicode xltxtra
```

**Windows:**
```powershell
Invoke-WebRequest -Uri "https://yihui.org/tinytex/install-bin-windows.bat" -OutFile "install-tinytex.bat"
.\install-tinytex.bat
```

### Step 2: Generate

```bash
python -m noteking.cli run "https://www.youtube.com/watch?v=xxxxx" --template latex_pdf
```

The generated PDF includes a cover page, table of contents, keyframe screenshots, highlight boxes, syntax-highlighted code, math formulas, and branded headers/footers.

---

## 8. Batch Processing a Course

```bash
# Process entire Bilibili collection
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC" --template detailed --batch

# Process YouTube playlist
python -m noteking.cli run "https://www.youtube.com/playlist?list=PLxxxx" --template detailed --batch

# Control concurrency
python -m noteking.cli run "..." --batch --workers 3
```

---

## 9. Template Guide

| Template | Name | Best For | Flag |
|----------|------|----------|------|
| `brief` | Brief Summary | Quick overview | `-t brief` |
| `detailed` | Detailed Notes | Structured study notes | `-t detailed` |
| `mindmap` | Mind Map | Knowledge structure | `-t mindmap` |
| `flashcard` | Flashcards | Anki-style review | `-t flashcard` |
| `quiz` | Quiz | Self-testing | `-t quiz` |
| `timeline` | Timeline | Time-stamped points | `-t timeline` |
| `exam` | Exam Review | Formula sheets + practice | `-t exam` |
| `tutorial` | Tutorial | Step-by-step guide | `-t tutorial` |
| `news` | News Brief | Journalism style | `-t news` |
| `podcast` | Podcast Summary | Audio content | `-t podcast` |
| `xhs_note` | Social Note | Social media sharing | `-t xhs_note` |
| `latex_pdf` | LaTeX PDF | Professional printable PDF | `-t latex_pdf` |
| `custom` | Custom | Your own prompt | `-t custom` |

---

## 10. Troubleshooting

**`git` not found**: Install Git from https://git-scm.com/

**`pip` not found**: Use `pip3` instead, or reinstall Python with "Add to PATH" checked.

**Docker won't start**: Ensure Docker Desktop is running. Windows users may need WSL 2 (`wsl --install`).

**API errors**: Double-check your API key, base URL, and that your plan has remaining credits.

**YouTube blocked**: Set a proxy: `export NOTEKING_PROXY="http://127.0.0.1:7890"`

**LaTeX missing packages**: Run `tlmgr install <package-name>` for any missing `.sty` files.

---

> Questions? Open an issue on [GitHub](https://github.com/bcefghj/noteking/issues).
