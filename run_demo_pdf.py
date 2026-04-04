"""
NoteKing Pro Demo — 为 OpenClaw 圆桌会议生成 LaTeX PDF 精美图文讲义
复用已有的 ASR 转录结果 + 提取视频关键帧 + MiniMax M2.7 生成讲义内容
"""
import sys
import re
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

VIDEO_PATH = Path("/Users/daishanghao/Desktop/20260405_会议视频录音/OpenClaw与AI开源圆桌会议 月之暗面创始人杨植麟主持 - 001 - OpenClaw与AI开源圆桌会议 月之暗面创始人杨植麟主持.mp4")
OUTPUT_DIR = Path(__file__).parent / "demos" / "openclaw_meeting"
FRAMES_DIR = OUTPUT_DIR / "frames"
LATEX_BUILD = OUTPUT_DIR / "latex_build"
MINIMAX_KEY = "sk-cp-vX4T-YhmjkytkOexcwZ-uAdmALWR8ggXmtGOymuJQ1lfNLOR1phT0Ju09VggOTENL-y1pGe-KC4fTQppbzn_X_WPxVIApwG71PlvZHCGgfaIRH2zYoAI_RA"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
LATEX_BUILD.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("NoteKing Pro — 生成 LaTeX PDF 精美图文讲义")
print("=" * 60)

# ── Step 1: 提取视频关键帧 ────────────────────────────────────────
print("\n[1/4] 提取视频关键帧...")
from core.pdf_engine import SmartFrameExtractor, ScoredFrame, SubtitleFrameAligner

extractor = SmartFrameExtractor(max_frames=12, min_interval=60.0)
frames = extractor.extract(VIDEO_PATH, FRAMES_DIR)
print(f"  ✓ 提取 {len(frames)} 张关键帧")
for i, f in enumerate(frames):
    print(f"    Fig.{i+1}: {f.time_str} (score={f.total_score:.2f})")

# ── Step 2: 加载转录结果并对齐帧 ──────────────────────────────────
print("\n[2/4] 加载转录并对齐字幕...")
with open(OUTPUT_DIR / "transcript.json", encoding="utf-8") as f:
    transcript_data = json.load(f)
full_text = transcript_data["full_text"]
segments = transcript_data["segments"]

aligner = SubtitleFrameAligner(tolerance=30.0)
frames = aligner.align(frames, segments)
for i, f in enumerate(frames):
    print(f"    Fig.{i+1} [{f.time_str}]: {f.subtitle_text[:50]}..." if f.subtitle_text else f"    Fig.{i+1} [{f.time_str}]: (无字幕)")

# ── Step 3: 用 MiniMax 生成图文讲义内容 ──────────────────────────
print("\n[3/4] 调用 MiniMax M2.7 生成图文讲义...")
from openai import OpenAI
import httpx

client = OpenAI(
    api_key=MINIMAX_KEY,
    base_url="https://api.minimax.chat/v1",
    timeout=httpx.Timeout(300.0, connect=60.0),
    max_retries=3,
)

frames_desc = "\n".join(
    f"  Fig.{i+1} at {f.time_str}: {f.subtitle_text[:80] or '(会议画面)'}"
    for i, f in enumerate(frames)
)

MAX_CHARS = 35000
transcript_for_llm = full_text[:MAX_CHARS]

prompt = f"""你是专业教育内容专家。请根据以下会议转录文字和关键帧信息，生成一份详细的中文图文讲义。

会议信息：
- 标题：OpenClaw与AI开源圆桌会议
- 主持人：杨植麟（月之暗面/Kimi创始人）
- 参与者：AI领域多位专家
- 时长：约35分钟
- 来源：Bilibili BV1GX9pB9E6N

可用关键帧（共{len(frames)}张）：
{frames_desc}

转录内容：
{transcript_for_llm}

格式要求：
1. 用 ## 作为大节标题，### 作为子节标题
2. 在适当位置插入 {{IMAGE:N}} 标记来引用关键帧（如 {{IMAGE:1}} 到 {{IMAGE:{len(frames)}}}）
3. 每张图片下方要有描述性文字说明
4. 使用高亮框标记重要概念：
   {{IMPORTANT}}重点内容{{/IMPORTANT}}
   {{KNOWLEDGE}}背景知识补充{{/KNOWLEDGE}}
   {{WARNING}}注意事项{{/WARNING}}
5. 适当使用表格对比不同观点
6. 全部使用中文

内容要求：
- 按会议议题分节，结构清晰
- 每节开头有要点概述
- 引用嘉宾原话时使用引用格式
- 对关键概念（OpenClaw、Agent、MCP等）进行解释
- 包含技术要点和行业洞察
- 末尾添加总结部分和学习要点回顾
- 适合存档学习和分享的高质量讲义

请直接输出讲义内容，不要输出其他说明。"""

system = "你是专业的技术教育内容创作者。直接输出结构化的Markdown讲义内容，使用指定的标记格式。不要使用think标签。"

print("  请求 MiniMax M2.7...")
resp = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    max_tokens=6000,
    temperature=0.2,
)
notes_md = resp.choices[0].message.content or ""
notes_md = re.sub(r"<think>.*?</think>\s*", "", notes_md, flags=re.DOTALL).strip()
print(f"  ✓ 讲义内容生成完成 ({len(notes_md)} 字)")

# ── Step 4: 编译 LaTeX PDF ────────────────────────────────────────
print("\n[4/4] 编译 LaTeX PDF...")
from core.pdf_engine import LaTeXNoteBuilder

latex_builder = LaTeXNoteBuilder()
title = "OpenClaw与AI开源圆桌会议 图文讲义"
meta = {
    "uploader": "杨植麟（月之暗面/Kimi创始人）",
    "duration": "35m42s",
    "url": "https://www.bilibili.com/video/BV1GX9pB9E6N/",
}

tex_content = latex_builder.build_tex(
    notes_md=notes_md,
    frames=frames,
    title=title,
    meta=meta,
)

tex_path = LATEX_BUILD / "meeting_notes.tex"
tex_path.write_text(tex_content, encoding="utf-8")
print(f"  ✓ LaTeX 源文件: {tex_path.name}")

pdf_result = latex_builder.compile_pdf(tex_path, LATEX_BUILD)

if pdf_result and pdf_result.exists():
    import shutil
    final_pdf = OUTPUT_DIR / "OpenClaw圆桌会议_图文讲义.pdf"
    shutil.copy2(pdf_result, final_pdf)
    print(f"  ✓ PDF 编译成功: {final_pdf.name} ({final_pdf.stat().st_size // 1024} KB)")

    notes_md_path = OUTPUT_DIR / "图文讲义.md"
    notes_md_path.write_text(f"# {title}\n\n{notes_md}", encoding="utf-8")
    print(f"  ✓ 讲义Markdown: {notes_md_path.name}")
else:
    print("  ⚠ LaTeX 编译失败，尝试 HTML → PDF...")
    from core.pdf_engine import HTMLNoteBuilder
    html_builder = HTMLNoteBuilder()
    html_content = html_builder.build_html(
        notes_md=notes_md,
        frames=frames,
        title=title,
        meta=meta,
        cover_path=frames[0].path if frames else None,
    )
    html_path = OUTPUT_DIR / "OpenClaw圆桌会议_图文讲义.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  ✓ HTML 讲义: {html_path.name}")

    final_pdf = OUTPUT_DIR / "OpenClaw圆桌会议_图文讲义.pdf"
    ok = HTMLNoteBuilder.html_to_pdf(html_path, final_pdf)
    if ok:
        print(f"  ✓ PDF (via Chrome): {final_pdf.name} ({final_pdf.stat().st_size // 1024} KB)")
    else:
        print("  ⚠ Chrome PDF 也失败了")

    notes_md_path = OUTPUT_DIR / "图文讲义.md"
    notes_md_path.write_text(f"# {title}\n\n{notes_md}", encoding="utf-8")
    print(f"  ✓ 讲义Markdown: {notes_md_path.name}")

print("\n" + "=" * 60)
print("✅ LaTeX PDF 讲义生成完成！")
print(f"输出目录: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        label = f"{size//1024}KB" if size > 1024 else f"{size}B"
        print(f"  {f.name} ({label})")
