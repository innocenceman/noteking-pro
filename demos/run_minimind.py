#!/usr/bin/env python3
"""
NoteKing Demo: Process all 26 episodes of MiniMind into illustrated PDF lecture notes.

Usage: python3.11 run_minimind.py
"""

import sys, os, time, re, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

# ── Config ──
API_KEY = "sk-cp-vX4T-YhmjkytkOexcwZ-uAdmALWR8ggXmtGOymuJQ1lfNLOR1phT0Ju09VggOTENL-y1pGe-KC4fTQppbzn_X_WPxVIApwG71PlvZHCGgfaIRH2zYoAI_RA"
BASE_URL = "https://api.minimax.chat/v1"
MODEL = "MiniMax-M2.7"
CONCURRENCY = 3
MAX_TOKENS = 5000

BASE_DIR = Path(__file__).resolve().parent / "minimind"
VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"
NOTES_DIR = BASE_DIR / "notes"
PDF_DIR = BASE_DIR / "pdf"

EPISODES = [
    (1,  "开篇",              170,  "课程介绍与背景"),
    (2,  "前言",              268,  "大模型基础概念、MiniMind技术栈"),
    (3,  "必看：前言补充",     123,  "勘误与补充说明"),
    (4,  "前置知识",          437,  "PyTorch张量、矩阵乘法、注意力机制原理"),
    (5,  "架构图解读",         346,  "Transformer架构全图逐层解析"),
    (6,  "初始化项目",         227,  "项目结构设计、ModelArgs配置类"),
    (7,  "理论：RMSNorm",      246,  "RMSNorm原理与数学推导"),
    (8,  "代码：RMSNorm",      385,  "RMSNorm的PyTorch实现"),
    (9,  "理论：RoPE&YaRN",    831,  "旋转位置编码RoPE + YaRN外推原理"),
    (10, "代码：RoPE&YaRN",   1040, "RoPE实现、旋转矩阵计算、YaRN缩放"),
    (11, "理论：GQA",          230,  "分组查询注意力 MHA/MQA/GQA对比"),
    (12, "代码：GQA 上",       788,  "Q/K/V投影、repeat_kv"),
    (13, "代码：GQA 下",      1162,  "注意力计算、因果掩码、输出投影"),
    (14, "理论：FFN",          376,  "前馈网络、SwiGLU激活函数"),
    (15, "代码：FFN",          339,  "FeedForward类的PyTorch实现"),
    (16, "拼接：Block",        347,  "TransformerBlock组装"),
    (17, "组装：Model",        703,  "Transformer完整实现"),
    (18, "封装：CausalLM",     600,  "因果语言模型封装、generate推理"),
    (19, "回顾与知识检验",     435,  "全架构回顾、核心概念测验"),
    (20, "必看：纠错补充",     475,  "代码勘误、补充说明"),
    (21, "重制Dataset：理论",  332,  "预训练数据格式、tokenizer"),
    (22, "重制Dataset：代码",  484,  "PretrainDataset类实现"),
    (23, "重制Pretrain：理论", 344,  "预训练目标、损失函数"),
    (24, "重制Pretrain：代码", 580,  "训练循环、优化器、学习率调度"),
    (25, "训练，启动！",       343,  "实际训练流程、日志分析"),
    (26, "Eval：完结！",       497,  "模型评估、推理测试、展望"),
]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def find_video(ep_num):
    """Find downloaded video file for episode."""
    patterns = [
        f"p{ep_num:02d}_*.mp4",
        f"*p{ep_num:02d}*.mp4",
        f"*_p{ep_num:02d}_*.mp4",
    ]
    for pat in patterns:
        matches = list(VIDEOS_DIR.glob(pat))
        if matches:
            return matches[0]
    return None


def extract_frames(video_path, ep_num, title):
    """Extract keyframes for one episode."""
    ep_frames_dir = FRAMES_DIR / f"p{ep_num:02d}"
    ep_frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(ep_frames_dir.glob("*.jpg"))
    if len(existing) >= 3:
        return existing

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.frames import extract_keyframes
        from core.frames import ExtractedFrame

        frames = extract_keyframes(
            video_path, ep_frames_dir,
            max_frames=12, threshold=25.0,
            dedup=True, score=True,
        )
        return [f.path for f in frames]
    except Exception as e:
        print(f"    SceneDetect/scoring failed ({e}), using ffmpeg uniform...")

    # Fallback: uniform
    duration_cmd = f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{video_path}"'
    r = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True, timeout=30)
    duration = float(r.stdout.strip()) if r.stdout.strip() else 300

    interval = max(15, duration / 12)
    frame_paths = []
    t = 3.0
    idx = 0
    while t < duration - 3 and idx < 15:
        out = ep_frames_dir / f"frame_{idx:03d}_{t:.0f}s.jpg"
        cmd = f'ffmpeg -ss {t:.2f} -i "{video_path}" -vframes 1 -q:v 3 -vf scale=960:-1 "{out}" -y -loglevel error'
        subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        if out.exists() and out.stat().st_size > 3000:
            frame_paths.append(out)
        t += interval
        idx += 1

    return frame_paths


def generate_notes(ep_num, title, duration, content_hint, frame_paths):
    """Generate illustrated lecture notes for one episode via LLM."""
    mins = duration // 60
    secs = duration % 60

    frames_desc = ""
    if frame_paths:
        frames_desc = "Available keyframes (insert {IMAGE:N} where appropriate):\n"
        for i, fp in enumerate(frame_paths):
            m = re.search(r'(\d+\.?\d*)s', fp.name)
            ts = m.group(1) if m else str(i * 30)
            frames_desc += f"  Fig.{i+1} at {ts}s\n"

    prompt = f"""Generate detailed illustrated lecture notes in Chinese for this MiniMind episode.

Episode: {ep_num}/26 "{title}" ({int(mins)}m{int(secs)}s)
Topic: {content_hint}
Course: MiniMind - PyTorch从零手敲大模型

{frames_desc}

REQUIRED FORMAT:
- Use ## for sections, ### for subsections
- Insert {{IMAGE:N}} markers for keyframe figures (e.g. {{IMAGE:1}})
- Use highlight boxes:
  {{IMPORTANT}}核心概念{{/IMPORTANT}}
  {{KNOWLEDGE}}背景知识{{/KNOWLEDGE}}
  {{WARNING}}易错点{{/WARNING}}
- Code blocks with ```python
- Math with $...$ or $$...$$
- Each section ends with brief summary

CONTENT:
- Write as a professional lecture note, 2000+ characters
- Cover all knowledge points for this topic
- Include code with comments when relevant
- Add formulas with explanation
- End with key takeaways and 1-2 thought questions"""

    for attempt in range(3):
        try:
            full = ""
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是专业的AI技术讲义编写专家。直接输出Markdown格式讲义。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS, stream=True, temperature=0.2, timeout=120,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
            return strip_think(full)
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 8)
            else:
                return f"*Episode {ep_num} generation failed: {e}*"
    return ""


def build_html(notes_md, frame_paths, ep_num, title, duration):
    """Build HTML lecture note with embedded images."""
    def replace_img(match):
        n = int(match.group(1))
        if 1 <= n <= len(frame_paths):
            fp = frame_paths[n - 1]
            return f'\n<img src="file://{fp.resolve()}" alt="fig{n}" style="max-width:100%;border-radius:6px;box-shadow:0 3px 12px rgba(0,0,0,.12);margin:1rem auto;display:block">\n<div style="text-align:center;color:#666;font-size:.85em;margin-bottom:1rem">Fig.{n}</div>\n'
        return f"*(Fig.{n})*"

    body = re.sub(r'\{IMAGE:(\d+)\}', replace_img, notes_md)
    body = re.sub(r'\{IMPORTANT\}(.*?)\{/IMPORTANT\}',
                  r'<div style="background:#fff8e1;border-left:4px solid #ffb300;padding:1rem;margin:1rem 0;border-radius:0 8px 8px 0"><b>⭐ 重点</b><br>\1</div>',
                  body, flags=re.DOTALL)
    body = re.sub(r'\{KNOWLEDGE\}(.*?)\{/KNOWLEDGE\}',
                  r'<div style="background:#e3f2fd;border-left:4px solid #1565c0;padding:1rem;margin:1rem 0;border-radius:0 8px 8px 0"><b>📖 知识补充</b><br>\1</div>',
                  body, flags=re.DOTALL)
    body = re.sub(r'\{WARNING\}(.*?)\{/WARNING\}',
                  r'<div style="background:#fce4ec;border-left:4px solid #c62828;padding:1rem;margin:1rem 0;border-radius:0 8px 8px 0"><b>⚠️ 注意</b><br>\1</div>',
                  body, flags=re.DOTALL)

    style = """<style>
body{font-family:"PingFang SC","Noto Sans CJK SC",sans-serif;max-width:900px;margin:0 auto;padding:2rem;line-height:1.8;color:#2d2d2d}
h1{color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:.5rem}
h2{color:#16213e;border-left:4px solid #0f3460;padding-left:.8rem;margin-top:2rem}
h3{color:#0f3460}
code{background:#f0f0f0;padding:.15em .4em;border-radius:3px;font-size:.88em;font-family:"SF Mono",Menlo,monospace}
pre{background:#1e1e2e;color:#cdd6f4;padding:1.2rem;border-radius:8px;overflow-x:auto;font-size:.85em}
pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ddd;padding:.6rem .8rem}
th{background:#0f3460;color:white}
tr:nth-child(even){background:#f8f9fa}
blockquote{background:#e8f4fd;border-left:4px solid #2196f3;padding:.8rem 1rem;margin:1rem 0;border-radius:0 8px 8px 0}
@media print{body{max-width:100%;padding:0}pre{white-space:pre-wrap}}
</style>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Ep.{ep_num}: {title}</title>{style}</head>
<body>
<h1>第{ep_num}集: {title}</h1>
<p style="color:#666">MiniMind · PyTorch从零手敲大模型 | {int(duration//60)}分{int(duration%60)}秒 | NoteKing</p>
<hr>
{body}
</body></html>"""


def html_to_pdf(html_path, pdf_path):
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        return False
    cmd = [chrome, "--headless=new", f"--print-to-pdf={pdf_path}",
           "--print-to-pdf-no-header", "--no-sandbox", "--disable-gpu",
           f"file://{html_path.resolve()}"]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return pdf_path.exists()


def process_one_episode(idx):
    """Process a single episode end-to-end."""
    ep_num, title, duration, content_hint = EPISODES[idx]

    # Find video
    video_path = find_video(ep_num)
    if not video_path or not video_path.exists():
        print(f"  [{ep_num:02d}/26] ⚠️ Video not found, generating notes without frames")
        frame_paths = []
    else:
        print(f"  [{ep_num:02d}/26] Extracting frames from {video_path.name}...")
        frame_paths = extract_frames(video_path, ep_num, title)
        print(f"  [{ep_num:02d}/26] Got {len(frame_paths)} frames")

    # Generate notes
    print(f"  [{ep_num:02d}/26] Generating notes: {title}...")
    notes = generate_notes(ep_num, title, duration, content_hint, frame_paths)
    print(f"  [{ep_num:02d}/26] Notes: {len(notes)} chars")

    # Save markdown
    md_path = NOTES_DIR / f"p{ep_num:02d}_{title.replace('：','_').replace(':','_')}.md"
    md_path.write_text(f"# 第{ep_num}集: {title}\n\n{notes}", encoding="utf-8")

    # Build HTML
    html_content = build_html(notes, frame_paths, ep_num, title, duration)
    html_path = PDF_DIR / f"p{ep_num:02d}_{title.replace('：','_').replace(':','_')}.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Convert to PDF
    pdf_path = html_path.with_suffix(".pdf")
    html_to_pdf(html_path, pdf_path)

    return {
        "ep": ep_num, "title": title,
        "notes_len": len(notes),
        "frames": len(frame_paths),
        "pdf": str(pdf_path) if pdf_path.exists() else None,
        "html": str(html_path),
    }


def build_merged_html(results):
    """Merge all episodes into one HTML document."""
    style = """<style>
body{font-family:"PingFang SC",sans-serif;max-width:900px;margin:0 auto;padding:2rem;line-height:1.8;color:#2d2d2d}
h1{color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:.5rem}
h2{color:#16213e;border-left:4px solid #0f3460;padding-left:.8rem;margin-top:2rem}
h3{color:#0f3460}
code{background:#f0f0f0;padding:.15em .4em;border-radius:3px;font-size:.88em;font-family:"SF Mono",Menlo,monospace}
pre{background:#1e1e2e;color:#cdd6f4;padding:1.2rem;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit;padding:0}
img{max-width:100%;border-radius:6px;box-shadow:0 3px 12px rgba(0,0,0,.12);margin:1rem auto;display:block}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ddd;padding:.5rem .7rem}
th{background:#0f3460;color:white}
tr:nth-child(even){background:#f8f9fa}
blockquote{background:#e8f4fd;border-left:4px solid #2196f3;padding:.8rem 1rem;margin:1rem 0;border-radius:0 8px 8px 0}
.cover{text-align:center;padding:3rem 0;page-break-after:always}
hr{margin:3rem 0}
@media print{body{max-width:100%;padding:0}pre{white-space:pre-wrap}}
</style>"""

    parts = [f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>MiniMind 完整图文讲义</title>{style}</head><body>
<div class="cover">
<h1 style="font-size:2.2rem;border:none">MiniMind 完整图文讲义</h1>
<p style="font-size:1.2rem;color:#666">PyTorch从零手敲大模型 · 架构到训练全教程</p>
<p>共 26 集 | NoteKing · {time.strftime('%Y-%m-%d')}</p>
</div>
<h2>目录</h2><ol>"""]

    for r in results:
        parts.append(f'<li>第{r["ep"]}集: {r["title"]}</li>')
    parts.append("</ol><hr>")

    for r in results:
        md_path = NOTES_DIR / f"p{r['ep']:02d}_{r['title'].replace('：','_').replace(':','_')}.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            # Inline images from frames
            ep_frames = sorted((FRAMES_DIR / f"p{r['ep']:02d}").glob("*.jpg")) if (FRAMES_DIR / f"p{r['ep']:02d}").exists() else []
            def _rep(m, fps=ep_frames):
                n = int(m.group(1))
                if 1 <= n <= len(fps):
                    return f'\n<img src="file://{fps[n-1].resolve()}" alt="fig">\n'
                return ""
            content = re.sub(r'\{IMAGE:(\d+)\}', _rep, content)
            content = re.sub(r'\{IMPORTANT\}(.*?)\{/IMPORTANT\}',
                             r'<div style="background:#fff8e1;border-left:4px solid #ffb300;padding:.8rem;margin:.5rem 0;border-radius:0 6px 6px 0"><b>重点</b>: \1</div>',
                             content, flags=re.DOTALL)
            content = re.sub(r'\{KNOWLEDGE\}(.*?)\{/KNOWLEDGE\}',
                             r'<div style="background:#e3f2fd;border-left:4px solid #1565c0;padding:.8rem;margin:.5rem 0;border-radius:0 6px 6px 0"><b>知识</b>: \1</div>',
                             content, flags=re.DOTALL)
            content = re.sub(r'\{WARNING\}(.*?)\{/WARNING\}',
                             r'<div style="background:#fce4ec;border-left:4px solid #c62828;padding:.8rem;margin:.5rem 0;border-radius:0 6px 6px 0"><b>注意</b>: \1</div>',
                             content, flags=re.DOTALL)
            parts.append(content)
        parts.append("<hr>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    print("=" * 65)
    print("  NoteKing · MiniMind 全集图文讲义生成器")
    print(f"  共 {len(EPISODES)} 集 | 并发 {CONCURRENCY} 路 | MiniMax M2.7 Max")
    print("=" * 65)

    for d in [NOTES_DIR, PDF_DIR, FRAMES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Check video downloads
    video_count = len(list(VIDEOS_DIR.glob("*.mp4"))) if VIDEOS_DIR.exists() else 0
    print(f"\n  已下载视频: {video_count}/26")
    if video_count == 0:
        print("  ⚠️ 视频尚未下载完成，将在无截图模式下生成笔记")
        print("  (视频下载完成后可重新运行以添加截图)")

    # Process all episodes concurrently
    results = [None] * len(EPISODES)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(process_one_episode, i): i for i in range(len(EPISODES))}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
                r = results[idx]
                print(f"  ✅ [{r['ep']:02d}/26] {r['title']} — {r['notes_len']} chars, {r['frames']} frames")
            except Exception as e:
                ep = EPISODES[idx]
                print(f"  ❌ [{ep[0]:02d}/26] {ep[1]} — FAILED: {e}")
                results[idx] = {"ep": ep[0], "title": ep[1], "notes_len": 0, "frames": 0, "pdf": None, "html": None}

    elapsed = time.time() - t0

    # Build merged HTML
    print("\n  Merging all episodes into one document...")
    valid_results = [r for r in results if r]
    valid_results.sort(key=lambda r: r["ep"])
    merged_html = build_merged_html(valid_results)
    merged_path = PDF_DIR / "MiniMind_全集_图文讲义.html"
    merged_path.write_text(merged_html, encoding="utf-8")

    # Convert merged to PDF
    merged_pdf = merged_path.with_suffix(".pdf")
    html_to_pdf(merged_path, merged_pdf)

    # Summary
    print("\n" + "=" * 65)
    print(f"  ✅ 全部完成！ 耗时 {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
    success = sum(1 for r in valid_results if r["notes_len"] > 200)
    total_chars = sum(r["notes_len"] for r in valid_results)
    total_frames = sum(r["frames"] for r in valid_results)
    print(f"  📊 成功: {success}/26 集")
    print(f"  📝 总字数: {total_chars:,}")
    print(f"  🖼️ 总截图: {total_frames}")
    print(f"  📄 合并HTML: {merged_path}")
    if merged_pdf.exists():
        print(f"  📕 合并PDF: {merged_pdf} ({merged_pdf.stat().st_size//1024}KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
