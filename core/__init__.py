"""NoteKing Core Engine: the ultimate video/blog/recording to notes tool.

Supports:
- Online video URLs (30+ platforms via yt-dlp)
- Local video/audio files (MP4, MP3, WAV, etc.)
- Meeting recordings with speaker diarization
- Multi-file batch processing with auto-merge
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .parser import parse_link, ParsedLink, Platform, LinkType, is_batch
from .downloader import get_video_info, VideoMeta
from .subtitle import extract_subtitles, SubtitleResult
from .llm import chat
from .cache import Cache
from .templates import get_template, TEMPLATES, TEMPLATE_LIST, TemplateContext
from .batch import get_batch_entries, process_batch, merge_batch_notes, BatchProgress

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


def summarize(
    url: str,
    template: str = "detailed",
    config: AppConfig | None = None,
    custom_prompt: str = "",
    use_cache: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Main entry point: convert a video/blog URL into notes.

    Args:
        url: Video URL or local file path
        template: Output template name (see TEMPLATE_LIST)
        config: Configuration (loads default if None)
        custom_prompt: Custom prompt for the 'custom' template
        use_cache: Whether to use cached results
        output_dir: Directory to save output files

    Returns:
        dict with keys: title, content, template, source, meta
    """
    if config is None:
        config = AppConfig.load()

    cache = Cache(config)
    if use_cache:
        cached = cache.get(url, template)
        if cached:
            return cached

    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_link(url)

    if is_batch(parsed):
        return _process_batch(parsed, template, config, custom_prompt, out_dir)

    return _process_single(
        parsed, template, config, custom_prompt, out_dir, cache, use_cache
    )


def _process_single(
    parsed: ParsedLink,
    template_name: str,
    config: AppConfig,
    custom_prompt: str,
    out_dir: Path,
    cache: Cache,
    use_cache: bool,
) -> dict[str, Any]:
    """Process a single video."""
    import tempfile

    work_dir = Path(tempfile.mkdtemp(prefix="noteking_"))

    # Step 1: Get video metadata
    if parsed.platform == Platform.LOCAL:
        meta = VideoMeta(title=Path(parsed.url).stem, webpage_url=parsed.url)
    else:
        meta = get_video_info(parsed.url, config)

    # Step 2: Extract subtitles (three-level fallback)
    subtitles = extract_subtitles(parsed, work_dir, config)

    if not subtitles.segments and subtitles.source == "visual":
        subtitles = SubtitleResult(
            segments=[],
            source="visual",
            raw_text=f"[视频无可用字幕，标题: {meta.title}，描述: {meta.description[:500]}]",
        )

    # Step 3: Generate notes using template
    tmpl = get_template(template_name, user_prompt=custom_prompt)
    ctx = TemplateContext(
        meta=meta,
        subtitles=subtitles,
        config=config,
        extra={"custom_prompt": custom_prompt},
    )
    content = tmpl.generate(ctx)

    # Step 4: Save output
    ext = tmpl.file_extension
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in meta.title)[:80]
    output_file = out_dir / f"{safe_title}_{template_name}{ext}"
    output_file.write_text(content, encoding="utf-8")

    # Save subtitle files
    if subtitles.segments:
        subtitles.save_srt(out_dir / f"{safe_title}.srt")
        subtitles.save_txt(out_dir / f"{safe_title}_transcript.txt")

    result = {
        "title": meta.title,
        "content": content,
        "template": template_name,
        "source": subtitles.source,
        "output_file": str(output_file),
        "url": parsed.url,
        "platform": parsed.platform.value,
        "duration": meta.duration,
        "uploader": meta.uploader,
    }

    if use_cache:
        cache.set(parsed.url, template_name, result)

    return result


def _process_batch(
    parsed: ParsedLink,
    template_name: str,
    config: AppConfig,
    custom_prompt: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Process a batch (playlist/collection)."""
    entries = get_batch_entries(parsed, config)
    cache = Cache(config)

    all_results: list[dict] = []

    def process_one(entry_url: str, idx: int) -> dict:
        entry_parsed = parse_link(entry_url)
        r = _process_single(
            entry_parsed, template_name, config, custom_prompt,
            out_dir, cache, True,
        )
        all_results.append(r)
        return r

    progress = process_batch(entries, process_one)

    merged = merge_batch_notes(
        [{"title": r.get("title", ""), "content": r.get("content", "")}
         for r in all_results],
        title=f"{parsed.video_id} 合集笔记",
    )

    merged_file = out_dir / f"batch_merged_{template_name}.md"
    merged_file.write_text(merged, encoding="utf-8")

    return {
        "title": f"合集 ({len(all_results)} 个视频)",
        "content": merged,
        "template": template_name,
        "source": "batch",
        "output_file": str(merged_file),
        "url": parsed.url,
        "platform": parsed.platform.value,
        "total": progress.total,
        "completed": progress.completed,
        "failed": progress.failed,
        "individual_results": all_results,
    }


def get_transcript(
    url: str,
    config: AppConfig | None = None,
) -> str:
    """Get just the transcript text for a video."""
    if config is None:
        config = AppConfig.load()

    parsed = parse_link(url)
    work_dir = Path(tempfile.mkdtemp(prefix="noteking_"))
    subtitles = extract_subtitles(parsed, work_dir, config)
    return subtitles.full_text


# ────────────────────────────────────────────────────────────────
# Recording Processing Pipeline (NEW in v2.0)
# ────────────────────────────────────────────────────────────────

def process_recording(
    input_files: list[str | Path],
    template: str = "meeting_minutes",
    context: str | None = None,
    scene: str | None = None,
    num_speakers: int | None = None,
    denoise_level: int = 1,
    output_formats: list[str] | None = None,
    config: AppConfig | None = None,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Process local recording/video files into structured notes.

    This is the main entry point for the recording processing pipeline:
    1. Preprocess: validate, merge, extract audio, denoise
    2. Transcribe: ASR with language auto-detection
    3. Diarize: speaker identification and alignment
    4. Generate: LLM-powered note generation with scene template
    5. Output: multi-format export (Markdown, PDF, SRT, etc.)

    Args:
        input_files: List of local file paths (video or audio)
        template: Output template name (meeting_minutes, lecture_notes, etc.)
        context: User description ("This is a meeting about AI open source projects")
        scene: Scene type (meeting/lecture/interview/brainstorm/news/exam/entertainment)
        num_speakers: Expected number of speakers (None for auto-detect)
        denoise_level: 0=none, 1=light, 2=medium, 3=heavy
        output_formats: List of formats (markdown, pdf, srt, vtt, transcript, json)
        config: App configuration
        output_dir: Directory for output files
        progress_callback: Optional progress callback(step_name, progress_pct)

    Returns:
        dict with keys: title, content, template, transcript, speakers,
        duration, output_files, diarization
    """
    from .preprocessor import preprocess, detect_media_type
    from .transcriber import transcribe, detect_language
    from .diarizer import diarize, DiarizationResult
    from .formatter import generate_outputs

    if config is None:
        config = AppConfig.load()

    if output_formats is None:
        output_formats = ["markdown"]

    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="noteking_rec_"))

    def _progress(step: str, pct: float):
        logger.info(f"[{step}] {pct*100:.0f}%")
        if progress_callback:
            progress_callback(step, pct)

    # Step 1: Preprocess
    _progress("preprocess", 0.0)
    pre_result = preprocess(
        input_files=input_files,
        work_dir=work_dir,
        denoise_level=denoise_level,
        progress_callback=lambda s, p: _progress(f"preprocess/{s}", p * 0.2),
    )
    _progress("preprocess", 0.2)

    # Step 2: Language detection + ASR transcription
    _progress("transcribe", 0.2)
    language = detect_language(pre_result.audio_path, config)
    transcript = transcribe(pre_result.audio_path, config, language=language)
    _progress("transcribe", 0.5)

    # Step 3: Speaker diarization
    _progress("diarize", 0.5)
    if config.recording.enable_diarization and len(transcript.segments) > 0:
        diar_result = diarize(
            audio_path=pre_result.audio_path,
            transcript=transcript,
            num_speakers=num_speakers or config.recording.num_speakers,
        )
    else:
        from .diarizer import DiarizationResult, DiarizedSegment
        diar_result = DiarizationResult(
            segments=[
                DiarizedSegment(s.start, s.end, s.text, "Speaker 1")
                for s in transcript.segments
            ],
            num_speakers=1,
            speakers=["Speaker 1"],
            language=language,
        )
    _progress("diarize", 0.6)

    # Step 4: Generate notes using template + LLM
    _progress("generate", 0.6)

    title = _derive_title(input_files, context)
    transcript_for_llm = _build_transcript_for_llm(diar_result, context, scene)

    meta = VideoMeta(
        title=title,
        description=context or "",
        duration=pre_result.duration,
        webpage_url=str(input_files[0]),
    )

    tmpl = get_template(template)
    sub_result = SubtitleResult(
        segments=transcript.segments,
        source="asr",
        language=language,
        raw_text=transcript_for_llm,
    )
    ctx = TemplateContext(
        meta=meta,
        subtitles=sub_result,
        config=config,
        extra={
            "context": context or "",
            "scene": scene or template,
            "num_speakers": diar_result.num_speakers,
            "speakers": diar_result.speakers,
            "diarized_transcript": diar_result.transcript_with_speakers,
        },
    )
    content = tmpl.generate(ctx)
    _progress("generate", 0.85)

    # Step 5: Multi-format output
    _progress("output", 0.85)
    base_name = _safe_title(title)
    bundle = generate_outputs(
        notes_content=content,
        diarization=diar_result,
        output_dir=out_dir,
        base_name=f"{base_name}_{template}",
        formats=output_formats,
        full_result={
            "title": title,
            "template": template,
            "language": language,
            "duration": pre_result.duration,
            "num_speakers": diar_result.num_speakers,
            "speakers": diar_result.speakers,
            "content": content,
        },
    )
    _progress("done", 1.0)

    return {
        "title": title,
        "content": content,
        "template": template,
        "source": "recording",
        "language": language,
        "duration": pre_result.duration,
        "num_speakers": diar_result.num_speakers,
        "speakers": diar_result.speakers,
        "output_files": {k: str(v) for k, v in bundle.all_files.items()},
        "transcript": diar_result.full_text,
        "diarized_transcript": diar_result.transcript_with_speakers,
    }


def _derive_title(input_files: list[str | Path], context: str | None) -> str:
    """Derive a title from filenames or user context."""
    if context:
        return context[:80]
    names = [Path(f).stem for f in input_files]
    if len(names) == 1:
        return names[0]
    return f"{names[0]} 等{len(names)}个文件"


def _build_transcript_for_llm(
    diar: Any,
    context: str | None,
    scene: str | None,
) -> str:
    """Build a rich transcript string for LLM consumption."""
    parts = []
    if context:
        parts.append(f"[内容描述]: {context}")
    if scene:
        parts.append(f"[场景类型]: {scene}")
    if diar.num_speakers > 1:
        parts.append(f"[说话人数量]: {diar.num_speakers}")
    parts.append("")
    parts.append(diar.transcript_with_speakers)
    return "\n".join(parts)


def _safe_title(text: str) -> str:
    return "".join(
        c if c.isalnum() or c in " _-" else "_" for c in text
    )[:80].strip()
