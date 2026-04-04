"""Multi-format output engine: generate Markdown, SRT, VTT, mind map, flashcards, JSON.

Takes a processing result and renders it in multiple output formats simultaneously.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .diarizer import DiarizationResult, DiarizedSegment


@dataclass
class OutputBundle:
    """All generated output files from a single processing run."""
    markdown_path: Path | None = None
    pdf_path: Path | None = None
    srt_path: Path | None = None
    vtt_path: Path | None = None
    mindmap_path: Path | None = None
    flashcard_path: Path | None = None
    transcript_path: Path | None = None
    action_items_path: Path | None = None
    json_path: Path | None = None
    all_files: dict[str, Path] = field(default_factory=dict)


def save_markdown(content: str, output_path: Path) -> Path:
    """Save generated notes as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def save_srt(
    diarization: DiarizationResult,
    output_path: Path,
    include_speaker: bool = True,
) -> Path:
    """Save transcript as SRT subtitle file with optional speaker labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, seg in enumerate(diarization.segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}")
        prefix = f"[{seg.speaker}] " if include_speaker and seg.speaker else ""
        lines.append(f"{prefix}{seg.text}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_vtt(
    diarization: DiarizationResult,
    output_path: Path,
    include_speaker: bool = True,
) -> Path:
    """Save transcript as WebVTT subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(diarization.segments, 1):
        lines.append(str(i))
        lines.append(f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}")
        prefix = f"<v {seg.speaker}>" if include_speaker and seg.speaker else ""
        lines.append(f"{prefix}{seg.text}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_transcript(
    diarization: DiarizationResult,
    output_path: Path,
) -> Path:
    """Save full transcript with speaker labels and timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        diarization.transcript_with_speakers, encoding="utf-8"
    )
    return output_path


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save structured result as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def generate_outputs(
    notes_content: str,
    diarization: DiarizationResult | None,
    output_dir: Path,
    base_name: str,
    formats: list[str],
    full_result: dict[str, Any] | None = None,
) -> OutputBundle:
    """Generate all requested output formats.

    Args:
        notes_content: Generated notes in Markdown
        diarization: Speaker-labeled transcript
        output_dir: Directory for output files
        base_name: Base filename (without extension)
        formats: List of formats: markdown, pdf, srt, vtt, transcript, json, mindmap, flashcard
        full_result: Complete result dict for JSON export
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(base_name)
    bundle = OutputBundle()

    if "markdown" in formats:
        path = output_dir / f"{safe_name}.md"
        save_markdown(notes_content, path)
        bundle.markdown_path = path
        bundle.all_files["markdown"] = path

    if "srt" in formats and diarization:
        path = output_dir / f"{safe_name}.srt"
        save_srt(diarization, path)
        bundle.srt_path = path
        bundle.all_files["srt"] = path

    if "vtt" in formats and diarization:
        path = output_dir / f"{safe_name}.vtt"
        save_vtt(diarization, path)
        bundle.vtt_path = path
        bundle.all_files["vtt"] = path

    if "transcript" in formats and diarization:
        path = output_dir / f"{safe_name}_transcript.txt"
        save_transcript(diarization, path)
        bundle.transcript_path = path
        bundle.all_files["transcript"] = path

    if "json" in formats and full_result:
        path = output_dir / f"{safe_name}.json"
        save_json(full_result, path)
        bundle.json_path = path
        bundle.all_files["json"] = path

    return bundle


def _safe_filename(text: str, max_len: int = 80) -> str:
    return "".join(
        c if c.isalnum() or c in " _-" else "_" for c in text
    )[:max_len].strip()


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
