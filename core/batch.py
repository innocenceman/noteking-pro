"""Batch processing: playlists, collections, and multi-part videos."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .downloader import list_playlist_entries, get_video_info, VideoMeta
from .parser import ParsedLink, is_batch


@dataclass
class BatchProgress:
    total: int = 0
    completed: int = 0
    current_title: str = ""
    failed: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0
        return (self.completed / self.total) * 100

    @property
    def status_line(self) -> str:
        return (
            f"[{self.completed}/{self.total}] "
            f"({self.percent:.0f}%) {self.current_title}"
        )


@dataclass
class BatchResult:
    progress: BatchProgress
    output_dir: Path
    merged_output: str = ""

    @property
    def success_count(self) -> int:
        return self.progress.completed - len(self.progress.failed)


def get_batch_entries(parsed: ParsedLink, config: AppConfig) -> list[dict]:
    """Get all entries from a playlist/collection."""
    if not is_batch(parsed):
        return [{"url": parsed.url, "title": ""}]

    entries = list_playlist_entries(parsed.url, config)
    results = []
    for entry in entries:
        url = entry.get("url") or entry.get("webpage_url", "")
        if not url and entry.get("id"):
            if "bilibili" in parsed.url:
                url = f"https://www.bilibili.com/video/{entry['id']}"
            elif "youtube" in parsed.url:
                url = f"https://www.youtube.com/watch?v={entry['id']}"
        if url:
            results.append({
                "url": url,
                "title": entry.get("title", ""),
                "duration": entry.get("duration", 0),
            })
    return results


def process_batch(
    entries: list[dict],
    process_fn: Callable[[str, int], dict],
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> BatchProgress:
    """Process a batch of entries, calling process_fn for each."""
    progress = BatchProgress(total=len(entries))

    for i, entry in enumerate(entries):
        url = entry.get("url", "")
        title = entry.get("title", f"Video {i + 1}")
        progress.current_title = title

        if progress_callback:
            progress_callback(progress)

        try:
            result = process_fn(url, i)
            progress.results.append(result)
        except Exception as e:
            progress.failed.append(f"{title}: {e}")

        progress.completed = i + 1
        if progress_callback:
            progress_callback(progress)

    return progress


def merge_batch_notes(
    results: list[dict],
    title: str = "课程笔记合集",
) -> str:
    """Merge multiple note results into a single document."""
    lines = [f"# {title}\n"]
    lines.append(f"共 {len(results)} 个视频\n")
    lines.append("---\n")

    lines.append("## 目录\n")
    for i, r in enumerate(results, 1):
        t = r.get("title", f"第 {i} 节")
        lines.append(f"{i}. [{t}](#{_slug(t)})")
    lines.append("\n---\n")

    for i, r in enumerate(results, 1):
        t = r.get("title", f"第 {i} 节")
        content = r.get("content", "")
        lines.append(f"\n## {i}. {t}\n")
        lines.append(content)
        lines.append("\n---\n")

    return "\n".join(lines)


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-").replace("/", "-")[:50]


def save_batch_checkpoint(
    progress: BatchProgress,
    checkpoint_path: Path,
) -> None:
    """Save progress for resume capability."""
    data = {
        "total": progress.total,
        "completed": progress.completed,
        "failed": progress.failed,
        "results_count": len(progress.results),
    }
    checkpoint_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_batch_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load a previous checkpoint for resuming."""
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
