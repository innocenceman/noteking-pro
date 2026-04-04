"""Base template class for all output formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..downloader import VideoMeta
from ..subtitle import SubtitleResult


@dataclass
class TemplateContext:
    meta: VideoMeta
    subtitles: SubtitleResult
    config: AppConfig
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def transcript(self) -> str:
        return self.subtitles.full_text

    @property
    def has_chapters(self) -> bool:
        return bool(self.meta.chapters)

    @property
    def chapter_texts(self) -> list[dict]:
        """Split transcript by video chapters."""
        if not self.meta.chapters or not self.subtitles.segments:
            return [{"title": self.meta.title, "text": self.transcript}]

        results = []
        for ch in self.meta.chapters:
            start = ch.get("start_time", 0)
            end = ch.get("end_time", float("inf"))
            title = ch.get("title", "")
            segs = [
                s.text for s in self.subtitles.segments
                if s.start >= start and s.start < end
            ]
            results.append({"title": title, "text": "\n".join(segs)})
        return results


class BaseTemplate(ABC):
    """All output templates inherit from this."""

    name: str = "base"
    display_name: str = "Base"
    description: str = ""
    file_extension: str = ".md"

    @abstractmethod
    def build_prompt(self, ctx: TemplateContext) -> str:
        """Build the LLM prompt for this template."""
        ...

    def system_prompt(self, ctx: TemplateContext) -> str:
        lang = ctx.config.llm.language
        lang_name = "中文" if "zh" in lang else "English"
        return (
            f"You are NoteKing, an expert at converting video/audio content into "
            f"high-quality structured notes. Always respond in {lang_name}. "
            f"Be thorough, accurate, and well-organized."
        )

    def post_process(self, result: str, ctx: TemplateContext) -> str:
        """Optional post-processing of LLM output."""
        return result

    def generate(self, ctx: TemplateContext) -> str:
        from ..llm import chat

        prompt = self.build_prompt(ctx)
        system = self.system_prompt(ctx)
        raw = chat(prompt, ctx.config, system=system)
        return self.post_process(raw, ctx)


def _truncate_transcript(text: str, max_chars: int = 60000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... 内容过长，中间部分已省略 ...]\n\n" + text[-half:]
