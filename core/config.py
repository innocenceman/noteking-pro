"""Global configuration management for NoteKing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path.home() / ".noteking"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_CACHE_DIR = DEFAULT_CONFIG_DIR / "cache"
DEFAULT_OUTPUT_DIR = Path.cwd() / "noteking_output"


@dataclass
class ProxyConfig:
    enabled: bool = False
    http: str = ""
    https: str = ""
    socks5: str = ""

    @property
    def for_requests(self) -> dict[str, str] | None:
        if not self.enabled:
            return None
        proxies: dict[str, str] = {}
        if self.http:
            proxies["http"] = self.http
        if self.https:
            proxies["https"] = self.https
        elif self.socks5:
            proxies["http"] = self.socks5
            proxies["https"] = self.socks5
        return proxies or None

    @property
    def for_ytdlp(self) -> str | None:
        if not self.enabled:
            return None
        return self.socks5 or self.https or self.http or None


@dataclass
class ASRConfig:
    default_engine: str = "auto"
    faster_whisper_model: str = "base"
    groq_api_key: str = ""
    openai_api_key: str = ""
    volcengine_app_id: str = ""
    volcengine_token: str = ""
    deepgram_api_key: str = ""


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 16000
    language: str = "zh-CN"


@dataclass
class RecordingConfig:
    """Configuration for recording/meeting processing."""
    denoise_level: int = 1  # 0=none, 1=light, 2=medium, 3=heavy
    num_speakers: int | None = None  # None=auto-detect
    enable_diarization: bool = True
    default_scene: str = "meeting"
    default_output_formats: list[str] = field(default_factory=lambda: ["markdown", "pdf"])
    hotwords: list[str] = field(default_factory=list)  # boost ASR accuracy
    context: str = ""  # user-provided description


@dataclass
class AppConfig:
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    bilibili_sessdata: str = ""
    default_template: str = "detailed"
    max_concurrent_downloads: int = 3

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or DEFAULT_CONFIG_FILE
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        try:
            data = json.loads(path.read_text())
            rec_data = data.get("recording", {})
            rec_data.pop("default_output_formats", None)  # avoid type issues
            return cls(
                proxy=ProxyConfig(**data.get("proxy", {})),
                asr=ASRConfig(**data.get("asr", {})),
                llm=LLMConfig(**data.get("llm", {})),
                recording=RecordingConfig(**{
                    k: v for k, v in rec_data.items()
                    if k in RecordingConfig.__dataclass_fields__
                }),
                cache_dir=data.get("cache_dir", str(DEFAULT_CACHE_DIR)),
                output_dir=data.get("output_dir", str(DEFAULT_OUTPUT_DIR)),
                bilibili_sessdata=data.get("bilibili_sessdata", ""),
                default_template=data.get("default_template", "detailed"),
                max_concurrent_downloads=data.get("max_concurrent_downloads", 3),
            )
        except Exception:
            return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
