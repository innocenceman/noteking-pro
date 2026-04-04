"""Cache management to avoid reprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import AppConfig


class Cache:
    def __init__(self, config: AppConfig):
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str, template: str = "") -> str:
        raw = f"{url}:{template}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, url: str, template: str = "") -> Path:
        return self.cache_dir / f"{self._key(url, template)}.json"

    def get(self, url: str, template: str = "") -> dict[str, Any] | None:
        p = self._path(url, template)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, url: str, template: str, data: dict[str, Any]) -> None:
        p = self._path(url, template)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def has(self, url: str, template: str = "") -> bool:
        return self._path(url, template).exists()

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count

    def get_transcript(self, url: str) -> str | None:
        data = self.get(url, "__transcript__")
        if data:
            return data.get("transcript")
        return None

    def set_transcript(self, url: str, transcript: str) -> None:
        self.set(url, "__transcript__", {"transcript": transcript})
