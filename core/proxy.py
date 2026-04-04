"""Proxy management for YouTube access from China."""

from __future__ import annotations

import subprocess
from .config import AppConfig


def test_youtube_access(config: AppConfig) -> bool:
    """Test if YouTube is directly accessible."""
    import httpx

    proxies = config.proxy.for_requests
    try:
        resp = httpx.get(
            "https://www.youtube.com/",
            timeout=10,
            follow_redirects=True,
            proxy=proxies.get("https") if proxies else None,
        )
        return resp.status_code == 200
    except Exception:
        return False


def test_proxy(proxy_url: str) -> bool:
    """Test if a proxy server is working."""
    import httpx

    try:
        resp = httpx.get(
            "https://www.google.com/",
            timeout=10,
            proxy=proxy_url,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_ytdlp_proxy_args(config: AppConfig) -> list[str]:
    """Get yt-dlp proxy command line arguments."""
    proxy = config.proxy.for_ytdlp
    if proxy:
        return ["--proxy", proxy]
    return []


def get_transcript_api_proxy(config: AppConfig):
    """Get proxy config for youtube-transcript-api."""
    proxy_dict = config.proxy.for_requests
    if not proxy_dict:
        return None

    try:
        from youtube_transcript_api.proxies import GenericProxyConfig
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http", "")
        if proxy_url:
            return GenericProxyConfig(proxy_url)
    except ImportError:
        pass
    return None
