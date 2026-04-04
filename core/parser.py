"""Link parser: platform detection, collection detection, short-link resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, parse_qs


class Platform(str, Enum):
    BILIBILI = "bilibili"
    YOUTUBE = "youtube"
    DOUYIN = "douyin"
    XIAOHONGSHU = "xiaohongshu"
    KUAISHOU = "kuaishou"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    TWITCH = "twitch"
    VIMEO = "vimeo"
    FACEBOOK = "facebook"
    REDDIT = "reddit"
    PODCAST = "podcast"
    LOCAL = "local"
    UNKNOWN = "unknown"


class LinkType(str, Enum):
    SINGLE = "single"
    PLAYLIST = "playlist"
    SERIES = "series"
    COLLECTION = "collection"
    FAVORITES = "favorites"
    CHANNEL = "channel"
    MULTI_PART = "multi_part"
    LOCAL_FILE = "local_file"


@dataclass
class ParsedLink:
    url: str
    platform: Platform
    link_type: LinkType
    video_id: str = ""
    playlist_id: str = ""
    title: str = ""
    extra: dict = field(default_factory=dict)


# ---------- pattern tables ----------

_BILIBILI_PATTERNS = [
    (r"bilibili\.com/video/(BV[\w]+)", LinkType.SINGLE),
    (r"bilibili\.com/video/av(\d+)", LinkType.SINGLE),
    (r"b23\.tv/([\w]+)", LinkType.SINGLE),
    (r"bilibili\.com/list/(\d+)", LinkType.PLAYLIST),
    (r"space\.bilibili\.com/\d+/channel/seriesdetail\?sid=(\d+)", LinkType.SERIES),
    (r"space\.bilibili\.com/\d+/favlist\?fid=(\d+)", LinkType.FAVORITES),
    (r"bilibili\.com/watchlater", LinkType.COLLECTION),
]

_YOUTUBE_PATTERNS = [
    (r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})", LinkType.SINGLE),
    (r"youtube\.com/playlist\?list=([\w-]+)", LinkType.PLAYLIST),
    (r"youtube\.com/(?:c/|channel/|@)([\w-]+)", LinkType.CHANNEL),
]


def _detect_bilibili_multipart(url: str) -> bool:
    """Check if a bilibili URL has ?p= indicating multi-part video."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return "p" in qs


def parse_link(url: str) -> ParsedLink:
    """Parse a URL or local file path into a structured ParsedLink."""
    import os

    stripped = url.strip()

    if os.path.exists(stripped):
        return ParsedLink(
            url=stripped,
            platform=Platform.LOCAL,
            link_type=LinkType.LOCAL_FILE,
            video_id=os.path.basename(stripped),
        )

    for pattern, link_type in _BILIBILI_PATTERNS:
        m = re.search(pattern, stripped)
        if m:
            vid = m.group(1)
            lt = link_type
            if lt == LinkType.SINGLE and _detect_bilibili_multipart(stripped):
                lt = LinkType.MULTI_PART
            return ParsedLink(
                url=stripped,
                platform=Platform.BILIBILI,
                link_type=lt,
                video_id=vid,
            )

    for pattern, link_type in _YOUTUBE_PATTERNS:
        m = re.search(pattern, stripped)
        if m:
            vid = m.group(1)
            extra = {}
            if link_type == LinkType.SINGLE:
                parsed = urlparse(stripped)
                qs = parse_qs(parsed.query)
                if "list" in qs:
                    extra["playlist_id"] = qs["list"][0]
            return ParsedLink(
                url=stripped,
                platform=Platform.YOUTUBE,
                link_type=link_type,
                video_id=vid,
                playlist_id=extra.get("playlist_id", ""),
                extra=extra,
            )

    platform = _guess_platform(stripped)
    return ParsedLink(
        url=stripped,
        platform=platform,
        link_type=LinkType.SINGLE,
    )


def _guess_platform(url: str) -> Platform:
    host = urlparse(url).hostname or ""
    mapping = {
        "douyin.com": Platform.DOUYIN,
        "iesdouyin.com": Platform.DOUYIN,
        "xiaohongshu.com": Platform.XIAOHONGSHU,
        "xhslink.com": Platform.XIAOHONGSHU,
        "kuaishou.com": Platform.KUAISHOU,
        "tiktok.com": Platform.TIKTOK,
        "twitter.com": Platform.TWITTER,
        "x.com": Platform.TWITTER,
        "instagram.com": Platform.INSTAGRAM,
        "twitch.tv": Platform.TWITCH,
        "vimeo.com": Platform.VIMEO,
        "facebook.com": Platform.FACEBOOK,
        "reddit.com": Platform.REDDIT,
    }
    for domain, plat in mapping.items():
        if domain in host:
            return plat
    return Platform.UNKNOWN


def is_batch(parsed: ParsedLink) -> bool:
    return parsed.link_type in (
        LinkType.PLAYLIST,
        LinkType.SERIES,
        LinkType.COLLECTION,
        LinkType.FAVORITES,
        LinkType.CHANNEL,
        LinkType.MULTI_PART,
    )
