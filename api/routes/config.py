"""Configuration API routes."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.schemas import ConfigUpdate
from core.config import AppConfig

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/")
async def get_config():
    """Get current configuration (sensitive fields masked)."""
    config = AppConfig.load()
    d = config.to_dict()
    if d.get("llm", {}).get("api_key"):
        d["llm"]["api_key"] = "***" + d["llm"]["api_key"][-4:]
    if d.get("asr", {}).get("groq_api_key"):
        d["asr"]["groq_api_key"] = "***" + d["asr"]["groq_api_key"][-4:]
    if d.get("bilibili_sessdata"):
        d["bilibili_sessdata"] = "***"
    return d


@router.put("/")
async def update_config(update: ConfigUpdate):
    """Update configuration."""
    config = AppConfig.load()

    if update.llm_api_key is not None:
        config.llm.api_key = update.llm_api_key
    if update.llm_base_url is not None:
        config.llm.base_url = update.llm_base_url
    if update.llm_model is not None:
        config.llm.model = update.llm_model
    if update.proxy_enabled is not None:
        config.proxy.enabled = update.proxy_enabled
    if update.proxy_url is not None:
        if update.proxy_url.startswith("socks"):
            config.proxy.socks5 = update.proxy_url
        else:
            config.proxy.https = update.proxy_url
            config.proxy.http = update.proxy_url
    if update.default_template is not None:
        config.default_template = update.default_template
    if update.bilibili_sessdata is not None:
        config.bilibili_sessdata = update.bilibili_sessdata

    config.save()
    return {"status": "ok", "message": "Configuration updated"}
