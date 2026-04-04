"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    url: str = Field(..., description="Video URL or local file path")
    template: str = Field("detailed", description="Output template name")
    custom_prompt: str = Field("", description="Custom prompt for 'custom' template")
    use_cache: bool = Field(True, description="Use cached results if available")


class BatchRequest(BaseModel):
    url: str = Field(..., description="Playlist/collection URL")
    template: str = Field("detailed", description="Output template name")
    merge: bool = Field(True, description="Merge all results into one document")


class ConfigUpdate(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    proxy_enabled: bool | None = None
    proxy_url: str | None = None
    default_template: str | None = None
    bilibili_sessdata: str | None = None


class VideoResponse(BaseModel):
    title: str
    content: str
    template: str
    source: str
    platform: str
    url: str
    duration: float = 0
    output_file: str = ""


class BatchResponse(BaseModel):
    title: str
    content: str
    template: str
    total: int
    completed: int
    failed: list[str] = []
    output_file: str = ""


class TemplateInfo(BaseModel):
    name: str
    display_name: str
    description: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
