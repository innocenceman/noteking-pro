"""Recording processing API routes: upload, process, and download."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Generator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse

from core.config import AppConfig

router = APIRouter(prefix="/api/v1/recording", tags=["recording"])

_tasks: dict[str, dict] = {}
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "noteking_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


def _load_config() -> AppConfig:
    cfg = AppConfig()
    if os.environ.get("NOTEKING_LLM_API_KEY"):
        cfg.llm.api_key = os.environ["NOTEKING_LLM_API_KEY"]
    if os.environ.get("NOTEKING_LLM_BASE_URL"):
        cfg.llm.base_url = os.environ["NOTEKING_LLM_BASE_URL"]
    if os.environ.get("NOTEKING_LLM_MODEL"):
        cfg.llm.model = os.environ["NOTEKING_LLM_MODEL"]
    if not cfg.llm.api_key:
        raise RuntimeError("No LLM API key configured.")
    return cfg


def _sse_event(stage: str, **kwargs) -> str:
    data = {"stage": stage, **kwargs}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a recording file for processing."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "upload").suffix
    save_path = _UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": save_path.stat().st_size,
        "path": str(save_path),
    }


@router.post("/process")
async def process_recording_endpoint(
    file_path: str = Form(None),
    file_id: str = Form(None),
    template: str = Form("meeting_minutes"),
    context: str = Form(None),
    scene: str = Form(None),
    num_speakers: int = Form(None),
    denoise_level: int = Form(1),
    output_formats: str = Form("markdown"),
):
    """Process an uploaded recording file."""
    if file_id:
        files = list(_UPLOAD_DIR.glob(f"{file_id}.*"))
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        actual_path = str(files[0])
    elif file_path:
        actual_path = file_path
    else:
        raise HTTPException(status_code=400, detail="Provide file_path or file_id")

    try:
        cfg = _load_config()
        from core import process_recording

        result = process_recording(
            input_files=[actual_path],
            template=template,
            context=context,
            scene=scene,
            num_speakers=num_speakers,
            denoise_level=denoise_level,
            output_formats=output_formats.split(","),
            config=cfg,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/stream")
async def process_recording_stream(
    file_path: str = Form(None),
    file_id: str = Form(None),
    template: str = Form("meeting_minutes"),
    context: str = Form(None),
    scene: str = Form(None),
    num_speakers: int = Form(None),
    denoise_level: int = Form(1),
    output_formats: str = Form("markdown"),
):
    """Process a recording with SSE streaming progress."""
    if file_id:
        files = list(_UPLOAD_DIR.glob(f"{file_id}.*"))
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        actual_path = str(files[0])
    elif file_path:
        actual_path = file_path
    else:
        raise HTTPException(status_code=400, detail="Provide file_path or file_id")

    def generate() -> Generator[str, None, None]:
        try:
            cfg = _load_config()
            from core import process_recording

            def _progress(step: str, pct: float):
                pass  # SSE events sent separately

            yield _sse_event("started", message="开始处理录音...")

            result = process_recording(
                input_files=[actual_path],
                template=template,
                context=context,
                scene=scene,
                num_speakers=num_speakers,
                denoise_level=denoise_level,
                output_formats=output_formats.split(","),
                config=cfg,
            )

            yield _sse_event("done",
                           title=result.get("title", ""),
                           content=result.get("content", ""),
                           duration=result.get("duration", 0),
                           num_speakers=result.get("num_speakers", 0),
                           output_files=result.get("output_files", {}))

        except Exception as e:
            yield _sse_event("error", message=str(e))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/scenes")
async def list_scenes():
    """List available scene types."""
    return [
        {"name": "meeting", "display": "会议", "description": "会议纪要、议题、决议、行动项"},
        {"name": "lecture", "display": "课堂", "description": "知识点、公式、习题、学习建议"},
        {"name": "interview", "display": "访谈", "description": "Q&A、观点、立场分析"},
        {"name": "brainstorm", "display": "灵感", "description": "想法、思维导图、行动建议"},
        {"name": "news", "display": "新闻", "description": "5W1H、引用、背景"},
        {"name": "exam", "display": "考试", "description": "闪卡、模拟题、要点清单"},
        {"name": "entertainment", "display": "娱乐", "description": "高光、金句、推荐指数"},
        {"name": "custom", "display": "自定义", "description": "自由定义输出格式"},
    ]
