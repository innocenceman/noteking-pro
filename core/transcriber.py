"""ASR engine pool: multiple transcription engines with factory pattern.

Supports:
- FunASR Paraformer-zh (best for Chinese, 12x faster, WER 8.4%)
- faster-whisper (best for multilingual/English)
- SenseVoice (50+ languages, 15x faster than Whisper)
- Groq Whisper API (cloud fallback)
- OpenAI Whisper API (cloud fallback)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .config import AppConfig
from .subtitle import SubtitleResult, SubtitleSegment

logger = logging.getLogger(__name__)


class ASREngine(ABC):
    """Abstract base for all ASR engines."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        ...

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        return True


class FunASREngine(ASREngine):
    """Alibaba FunASR Paraformer — best Chinese ASR (WER 8.4%, 12x faster).

    Supports Chinese, English, and code-switching natively.
    Built-in VAD, punctuation restoration, and timestamps.
    """

    name = "funasr"

    def __init__(self, model: str = "paraformer-zh"):
        self.model_name = model
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from funasr import AutoModel
            self._pipeline = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
            )
        return self._pipeline

    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        pipeline = self._get_pipeline()
        result = pipeline.generate(
            input=str(audio_path),
            batch_size_s=300,
        )

        segments = []
        if result and len(result) > 0:
            item = result[0]
            text = item.get("text", "")
            timestamps = item.get("timestamp", [])
            sentence_info = item.get("sentence_info", [])

            if sentence_info:
                for sent in sentence_info:
                    seg_text = sent.get("text", "")
                    start_ms = sent.get("start", 0)
                    end_ms = sent.get("end", 0)
                    segments.append(SubtitleSegment(
                        start=start_ms / 1000.0,
                        end=end_ms / 1000.0,
                        text=seg_text.strip(),
                    ))
            elif timestamps:
                for ts in timestamps:
                    if len(ts) >= 2:
                        start_ms = ts[0]
                        end_ms = ts[1]
                        segments.append(SubtitleSegment(
                            start=start_ms / 1000.0,
                            end=end_ms / 1000.0,
                            text="",
                        ))
                if text and not any(s.text for s in segments):
                    if segments:
                        segments[0] = SubtitleSegment(
                            start=segments[0].start,
                            end=segments[-1].end,
                            text=text.strip(),
                        )
                        segments = [segments[0]]
            elif text:
                segments.append(SubtitleSegment(start=0, end=0, text=text.strip()))

        return SubtitleResult(segments=segments, source="asr", language=language)

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        try:
            import funasr
            return True
        except ImportError:
            return False


class SenseVoiceEngine(ASREngine):
    """FunAudioLLM SenseVoice — 50+ languages, 15x faster than Whisper-Large."""

    name = "sensevoice"

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from funasr import AutoModel
            self._model = AutoModel(model="iic/SenseVoiceSmall")
        return self._model

    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        model = self._get_model()
        result = model.generate(
            input=str(audio_path),
            language="auto",
            use_itn=True,
        )

        segments = []
        if result and len(result) > 0:
            text = result[0].get("text", "")
            if text:
                segments.append(SubtitleSegment(start=0, end=0, text=text.strip()))

        return SubtitleResult(segments=segments, source="asr", language=language)

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        try:
            import funasr
            return True
        except ImportError:
            return False


class FasterWhisperEngine(ASREngine):
    """Local faster-whisper engine (free, GPU-accelerated)."""

    name = "faster_whisper"

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device="auto", compute_type="auto"
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        model = self._get_model()
        segs, info = model.transcribe(
            str(audio_path), language=language, vad_filter=True
        )
        segments = []
        for seg in segs:
            segments.append(SubtitleSegment(
                start=seg.start, end=seg.end, text=seg.text.strip()
            ))
        return SubtitleResult(
            segments=segments, source="asr", language=info.language
        )

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False


class GroqWhisperEngine(ASREngine):
    """Groq cloud Whisper API (free tier available)."""

    name = "groq"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        import httpx

        with open(audio_path, "rb") as f:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (audio_path.name, f, "audio/wav")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "language": language,
                },
                timeout=300,
            )
        resp.raise_for_status()
        data = resp.json()
        segments = []
        for seg in data.get("segments", []):
            segments.append(SubtitleSegment(
                start=seg["start"], end=seg["end"], text=seg["text"].strip()
            ))
        if not segments and data.get("text"):
            segments.append(SubtitleSegment(start=0, end=0, text=data["text"]))
        return SubtitleResult(segments=segments, source="asr", language=language)

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        return bool(config.asr.groq_api_key)


class OpenAIWhisperEngine(ASREngine):
    """OpenAI Whisper API."""

    name = "openai_whisper"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_path: Path, language: str = "zh") -> SubtitleResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                language=language,
            )

        segments = []
        for seg in getattr(transcript, "segments", []) or []:
            segments.append(SubtitleSegment(
                start=seg["start"], end=seg["end"], text=seg["text"].strip()
            ))
        if not segments and transcript.text:
            segments.append(SubtitleSegment(start=0, end=0, text=transcript.text))
        return SubtitleResult(segments=segments, source="asr", language=language)

    @classmethod
    def is_available(cls, config: AppConfig) -> bool:
        if config.asr.openai_api_key:
            return True
        # Only use LLM API key if it's an actual OpenAI endpoint
        if config.llm.api_key and (
            not config.llm.base_url
            or "openai" in config.llm.base_url.lower()
        ):
            return True
        return False


# ---------- language detection ----------

def detect_language(audio_path: Path, config: AppConfig) -> str:
    """Detect the primary language of an audio file.

    Strategy:
    1. Try faster-whisper's language detection (fast, ~1 second)
    2. Fall back to default 'zh'

    Returns ISO 639-1 code (e.g. 'zh', 'en', 'ja')
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="auto", compute_type="auto")
        _, info = model.transcribe(str(audio_path), language=None, vad_filter=True)
        detected = info.language
        prob = info.language_probability
        logger.info(f"Detected language: {detected} (confidence: {prob:.2f})")
        if prob > 0.5:
            return detected
    except (ImportError, Exception) as e:
        logger.debug(f"Language detection failed: {e}")

    return "zh"


def is_chinese_dominant(lang: str) -> bool:
    """Check if the detected language is primarily Chinese."""
    return lang in ("zh", "yue", "wuu", "nan", "hak")


# ---------- factory ----------

_ENGINES: list[type[ASREngine]] = [
    FunASREngine,
    FasterWhisperEngine,
    SenseVoiceEngine,
    GroqWhisperEngine,
    OpenAIWhisperEngine,
]


def _create_engine(config: AppConfig, language: str = "zh") -> ASREngine:
    """Create the best available ASR engine based on config and language."""
    pref = config.asr.default_engine

    # Explicit engine preference
    if pref == "funasr" and FunASREngine.is_available(config):
        return FunASREngine()
    if pref == "sensevoice" and SenseVoiceEngine.is_available(config):
        return SenseVoiceEngine()
    if pref == "faster_whisper" and FasterWhisperEngine.is_available(config):
        return FasterWhisperEngine(config.asr.faster_whisper_model)
    if pref == "groq" and GroqWhisperEngine.is_available(config):
        return GroqWhisperEngine(config.asr.groq_api_key)
    if pref == "openai" and OpenAIWhisperEngine.is_available(config):
        key = config.asr.openai_api_key or config.llm.api_key
        return OpenAIWhisperEngine(key)

    # Auto mode: select by language
    if pref == "auto":
        if is_chinese_dominant(language):
            if FunASREngine.is_available(config):
                return FunASREngine()
        if FasterWhisperEngine.is_available(config):
            return FasterWhisperEngine(config.asr.faster_whisper_model)
        if SenseVoiceEngine.is_available(config):
            return SenseVoiceEngine()

    # Cloud fallback
    if GroqWhisperEngine.is_available(config):
        return GroqWhisperEngine(config.asr.groq_api_key)
    if OpenAIWhisperEngine.is_available(config):
        key = config.asr.openai_api_key or config.llm.api_key
        return OpenAIWhisperEngine(key)
    if FasterWhisperEngine.is_available(config):
        return FasterWhisperEngine(config.asr.faster_whisper_model)

    raise RuntimeError(
        "No ASR engine available. Install funasr, faster-whisper, or provide "
        "an API key for Groq/OpenAI in the config."
    )


def transcribe(
    audio_path: Path,
    config: AppConfig,
    language: str | None = None,
) -> SubtitleResult:
    """Transcribe audio using the best available ASR engine.

    If language is None, auto-detects the language first.
    """
    if language is None:
        language = detect_language(audio_path, config)

    engine = _create_engine(config, language)
    logger.info(f"Using ASR engine: {engine.name} (language: {language})")
    return engine.transcribe(audio_path, language)
