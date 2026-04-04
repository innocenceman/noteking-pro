"""NoteKing CLI: command-line interface for video/recording to notes conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

ALL_TEMPLATES = [
    "brief", "detailed", "mindmap", "flashcard", "quiz", "timeline",
    "exam", "tutorial", "news", "podcast", "xhs_note", "latex_pdf", "custom",
    "meeting_minutes", "lecture_notes", "interview", "brainstorm",
    "news_digest", "exam_prep", "cornell_notes", "podcast_shownotes",
    "entertainment", "smart_summary",
]


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="2.0.0", prog_name="noteking")
def main(ctx):
    """NoteKing Pro - The ultimate video/recording to learning notes tool.

    Supports 30+ platforms, local video/audio files, meeting recordings,
    speaker diarization, noise reduction, and 23 output templates.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ────────────────────────────────────────────────────────────────
# Original commands
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("url")
@click.option("-t", "--template", default="detailed",
              help="Output template name")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--api-key", default=None, help="LLM API key")
@click.option("--base-url", default=None, help="LLM API base URL")
@click.option("--model", default=None, help="LLM model name")
@click.option("--proxy", default=None, help="Proxy URL")
@click.option("--custom-prompt", default="", help="Custom prompt for 'custom' template")
@click.option("--no-cache", is_flag=True, help="Disable result caching")
def run(url, template, output, api_key, base_url, model, proxy, custom_prompt, no_cache):
    """Process a video/blog URL and generate notes.

    Examples:

      noteking run "https://www.bilibili.com/video/BV1xx" -t detailed

      noteking run "https://youtu.be/xxx" -t mindmap

      noteking run "./lecture.mp4" -t exam
    """
    from core.config import AppConfig
    from core import summarize

    config = AppConfig.load()
    _apply_overrides(config, api_key, base_url, model, proxy)

    console.print(Panel(
        f"[bold cyan]NoteKing[/bold cyan] - Video to Notes\n"
        f"URL: {url}\nTemplate: {template}",
        title="Processing",
    ))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Fetching video info and subtitles...", total=None)
        try:
            result = summarize(url=url, template=template, config=config,
                             custom_prompt=custom_prompt, use_cache=not no_cache, output_dir=output)
            progress.update(task, description="Done!")
        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}[/red]")
            console.print(f"\n[red]Error:[/red] {e}")
            sys.exit(1)

    _print_result(result)


@main.command()
@click.argument("url")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--proxy", default=None, help="Proxy URL")
def transcript(url, output, proxy):
    """Extract only the transcript/subtitles from a video."""
    from core.config import AppConfig
    from core import get_transcript

    config = AppConfig.load()
    if proxy:
        config.proxy.enabled = True
        config.proxy.https = proxy

    text = get_transcript(url, config)
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        out_file = Path(output) / "transcript.txt"
        out_file.write_text(text, encoding="utf-8")
        console.print(f"[green]Saved to {out_file}[/green]")
    else:
        console.print(text)


# ────────────────────────────────────────────────────────────────
# New recording processing commands
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-t", "--template", default="meeting_minutes", help="Output template name")
@click.option("-c", "--context", default=None, help="Content description (e.g. 'AI开源项目圆桌讨论')")
@click.option("-s", "--scene", default=None,
              type=click.Choice(["meeting", "lecture", "interview", "brainstorm",
                                "news", "exam", "entertainment", "custom"]),
              help="Scene type")
@click.option("--speakers", default=None, type=int, help="Number of speakers (auto-detect if not set)")
@click.option("--denoise", default=1, type=click.IntRange(0, 3), help="Noise reduction level (0-3)")
@click.option("--format", "output_formats", default="markdown", help="Output formats, comma-separated (markdown,pdf,srt,vtt,transcript,json)")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--api-key", default=None, help="LLM API key")
@click.option("--base-url", default=None, help="LLM API base URL")
@click.option("--model", default=None, help="LLM model name")
def process(files, template, context, scene, speakers, denoise, output_formats, output, api_key, base_url, model):
    """Process local recording/video files and generate notes.

    Supports video (MP4, MKV, AVI, MOV) and audio (MP3, WAV, FLAC, M4A) files.
    Multiple files will be merged in order.

    Examples:

      noteking process meeting.mp4 -t meeting_minutes -c "产品周会"

      noteking process lecture.mp3 -t lecture_notes --denoise 2

      noteking process part1.mp4 part2.mp4 -t meeting_minutes --speakers 4

      noteking process interview.wav -t interview --format markdown,srt
    """
    from core.config import AppConfig
    from core import process_recording

    config = AppConfig.load()
    _apply_overrides(config, api_key, base_url, model, None)

    formats = [f.strip() for f in output_formats.split(",")]

    console.print(Panel(
        f"[bold cyan]NoteKing Pro[/bold cyan] - Recording Processor\n"
        f"Files: {', '.join(str(f) for f in files)}\n"
        f"Template: {template}\n"
        f"Scene: {scene or 'auto'}\n"
        f"Denoise: Level {denoise}\n"
        f"Speakers: {speakers or 'auto-detect'}\n"
        f"Formats: {', '.join(formats)}",
        title="Processing Recording",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=100)

        def _progress(step: str, pct: float):
            step_names = {
                "preprocess/validate": "Validating files...",
                "preprocess/merge": "Merging files...",
                "preprocess/extract_audio": "Extracting audio...",
                "preprocess/denoise": "Reducing noise...",
                "preprocess/done": "Preprocessing done",
                "preprocess": "Preprocessing...",
                "transcribe": "Transcribing audio...",
                "diarize": "Identifying speakers...",
                "generate": "Generating notes with AI...",
                "output": "Saving output files...",
                "done": "Complete!",
            }
            desc = step_names.get(step, step)
            progress.update(task, completed=int(pct * 100), description=desc)

        try:
            result = process_recording(
                input_files=list(files),
                template=template,
                context=context,
                scene=scene,
                num_speakers=speakers,
                denoise_level=denoise,
                output_formats=formats,
                config=config,
                output_dir=output,
                progress_callback=_progress,
            )
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    console.print()
    _print_recording_result(result)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--lang", default=None, help="Language code (auto-detect if not set)")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--api-key", default=None, help="LLM API key")
def transcribe(file, lang, output, api_key):
    """Transcribe a local audio/video file to text.

    Example:

      noteking transcribe meeting.mp4

      noteking transcribe lecture.mp3 --lang en
    """
    from core.config import AppConfig
    from core.preprocessor import preprocess
    from core.transcriber import transcribe as do_transcribe, detect_language

    config = AppConfig.load()
    if api_key:
        config.llm.api_key = api_key

    console.print(Panel(f"[cyan]Transcribing:[/cyan] {file}", title="ASR"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Extracting audio...", total=None)
        pre = preprocess([file], denoise_level=0)

        progress.update(task, description="Detecting language...")
        language = lang or detect_language(pre.audio_path, config)

        progress.update(task, description=f"Transcribing ({language})...")
        result = do_transcribe(pre.audio_path, config, language=language)
        progress.update(task, description="Done!")

    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{Path(file).stem}_transcript.txt"
        srt_path = out_dir / f"{Path(file).stem}.srt"
        result.save_txt(txt_path)
        result.save_srt(srt_path)
        console.print(f"[green]Saved:[/green] {txt_path}")
        console.print(f"[green]Saved:[/green] {srt_path}")
    else:
        console.print(result.full_text)

    console.print(f"\nLanguage: {language}")
    console.print(f"Segments: {len(result.segments)}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--level", default=2, type=click.IntRange(1, 3), help="Denoise level (1-3)")
@click.option("-o", "--output", default=None, help="Output file path")
def denoise(file, level, output):
    """Apply noise reduction to an audio file.

    Example:

      noteking denoise noisy_recording.wav --level 2
    """
    from core.preprocessor import extract_audio, denoise_audio

    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="noteking_dn_"))

    console.print(Panel(f"[cyan]Denoising:[/cyan] {file}\nLevel: {level}", title="Noise Reduction"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Extracting audio...", total=None)
        audio_path = work_dir / "audio.wav"
        extract_audio(Path(file), audio_path)

        progress.update(task, description=f"Applying noise reduction (level {level})...")
        out_path = Path(output) if output else Path(file).parent / f"{Path(file).stem}_denoised.wav"
        result = denoise_audio(audio_path, out_path, level=level)
        progress.update(task, description="Done!")

    console.print(f"[green]Saved:[/green] {result}")


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output file path")
def merge(files, output):
    """Merge multiple audio/video files into one.

    Example:

      noteking merge part1.mp4 part2.mp4 -o merged.mp4
    """
    from core.preprocessor import merge_files

    console.print(Panel(
        f"[cyan]Merging {len(files)} files[/cyan]",
        title="Merge",
    ))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Merging...", total=None)
        paths = [Path(f) for f in files]
        merged, segments = merge_files(paths, Path(output))
        progress.update(task, description="Done!")

    console.print(f"[green]Merged:[/green] {merged}")
    for seg in segments:
        console.print(f"  {seg.file_path} (offset: {seg.start_offset:.1f}s, duration: {seg.duration:.1f}s)")


# ────────────────────────────────────────────────────────────────
# Utility commands
# ────────────────────────────────────────────────────────────────

@main.command()
def templates():
    """List all available output templates."""
    from core.templates import TEMPLATE_LIST

    table = Table(title="Available Templates (23)")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Description")

    for t in TEMPLATE_LIST:
        table.add_row(t["name"], t["display_name"], t["description"])

    console.print(table)


@main.command()
@click.option("--api-key", prompt="LLM API Key", help="Your LLM API key")
@click.option("--base-url", default="", help="LLM API base URL (for MiniMax/DeepSeek/Ollama)")
@click.option("--model", default="gpt-4o-mini", help="LLM model name")
@click.option("--proxy", default="", help="Proxy URL for YouTube access")
def setup(api_key, base_url, model, proxy):
    """Interactive setup wizard for first-time configuration."""
    from core.config import AppConfig

    config = AppConfig.load()
    config.llm.api_key = api_key
    if base_url:
        config.llm.base_url = base_url
    config.llm.model = model
    if proxy:
        config.proxy.enabled = True
        if proxy.startswith("socks"):
            config.proxy.socks5 = proxy
        else:
            config.proxy.https = proxy
    config.save()

    console.print("[green]Configuration saved![/green]")
    console.print(f"Config: ~/.noteking/config.json")


@main.command()
def cache_clear():
    """Clear all cached results."""
    from core.config import AppConfig
    from core.cache import Cache

    config = AppConfig.load()
    c = Cache(config)
    count = c.clear()
    console.print(f"[green]Cleared {count} cached entries.[/green]")


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _apply_overrides(config, api_key, base_url, model, proxy):
    if api_key:
        config.llm.api_key = api_key
    if base_url:
        config.llm.base_url = base_url
    if model:
        config.llm.model = model
    if proxy:
        config.proxy.enabled = True
        if proxy.startswith("socks"):
            config.proxy.socks5 = proxy
        else:
            config.proxy.https = proxy
            config.proxy.http = proxy


def _print_result(result):
    console.print(f"\n[green]Success![/green]")
    console.print(f"Title: {result.get('title', 'N/A')}")
    console.print(f"Platform: {result.get('platform', 'N/A')}")
    console.print(f"Source: {result.get('source', 'N/A')}")
    console.print(f"Output: {result.get('output_file', 'N/A')}")


def _print_recording_result(result):
    console.print(Panel("[bold green]Processing Complete![/bold green]", title="Result"))
    console.print(f"Title: {result.get('title', 'N/A')}")
    console.print(f"Template: {result.get('template', 'N/A')}")
    console.print(f"Language: {result.get('language', 'N/A')}")
    console.print(f"Duration: {result.get('duration', 0):.0f} seconds")
    console.print(f"Speakers: {result.get('num_speakers', 'N/A')}")

    files = result.get("output_files", {})
    if files:
        console.print("\n[bold]Output Files:[/bold]")
        for fmt, path in files.items():
            console.print(f"  [{fmt}] {path}")


if __name__ == "__main__":
    main()
