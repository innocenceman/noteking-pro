# NoteKing Pro Desktop

Cross-platform desktop app built with Tauri 2.0.

## Prerequisites

- [Rust](https://rustup.rs/) (for Tauri backend)
- [Node.js 18+](https://nodejs.org/)
- [pnpm](https://pnpm.io/)
- [FFmpeg](https://ffmpeg.org/)
- [Python 3.11+](https://www.python.org/)

## Setup

```bash
# Install Tauri CLI
cargo install tauri-cli

# Install frontend dependencies
cd ../web && pnpm install && cd ../desktop

# Run in development mode
cargo tauri dev

# Build for production
cargo tauri build
```

## Architecture

The desktop app reuses the same Web frontend (Next.js) and embeds the Python
backend via a sidecar process. Tauri provides the native window shell with
minimal resource usage (~10MB vs Electron's 200MB).

### Features
- Local ASR (faster-whisper / FunASR) — no cloud dependency for transcription
- Speaker diarization (pyannote-audio)
- Noise reduction (noisereduce)
- User provides their own LLM API key (MiniMax/OpenAI/DeepSeek)
- File drag & drop processing
- All data stays local

### Key Files

```
desktop/
├── src-tauri/
│   ├── Cargo.toml        # Rust project config
│   ├── tauri.conf.json   # Tauri configuration
│   └── src/
│       └── main.rs       # Rust entry point
└── README.md
```

## Building Installers

```bash
# macOS (.dmg)
cargo tauri build --target universal-apple-darwin

# Windows (.msi)
cargo tauri build --target x86_64-pc-windows-msvc

# Linux (.deb, .AppImage)
cargo tauri build --target x86_64-unknown-linux-gnu
```

The desktop app bundles `yt-dlp` and `ffmpeg` as sidecar binaries
and runs the Python API server as a background process.
