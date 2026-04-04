#!/bin/bash
# NoteKing One-Click Install Script
# Usage: curl -sSL https://raw.githubusercontent.com/bcefghj/noteking/main/install.sh | bash

set -e

INSTALL_DIR="/opt/noteking"
REPO_URL="https://github.com/bcefghj/noteking.git"

echo "========================================"
echo "  NoteKing - One-Click Install"
echo "  Video/Blog to Learning Notes Tool"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash install.sh"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[1/4] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl start docker
    systemctl enable docker
    echo "Docker installed successfully."
else
    echo "[1/4] Docker already installed."
fi

# Install Docker Compose if not present
if ! command -v docker compose &> /dev/null; then
    echo "[2/4] Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin 2>/dev/null || true
fi
echo "[2/4] Docker Compose ready."

# Clone NoteKing
echo "[3/4] Downloading NoteKing..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    git pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Create .env if not exists
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    echo ""
    echo "[IMPORTANT] Please edit $INSTALL_DIR/.env and set your LLM API Key!"
    echo "Run: nano $INSTALL_DIR/.env"
fi

# Start services
echo "[4/4] Starting NoteKing..."
cd "$INSTALL_DIR"
docker compose up -d --build

echo ""
echo "========================================"
echo "  NoteKing installed successfully!"
echo ""
echo "  Web UI: http://$(hostname -I | awk '{print $1}'):3000"
echo "  API:    http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "  Config: $INSTALL_DIR/.env"
echo "  Logs:   cd $INSTALL_DIR && docker compose logs -f"
echo "========================================"
