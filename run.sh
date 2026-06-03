#!/usr/bin/env bash
set -e

# ── FinWolf OCR — quick launcher ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌  Python 3 not found. Install from https://python.org"
    exit 1
fi

# 2. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "📦  Creating virtual environment…"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install / upgrade deps
echo "📦  Installing dependencies…"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 4. Check .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env — add your ANTHROPIC_API_KEY before continuing."
    echo "    Open .env and replace: sk-ant-your-key-here"
    echo ""
    exit 1
fi

# 5. Launch
echo ""
echo "🐺  Launching FinWolf OCR…"
echo "    → http://localhost:8501"
echo ""
streamlit run app/ui/streamlit_app.py --server.port 8501
