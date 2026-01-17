#!/bin/bash
# AutoPRISMA CLI Launcher for Linux/Mac
# This script helps run systematic reviews from the command line

echo ""
echo "========================================"
echo " AutoPRISMA - Systematic Review CLI"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[INFO] Virtual environment not detected"
    if [ -f "venv/bin/activate" ]; then
        echo "[INFO] Activating virtual environment..."
        source venv/bin/activate
    else
        echo "[WARNING] No virtual environment found. Using system Python."
        echo "[TIP] Create one with: python -m venv venv"
    fi
fi

# Check if Ollama is running
echo "[INFO] Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[WARNING] Ollama is not responding on http://localhost:11434"
    echo "[ACTION] Start Ollama with: ollama serve"
    echo "[ACTION] Then pull model with: ollama pull qwen2.5:32b"
    echo ""
    exit 1
fi

echo "[OK] Ollama is running"
echo ""

# Run the CLI with all provided arguments
python cli.py "$@"

echo ""
