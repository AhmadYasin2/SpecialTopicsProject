#!/bin/bash
# AutoPRISMA Setup Script for Linux/Mac
# Run with: bash setup.sh or ./setup.sh (after chmod +x setup.sh)

set -e  # Exit on error

echo ""
echo "========================================"
echo " AutoPRISMA Setup for Linux/Mac"
echo "========================================"
echo ""

# Check Python
echo "[1/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "      Found: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "      Found: $(python --version)"
else
    echo "      ERROR: Python not found!"
    echo "      Please install Python 3.10+ from python.org"
    exit 1
fi

# Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
echo "      OK: Dependencies installed"

# Check Ollama
echo ""
echo "[3/5] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "      OK: Ollama is installed"
    
    # Check if Ollama is running
    echo ""
    echo "[4/5] Checking Ollama server..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "      OK: Ollama server is running"
    else
        echo "      WARNING: Ollama server not responding"
        echo "      Starting Ollama server in background..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
        
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "      OK: Ollama server started"
        else
            echo "      ERROR: Could not start Ollama server"
            echo "      Please start it manually: ollama serve"
            exit 1
        fi
    fi
    
    # Pull model
    echo ""
    echo "[5/5] Checking Qwen 32B model..."
    if ollama list | grep -q "qwen2.5:32b"; then
        echo "      OK: Model already downloaded"
    else
        echo "      Downloading Qwen 32B model (~20GB, this will take a while)..."
        ollama pull qwen2.5:32b
        if [ $? -eq 0 ]; then
            echo "      OK: Model downloaded successfully"
        else
            echo "      WARNING: Failed to download model"
            echo "      You can download it manually later with: ollama pull qwen2.5:32b"
        fi
    fi
else
    echo "      WARNING: Ollama not found"
    echo ""
    echo "      Please install Ollama:"
    echo "      curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "      Then run this script again."
    exit 1
fi

# Make scripts executable
chmod +x run_cli.sh
chmod +x setup.sh

# Success
echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "You can now run reviews with:"
echo '  python cli.py "Your research question"'
echo ""
echo "Or use the launcher:"
echo '  ./run_cli.sh "Your research question"'
echo ""
echo "Examples:"
echo '  python cli.py "Effects of vitamin D on bone health"'
echo ""
echo '  python cli.py "Machine learning in healthcare" --databases pubmed'
echo ""
echo "For help:"
echo "  python cli.py --help"
echo ""
echo "Documentation:"
echo "  - CLI_USAGE.md - Complete usage guide"
echo "  - HEADLESS_SETUP.md - Server setup guide"
echo "  - CLI_QUICKSTART.md - Quick reference"
echo ""
