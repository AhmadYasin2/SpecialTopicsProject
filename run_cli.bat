@echo off
REM AutoPRISMA CLI Launcher for Windows
REM This script helps run systematic reviews from the command line

echo.
echo ========================================
echo  AutoPRISMA - Systematic Review CLI
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Virtual environment not detected
    if exist "venv\Scripts\activate.bat" (
        echo [INFO] Activating virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo [WARNING] No virtual environment found. Using system Python.
        echo [TIP] Create one with: python -m venv venv
    )
)

REM Check if Ollama is running
echo [INFO] Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not responding on http://localhost:11434
    echo [ACTION] Start Ollama with: ollama serve
    echo [ACTION] Then pull model with: ollama pull qwen2.5:32b
    echo.
    pause
    exit /b 1
)

echo [OK] Ollama is running
echo.

REM Run the CLI with all provided arguments
python cli.py %*

echo.
pause
