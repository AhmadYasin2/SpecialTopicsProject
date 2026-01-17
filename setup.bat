@echo off
REM AutoPRISMA Setup Script for Windows
echo.
echo ========================================
echo  AutoPRISMA Setup
echo ========================================
echo.

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo [2/4] Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found in PATH
    echo [ACTION] Please install Ollama from: https://ollama.ai
    echo.
) else (
    echo [OK] Ollama is installed
    echo.
    
    echo [3/4] Starting Ollama server...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
    
    echo [4/4] Pulling Qwen 32B model (this may take a while, ~20GB)...
    ollama pull qwen2.5:32b
    if errorlevel 1 (
        echo [WARNING] Failed to pull model. You can do this manually later:
        echo           ollama pull qwen2.5:32b
    ) else (
        echo [OK] Model downloaded successfully
    )
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo You can now run reviews with:
echo   python cli.py "Your research question"
echo.
echo Or use the launcher:
echo   run_cli.bat "Your research question"
echo.
echo For help:
echo   python cli.py --help
echo.
pause
