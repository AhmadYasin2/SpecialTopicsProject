# AutoPRISMA Setup Script for PowerShell
# Run this script to set up the system: .\setup.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AutoPRISMA Setup for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python not found!" -ForegroundColor Red
    Write-Host "      Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host ""
Write-Host "[2/5] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "      OK: Dependencies installed" -ForegroundColor Green

# Check Ollama
Write-Host ""
Write-Host "[3/5] Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaCheck = Get-Command ollama -ErrorAction Stop
    Write-Host "      OK: Ollama is installed" -ForegroundColor Green
    
    # Check if Ollama is running
    Write-Host ""
    Write-Host "[4/5] Checking Ollama server..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction Stop
        Write-Host "      OK: Ollama server is running" -ForegroundColor Green
    } catch {
        Write-Host "      WARNING: Ollama server not responding" -ForegroundColor Yellow
        Write-Host "      Starting Ollama server..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        Write-Host "      OK: Ollama server started" -ForegroundColor Green
    }
    
    # Pull model
    Write-Host ""
    Write-Host "[5/5] Checking Qwen 32B model..." -ForegroundColor Yellow
    $models = ollama list | Select-String "qwen2.5:32b"
    if ($models) {
        Write-Host "      OK: Model already downloaded" -ForegroundColor Green
    } else {
        Write-Host "      Downloading Qwen 32B model (~20GB, this will take a while)..." -ForegroundColor Yellow
        ollama pull qwen2.5:32b
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      OK: Model downloaded successfully" -ForegroundColor Green
        } else {
            Write-Host "      WARNING: Failed to download model" -ForegroundColor Yellow
            Write-Host "      You can download it manually later with: ollama pull qwen2.5:32b" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host "      WARNING: Ollama not found" -ForegroundColor Yellow
    Write-Host "" 
    Write-Host "      Please install Ollama from: https://ollama.ai" -ForegroundColor Cyan
    Write-Host "      Then run this script again." -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Success
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run reviews with:" -ForegroundColor Cyan
Write-Host '  python cli.py "Your research question"' -ForegroundColor White
Write-Host ""
Write-Host "Examples:" -ForegroundColor Cyan
Write-Host '  python cli.py "Effects of vitamin D on bone health"' -ForegroundColor White
Write-Host ""
Write-Host '  python cli.py "Machine learning in healthcare" --databases pubmed' -ForegroundColor White
Write-Host ""
Write-Host "For help:" -ForegroundColor Cyan
Write-Host "  python cli.py --help" -ForegroundColor White
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - CLI_USAGE.md - Complete usage guide" -ForegroundColor White
Write-Host "  - HEADLESS_SETUP.md - Server setup guide" -ForegroundColor White
Write-Host "  - CLI_QUICKSTART.md - Quick reference" -ForegroundColor White
Write-Host ""
