@echo off
REM Quick Start Script for AutoPRISMA
REM This script helps you set up and run AutoPRISMA quickly

echo ========================================
echo   AutoPRISMA Setup and Launch Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo [1/5] Python found!
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment already exists
)
echo.

REM Activate virtual environment and install dependencies
echo [3/5] Installing dependencies...
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some packages may have failed to install
)
echo.

REM Check if .env file exists
if not exist ".env" (
    echo [4/5] Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env and add your OpenAI API key
    echo Open .env in notepad: notepad .env
    echo.
    pause
) else (
    echo [4/5] .env file already exists
)
echo.

REM Create data directories
if not exist "data\" mkdir data
if not exist "data\vector_store\" mkdir data\vector_store
if not exist "data\documents\" mkdir data\documents
if not exist "data\state\" mkdir data\state

echo [5/5] Setup complete!
echo.
echo ========================================
echo   Choose how to run AutoPRISMA:
echo ========================================
echo.
echo 1. Streamlit UI (Recommended - Interactive Interface)
echo 2. FastAPI Backend Only (For API access)
echo 3. Command Line (Direct execution)
echo 4. Test Individual Agents
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting AutoPRISMA with Streamlit UI...
    echo ========================================
    echo.
    echo Opening two terminals:
    echo   - Terminal 1: FastAPI Backend (port 8000)
    echo   - Terminal 2: Streamlit UI (port 8501)
    echo.
    echo Access the UI at: http://localhost:8501
    echo API docs at: http://localhost:8000/docs
    echo.
    echo Press Ctrl+C in each terminal to stop
    echo.
    
    REM Start FastAPI in new window
    start "AutoPRISMA API" cmd /k ".venv\Scripts\activate.bat && python main.py"
    
    REM Wait a bit for API to start
    timeout /t 3 /nobreak >nul
    
    REM Start Streamlit in new window
    start "AutoPRISMA UI" cmd /k ".venv\Scripts\activate.bat && streamlit run app.py"
    
    echo.
    echo Services started! Check the new terminal windows.
    echo.
    pause
    
) else if "%choice%"=="2" (
    echo.
    echo Starting FastAPI Backend...
    echo Access API docs at: http://localhost:8000/docs
    echo.
    python main.py
    
) else if "%choice%"=="3" (
    echo.
    set /p query="Enter your research question: "
    echo.
    echo Running systematic review...
    python orchestrator.py --query "%query%" --databases semantic_scholar arxiv
    echo.
    pause
    
) else if "%choice%"=="4" (
    echo.
    echo Testing Individual Agents:
    echo ========================================
    echo.
    echo 1. Query Strategist
    echo 2. Literature Retrieval
    echo 3. Screening Criteria
    echo 4. Abstract Evaluator
    echo 5. Synthesis Analysis
    echo 6. Report Generator
    echo 7. All Agents
    echo.
    
    set /p agent_choice="Choose agent to test (1-7): "
    
    if "%agent_choice%"=="1" python agent_query_strategist.py
    if "%agent_choice%"=="2" python agent_literature_retrieval.py
    if "%agent_choice%"=="3" python agent_screening_criteria.py
    if "%agent_choice%"=="4" python agent_abstract_evaluator.py
    if "%agent_choice%"=="5" python agent_synthesis_analysis.py
    if "%agent_choice%"=="6" python agent_report_generator.py
    if "%agent_choice%"=="7" (
        python agent_query_strategist.py
        python agent_literature_retrieval.py
        python agent_screening_criteria.py
        python agent_abstract_evaluator.py
        python agent_synthesis_analysis.py
        python agent_report_generator.py
    )
    
    echo.
    pause
    
) else if "%choice%"=="5" (
    echo.
    echo Exiting...
    exit /b 0
    
) else (
    echo.
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

REM Deactivate virtual environment
call .venv\Scripts\deactivate.bat

echo.
echo Done!
pause
