@echo off
setlocal enabledelayedexpansion

echo Setting up TALLMesh application...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.7 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed. Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Clone the repository
if not exist "TALLMesh" (
    echo Cloning TALLMesh repository...
    git clone https://github.com/yourusername/TALLMesh.git
) else (
    echo TALLMesh directory already exists. Updating...
    cd TALLMesh
    git pull
    cd ..
)

REM Create and activate virtual environment
if not exist "TALLMesh\venv" (
    echo Creating virtual environment...
    python -m venv TALLMesh\venv
)

echo Activating virtual environment...
call TALLMesh\venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r TALLMesh\requirements.txt

REM Run the Streamlit app
echo Starting TALLMesh application...
streamlit run TALLMesh\TALLMesh.py

REM Deactivate virtual environment
deactivate

echo TALLMesh application has been closed.
pause