@echo off
SETLOCAL ENABLEEXTENSIONS

SET ENV_NAME=detectron2-env
SET CONDA_BASE=C:\ProgramData\anaconda3

REM Check if conda is available
where conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Conda is not installed or not added to PATH. Exiting...
    pause
    exit /b
)

REM Activate Conda base
CALL "%CONDA_BASE%\Scripts\activate.bat"

REM Check if environment already exists
conda env list | findstr /i "%ENV_NAME%" >nul
IF %ERRORLEVEL% EQU 0 (
    echo ✅ Environment "%ENV_NAME%" already exists!
) ELSE (
    echo 🔧 Creating new environment: %ENV_NAME%
    conda create -y -n %ENV_NAME% python=3.10 pip
)

REM Activate the environment
CALL conda activate %ENV_NAME%

REM Go to the folder where this script is located (project root)
cd /d "%~dp0"

REM Install standard Python requirements
IF EXIST requirements.txt (
    echo 📦 Installing Python requirements from requirements.txt...
    pip install -r requirements.txt
) ELSE (
    echo ⚠️ requirements.txt not found, skipping package install.
)

REM Install local Detectron2 in editable mode
IF EXIST detectron2\setup.py (
    echo 🧠 Installing local Detectron2 from ./detectron2...
    pip install -e detectron2
) ELSE (
    echo ❌ detectron2 folder not found. Skipping Detectron2 installation.
)

echo ✅ Environment setup complete! You're ready to run the project!
pause
