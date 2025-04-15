@echo off
echo 🔥 Autonomous Drone Detection System
echo Activating Conda environment: detectron2-env

CALL C:\ProgramData\anaconda3\Scripts\activate.bat C:\Users\sharm\.conda\envs\detectron2-env

REM Go to the folder where this .bat file is located
cd /d "%~dp0"

REM Run your Python script (change it if your entry point is different)
python ADDS.py

pause
