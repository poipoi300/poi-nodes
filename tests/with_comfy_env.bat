@echo off
REM Get the directory where this batch script is located.
SET "SCRIPT_DIR=%~dp0"

REM Navigate up three directories to the main ComfyUI folder.
cd /d "%SCRIPT_DIR%\..\..\.."

REM Check if the venv exists before trying to activate it.
IF NOT EXIST ".\venv\Scripts\activate.bat" (
    echo Error: Could not find the activation script.
    echo Make sure this script is placed correctly within your ComfyUI folder structure.
    pause
    exit /b
)

echo Activating ComfyUI Virtual Environment...
echo The new prompt will open in your 'poi-nodes' folder.

REM Corrected Path: Navigate up ONE level from the script's location to 'poi-nodes'.
cmd /k ".\venv\Scripts\activate.bat && cd /d "%SCRIPT_DIR%\..""
