@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\venv\Scripts\python.exe" (
    "..\..\work\venv\Scripts\python.exe" train_reader.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" train_reader.py
) else (
    echo Python setup is needed first. Open README.md.
    pause
    exit /b 1
)
if errorlevel 1 pause
endlocal
