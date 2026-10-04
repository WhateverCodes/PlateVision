@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\venv\Scripts\python.exe" (
    "..\..\work\venv\Scripts\python.exe" train_models.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" train_models.py
) else (
    echo Python setup is needed first. Open README.md for installation steps.
)
echo.
pause
endlocal
