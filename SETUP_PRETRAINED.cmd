@echo off
setlocal
cd /d "%~dp0"
set "PV_PYTHON=.venv\Scripts\python.exe"
if exist "..\..\work\venv\Scripts\python.exe" set "PV_PYTHON=..\..\work\venv\Scripts\python.exe"
if not exist "%PV_PYTHON%" (
 echo Set up Python using README.md first.
 pause
 exit /b 1
)
"%PV_PYTHON%" -m pip install onnxruntime==1.23.2 PyYAML==6.0.2 --timeout 20 --retries 1 --disable-pip-version-check
if errorlevel 1 goto failed
"%PV_PYTHON%" -m training.download_pretrained
if errorlevel 1 goto failed
"%PV_PYTHON%" -m training.download_whole_line
if errorlevel 1 goto failed
echo Ready. Open OPEN_PLATEVISION.cmd and select Compare models.
pause
exit /b 0
:failed
echo Setup did not finish. Keep this error message and ask for help.
pause
exit /b 1
