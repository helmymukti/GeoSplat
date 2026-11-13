@echo off
setlocal
REM GeoSplat base setup: Python virtual environment (.venv) + Python packages + portable FFmpeg/COLMAP.
cd /d "%~dp0.."
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY ( py -%%V --version >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo [ERROR] Python 3.10-3.13 not found. Install it with: winget install Python.Python.3.12
  exit /b 1
)
echo [1/3] Python: & %PY% --version
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || exit /b 1
echo [2/3] Python packages (requirements.txt)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
echo [3/3] FFmpeg and COLMAP...
".venv\Scripts\python.exe" setup\bootstrap_tools.py
if errorlevel 1 echo [WARNING] FFmpeg/COLMAP incomplete - see the messages above.
echo.
echo Base setup complete. For Gaussian Splatting (GPU), click "Set up GPU..." in the app.
exit /b 0
