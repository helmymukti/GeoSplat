@echo off
setlocal
REM GeoSplat - one-click launcher. First run prepares everything inside this folder.
REM Tip: drag a DJI .MP4 (and .SRT) onto this file to open it directly.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [GeoSplat] Persiapan pertama kali - mohon tunggu...
  call "%~dp0setup\install.bat"
  if errorlevel 1 ( echo. & echo [GeoSplat] Setup gagal - lihat pesan di atas. & pause & exit /b 1 )
)
start "" ".venv\Scripts\pythonw.exe" -m geosplat.gui %*
endlocal
