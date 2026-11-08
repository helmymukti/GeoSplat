@echo off
REM GeoSplat GPU setup: Graphdeco 3D Gaussian Splatting (cloned from the official repository) + PyTorch CUDA.
REM Needs: NVIDIA driver, CUDA Toolkit 11.8/12.x, Visual Studio 2022 Build Tools (C++), git, conda or Python 3.10-3.12.
cd /d "%~dp0.."
".venv\Scripts\python.exe" setup\install_gaussian_splatting.py
echo.
pause
