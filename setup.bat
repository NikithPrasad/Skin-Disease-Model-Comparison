@echo off
rem One-time setup for the website. Needs Python 3.10 or newer and an internet connection.
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  echo Python is not installed. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation, then run this again.
  pause
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
rem CPU-only PyTorch: ~200 MB instead of ~2.5 GB for the GPU build. No GPU needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r app\requirements.txt
echo.
echo Setup complete. Double-click run.bat to start the website.
pause
