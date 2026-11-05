@echo off
title HridyaVayu Live Platform
echo ==========================================================
echo    Starting HridyaVayu: Live Connected Cloud Platform
echo ==========================================================
cd /d "%~dp0"

set PYTHON_EXE=python
if exist ".\venv\Scripts\python.exe" (
    set PYTHON_EXE=".\venv\Scripts\python.exe"
) else if exist "C:\Users\HP\HridyaVayu\venv\Scripts\python.exe" (
    set PYTHON_EXE="C:\Users\HP\HridyaVayu\venv\Scripts\python.exe"
)

echo [1/2] Starting Flask Backend in background using %PYTHON_EXE%...
start "HridyaVayu Backend" /B %PYTHON_EXE% app.py

echo Waiting for Flask to initialize on port 7860...
timeout /t 4 /nobreak >nul

if exist ".\cloudflared_win.exe" (
    echo [2/2] Launching Cloudflare Public HTTPS Tunnel...
    ".\cloudflared_win.exe" tunnel --url http://127.0.0.1:7860
) else (
    echo [2/2] Opening Local Application at http://127.0.0.1:7860...
    start http://127.0.0.1:7860
)
pause
