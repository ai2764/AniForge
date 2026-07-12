@echo off
REM Start the AniForge server on http://127.0.0.1:8500
REM Requires: a running ComfyUI at 127.0.0.1:8188 (Kimodo + SCAIL2 nodes) and ffmpeg on PATH.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Load machine-local config from .env (KEY=VALUE lines, # for comments), if present.
REM This is where COMFY_PYTHON (torch interpreter for Kimodo/matting) gets set.
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
)

REM Interpreter to launch the server. Override via SERVER_PYTHON in .env; else use PATH's python.
if not defined SERVER_PYTHON set "SERVER_PYTHON=python"

echo Starting AniForge on http://127.0.0.1:8500  (Ctrl+C to stop)
"%SERVER_PYTHON%" server\app.py --port 8500
pause
