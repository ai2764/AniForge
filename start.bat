@echo off
REM Start the Motion Portrait server on http://127.0.0.1:8500
REM Requires: a running ComfyUI at 127.0.0.1:8188 (Kimodo + SCAIL2 nodes) and ffmpeg on PATH.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo Starting Motion Portrait on http://127.0.0.1:8500  (Ctrl+C to stop)
python server\app.py --port 8500
pause
