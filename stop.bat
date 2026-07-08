@echo off
REM Stop the Motion Portrait server (whatever process is listening on port 8500).
setlocal
set PORT=8500
set FOUND=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"127.0.0.1:%PORT% .*LISTENING"') do (
    set FOUND=%%p
    echo Stopping Motion Portrait server on port %PORT% (PID %%p)...
    taskkill /F /PID %%p >nul 2>&1
)
if not defined FOUND echo No server is listening on port %PORT%.
endlocal
