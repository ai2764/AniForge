@echo off
REM Rename this folder from motion-portrait -> AniForge.
REM Close Grok/IDE terminals that have cwd inside this repo first.
cd /d "%~dp0.."
if not exist "motion-portrait" (
  echo Already renamed or path not found: motion-portrait
  exit /b 1
)
if exist "AniForge" (
  echo Target already exists: AniForge
  exit /b 1
)
ren "motion-portrait" "AniForge"
if errorlevel 1 (
  echo Rename failed — a process still holds the folder open.
  exit /b 1
)
echo OK: C:\Users\AIBOX\dev\AniForge
exit /b 0
