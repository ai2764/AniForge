@echo off
REM Rename this folder from motion-portrait -> AniForge (one-time helper).
REM Close IDEs/terminals with cwd inside this folder first.
cd /d "%~dp0.."
if not exist "motion-portrait" (
  echo Source folder motion-portrait not found (already renamed?)
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
echo OK: renamed to AniForge under %CD%
exit /b 0
