@echo off
setlocal
cd /d "%~dp0"
python -m ork2cfd %*
if errorlevel 1 (
    echo.
    echo ork2cfd exited with an error. Press any key to close.
    pause >nul
)
