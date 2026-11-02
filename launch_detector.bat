@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 scripts\detector_gui.py
) else (
    python scripts\detector_gui.py
)
if errorlevel 1 (
    echo.
    echo Please install Python 3.10 or newer with Tkinter and try again.
    pause
)
