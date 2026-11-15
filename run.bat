@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py "%SCRIPT_DIR%bootstrap.py" %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        python "%SCRIPT_DIR%bootstrap.py" %*
    ) else (
        echo Python not found on PATH. Please install Python and try again.
        exit /b 1
    )
)
