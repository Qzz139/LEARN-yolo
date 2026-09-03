@echo off
setlocal

set "PREVIEW_DIR=%~dp0"
set "PROJECT_DIR=%PREVIEW_DIR%..\.."
set "PREVIEW_PYTHON="

if exist "%PREVIEW_DIR%.venv\Scripts\python.exe" set "PREVIEW_PYTHON=%PREVIEW_DIR%.venv\Scripts\python.exe"
if not defined PREVIEW_PYTHON if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" set "PREVIEW_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not defined PREVIEW_PYTHON (
  for /f "delims=" %%I in ('where python.exe 2^>nul') do if not defined PREVIEW_PYTHON set "PREVIEW_PYTHON=%%I"
)

if not defined PREVIEW_PYTHON (
  echo No Python installation was found.
  echo Create the application environment with:
  echo   cd /d "%PREVIEW_DIR%"
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%PREVIEW_PYTHON%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
  echo Python was found, but OpenCV and NumPy are missing:
  echo   "%PREVIEW_PYTHON%" -m pip install -r "%PREVIEW_DIR%requirements.txt"
  exit /b 1
)

"%PREVIEW_PYTHON%" "%PREVIEW_DIR%launcher.py" %*
exit /b %errorlevel%
