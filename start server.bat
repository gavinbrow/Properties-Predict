@echo off
setlocal

cd /d "%~dp0"

echo Starting Properties Predict production server...
echo URL: http://127.0.0.1:8000
echo.

python app.py
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%
