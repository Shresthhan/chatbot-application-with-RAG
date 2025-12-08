@echo off
REM Start FastAPI Backend Server

echo Starting FastAPI Backend...
cd /d "%~dp0"
call .venv\Scripts\activate
.venv\Scripts\python.exe -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

pause
