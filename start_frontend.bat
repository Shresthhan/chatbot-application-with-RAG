@echo off
REM Start Streamlit Frontend

echo Starting Streamlit Frontend...
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run frontend\app_api.py

pause
