@echo off
REM Start Legacy Streamlit App (without API)

echo Starting Legacy Streamlit App...
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app\app.py

pause
