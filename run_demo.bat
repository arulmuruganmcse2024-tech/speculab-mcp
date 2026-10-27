@echo off
REM One-Click Streamlit Demo Launcher for SpecuLab-MCP (Windows)

setlocal

if exist venv\Scripts\python.exe (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ============================================================
echo Starting SpecuLab-MCP Interactive Active Vision Dashboard...
echo ============================================================
"%PYTHON_EXE%" -m streamlit run app.py %*
