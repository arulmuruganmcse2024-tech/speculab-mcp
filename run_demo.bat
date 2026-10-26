@echo off
REM One-Click Streamlit Demo Launcher for SpecuLab-MCP (Windows)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Starting SpecuLab-MCP Streamlit Dashboard...
streamlit run app.py
