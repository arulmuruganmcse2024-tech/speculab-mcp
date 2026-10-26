#!/usr/bin/env bash
# One-Click Streamlit Demo Launcher for SpecuLab-MCP (Linux / Graviton)
set -e

if [ -f ~/venv_cool_app/bin/activate ]; then
    source ~/venv_cool_app/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "Starting SpecuLab-MCP Streamlit Dashboard..."
streamlit run app.py
