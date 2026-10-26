#!/usr/bin/env bash
set -euo pipefail

echo "=== Deploying SpecuLab-MCP Systemd Service ==="

APP_DIR=$(pwd)
SERVICE_NAME="speculab-mcp"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_PYTHON="${HOME}/venv_cool_app/bin/streamlit"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(which streamlit || true)"
fi

echo "App Directory: $APP_DIR"
echo "Streamlit Binary: $VENV_PYTHON"

# 1. Create CloudWatch Log Group once via AWS CLI (does not run inside application runtime)
if command -v aws &> /dev/null; then
    echo "Ensuring CloudWatch log group /aws/opencv2026/speculab-mcp exists..."
    aws logs create-log-group --log-group-name "/aws/opencv2026/speculab-mcp" --region "us-east-1" 2>/dev/null || true
fi

# 2. Write Systemd Unit File
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=SpecuLab-MCP Active Vision Metrology Dashboard
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${HOME}/venv_cool_app/bin:/usr/local/bin:/usr/bin"
Environment="SPECULAB_JUDGE_PASSCODE=${SPECULAB_JUDGE_PASSCODE:-}"
ExecStart=${VENV_PYTHON} run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

# 3. Reload and enable service
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
echo "Service ${SERVICE_NAME} deployed. Start with: sudo systemctl start ${SERVICE_NAME}"
