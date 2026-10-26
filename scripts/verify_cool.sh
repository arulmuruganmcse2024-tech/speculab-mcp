#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "=== Verifying AWS COOL & OpenCV Environment ==="

# Auto-detect COOL virtualenv
COOL_VENV=$(ls -d /opt/cool/python_*/bin/activate 2>/dev/null | head -n 1 || true)
if [ -n "$COOL_VENV" ] && [ -f "$COOL_VENV" ]; then
    echo "Sourcing COOL virtual environment: $COOL_VENV"
    source "$COOL_VENV"
    PYTHON_CMD="python3"
else
    echo "No /opt/cool activate script found; using current Python"
    PYTHON_CMD="python3"
fi

# IMDSv2 EC2 Metadata Query
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)
if [ -n "$TOKEN" ]; then
    INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "LOCAL_DEV")
    AMI_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || echo "LOCAL_DEV")
else
    INSTANCE_TYPE="LOCAL_DEV"
    AMI_ID="LOCAL_DEV"
fi

ARCH=$(uname -m)

$PYTHON_CMD - <<EOF > logs/cool_verification.json
import json
import os
import sys

try:
    import cv2
    cv_ver = cv2.__version__
    cv_file = cv2.__file__
    build_info = cv2.getBuildInformation()
except ImportError:
    cv_ver = "NOT_INSTALLED"
    cv_file = "NONE"
    build_info = ""

try:
    import numpy as np
    np_ver = np.__version__
except ImportError:
    np_ver = "NOT_INSTALLED"

kleidi_flags = []
for line in build_info.splitlines():
    l_clean = line.strip()
    if any(k in l_clean.lower() for k in ["kleidicv", "neon", "simd"]):
        kleidi_flags.append(l_clean)

report = {
    "instance_type": "$INSTANCE_TYPE",
    "ami_id": "$AMI_ID",
    "architecture": "$ARCH",
    "cool_venv_path": "$COOL_VENV" if "$COOL_VENV" else "NONE",
    "cv2_version": cv_ver,
    "cv2_file": cv_file,
    "numpy_version": np_ver,
    "kleidicv_simd_flags": kleidi_flags,
}

with open("logs/cool_verification.txt", "w") as tf:
    tf.write(f"Instance Type: {report['instance_type']}\n")
    tf.write(f"AMI ID: {report['ami_id']}\n")
    tf.write(f"Architecture: {report['architecture']}\n")
    tf.write(f"OpenCV: {report['cv2_version']} ({report['cv2_file']})\n")
    tf.write(f"NumPy: {report['numpy_version']}\n")
    tf.write("Build flags:\n" + "\n".join(kleidi_flags) + "\n")

print(json.dumps(report, indent=2))
EOF

echo "Verification complete. Logs written to logs/cool_verification.json and logs/cool_verification.txt"
