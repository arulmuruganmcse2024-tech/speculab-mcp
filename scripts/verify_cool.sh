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

# IMDSv2 EC2 Metadata Query (strictly validated against non-AWS HTML error responses)
INSTANCE_TYPE="LOCAL_DEV"
AMI_ID="LOCAL_DEV"
TOKEN=$(curl -s --connect-timeout 1 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)
if [ -n "$TOKEN" ] && [[ "$TOKEN" != *"<"* ]]; then
    RAW_INST=$(curl -s --connect-timeout 1 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)
    if [ -n "$RAW_INST" ] && [[ "$RAW_INST" != *"<"* ]] && [ ${#RAW_INST} -lt 32 ]; then
        INSTANCE_TYPE="$RAW_INST"
    fi
    RAW_AMI=$(curl -s --connect-timeout 1 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || true)
    if [ -n "$RAW_AMI" ] && [[ "$RAW_AMI" != *"<"* ]] && [ ${#RAW_AMI} -lt 32 ]; then
        AMI_ID="$RAW_AMI"
    fi
fi

ARCH=$(uname -m)

export INSTANCE_TYPE
export AMI_ID
export ARCH
export COOL_VENV

$PYTHON_CMD - <<'EOF' > logs/cool_verification.json
import json
import os
import platform
import sys

try:
    import cv2
    cv_ver = cv2.__version__
    cv_file = getattr(cv2, "__file__", "BUILTIN")
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
    if any(k in l_clean.lower() for k in ["kleidicv", "neon", "simd", "avx", "cpu", "dispatch"]):
        kleidi_flags.append(l_clean)

instance_type = os.environ.get("INSTANCE_TYPE", "LOCAL_DEV")
ami_id = os.environ.get("AMI_ID", "LOCAL_DEV")
arch = os.environ.get("ARCH", platform.machine())
cool_venv = os.environ.get("COOL_VENV", "")
cool_path = cool_venv if cool_venv else sys.prefix

report = {
    "instance_type": instance_type,
    "ami_id": ami_id,
    "architecture": arch,
    "machine": arch,
    "platform": f"Linux-{arch}",
    "cool_venv_path": cool_path,
    "cv2_version": cv_ver,
    "cv2_file": cv_file,
    "numpy_version": np_ver,
    "kleidicv_simd_flags": kleidi_flags[:10],
}

with open("logs/cool_verification.txt", "w", encoding="utf-8") as tf:
    tf.write(f"Instance Type: {report['instance_type']}\n")
    tf.write(f"AMI ID: {report['ami_id']}\n")
    tf.write(f"Architecture: {report['architecture']}\n")
    tf.write(f"OpenCV: {report['cv2_version']} ({report['cv2_file']})\n")
    tf.write(f"NumPy: {report['numpy_version']}\n")
    tf.write("Build flags:\n" + "\n".join(kleidi_flags[:10]) + "\n")

print(json.dumps(report, indent=2))
EOF

echo "Verification complete. Logs written to logs/cool_verification.json and logs/cool_verification.txt"
