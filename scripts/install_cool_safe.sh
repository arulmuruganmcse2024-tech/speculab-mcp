#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "=== Creating Safe Child Venv Preserving COOL OpenCV and NumPy ==="

BASE_PYTHON=$(ls -d /opt/cool/python_*/bin/python3 2>/dev/null | head -n 1 || which python3)
TARGET_VENV="$HOME/venv_cool_app"

echo "Base Python: $BASE_PYTHON"
echo "Target Venv: $TARGET_VENV"

# 1. Create --system-site-packages virtualenv
"$BASE_PYTHON" -m venv --system-site-packages "$TARGET_VENV"
source "$TARGET_VENV/bin/activate"

# 2. Record before state
python3 - <<EOF > logs/cool_venv_before.json
import json, cv2, numpy as np
state = {
    "cv2_version": cv2.__version__,
    "cv2_file": cv2.__file__,
    "numpy_version": np.__version__
}
print(json.dumps(state, indent=2))
EOF

# 3. Lock numpy version to avoid overwrite
INSTALLED_NP=$(python3 -c "import numpy; print(numpy.__version__)")
echo "numpy==$INSTALLED_NP" > /tmp/cool_constraints.txt

# 4. Install requirements.txt using constraints
pip install --no-cache-dir -r requirements.txt -c /tmp/cool_constraints.txt

# 5. Record after state and assert zero mutation of OpenCV & NumPy
python3 - <<EOF > logs/cool_venv_after.json
import json, cv2, numpy as np
state = {
    "cv2_version": cv2.__version__,
    "cv2_file": cv2.__file__,
    "numpy_version": np.__version__
}
print(json.dumps(state, indent=2))
EOF

python3 - <<EOF
import json
with open("logs/cool_venv_before.json") as f:
    before = json.load(f)
with open("logs/cool_venv_after.json") as f:
    after = json.load(f)

assert before["cv2_file"] == after["cv2_file"], "COOL OpenCV binary path was overwritten!"
assert before["cv2_version"] == after["cv2_version"], "OpenCV version was changed!"
assert before["numpy_version"] == after["numpy_version"], "NumPy version was changed!"
print("[SUCCESS] Verified zero change to pre-installed COOL OpenCV and NumPy binaries.")
EOF

# 6. Verify FastMCP import
python3 -c "from mcp.server.fastmcp import FastMCP; print('[SUCCESS] FastMCP imported successfully in venv.')"
