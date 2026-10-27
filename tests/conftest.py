"""Pytest root configuration and virtual environment bootstrap for SpecuLab-MCP."""

import os
import subprocess
import sys

# Ensure workspace virtual environment is used if run with an unconfigured python
if os.environ.get("SPECULAB_VENV_ACTIVE") != "1":
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(repo_root, "venv", "Scripts", "python.exe"),
            os.path.join(repo_root, ".venv", "Scripts", "python.exe"),
            os.path.join(repo_root, "venv", "bin", "python"),
            os.path.join(repo_root, ".venv", "bin", "python"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate) and os.path.abspath(sys.executable).lower() != os.path.abspath(candidate).lower():
                env = dict(os.environ)
                env["SPECULAB_VENV_ACTIVE"] = "1"
                cmd = [candidate, "-m", "pytest"] + sys.argv[1:]
                res = subprocess.run(cmd, env=env)
                sys.exit(res.returncode)
