"""Automated virtual environment bootstrap helper for SpecuLab-MCP.

Ensures that scripts executed directly from the terminal or IDE (e.g. via 'py script.py'
or system Python) automatically detect and trampoline into the workspace virtual
environment (./venv) containing OpenCV 5.0.0, NumPy, FastMCP, and other required tools.
"""

import os
import subprocess
import sys


def ensure_venv() -> None:
    """Trampolines into the project's virtual environment if running outside it
    and core modules cannot be imported.
    """
    if os.environ.get("SPECULAB_VENV_ACTIVE") == "1":
        return

    # Check if essential packages are already importable in the current interpreter
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return  # Environment already has necessary packages
    except ImportError:
        pass

    # Find the repository root (directory containing 'config.yaml' or parent of 'src')
    cur = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(cur) if os.path.basename(cur) == "src" else cur
    if not os.path.isfile(os.path.join(repo_root, "config.yaml")):
        repo_root = os.getcwd()

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
            res = subprocess.run([candidate] + sys.argv, env=env)
            sys.exit(res.returncode)
