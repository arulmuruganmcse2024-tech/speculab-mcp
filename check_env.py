"""Environment validation and pre-flight check script for SpecuLab-MCP."""

import os
import subprocess
import sys


def _bootstrap_venv():
    """If running outside the project virtual environment and core modules cannot be imported,
    automatically trampoline into the local venv Python if available.
    """
    if os.environ.get("SPECULAB_VENV_ACTIVE") == "1":
        return

    # Quick check if core modules are already available
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return
    except ImportError:
        pass

    # Look for workspace virtual environment
    repo_root = os.path.dirname(os.path.abspath(__file__))
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


_bootstrap_venv()


def main() -> int:
    import argparse
    import json
    import platform

    parser = argparse.ArgumentParser(description="SpecuLab-MCP Environment Diagnostic & Pre-flight Check")
    parser.add_argument("--write-verification-json", action="store_true", help="Write logs/cool_verification.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress banner and non-essential output")
    args, _ = parser.parse_known_args()

    if not args.quiet:
        print("=" * 60)
        print("SpecuLab-MCP Environment Diagnostic & Pre-flight Check")
        print("=" * 60)

    failures = []

    # 1. Python version
    py_ver = sys.version.split()[0]
    if not args.quiet:
        print(f"[OK] Python version: {py_ver} ({sys.executable})")

    # 2. NumPy
    np_module = None
    try:
        import numpy as np
        np_module = np
        if not args.quiet:
            print(f"[OK] NumPy version: {np.__version__}")
    except Exception as e:
        print(f"[FAIL] NumPy not found or failed to load: {e}")
        failures.append("numpy")

    # 3. OpenCV
    cv2_module = None
    try:
        import cv2
        cv2_module = cv2
        if not args.quiet:
            print(f"[OK] OpenCV version: {cv2.__version__} (Threads: {cv2.getNumThreads()}, Optimized: {cv2.useOptimized()})")
    except Exception as e:
        print(f"[FAIL] OpenCV not found or failed to load: {e}")
        failures.append("opencv")

    # 4. PyYAML
    try:
        import yaml
        if not args.quiet:
            print(f"[OK] PyYAML version: {yaml.__version__}")
    except Exception as e:
        print(f"[FAIL] PyYAML not found: {e}")
        failures.append("pyyaml")

    # 5. FastMCP
    try:
        from mcp.server.fastmcp import FastMCP
        if not args.quiet:
            print("[OK] FastMCP imported successfully from mcp.server.fastmcp")
    except Exception as e:
        print(f"[FAIL] FastMCP import failed: {e}")
        failures.append("fastmcp")

    # 6. Streamlit (Fast version check with fallback)
    try:
        import importlib.metadata
        st_ver = importlib.metadata.version("streamlit")
        if not args.quiet:
            print(f"[OK] Streamlit version: {st_ver}")
    except Exception:
        try:
            import streamlit as st
            if not args.quiet:
                print(f"[OK] Streamlit version: {st.__version__}")
        except Exception as e:
            print(f"[FAIL] Streamlit not found: {e}")
            failures.append("streamlit")

    # 7. Boto3 & Bedrock
    try:
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        import boto3
        from botocore.config import Config
        if not args.quiet:
            print(f"[OK] Boto3 version: {boto3.__version__}")
        boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=Config(connect_timeout=1.0, read_timeout=1.0),
        )
        if not args.quiet:
            print("[INFO] Boto3 Bedrock client instantiated.")
    except Exception as e:
        if not args.quiet:
            print(f"[WARN] Boto3 Bedrock client notice: {e}")

    # 8. Check COOL / Graviton path
    if not args.quiet:
        if os.path.exists("/opt/cool"):
            print("[INFO] AWS COOL optimized OpenCV library detected at /opt/cool")
        else:
            print("[INFO] Standard environment (no /opt/cool directory detected)")

        print("=" * 60)

    if failures:
        print(f"[ERROR] Environment check failed: missing {len(failures)} required component(s): {', '.join(failures)}")
        print("[HINT] Activate the project virtual environment before running:")
        print("       PowerShell:  .\\venv\\Scripts\\Activate.ps1")
        print("       CMD:         .\\venv\\Scripts\\activate.bat")
        print("       Bash:        source venv/bin/activate")
        print("       Or run with: .\\venv\\Scripts\\python.exe check_env.py")
        print("=" * 60)
        return 1

    if not args.quiet:
        print("All required core components verified.")
        print("=" * 60)

    if args.write_verification_json and cv2_module and np_module:
        os.makedirs("logs", exist_ok=True)
        arch = platform.machine()
        cv_ver = cv2_module.__version__
        cv_file = getattr(cv2_module, "__file__", "BUILTIN")
        build_info = cv2_module.getBuildInformation()
        kleidi_flags = []
        for line in build_info.splitlines():
            l_clean = line.strip()
            if any(k in l_clean.lower() for k in ["kleidicv", "neon", "simd", "avx", "cpu", "dispatch"]):
                kleidi_flags.append(l_clean)

        cool_path = "/opt/cool" if os.path.exists("/opt/cool") else sys.prefix
        rep = {
            "instance_type": "LOCAL_DEV",
            "ami_id": "LOCAL_DEV",
            "architecture": arch,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cool_venv_path": cool_path,
            "cv2_version": cv_ver,
            "cv2_file": cv_file,
            "numpy_version": np_module.__version__,
            "kleidicv_simd_flags": kleidi_flags[:10],
        }
        with open("logs/cool_verification.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        with open("logs/cool_verification.txt", "w", encoding="utf-8") as tf:
            tf.write(f"Instance Type: {rep['instance_type']}\n")
            tf.write(f"AMI ID: {rep['ami_id']}\n")
            tf.write(f"Architecture: {rep['architecture']}\n")
            tf.write(f"Platform: {rep['platform']}\n")
            tf.write(f"Machine: {rep['machine']}\n")
            tf.write(f"OpenCV: {rep['cv2_version']} ({rep['cv2_file']})\n")
            tf.write(f"NumPy: {rep['numpy_version']}\n")
            tf.write("Build flags:\n" + "\n".join(kleidi_flags[:10]) + "\n")
        print("[OK] Written logs/cool_verification.json and logs/cool_verification.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
