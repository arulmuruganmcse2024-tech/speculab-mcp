"""Environment validation and pre-flight check script for SpecuLab-MCP."""

import os
import sys

def main() -> int:
    print("=" * 60)
    print("SpecuLab-MCP Environment Diagnostic & Pre-flight Check")
    print("=" * 60)

    # 1. Python version
    py_ver = sys.version.split()[0]
    print(f"[OK] Python version: {py_ver} ({sys.executable})")

    # 2. NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy not found: {e}")
        return 1

    # 3. OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__} (Threads: {cv2.getNumThreads()}, Optimized: {cv2.useOptimized()})")
    except ImportError as e:
        print(f"[FAIL] OpenCV not found: {e}")
        return 1

    # 4. PyYAML
    try:
        import yaml
        print(f"[OK] PyYAML version: {yaml.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyYAML not found: {e}")
        return 1

    # 5. FastMCP
    try:
        from mcp.server.fastmcp import FastMCP
        print("[OK] FastMCP imported successfully from mcp.server.fastmcp")
    except ImportError as e:
        print(f"[FAIL] FastMCP import failed: {e}")
        return 1

    # 6. Streamlit
    try:
        import streamlit as st
        print(f"[OK] Streamlit version: {st.__version__}")
    except ImportError as e:
        print(f"[FAIL] Streamlit not found: {e}")
        return 1

    # 7. Boto3 & Bedrock
    try:
        import boto3
        from botocore.config import Config
        print(f"[OK] Boto3 version: {boto3.__version__}")
        client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(connect_timeout=1.0, read_timeout=1.0))
        print("[INFO] Boto3 Bedrock client instantiated.")
    except Exception as e:
        print(f"[WARN] Boto3 Bedrock client notice: {e}")

    # 8. Check COOL / Graviton path
    if os.path.exists("/opt/cool"):
        print("[INFO] AWS COOL optimized OpenCV library detected at /opt/cool")
    else:
        print("[INFO] Standard environment (no /opt/cool directory detected)")

    print("=" * 60)
    print("All required core components verified.")
    print("=" * 60)

    import argparse
    import json
    import platform

    parser = argparse.ArgumentParser(description="Check environment")
    parser.add_argument("--write-verification-json", action="store_true", help="Write logs/cool_verification.json")
    args, _ = parser.parse_known_args()

    if args.write_verification_json:
        os.makedirs("logs", exist_ok=True)
        arch = platform.machine()
        cv_ver = cv2.__version__
        cv_file = getattr(cv2, "__file__", "BUILTIN")
        build_info = cv2.getBuildInformation()
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
            "numpy_version": np.__version__,
            "kleidicv_simd_flags": kleidi_flags[:10],
        }
        with open("logs/cool_verification.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        with open("logs/cool_verification.txt", "w", encoding="utf-8") as tf:
            tf.write(f"Instance Type: {rep['instance_type']}\n")
            tf.write(f"AMI ID: {rep['ami_id']}\n")
            tf.write(f"Architecture: {rep['architecture']}\n")
            tf.write(f"OpenCV: {rep['cv2_version']} ({rep['cv2_file']})\n")
            tf.write(f"NumPy: {rep['numpy_version']}\n")
            tf.write("Build flags:\n" + "\n".join(kleidi_flags[:10]) + "\n")
        print("[OK] Written logs/cool_verification.json and logs/cool_verification.txt")

    return 0

if __name__ == "__main__":
    sys.exit(main())
