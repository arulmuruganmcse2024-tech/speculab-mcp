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
    return 0

if __name__ == "__main__":
    sys.exit(main())
