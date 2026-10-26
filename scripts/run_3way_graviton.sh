#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Authentic Graviton Benchmark Comparison ==="

# 1. Run Graviton COOL benchmark inside ~/venv_cool_app
if [ -f "$HOME/venv_cool_app/bin/activate" ]; then
    echo "Activating ~/venv_cool_app for Graviton COOL profile..."
    source "$HOME/venv_cool_app/bin/activate"
    python3 run_authentic_benchmark.py --label graviton_cool --threads 4
else
    echo "Running with current Python environment as graviton_cool..."
    python3 run_authentic_benchmark.py --label graviton_cool --threads 4
fi

# 2. Setup isolated stock OpenCV venv for baseline comparison
STOCK_VENV="$HOME/venv_stock_cv"
if [ ! -d "$STOCK_VENV" ]; then
    echo "Creating isolated stock OpenCV environment: $STOCK_VENV"
    python3 -m venv "$STOCK_VENV"
    "$STOCK_VENV/bin/pip" install --upgrade pip
    "$STOCK_VENV/bin/pip" install opencv-python-headless numpy pyyaml
fi

echo "Activating ~/venv_stock_cv for Graviton Stock OpenCV profile..."
source "$STOCK_VENV/bin/activate"
python3 run_authentic_benchmark.py --label graviton_stock --threads 4

# 3. Compare benchmark results
echo "Generating comparative benchmark Markdown tables..."
python3 compare_benchmarks.py --inputs benchmark_graviton_cool.json benchmark_graviton_stock.json --output benchmark_comparison.json
