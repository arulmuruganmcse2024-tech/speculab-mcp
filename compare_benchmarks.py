"""Comparison engine for authentic OpenCV benchmark runs (M-19, M-29, D5).

Merges benchmark_*.json files into benchmark_comparison.json and renders
Markdown tables comparing 1-thread SIMD and N-thread scaling profiles.
Labels differences within combined standard error as "Parity / Within Noise".
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def format_cell(mean_ms: float, std_ms: float) -> str:
    """Format mean +/- std."""
    return f"{mean_ms:.2f} +/- {std_ms:.2f} ms"


def compute_comparison_verdict(
    mean_base: float, std_base: float, mean_test: float, std_test: float
) -> Tuple[float, str]:
    """Compute speedup ratio and verify if difference is outside combined standard error."""
    diff = abs(mean_base - mean_test)
    combined_stderr = math.sqrt(std_base**2 + std_test**2)

    speedup = round(mean_base / max(1e-6, mean_test), 2)
    if diff <= combined_stderr:
        verdict = f"{speedup:.2f}x (Parity / Within Noise)"
    elif mean_test < mean_base:
        pct_faster = round(((mean_base - mean_test) / mean_base) * 100.0, 1)
        verdict = f"{speedup:.2f}x (+{pct_faster}% faster)"
    else:
        pct_slower = round(((mean_test - mean_base) / mean_base) * 100.0, 1)
        verdict = f"{speedup:.2f}x (-{pct_slower}% slower)"
    return speedup, verdict


def render_markdown_table(
    profile_key: str,
    runs: List[Dict[str, Any]],
) -> str:
    """Render comparison Markdown table for a specific thread profile."""
    if not runs:
        return "*No benchmark runs available to compare.*"

    stages = [
        ("resize_ms", "cv2.resize (1280 proxy)"),
        ("cvt_color_ms", "cv2.cvtColor (BGR->GRAY)"),
        ("gaussian_blur_ms", "cv2.GaussianBlur (3x3)"),
        ("aruco_detect_ms", "ArUco Detect & Metric"),
        ("warp_perspective_ms", "cv2.warpPerspective (2000x800)"),
        ("glare_mask_and_entropy_ms", "Hysteresis Glare & Entropy"),
        ("orb_extract_ms", "ORB Feature Extraction"),
        ("usac_magsac_ms", "USAC_MAGSAC & Fusion"),
        ("detection_morphology_ms", "BLACKHAT 61x61 Denoised"),
        ("total_pipeline_ms", "Total Pipeline Inspection"),
    ]

    labels = [r.get("label", f"Run {idx}") for idx, r in enumerate(runs)]
    header = ["Inspection Stage"] + labels
    if len(runs) == 2:
        header.append(f"Speedup ({labels[1]} vs {labels[0]})")

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for stage_key, stage_title in stages:
        row = [stage_title]
        stats_list = []
        for r in runs:
            prof = r.get(profile_key, {}).get("stages", {})
            st_data = prof.get(stage_key, {})
            mean_ms = st_data.get("mean_ms", 0.0)
            std_ms = st_data.get("std_ms", 0.0)
            stats_list.append((mean_ms, std_ms))
            row.append(format_cell(mean_ms, std_ms))

        if len(runs) == 2:
            m0, s0 = stats_list[0]
            m1, s1 = stats_list[1]
            _, verdict = compute_comparison_verdict(m0, s0, m1, s1)
            row.append(verdict)

        lines.append("| " + " | ".join(row) + " |")

    # Add Throughput and Peak RSS rows
    fps_row = ["Throughput (FPS)"]
    for r in runs:
        fps = r.get(profile_key, {}).get("throughput_fps", 0.0)
        fps_row.append(f"{fps:.1f} fps")
    if len(runs) == 2:
        fps_base = runs[0].get(profile_key, {}).get("throughput_fps", 0.0)
        fps_test = runs[1].get(profile_key, {}).get("throughput_fps", 0.0)
        ratio = round(fps_test / max(1e-6, fps_base), 2)
        fps_row.append(f"{ratio:.2f}x throughput")
    lines.append("| " + " | ".join(fps_row) + " |")

    rss_row = ["Peak RSS Memory (MB)"]
    for r in runs:
        rss = r.get(profile_key, {}).get("peak_rss_mb", 0.0)
        rss_row.append(f"{rss:.1f} MB")
    if len(runs) == 2:
        rss_base = runs[0].get(profile_key, {}).get("peak_rss_mb", 0.0)
        rss_test = runs[1].get(profile_key, {}).get("peak_rss_mb", 0.0)
        diff_mb = round(rss_test - rss_base, 1)
        rss_row.append(f"{diff_mb:+.1f} MB delta")
    lines.append("| " + " | ".join(rss_row) + " |")

    return "\n".join(lines)


def compare_benchmarks(
    input_files: Optional[List[str]] = None,
    output_path: str = "benchmark_comparison.json",
) -> Optional[Dict[str, Any]]:
    """Merge benchmark JSONs and output comparison tables and structured comparison JSON."""
    if not input_files:
        input_files = sorted(glob.glob("benchmark_*.json"))
        # Exclude comparison file itself
        input_files = [f for f in input_files if not f.endswith("benchmark_comparison.json")]

    if not input_files:
        print("[INFO] No benchmark_*.json files found to compare.")
        return None

    runs = []
    for fp in input_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")

    if not runs:
        print("[INFO] No valid benchmark runs loaded.")
        return None

    table_1thread = render_markdown_table("profile_1thread", runs)
    table_nthread = render_markdown_table("profile_nthread", runs)

    comparison_report = {
        "compared_runs": [r.get("label", "unnamed") for r in runs],
        "input_files": input_files,
        "runs": runs,
        "markdown_tables": {
            "table_1thread": table_1thread,
            "table_nthread": table_nthread,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\n### 1-Thread SIMD Profile (KleidiCV / Single-Core Isolation)\n")
    print(table_1thread)
    print(f"\n### Multi-Thread Profile (Scaling)\n")
    print(table_nthread)
    print(f"\nComparison written to {output_path}")

    return comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare authentic benchmark runs")
    parser.add_argument("--inputs", nargs="*", default=None, help="Input benchmark JSON paths")
    parser.add_argument("--output", type=str, default="benchmark_comparison.json", help="Output comparison JSON path")
    args = parser.parse_args()

    compare_benchmarks(input_files=args.inputs, output_path=args.output)


if __name__ == "__main__":
    main()
