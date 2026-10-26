"""Empirical micro-benchmarking engine for SpecuLab-MCP.

Measures authentic OpenCV pipeline latencies across 1-thread SIMD and N-thread scaling (M-19, M-29, D5).
Excludes pricing when --hourly-cost == 0 (marks UNPRICED_EXCLUDED_FROM_COST).
Zero fabricated metrics: outputs empirical measurements only.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scripts.generate_printable_card import generate_card
from src.vision_engine import PlanarActiveVisionEngine, get_opencv_runtime_metadata, load_image_exif_safe


def get_peak_rss_mb() -> float:
    """Return peak resident set size in megabytes cross-platform."""
    try:
        import resource
        if sys.platform == "darwin":
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 2)
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except ImportError:
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
            if fn is None:
                fn = getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
            if fn is not None:
                fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
                fn.restype = wintypes.BOOL
                if fn(handle, ctypes.byref(counters), counters.cb):
                    return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 2)
        except Exception:
            pass
        return 0.0


def benchmark_single_iteration(
    engine: PlanarActiveVisionEngine,
    view0: np.ndarray,
    view1: np.ndarray,
) -> Dict[str, float]:
    """Execute one full active inspection cycle, recording granular stage latencies in ms."""
    t_tot_start = time.perf_counter()

    # 1. Resize
    t0 = time.perf_counter()
    orig_h, orig_w = view0.shape[:2]
    proxy_w = 1280
    proxy_h = int(round(orig_h * (proxy_w / float(orig_w))))
    proxy_bgr = cv2.resize(view0, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)
    resize_ms = (time.perf_counter() - t0) * 1000.0

    # 2. CvtColor
    t0 = time.perf_counter()
    proxy_gray = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2GRAY)
    cvt_color_ms = (time.perf_counter() - t0) * 1000.0

    # 3. GaussianBlur
    t0 = time.perf_counter()
    proxy_blur = cv2.GaussianBlur(proxy_gray, (3, 3), 0)
    gaussian_blur_ms = (time.perf_counter() - t0) * 1000.0

    # 4. ArUco detection & rectification
    t0 = time.perf_counter()
    rect0, _ = engine.rectify_to_canonical_plane(view0)
    aruco_detect_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Warp perspective
    t0 = time.perf_counter()
    rect1, _ = engine.rectify_to_canonical_plane(view1)
    warp_perspective_ms = (time.perf_counter() - t0) * 1000.0

    # 6. Glare mask and Shannon entropy
    t0 = time.perf_counter()
    mask0, _, entropy0, _ = engine.compute_glare_mask(rect0)
    glare_mask_and_entropy_ms = (time.perf_counter() - t0) * 1000.0

    # 7. ORB extraction
    t0 = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=int(engine.healing_cfg.get("orb_nfeatures", 1500)))
    gray_r0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
    kp0, des0 = orb.detectAndCompute(gray_r0, mask=None)
    orb_extract_ms = (time.perf_counter() - t0) * 1000.0

    # 8. USAC-MAGSAC alignment & healing
    t0 = time.perf_counter()
    healed, still_unhealed, _, _, _ = engine.heal_accumulated_canvas(rect0, mask0, rect1)
    usac_magsac_ms = (time.perf_counter() - t0) * 1000.0

    # 9. Detection morphology (61x61 BLACKHAT)
    t0 = time.perf_counter()
    det_res = engine.detect_defects(healed, exclude_mask=still_unhealed)
    detection_morphology_ms = (time.perf_counter() - t0) * 1000.0

    total_pipeline_ms = (time.perf_counter() - t_tot_start) * 1000.0

    return {
        "resize_ms": resize_ms,
        "cvt_color_ms": cvt_color_ms,
        "gaussian_blur_ms": gaussian_blur_ms,
        "aruco_detect_ms": aruco_detect_ms,
        "warp_perspective_ms": warp_perspective_ms,
        "glare_mask_and_entropy_ms": glare_mask_and_entropy_ms,
        "orb_extract_ms": orb_extract_ms,
        "usac_magsac_ms": usac_magsac_ms,
        "detection_morphology_ms": detection_morphology_ms,
        "total_pipeline_ms": total_pipeline_ms,
    }


def run_profile(
    engine: PlanarActiveVisionEngine,
    view0: np.ndarray,
    view1: np.ndarray,
    thread_count: int,
    warmup_iters: int = 5,
    timed_iters: int = 30,
) -> Dict[str, Any]:
    """Run benchmark iterations under a specific thread setting."""
    cv2.setNumThreads(thread_count)
    cv2.setRNGSeed(42)

    # Warmup
    for _ in range(warmup_iters):
        benchmark_single_iteration(engine, view0, view1)

    cv2.setRNGSeed(42)

    # Timed runs
    records: Dict[str, List[float]] = {
        "resize_ms": [],
        "cvt_color_ms": [],
        "gaussian_blur_ms": [],
        "aruco_detect_ms": [],
        "warp_perspective_ms": [],
        "glare_mask_and_entropy_ms": [],
        "orb_extract_ms": [],
        "usac_magsac_ms": [],
        "detection_morphology_ms": [],
        "total_pipeline_ms": [],
    }

    t0_wall = time.perf_counter()
    for _ in range(timed_iters):
        row = benchmark_single_iteration(engine, view0, view1)
        for k, v in row.items():
            records[k].append(v)
    total_wall_sec = time.perf_counter() - t0_wall

    stats: Dict[str, Dict[str, float]] = {}
    for k, vals in records.items():
        stats[k] = {
            "mean_ms": round(float(np.mean(vals)), 3),
            "std_ms": round(float(np.std(vals)), 3),
            "p50_ms": round(float(np.percentile(vals, 50)), 3),
            "p95_ms": round(float(np.percentile(vals, 95)), 3),
        }

    throughput_fps = round(float(timed_iters) / max(1e-6, total_wall_sec), 2)
    peak_rss = get_peak_rss_mb()

    return {
        "threads": thread_count,
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "throughput_fps": throughput_fps,
        "peak_rss_mb": peak_rss,
        "stages": stats,
    }


def run_benchmark(
    label: str,
    specimens_dir: str = "assets/specimens",
    threads: int = 4,
    iterations: int = 30,
    disable_cv_opt: bool = False,
    hourly_cost: float = 0.0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run 1-thread and N-thread authentic benchmarks on loaded image pairs."""
    engine = PlanarActiveVisionEngine("config.yaml")

    # Load pair from real specimens if available, otherwise generate authentic card pair
    v0_img = None
    v1_img = None
    folders = sorted(glob.glob(os.path.join(specimens_dir, "specimen_*")))
    for sdir in folders:
        views = sorted(glob.glob(os.path.join(sdir, "*.jpg")) + glob.glob(os.path.join(sdir, "*.png")))
        if len(views) >= 2:
            try:
                v0_img = load_image_exif_safe(views[0])
                v1_img = load_image_exif_safe(views[1])
                break
            except Exception:
                pass

    if v0_img is None or v1_img is None:
        # Pre-load canonical printable card pair
        v0_img = generate_card()
        v1_img = v0_img.copy()
        v1_img[200:600, 600:1400] = 175  # Slightly shifted exposure

    orig_opt = cv2.useOptimized()
    # Explicitly configure SIMD optimization state BEFORE any warmups or executions
    cv2.setUseOptimized(not disable_cv_opt)

    try:
        runtime_meta = get_opencv_runtime_metadata()
        # Profile 1: Single-core (threads=1) for KleidiCV/SIMD isolation (M-29)
        profile_1thread = run_profile(
            engine, v0_img, v1_img, thread_count=1, warmup_iters=5, timed_iters=iterations
        )

        # Profile 2: Multi-core (threads=N) for multi-threaded scaling
        profile_nthread = run_profile(
            engine, v0_img, v1_img, thread_count=threads, warmup_iters=5, timed_iters=iterations
        )
    finally:
        cv2.setUseOptimized(orig_opt)

    # Cost calculations (D5: Unpriced when --hourly-cost == 0)
    cost_info: Dict[str, Any] = {}
    if hourly_cost > 0.0:
        # Cost per 1M inspections = (1,000,000 / (fps * 3600)) * hourly_cost
        fps_1t = profile_1thread["throughput_fps"]
        fps_nt = profile_nthread["throughput_fps"]
        cost_1m_1t = round((1_000_000.0 / (fps_1t * 3600.0)) * hourly_cost, 4) if fps_1t > 0 else None
        cost_1m_nt = round((1_000_000.0 / (fps_nt * 3600.0)) * hourly_cost, 4) if fps_nt > 0 else None
        cost_info = {
            "pricing_status": "PRICED",
            "hourly_ec2_cost_usd": hourly_cost,
            "cost_per_1m_inspections_1thread_usd": cost_1m_1t,
            "cost_per_1m_inspections_nthread_usd": cost_1m_nt,
        }
    else:
        cost_info = {
            "pricing_status": "UNPRICED_EXCLUDED_FROM_COST",
            "hourly_ec2_cost_usd": None,
            "cost_per_1m_inspections_1thread_usd": None,
            "cost_per_1m_inspections_nthread_usd": None,
        }

    report = {
        "label": label,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opencv_runtime": runtime_meta,
        "cost_analysis": cost_info,
        "profile_1thread": profile_1thread,
        "profile_nthread": profile_nthread,
    }

    if output_path is None:
        output_path = f"benchmark_{label}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Benchmark '{label}' completed. Report written to {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run authentic OpenCV micro-benchmarks")
    parser.add_argument("--label", type=str, default="authentic_run", help="Benchmark run identifier")
    parser.add_argument("--specimens-dir", type=str, default="assets/specimens", help="Specimens directory")
    parser.add_argument("--threads", type=int, default=4, help="Thread count for multi-thread profile")
    parser.add_argument("--iterations", type=int, default=30, help="Number of timed benchmark iterations")
    parser.add_argument("--disable-cv-opt", action="store_true", help="Disable OpenCV SIMD hardware optimizations")
    parser.add_argument("--hourly-cost", type=float, default=0.0, help="EC2 instance hourly cost in USD (0.0 = unpriced)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(
        label=args.label,
        specimens_dir=args.specimens_dir,
        threads=args.threads,
        iterations=args.iterations,
        disable_cv_opt=args.disable_cv_opt,
        hourly_cost=args.hourly_cost,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
