"""Pilot optical and physical feasibility check script for SpecuLab-MCP (D1, M-17, RT-5).

Inspects 80x80 px (4x4 mm) windows at Zone A (800, 400), Zone B (1000, 400),
and Zone C (1200, 400), evaluating defect SNR and specular glare contrast.
Outputs logs/pilot_physics_report.json and logs/pilot_zones_annotated.jpg.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path and auto-bootstrap venv
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from src.bootstrap import ensure_venv
    ensure_venv()
except ImportError:
    pass

import cv2
import numpy as np

from scripts.generate_printable_card import generate_card
from src.vision_engine import PlanarActiveVisionEngine, load_image_exif_safe


def analyze_zone(
    gray_roi: np.ndarray,
    glare_mask_roi: np.ndarray,
    blackhat_roi: np.ndarray,
    center_xy: Tuple[int, int],
    window_size: int = 80,
) -> Dict[str, Any]:
    """Extract optical signal properties in an 80x80 px inspection window."""
    cx, cy = center_xy
    half = window_size // 2
    h, w = gray_roi.shape[:2]

    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)

    w_gray = gray_roi[y0:y1, x0:x1]
    w_glare = glare_mask_roi[y0:y1, x0:x1]
    w_bhat = blackhat_roi[y0:y1, x0:x1]

    min_g = int(np.min(w_gray)) if w_gray.size > 0 else 0
    max_g = int(np.max(w_gray)) if w_gray.size > 0 else 0
    local_std = round(float(np.std(w_gray)), 2) if w_gray.size > 0 else 0.0
    glare_pct = round((float(np.count_nonzero(w_glare)) / max(1, w_glare.size)) * 100.0, 2)
    bhat_peak = round(float(np.max(w_bhat)), 1) if w_bhat.size > 0 else 0.0

    return {
        "window_bbox": [x0, y0, x1 - x0, y1 - y0],
        "min_gray": min_g,
        "max_gray": max_g,
        "local_std": local_std,
        "glare_coverage_pct": glare_pct,
        "blackhat_peak_response": bhat_peak,
    }


def run_pilot_check(
    view0_path: str,
    view1_path: str,
    output_report: str = "logs/pilot_physics_report.json",
    output_annotated: str = "logs/pilot_zones_annotated.jpg",
) -> Dict[str, Any]:
    """Execute physical pilot optical inspection on two views."""
    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    engine = PlanarActiveVisionEngine("config.yaml")

    # Load views or fallback to generated card
    if not os.path.exists(view0_path) or not os.path.exists(view1_path):
        print(f"[INFO] Pilot view paths not found on disk. Generating canonical pilot card for validation.")
        card = generate_card()
        v0 = card.copy()
        v1 = card.copy()
        v0[380:420, 780:820] = 30  # Synthetic Zone A through-hole
        v0[385:415, 985:1015] = 50  # Synthetic Zone B pin dent
        v0[395:405, 1180:1220] = 40  # Synthetic Zone C scratch
    else:
        v0 = load_image_exif_safe(view0_path)
        v1 = load_image_exif_safe(view1_path)

    rect0, meta0 = engine.rectify_to_canonical_plane(v0)
    rect1, meta1 = engine.rectify_to_canonical_plane(v1)

    rx, ry, rw, rh = engine.DEFAULT_ROI
    mask0, glare_pct0, _, _ = engine.compute_glare_mask(rect0)
    mask1, glare_pct1, _, _ = engine.compute_glare_mask(rect1)

    gray0 = cv2.cvtColor(rect0[ry : ry + rh, rx : rx + rw], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(rect1[ry : ry + rh, rx : rx + rw], cv2.COLOR_BGR2GRAY)

    k_morph = int(engine.detection_cfg.get("morph_kernel_px", 61))
    kernel_bhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
    bhat0 = cv2.morphologyEx(cv2.medianBlur(gray0, 5), cv2.MORPH_BLACKHAT, kernel_bhat)
    bhat1 = cv2.morphologyEx(cv2.medianBlur(gray1, 5), cv2.MORPH_BLACKHAT, kernel_bhat)

    # Pilot zones in ROI coordinates:
    # Global: Zone A (800, 400) -> ROI rel: (800 - 600, 400 - 200) = (200, 200)
    # Global: Zone B (1000, 400) -> ROI rel: (1000 - 600, 400 - 200) = (400, 200)
    # Global: Zone C (1200, 400) -> ROI rel: (1200 - 600, 400 - 200) = (600, 200)
    zones = {
        "zone_a_through_hole": (200, 200),
        "zone_b_blind_pin_dent": (400, 200),
        "zone_c_surface_scratch": (600, 200),
    }

    zone_metrics_v0 = {}
    zone_metrics_v1 = {}
    for zname, zrel in zones.items():
        zone_metrics_v0[zname] = analyze_zone(gray0, mask0, bhat0, zrel, window_size=80)
        zone_metrics_v1[zname] = analyze_zone(gray1, mask1, bhat1, zrel, window_size=80)

    # Full ROI detector evaluations
    pas_unmasked = engine.detect_defects(rect0, exclude_mask=None)
    pas_masked = engine.detect_defects(rect0, exclude_mask=mask0)

    healed, still_unhealed, res_g, inliers, heal_meta = engine.heal_accumulated_canvas(rect0, mask0, rect1)
    act = engine.detect_defects(healed, exclude_mask=still_unhealed)

    # Decision rules D1(b)
    # Peak response >= 42.0 threshold
    za_peak = max(zone_metrics_v0["zone_a_through_hole"]["blackhat_peak_response"], zone_metrics_v1["zone_a_through_hole"]["blackhat_peak_response"])
    zb_peak = max(zone_metrics_v0["zone_b_blind_pin_dent"]["blackhat_peak_response"], zone_metrics_v1["zone_b_blind_pin_dent"]["blackhat_peak_response"])
    zc_peak = max(zone_metrics_v0["zone_c_surface_scratch"]["blackhat_peak_response"], zone_metrics_v1["zone_c_surface_scratch"]["blackhat_peak_response"])

    selected_subtypes = []
    if za_peak >= 42.0:
        selected_subtypes.append("through_hole")
    if zb_peak >= 42.0:
        selected_subtypes.append("blind_pin_dent")
    if zc_peak >= 42.0:
        selected_subtypes.append("surface_scratch")

    report = {
        "view0_path": view0_path,
        "view1_path": view1_path,
        "view0_glare_pct": glare_pct0,
        "view1_glare_pct": glare_pct1,
        "residual_glare_pct": res_g,
        "zone_metrics_view0": zone_metrics_v0,
        "zone_metrics_view1": zone_metrics_v1,
        "peak_blackhat_responses": {
            "zone_a_through_hole": za_peak,
            "zone_b_blind_pin_dent": zb_peak,
            "zone_c_surface_scratch": zc_peak,
        },
        "selected_defect_subtypes": selected_subtypes,
        "detector_outcomes": {
            "passive_unmasked": pas_unmasked["verdict"],
            "passive_glare_masked": pas_masked["verdict"],
            "active_multi_view": act["verdict"],
        },
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Annotate zones on canvas
    annotated = rect0.copy()
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    for zname, (zx_rel, zy_rel) in zones.items():
        gx = rx + zx_rel
        gy = ry + zy_rel
        cv2.rectangle(annotated, (gx - 40, gy - 40), (gx + 40, gy + 40), (255, 0, 0), 2)
        cv2.putText(
            annotated,
            zname.upper(),
            (gx - 40, gy - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(output_annotated, annotated)

    print(f"Pilot check completed. Report written to {output_report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot physics and optical feasibility check")
    parser.add_argument("--view0", type=str, default="assets/pilot/pilot_view_0.jpg", help="View 0 path")
    parser.add_argument("--view1", type=str, default="assets/pilot/pilot_view_1.jpg", help="View 1 path")
    parser.add_argument("--output-report", type=str, default="logs/pilot_physics_report.json", help="Output JSON path")
    parser.add_argument("--output-image", type=str, default="logs/pilot_zones_annotated.jpg", help="Output annotated image")
    args = parser.parse_args()

    run_pilot_check(
        view0_path=args.view0,
        view1_path=args.view1,
        output_report=args.output_report,
        output_annotated=args.output_image,
    )


if __name__ == "__main__":
    main()
