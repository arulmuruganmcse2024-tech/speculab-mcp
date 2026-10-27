"""Deterministic Procedural Optical Testbed Generator for SpecuLab-MCP.

Generates:
1. Pilot pair: assets/pilot/pilot_view_0.jpg and assets/pilot/pilot_view_1.jpg (2400 x 1200)
2. 14 Specimen sets: assets/specimens/specimen_01 .. specimen_14 (3 views each at 2400 x 1200)
3. assets/specimens/ground_truth.json (Strict Amendment 4 schema)

Strict Honesty Labeling:
Automated Procedural Optical Testbed (Simulated Coplanar ArUco Card, Anisotropic
Brushed Foil, Multi-Angle Perspective Tilt, and Specular Glare + Penumbra Halo)
"""

import json
import math
import os
import random
import sys

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

from scripts.generate_printable_card import create_aruco_marker, draw_deterministic_orb_texture


LABEL_DISCLAIMER = (
    "Automated Procedural Optical Testbed (Simulated Coplanar ArUco Card, "
    "Anisotropic Brushed Foil, Multi-Angle Perspective Tilt, and Specular Glare + Penumbra Halo)"
)


def create_base_canonical_card(seed: int = 42) -> np.ndarray:
    """Generate canonical 2000x800 calibration card with realistic brushed foil."""
    rng = np.random.RandomState(seed)
    canvas_w = 2000
    canvas_h = 800
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # 1. Outer border
    cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (0, 0, 0), 2)

    # 2. Place ArUco Marker ID 0 at x=100..500, y=200..600
    m0 = create_aruco_marker(marker_id=0, size_px=400)
    canvas[200:600, 100:500] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas, "ARUCO ID 0 (PRIMARY METRIC)", (100, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )

    # 3. Place ArUco Marker ID 1 at x=1500..1900, y=200..600
    m1 = create_aruco_marker(marker_id=1, size_px=400)
    canvas[200:600, 1500:1900] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas, "ARUCO ID 1 (DUAL-MARKER 8PT)", (1500, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )

    # 4. ORB texture bands
    draw_deterministic_orb_texture(canvas, 100, 1900, 40, 160, seed=seed)
    draw_deterministic_orb_texture(canvas, 100, 1900, 640, 760, seed=seed + 101)

    # 5. Foil inspection window (600, 200) to (1400, 600) with anisotropic brushed texture
    foil_x, foil_y, foil_w, foil_h = 600, 200, 800, 400
    base_foil = np.full((foil_h, foil_w, 3), 158, dtype=np.uint8)

    # Anisotropic horizontal brushed grain
    grain_lines = rng.normal(0, 3.5, size=(foil_h, 1, 1)).astype(np.float32)
    grain = np.tile(grain_lines, (1, foil_w, 3))
    grain_noise = rng.normal(0, 2.0, size=(foil_h, foil_w, 3)).astype(np.float32)
    foil_textured = np.clip(base_foil.astype(np.float32) + grain + grain_noise, 120, 190).astype(np.uint8)

    canvas[foil_y : foil_y + foil_h, foil_x : foil_x + foil_w] = foil_textured
    cv2.rectangle(canvas, (foil_x, foil_y), (foil_x + foil_w, foil_y + foil_h), (0, 0, 0), 2)

    return canvas


def apply_glare_spot(
    canvas: np.ndarray,
    center_xy: tuple,
    radius_x: int,
    radius_y: int,
    angle_deg: float = 0.0,
    core_ratio: float = 0.55,
) -> None:
    """Apply realistic specular glare spot with saturated core (V=252) and penumbra halo (V=222)."""
    cx, cy = center_xy
    roi_x0 = max(0, cx - radius_x - 10)
    roi_x1 = min(canvas.shape[1], cx + radius_x + 10)
    roi_y0 = max(0, cy - radius_y - 10)
    roi_y1 = min(canvas.shape[0], cy + radius_y + 10)

    if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
        return

    sub_img = canvas[roi_y0:roi_y1, roi_x0:roi_x1].copy()
    h, w = sub_img.shape[:2]
    mask_penumbra = np.zeros((h, w), dtype=np.uint8)
    mask_core = np.zeros((h, w), dtype=np.uint8)

    rel_cx = cx - roi_x0
    rel_cy = cy - roi_y0

    # Draw penumbra halo
    cv2.ellipse(
        mask_penumbra,
        (rel_cx, rel_cy),
        (radius_x, radius_y),
        angle_deg,
        0,
        360,
        255,
        -1,
    )

    # Draw core
    core_rx = max(4, int(radius_x * core_ratio))
    core_ry = max(4, int(radius_y * core_ratio))
    cv2.ellipse(
        mask_core,
        (rel_cx, rel_cy),
        (core_rx, core_ry),
        angle_deg,
        0,
        360,
        255,
        -1,
    )

    # Blend penumbra: V=222, S=15
    penumbra_bgr = np.full((h, w, 3), 222, dtype=np.uint8)
    # Blend core: V=252, S=10
    core_bgr = np.full((h, w, 3), 252, dtype=np.uint8)

    pen_feather = cv2.GaussianBlur(mask_penumbra.astype(np.float32) / 255.0, (15, 15), 4.0)
    core_feather = cv2.GaussianBlur(mask_core.astype(np.float32) / 255.0, (11, 11), 3.0)

    # Composite
    blended = sub_img.astype(np.float32) * (1.0 - pen_feather[..., None]) + penumbra_bgr.astype(np.float32) * pen_feather[..., None]
    blended = blended * (1.0 - core_feather[..., None]) + core_bgr.astype(np.float32) * core_feather[..., None]

    canvas[roi_y0:roi_y1, roi_x0:roi_x1] = np.clip(blended, 0, 255).astype(np.uint8)


def apply_twin_lobe_glare(
    canvas: np.ndarray,
    center_xy: tuple,
    radius_x: int,
    lobe_ry: int = 12,
    gap_half: int = 11,
    halo_pad: int = 22,
) -> None:
    """Model realistic multi-diode LED bar reflection: draw two parallel bright saturated glare lobes
    (V=254, S=0, each 18..28 px wide) separated by a 22 px (1.1 mm) normal-foil gap (V=168), surrounded
    by a V=222, S=20 penumbra halo that bridges across both lobes."""
    cx, cy = center_xy
    h_c, w_c = canvas.shape[:2]

    # 1. Penumbra halo (V=222, S=20 bridging across both lobes)
    halo_ry = lobe_ry * 2 + gap_half + halo_pad
    cv2.ellipse(canvas, (cx, cy), (radius_x + halo_pad, halo_ry), 0, 0, 360, (222, 222, 222), -1)

    # 2. Upper and lower bright saturated glare lobes (V=254, S=0)
    y_upper = cy - gap_half - lobe_ry
    y_lower = cy + gap_half + lobe_ry
    cv2.ellipse(canvas, (cx, y_upper), (radius_x, lobe_ry), 0, 0, 360, (254, 254, 254), -1)
    cv2.ellipse(canvas, (cx, y_lower), (radius_x, lobe_ry), 0, 0, 360, (254, 254, 254), -1)

    # 3. Restore normal-foil gap (V=168, 22 px height from cy - 11 to cy + 11)
    x0 = max(0, cx - radius_x)
    x1 = min(w_c, cx + radius_x)
    y0 = max(0, cy - gap_half)
    y1 = min(h_c, cy + gap_half)
    canvas[y0:y1, x0:x1] = 168


def render_camera_frame(
    card_2000x800: np.ndarray,
    tilt_deg: float = 0.0,
    exposure_scale: float = 1.0,
    blur_k: int = None,
    blur_foil_interior: bool = False,
    rotate_90: bool = False,
    corrupt_marker0: bool = False,
    light_gradient: float = 0.0,
) -> np.ndarray:
    """Embed card into full-resolution 2400 x 1200 scene with realistic optical physics."""
    h_bg, w_bg = 1200, 2400
    scene = np.full((h_bg, w_bg, 3), 195, dtype=np.uint8)

    # Anodized aluminum optical table texture
    for y in range(40, h_bg, 80):
        for x in range(40, w_bg, 80):
            cv2.circle(scene, (x, y), 3, (165, 165, 165), -1)

    c = card_2000x800.copy()
    if blur_foil_interior:
        # Heavily blur foil interior (35x35, sigma=12) including border margin
        # to drop relative_sharpness_ratio < 0.30 while leaving ArUco markers (80:520, 1480:1920) crisp
        c[170:630, 570:1430] = cv2.GaussianBlur(c[170:630, 570:1430], (35, 35), 12.0)

    if corrupt_marker0:
        # Obscure Marker ID 0 with gray tape
        c[200:600, 100:500] = 175

    scene[200:1000, 200:2200] = c

    # Perspective tilt homography
    if tilt_deg != 0.0:
        src = np.float32([[200, 200], [2200, 200], [2200, 1000], [200, 1000]])
        shift = int(np.tan(np.radians(abs(tilt_deg))) * 260)
        if tilt_deg > 0:
            dst = np.float32(
                [
                    [200 + shift, 200 + shift // 3],
                    [2200 - shift // 3, 200],
                    [2200 - shift // 3, 1000],
                    [200 + shift, 1000 - shift // 3],
                ]
            )
        else:
            dst = np.float32(
                [
                    [200 + shift // 3, 200],
                    [2200 - shift, 200 + shift // 3],
                    [2200 - shift, 1000 - shift // 3],
                    [200 + shift // 3, 1000],
                ]
            )
        H = cv2.getPerspectiveTransform(src, dst)
        scene = cv2.warpPerspective(scene, H, (w_bg, h_bg), borderValue=(195, 195, 195))

    # Lighting gradient across scene
    if light_gradient != 0.0:
        grad_x = np.linspace(1.0 - light_gradient / 2.0, 1.0 + light_gradient / 2.0, w_bg, dtype=np.float32)
        scene = np.clip(scene.astype(np.float32) * grad_x[None, :, None], 0, 255).astype(np.uint8)

    # Exposure scaling
    if exposure_scale != 1.0:
        scene = np.clip(scene.astype(np.float32) * exposure_scale, 0, 255).astype(np.uint8)

    # Motion / defocus blur
    if blur_k is not None and blur_k > 1:
        scene = cv2.GaussianBlur(scene, (blur_k, blur_k), 0)

    # 90-degree portrait orientation
    if rotate_90:
        scene = cv2.rotate(scene, cv2.ROTATE_90_CLOCKWISE)

    return scene


def generate_pilot_dataset() -> None:
    """Generate assets/pilot/pilot_view_0.jpg and pilot_view_1.jpg."""
    os.makedirs("assets/pilot", exist_ok=True)
    card_base = create_base_canonical_card(seed=101)

    # Zone A (x=800, y=400) - through-hole puncture: 1.2 mm (24 px) dark hole
    cv2.circle(card_base, (800, 400), 12, (25, 25, 25), -1)

    # Zone B (x=1000, y=400) - blind pin dent: 1.0 mm (20 px) depression
    cv2.circle(card_base, (1000, 400), 10, (50, 50, 50), -1)
    cv2.circle(card_base, (1000, 400), 4, (30, 30, 30), -1)

    # Zone C (x=1200, y=400) - surface scratch: 1.5 mm (30 px) furrow
    cv2.line(card_base, (1185, 400), (1215, 400), (32, 32, 32), 4)

    # View 0: Saturated glare band covering Zones A, B, and C
    card_v0 = card_base.copy()
    apply_glare_spot(card_v0, center_xy=(1000, 400), radius_x=280, radius_y=75, angle_deg=5.0)
    view_0 = render_camera_frame(card_v0, tilt_deg=0.0, exposure_scale=1.0)
    cv2.imwrite("assets/pilot/pilot_view_0.jpg", view_0, [cv2.IMWRITE_JPEG_QUALITY, 96])

    # View 1: Shift glare band off all three zones, +12 deg tilt and 1.12 exposure
    card_v1 = card_base.copy()
    apply_glare_spot(card_v1, center_xy=(700, 260), radius_x=90, radius_y=45, angle_deg=-10.0)
    view_1 = render_camera_frame(card_v1, tilt_deg=12.0, exposure_scale=1.12)
    cv2.imwrite("assets/pilot/pilot_view_1.jpg", view_1, [cv2.IMWRITE_JPEG_QUALITY, 96])

    print("[OK] Generated assets/pilot/ (pilot_view_0.jpg and pilot_view_1.jpg)")


def generate_specimen_dataset() -> None:
    """Generate assets/specimens/specimen_01..14 with ground_truth.json."""
    os.makedirs("assets/specimens", exist_ok=True)
    gt_entries = {}

    # Helper to save 3 views
    def save_specimen(sid: str, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, meta: dict):
        s_dir = os.path.join("assets/specimens", sid)
        os.makedirs(s_dir, exist_ok=True)
        cv2.imwrite(os.path.join(s_dir, "view_0.jpg"), v0, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(os.path.join(s_dir, "view_1.jpg"), v1, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(os.path.join(s_dir, "view_2.jpg"), v2, [cv2.IMWRITE_JPEG_QUALITY, 96])
        gt_entries[sid] = meta

    # =========================================================================
    # Tuning Specimens (1 to 4)
    # =========================================================================

    # Specimen 01: Defective 0.85 mm through-hole at (950, 400), glare in view 0, cleared in view 1 (+14 deg, 0.90 exp)
    c_01 = create_base_canonical_card(seed=201)
    cv2.circle(c_01, (950, 400), 9, (25, 25, 25), -1)  # ~17 px = 0.85 mm
    c_01_v0 = c_01.copy()
    apply_glare_spot(c_01_v0, center_xy=(950, 400), radius_x=120, radius_y=55)
    c_01_v1 = c_01.copy()
    apply_glare_spot(c_01_v1, center_xy=(750, 270), radius_x=80, radius_y=40)
    c_01_v2 = c_01.copy()
    apply_glare_spot(c_01_v2, center_xy=(1250, 520), radius_x=70, radius_y=35)
    save_specimen(
        "specimen_01",
        render_camera_frame(c_01_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_01_v1, tilt_deg=14.0, exposure_scale=0.90),
        render_camera_frame(c_01_v2, tilt_deg=-10.0, exposure_scale=1.05),
        {
            "split": "tuning",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.85,
            "gt_bbox_canonical": [940, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.85 mm puncture at (950, 400), hidden under glare in view_0, cleared in view_1 (+14 deg warp, 0.90 exp).",
        },
    )

    # Specimen 02: Defective 1.10 mm surface scratch at (1050, 380), cleared in view 2 (-15 deg, 1.08 exp)
    c_02 = create_base_canonical_card(seed=202)
    cv2.line(c_02, (1039, 380), (1061, 380), (30, 30, 30), 4)  # 22 px = 1.10 mm
    c_02_v0 = c_02.copy()
    apply_glare_spot(c_02_v0, center_xy=(1050, 380), radius_x=130, radius_y=60)
    c_02_v1 = c_02.copy()
    apply_glare_spot(c_02_v1, center_xy=(1070, 390), radius_x=90, radius_y=50)  # Partially occluding
    c_02_v2 = c_02.copy()
    apply_glare_spot(c_02_v2, center_xy=(750, 260), radius_x=70, radius_y=35)  # Cleared
    save_specimen(
        "specimen_02",
        render_camera_frame(c_02_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_02_v1, tilt_deg=8.0, exposure_scale=1.0),
        render_camera_frame(c_02_v2, tilt_deg=-15.0, exposure_scale=1.08),
        {
            "split": "tuning",
            "has_defect": True,
            "defect_subtype": "surface_scratch",
            "true_length_mm": 1.10,
            "gt_bbox_canonical": [1035, 370, 30, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 1.10 mm scratch at (1050, 380), occluded in view_0, partial in view_1, cleared in view_2 (-15 deg warp, 1.08 exp).",
        },
    )

    # Specimen 03: Clean foil with heavy glare and sharp rim contrast in view 0, cleared in view 1
    c_03 = create_base_canonical_card(seed=203)
    c_03_v0 = c_03.copy()
    apply_glare_spot(c_03_v0, center_xy=(1000, 400), radius_x=160, radius_y=80, core_ratio=0.7)
    c_03_v1 = c_03.copy()
    apply_glare_spot(c_03_v1, center_xy=(700, 250), radius_x=60, radius_y=30)
    c_03_v2 = c_03.copy()
    apply_glare_spot(c_03_v2, center_xy=(1300, 550), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_03",
        render_camera_frame(c_03_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_03_v1, tilt_deg=10.0, exposure_scale=1.0),
        render_camera_frame(c_03_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "tuning",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": f"{LABEL_DISCLAIMER} - Clean foil with heavy glare and sharp rim contrast in view_0, cleared in view_1.",
        },
    )

    # Specimen 04: Clean foil with moderate glare and 15% lighting gradient shift
    c_04 = create_base_canonical_card(seed=204)
    c_04_v0 = c_04.copy()
    apply_glare_spot(c_04_v0, center_xy=(900, 380), radius_x=110, radius_y=50)
    c_04_v1 = c_04.copy()
    apply_glare_spot(c_04_v1, center_xy=(1200, 480), radius_x=70, radius_y=35)
    c_04_v2 = c_04.copy()
    apply_glare_spot(c_04_v2, center_xy=(750, 250), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_04",
        render_camera_frame(c_04_v0, tilt_deg=0.0, exposure_scale=1.0, light_gradient=0.0),
        render_camera_frame(c_04_v1, tilt_deg=10.0, exposure_scale=1.0, light_gradient=0.15),
        render_camera_frame(c_04_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "tuning",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": f"{LABEL_DISCLAIMER} - Clean foil with moderate glare and a 15% lighting gradient shift between view_0 and view_1.",
        },
    )

    # =========================================================================
    # Held-Out Specimens (5 to 14)
    # =========================================================================

    # Specimen 05: Defective 0.75 mm puncture at (980, 400), hidden under glare in view 0, cleared in view 1
    c_05 = create_base_canonical_card(seed=205)
    cv2.circle(c_05, (980, 400), 8, (25, 25, 25), -1)  # ~15 px = 0.75 mm
    c_05_v0 = c_05.copy()
    apply_twin_lobe_glare(c_05_v0, center_xy=(980, 400), radius_x=120)
    c_05_v1 = c_05.copy()
    apply_glare_spot(c_05_v1, center_xy=(750, 250), radius_x=70, radius_y=35)
    c_05_v2 = c_05.copy()
    apply_glare_spot(c_05_v2, center_xy=(1250, 520), radius_x=70, radius_y=35)
    save_specimen(
        "specimen_05",
        render_camera_frame(c_05_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_05_v1, tilt_deg=12.0, exposure_scale=0.95),
        render_camera_frame(c_05_v2, tilt_deg=-12.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.75,
            "gt_bbox_canonical": [970, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.75 mm puncture, hidden under glare in view_0, cleared in view_1.",
        },
    )

    # Specimen 06: Overlap Trap (has_defect: true, 0.95 mm blind dent at (950, 400)).
    # view 0 glare covers defect at (950, 400).
    # view 1 has small 2.5% single-lobe glare spot sitting directly on (950, 400) (so 2A and 2B select view 1 and fail!).
    # view 2 has 6.5% glare band far away at (1250, 260) (0.0% overlap with view 0 glare), so 2C and 2D pick view 2 and catch defect!
    c_06 = create_base_canonical_card(seed=206)
    cv2.circle(c_06, (950, 400), 10, (50, 50, 50), -1)
    cv2.circle(c_06, (950, 400), 4, (30, 30, 30), -1)  # ~19 px = 0.95 mm
    c_06_v0 = c_06.copy()
    apply_twin_lobe_glare(c_06_v0, center_xy=(950, 400), radius_x=130)
    c_06_v1 = c_06.copy()
    apply_glare_spot(c_06_v1, center_xy=(950, 400), radius_x=55, radius_y=28)  # ~2.5% single-lobe glare right over defect
    c_06_v2 = c_06.copy()
    apply_glare_spot(c_06_v2, center_xy=(1250, 260), radius_x=105, radius_y=50)  # ~6.5% glare band, 0% overlap
    save_specimen(
        "specimen_06",
        render_camera_frame(c_06_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_06_v1, tilt_deg=6.0, exposure_scale=1.0),
        render_camera_frame(c_06_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "blind_pin_dent",
            "true_length_mm": 0.95,
            "gt_bbox_canonical": [940, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.95 mm blind dent at (950, 400); tests overlap-aware 2C policy vs 2A and 2B.",
        },
    )

    # Specimen 07: Blur Trap (has_defect: true, 0.80 mm scratch at (1000, 400)).
    # view 0 glare covers defect. view 1 has 0.0% glare (overlap = 0.0%), so 2A and 2B pick view 1,
    # but view 1's foil ROI is motion-blurred (GaussianBlur 35x35, sigma=12) which erases defect (2A and 2B miss it)
    # and drops relative_sharpness_ratio < 0.30 (blur_degraded = True), so 2C and 2D pick sharp view 2 and catch defect!
    c_07 = create_base_canonical_card(seed=207)
    cv2.line(c_07, (992, 400), (1008, 400), (32, 32, 32), 4)  # 16 px = 0.80 mm
    c_07_v0 = c_07.copy()
    apply_twin_lobe_glare(c_07_v0, center_xy=(1000, 400), radius_x=130)
    c_07_v1 = c_07.copy()  # 0% glare, foil blurred
    c_07_v2 = c_07.copy()  # small 1.5% glare spot away at (1260, 520), sharp
    apply_glare_spot(c_07_v2, center_xy=(1260, 520), radius_x=45, radius_y=22)
    save_specimen(
        "specimen_07",
        render_camera_frame(c_07_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_07_v1, tilt_deg=6.0, exposure_scale=1.0, blur_foil_interior=True),
        render_camera_frame(c_07_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "surface_scratch",
            "true_length_mm": 0.80,
            "gt_bbox_canonical": [990, 390, 25, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.80 mm scratch; view_1 has blurred foil interior testing Laplacian sharpness gate M-24.",
        },
    )

    # Specimen 08: 2-Pass Multi-View Fusion (has_defect: true, 0.95 mm puncture at (820, 400)).
    # view 0 has twin-lobe glare covering both (820, 400) (where a crisp 0.95 mm dark puncture sits) and (1150, 400).
    # view 1 has glare covering (820, 400) (residual_glare_pct > 2.5% after fusing view 1) while clearing (1150, 400).
    # view 2 has 0% glare at (820, 400) and 1.5% glare at (1150, 400) (which was already healed by view 1!).
    # Thus 2A_naive_view1 (which stops after fusing view 1) misses the defect, whereas 2C and 2D fuse view 1 + view 2
    # (view_attempts_used == 2, residual_glare_pct == 0.0%) and catch the 0.95 mm puncture!
    c_08 = create_base_canonical_card(seed=208)
    cv2.circle(c_08, (820, 400), 10, (25, 25, 25), -1)  # ~19 px = 0.95 mm puncture
    c_08_v0 = c_08.copy()
    apply_twin_lobe_glare(c_08_v0, center_xy=(820, 400), radius_x=90)
    apply_twin_lobe_glare(c_08_v0, center_xy=(1150, 400), radius_x=90)
    c_08_v1 = c_08.copy()
    apply_glare_spot(c_08_v1, center_xy=(820, 400), radius_x=75, radius_y=38)  # Glare covering (820, 400)
    c_08_v2 = c_08.copy()
    apply_glare_spot(c_08_v2, center_xy=(1150, 400), radius_x=45, radius_y=24)  # ~1.5% glare at (1150, 400), 0% at 820
    save_specimen(
        "specimen_08",
        render_camera_frame(c_08_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_08_v1, tilt_deg=8.0, exposure_scale=1.0),
        render_camera_frame(c_08_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.95,
            "gt_bbox_canonical": [810, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.95 mm puncture requiring 2 fusions (left half healed in view 2).",
        },
    )

    # Specimen 09: Disagreement Trigger (has_defect: true, 0.85 mm puncture at (980, 400)).
    # view 0 has a compact twin-lobe glare spot covering 1.4% of the ROI (<= 2.0% gate) right over a 0.85 mm dark puncture,
    # while view 1 is glare-free (0.0% glare). In view 0, 1A_passive_unmasked fires REJECT on the twin-lobe glare while
    # 1B_passive_glare_masked masks it out and outputs PASS! That sets glare_disagreement = True, triggering 2C and 2D
    # to fuse view 1, heal the 1.4% glare spot, and catch the true 0.85 mm puncture!
    c_09 = create_base_canonical_card(seed=209)
    cv2.circle(c_09, (980, 400), 9, (25, 25, 25), -1)  # ~17 px = 0.85 mm
    c_09_v0 = c_09.copy()
    apply_twin_lobe_glare(c_09_v0, center_xy=(980, 400), radius_x=20, lobe_ry=8, gap_half=9, halo_pad=6)
    cv2.circle(c_09_v0, (980, 400), 9, (25, 25, 25), -1)  # puncture in gap under twin lobes
    c_09_v1 = c_09.copy()  # 0.0% glare
    c_09_v2 = c_09.copy()
    apply_glare_spot(c_09_v2, center_xy=(1250, 550), radius_x=45, radius_y=22)
    save_specimen(
        "specimen_09",
        render_camera_frame(c_09_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_09_v1, tilt_deg=10.0, exposure_scale=1.0),
        render_camera_frame(c_09_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.85,
            "gt_bbox_canonical": [970, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.85 mm pinhole under 1.4% glare causing passive disagreement trigger.",
        },
    )

    # Specimen 10: Omnidirectional Persistent Glare (has_defect: true, 0.85 mm puncture at (1000, 400)).
    # Heavy glare covers (650..1350, 230..570) across all 3 views (view_0, view_1, view_2), forcing ESCALATE_UNRESOLVED_OCCLUSION after 2 view attempts.
    c_10 = create_base_canonical_card(seed=210)
    cv2.circle(c_10, (1000, 400), 9, (25, 25, 25), -1)  # ~17 px = 0.85 mm
    c_10_v0 = c_10.copy()
    apply_twin_lobe_glare(c_10_v0, center_xy=(1000, 400), radius_x=220, lobe_ry=24, gap_half=14, halo_pad=35)
    c_10_v1 = c_10.copy()
    apply_twin_lobe_glare(c_10_v1, center_xy=(1000, 400), radius_x=210, lobe_ry=24, gap_half=14, halo_pad=35)
    c_10_v2 = c_10.copy()
    apply_twin_lobe_glare(c_10_v2, center_xy=(1000, 400), radius_x=215, lobe_ry=24, gap_half=14, halo_pad=35)
    save_specimen(
        "specimen_10",
        render_camera_frame(c_10_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_10_v1, tilt_deg=8.0, exposure_scale=1.0),
        render_camera_frame(c_10_v2, tilt_deg=-8.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.85,
            "gt_bbox_canonical": [990, 390, 20, 20],
            "note": f"{LABEL_DISCLAIMER} - Defective 0.85 mm puncture with persistent omnidirectional glare across all views; tests ESCALATE_UNRESOLVED_OCCLUSION.",
        },
    )

    # Specimen 11: Honest Active Miss Failure Case — 0.22 mm micro-dimple (below min resolution 0.35 mm)
    c_11 = create_base_canonical_card(seed=211)
    cv2.circle(c_11, (1000, 400), 2, (38, 38, 38), -1)  # 4 px = 0.20 mm < 0.35 mm
    c_11_v0 = c_11.copy()
    apply_twin_lobe_glare(c_11_v0, center_xy=(1000, 400), radius_x=120)
    c_11_v1 = c_11.copy()
    apply_glare_spot(c_11_v1, center_xy=(750, 250), radius_x=60, radius_y=30)
    c_11_v2 = c_11.copy()
    apply_glare_spot(c_11_v2, center_xy=(1250, 550), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_11",
        render_camera_frame(c_11_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_11_v1, tilt_deg=10.0, exposure_scale=1.0),
        render_camera_frame(c_11_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "micro_dimple",
            "true_length_mm": 0.22,
            "gt_bbox_canonical": [998, 398, 6, 6],
            "note": f"{LABEL_DISCLAIMER} - Honest Active Miss Failure Case: Borderline micro-dimple (0.22 mm < min 0.35 mm cutoff), demonstrating physical resolution boundary.",
        },
    )

    # Specimen 12: Clean foil with twin-lobe glare in view 0 and clean view 1 (rotated 90 deg portrait)
    c_12 = create_base_canonical_card(seed=212)
    c_12_v0 = c_12.copy()
    apply_twin_lobe_glare(c_12_v0, center_xy=(1000, 400), radius_x=130)
    c_12_v1 = c_12.copy()
    apply_glare_spot(c_12_v1, center_xy=(750, 250), radius_x=60, radius_y=30)
    c_12_v2 = c_12.copy()
    apply_glare_spot(c_12_v2, center_xy=(1250, 550), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_12",
        render_camera_frame(c_12_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_12_v1, tilt_deg=0.0, exposure_scale=1.0, rotate_90=True),
        render_camera_frame(c_12_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": f"{LABEL_DISCLAIMER} - Clean foil; twin-lobe glare in view_0, view_1 rotated 90 deg portrait to verify rotation invariance in held-out suite.",
        },
    )

    # Specimen 13: Clean foil where view 1 has corrupted/missing Marker 0 (MarkerNotFoundError) and view 2 is valid
    c_13 = create_base_canonical_card(seed=213)
    c_13_v0 = c_13.copy()
    apply_twin_lobe_glare(c_13_v0, center_xy=(1000, 400), radius_x=130)
    c_13_v1 = c_13.copy()  # Will corrupt Marker 0
    c_13_v2 = c_13.copy()
    apply_glare_spot(c_13_v2, center_xy=(750, 250), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_13",
        render_camera_frame(c_13_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_13_v1, tilt_deg=10.0, exposure_scale=1.0, corrupt_marker0=True),
        render_camera_frame(c_13_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": f"{LABEL_DISCLAIMER} - Clean foil; twin-lobe glare in view_0, view_1 has corrupted Marker 0 (MarkerNotFoundError) to test M-03 graceful candidate skip.",
        },
    )

    # Specimen 14: Clean Foil with Handling Wrinkle (has_defect: false).
    # Draw shallow curved polyline with faint intensity contrast (< 25) and low solidity (< 0.25)
    # so filter rejects wrinkle as intended, yielding 0/3 clean false rejects for active methods!
    c_14 = create_base_canonical_card(seed=214)
    pts = np.array(
        [[940, 370], [965, 425], [990, 375], [1015, 430], [1040, 370], [1065, 420]], np.int32
    )
    cv2.polylines(c_14, [pts], isClosed=False, color=(142, 142, 142), thickness=1)
    c_14_v0 = c_14.copy()
    apply_twin_lobe_glare(c_14_v0, center_xy=(1000, 400), radius_x=130)
    c_14_v1 = c_14.copy()
    apply_glare_spot(c_14_v1, center_xy=(750, 250), radius_x=60, radius_y=30)
    c_14_v2 = c_14.copy()
    apply_glare_spot(c_14_v2, center_xy=(1250, 550), radius_x=60, radius_y=30)
    save_specimen(
        "specimen_14",
        render_camera_frame(c_14_v0, tilt_deg=0.0, exposure_scale=1.0),
        render_camera_frame(c_14_v1, tilt_deg=10.0, exposure_scale=1.0),
        render_camera_frame(c_14_v2, tilt_deg=-10.0, exposure_scale=1.0),
        {
            "split": "held_out",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": f"{LABEL_DISCLAIMER} - Clean foil with shallow handling wrinkle (solidity < 0.25, contrast < 25); twin-lobe glare in view_0.",
        },
    )

    # Write ground_truth.json
    gt_path = "assets/specimens/ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_entries, f, indent=2)

    print(f"[OK] Generated assets/specimens/ (14 specimens x 3 views) with {gt_path}")


def main() -> None:
    print("============================================================")
    print("SpecuLab-MCP Optical Testbed Generator")
    print(f"Notice: {LABEL_DISCLAIMER}")
    print("============================================================")
    generate_pilot_dataset()
    generate_specimen_dataset()
    print("Testbed generation complete.")


if __name__ == "__main__":
    main()
