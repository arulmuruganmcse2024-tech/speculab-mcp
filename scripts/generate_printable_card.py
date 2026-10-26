"""Generate the canonical printable physical calibration card.

Card Geometry:
- Active Card Area: 100.0 mm x 40.0 mm
- Canonical Pixel Canvas: 2000 px x 800 px (nominal scale 20.0 px/mm)
- Marker 0: (100, 200) to (500, 600) [400x400 px] (DICT_4X4_50)
- Foil Box: (600, 200) to (1400, 600) [800x400 px]
- Marker 1: (1500, 200) to (1900, 600) [400x400 px] (DICT_4X4_50)
- Top ORB Band: x=100..1900, y=40..160
- Bottom ORB Band: x=100..1900, y=640..760
"""

import argparse
import os
import random
import cv2
import numpy as np


def create_aruco_marker(marker_id: int, size_px: int = 400) -> np.ndarray:
    """Generate ArUco marker image using DICT_4X4_50."""
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    elif hasattr(cv2.aruco, "drawMarker"):
        marker = cv2.aruco.drawMarker(dictionary, marker_id, size_px)
    else:
        raise RuntimeError("No supported ArUco marker generation function found in cv2.aruco")
    return marker


def draw_deterministic_orb_texture(
    canvas: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int, seed: int
) -> None:
    """Populate band with deterministic aperiodic geometric primitives for ORB anchoring."""
    rng = random.Random(seed)

    # 1. Alphanumeric fiducial strings along the band
    fiducial_strings = [
        "SPECULAB-ORB-A1",
        "KLEIDICV-CALIB-02",
        "CANONICAL-20PXMM",
        "FIDUCIAL-X709",
        "SPECULAB-REF-44",
        "ACTIVE-FUSION-Z9",
        "OPENCV-CALIB-3X",
        "CANVAS-ANCHOR-W2",
    ]
    step_x = (x_end - x_start) // len(fiducial_strings)
    for idx, text in enumerate(fiducial_strings):
        tx = x_start + idx * step_x + rng.randint(5, 20)
        ty = y_start + (y_end - y_start) // 2 + rng.randint(-15, 15)
        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60)),
            2,
            cv2.LINE_AA,
        )

    # 2. Scalene triangles
    for _ in range(35):
        cx = rng.randint(x_start + 15, x_end - 15)
        cy = rng.randint(y_start + 15, y_end - 15)
        pts = np.array(
            [
                [cx + rng.randint(-25, 25), cy + rng.randint(-20, 20)],
                [cx + rng.randint(-25, 25), cy + rng.randint(-20, 20)],
                [cx + rng.randint(-25, 25), cy + rng.randint(-20, 20)],
            ],
            np.int32,
        )
        color = (rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50))
        if rng.random() > 0.5:
            cv2.fillPoly(canvas, [pts], color)
        else:
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    # 3. Concentric and varying circles
    for _ in range(30):
        cx = rng.randint(x_start + 10, x_end - 10)
        cy = rng.randint(y_start + 10, y_end - 10)
        radius = rng.randint(4, 18)
        color = (rng.randint(15, 55), rng.randint(15, 55), rng.randint(15, 55))
        cv2.circle(canvas, (cx, cy), radius, color, rng.choice([1, 2, -1]))

    # 4. Asymmetric crosshairs and stars
    for _ in range(25):
        cx = rng.randint(x_start + 10, x_end - 10)
        cy = rng.randint(y_start + 10, y_end - 10)
        arm_l = rng.randint(6, 18)
        arm_r = rng.randint(6, 18)
        arm_u = rng.randint(6, 18)
        arm_d = rng.randint(6, 18)
        color = (rng.randint(0, 40), rng.randint(0, 40), rng.randint(0, 40))
        cv2.line(canvas, (cx - arm_l, cy), (cx + arm_r, cy), color, 2)
        cv2.line(canvas, (cx, cy - arm_u), (cx, cy + arm_d), color, 2)


def generate_card(output_path: str = "assets/printable_test_card.png") -> np.ndarray:
    """Generate and save the 2000x800 canonical printable card image."""
    canvas_w = 2000
    canvas_h = 800
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Outer border of active card area
    cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (0, 0, 0), 2)

    # 1. Place ArUco Marker ID 0 at x=100..500, y=200..600
    marker_0 = create_aruco_marker(marker_id=0, size_px=400)
    canvas[200:600, 100:500] = cv2.cvtColor(marker_0, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas,
        "ARUCO ID 0 (PRIMARY METRIC)",
        (100, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # 2. Place ArUco Marker ID 1 at x=1500..1900, y=200..600
    marker_1 = create_aruco_marker(marker_id=1, size_px=400)
    canvas[200:600, 1500:1900] = cv2.cvtColor(marker_1, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas,
        "ARUCO ID 1 (DUAL-MARKER 8PT)",
        (1500, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # 3. Foil Inspection Box outline at x=600..1400, y=200..600 (40 mm x 20 mm)
    foil_x, foil_y, foil_w, foil_h = 600, 200, 800, 400
    cv2.rectangle(
        canvas,
        (foil_x, foil_y),
        (foil_x + foil_w, foil_y + foil_h),
        (0, 0, 0),
        3,
    )

    # Interior instruction text
    instruction_lines = [
        "MOUNT CARD ON RIGID FLAT BACKING.",
        "TAPE MATTE BLACK PAPER + TAUT FLAT FOIL HERE (40x20 mm)",
    ]
    cv2.putText(
        canvas,
        instruction_lines[0],
        (foil_x + 80, foil_y + 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (100, 100, 100),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        instruction_lines[1],
        (foil_x + 30, foil_y + 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (100, 100, 100),
        2,
        cv2.LINE_AA,
    )

    # Pilot zone tick marks and labels outside top/bottom edges
    # Zone A: x = 800, Zone B: x = 1000, Zone C: x = 1200
    zones = [
        (800, "ZONE A: THROUGH-HOLE"),
        (1000, "ZONE B: BLIND PIN DENT"),
        (1200, "ZONE C: SURFACE SCRATCH"),
    ]
    for zx, zlabel in zones:
        # Top tick & label
        cv2.line(canvas, (zx, 180), (zx, 200), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            zlabel,
            (zx - 80, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        # Bottom tick
        cv2.line(canvas, (zx, 600), (zx, 620), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            zlabel,
            (zx - 80, 635),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Horizontal center ticks at y = 400
    cv2.line(canvas, (foil_x - 15, 400), (foil_x, 400), (0, 0, 0), 2)
    cv2.line(canvas, (foil_x + foil_w, 400), (foil_x + foil_w + 15, 400), (0, 0, 0), 2)

    # 4. Vertical Y-Axis Non-Repetitive ORB Texture Bands
    # Top Band: x = 100..1900, y = 40..160
    draw_deterministic_orb_texture(canvas, 100, 1900, 40, 160, seed=101)
    cv2.rectangle(canvas, (100, 40), (1900, 160), (180, 180, 180), 1)

    # Bottom Band: x = 100..1900, y = 640..760
    draw_deterministic_orb_texture(canvas, 100, 1900, 640, 760, seed=202)
    cv2.rectangle(canvas, (100, 640), (1900, 760), (180, 180, 180), 1)

    # 5. Ruler verification scale bar (20.0 mm = 400 px)
    scale_x0 = 100
    scale_y0 = 780
    scale_x1 = 500  # 400 px
    cv2.line(canvas, (scale_x0, scale_y0), (scale_x1, scale_y0), (0, 0, 0), 3)
    cv2.line(canvas, (scale_x0, scale_y0 - 10), (scale_x0, scale_y0 + 10), (0, 0, 0), 2)
    cv2.line(canvas, (scale_x1, scale_y0 - 10), (scale_x1, scale_y0 + 10), (0, 0, 0), 2)
    cv2.putText(
        canvas,
        "400 px = 20.0 mm | MEASURE MARKER 0 BLACK SQUARE IN MM WITH A RULER AND SET marker_size_mm IN config.yaml",
        (scale_x1 + 20, scale_y0 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Ensure parent directory exists and save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical printable test card")
    parser.add_argument(
        "--output",
        type=str,
        default="assets/printable_test_card.png",
        help="Output image path",
    )
    args = parser.parse_args()
    card = generate_card(args.output)
    print(f"Generated printable card at {args.output} (shape={card.shape})")


if __name__ == "__main__":
    main()
