"""Dump candidate contours and morphological inspection data for debugging."""

import argparse
import json
import os
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

from src.vision_engine import PlanarActiveVisionEngine, load_image_exif_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump candidate contours and metrics")
    parser.add_argument(
        "--image",
        type=str,
        default="assets/specimens/specimen_01/view_1.jpg",
        help="Input image path (defaults to specimen_01 view_1.jpg)",
    )
    parser.add_argument("--thresh", type=int, default=30, help="Morphology threshold")
    parser.add_argument(
        "--output-json",
        type=str,
        default="logs/tuning_contours_dump.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    engine = PlanarActiveVisionEngine("config.yaml")
    img = load_image_exif_safe(args.image)
    rect, meta = engine.rectify_to_canonical_plane(img)

    candidates, bhat = engine.extract_raw_candidate_contours(rect, morph_intensity_thresh=args.thresh)
    serializable = []
    for c in candidates:
        entry = {k: v for k, v in c.items() if k != "raw_contour"}
        serializable.append(entry)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "candidate_count": len(serializable), "candidates": serializable}, f, indent=2)

    print(f"Dumped {len(serializable)} candidate contours to {args.output_json}")


if __name__ == "__main__":
    main()
