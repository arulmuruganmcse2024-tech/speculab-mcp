"""Dump candidate contours and morphological inspection data for debugging."""

import argparse
import json
import cv2
import numpy as np

from src.vision_engine import PlanarActiveVisionEngine, load_image_exif_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump candidate contours and metrics")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--thresh", type=int, default=30, help="Morphology threshold")
    parser.add_argument("--output-json", type=str, default="logs/dumped_contours.json", help="Output JSON path")
    args = parser.parse_args()

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
