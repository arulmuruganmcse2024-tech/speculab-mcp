"""Planar active vision engine for SpecuLab-MCP.

Provides:
- MarkerNotFoundError
- get_opencv_runtime_metadata()
- load_image_exif_safe()
- PlanarActiveVisionEngine (Stateless per-frame execution, M-20)
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps


class MarkerNotFoundError(RuntimeError):
    """Raised when primary metric ArUco marker ID 0 is missing."""
    pass


def get_opencv_runtime_metadata() -> Dict[str, Any]:
    """Empirically inspect OpenCV version, threads, optimizations, and build metadata."""
    ver = cv2.__version__
    major = int(ver.split(".")[0])
    build_info = cv2.getBuildInformation()

    simd_lines = []
    for line in build_info.splitlines():
        line_clean = line.strip()
        if any(term in line_clean.lower() for term in ["simd", "neon", "avx", "kleidi", "cpu", "dispatch"]):
            simd_lines.append(line_clean)

    return {
        "cv2_version": ver,
        "cv2_major": major,
        "threads_configured": cv2.getNumThreads(),
        "optimizations_enabled": bool(cv2.useOptimized()),
        "simd_metadata": simd_lines[:15],
        "build_information": build_info,
    }


def load_image_exif_safe(path: str) -> np.ndarray:
    """Safely load an image from disk, respecting EXIF orientation metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")

    with Image.open(path) as img:
        transposed = ImageOps.exif_transpose(img)
        rgb = np.array(transposed.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if bgr is None or bgr.size == 0:
        raise ValueError(f"Failed to decode image from path: {path}")

    return bgr


class PlanarActiveVisionEngine:
    """Stateless active vision inspection engine operating on canonical 2000x800 plane."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        calib = self.config.get("calibration", {})
        self.MARKER_0_ID: int = int(calib.get("marker_0_id", 0))
        self.MARKER_1_ID: int = int(calib.get("marker_1_id", 1))
        self.MARKER_SIZE_MM: float = float(calib.get("marker_size_mm", 20.0))
        self.CANVAS_W: int = int(calib.get("canvas_width_px", 2000))
        self.CANVAS_H: int = int(calib.get("canvas_height_px", 800))
        self.DEFAULT_ROI: Tuple[int, int, int, int] = tuple(
            calib.get("default_roi", [600, 200, 800, 400])
        )

        # Scale calibration
        self.CANONICAL_PX_PER_MM: float = 400.0 / self.MARKER_SIZE_MM
        self.MM_PER_PX: float = self.MARKER_SIZE_MM / 400.0

        self.glare_cfg = self.config.get("glare", {})
        self.sharpness_cfg = self.config.get("sharpness", {})
        self.healing_cfg = self.config.get("healing", {})
        self.detection_cfg = self.config.get("detection", {})
        self.optional_features = self.config.get("optional_features", {})

        # Canonical destination corners
        self.dst_0 = np.array(
            [[100.0, 200.0], [500.0, 200.0], [500.0, 600.0], [100.0, 600.0]],
            dtype=np.float32,
        )
        self.dst_1 = np.array(
            [[1500.0, 200.0], [1900.0, 200.0], [1900.0, 600.0], [1500.0, 600.0]],
            dtype=np.float32,
        )

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

    def _detect_markers(self, gray_img: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Run ArUco marker detection compatible with OpenCV 4.x and 5.x."""
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def rectify_to_canonical_plane(
        self,
        frame_bgr: np.ndarray,
        optional_flags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Rectify input image to the 2000x800 canonical metric inspection plane."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or invalid")

        opt = optional_flags if optional_flags is not None else self.optional_features
        orig_h, orig_w = frame_bgr.shape[:2]

        # 1. Resize to proxy width 1280 for fast ArUco detection
        t0 = time.perf_counter()
        proxy_w = 1280
        if orig_w != proxy_w:
            proxy_h = int(round(orig_h * (proxy_w / float(orig_w))))
            proxy_bgr = cv2.resize(frame_bgr, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)
        else:
            proxy_bgr = frame_bgr
            proxy_h = orig_h
        t1 = time.perf_counter()
        resize_ms = round((t1 - t0) * 1000.0, 3)

        # 2. Convert to grayscale
        proxy_gray = cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2GRAY)
        t2 = time.perf_counter()
        cvt_color_ms = round((t2 - t1) * 1000.0, 3)

        # 3. Gaussian blur
        proxy_blur = cv2.GaussianBlur(proxy_gray, (3, 3), 0)
        t3 = time.perf_counter()
        gaussian_blur_ms = round((t3 - t2) * 1000.0, 3)

        # 4. ArUco detection on proxy
        t4 = time.perf_counter()
        corners_list, ids = self._detect_markers(proxy_blur)
        aruco_detect_ms = round((time.perf_counter() - t4) * 1000.0, 3)

        if ids is None or len(ids) == 0:
            raise MarkerNotFoundError(f"Primary marker ID {self.MARKER_0_ID} not detected")

        id_to_proxy_corners: Dict[int, np.ndarray] = {}
        for c, mid in zip(corners_list, ids.flatten()):
            id_to_proxy_corners[int(mid)] = c[0]

        if self.MARKER_0_ID not in id_to_proxy_corners:
            raise MarkerNotFoundError(f"Primary marker ID {self.MARKER_0_ID} not detected")

        # Scale corners back to full resolution
        sx = orig_w / float(proxy_w)
        sy = orig_h / float(proxy_h)

        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        def refine_marker_corners(proxy_pts: np.ndarray) -> np.ndarray:
            full_pts = proxy_pts.copy()
            full_pts[:, 0] *= sx
            full_pts[:, 1] *= sy

            # Bounds-Checked cornerSubPix (M-15): Check 8 px margin inside image bounds
            inside_margin = all(
                8.0 <= pt[0] <= (orig_w - 9.0) and 8.0 <= pt[1] <= (orig_h - 9.0)
                for pt in full_pts
            )
            if inside_margin:
                refined = cv2.cornerSubPix(
                    gray_full,
                    full_pts.reshape(-1, 1, 2).astype(np.float32),
                    (5, 5),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05),
                )
                return refined.reshape(4, 2)
            return full_pts.astype(np.float32)

        corners_0 = refine_marker_corners(id_to_proxy_corners[self.MARKER_0_ID])

        # Edge lengths & metric computation on full resolution
        e01 = float(np.linalg.norm(corners_0[1] - corners_0[0]))
        e12 = float(np.linalg.norm(corners_0[2] - corners_0[1]))
        e23 = float(np.linalg.norm(corners_0[3] - corners_0[2]))
        e30 = float(np.linalg.norm(corners_0[0] - corners_0[3]))
        edges = [e01, e12, e23, e30]
        mean_marker0_edge_full_px = float(np.mean(edges))
        min_edge_px = float(min(edges))
        max_edge_px = float(max(edges))

        native_px_per_mm = round(mean_marker0_edge_full_px / self.MARKER_SIZE_MM, 2)
        native_mm_per_px = round(self.MARKER_SIZE_MM / max(1e-6, mean_marker0_edge_full_px), 4)
        tilt_quality_ratio = round(min_edge_px / max(1.0, max_edge_px), 3)

        # Dual-Marker 8-Point vs Single-Marker 4-Point Homography (M-12, D3)
        rectification_mode = "SINGLE_MARKER_4PT"
        H_metric = None

        if self.MARKER_1_ID in id_to_proxy_corners:
            corners_1 = refine_marker_corners(id_to_proxy_corners[self.MARKER_1_ID])
            src_8 = np.vstack([corners_0, corners_1]).astype(np.float32)
            dst_8 = np.vstack([self.dst_0, self.dst_1]).astype(np.float32)
            method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
            H_8pt, _ = cv2.findHomography(src_8, dst_8, method=method, ransacReprojThreshold=2.0)
            if H_8pt is not None:
                H_metric = H_8pt
                rectification_mode = "DUAL_MARKER_8PT"

        if H_metric is None:
            H_metric, _ = cv2.findHomography(corners_0.astype(np.float32), self.dst_0, method=0)
            rectification_mode = "SINGLE_MARKER_4PT"

        if H_metric is None:
            raise RuntimeError("Failed to compute perspective homography from marker corners")

        # Warp perspective
        t_warp = time.perf_counter()
        rectified_bgr = cv2.warpPerspective(
            frame_bgr,
            H_metric,
            (self.CANVAS_W, self.CANVAS_H),
            flags=cv2.INTER_LINEAR,
        )
        warp_perspective_ms = round((time.perf_counter() - t_warp) * 1000.0, 3)

        # Optional M-23: NBV Pose Prediction via solvePnP
        approx_roll_deg = None
        approx_pitch_deg = None
        if opt.get("enable_nbv_pose_prediction", False):
            try:
                half_s = self.MARKER_SIZE_MM / 2.0
                obj_pts = np.array(
                    [
                        [-half_s, half_s, 0.0],
                        [half_s, half_s, 0.0],
                        [half_s, -half_s, 0.0],
                        [-half_s, -half_s, 0.0],
                    ],
                    dtype=np.float32,
                )
                f_focal = float(max(orig_w, orig_h))
                k_cam = np.array(
                    [[f_focal, 0.0, orig_w / 2.0], [0.0, f_focal, orig_h / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
                succ, rvec, _ = cv2.solvePnP(
                    obj_pts,
                    corners_0.astype(np.float32),
                    k_cam,
                    dist_coeffs,
                    flags=pnp_flag,
                )
                if succ:
                    rmat, _ = cv2.Rodrigues(rvec)
                    roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
                    pitch = math.degrees(
                        math.atan2(-rmat[2, 0], math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
                    )
                    approx_roll_deg = round(roll, 2)
                    approx_pitch_deg = round(pitch, 2)
            except Exception:
                approx_roll_deg = None
                approx_pitch_deg = None

        metadata = {
            "resize_ms": resize_ms,
            "cvt_color_ms": cvt_color_ms,
            "gaussian_blur_ms": gaussian_blur_ms,
            "aruco_detect_ms": aruco_detect_ms,
            "warp_perspective_ms": warp_perspective_ms,
            "rectification_mode": rectification_mode,
            "native_px_per_mm": native_px_per_mm,
            "native_mm_per_px": native_mm_per_px,
            "tilt_quality_ratio": tilt_quality_ratio,
            "approx_roll_deg": approx_roll_deg,
            "approx_pitch_deg": approx_pitch_deg,
            "orig_dimensions": [orig_w, orig_h],
        }
        return rectified_bgr, metadata

    def compute_glare_mask(
        self,
        rectified_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        optional_flags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """Compute two-stage hysteresis glare mask and glare-excluded sharpness (M-26, M-24, M-23)."""
        opt = optional_flags if optional_flags is not None else self.optional_features
        rx, ry, rw, rh = roi if roi is not None else self.DEFAULT_ROI

        # Bounds validation
        rx = max(0, min(self.CANVAS_W - 1, rx))
        ry = max(0, min(self.CANVAS_H - 1, ry))
        rw = max(1, min(self.CANVAS_W - rx, rw))
        rh = max(1, min(self.CANVAS_H - ry, rh))

        roi_bgr = rectified_bgr[ry : ry + rh, rx : rx + rw]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1]
        v = hsv[..., 2]

        core_val_thresh = int(self.glare_cfg.get("core_val_thresh", 240))
        core_sat_thresh = int(self.glare_cfg.get("core_sat_thresh", 35))
        penumbra_min_val = int(self.glare_cfg.get("penumbra_min_val", 210))
        penumbra_median_offset = int(self.glare_cfg.get("penumbra_median_offset", 35))
        penumbra_sat_thresh = int(self.glare_cfg.get("penumbra_sat_thresh", 50))
        seed_dilation_px = int(self.glare_cfg.get("seed_dilation_px", 21))
        final_dilation_px = int(self.glare_cfg.get("final_dilation_px", 9))

        core_seed = (v >= core_val_thresh) & (s <= core_sat_thresh)

        if np.count_nonzero(core_seed) == 0:
            mask_dilated = np.zeros((rh, rw), dtype=np.uint8)
            glare_pct = 0.0
        else:
            median_v = float(np.median(v))
            penumbra_val_cutoff = max(float(penumbra_min_val), median_v + penumbra_median_offset)
            penumbra_candidates = (v >= penumbra_val_cutoff) & (s <= penumbra_sat_thresh)

            k_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_dilation_px, seed_dilation_px))
            core_dilated = cv2.dilate(core_seed.astype(np.uint8) * 255, k_seed)
            penumbra_connected = (core_dilated > 0) & penumbra_candidates

            full_seed = (core_seed | penumbra_connected).astype(np.uint8) * 255
            k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_dilation_px, final_dilation_px))
            mask_dilated = cv2.dilate(full_seed, k_final)
            glare_pct = round((float(np.count_nonzero(mask_dilated)) / float(rw * rh)) * 100.0, 3)

        # Sobel gradient Shannon entropy
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        hist, _ = np.histogram(mag, bins=64, range=(0, 255))
        probs = hist.astype(np.float64) / max(1.0, float(np.sum(hist)))
        probs = probs[probs > 0]
        entropy_bits = round(float(-np.sum(probs * np.log2(probs))), 3)

        # Glare-excluded sharpness (M-24)
        lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
        glare_free_pixels = lap[mask_dilated == 0]
        if len(glare_free_pixels) > 0:
            glare_free_sharpness = round(float(np.var(glare_free_pixels)), 3)
        else:
            glare_free_sharpness = round(float(np.var(lap)), 3)

        # Optional M-23: Centroid & NBV instruction
        glare_centroid_xy = None
        predicted_nbv_instruction = None
        if opt.get("enable_nbv_pose_prediction", False) and glare_pct > 0:
            m = cv2.moments(mask_dilated)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                glare_centroid_xy = [round(float(cx), 1), round(float(cy), 1)]
                dx = cx - (rw / 2.0)
                dy = cy - (rh / 2.0)
                h_dir = "left" if dx > 0 else "right"
                v_dir = "up" if dy > 0 else "down"
                predicted_nbv_instruction = f"Tilt phone {h_dir} by ~5 deg and {v_dir} by ~5 deg"

        metadata = {
            "glare_free_sharpness": glare_free_sharpness,
            "glare_centroid_xy": glare_centroid_xy,
            "predicted_nbv_instruction": predicted_nbv_instruction,
            "roi": [rx, ry, rw, rh],
        }
        return mask_dilated, glare_pct, entropy_bits, metadata

    def heal_accumulated_canvas(
        self,
        acc_canvas: np.ndarray,
        unhealed_roi_mask: np.ndarray,
        rect_alt: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        optional_flags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, Dict[str, Any]]:
        """Seamlessly blend non-glared alternative pixels into accumulated canvas (M-11, M-12, M-22, M-25, M-27, M-28)."""
        opt = optional_flags if optional_flags is not None else self.optional_features
        rx, ry, rw, rh = roi if roi is not None else self.DEFAULT_ROI

        # 1. Build ORB exclusion mask
        orb_mask = np.full((self.CANVAS_H, self.CANVAS_W), 255, dtype=np.uint8)
        # Exclude Marker 0 + margin
        orb_mask[180:620, 80:520] = 0
        # Exclude Marker 1 + margin
        orb_mask[180:620, 1480:1920] = 0
        # Exclude currently unhealed glare region inside ROI
        orb_mask[ry : ry + rh, rx : rx + rw][unhealed_roi_mask > 0] = 0

        # 2. Extract ORB features outside markers and glare
        nfeatures = int(self.healing_cfg.get("orb_nfeatures", 1500))
        orb = cv2.ORB_create(nfeatures=nfeatures)
        gray_acc = cv2.cvtColor(acc_canvas, cv2.COLOR_BGR2GRAY)
        gray_alt = cv2.cvtColor(rect_alt, cv2.COLOR_BGR2GRAY)

        kp_acc, des_acc = orb.detectAndCompute(gray_acc, mask=orb_mask)
        kp_alt, des_alt = orb.detectAndCompute(gray_alt, mask=orb_mask)

        inliers = 0
        aligned_alt = rect_alt.copy()
        alignment_mode = "ARUCO_METRIC_ONLY"

        min_inliers = int(self.healing_cfg.get("min_non_marker_inliers", 10))
        reproj_thresh = float(self.healing_cfg.get("usac_reproj_thresh_px", 1.5))

        if (
            des_acc is not None
            and des_alt is not None
            and len(des_acc) >= min_inliers
            and len(des_alt) >= min_inliers
        ):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des_alt, des_acc)
            if len(matches) >= min_inliers:
                src_pts = np.float32([kp_alt[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_acc[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                H_refine, inlier_mask = cv2.findHomography(
                    src_pts, dst_pts, method=method, ransacReprojThreshold=reproj_thresh
                )
                if H_refine is not None and inlier_mask is not None:
                    count_inliers = int(np.sum(inlier_mask))
                    if count_inliers >= min_inliers:
                        inliers = count_inliers
                        aligned_alt = cv2.warpPerspective(
                            rect_alt,
                            H_refine,
                            (self.CANVAS_W, self.CANVAS_H),
                            flags=cv2.INTER_LINEAR,
                        )
                        alignment_mode = "ARUCO_PLUS_USAC_MAGSAC_REFINEMENT"

        # Optional M-27: Euclidean ECC subpixel refinement
        if opt.get("enable_ecc_subpixel_refinement", False):
            try:
                gray_al_alt = cv2.cvtColor(aligned_alt, cv2.COLOR_BGR2GRAY)
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.001)
                succ, warp_matrix = cv2.findTransformECC(
                    gray_acc[ry : ry + rh, rx : rx + rw],
                    gray_al_alt[ry : ry + rh, rx : rx + rw],
                    warp_matrix,
                    cv2.MOTION_EUCLIDEAN,
                    criteria,
                )
                if succ:
                    dx = warp_matrix[0, 2]
                    dy = warp_matrix[1, 2]
                    rot_deg = abs(math.degrees(math.atan2(warp_matrix[1, 0], warp_matrix[0, 0])))
                    if math.hypot(dx, dy) <= 8.0 and rot_deg <= 2.0:
                        aligned_alt = cv2.warpAffine(
                            aligned_alt,
                            warp_matrix,
                            (self.CANVAS_W, self.CANVAS_H),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                        )
            except Exception:
                pass

        # 3. Compute glare mask on aligned alternative image
        mask_alt_aligned, _, _, _ = self.compute_glare_mask(aligned_alt, roi=roi, optional_flags=optional_flags)

        newly_healable = (unhealed_roi_mask > 0) & (mask_alt_aligned == 0)
        still_unhealed = ((unhealed_roi_mask > 0) & (mask_alt_aligned > 0)).astype(np.uint8) * 255

        roi_cur = acc_canvas[ry : ry + rh, rx : rx + rw].copy()
        roi_alt = aligned_alt[ry : ry + rh, rx : rx + rw].copy()

        mutually_valid = (unhealed_roi_mask == 0) & (mask_alt_aligned == 0)
        n_mutually_valid = int(np.count_nonzero(mutually_valid))

        # 4. Radiometric Gain Matching (M-22, M-25)
        roi_alt_gain = roi_alt.astype(np.float32)

        if not opt.get("enable_2d_illumination_field_gain", False):
            if n_mutually_valid >= 200:
                clip_min = float(self.healing_cfg.get("gain_clip_min", 0.60))
                clip_max = float(self.healing_cfg.get("gain_clip_max", 1.60))
                for c in range(3):
                    cur_mean = float(np.mean(roi_cur[..., c][mutually_valid]))
                    alt_mean = max(1.0, float(np.mean(roi_alt[..., c][mutually_valid])))
                    gain_c = float(np.clip(cur_mean / alt_mean, clip_min, clip_max))
                    roi_alt_gain[..., c] = roi_alt[..., c].astype(np.float32) * gain_c
            else:
                # Border median fallback (M-22)
                bw = min(20, rh // 4, rw // 4)
                border_mask = np.zeros((rh, rw), dtype=bool)
                border_mask[:bw, :] = True
                border_mask[-bw:, :] = True
                border_mask[:, :bw] = True
                border_mask[:, -bw:] = True

                clip_min = float(self.healing_cfg.get("low_overlap_clip_min", 0.75))
                clip_max = float(self.healing_cfg.get("low_overlap_clip_max", 1.35))
                for c in range(3):
                    cur_med = float(np.median(roi_cur[..., c][border_mask]))
                    alt_med = max(1.0, float(np.median(roi_alt[..., c][border_mask])))
                    gain_c = float(np.clip(cur_med / alt_med, clip_min, clip_max))
                    roi_alt_gain[..., c] = roi_alt[..., c].astype(np.float32) * gain_c
        else:
            # 2D illumination field gain (M-25)
            cur_f = roi_cur.astype(np.float32)
            alt_f = roi_alt.astype(np.float32)
            blur_cur = cv2.GaussianBlur(cur_f, (101, 101), 35)
            blur_alt = np.maximum(1.0, cv2.GaussianBlur(alt_f, (101, 101), 35))
            gain_2d = np.clip(blur_cur / blur_alt, 0.60, 1.60)
            roi_alt_gain = alt_f * gain_2d

        roi_alt_gain = np.clip(roi_alt_gain, 0.0, 255.0)

        # 5. Dilated Feather Blending (M-11, FQ7)
        pre_feather_px = int(self.healing_cfg.get("pre_feather_dilation_px", 11))
        k_pfd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre_feather_px, pre_feather_px))
        nh_dilated = cv2.dilate((newly_healable.astype(np.uint8) * 255), k_pfd)
        nh_dilated = (nh_dilated > 0) & (mask_alt_aligned == 0)

        feather_k = int(self.healing_cfg.get("feather_kernel_px", 21))
        feather_sigma = float(self.healing_cfg.get("feather_sigma", 5.0))
        alpha = cv2.GaussianBlur(nh_dilated.astype(np.float32), (feather_k, feather_k), feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        blended_roi = (
            roi_cur.astype(np.float32) * (1.0 - alpha) + roi_alt_gain * alpha
        )
        new_acc_canvas = acc_canvas.copy()
        new_acc_canvas[ry : ry + rh, rx : rx + rw] = np.clip(blended_roi, 0.0, 255.0).astype(np.uint8)

        # Optional M-28: Seam gradient spike ratio
        seam_gradient_spike_ratio = 1.0
        if np.count_nonzero(newly_healable) > 0:
            k_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            seam_mask = cv2.morphologyEx(newly_healable.astype(np.uint8), cv2.MORPH_GRADIENT, k_grad)
            if np.count_nonzero(seam_mask) > 0:
                blended_gray = cv2.cvtColor(new_acc_canvas[ry : ry + rh, rx : rx + rw], cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(blended_gray, cv2.CV_64F, 1, 0)
                gy = cv2.Sobel(blended_gray, cv2.CV_64F, 0, 1)
                sobel_mag = cv2.magnitude(gx, gy)
                seam_grad = float(np.mean(sobel_mag[seam_mask > 0]))
                roi_grad = float(np.mean(sobel_mag))
                seam_gradient_spike_ratio = round(seam_grad / max(1.0, roi_grad), 3)

        residual_glare_pct = round((float(np.count_nonzero(still_unhealed)) / float(rw * rh)) * 100.0, 3)

        metadata = {
            "alignment_mode": alignment_mode,
            "non_marker_usac_inliers": inliers,
            "residual_glare_pct": residual_glare_pct,
            "mutually_valid_pixels": n_mutually_valid,
            "seam_gradient_spike_ratio": seam_gradient_spike_ratio,
        }
        return new_acc_canvas, still_unhealed, residual_glare_pct, inliers, metadata

    def extract_raw_candidate_contours(
        self,
        canvas: np.ndarray,
        exclude_mask: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        morph_intensity_thresh: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Extract candidate defect contours using single unrigged 61x61 BLACKHAT detector (M-13, D2)."""
        rx, ry, rw, rh = roi if roi is not None else self.DEFAULT_ROI
        thresh_val = (
            morph_intensity_thresh
            if morph_intensity_thresh is not None
            else int(self.detection_cfg.get("morph_intensity_thresh", 30))
        )
        margin = int(self.detection_cfg.get("roi_border_margin_px", 12))

        inner_x = rx + margin
        inner_y = ry + margin
        inner_w = max(1, rw - 2 * margin)
        inner_h = max(1, rh - 2 * margin)

        roi_inner_bgr = canvas[inner_y : inner_y + inner_h, inner_x : inner_x + inner_w]
        gray = cv2.cvtColor(roi_inner_bgr, cv2.COLOR_BGR2GRAY)

        k_blur = int(self.detection_cfg.get("median_blur_px", 5))
        denoised = cv2.medianBlur(gray, k_blur)

        k_morph = int(self.detection_cfg.get("morph_kernel_px", 61))
        kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel_blackhat)

        _, thresh = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

        if exclude_mask is not None:
            # Exclude mask must be cropped to inner ROI and dilated by 11 px
            inner_exclude = (exclude_mask[margin : margin + inner_h, margin : margin + inner_w] > 0).astype(np.uint8) * 255
            k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dilated_ex = cv2.dilate(inner_exclude, k_ex)
            thresh[dilated_ex > 0] = 0

        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area / max(1.0, hull_area))

            mask_cnt = np.zeros_like(blackhat)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            peak_contrast = float(np.max(blackhat[mask_cnt > 0])) if np.any(mask_cnt > 0) else 0.0

            bx, by, bw, bh = cv2.boundingRect(cnt)
            length_mm = round(max(bw, bh) * self.MM_PER_PX, 3)
            width_mm = round(min(bw, bh) * self.MM_PER_PX, 3)

            peri = float(cv2.arcLength(cnt, True))
            circularity = round(float(4.0 * math.pi * area / max(1.0, peri * peri)), 3)

            candidates.append(
                {
                    "area_px": area,
                    "solidity": round(solidity, 3),
                    "peak_contrast": round(peak_contrast, 1),
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                    "circularity": circularity,
                    "bbox_canonical": [inner_x + bx, inner_y + by, bw, bh],
                    "raw_contour": cnt,
                }
            )

        return candidates, blackhat

    def detect_defects(
        self,
        canvas: np.ndarray,
        exclude_mask: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        morph_intensity_thresh: Optional[int] = None,
        optional_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Detect foil defects using filtered blackhat contours (M-13, D2)."""
        t0 = time.perf_counter()
        candidates, blackhat = self.extract_raw_candidate_contours(
            canvas=canvas,
            exclude_mask=exclude_mask,
            roi=roi,
            morph_intensity_thresh=morph_intensity_thresh,
        )

        min_area = float(self.detection_cfg.get("min_area_px", 14.0))
        min_solidity = float(self.detection_cfg.get("min_solidity", 0.35))
        min_contrast = float(self.detection_cfg.get("min_peak_contrast", 42.0))
        min_defect_len = float(self.detection_cfg.get("min_defect_length_mm", 0.35))

        defects = []
        for c in candidates:
            if (
                c["area_px"] >= min_area
                and c["solidity"] >= min_solidity
                and c["peak_contrast"] >= min_contrast
                and c["length_mm"] >= min_defect_len
            ):
                defects.append(
                    {
                        "length_mm": c["length_mm"],
                        "width_mm": c["width_mm"],
                        "area_px": c["area_px"],
                        "solidity": c["solidity"],
                        "peak_contrast": c["peak_contrast"],
                        "circularity": c["circularity"],
                        "bbox_canonical": c["bbox_canonical"],
                    }
                )

        max_defect_length_mm = max([d["length_mm"] for d in defects], default=0.0)
        verdict = "REJECT" if max_defect_length_mm >= min_defect_len else "PASS"

        # Annotation
        annotated_canvas = canvas.copy()
        rx, ry, rw, rh = roi if roi is not None else self.DEFAULT_ROI
        box_color = (0, 0, 255) if verdict == "REJECT" else (0, 200, 0)
        cv2.rectangle(annotated_canvas, (rx, ry), (rx + rw, ry + rh), box_color, 2)

        for d in defects:
            bx, by, bw, bh = d["bbox_canonical"]
            cv2.rectangle(annotated_canvas, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            label = f"{d['length_mm']:.2f}mm (c={d['peak_contrast']:.0f})"
            cv2.putText(
                annotated_canvas,
                label,
                (bx, max(20, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        inspect_total_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "verdict": verdict,
            "defects": defects,
            "max_defect_length_mm": max_defect_length_mm,
            "annotated_canvas": annotated_canvas,
            "inspect_total_ms": inspect_total_ms,
        }
