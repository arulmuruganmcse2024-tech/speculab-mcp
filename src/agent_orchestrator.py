"""Agent orchestrator and Bedrock MCP active vision controller for SpecuLab-MCP.

Provides:
- GlobalQuotaLimiter: 3 isolated quota buckets, atomic locking, HMAC judge verification.
- CloudWatchAuditLogger: Human supervisor quarantine approval & audit logging.
- BedrockMCPAgent: MCP active vision orchestrator with tool execution, caching,
  and premature finalize guards.
"""

import copy
import hmac
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
import cv2
import numpy as np
import yaml

from src.vision_engine import (
    MarkerNotFoundError,
    PlanarActiveVisionEngine,
    get_opencv_runtime_metadata,
    load_image_exif_safe,
)

# Cross-platform file locking imports
fcntl = None
if sys.platform != "win32":
    try:
        import fcntl  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        fcntl = None


class GlobalQuotaLimiter:
    """Multi-bucket rate limiter with atomic locking and tamper-proof quotas."""

    QUOTA_FILE = "logs/global_llm_quota.json"
    LOCK_FILE = "logs/global_llm_quota.lock"

    BUCKET_LIMITS = {
        "public_anonymous": {"per_hour": 15, "per_day": 50},
        "judge_vip": {"per_hour": 100, "per_day": 300},
        "cli_evaluation": {"per_hour": 100, "per_day": 300},
    }

    def __init__(self, quota_file: Optional[str] = None):
        if quota_file:
            self.quota_file = quota_file
            self.lock_file = quota_file + ".lock"
        else:
            self.quota_file = self.QUOTA_FILE
            self.lock_file = self.LOCK_FILE

        os.makedirs(os.path.dirname(os.path.abspath(self.quota_file)), exist_ok=True)
        if not os.path.exists(self.quota_file):
            self._write_state({k: [] for k in self.BUCKET_LIMITS})

    def _acquire_lock(self):
        """Acquire lock safely on POSIX (fcntl) or Windows (atomic lockfile)."""
        if fcntl is not None:
            f = open(self.lock_file, "w")
            fcntl.flock(f, fcntl.LOCK_EX)
            return f
        else:
            # Atomic lockfile retry loop on Windows
            start = time.time()
            while time.time() - start < 5.0:
                try:
                    fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    return fd
                except FileExistsError:
                    time.sleep(0.05)
            # Stale lock fallback
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
            return os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)

    def _release_lock(self, lock_handle):
        """Release lock handle."""
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
            except Exception:
                pass
        else:
            try:
                os.close(lock_handle)
                os.remove(self.lock_file)
            except Exception:
                pass

    def _read_state(self) -> Dict[str, List[float]]:
        try:
            with open(self.quota_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {k: [] for k in self.BUCKET_LIMITS}

    def _write_state(self, state: Dict[str, List[float]]) -> None:
        tmp_path = self.quota_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        if os.path.exists(self.quota_file):
            try:
                os.replace(tmp_path, self.quota_file)
            except OSError:
                os.remove(self.quota_file)
                os.rename(tmp_path, self.quota_file)
        else:
            os.rename(tmp_path, self.quota_file)

    @staticmethod
    def verify_judge_passcode(provided: str) -> bool:
        """Verify judge VIP passcode against SPECULAB_JUDGE_PASSCODE environment variable."""
        expected = os.environ.get("SPECULAB_JUDGE_PASSCODE", "")
        if len(expected) < 8 or not provided:
            return False
        return hmac.compare_digest(provided.strip().encode("utf-8"), expected.strip().encode("utf-8"))

    def check_and_consume(self, bucket: str) -> Tuple[bool, str]:
        """Check quota and consume 1 token atomically. Returns (allowed, reason)."""
        if bucket not in self.BUCKET_LIMITS:
            return False, f"Unknown quota bucket: {bucket}"

        lock = self._acquire_lock()
        try:
            state = self._read_state()
            now = time.time()
            one_hour_ago = now - 3600.0
            one_day_ago = now - 86400.0

            timestamps = state.get(bucket, [])
            timestamps_24h = [t for t in timestamps if t >= one_day_ago]
            timestamps_1h = [t for t in timestamps_24h if t >= one_hour_ago]

            limits = self.BUCKET_LIMITS[bucket]
            if len(timestamps_1h) >= limits["per_hour"]:
                state[bucket] = timestamps_24h
                self._write_state(state)
                return False, f"Bucket '{bucket}' hourly limit reached ({len(timestamps_1h)}/{limits['per_hour']})."

            if len(timestamps_24h) >= limits["per_day"]:
                state[bucket] = timestamps_24h
                self._write_state(state)
                return False, f"Bucket '{bucket}' daily limit reached ({len(timestamps_24h)}/{limits['per_day']})."

            timestamps_24h.append(now)
            state[bucket] = timestamps_24h
            self._write_state(state)
            return True, "OK"
        finally:
            self._release_lock(lock)


class CloudWatchAuditLogger:
    """Logs human-in-the-loop quarantine approvals and audit events to CloudWatch + local JSONL."""

    LOG_GROUP = "/aws/opencv2026/speculab-mcp"
    AUDIT_LOG_FILE = "logs/audit_events.jsonl"

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.stream_name = f"supervisor-audit-{time.strftime('%Y%m%d')}"
        os.makedirs(os.path.dirname(os.path.abspath(self.AUDIT_LOG_FILE)), exist_ok=True)
        try:
            self.client = boto3.client(
                "logs",
                region_name=self.region,
                config=Config(connect_timeout=1.0, read_timeout=1.5, retries={"max_attempts": 1}),
            )
        except Exception:
            self.client = None

    def log_approval_event(self, event_data: Dict[str, Any]) -> None:
        """Log quarantine approval event locally and push to CloudWatch."""
        event = copy.deepcopy(event_data)
        event["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        event["timestamp_ms"] = int(time.time() * 1000)

        # Append to durable local JSONL audit trail
        with open(self.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        # Push to AWS CloudWatch if client available
        if self.client is not None:
            try:
                try:
                    self.client.create_log_stream(
                        logGroupName=self.LOG_GROUP,
                        logStreamName=self.stream_name,
                    )
                except Exception:
                    pass

                self.client.put_log_events(
                    logGroupName=self.LOG_GROUP,
                    logStreamName=self.stream_name,
                    logEvents=[
                        {
                            "timestamp": event["timestamp_ms"],
                            "message": json.dumps(event),
                        }
                    ],
                )
            except Exception:
                pass


class BedrockMCPAgent:
    """Bedrock Model Context Protocol orchestrator with active multi-view fusion."""

    MAX_VIEW_ATTEMPTS: int = 2

    def __init__(
        self,
        config_path: str = "config.yaml",
        engine: Optional[PlanarActiveVisionEngine] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.engine = engine if engine is not None else PlanarActiveVisionEngine(config_path)
        self.quota_limiter = GlobalQuotaLimiter()
        self.audit_logger = CloudWatchAuditLogger()

        agent_cfg = self.config.get("agent", {})
        self.default_region: str = agent_cfg.get("default_region", "us-east-1")
        self.default_model_id: str = agent_cfg.get("default_model_id", "us.amazon.nova-lite-v1:0")

        # Unverified pricing estimates (Amendment 5)
        self.cost_per_1m_in: float = float(agent_cfg.get("input_token_cost_per_1m_usd", 0.06))
        self.cost_per_1m_out: float = float(agent_cfg.get("output_token_cost_per_1m_usd", 0.24))

        # Bedrock runtime client with mandatory read_timeout >= 10.0 (M-01)
        self.bedrock_config = Config(
            connect_timeout=3.0,
            read_timeout=12.0,
            retries={"max_attempts": 1},
        )
        try:
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.default_region,
                config=self.bedrock_config,
            )
        except Exception:
            self.bedrock_client = None

        # Tools definition for Bedrock Converse API (D6: No agent threshold-tuning parameter)
        self.tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "rank_candidate_views",
                        "description": (
                            "Ranks candidate alternative views based on unhealed overlap, "
                            "glare percentage, tilt quality, and Laplacian sharpness gate."
                        ),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "candidate_paths": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of candidate file paths to evaluate.",
                                    }
                                },
                                "required": ["candidate_paths"],
                            }
                        },
                    }
                },
                {
                    "toolSpec": {
                        "name": "fuse_candidate_view",
                        "description": (
                            "Fuses an alternative candidate view into the canvas to heal specular glare. "
                            "Enforces hard cap of MAX_VIEW_ATTEMPTS = 2."
                        ),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "candidate_path": {
                                        "type": "string",
                                        "description": "File path of the candidate view to fuse.",
                                    },
                                    "rollback_previous_fusion": {
                                        "type": "boolean",
                                        "description": "Roll back the previous fusion if seam gradient spiked.",
                                    },
                                },
                                "required": ["candidate_path"],
                            }
                        },
                    }
                },
                {
                    "toolSpec": {
                        "name": "finalize_inspection_or_escalate",
                        "description": (
                            "Finalize inspection by running defect metrology or escalate to human supervisor "
                            "if residual glare remains occluding the foil surface."
                        ),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "decision_note": {
                                        "type": "string",
                                        "description": "Reasoning summary for finalization or escalation.",
                                    }
                                },
                                "required": ["decision_note"],
                            }
                        },
                    }
                },
            ]
        }

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tool call on internal inspection state."""
        roi = state["roi"]
        gate = state["glare_gate_pct"]
        opt = state["optional_flags"]

        if tool_name == "rank_candidate_views":
            candidate_paths = tool_args.get("candidate_paths", [])
            allowed_map = state.get("allowed_map", {})
            unusable_candidates = []
            rejected_paths = []
            usable_candidates = []

            ref_sharpness = state.get("ref_sharpness", 1.0)
            unhealed_mask = state["unhealed_mask"]
            total_unhealed_px = max(1, int(np.count_nonzero(unhealed_mask > 0)))

            for p in candidate_paths:
                p_norm = os.path.abspath(p)
                if allowed_map and p_norm not in allowed_map:
                    rejected_paths.append(p)
                    continue

                try:
                    img = load_image_exif_safe(p)
                    rect_cand, meta_cand = self.engine.rectify_to_canonical_plane(img, optional_flags=opt)
                    mask_cand, cand_glare_pct, _, cand_meta = self.engine.compute_glare_mask(
                        rect_cand, roi=roi, optional_flags=opt
                    )

                    # Overlap with unhealed glare region
                    cand_glare_free_in_roi = mask_cand == 0
                    overlap_px = int(np.count_nonzero((unhealed_mask > 0) & cand_glare_free_in_roi))
                    overlap_pct = round((overlap_px / float(total_unhealed_px)) * 100.0, 2)

                    cand_sharp = cand_meta.get("glare_free_sharpness", 0.0)
                    rel_sharp = round(cand_sharp / max(1.0, ref_sharpness), 3)
                    min_ratio = float(self.config.get("sharpness", {}).get("min_relative_sharpness_ratio", 0.45))
                    blur_degraded = bool(rel_sharp < min_ratio)

                    usable_candidates.append(
                        {
                            "path": p,
                            "rectified_bgr": rect_cand,
                            "overlap_with_unhealed_pct": overlap_pct,
                            "candidate_glare_pct": cand_glare_pct,
                            "tilt_quality_ratio": meta_cand.get("tilt_quality_ratio", 1.0),
                            "native_px_per_mm": meta_cand.get("native_px_per_mm", 20.0),
                            "glare_free_sharpness": cand_sharp,
                            "relative_sharpness_ratio": rel_sharp,
                            "blur_degraded": blur_degraded,
                        }
                    )
                except (MarkerNotFoundError, ValueError, Exception) as e:
                    unusable_candidates.append({"path": p, "error": str(e)})

            # Sort: non-blurry first, highest overlap, lowest glare, highest tilt quality, highest resolution
            usable_candidates.sort(
                key=lambda c: (
                    -int(not c["blur_degraded"]),
                    -c["overlap_with_unhealed_pct"],
                    c["candidate_glare_pct"],
                    -c["tilt_quality_ratio"],
                    -c["native_px_per_mm"],
                )
            )

            state["ranked_candidates"] = usable_candidates
            state["unusable_candidates"] = unusable_candidates
            state["rejected_paths"] = rejected_paths

            ranked_summary = [
                {
                    "path": c["path"],
                    "overlap_with_unhealed_pct": c["overlap_with_unhealed_pct"],
                    "candidate_glare_pct": c["candidate_glare_pct"],
                    "blur_degraded": c["blur_degraded"],
                    "relative_sharpness_ratio": c["relative_sharpness_ratio"],
                    "tilt_quality_ratio": c["tilt_quality_ratio"],
                }
                for c in usable_candidates
            ]
            return {
                "ranked_candidates": ranked_summary,
                "unusable_candidates": unusable_candidates,
                "rejected_non_whitelisted_paths": rejected_paths,
            }

        elif tool_name == "fuse_candidate_view":
            candidate_path = tool_args.get("candidate_path")
            rollback = bool(tool_args.get("rollback_previous_fusion", False))

            if (
                rollback
                and opt.get("enable_seam_rollback", False)
                and state.get("prev_snapshot") is not None
            ):
                snap = state["prev_snapshot"]
                state["acc_canvas"] = snap["acc_canvas"].copy()
                state["unhealed_mask"] = snap["unhealed_mask"].copy()
                state["residual_glare_pct"] = snap["residual_glare_pct"]

            if state["view_attempts"] >= self.MAX_VIEW_ATTEMPTS:
                return {
                    "error": f"Hard cap MAX_VIEW_ATTEMPTS = {self.MAX_VIEW_ATTEMPTS} reached",
                    "view_attempts_used": state["view_attempts"],
                    "max_view_attempts": self.MAX_VIEW_ATTEMPTS,
                    "residual_glare_pct": state["residual_glare_pct"],
                    "glare_cleared_below_gate": state["residual_glare_pct"] <= gate,
                    "recommended_next_step": "finalize_inspection_or_escalate",
                }

            # Find rectified alternative view
            rect_alt = None
            for c in state.get("ranked_candidates", []):
                if c["path"] == candidate_path:
                    rect_alt = c["rectified_bgr"]
                    break

            if rect_alt is None:
                # Load on demand if not in cache
                try:
                    img_alt = load_image_exif_safe(candidate_path)
                    if img_alt is None:
                        raise ValueError(f"Could not load image at {candidate_path}")
                    rect_alt, _ = self.engine.rectify_to_canonical_plane(img_alt, optional_flags=opt)
                except (MarkerNotFoundError, ValueError, Exception) as e:
                    state["view_attempts"] += 1
                    return {
                        "error": f"Candidate view {candidate_path} unusable: {e}",
                        "view_attempts_used": state["view_attempts"],
                        "max_view_attempts": self.MAX_VIEW_ATTEMPTS,
                        "residual_glare_pct": state["residual_glare_pct"],
                        "glare_cleared_below_gate": state["residual_glare_pct"] <= gate,
                        "recommended_next_step": "finalize_inspection_or_escalate",
                    }

            # Snapshot state before fusion
            state["prev_snapshot"] = {
                "acc_canvas": state["acc_canvas"].copy(),
                "unhealed_mask": state["unhealed_mask"].copy(),
                "residual_glare_pct": state["residual_glare_pct"],
            }
            state["view_attempts"] += 1

            new_canvas, still_unhealed, residual_glare, inliers, heal_meta = self.engine.heal_accumulated_canvas(
                state["acc_canvas"],
                state["unhealed_mask"],
                rect_alt,
                roi=roi,
                optional_flags=opt,
            )

            state["acc_canvas"] = new_canvas
            state["unhealed_mask"] = still_unhealed
            state["residual_glare_pct"] = residual_glare

            cleared = bool(residual_glare <= gate)
            next_step = (
                "finalize_inspection_or_escalate"
                if (cleared or state["view_attempts"] >= self.MAX_VIEW_ATTEMPTS)
                else "fuse_candidate_view"
            )

            return {
                "fused_view": candidate_path,
                "view_attempts_used": state["view_attempts"],
                "max_view_attempts": self.MAX_VIEW_ATTEMPTS,
                "non_marker_usac_inliers": inliers,
                "alignment_mode": heal_meta.get("alignment_mode", "UNKNOWN"),
                "residual_glare_pct": residual_glare,
                "glare_cleared_below_gate": cleared,
                "recommended_next_step": next_step,
            }

        elif tool_name == "finalize_inspection_or_escalate":
            decision_note = tool_args.get("decision_note", "Standard completion")

            # Premature Finalize Guard (M-02)
            if state["residual_glare_pct"] > gate and state["view_attempts"] < self.MAX_VIEW_ATTEMPTS:
                if not state.get("ranked_candidates"):
                    all_cands = [p for p in state.get("candidate_paths", [])[1:]]
                    self._execute_tool("rank_candidate_views", {"candidate_paths": all_cands}, state)

                used_fusions = state.get("fused_paths", set())
                for cand in state.get("ranked_candidates", []):
                    cpath = cand["path"]
                    if cpath not in used_fusions and state["view_attempts"] < self.MAX_VIEW_ATTEMPTS:
                        fuse_res = self._execute_tool("fuse_candidate_view", {"candidate_path": cpath}, state)
                        used_fusions.add(cpath)
                        state["fused_paths"] = used_fusions
                        state["trace"].append(
                            {
                                "timeline_phase": "Premature Finalize Guard Auto-Fuse (M-02)",
                                "auto_fused_candidate": cpath,
                                "residual_glare_pct": fuse_res["residual_glare_pct"],
                            }
                        )
                        if state["residual_glare_pct"] <= gate:
                            break

            # Outcome evaluation
            if state["residual_glare_pct"] > gate:
                return {
                    "status": "UNRESOLVED_OCCLUSION",
                    "verdict": "ESCALATE_UNRESOLVED_OCCLUSION",
                    "requires_human_approval": True,
                    "residual_glare_pct": state["residual_glare_pct"],
                    "decision_note": decision_note,
                    "view_attempts_used": state["view_attempts"],
                }
            else:
                det = self.engine.detect_defects(
                    state["acc_canvas"],
                    exclude_mask=state["unhealed_mask"],
                    roi=roi,
                    morph_intensity_thresh=state["morph_thresh"],
                    optional_flags=opt,
                )
                return {
                    "status": "HEALED_AND_INSPECTED",
                    "verdict": det["verdict"],
                    "max_defect_length_mm": det["max_defect_length_mm"],
                    "defects": det["defects"],
                    "requires_human_approval": bool(det["verdict"] == "REJECT"),
                    "decision_note": decision_note,
                    "view_attempts_used": state["view_attempts"],
                }
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

    def run_inspection_cycle(
        self,
        candidate_paths: List[str],
        policy_mode: str = "bedrock_llm",
        quota_bucket: str = "public_anonymous",
        judge_passcode: Optional[str] = None,
        allow_cached_replay: bool = True,
        specimen_id: Optional[str] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        glare_gate_pct: Optional[float] = None,
        morph_intensity_thresh: Optional[int] = None,
        optional_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a full active vision inspection cycle across candidate views."""
        cycle_t0 = time.perf_counter()

        # Resolve parameters without mutating self (M-20)
        resolved_roi = roi if roi is not None else self.engine.DEFAULT_ROI
        resolved_gate = (
            glare_gate_pct
            if glare_gate_pct is not None
            else float(self.engine.glare_cfg.get("default_gate_pct", 2.0))
        )
        resolved_thresh = (
            morph_intensity_thresh
            if morph_intensity_thresh is not None
            else int(self.engine.detection_cfg.get("morph_intensity_thresh", 30))
        )
        resolved_opt = (
            optional_flags if optional_flags is not None else self.engine.optional_features.copy()
        )

        if not candidate_paths:
            raise ValueError("Candidate paths list is empty")

        view0_path = candidate_paths[0]
        view0_img = load_image_exif_safe(view0_path)

        # 1. Rectify View 0 to canonical 2000x800 plane
        rect_view0, calib_meta = self.engine.rectify_to_canonical_plane(
            view0_img, optional_flags=resolved_opt
        )

        # 2. Pass 1 Glare & Sharpness
        mask_0, glare_pct_0, entropy_0, glare_meta_0 = self.engine.compute_glare_mask(
            rect_view0, roi=resolved_roi, optional_flags=resolved_opt
        )

        # 3. Two baseline passive evaluations (unmasked vs glare-masked)
        pas_unmasked = self.engine.detect_defects(
            rect_view0,
            exclude_mask=None,
            roi=resolved_roi,
            morph_intensity_thresh=resolved_thresh,
            optional_flags=resolved_opt,
        )
        pas_masked = self.engine.detect_defects(
            rect_view0,
            exclude_mask=mask_0,
            roi=resolved_roi,
            morph_intensity_thresh=resolved_thresh,
            optional_flags=resolved_opt,
        )

        # 4. Active trigger condition (M-06)
        glare_disagreement = bool(
            (pas_unmasked["verdict"] != pas_masked["verdict"]) and (glare_pct_0 > 0.2)
        )
        trigger_active = bool((glare_pct_0 > resolved_gate) or glare_disagreement)

        # Allowed whitelist map
        allowed_map = {os.path.abspath(p): p for p in candidate_paths}

        state: Dict[str, Any] = {
            "candidate_paths": candidate_paths,
            "allowed_map": allowed_map,
            "roi": resolved_roi,
            "glare_gate_pct": resolved_gate,
            "morph_thresh": resolved_thresh,
            "optional_flags": resolved_opt,
            "acc_canvas": rect_view0.copy(),
            "unhealed_mask": mask_0.copy(),
            "residual_glare_pct": glare_pct_0,
            "view_attempts": 0,
            "ranked_candidates": [],
            "unusable_candidates": [],
            "rejected_paths": [],
            "fused_paths": set(),
            "ref_sharpness": glare_meta_0.get("glare_free_sharpness", 1.0),
            "trace": [],
        }

        # Step 1 in Timeline Trace
        state["trace"].append(
            {
                "timeline_phase": "Pass 1 Glare Gate & Trigger Assessment",
                "glare_pct": glare_pct_0,
                "glare_gate_pct": resolved_gate,
                "glare_disagreement": glare_disagreement,
                "trigger_active": trigger_active,
                "passive_unmasked_verdict": pas_unmasked["verdict"],
                "passive_masked_verdict": pas_masked["verdict"],
            }
        )

        execution_badge = "DETERMINISTIC POLICY (NO LLM)"
        input_tokens = 0
        output_tokens = 0
        tool_calls_count = 0
        llm_latency_ms = 0.0
        fallback_to_deterministic = False

        # Check cached replay (RT-7, D4)
        cached_path = None
        if specimen_id:
            cached_path = os.path.join("logs", "cached_bedrock_runs", f"{specimen_id}.json")

        if (
            allow_cached_replay
            and cached_path
            and os.path.exists(cached_path)
            and policy_mode == "bedrock_llm"
        ):
            try:
                with open(cached_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                active_res = cached_data["active_result"]
                llm_telem = cached_data["llm_telemetry"]
                trace = cached_data["trace"]
                execution_badge = "CACHED REAL BEDROCK TRACE (NOT LIVE — QUOTA/FALLBACK REPLAY)"

                annotated_active = self.engine.detect_defects(
                    state["acc_canvas"],
                    exclude_mask=state["unhealed_mask"],
                    roi=resolved_roi,
                    morph_intensity_thresh=resolved_thresh,
                )["annotated_canvas"]

                return {
                    "opencv_runtime": get_opencv_runtime_metadata(),
                    "orchestrator_mode": policy_mode,
                    "execution_badge": execution_badge,
                    "cycle_latency_ms": round((time.perf_counter() - cycle_t0) * 1000.0, 2),
                    "llm_telemetry": llm_telem,
                    "calibration_meta": calib_meta,
                    "passive_unmasked": pas_unmasked,
                    "passive_masked": pas_masked,
                    "active_result": active_res,
                    "annotated_passive_unmasked": pas_unmasked["annotated_canvas"],
                    "annotated_passive_masked": pas_masked["annotated_canvas"],
                    "annotated_active": annotated_active,
                    "trace": trace,
                }
            except Exception:
                pass

        # If active trigger not met and views > 1, no fusion required
        if not trigger_active:
            final_active_result = {
                "status": "HEALED_AND_INSPECTED",
                "verdict": pas_masked["verdict"],
                "max_defect_length_mm": pas_masked["max_defect_length_mm"],
                "defects": pas_masked["defects"],
                "requires_human_approval": bool(pas_masked["verdict"] == "REJECT"),
                "residual_glare_pct": glare_pct_0,
                "view_attempts_used": 0,
                "decision_note": "Foil glare below gate on Pass 1; no view fusion required.",
            }
        else:
            # Active Multi-View Fusion Execution
            alt_views = candidate_paths[1:]

            if policy_mode == "naive_view1":
                execution_badge = "DETERMINISTIC POLICY (NO LLM)"
                if alt_views:
                    self._execute_tool("fuse_candidate_view", {"candidate_path": alt_views[0]}, state)
                final_active_result = self._execute_tool(
                    "finalize_inspection_or_escalate", {"decision_note": "Naive view 1 policy"}, state
                )

            elif policy_mode == "greedy_lowest_glare":
                execution_badge = "DETERMINISTIC POLICY (NO LLM)"
                rank_res = self._execute_tool(
                    "rank_candidate_views", {"candidate_paths": alt_views}, state
                )
                ranked = sorted(state.get("ranked_candidates", []), key=lambda c: c["candidate_glare_pct"])
                if ranked:
                    self._execute_tool("fuse_candidate_view", {"candidate_path": ranked[0]["path"]}, state)
                final_active_result = self._execute_tool(
                    "finalize_inspection_or_escalate", {"decision_note": "Greedy lowest glare policy"}, state
                )

            elif policy_mode == "active_deterministic":
                execution_badge = "DETERMINISTIC POLICY (NO LLM)"
                self._execute_tool("rank_candidate_views", {"candidate_paths": alt_views}, state)
                for cand in state.get("ranked_candidates", []):
                    if (
                        state["view_attempts"] < self.MAX_VIEW_ATTEMPTS
                        and state["residual_glare_pct"] > resolved_gate
                    ):
                        self._execute_tool("fuse_candidate_view", {"candidate_path": cand["path"]}, state)
                final_active_result = self._execute_tool(
                    "finalize_inspection_or_escalate", {"decision_note": "Active deterministic policy"}, state
                )

            elif policy_mode == "bedrock_llm":
                # Check quota
                active_bucket = quota_bucket
                if judge_passcode and self.quota_limiter.verify_judge_passcode(judge_passcode):
                    active_bucket = "judge_vip"

                allowed, q_reason = self.quota_limiter.check_and_consume(active_bucket)
                if not allowed or self.bedrock_client is None:
                    fallback_to_deterministic = True
                    execution_badge = "DETERMINISTIC POLICY (FALLBACK — QUOTA/NO BEDROCK)"
                    self._execute_tool("rank_candidate_views", {"candidate_paths": alt_views}, state)
                    for cand in state.get("ranked_candidates", []):
                        if (
                            state["view_attempts"] < self.MAX_VIEW_ATTEMPTS
                            and state["residual_glare_pct"] > resolved_gate
                        ):
                            self._execute_tool("fuse_candidate_view", {"candidate_path": cand["path"]}, state)
                    final_active_result = self._execute_tool(
                        "finalize_inspection_or_escalate",
                        {"decision_note": f"Fallback: {q_reason}"},
                        state,
                    )
                else:
                    # Live Bedrock Converse execution
                    execution_badge = "LIVE BEDROCK CONVERSE"
                    llm_start = time.perf_counter()
                    system_prompt = (
                        "You are an expert autonomous active vision metrology controller. "
                        "Specular glare has occluded a critical metallic foil inspection region. "
                        "You have 3 tools: rank_candidate_views, fuse_candidate_view, finalize_inspection_or_escalate. "
                        "Evaluate available candidate views, fuse up to MAX_VIEW_ATTEMPTS=2 views to eliminate glare, "
                        "and finalize or escalate if occlusion cannot be resolved."
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": (
                                        f"Inspection cycle initiated. Initial glare is {glare_pct_0:.2f}%, "
                                        f"exceeding target gate of {resolved_gate:.2f}%. "
                                        f"Alternative candidate views available: {json.dumps(alt_views)}. "
                                        "Proceed with multi-view active healing."
                                    )
                                }
                            ],
                        }
                    ]

                    # Agent Converse multi-turn tool execution loop
                    conversation_active = True
                    turns = 0
                    final_active_result = None

                    while conversation_active and turns < 5:
                        turns += 1
                        try:
                            response = self.bedrock_client.converse(
                                modelId=self.default_model_id,
                                system=[{"text": system_prompt}],
                                messages=messages,
                                toolConfig=self.tool_config,
                            )
                        except Exception as e:
                            fallback_to_deterministic = True
                            state["trace"].append({"bedrock_error": str(e)})
                            break

                        usage = response.get("usage", {})
                        input_tokens += usage.get("inputTokens", 0)
                        output_tokens += usage.get("outputTokens", 0)

                        output_msg = response.get("output", {}).get("message", {})
                        messages.append(output_msg)

                        stop_reason = response.get("stopReason")
                        if stop_reason == "tool_use":
                            tool_results = []
                            for content_block in output_msg.get("content", []):
                                if "toolUse" in content_block:
                                    tool_calls_count += 1
                                    tuse = content_block["toolUse"]
                                    tname = tuse["name"]
                                    targs = tuse.get("input", {})
                                    tid = tuse["toolUseId"]

                                    res_dict = self._execute_tool(tname, targs, state)
                                    state["trace"].append(
                                        {
                                            "timeline_phase": f"LLM Tool: {tname}",
                                            "arguments": targs,
                                            "result": {
                                                k: v for k, v in res_dict.items() if k != "rectified_bgr"
                                            },
                                        }
                                    )

                                    if tname == "finalize_inspection_or_escalate":
                                        final_active_result = res_dict
                                        conversation_active = False

                                    tool_results.append(
                                        {
                                            "toolResult": {
                                                "toolUseId": tid,
                                                "content": [{"json": res_dict}],
                                            }
                                        }
                                    )

                            messages.append({"role": "user", "content": tool_results})
                        else:
                            conversation_active = False

                    llm_latency_ms = round((time.perf_counter() - llm_start) * 1000.0, 2)

                    # If LLM failed to call finalize, execute deterministic safety finalization
                    if final_active_result is None:
                        final_active_result = self._execute_tool(
                            "finalize_inspection_or_escalate",
                            {"decision_note": "Safety finalizer post-LLM conversation"},
                            state,
                        )

            # Mandatory: when trigger_active is True and alt views exist, guarantee view_attempts_used >= 1 (M-06)
            if trigger_active and alt_views and final_active_result.get("view_attempts_used", 0) == 0:
                self._execute_tool("fuse_candidate_view", {"candidate_path": alt_views[0]}, state)
                final_active_result = self._execute_tool(
                    "finalize_inspection_or_escalate",
                    {"decision_note": "Trigger active guarantee fallback"},
                    state,
                )

        # Estimate costs (Amendment 5: unverified estimates)
        est_cost_usd = round(
            (input_tokens / 1e6) * self.cost_per_1m_in + (output_tokens / 1e6) * self.cost_per_1m_out, 6
        )

        llm_telemetry = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_token_cost_usd_unverified": est_cost_usd,
            "cost_notice": "[ESTIMATE ONLY - VERIFY ON AWS BEDROCK PRICING PAGE]",
            "tool_calls_count": tool_calls_count,
            "latency_ms": llm_latency_ms,
            "fallback_to_deterministic": fallback_to_deterministic,
            "model_id": self.default_model_id,
        }

        # Annotate active inspection canvas
        det_active = self.engine.detect_defects(
            state["acc_canvas"],
            exclude_mask=state["unhealed_mask"],
            roi=resolved_roi,
            morph_intensity_thresh=resolved_thresh,
        )
        annotated_active = det_active["annotated_canvas"]

        # Cache live run safely without ndarrays (RT-7, D4)
        if specimen_id and policy_mode == "bedrock_llm" and not fallback_to_deterministic:
            try:
                os.makedirs("logs/cached_bedrock_runs", exist_ok=True)
                cached_run_file = os.path.join("logs", "cached_bedrock_runs", f"{specimen_id}.json")
                serializable_trace = copy.deepcopy(state["trace"])
                cache_payload = {
                    "specimen_id": specimen_id,
                    "active_result": final_active_result,
                    "llm_telemetry": llm_telemetry,
                    "trace": serializable_trace,
                }
                with open(cached_run_file, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
            except Exception:
                pass

        total_cycle_ms = round((time.perf_counter() - cycle_t0) * 1000.0, 2)

        return {
            "opencv_runtime": get_opencv_runtime_metadata(),
            "orchestrator_mode": policy_mode,
            "execution_badge": execution_badge,
            "cycle_latency_ms": total_cycle_ms,
            "llm_telemetry": llm_telemetry,
            "calibration_meta": calib_meta,
            "passive_unmasked": pas_unmasked,
            "passive_masked": pas_masked,
            "active_result": final_active_result,
            "annotated_passive_unmasked": pas_unmasked["annotated_canvas"],
            "annotated_passive_masked": pas_masked["annotated_canvas"],
            "annotated_active": annotated_active,
            "trace": state["trace"],
        }
