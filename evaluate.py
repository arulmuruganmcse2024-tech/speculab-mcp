"""Empirical evaluation and sweep engine for SpecuLab-MCP.

Executes:
1. 4x3 Tuning Sweep (glare_gate_pct x morph_intensity_thresh) on tuning split (M-04).
2. Held-out 6-method multi-policy evaluation (M-05, M-18, D4, D6).
3. Schema validation for ground_truth.json and graceful empty-folder handling (Amendment 4).
4. Unverified token cost estimation labeling (Amendment 5).
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.agent_orchestrator import BedrockMCPAgent
from src.vision_engine import (
    PlanarActiveVisionEngine,
    get_opencv_runtime_metadata,
    load_image_exif_safe,
)


def validate_ground_truth(gt: Dict[str, Any]) -> None:
    """Validate ground_truth.json schema strictly according to Amendment 4.

    Schema:
    {specimen_id: {
        "split": "tuning" | "held_out",
        "has_defect": bool,
        "defect_subtype": str | None,
        "true_length_mm": float | int | None,
        "gt_bbox_canonical": [x, y, w, h] | None,
        "note": str
    }}
    """
    if not isinstance(gt, dict):
        raise ValueError("Ground truth root must be a dictionary keyed by specimen_id")

    for specimen_id, entry in gt.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Specimen '{specimen_id}' ground truth entry must be a dictionary")

        split = entry.get("split")
        if split not in ["tuning", "held_out"]:
            raise ValueError(
                f"Specimen '{specimen_id}' field 'split' must be 'tuning' or 'held_out', got '{split}'"
            )

        has_defect = entry.get("has_defect")
        if not isinstance(has_defect, bool):
            raise ValueError(
                f"Specimen '{specimen_id}' field 'has_defect' must be a boolean, got {type(has_defect).__name__}"
            )

        subtype = entry.get("defect_subtype")
        if subtype is not None and not isinstance(subtype, str):
            raise ValueError(
                f"Specimen '{specimen_id}' field 'defect_subtype' must be str or None, got {type(subtype).__name__}"
            )

        tlen = entry.get("true_length_mm")
        if tlen is not None and not isinstance(tlen, (int, float)):
            raise ValueError(
                f"Specimen '{specimen_id}' field 'true_length_mm' must be float/int or None, got {type(tlen).__name__}"
            )

        bbox = entry.get("gt_bbox_canonical")
        if bbox is not None:
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
            ):
                raise ValueError(
                    f"Specimen '{specimen_id}' field 'gt_bbox_canonical' must be [x, y, w, h] or None, got {bbox}"
                )

        note = entry.get("note")
        if not isinstance(note, str):
            raise ValueError(
                f"Specimen '{specimen_id}' field 'note' must be a string, got {type(note).__name__}"
            )


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two [x, y, w, h] bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = (w1 * h1) + (w2 * h2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_specimens(specimens_dir: str = "assets/specimens") -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Load and validate ground truth and image file paths for all specimens."""
    gt_path = os.path.join(specimens_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        return {}, {}

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    validate_ground_truth(gt)

    specimen_views: Dict[str, List[str]] = {}
    for sid in gt:
        s_folder = os.path.join(specimens_dir, sid)
        if os.path.isdir(s_folder):
            views = sorted(
                glob.glob(os.path.join(s_folder, "*.jpg"))
                + glob.glob(os.path.join(s_folder, "*.png"))
                + glob.glob(os.path.join(s_folder, "*.jpeg"))
            )
            if views:
                specimen_views[sid] = views

    return gt, specimen_views


def evaluate_method_on_split(
    method_key: str,
    specimen_ids: List[str],
    gt: Dict[str, Any],
    views_map: Dict[str, List[str]],
    agent: BedrockMCPAgent,
    glare_gate_pct: float,
    morph_intensity_thresh: int,
    with_bedrock: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single policy method across specified specimen set."""
    policy_mode = "active_deterministic"
    if method_key == "1A_passive_unmasked":
        policy_mode = "passive_unmasked"
    elif method_key == "1B_passive_glare_masked":
        policy_mode = "passive_glare_masked"
    elif method_key == "2A_naive_view1":
        policy_mode = "naive_view1"
    elif method_key == "2B_greedy_lowest_glare":
        policy_mode = "greedy_lowest_glare"
    elif method_key == "2C_active_deterministic":
        policy_mode = "active_deterministic"
    elif method_key == "2D_active_bedrock_live":
        policy_mode = "bedrock_llm"

    n_total = len(specimen_ids)
    n_def = sum(1 for sid in specimen_ids if gt[sid]["has_defect"])
    n_clean = n_total - n_def

    tp_reject = 0
    fp_reject = 0
    def_escalated = 0
    clean_escalated = 0

    llm_attempted_count = 0
    llm_completion_without_fallback_count = 0
    fallback_triggered_count = 0
    llm_vs_deterministic_verdict_agreement_count = 0
    llm_divergent_view_choice_count = 0
    tool_calls_list = []
    latencies_list = []
    total_estimated_token_cost_usd_unverified = 0.0

    for sid in specimen_ids:
        cands = views_map.get(sid, [])
        if not cands:
            continue

        gt_entry = gt[sid]
        has_defect = gt_entry["has_defect"]
        gt_bbox = gt_entry.get("gt_bbox_canonical")

        if method_key in ["1A_passive_unmasked", "1B_passive_glare_masked"]:
            v0 = cands[0]
            img0 = agent.engine.rectify_to_canonical_plane(load_image_exif_safe(v0))[0]
            if method_key == "1A_passive_unmasked":
                res = agent.engine.detect_defects(
                    img0, exclude_mask=None, morph_intensity_thresh=morph_intensity_thresh
                )
            else:
                m0, _, _, _ = agent.engine.compute_glare_mask(img0)
                res = agent.engine.detect_defects(
                    img0, exclude_mask=m0, morph_intensity_thresh=morph_intensity_thresh
                )
            verdict = res["verdict"]
            defects = res["defects"]
            status = "HEALED_AND_INSPECTED"
        else:
            cycle_out = agent.run_inspection_cycle(
                candidate_paths=cands,
                policy_mode=policy_mode,
                quota_bucket="cli_evaluation",
                specimen_id=sid,
                glare_gate_pct=glare_gate_pct,
                morph_intensity_thresh=morph_intensity_thresh,
            )
            act_res = cycle_out["active_result"]
            verdict = act_res.get("verdict", "PASS")
            defects = act_res.get("defects", [])
            status = act_res.get("status", "HEALED_AND_INSPECTED")

            if method_key == "2D_active_bedrock_live":
                telem = cycle_out.get("llm_telemetry", {})
                llm_attempted_count += 1
                if not telem.get("fallback_to_deterministic", False):
                    llm_completion_without_fallback_count += 1
                else:
                    fallback_triggered_count += 1
                tool_calls_list.append(telem.get("tool_calls_count", 0))
                latencies_list.append(telem.get("latency_ms", 0.0))
                total_estimated_token_cost_usd_unverified += telem.get(
                    "estimated_token_cost_usd_unverified", 0.0
                )

        # Check classification outcome
        is_reject = verdict == "REJECT"
        is_escalated = status == "UNRESOLVED_OCCLUSION"

        # Check bbox localization if gt_bbox provided
        if is_reject and has_defect and gt_bbox is not None:
            matched_iou = any(compute_iou(gt_bbox, d["bbox_canonical"]) >= 0.15 for d in defects)
            if not matched_iou:
                is_reject = False

        if has_defect:
            if is_reject:
                tp_reject += 1
            elif is_escalated:
                def_escalated += 1
        else:
            if is_reject:
                fp_reject += 1
            elif is_escalated:
                clean_escalated += 1

    total_escalated = def_escalated + clean_escalated

    strict_recall_str = f"{tp_reject}/{n_def}" if n_def > 0 else "0/0"
    safety_capture_str = f"{tp_reject + def_escalated}/{n_def}" if n_def > 0 else "0/0"
    clean_fp_str = f"{fp_reject}/{n_clean}" if n_clean > 0 else "0/0"
    unresolved_str = f"{total_escalated}/{n_total}" if n_total > 0 else "0/0"

    out = {
        "method_key": method_key,
        "n_total": n_total,
        "n_def": n_def,
        "n_clean": n_clean,
        "tp_reject": tp_reject,
        "fp_reject": fp_reject,
        "def_escalated": def_escalated,
        "clean_escalated": clean_escalated,
        "total_escalated": total_escalated,
        "strict_defect_recall": strict_recall_str,
        "safety_capture_rate_incl_escalation": safety_capture_str,
        "clean_false_rejects": clean_fp_str,
        "unresolved_occlusion_flagged": unresolved_str,
    }

    if method_key == "2D_active_bedrock_live":
        out.update(
            {
                "llm_attempted_count": llm_attempted_count,
                "llm_completion_without_fallback_count": llm_completion_without_fallback_count,
                "fallback_triggered_count": fallback_triggered_count,
                "llm_vs_deterministic_verdict_agreement_count": llm_vs_deterministic_verdict_agreement_count,
                "llm_divergent_view_choice_count": llm_divergent_view_choice_count,
                "mean_tool_calls": round(float(np.mean(tool_calls_list)), 2) if tool_calls_list else 0.0,
                "mean_latency_ms": round(float(np.mean(latencies_list)), 2) if latencies_list else 0.0,
                "estimated_token_cost_usd_unverified": round(total_estimated_token_cost_usd_unverified, 6),
                "cost_notice": "[ESTIMATE ONLY - VERIFY ON AWS BEDROCK PRICING PAGE]",
            }
        )

    return out


def run_evaluation(
    specimens_dir: str = "assets/specimens",
    with_bedrock: bool = False,
    output_report: str = "evaluation_report.json",
) -> Optional[Dict[str, Any]]:
    """Run exploratory 4x3 tuning sweep and held-out 6-method evaluation."""
    gt, views_map = load_specimens(specimens_dir)

    if not gt or not views_map:
        print(
            f"[INFO] No specimen dataset found at '{specimens_dir}' (ground_truth.json or specimen folders missing).\n"
            "Gracefully stopping evaluation without errors (Amendment 4)."
        )
        return None

    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    tuning_ids = [sid for sid, e in gt.items() if e["split"] == "tuning" and sid in views_map]
    held_out_ids = [sid for sid, e in gt.items() if e["split"] == "held_out" and sid in views_map]

    print(
        f"Loaded {len(gt)} specimens ({len(tuning_ids)} tuning, {len(held_out_ids)} held_out). "
        f"Views available for {len(views_map)} specimens."
    )

    # Step 1: 4x3 Exploratory Tuning Sweep (M-04)
    gate_sweep = [1.0, 2.0, 5.0, 10.0]
    thresh_sweep = [24, 30, 36]
    sweep_results = []

    best_score = -999999
    best_config = (2.0, 30)

    for g in gate_sweep:
        for th in thresh_sweep:
            res = evaluate_method_on_split(
                "2C_active_deterministic",
                tuning_ids,
                gt,
                views_map,
                agent,
                glare_gate_pct=g,
                morph_intensity_thresh=th,
            )
            score_tuple = (res["tp_reject"] - res["fp_reject"], res["tp_reject"] + res["def_escalated"])
            sweep_results.append(
                {
                    "glare_gate_pct": g,
                    "morph_intensity_thresh": th,
                    "tp_reject": res["tp_reject"],
                    "fp_reject": res["fp_reject"],
                    "def_escalated": res["def_escalated"],
                    "strict_defect_recall": res["strict_defect_recall"],
                    "safety_capture_rate": res["safety_capture_rate_incl_escalation"],
                    "clean_false_rejects": res["clean_false_rejects"],
                }
            )

            # Tie-breaking in favor of simplest default config (2.0, 30)
            score_val = score_tuple[0] * 1000 + score_tuple[1]
            if (g == 2.0 and th == 30) and score_val == best_score:
                best_config = (g, th)
            elif score_val > best_score:
                best_score = score_val
                best_config = (g, th)

    locked_gate_pct, locked_morph_thresh = best_config
    print(f"Locked sweep config: glare_gate_pct={locked_gate_pct}, morph_intensity_thresh={locked_morph_thresh}")

    # Step 2: Held-Out 6-Method Evaluation (M-05, M-18)
    methods = [
        "1A_passive_unmasked",
        "1B_passive_glare_masked",
        "2A_naive_view1",
        "2B_greedy_lowest_glare",
        "2C_active_deterministic",
    ]
    if with_bedrock:
        methods.append("2D_active_bedrock_live")

    held_out_reports = {}
    for m in methods:
        res_m = evaluate_method_on_split(
            m,
            held_out_ids,
            gt,
            views_map,
            agent,
            glare_gate_pct=locked_gate_pct,
            morph_intensity_thresh=locked_morph_thresh,
            with_bedrock=with_bedrock,
        )
        held_out_reports[m] = res_m

    full_report = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opencv_runtime": get_opencv_runtime_metadata(),
        "dataset_summary": {
            "total_specimens": len(gt),
            "tuning_count": len(tuning_ids),
            "held_out_count": len(held_out_ids),
            "defective_count": sum(1 for e in gt.values() if e["has_defect"]),
            "clean_count": sum(1 for e in gt.values() if not e["has_defect"]),
        },
        "locked_tuning_config": {
            "glare_gate_pct": locked_gate_pct,
            "morph_intensity_thresh": locked_morph_thresh,
        },
        "tuning_sweep": sweep_results,
        "held_out_methods": held_out_reports,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Evaluation completed. Report written to {output_report}")
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SpecuLab-MCP on physical specimens")
    parser.add_argument(
        "--specimens-dir",
        type=str,
        default="assets/specimens",
        help="Path to specimens directory",
    )
    parser.add_argument(
        "--with-bedrock",
        action="store_true",
        help="Include live Bedrock Converse active policy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output evaluation JSON report path",
    )
    args = parser.parse_args()

    run_evaluation(
        specimens_dir=args.specimens_dir,
        with_bedrock=args.with_bedrock,
        output_report=args.output,
    )


if __name__ == "__main__":
    main()
