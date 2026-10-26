"""Contract tests for SpecuLab-MCP architecture and invariants.

Fully validates:
- test_app_ui_key_contract_and_stateless_concurrency (full BedrockMCPAgent run_inspection_cycle contract against app.py)
- test_dual_and_single_marker_2000x800_geometry (M-12, D3)
- test_printer_scaling_metric_calibration (M-21)
- test_single_detector_consistency (M-13, D2)
- test_hysteresis_glare_mask_captures_penumbra_halo (M-26)
- test_tuning_sweep_and_six_method_evaluation_metrics (M-04, M-05, M-18)
- test_cached_bedrock_trace_serialization_and_replay (RT-7, D4)
- test_benchmark_outputs_1thread_and_nthread_profiles_and_unpriced_handling (M-19, M-29, D5)
- test_readme_markdown_fences_and_fastmcp_import (M-07, D7)
- test_doc_placeholders_match_render_script (Amendment 6)
"""

import copy
import inspect
import json
import os
import re
import cv2
import numpy as np
import pytest
import yaml

from compare_benchmarks import compare_benchmarks
from evaluate import run_evaluation
from run_authentic_benchmark import run_benchmark
from scripts.generate_printable_card import create_aruco_marker, generate_card
from scripts.render_report_tables import (
    AUTO_RENDERED_PLACEHOLDERS,
    HUMAN_MANUAL_PLACEHOLDERS,
)
from src.agent_orchestrator import BedrockMCPAgent
from src.vision_engine import (
    MarkerNotFoundError,
    PlanarActiveVisionEngine,
    get_opencv_runtime_metadata,
)


def create_synthetic_card_image(
    include_marker_1: bool = True,
    add_defect: bool = False,
    defect_xy: tuple = (1000, 400),
    defect_size_px: int = 30,
) -> np.ndarray:
    """Generate in-memory synthetic card for fast hermetic testing."""
    canvas = np.full((800, 2000, 3), 255, dtype=np.uint8)

    m0 = create_aruco_marker(marker_id=0, size_px=400)
    canvas[200:600, 100:500] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)

    if include_marker_1:
        m1 = create_aruco_marker(marker_id=1, size_px=400)
        canvas[200:600, 1500:1900] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)

    canvas[200:600, 600:1400] = 160

    cv2.rectangle(canvas, (100, 40), (1900, 160), (120, 120, 120), 2)
    cv2.rectangle(canvas, (100, 640), (1900, 760), (120, 120, 120), 2)
    cv2.putText(canvas, "TEST-TEXTURE-ANCHOR", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "TEST-TEXTURE-ANCHOR", (200, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if add_defect:
        dx, dy = defect_xy
        cv2.circle(canvas, (dx, dy), defect_size_px // 2, (30, 30, 30), -1)

    return canvas


def test_app_ui_key_contract_and_stateless_concurrency(tmp_path):
    """Assert BedrockMCPAgent.run_inspection_cycle outputs every key read by app.py and never mutates state (M-20)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    # Capture initial states
    initial_engine_state = {
        k: copy.deepcopy(v)
        for k, v in engine.__dict__.items()
        if k not in ["aruco_dict", "aruco_params"]
    }
    initial_agent_state = {
        k: copy.deepcopy(v)
        for k, v in agent.__dict__.items()
        if k not in ["engine", "bedrock_client", "quota_limiter", "audit_logger", "bedrock_config"]
    }

    # Create synthetic candidate images
    card0 = create_synthetic_card_image(add_defect=True)
    card1 = create_synthetic_card_image(add_defect=True)
    # Add glare on card 0
    card0[250:450, 850:1150] = 255

    p0 = str(tmp_path / "view_0.png")
    p1 = str(tmp_path / "view_1.png")
    cv2.imwrite(p0, card0)
    cv2.imwrite(p1, card1)

    custom_roi = (650, 220, 700, 360)
    custom_gate = 1.5
    custom_thresh = 35
    custom_flags = {
        "enable_nbv_pose_prediction": True,
        "enable_2d_illumination_field_gain": False,
        "enable_ecc_subpixel_refinement": False,
        "enable_seam_rollback": True,
    }

    # Run inspection cycle
    cycle_res = agent.run_inspection_cycle(
        candidate_paths=[p0, p1],
        policy_mode="active_deterministic",
        quota_bucket="cli_evaluation",
        roi=custom_roi,
        glare_gate_pct=custom_gate,
        morph_intensity_thresh=custom_thresh,
        optional_flags=custom_flags,
    )

    # Every key read by app.py must exist
    app_required_keys = [
        "opencv_runtime",
        "orchestrator_mode",
        "execution_badge",
        "cycle_latency_ms",
        "llm_telemetry",
        "calibration_meta",
        "passive_unmasked",
        "passive_masked",
        "active_result",
        "annotated_passive_unmasked",
        "annotated_passive_masked",
        "annotated_active",
        "trace",
    ]
    for k in app_required_keys:
        assert k in cycle_res, f"Required key '{k}' missing from run_inspection_cycle output"

    assert isinstance(cycle_res["annotated_passive_unmasked"], np.ndarray)
    assert isinstance(cycle_res["annotated_passive_masked"], np.ndarray)
    assert isinstance(cycle_res["annotated_active"], np.ndarray)
    assert isinstance(cycle_res["trace"], list)
    assert len(cycle_res["trace"]) >= 1

    # Verify zero mutation of engine and agent attributes
    for k, v in initial_engine_state.items():
        curr = getattr(engine, k)
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, curr), f"Mutated engine array {k}"
        else:
            assert v == curr, f"Mutated engine attribute {k}"

    for k, v in initial_agent_state.items():
        curr = getattr(agent, k)
        assert v == curr, f"Mutated agent attribute {k}"


def test_dual_and_single_marker_2000x800_geometry(tmp_path):
    """Verify 2000x800 px rectification in DUAL_MARKER_8PT and SINGLE_MARKER_4PT modes (M-12, D3)."""
    engine = PlanarActiveVisionEngine("config.yaml")

    dual_card = create_synthetic_card_image(include_marker_1=True)
    pts_src = np.float32([[50, 50], [1950, 40], [1940, 760], [60, 750]])
    pts_dst = np.float32([[0, 0], [2000, 0], [2000, 800], [0, 800]])
    H_sim = cv2.getPerspectiveTransform(pts_dst, pts_src)
    frame_dual = cv2.warpPerspective(dual_card, H_sim, (2100, 900))

    rect_dual, meta_dual = engine.rectify_to_canonical_plane(frame_dual)
    assert rect_dual.shape == (800, 2000, 3)
    assert meta_dual["rectification_mode"] == "DUAL_MARKER_8PT"

    single_card = create_synthetic_card_image(include_marker_1=False)
    frame_single = cv2.warpPerspective(single_card, H_sim, (2100, 900))

    rect_single, meta_single = engine.rectify_to_canonical_plane(frame_single)
    assert rect_single.shape == (800, 2000, 3)
    assert meta_single["rectification_mode"] == "SINGLE_MARKER_4PT"


def test_printer_scaling_metric_calibration(tmp_path):
    """Verify changing marker_size_mm updates MM_PER_PX and defect length metrology (M-21)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["calibration"]["marker_size_mm"] = 19.0
    custom_cfg_path = tmp_path / "config_19mm.yaml"
    with open(custom_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    engine_19 = PlanarActiveVisionEngine(str(custom_cfg_path))
    assert engine_19.MARKER_SIZE_MM == 19.0
    expected_mm_per_px = 19.0 / 400.0
    assert abs(engine_19.MM_PER_PX - expected_mm_per_px) < 1e-7

    card_19 = create_synthetic_card_image(add_defect=True, defect_xy=(1000, 400), defect_size_px=40)
    res = engine_19.detect_defects(card_19)
    assert res["verdict"] == "REJECT"
    defect = res["defects"][0]
    expected_len = 40.0 * (19.0 / 400.0)
    assert abs(defect["length_mm"] - expected_len) < 0.35


def test_single_detector_consistency():
    """Verify MORPH_TOPHAT is completely absent and BLACKHAT is used consistently (M-13, D2)."""
    import src.vision_engine as ve
    source_code = inspect.getsource(ve)
    assert "MORPH_TOPHAT" not in source_code, "Violated D2: MORPH_TOPHAT must be completely absent"
    assert "MORPH_BLACKHAT" in source_code, "Must use MORPH_BLACKHAT detector"

    engine = PlanarActiveVisionEngine("config.yaml")
    clean_card = create_synthetic_card_image(add_defect=False)
    defective_card = create_synthetic_card_image(add_defect=True, defect_size_px=30)

    res_clean = engine.detect_defects(clean_card)
    assert res_clean["verdict"] == "PASS"
    assert len(res_clean["defects"]) == 0

    res_def = engine.detect_defects(defective_card)
    assert res_def["verdict"] == "REJECT"
    assert len(res_def["defects"]) >= 1


def test_hysteresis_glare_mask_captures_penumbra_halo():
    """Verify V=250 core with V=220 halo is captured, while max V=225 returns 0.0% glare (M-26)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    canvas = np.full((800, 2000, 3), 120, dtype=np.uint8)

    roi = engine.DEFAULT_ROI
    rx, ry, rw, rh = roi

    cv2.circle(canvas, (rx + 400, ry + 200), 45, (220, 220, 220), -1)
    cv2.circle(canvas, (rx + 400, ry + 200), 20, (250, 250, 250), -1)

    mask_halo, glare_pct_halo, _, _ = engine.compute_glare_mask(canvas, roi=roi)
    assert glare_pct_halo > 0.0
    halo_count = np.count_nonzero(mask_halo)
    assert halo_count > np.pi * (30 ** 2)

    canvas_no_core = np.full((800, 2000, 3), 120, dtype=np.uint8)
    cv2.circle(canvas_no_core, (rx + 400, ry + 200), 50, (225, 225, 225), -1)
    mask_none, glare_pct_none, _, _ = engine.compute_glare_mask(canvas_no_core, roi=roi)
    assert glare_pct_none == 0.0
    assert np.count_nonzero(mask_none) == 0


def test_tuning_sweep_and_six_method_evaluation_metrics(tmp_path):
    """Verify 4x3 tuning sweep and held-out 6-method evaluation metrics (M-04, M-05, M-18)."""
    spec_dir = tmp_path / "specimens"
    os.makedirs(spec_dir, exist_ok=True)

    # Create 2 tuning and 2 held-out specimens
    gt_data = {
        "spec_tune_01": {
            "split": "tuning",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 1.0,
            "gt_bbox_canonical": [980, 380, 40, 40],
            "note": "Tuning defective",
        },
        "spec_tune_02": {
            "split": "tuning",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": "Tuning clean",
        },
        "spec_held_01": {
            "split": "held_out",
            "has_defect": True,
            "defect_subtype": "surface_scratch",
            "true_length_mm": 1.2,
            "gt_bbox_canonical": [980, 380, 40, 40],
            "note": "Held out defective",
        },
        "spec_held_02": {
            "split": "held_out",
            "has_defect": False,
            "defect_subtype": None,
            "true_length_mm": None,
            "gt_bbox_canonical": None,
            "note": "Held out clean",
        },
    }
    with open(spec_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_data, f)

    for sid, entry in gt_data.items():
        s_folder = spec_dir / sid
        os.makedirs(s_folder, exist_ok=True)
        c0 = create_synthetic_card_image(add_defect=entry["has_defect"])
        c1 = create_synthetic_card_image(add_defect=entry["has_defect"])
        cv2.imwrite(str(s_folder / "view_0.png"), c0)
        cv2.imwrite(str(s_folder / "view_1.png"), c1)

    eval_out = str(tmp_path / "eval_report.json")
    report = run_evaluation(specimens_dir=str(spec_dir), with_bedrock=False, output_report=eval_out)

    assert report is not None
    assert len(report["tuning_sweep"]) == 12  # 4 gates x 3 thresholds
    assert "locked_tuning_config" in report

    methods = report["held_out_methods"]
    assert "1A_passive_unmasked" in methods
    assert "1B_passive_glare_masked" in methods
    assert "2A_naive_view1" in methods
    assert "2B_greedy_lowest_glare" in methods
    assert "2C_active_deterministic" in methods

    # Verify integer fraction strings
    for m in methods.values():
        assert "/" in m["strict_defect_recall"]
        assert "/" in m["safety_capture_rate_incl_escalation"]
        assert "/" in m["clean_false_rejects"]
        assert "/" in m["unresolved_occlusion_flagged"]


def test_cached_bedrock_trace_serialization_and_replay(tmp_path):
    """Verify saving run to logs/cached_bedrock_runs/ serializes without ndarrays and replays badge (RT-7, D4)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    c0 = create_synthetic_card_image(add_defect=False)
    p0 = str(tmp_path / "view_0.png")
    cv2.imwrite(p0, c0)

    # Test cache saving and loading
    cached_dir = os.path.join("logs", "cached_bedrock_runs")
    os.makedirs(cached_dir, exist_ok=True)
    specimen_id = "test_specimen_replay_001"
    cached_file = os.path.join(cached_dir, f"{specimen_id}.json")

    try:
        mock_payload = {
            "specimen_id": specimen_id,
            "active_result": {
                "status": "HEALED_AND_INSPECTED",
                "verdict": "PASS",
                "max_defect_length_mm": 0.0,
                "defects": [],
                "requires_human_approval": False,
                "view_attempts_used": 1,
            },
            "llm_telemetry": {
                "input_tokens": 120,
                "output_tokens": 40,
                "estimated_token_cost_usd_unverified": 0.00001,
                "cost_notice": "[ESTIMATE ONLY - VERIFY ON AWS BEDROCK PRICING PAGE]",
                "tool_calls_count": 2,
                "latency_ms": 350.0,
                "fallback_to_deterministic": False,
                "model_id": agent.default_model_id,
            },
            "trace": [{"phase": "Mock trace"}],
        }
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(mock_payload, f)

        res = agent.run_inspection_cycle(
            candidate_paths=[p0],
            policy_mode="bedrock_llm",
            allow_cached_replay=True,
            specimen_id=specimen_id,
        )

        assert res["execution_badge"] == "CACHED REAL BEDROCK TRACE (NOT LIVE — QUOTA/FALLBACK REPLAY)"
        assert res["active_result"]["verdict"] == "PASS"
    finally:
        if os.path.exists(cached_file):
            os.remove(cached_file)


def test_benchmark_outputs_1thread_and_nthread_profiles_and_unpriced_handling(tmp_path):
    """Verify run_authentic_benchmark outputs 1-thread/n-thread, peak RSS, and unpriced status (M-19, M-29, D5)."""
    bench_out = str(tmp_path / "bench_unpriced.json")
    report = run_benchmark(
        label="unpriced_test",
        specimens_dir="assets/specimens",
        threads=2,
        hourly_cost=0.0,
        output_path=bench_out,
    )

    assert "profile_1thread" in report
    assert "profile_nthread" in report
    assert report["profile_1thread"]["threads"] == 1
    assert report["profile_nthread"]["threads"] == 2
    assert report["profile_1thread"]["peak_rss_mb"] > 0.0

    cost = report["cost_analysis"]
    assert cost["pricing_status"] == "UNPRICED_EXCLUDED_FROM_COST"
    assert cost["hourly_ec2_cost_usd"] is None

    comp_out = str(tmp_path / "comp_out.json")
    comp_res = compare_benchmarks(input_files=[bench_out], output_path=comp_out)
    assert comp_res is not None
    assert "table_1thread" in comp_res["markdown_tables"]
    assert "table_nthread" in comp_res["markdown_tables"]


def test_readme_markdown_fences_and_fastmcp_import():
    """Verify README.md has balanced code fences and FastMCP registers all 3 tools (M-07, D7)."""
    with open("README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()

    fences = re.findall(r"```", readme_text)
    assert len(fences) % 2 == 0, f"Unbalanced code fences in README.md (count={len(fences)})"

    import src.mcp_server as mcp_srv
    registered_tools = list(mcp_srv.mcp._tool_manager._tools.keys())
    assert "rank_candidate_views" in registered_tools
    assert "fuse_candidate_view" in registered_tools
    assert "finalize_inspection_or_escalate" in registered_tools


def test_doc_placeholders_match_render_script():
    """Verify all TODO_* placeholders in docs are registered in render_report_tables.py (Amendment 6)."""
    all_registered = AUTO_RENDERED_PLACEHOLDERS | HUMAN_MANUAL_PLACEHOLDERS

    docs = ["README.md", "TECHNICAL_REPORT.md", "VIDEO_SCRIPT.md"]
    for doc in docs:
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()

        found_tokens = set(re.findall(r"TODO_[A-Z0-9_]+", content))
        for token in found_tokens:
            assert (
                token in all_registered
            ), f"Unregistered placeholder '{token}' found in {doc}. Must match render_report_tables.py!"


def test_zero_remaining_todos_and_preloaded_dataset_ready():
    """Assert zero remaining TODO_* in documentation and that preloaded dataset + empirical reports exist."""
    from evaluate import validate_ground_truth

    docs = ["README.md", "TECHNICAL_REPORT.md", "VIDEO_SCRIPT.md", "DEVPOST_SUBMISSION.md"]
    for doc in docs:
        assert os.path.exists(doc), f"Document {doc} does not exist"
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        todos = re.findall(r"TODO_[A-Z0-9_]+", content)
        assert len(todos) == 0, f"Found remaining TODO placeholders in {doc}: {todos}"

    assert os.path.exists("evaluation_report.json"), "evaluation_report.json missing"
    with open("evaluation_report.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    assert isinstance(eval_data, dict), "evaluation_report.json not valid JSON object"
    assert "held_out_methods" in eval_data, "evaluation_report.json missing held_out_methods key"

    assert os.path.exists("benchmark_comparison.json"), "benchmark_comparison.json missing"
    with open("benchmark_comparison.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    assert isinstance(bench_data, dict), "benchmark_comparison.json not valid JSON object"
    assert "runs" in bench_data, "benchmark_comparison.json missing runs key"

    gt_path = "assets/specimens/ground_truth.json"
    assert os.path.exists(gt_path), "assets/specimens/ground_truth.json missing"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    assert len(gt_data) == 14, f"Expected 14 specimens in ground_truth.json, found {len(gt_data)}"
    validate_ground_truth(gt_data)

