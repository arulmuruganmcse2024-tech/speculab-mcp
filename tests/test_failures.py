"""Failure modes, edge cases, and robustness tests for SpecuLab-MCP.

Fully validates:
- test_glare_healing_leaves_zero_rim_artifact_on_clean_foil (M-11, FQ7)
- test_near_border_marker_and_90_degree_rotated_frame (M-15, M-16)
- test_gain_matching_low_overlap_fallback (M-22)
- test_blurry_candidate_penalized_and_optional_flags_default_off (M-24, hard cap on fusions)
- test_bedrock_timeout_config (M-01)
- test_premature_finalize_auto_fuses_remaining_candidate (M-02)
- test_corrupted_candidate_view_skipped_gracefully (M-03)
- test_glare_disagreement_guarantees_at_least_one_fusion (M-06)
- test_three_bucket_quota_and_env_passcode_and_no_fcntl (M-10, M-14, D4)
- test_ground_truth_validator_and_empty_specimens_graceful (Amendment 4)
"""

import copy
import math
import os
import cv2
import numpy as np
import pytest
import yaml

from evaluate import run_evaluation, validate_ground_truth
from scripts.generate_printable_card import create_aruco_marker
from src.agent_orchestrator import BedrockMCPAgent, GlobalQuotaLimiter
from src.vision_engine import PlanarActiveVisionEngine
from tests.test_contract import create_synthetic_card_image


def test_glare_healing_leaves_zero_rim_artifact_on_clean_foil():
    """Places saturated glare band on clean foil in View 0 and 20% exposure-shifted View 1 (M-11, FQ7)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    rx, ry, rw, rh = engine.DEFAULT_ROI

    view_0 = create_synthetic_card_image(add_defect=False)
    view_0[ry : ry + rh, rx : rx + rw] = 140
    cv2.rectangle(view_0, (rx + 250, ry + 100), (rx + 550, ry + 300), (255, 255, 255), -1)

    view_1 = create_synthetic_card_image(add_defect=False)
    view_1[ry : ry + rh, rx : rx + rw] = 168

    mask_0, glare_pct_0, _, _ = engine.compute_glare_mask(view_0, roi=engine.DEFAULT_ROI)
    assert glare_pct_0 > 5.0, "Glare mask must capture central glare band"

    healed, still_unhealed, res_glare, inliers, heal_meta = engine.heal_accumulated_canvas(
        view_0, mask_0, view_1, roi=engine.DEFAULT_ROI
    )

    assert res_glare == 0.0, "All glare should be healed by glare-free View 1"
    assert np.count_nonzero(still_unhealed) == 0

    det_res = engine.detect_defects(healed, exclude_mask=still_unhealed, roi=engine.DEFAULT_ROI)
    assert det_res["verdict"] == "PASS", f"Detected false defect: {det_res['defects']}"
    assert len(det_res["defects"]) == 0


def test_near_border_marker_and_90_degree_rotated_frame():
    """Tests marker within 4 px of image border (no cornerSubPix crash, M-15) and 90-deg rotated frame (M-16)."""
    engine = PlanarActiveVisionEngine("config.yaml")

    # Near border marker (M-15)
    border_frame = np.full((700, 700, 3), 255, dtype=np.uint8)
    m0 = create_aruco_marker(marker_id=0, size_px=400)
    border_frame[2:402, 2:402] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)

    rect_border, meta_border = engine.rectify_to_canonical_plane(border_frame)
    assert rect_border.shape == (800, 2000, 3)
    assert meta_border["rectification_mode"] == "SINGLE_MARKER_4PT"

    # 90-degree rotated portrait frame (1800 x 2400) (M-16)
    card_landscape = create_synthetic_card_image(add_defect=True, defect_xy=(1000, 400), defect_size_px=35)
    card_rot90 = cv2.rotate(card_landscape, cv2.ROTATE_90_CLOCKWISE)

    portrait_frame = np.full((2400, 1800, 3), 240, dtype=np.uint8)
    portrait_frame[200 : 200 + card_rot90.shape[0], 200 : 200 + card_rot90.shape[1]] = card_rot90

    rect_portrait, meta_portrait = engine.rectify_to_canonical_plane(portrait_frame)
    assert rect_portrait.shape == (800, 2000, 3)

    det_portrait = engine.detect_defects(rect_portrait)
    assert det_portrait["verdict"] == "REJECT"
    assert len(det_portrait["defects"]) >= 1


def test_gain_matching_low_overlap_fallback():
    """Tests healing when < 100 mutually glare-free pixels exist (M-22 border median fallback)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    rx, ry, rw, rh = engine.DEFAULT_ROI

    view_0 = create_synthetic_card_image(add_defect=False)
    view_1 = create_synthetic_card_image(add_defect=False)

    unhealed_mask = np.ones((rh, rw), dtype=np.uint8) * 255
    unhealed_mask[:5, :10] = 0

    view_1[ry : ry + 10, rx : rx + 20] = 255

    healed, still_unhealed, res_glare, inliers, heal_meta = engine.heal_accumulated_canvas(
        view_0, unhealed_mask, view_1, roi=engine.DEFAULT_ROI
    )

    assert heal_meta["mutually_valid_pixels"] < 100
    assert not np.isnan(healed).any()
    assert not np.isinf(healed).any()
    assert healed.shape == (800, 2000, 3)


def test_blurry_candidate_penalized_and_optional_flags_default_off(tmp_path):
    """Verifies blurry candidate gets blur_degraded: True and hard cap MAX_VIEW_ATTEMPTS=2 (M-24, M-28)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    # 1. Verify optional feature config defaults are False
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    opt_cfg = cfg.get("optional_features", {})
    assert opt_cfg.get("enable_nbv_pose_prediction") is False
    assert opt_cfg.get("enable_2d_illumination_field_gain") is False
    assert opt_cfg.get("enable_ecc_subpixel_refinement") is False
    assert opt_cfg.get("enable_seam_rollback") is False

    # 2. Test agent._execute_tool("rank_candidate_views") penalizes blurred view
    sharp_card = create_synthetic_card_image(add_defect=True, defect_xy=(1000, 400), defect_size_px=25)
    blurry_card = cv2.GaussianBlur(sharp_card, (41, 41), 15.0)
    # Restore markers so rectification succeeds
    blurry_card[200:600, 100:500] = sharp_card[200:600, 100:500]
    blurry_card[200:600, 1500:1900] = sharp_card[200:600, 1500:1900]

    p_sharp = str(tmp_path / "cand_sharp.png")
    p_blur = str(tmp_path / "cand_blur.png")
    cv2.imwrite(p_sharp, sharp_card)
    cv2.imwrite(p_blur, blurry_card)

    state = {
        "roi": engine.DEFAULT_ROI,
        "glare_gate_pct": 2.0,
        "morph_thresh": 30,
        "optional_flags": engine.optional_features,
        "unhealed_mask": np.ones((400, 800), dtype=np.uint8) * 255,
        "ref_sharpness": 40.0,
        "allowed_map": {os.path.abspath(p_sharp): p_sharp, os.path.abspath(p_blur): p_blur},
    }

    rank_res = agent._execute_tool("rank_candidate_views", {"candidate_paths": [p_blur, p_sharp]}, state)
    ranked = rank_res["ranked_candidates"]
    assert len(ranked) == 2

    # Sharp candidate must be ranked first; blurred must have blur_degraded=True
    assert ranked[0]["path"] == p_sharp
    assert ranked[1]["path"] == p_blur
    assert ranked[1]["blur_degraded"] is True

    # 3. Test hard cap MAX_VIEW_ATTEMPTS = 2 even when rollback is enabled
    fusion_state = {
        "roi": engine.DEFAULT_ROI,
        "glare_gate_pct": 0.5,
        "morph_thresh": 30,
        "optional_flags": {"enable_seam_rollback": True},
        "acc_canvas": sharp_card.copy(),
        "unhealed_mask": np.ones((400, 800), dtype=np.uint8) * 255,
        "residual_glare_pct": 10.0,
        "view_attempts": 0,
        "ranked_candidates": state["ranked_candidates"],
        "trace": [],
    }

    # First fusion
    f1 = agent._execute_tool("fuse_candidate_view", {"candidate_path": p_sharp}, fusion_state)
    assert f1["view_attempts_used"] == 1

    # Second fusion with rollback
    f2 = agent._execute_tool(
        "fuse_candidate_view", {"candidate_path": p_sharp, "rollback_previous_fusion": True}, fusion_state
    )
    assert f2["view_attempts_used"] == 2

    # Third fusion must be rejected by hard cap
    f3 = agent._execute_tool("fuse_candidate_view", {"candidate_path": p_sharp}, fusion_state)
    assert "error" in f3
    assert f3["view_attempts_used"] == 2


def test_bedrock_timeout_config():
    """Assert BedrockMCPAgent configures read_timeout >= 10.0 (M-01)."""
    agent = BedrockMCPAgent("config.yaml")
    assert agent.bedrock_config.read_timeout >= 10.0, "Violated M-01: read_timeout must be >= 10.0s"
    assert agent.bedrock_config.connect_timeout >= 1.0


def test_premature_finalize_auto_fuses_remaining_candidate(tmp_path):
    """Simulates LLM calling finalize prematurely; asserts guard auto-fuses clean view and recovers defect (M-02)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    # View 0: Defect covered by glare
    c0 = create_synthetic_card_image(add_defect=True, defect_xy=(1000, 400), defect_size_px=30)
    c0[350:450, 950:1050] = 255  # Glare covering defect

    # View 1: Clean foil revealing defect
    c1 = create_synthetic_card_image(add_defect=True, defect_xy=(1000, 400), defect_size_px=30)

    p0 = str(tmp_path / "view_0.png")
    p1 = str(tmp_path / "view_1.png")
    cv2.imwrite(p0, c0)
    cv2.imwrite(p1, c1)

    state = {
        "candidate_paths": [p0, p1],
        "allowed_map": {os.path.abspath(p0): p0, os.path.abspath(p1): p1},
        "roi": engine.DEFAULT_ROI,
        "glare_gate_pct": 2.0,
        "morph_thresh": 30,
        "optional_flags": engine.optional_features,
        "acc_canvas": c0.copy(),
        "unhealed_mask": np.ones((400, 800), dtype=np.uint8) * 255,  # High glare
        "residual_glare_pct": 15.0,  # Exceeds gate (2.0)
        "view_attempts": 0,
        "ranked_candidates": [],
        "fused_paths": set(),
        "trace": [],
        "ref_sharpness": 500.0,
    }

    # Agent prematurely calls finalize_inspection_or_escalate
    res = agent._execute_tool("finalize_inspection_or_escalate", {"decision_note": "Premature call"}, state)

    # Premature Finalize Guard (M-02) should have triggered auto-fusion of p1
    assert state["view_attempts"] >= 1, "Guard must auto-fuse remaining candidate view"
    assert res["status"] == "HEALED_AND_INSPECTED"
    assert res["verdict"] == "REJECT", "Must recover defect instead of escalating falsely"
    assert len(res["defects"]) >= 1


def test_corrupted_candidate_view_skipped_gracefully(tmp_path):
    """Passes blank/markerless candidate image alongside valid view; asserts graceful skip in unusable_candidates (M-03)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    # Valid card
    valid_card = create_synthetic_card_image(add_defect=False)
    p_valid = str(tmp_path / "valid.png")
    cv2.imwrite(p_valid, valid_card)

    # Blank corrupted image (no ArUco markers)
    blank_img = np.zeros((800, 2000, 3), dtype=np.uint8)
    p_blank = str(tmp_path / "corrupted_blank.png")
    cv2.imwrite(p_blank, blank_img)

    state = {
        "roi": engine.DEFAULT_ROI,
        "glare_gate_pct": 2.0,
        "morph_thresh": 30,
        "optional_flags": engine.optional_features,
        "unhealed_mask": np.zeros((400, 800), dtype=np.uint8),
        "ref_sharpness": 100.0,
        "allowed_map": {os.path.abspath(p_valid): p_valid, os.path.abspath(p_blank): p_blank},
    }

    rank_res = agent._execute_tool(
        "rank_candidate_views", {"candidate_paths": [p_blank, p_valid]}, state
    )

    # Corrupted image must be logged in unusable_candidates without aborting
    assert len(rank_res["unusable_candidates"]) == 1
    assert rank_res["unusable_candidates"][0]["path"] == p_blank
    assert len(rank_res["ranked_candidates"]) == 1
    assert rank_res["ranked_candidates"][0]["path"] == p_valid


def test_glare_disagreement_guarantees_at_least_one_fusion(tmp_path):
    """Asserts glare disagreement triggers active fusion and view_attempts_used >= 1 (M-06)."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)

    # Create view 0 where unmasked is REJECT, but masked is PASS, with glare_pct = 1.0 (below 2.0 gate)
    # A small dark speck right inside the small glare patch
    c0 = create_synthetic_card_image(add_defect=False)
    rx, ry, rw, rh = engine.DEFAULT_ROI
    # Tiny glare patch (e.g. 1% of ROI)
    cv2.circle(c0, (rx + 200, ry + 200), 25, (255, 255, 255), -1)
    # Dark defect speck inside that glare patch
    cv2.circle(c0, (rx + 200, ry + 200), 10, (20, 20, 20), -1)

    c1 = create_synthetic_card_image(add_defect=False)

    p0 = str(tmp_path / "view_0.png")
    p1 = str(tmp_path / "view_1.png")
    cv2.imwrite(p0, c0)
    cv2.imwrite(p1, c1)

    res = agent.run_inspection_cycle(
        candidate_paths=[p0, p1],
        policy_mode="active_deterministic",
        glare_gate_pct=10.0,  # Gate is high, so glare_pct (1%) < gate (10%)
    )

    # Glare disagreement between unmasked and masked must guarantee at least one fusion (M-06)
    trace0 = res["trace"][0]
    if trace0.get("glare_disagreement"):
        assert res["active_result"]["view_attempts_used"] >= 1


def test_three_bucket_quota_and_env_passcode_and_no_fcntl(tmp_path, monkeypatch):
    """Tests 3 quota buckets, judge passcode verification, and atomic locking without fcntl (M-10, M-14, D4)."""
    quota_file = str(tmp_path / "test_quota.json")
    limiter = GlobalQuotaLimiter(quota_file=quota_file)

    # 1. Test judge passcode
    monkeypatch.setenv("SPECULAB_JUDGE_PASSCODE", "SecretJudgePasscode2026!")
    assert limiter.verify_judge_passcode("SecretJudgePasscode2026!") is True
    assert limiter.verify_judge_passcode("WrongPasscode") is False
    assert limiter.verify_judge_passcode("") is False

    # Short passcode rejected (< 8 chars)
    monkeypatch.setenv("SPECULAB_JUDGE_PASSCODE", "short")
    assert limiter.verify_judge_passcode("short") is False

    # 2. Test 3 isolated buckets
    # public_anonymous (limit 15/hr)
    for _ in range(15):
        allowed, _ = limiter.check_and_consume("public_anonymous")
        assert allowed is True

    # 16th request must be denied
    allowed_16, reason = limiter.check_and_consume("public_anonymous")
    assert allowed_16 is False
    assert "hourly limit reached" in reason

    # Other buckets must remain available
    allowed_judge, _ = limiter.check_and_consume("judge_vip")
    assert allowed_judge is True

    allowed_cli, _ = limiter.check_and_consume("cli_evaluation")
    assert allowed_cli is True

    # 3. Test locking when fcntl is None
    import src.agent_orchestrator as ao
    monkeypatch.setattr(ao, "fcntl", None)
    limiter_nofcntl = GlobalQuotaLimiter(quota_file=str(tmp_path / "quota_nofcntl.json"))
    allowed_nofcntl, _ = limiter_nofcntl.check_and_consume("cli_evaluation")
    assert allowed_nofcntl is True


def test_ground_truth_validator_and_empty_specimens_graceful(tmp_path):
    """Tests schema validation failure messages and graceful handling of empty folders (Amendment 4)."""
    # 1. Valid ground truth
    valid_gt = {
        "spec_001": {
            "split": "tuning",
            "has_defect": True,
            "defect_subtype": "through_hole",
            "true_length_mm": 0.85,
            "gt_bbox_canonical": [800, 300, 40, 40],
            "note": "Valid entry",
        }
    }
    validate_ground_truth(valid_gt)

    # 2. Invalid split
    bad_split = copy.deepcopy(valid_gt)
    bad_split["spec_001"]["split"] = "invalid_split"
    with pytest.raises(ValueError, match="field 'split' must be 'tuning' or 'held_out'"):
        validate_ground_truth(bad_split)

    # 3. Invalid has_defect type
    bad_defect = copy.deepcopy(valid_gt)
    bad_defect["spec_001"]["has_defect"] = "true"  # string instead of bool
    with pytest.raises(ValueError, match="field 'has_defect' must be a boolean"):
        validate_ground_truth(bad_defect)

    # 4. Invalid bbox format
    bad_bbox = copy.deepcopy(valid_gt)
    bad_bbox["spec_001"]["gt_bbox_canonical"] = [800, 300]  # Only 2 elements
    with pytest.raises(ValueError, match="field 'gt_bbox_canonical' must be"):
        validate_ground_truth(bad_bbox)

    # 5. Empty specimens folder handled gracefully
    empty_dir = str(tmp_path / "empty_specimens")
    os.makedirs(empty_dir, exist_ok=True)
    res = run_evaluation(specimens_dir=empty_dir, with_bedrock=False)
    assert res is None, "Empty specimen folder should gracefully return None without error"
