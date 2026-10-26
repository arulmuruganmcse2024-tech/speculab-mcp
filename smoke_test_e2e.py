"""End-to-end smoke test validating all SpecuLab-MCP pipeline components.

Requirements:
- Must exit with code 0.
- Must leave zero synthetic files in tmp_smoke_test/, assets/specimens/, or root report JSONs.
- Validates the complete pipeline: check_env, card generation, vision engine, agent,
  MCP server, evaluate, benchmark, and render scripts.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import cv2
import numpy as np

SMOKE_TMP_DIR = "tmp_smoke_test"


def cleanup():
    """Ensure all temporary smoke test directories and files are cleaned up."""
    if os.path.exists(SMOKE_TMP_DIR):
        try:
            shutil.rmtree(SMOKE_TMP_DIR)
        except Exception:
            pass

    for cleanup_file in ["smoke_eval.json", "smoke_bench.json", "smoke_comp.json"]:
        if os.path.exists(cleanup_file):
            try:
                os.remove(cleanup_file)
            except Exception:
                pass


def main() -> int:
    cleanup()
    os.makedirs(SMOKE_TMP_DIR, exist_ok=True)
    python_bin = sys.executable

    try:
        print("[1/8] Testing check_env.py...")
        subprocess.check_call([python_bin, "check_env.py"])

        print("[2/8] Testing scripts/generate_printable_card.py...")
        card_out = os.path.join(SMOKE_TMP_DIR, "smoke_card.png")
        subprocess.check_call([python_bin, "scripts/generate_printable_card.py", "--output", card_out])
        assert os.path.exists(card_out), "Printable card output was not created"

        print("[3/8] Testing PlanarActiveVisionEngine in isolation...")
        from scripts.generate_printable_card import create_aruco_marker
        from src.vision_engine import PlanarActiveVisionEngine

        engine = PlanarActiveVisionEngine("config.yaml")

        # Create 2 synthetic views with defect and glare
        view0 = np.full((800, 2000, 3), 255, dtype=np.uint8)
        m0 = create_aruco_marker(0, 400)
        m1 = create_aruco_marker(1, 400)
        view0[200:600, 100:500] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)
        view0[200:600, 1500:1900] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
        view0[200:600, 600:1400] = 150  # Foil area
        # Specular glare band on view 0
        cv2.rectangle(view0, (850, 300), (1150, 500), (255, 255, 255), -1)

        view1 = view0.copy()
        # View 1 without glare + with circular defect
        view1[200:600, 600:1400] = 165
        cv2.circle(view1, (1000, 400), 20, (30, 30, 30), -1)

        v0_path = os.path.join(SMOKE_TMP_DIR, "view_0.png")
        v1_path = os.path.join(SMOKE_TMP_DIR, "view_1.png")
        cv2.imwrite(v0_path, view0)
        cv2.imwrite(v1_path, view1)

        print("[4/8] Testing BedrockMCPAgent.run_inspection_cycle()...")
        from src.agent_orchestrator import BedrockMCPAgent

        agent = BedrockMCPAgent("config.yaml", engine=engine)
        cycle_res = agent.run_inspection_cycle(
            candidate_paths=[v0_path, v1_path],
            policy_mode="active_deterministic",
            quota_bucket="cli_evaluation",
        )

        expected_keys = [
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
        for k in expected_keys:
            assert k in cycle_res, f"Key '{k}' missing from cycle_res"

        assert cycle_res["active_result"]["verdict"] == "REJECT", "Active result should reject defect"

        print("[5/8] Testing src/mcp_server.py stateless tools...")
        import src.mcp_server as mcp_srv

        rank_out = mcp_srv.rank_candidate_views(v0_path, [v1_path])
        assert "ranked_candidates" in rank_out, "rank_candidate_views failed"

        fuse_out = mcp_srv.fuse_candidate_view(v0_path, v1_path)
        assert "fused_view" in fuse_out, "fuse_candidate_view failed"

        final_out = mcp_srv.finalize_inspection_or_escalate(v0_path, [v1_path])
        assert "verdict" in final_out, "finalize_inspection_or_escalate failed"

        print("[6/8] Testing evaluate.py with isolated synthetic ground truth...")
        from evaluate import run_evaluation, validate_ground_truth

        # Test validator on valid schema
        valid_gt = {
            "specimen_smoke_01": {
                "split": "tuning",
                "has_defect": True,
                "defect_subtype": "through_hole",
                "true_length_mm": 1.0,
                "gt_bbox_canonical": [980, 380, 40, 40],
                "note": "Smoke test valid entry",
            }
        }
        validate_ground_truth(valid_gt)

        # Setup smoke specimen folder
        sdir = os.path.join(SMOKE_TMP_DIR, "specimens", "specimen_smoke_01")
        os.makedirs(sdir, exist_ok=True)
        shutil.copy(v0_path, os.path.join(sdir, "view_0.png"))
        shutil.copy(v1_path, os.path.join(sdir, "view_1.png"))
        with open(os.path.join(SMOKE_TMP_DIR, "specimens", "ground_truth.json"), "w") as f:
            json.dump(valid_gt, f)

        eval_report_path = os.path.join(SMOKE_TMP_DIR, "smoke_eval_report.json")
        eval_res = run_evaluation(
            specimens_dir=os.path.join(SMOKE_TMP_DIR, "specimens"),
            with_bedrock=False,
            output_report=eval_report_path,
        )
        assert eval_res is not None, "Evaluation run failed"
        assert os.path.exists(eval_report_path)

        print("[7/8] Testing run_authentic_benchmark.py and compare_benchmarks.py...")
        bench_out = os.path.join(SMOKE_TMP_DIR, "smoke_bench.json")
        from run_authentic_benchmark import run_benchmark

        run_benchmark(
            label="smoke_test",
            specimens_dir=os.path.join(SMOKE_TMP_DIR, "specimens"),
            threads=2,
            hourly_cost=0.0,
            output_path=bench_out,
        )
        assert os.path.exists(bench_out)

        from compare_benchmarks import compare_benchmarks

        comp_out = os.path.join(SMOKE_TMP_DIR, "smoke_comp.json")
        comp_res = compare_benchmarks(input_files=[bench_out], output_path=comp_out)
        assert comp_res is not None

        print("[8/8] Testing scripts/render_report_tables.py...")
        from scripts.render_report_tables import build_replacements

        reps = build_replacements()
        assert isinstance(reps, dict)

        print("\n========================================================")
        print("[SUCCESS] ALL END-TO-END SMOKE TESTS PASSED CLEANLY (EXIT 0)")
        print("========================================================")
        return 0

    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
