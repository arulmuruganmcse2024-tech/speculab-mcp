"""Streamlit interactive dashboard for SpecuLab-MCP active vision metrology.

Features:
- Stateless execution via cached engine & agent (M-09, M-20).
- 3-Column inspection comparison (Passive Unmasked, Passive Glare-Masked, Active Healed).
- 4-Row Plain-English Agent Decision Timeline Table above collapsible raw JSON trace.
- Human-in-the-Loop Supervisor Quarantine Approval writing to CloudWatch + local audit JSONL.
- 60-Second Judge Quick-Verify Tab reading authentic verification logs and SHA-256 hashes.
- Graceful empty specimens folder handling (Amendment 4).
"""

import glob
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from src.agent_orchestrator import BedrockMCPAgent
from src.vision_engine import PlanarActiveVisionEngine, get_opencv_runtime_metadata

st.set_page_config(
    page_title="SpecuLab-MCP | Autonomous Active Vision Inspection",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_stateless_system():
    """Initialize stateless vision engine and Bedrock agent."""
    engine = PlanarActiveVisionEngine("config.yaml")
    agent = BedrockMCPAgent("config.yaml", engine=engine)
    return engine, agent


def compute_sha256(filepath: str) -> str:
    """Compute file SHA-256 hash."""
    if not os.path.exists(filepath):
        return "N/A (File not found)"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    st.title("🔬 SpecuLab-MCP: Autonomous Active Vision Metrology")
    st.caption("Arm Graviton / AWS Bedrock / FastMCP Multi-View Specular Glare Elimination")
    st.info(
        "ℹ️ **Dataset Notice**: Automated Procedural Optical Testbed "
        "(Simulated Coplanar ArUco Card, Anisotropic Brushed Foil, Multi-Angle Perspective Tilt, and Specular Glare + Penumbra Halo)"
    )

    engine, agent = get_stateless_system()
    runtime_meta = get_opencv_runtime_metadata()

    # Sidebar controls
    st.sidebar.header("Inspection Controls")

    # Specimen selector
    specimens_dir = "assets/specimens"
    specimen_folders = sorted(glob.glob(os.path.join(specimens_dir, "specimen_*")))

    selected_specimen = None
    candidate_views: List[str] = []

    if not specimen_folders:
        st.sidebar.info("ℹ️ No physical specimens found in assets/specimens/ (Amendment 4).")
        # Fallback to test card image for demonstration
        if os.path.exists("assets/printable_test_card.png"):
            candidate_views = ["assets/printable_test_card.png"]
            selected_specimen = "printable_test_card_demo"
    else:
        folder_names = [os.path.basename(f) for f in specimen_folders]
        selected_folder_name = st.sidebar.selectbox("Select Specimen", folder_names)
        if selected_folder_name:
            selected_specimen = selected_folder_name
            selected_path = os.path.join(specimens_dir, selected_folder_name)
            candidate_views = sorted(
                glob.glob(os.path.join(selected_path, "*.jpg"))
                + glob.glob(os.path.join(selected_path, "*.png"))
                + glob.glob(os.path.join(selected_path, "*.jpeg"))
            )

    st.sidebar.markdown(f"**Loaded Views:** {len(candidate_views)}")

    policy_mode = st.sidebar.selectbox(
        "Orchestration Policy",
        [
            "bedrock_llm",
            "active_deterministic",
            "naive_view1",
            "greedy_lowest_glare",
        ],
        index=0,
    )

    glare_gate_pct = st.sidebar.slider(
        "Glare Acceptance Gate (%)",
        min_value=0.5,
        max_value=10.0,
        value=float(engine.glare_cfg.get("default_gate_pct", 2.0)),
        step=0.5,
    )

    morph_intensity_thresh = st.sidebar.slider(
        "Blackhat Morphology Threshold",
        min_value=15,
        max_value=50,
        value=int(engine.detection_cfg.get("morph_intensity_thresh", 30)),
        step=1,
    )

    with st.sidebar.expander("Foil Inspection ROI (px)"):
        def_roi = engine.DEFAULT_ROI
        roi_x = st.number_input("ROI X", value=def_roi[0], step=10)
        roi_y = st.number_input("ROI Y", value=def_roi[1], step=10)
        roi_w = st.number_input("ROI Width", value=def_roi[2], step=20)
        roi_h = st.number_input("ROI Height", value=def_roi[3], step=20)
        current_roi = (int(roi_x), int(roi_y), int(roi_w), int(roi_h))

    with st.sidebar.expander("Optional Features (Default Off)"):
        f_nbv = st.checkbox("Enable NBV Pose Prediction (M-23)", value=False)
        f_2d = st.checkbox("Enable 2D Illumination Gain (M-25)", value=False)
        f_ecc = st.checkbox("Enable ECC Subpixel Refinement (M-27)", value=False)
        f_roll = st.checkbox("Enable Seam Rollback (M-28)", value=False)
        optional_flags = {
            "enable_nbv_pose_prediction": f_nbv,
            "enable_2d_illumination_field_gain": f_2d,
            "enable_ecc_subpixel_refinement": f_ecc,
            "enable_seam_rollback": f_roll,
        }

    with st.sidebar.expander("Judge / VIP Access"):
        judge_passcode = st.text_input("VIP Judge Passcode", type="password")

    tab1, tab2 = st.tabs(["🚀 Live Active Inspection & Trace", "📋 60-Second Judge Quick-Verify & Logs"])

    with tab1:
        if not candidate_views:
            st.warning(
                "⚠️ No specimen images are available. Please place multi-view specimen folders "
                "inside `assets/specimens/specimen_001/` or run `scripts/pilot_physics_check.py`."
            )
            return

        run_btn = st.button("▶️ Run Active Multi-View Inspection Cycle", type="primary")

        if run_btn:
            with st.spinner("Executing active inspection cycle..."):
                cycle_results = agent.run_inspection_cycle(
                    candidate_paths=candidate_views,
                    policy_mode=policy_mode,
                    quota_bucket="public_anonymous",
                    judge_passcode=judge_passcode,
                    specimen_id=selected_specimen,
                    roi=current_roi,
                    glare_gate_pct=glare_gate_pct,
                    morph_intensity_thresh=morph_intensity_thresh,
                    optional_flags=optional_flags,
                )
            st.session_state["cycle_results"] = cycle_results

        if "cycle_results" in st.session_state:
            res = st.session_state["cycle_results"]

            # Prominent execution mode badge
            badge_text = res["execution_badge"]
            badge_color = "green" if "LIVE BEDROCK" in badge_text else "blue"
            st.markdown(
                f"<div style='padding:10px; border-radius:5px; background-color:#1e293b; color:#38bdf8; "
                f"font-weight:bold; font-size:16px; margin-bottom:15px; border-left: 5px solid {badge_color};'>"
                f"EXECUTION BADGE: {badge_text} | Total Latency: {res['cycle_latency_ms']} ms"
                f"</div>",
                unsafe_allow_html=True,
            )

            # 3-Column Visual Comparison
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("1A. Passive Unmasked")
                st.image(
                    cv2.cvtColor(res["annotated_passive_unmasked"], cv2.COLOR_BGR2RGB),
                    caption=f"Verdict: {res['passive_unmasked']['verdict']} | Defects: {len(res['passive_unmasked']['defects'])}",
                    use_column_width=True,
                )

            with col2:
                st.subheader("1B. Passive Glare-Masked")
                st.image(
                    cv2.cvtColor(res["annotated_passive_masked"], cv2.COLOR_BGR2RGB),
                    caption=f"Verdict: {res['passive_masked']['verdict']} | Defects: {len(res['passive_masked']['defects'])}",
                    use_column_width=True,
                )

            with col3:
                act = res["active_result"]
                st.subheader("2. Active Multi-View Healed")
                st.image(
                    cv2.cvtColor(res["annotated_active"], cv2.COLOR_BGR2RGB),
                    caption=f"Status: {act.get('status')} | Verdict: {act.get('verdict')} | Fusions: {act.get('view_attempts_used', 0)}/2",
                    use_column_width=True,
                )

            st.divider()

            # 4-Row Plain-English Agent Decision Timeline Table
            st.markdown("### 📋 Agent Decision Timeline")
            timeline_rows = [
                {
                    "Phase": "1. Pass 1 Glare Gate",
                    "Observation": f"Initial glare = {res['trace'][0].get('glare_pct', 0.0):.2f}% (Gate: {glare_gate_pct:.2f}%)",
                    "Action Taken": "Triggered active multi-view fusion" if res['trace'][0].get('trigger_active') else "Within tolerance, passive inspection accepted",
                },
                {
                    "Phase": "2. Candidate View Ranking",
                    "Observation": f"Ranked {len(candidate_views) - 1} alternative views against unhealed glare & sharpness gate",
                    "Action Taken": "Sorted by overlap, glare-free Laplacian sharpness, and tilt quality",
                },
                {
                    "Phase": "3. View Fusion & Radiometric Gain",
                    "Observation": f"Fused {act.get('view_attempts_used', 0)} candidate view(s) (Hard cap MAX_VIEW_ATTEMPTS = 2)",
                    "Action Taken": "ORB + USAC-MAGSAC alignment with dilated feathering (zero glare rim artifact)",
                },
                {
                    "Phase": "4. Final Metrology & Decision",
                    "Observation": f"Residual glare = {act.get('residual_glare_pct', 0.0):.2f}% | Max Defect: {act.get('max_defect_length_mm', 0.0):.2f} mm",
                    "Action Taken": f"Verdict: {act.get('verdict')} | Escalation: {act.get('requires_human_approval')}",
                },
            ]
            st.table(timeline_rows)

            # Human-in-the-Loop Quarantine Approval
            if act.get("requires_human_approval") or act.get("verdict") == "REJECT":
                st.warning("⚠️ High-Risk Defect or Occlusion detected. Requires Human Supervisor Approval.")
                supervisor_name = st.text_input("Supervisor Name / Badge ID", key="sup_id")
                approval_reason = st.text_area("Quarantine Override / Disposition Note", key="sup_note")
                if st.button("🔒 Approve Quarantine & Log to AWS CloudWatch"):
                    event = {
                        "event_type": "HUMAN_SUPERVISOR_QUARANTINE_DISPOSITION",
                        "supervisor": supervisor_name,
                        "disposition_note": approval_reason,
                        "specimen_id": selected_specimen,
                        "verdict": act.get("verdict"),
                        "status": act.get("status"),
                    }
                    agent.audit_logger.log_approval_event(event)
                    st.success("✅ Quarantine event logged to CloudWatch Log Group /aws/opencv2026/speculab-mcp and logs/audit_events.jsonl")

            with st.expander("🔍 View Raw Agent Execution Trace & Telemetry"):
                st.json(res["llm_telemetry"])
                st.json(res["trace"])

    with tab2:
        st.header("60-Second Judge Quick-Verify & Verification Artifacts")

        st.subheader("1. Runtime Environment & OpenCV Build Information")
        st.json(runtime_meta)

        st.subheader("2. System Artifacts & Cryptographic Checksums")
        tracked_files = [
            "config.yaml",
            "assets/printable_test_card.png",
            "logs/cool_verification.json",
            "evaluation_report.json",
            "benchmark_comparison.json",
        ]
        file_records = []
        for tf in tracked_files:
            file_records.append(
                {
                    "File": tf,
                    "Status": "Present" if os.path.exists(tf) else "Pending human hardware run",
                    "SHA-256": compute_sha256(tf),
                }
            )
        st.table(file_records)

        st.subheader("3. Benchmark Comparison Tables")
        if os.path.exists("benchmark_comparison.json"):
            with open("benchmark_comparison.json", "r", encoding="utf-8") as f:
                bcomp = json.load(f)
            md_tables = bcomp.get("markdown_tables", {})
            st.markdown("#### 1-Thread SIMD Isolation Profile")
            st.markdown(md_tables.get("table_1thread", "*No table data*"))
            st.markdown("#### Multi-Thread Scaling Profile")
            st.markdown(md_tables.get("table_nthread", "*No table data*"))
        else:
            st.info("ℹ️ Run scripts/run_3way_graviton.sh on AWS Graviton to generate authentic comparative benchmarks.")


if __name__ == "__main__":
    main()
