# SpecuLab-MCP: Autonomous Active Vision Metrology with FastMCP & Arm Graviton

SpecuLab-MCP is an industrial computer-vision system for high-specularity metallic foil surface inspection. By integrating OpenCV with the Model Context Protocol (FastMCP) and AWS Bedrock runtime, it autonomously eliminates specular glare through bounded, multi-view active homography fusion and detects micro-scale defects using a single unrigged morphological blackhat detector.

---

## 1. System Architecture & Core Capabilities

- **Canonical Metric Plane**: Every frame is rectified to a metric $2000 \times 800\text{ px}$ canvas ($100.0\text{ mm} \times 40.0\text{ mm}$ physical active card area at nominal $20.0\text{ px/mm}$).
- **Dual-Marker 8-Point & Single-Marker 4-Point Homography (`M-12`, `D3`)**: ArUco Marker ID 0 (`400x400 px`) and optional Marker ID 1 (`400x400 px`) provide metric scale and perspective invariance with bounds-checked subpixel refinement (`M-15`).
- **Two-Stage Hysteresis Glare Mask (`M-26`)**: Isolates saturated core reflection ($V \ge 240, S \le 35$) and penumbra halo ($V \ge \max(210, \text{median}(V)+35), S \le 50$) with morphological elliptical dilation.
- **Glare-Excluded Laplacian Sharpness Gate (`M-24`)**: Measures focal variance strictly outside the glare mask, penalizing out-of-focus views.
- **Dilated Feather Blending & Radiometric Gain (`M-11`, `M-22`, `FQ7`)**: Seamlessly stitches glare-free alternative pixels with 100% replacement in the core and border-median exposure balancing.
- **Single Unrigged Blackhat Detector (`M-13`, `D2`)**: Detects pits, dents, through-holes, and scratches with $61\times 61$ `cv2.MORPH_BLACKHAT` exclusively; `MORPH_TOPHAT` is completely absent.
- **FastMCP Stdio Server (Amendment 7)**: Exposes active vision tools (`rank_candidate_views`, `fuse_candidate_view`, `finalize_inspection_or_escalate`) with stateless per-call execution.
- **Hard Cap on Fusions**: Strictly caps candidate view fusions at `MAX_VIEW_ATTEMPTS = 2`.

---

## 2. Empirical Verification & Measured Data Placeholders

*All values below are populated directly from real JSON run outputs via `python3 scripts/render_report_tables.py`:*

### Runtime Environment & Hardware
- **OpenCV Version**: TODO_CV2_VERSION
- **Architecture & EC2 Instance**: TODO_ARCH_AND_INSTANCE
- **AMI ID**: TODO_AMI_ID
- **COOL Venv Path**: TODO_COOL_VEN_PATH
- **KleidiCV SIMD Compilation Flags**: TODO_KLEIDICV_FLAGS
- **Physical Ruler Measured Marker 0 Edge**: TODO_MEASURED_MARKER_MM mm

### Pilot Feasibility & Optical Peaks
- **Zone A (Through-Hole) Blackhat Peak**: TODO_PILOT_ZONE_A_PEAK
- **Zone B (Blind Pin Dent) Blackhat Peak**: TODO_PILOT_ZONE_B_PEAK
- **Zone C (Surface Scratch) Blackhat Peak**: TODO_PILOT_ZONE_C_PEAK
- **Selected Defect Subtypes**: TODO_SELECTED_DEFECT_SUBTYPES

### Dataset & Evaluation Summary
- **Total Physical Specimens**: TODO_TOTAL_SPECIMENS (Tuning: TODO_TUNING_COUNT, Held-Out: TODO_HELD_OUT_COUNT)
- **Class Balance**: Defective: TODO_DEFECTIVE_COUNT, Clean: TODO_CLEAN_COUNT
- **Locked Tuning Config**: Glare Gate = TODO_LOCKED_GATE_PCT, Blackhat Threshold = TODO_LOCKED_MORPH_THRESH

### Held-Out 6-Method Multi-Policy Comparison
TODO_HELD_OUT_6METHOD_MARKDOWN_TABLE

### Live Bedrock Agent Telemetry
- **Model ID**: TODO_BEDROCK_MODEL_ID
- **LLM Completion Rate**: TODO_LLM_COMPLETION_RATE
- **Fallback Rate**: TODO_LLM_FALLBACK_RATE
- **Verdict Agreement with Deterministic**: TODO_LLM_AGREEMENT_COUNT
- **Divergent View Choices**: TODO_LLM_DIVERGENT_VIEW_COUNT
- **Mean Cycle Latency**: TODO_LLM_MEAN_LATENCY_MS
- **Total Estimated Token Cost**: TODO_LLM_TOTAL_COST_USD

### Authentic Micro-Benchmark Profiles
#### 1-Thread SIMD KleidiCV Isolation Profile
TODO_BENCHMARK_1THREAD_MARKDOWN_TABLE

#### Multi-Thread Scaling Profile
TODO_BENCHMARK_NTHREAD_MARKDOWN_TABLE

- **On-Demand Hourly Reference (x86 c6i.xlarge)**: TODO_X86_EC2_HOURLY_USD
- **On-Demand Hourly Reference (Graviton c7g.xlarge)**: TODO_GRAVITON_EC2_HOURLY_USD
- **Empirical Benchmark Analysis**: TODO_BENCHMARK_HONEST_INTERPRETATION

---

## 3. Installation & Local Development

1. Create a dedicated virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install runtime dependencies:
```bash
pip install -r requirements.txt
# For local development and testing:
pip install opencv-python-headless numpy
```

3. Run diagnostic check:
```bash
python3 check_env.py
```

4. Generate the printable calibration card:
```bash
python3 scripts/generate_printable_card.py
```

5. Run test suite:
```bash
pytest -v
python3 smoke_test_e2e.py
```

6. Launch interactive Streamlit dashboard:
```bash
streamlit run app.py
```
