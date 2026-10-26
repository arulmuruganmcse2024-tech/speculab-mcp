# Technical Report: SpecuLab-MCP Autonomous Active Vision Metrology

## 1. Executive Summary & Problem Statement

High-specularity surfaces such as rolled battery current collectors, pouch packaging, and cold-formed aerospace foils present a notorious challenge for automated optical inspection (AOI). Saturated directional reflections ("specular glare") blind fixed-view cameras, causing true pinholes, scratches, and through-dents to be either missed entirely (false passes) or obscured by specular halos (unresolved occlusions).

SpecuLab-MCP resolves this limitation via **Autonomous Active Vision**:
1. It projects multi-view candidate captures onto a unified canonical metric plane ($2000 \times 800\text{ px}$, $100.0\text{ mm} \times 40.0\text{ mm}$).
2. It detects glare regions with a two-stage hysteresis mask, isolating high-intensity reflection cores and their penumbras.
3. It selectively blends glare-free candidate pixels using non-marker ORB matching, USAC-MAGSAC homography refinement, and dilated feathering with radiometric gain matching.
4. It detects defects with a single unrigged $61\times 61$ blackhat morphological detector.
5. All tool operations are coordinated by an autonomous LLM agent or deterministic active policy over FastMCP, strictly capped at `MAX_VIEW_ATTEMPTS = 2`.

---

## 2. Canonical Card Geometry & Coordinate Frame

The physical card is mounted on a rigid backing and measured with a millimeter ruler to calibrate metric scale:
- Active Card Width: $100.0\text{ mm}$
- Active Card Height: $40.0\text{ mm}$
- Canonical Resolution: $2000 \times 800\text{ px}$
- Nominal Scale: $20.0\text{ px/mm}$
- Measured Marker 0 Edge: TODO_MEASURED_MARKER_MM mm

Layout along the X-axis:
- $x = 0..100\text{ px}$: Left border margin.
- $x = 100..500\text{ px}, y = 200..600\text{ px}$: Primary ArUco Marker ID 0 (`DICT_4X4_50`, $400\times 400\text{ px}$).
- $x = 600..1400\text{ px}, y = 200..600\text{ px}$: Central Foil Inspection Box ($800\times 400\text{ px}$ / $40.0\text{ mm} \times 20.0\text{ mm}$).
- $x = 1500..1900\text{ px}, y = 200..600\text{ px}$: Secondary ArUco Marker ID 1 ($400\times 400\text{ px}$).
- $y = 40..160\text{ px}$ and $y = 640..760\text{ px}$: Non-repetitive aperiodic geometric ORB texture bands.

---

## 3. Image Processing Formulations

### 3.1 Two-Stage Hysteresis Glare Mask
Saturated specular reflections exhibit a bright core surrounded by a diffuse penumbra halo:
- Core seed: $V \ge 240 \land S \le 35$
- Penumbra candidates: $V \ge \max(210, \text{median}(V) + 35) \land S \le 50$
- If core seed is empty, return $0.0\%$ glare. Otherwise:
$$\text{Full Seed} = \text{Core} \cup (\text{Dilate}(\text{Core}, K_{21}) \cap \text{Penumbra})$$
$$\text{Glare Mask} = \text{Dilate}(\text{Full Seed}, K_{9})$$

### 3.2 Glare-Excluded Laplacian Sharpness
Sharpness is computed strictly over non-glared foil pixels:
$$\text{Sharpness} = \text{Var}\left( \nabla^2 I(x, y) \mid (x, y) \in \text{ROI} \land \text{Mask}(x, y) = 0 \right)$$
Views with relative sharpness ratio $< 0.45$ relative to reference are flagged as `blur_degraded: True` and penalized during ranking.

### 3.3 Seamless Dilated Feather Blending
Newly healable pixels are dilated by an 11 px ellipse kernel before applying a $21\times 21$ Gaussian blur ($\sigma=5.0$). This ensures the blending weight $\alpha=1.0$ covers 100% of the specular core and rim, preventing false edge artifacts on clean foil.

### 3.4 Single Unrigged Blackhat Defect Metrology
Defects are detected exclusively with morphological blackhat:
$$\text{Blackhat} = \text{Closing}(I, K_{61\times 61}) - I$$
Defects are filtered by minimum area ($14.0\text{ px}$), solidity ($\ge 0.35$), peak contrast ($\ge 42.0$), and minimum defect length ($\ge 0.35\text{ mm}$).

---

## 4. Empirical Evaluation Results

### 4.1 System & Hardware Environment
- **OpenCV Version**: TODO_CV2_VERSION
- **Hardware Architecture**: TODO_ARCH_AND_INSTANCE
- **AMI ID**: TODO_AMI_ID
- **Venv Path**: TODO_COOL_VEN_PATH
- **KleidiCV Acceleration Flags**: TODO_KLEIDICV_FLAGS

### 4.2 Pilot Physics Feasibility
- **Zone A Peak Response**: TODO_PILOT_ZONE_A_PEAK
- **Zone B Peak Response**: TODO_PILOT_ZONE_B_PEAK
- **Zone C Peak Response**: TODO_PILOT_ZONE_C_PEAK
- **Validated Subtypes**: TODO_SELECTED_DEFECT_SUBTYPES

### 4.3 Exploratory 4x3 Parameter Tuning Sweep
- **Specimen Counts**: Total = TODO_TOTAL_SPECIMENS (Tuning = TODO_TUNING_COUNT, Held-Out = TODO_HELD_OUT_COUNT)
- **Class Counts**: Defective = TODO_DEFECTIVE_COUNT, Clean = TODO_CLEAN_COUNT
- **Locked Config**: Gate = TODO_LOCKED_GATE_PCT, Threshold = TODO_LOCKED_MORPH_THRESH

TODO_TUNING_SWEEP_MARKDOWN_TABLE

### 4.4 Held-Out 6-Method Multi-Policy Benchmark
TODO_HELD_OUT_6METHOD_MARKDOWN_TABLE

### 4.5 LLM Orchestrator Telemetry (AWS Bedrock)
- **Model**: TODO_BEDROCK_MODEL_ID
- **Completion Rate**: TODO_LLM_COMPLETION_RATE
- **Fallback Rate**: TODO_LLM_FALLBACK_RATE
- **Agreement with Deterministic Policy**: TODO_LLM_AGREEMENT_COUNT
- **Divergent View Selection Count**: TODO_LLM_DIVERGENT_VIEW_COUNT
- **Mean Cycle Latency**: TODO_LLM_MEAN_LATENCY_MS
- **Total Estimated Token Cost**: TODO_LLM_TOTAL_COST_USD

---

## 5. Micro-Benchmark Profiles

### 5.1 Single-Core 1-Thread Isolation Profile (KleidiCV SIMD)
TODO_BENCHMARK_1THREAD_MARKDOWN_TABLE

### 5.2 Multi-Core Scaling Profile
TODO_BENCHMARK_NTHREAD_MARKDOWN_TABLE

- **Hourly Instance Cost Reference (x86)**: TODO_X86_EC2_HOURLY_USD
- **Hourly Instance Cost Reference (Graviton)**: TODO_GRAVITON_EC2_HOURLY_USD
- **Empirical Interpretation**: TODO_BENCHMARK_HONEST_INTERPRETATION

---

## 6. Conclusions & Engineering Principles

1. **Zero Fabricated Metrics**: All performance indicators, recall percentages, latencies, and speedup ratios reflect authentic empirical measurements.
2. **Stateless FastMCP Decoupling**: Visual processing and decision orchestration remain completely stateless per call, guaranteeing reproducibility and safe multi-threaded concurrency.
3. **Bounded Autonomous Healing**: Hard limits (`MAX_VIEW_ATTEMPTS = 2`) and premature finalization guards prevent unbounded agent loops or false approvals.
