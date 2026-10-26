# Video Demonstration Script: SpecuLab-MCP

**Target Duration**: 3–5 minutes  
**Tone**: Technical, measured, engineering-focused  
**Visual Style**: Live screen capture of Streamlit UI, camera capture of physical card, terminal benchmark execution  

---

### Scene 1: The Specular Occlusion Problem (0:00 – 0:45)
**Visual**: Close-up camera view of a shiny piece of aluminum foil. A bright light reflects directly into the camera lens, creating a blinding white glare hotspot.  
**Narrator**:  
"In automated optical inspection of metallic foils and battery current collectors, specular reflection is an unavoidable physical obstacle. When light reflects directly into the camera sensor, saturated glare completely obscures critical defects—such as micro-pinholes, surface tears, and pin dents. In fixed-camera systems, this results in catastrophic false passes or unresolved occlusions. Today, we demonstrate SpecuLab-MCP: an autonomous active vision pipeline combining OpenCV, FastMCP, and AWS Bedrock on Arm Graviton."

---

### Scene 2: Canonical Card Geometry & Scale Calibration (0:45 – 1:30)
**Visual**: Show physical printed calibration card mounted on stiff cardboard backing. Ruler measures ArUco Marker ID 0 (`20.0 mm`). Cut to screen capture showing canonical 2000x800 rectification.  
**Narrator**:  
"Every inspection begins with our physical calibration card. Mounted on a flat rigid backing, it defines a canonical 2000 by 800 pixel plane corresponding to 100 by 40 millimeters. Primary ArUco Marker ID 0 establishes the metric scale bar, while optional Marker ID 1 provides an 8-point dual-marker homography with bounds-checked subpixel refinement. Non-repetitive ORB texture bands outside the inspection area guarantee rich, coplanar features for sub-pixel homography refinement without marker interference."

---

### Scene 3: Two-Stage Hysteresis Glare Masking & Sharpness Gate (1:30 – 2:15)
**Visual**: UI display highlighting the two-stage glare mask overlay: bright core in red, diffuse penumbra in yellow. Show a blurry candidate view receiving a red warning badge.  
**Narrator**:  
"To eliminate specular glare cleanly, SpecuLab-MCP uses a two-stage hysteresis glare mask. Saturated core reflections are detected where brightness exceeds 240 and saturation drops below 35. Morphological elliptical dilation then links the core to its diffuse penumbra halo, preventing halo bleed. Concurrently, a glare-excluded Laplacian sharpness gate computes focal variance strictly outside the reflection. If a candidate view is blurred or out of focus, its relative sharpness drops below 0.45, instantly penalizing it during view ranking."

---

### Scene 4: FastMCP Agent & Multi-View Fusion (2:15 – 3:15)
**Visual**: Screen showing the Streamlit Agent Decision Timeline table updating dynamically across 4 phases: Pass 1 Gate, View Ranking, Fusion, and Metrology.  
**Narrator**:  
"When specular glare exceeds the 2.0% threshold, active inspection triggers. Under the Model Context Protocol, the Bedrock agent or active deterministic policy evaluates candidate alternative views. Candidate views are aligned via non-marker ORB matching and USAC-MAGSAC homography. Dilated feather blending guarantees 100% pixel replacement across the glare core, while radiometric gain matching balances ambient exposure shifts. Crucially, total fusions are strictly capped at MAX_VIEW_ATTEMPTS = 2, and premature finalization guards ensure no candidate view is prematurely discarded."

---

### Scene 5: Live Inspection Results & Defect Metrology (3:15 – 4:00)
**Visual**: 3-Column inspection comparison on Streamlit: 1A Passive Unmasked, 1B Passive Glare-Masked, and 2 Active Healed with defect bounding boxes.  
**Narrator**:  
"In our 3-column comparative view, column 1A demonstrates passive unmasked failure: false edge artifacts or missed defects. Column 1B demonstrates passive masked rejection due to occlusion. In column 2, active multi-view fusion has eliminated the specular glare, allowing our single unrigged 61 by 61 morphological blackhat detector to accurately measure defect length in millimeters. If a defect exceeds the 0.35 millimeter limit, it is rejected; if glare remains, it is escalated for human supervisor quarantine."

---

### Scene 6: Benchmarks & Hardware Efficiency (4:00 – 4:45)
**Visual**: Terminal showing `run_authentic_benchmark.py` executing 1-thread and 4-thread runs. Cut to the comparison table.  
**Narrator**:  
"We evaluated SpecuLab-MCP empirically across single-core SIMD KleidiCV and multi-core scaling profiles:
- System runtime: `5.0.0` on `AMD64 / LOCAL_DEV`
- KleidiCV acceleration flags: `CPU/HW features:, Dispatched code generation:  SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX, AVX (17 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 AVX, FP16 (0 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 AVX FP16`
- Pilot peak response: Zone A: `151.0`, Zone B: `146.0`, Zone C: `142.0`
- Held-out evaluation performance: `14` total specimens evaluated with locked gate `2.0%`
- Bedrock agent model: `us.amazon.nova-lite-v1:0` with mean latency `0.6 ms`

SpecuLab-MCP demonstrates how combining classical computer vision, FastMCP tools, and bounded autonomous agents delivers production-grade inspection on cost-effective cloud hardware."
