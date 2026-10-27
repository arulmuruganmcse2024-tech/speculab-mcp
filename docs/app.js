/**
 * SpecuLab-MCP Showcase Web App — Interactive Inspection Engine & Telemetry
 */

const SPECIMENS = {
  specimen_06: {
    id: "specimen_06",
    title: "Specimen 06: Overlap Trap",
    category: "Defective (Adversarial Overlap)",
    description: "View 0 has heavy glare covering defect at (950, 400). View 1 has a low 2.5% glare spot sitting right over (950, 400), trapping greedy methods. SpecuLab-MCP scores overlap = 0.0% with View 2, catching the flaw!",
    views: {
      view_0: "assets/specimens/specimen_06/view_0.jpg",
      view_1: "assets/specimens/specimen_06/view_1.jpg",
      view_2: "assets/specimens/specimen_06/view_2.jpg",
      healed: "assets/specimens/specimen_06/view_2.jpg"
    },
    telemetry: {
      ground_truth: "0.95 mm Puncture at (950, 400)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "16.4%",
      healed_glare: "0.0%",
      views_fused: "View 2 (View 1 Rejected for 94% Glare Overlap Trap)",
      passive_unmasked: { verdict: "REJECT", reason: "Blocked by Glare Trough", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Defect Masked Out (Missed)", class: "verdict-pass" },
      active_healed: { verdict: "REJECT", reason: "Defect Confirmed (0.95 mm >= 0.35 mm)", class: "verdict-reject" }
    },
    trace: [
      {
        step: 1,
        title: "Primary Frame Analysis",
        tool: "engine.compute_glare_mask(view_0)",
        desc: "Initial view_0 has 16.4% glare covering central foil ROI. Exceeds 2.0% threshold."
      },
      {
        step: 2,
        title: "Multi-Criteria View Ranking",
        tool: "fastmcp.rank_candidate_views([view_1, view_2])",
        desc: "View 1 has 94.2% glare overlap over defect zone (TRAP). View 2 has 0.0% overlap and 1.05 sharpness ratio."
      },
      {
        step: 3,
        title: "USAC-MAGSAC Homography Fusion",
        tool: "fastmcp.fuse_candidate_view(view_0, view_2)",
        desc: "Aligned View 2 onto canonical canvas. Applied gain matching and 11 px feather blend. Residual glare: 0.0%."
      },
      {
        step: 4,
        title: "Defect Metrology & Finalization",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Morphological Blackhat detected 0.95 mm puncture at (950, 400). Final Verdict: REJECT."
      }
    ]
  },
  specimen_07: {
    id: "specimen_07",
    title: "Specimen 07: Motion Blur Trap",
    category: "Defective (Motion Blur Trap)",
    description: "View 0 glare covers defect. View 1 has 0.0% glare, but is motion-blurred (Laplacian variance < 0.30), which erases high frequencies. SpecuLab-MCP penalizes blur and selects sharp View 2!",
    views: {
      view_0: "assets/specimens/specimen_07/view_0.jpg",
      view_1: "assets/specimens/specimen_07/view_1.jpg",
      view_2: "assets/specimens/specimen_07/view_2.jpg",
      healed: "assets/specimens/specimen_07/view_2.jpg"
    },
    telemetry: {
      ground_truth: "0.85 mm Crack at (1050, 400)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "18.2%",
      healed_glare: "0.0%",
      views_fused: "View 2 (View 1 Rejected for Motion Blur: Sharpness 0.19 < 0.30)",
      passive_unmasked: { verdict: "REJECT", reason: "Blocked by Glare Trough", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Defect Masked Out (Missed)", class: "verdict-pass" },
      active_healed: { verdict: "REJECT", reason: "Defect Confirmed (0.85 mm >= 0.35 mm)", class: "verdict-reject" }
    },
    trace: [
      {
        step: 1,
        title: "Primary Frame Analysis",
        tool: "engine.compute_glare_mask(view_0)",
        desc: "View 0 has 18.2% glare covering target foil. Exceeds 2.0% acceptance gate."
      },
      {
        step: 2,
        title: "Sharpness & Degradation Check",
        tool: "fastmcp.rank_candidate_views([view_1, view_2])",
        desc: "View 1 has 0% glare but relative sharpness 0.19 (motion blur trap). View 2 has sharpness 1.02 and 0% glare."
      },
      {
        step: 3,
        title: "Active View Fusion",
        tool: "fastmcp.fuse_candidate_view(view_0, view_2)",
        desc: "Fused sharp View 2. Dilated alpha feather blend eliminated boundary step. Residual glare: 0.0%."
      },
      {
        step: 4,
        title: "Inspection Verdict",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Extracted 0.85 mm crack (solidity 0.72, contrast 78 >= 42). Final Verdict: REJECT."
      }
    ]
  },
  specimen_08: {
    id: "specimen_08",
    title: "Specimen 08: 2-Pass Multi-View Fusion",
    category: "Defective (Compound Glare)",
    description: "View 0 has heavy glare across two zones. View 1 clears the right zone while leaving left glare. View 2 clears the left zone. Requires 2 sequential fusions to achieve 0.0% residual glare!",
    views: {
      view_0: "assets/specimens/specimen_08/view_0.jpg",
      view_1: "assets/specimens/specimen_08/view_1.jpg",
      view_2: "assets/specimens/specimen_08/view_2.jpg",
      healed: "assets/specimens/specimen_08/view_2.jpg"
    },
    telemetry: {
      ground_truth: "0.95 mm Puncture at (820, 400)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "22.5%",
      healed_glare: "0.0%",
      views_fused: "View 1 + View 2 (Two Sequential Fusions)",
      passive_unmasked: { verdict: "REJECT", reason: "Twin-Lobe Trough Response", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Defect Swallowed by Glare Mask", class: "verdict-pass" },
      active_healed: { verdict: "REJECT", reason: "Defect Confirmed (0.95 mm >= 0.35 mm)", class: "verdict-reject" }
    },
    trace: [
      {
        step: 1,
        title: "Pass 1 Fusion",
        tool: "fastmcp.fuse_candidate_view(view_0, view_1)",
        desc: "View 1 fused. Cleared right lobe, but residual glare remains 3.2% > 2.0% gate."
      },
      {
        step: 2,
        title: "Pass 2 Fusion (2-Pass Limit)",
        tool: "fastmcp.fuse_candidate_view(canvas, view_2)",
        desc: "Second fusion with View 2 cleared remaining left lobe. Residual glare: 0.0%."
      },
      {
        step: 3,
        title: "Final Metrology",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Detected 0.95 mm puncture under previously occluded left zone. Final Verdict: REJECT."
      }
    ]
  },
  specimen_09: {
    id: "specimen_09",
    title: "Specimen 09: Disagreement Trigger",
    category: "Defective (Compact Glare Spot)",
    description: "View 0 has a compact 1.4% twin-lobe glare spot (below 2.0% threshold) right over a defect. 1A rejects on glare, 1B passes by masking. Disagreement triggers active fusion with View 1!",
    views: {
      view_0: "assets/specimens/specimen_09/view_0.jpg",
      view_1: "assets/specimens/specimen_09/view_1.jpg",
      view_2: "assets/specimens/specimen_09/view_2.jpg",
      healed: "assets/specimens/specimen_09/view_1.jpg"
    },
    telemetry: {
      ground_truth: "0.85 mm Puncture at (1000, 400)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "1.4%",
      healed_glare: "0.0%",
      views_fused: "View 1 (Triggered by 1A != 1B Disagreement)",
      passive_unmasked: { verdict: "REJECT", reason: "Glare Peak Contrast >= 42", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Masked Out Spot (Missed)", class: "verdict-pass" },
      active_healed: { verdict: "REJECT", reason: "Defect Confirmed (0.85 mm >= 0.35 mm)", class: "verdict-reject" }
    },
    trace: [
      {
        step: 1,
        title: "Passive Disagreement Check",
        tool: "orchestrator.evaluate_disagreement()",
        desc: "1A_unmasked outputs REJECT on glare. 1B_masked outputs PASS. Glare Disagreement = TRUE."
      },
      {
        step: 2,
        title: "Mandatory Safety Fusion",
        tool: "fastmcp.fuse_candidate_view(view_0, view_1)",
        desc: "Disagreement forces active fusion despite glare <= 2.0%. Fused glare-free View 1."
      },
      {
        step: 3,
        title: "True Defect Recovery",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Revealed true 0.85 mm puncture beneath the 1.4% spot. Final Verdict: REJECT."
      }
    ]
  },
  specimen_10: {
    id: "specimen_10",
    title: "Specimen 10: Omnidirectional Glare",
    category: "Defective (Unresolvable Occlusion)",
    description: "Omnidirectional multi-head lighting produces heavy glare across all 3 views. SpecuLab-MCP safely halts the line with ESCALATE_UNRESOLVED_OCCLUSION for human supervisor sign-off.",
    views: {
      view_0: "assets/specimens/specimen_10/view_0.jpg",
      view_1: "assets/specimens/specimen_10/view_1.jpg",
      view_2: "assets/specimens/specimen_10/view_2.jpg",
      healed: "assets/specimens/specimen_10/view_0.jpg"
    },
    telemetry: {
      ground_truth: "1.20 mm Gouge (Heavily Glared)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "34.1%",
      healed_glare: "28.4%",
      views_fused: "View 1 + View 2 (Both Glare-Saturated)",
      passive_unmasked: { verdict: "REJECT", reason: "Massive Glare Artifact", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Silently Passed Entire Foil!", class: "verdict-pass" },
      active_healed: { verdict: "ESCALATE", reason: "Residual Glare 28.4% > 2.0%", class: "verdict-escalate" }
    },
    trace: [
      {
        step: 1,
        title: "Max Fusions Reached",
        tool: "orchestrator.run_active_cycle()",
        desc: "Fused 2 candidate views. Residual glare remains 28.4% (well above 2.0% threshold)."
      },
      {
        step: 2,
        title: "Supervisor Quarantine Action",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Part flagged as ESCALATE_UNRESOLVED_OCCLUSION. Production halted pending human review."
      },
      {
        step: 3,
        title: "CloudWatch Audit Log",
        tool: "CloudWatchAuditLogger.log_quarantine()",
        desc: "Logged tamper-proof audit record with SHA-256 image hashes and inspector sign-off."
      }
    ]
  },
  specimen_12: {
    id: "specimen_12",
    title: "Specimen 12: Clean Pristine Foil",
    category: "Clean Foil Held-Out Card",
    description: "Perfect defect-free metallic foil seal. Demonstrates 0/3 clean false rejects: passive unmasked false-rejects on glare, while SpecuLab-MCP correctly outputs PASS!",
    views: {
      view_0: "assets/specimens/specimen_12/view_0.jpg",
      view_1: "assets/specimens/specimen_12/view_0.jpg",
      view_2: "assets/specimens/specimen_12/view_0.jpg",
      healed: "assets/specimens/specimen_12/view_0.jpg"
    },
    telemetry: {
      ground_truth: "CLEAN FOIL (No Defect Present)",
      scale: "20.0 px/mm (ArUco Metric Calibrated)",
      view0_glare: "14.5%",
      healed_glare: "0.0%",
      views_fused: "View 1 (Healed Glare cleanly without false alarms)",
      passive_unmasked: { verdict: "REJECT", reason: "FALSE ALARM (Twin-Lobe Glare)", class: "verdict-reject" },
      passive_masked: { verdict: "PASS", reason: "Clean Foil Verified", class: "verdict-pass" },
      active_healed: { verdict: "PASS", reason: "Clean Foil Verified (0.0% False Rejects)", class: "verdict-pass" }
    },
    trace: [
      {
        step: 1,
        title: "1A False Reject Reproduction",
        tool: "engine.detect_defects(unmasked)",
        desc: "Blackhat closing across 22 px twin-lobe gap produces 86 contrast trough, false-rejecting clean foil!"
      },
      {
        step: 2,
        title: "Active Glare Healing",
        tool: "fastmcp.fuse_candidate_view(view_0, view_1)",
        desc: "11 px feather blending leaves zero rim artifact on clean foil. Blackhat output is completely flat."
      },
      {
        step: 3,
        title: "Verified Clean Release",
        tool: "fastmcp.finalize_inspection_or_escalate()",
        desc: "Max detected contour length 0.0 mm. Part passed cleanly: 0/3 False Rejects achieved."
      }
    ]
  }
};

let currentSpecimenId = "specimen_06";
let currentViewMode = "active_healed";

function updateSpecimenUI(specimenId) {
  currentSpecimenId = specimenId;
  const specimen = SPECIMENS[specimenId];
  if (!specimen) return;

  // Update pills
  document.querySelectorAll(".specimen-pill").forEach(pill => {
    pill.classList.toggle("active", pill.dataset.specimen === specimenId);
  });

  // Update images
  const baseImg = document.getElementById("canvas-base-img");
  const healedImg = document.getElementById("canvas-healed-img");

  if (currentViewMode === "view_0") {
    baseImg.src = specimen.views.view_0;
    healedImg.style.display = "none";
  } else if (currentViewMode === "view_1") {
    baseImg.src = specimen.views.view_1;
    healedImg.style.display = "none";
  } else if (currentViewMode === "view_2") {
    baseImg.src = specimen.views.view_2;
    healedImg.style.display = "none";
  } else {
    // Active Healed with interactive split slider
    baseImg.src = specimen.views.view_0;
    healedImg.src = specimen.views.healed;
    healedImg.style.display = "block";
  }

  // Update HUD
  document.getElementById("hud-ground-truth").textContent = specimen.telemetry.ground_truth;
  document.getElementById("hud-glare").textContent = currentViewMode === "view_0" ? specimen.telemetry.view0_glare : specimen.telemetry.healed_glare;
  document.getElementById("hud-views-fused").textContent = specimen.telemetry.views_fused;

  const verdictEl = document.getElementById("hud-verdict");
  const verdictData = specimen.telemetry[currentViewMode === "view_0" ? "passive_unmasked" : "active_healed"];
  verdictEl.textContent = `${verdictData.verdict} — ${verdictData.reason}`;
  verdictEl.className = `hud-verdict-pill ${verdictData.class}`;

  // Update Timeline
  renderTimeline(specimen.trace);
}

function renderTimeline(trace) {
  const container = document.getElementById("agent-timeline-container");
  if (!container) return;

  container.innerHTML = trace.map(item => `
    <div class="timeline-step">
      <div class="step-node">${item.step}</div>
      <div class="step-content">
        <div class="step-header">${item.title}</div>
        <div class="step-tool">${item.tool}</div>
        <div class="step-desc">${item.desc}</div>
      </div>
    </div>
  `).join("");
}

// Split Slider Drag Controller
function initSplitSlider() {
  const wrapper = document.querySelector(".inspection-canvas-wrapper");
  const handle = document.getElementById("split-handle");
  const overlay = document.getElementById("overlay-healed");
  if (!wrapper || !handle || !overlay) return;

  let isDragging = false;

  function setSliderPosition(clientX) {
    const rect = wrapper.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;
    handle.style.left = `${pct}%`;
    overlay.style.width = `${pct}%`;
  }

  handle.addEventListener("mousedown", () => { isDragging = true; });
  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (isDragging) setSliderPosition(e.clientX);
  });

  // Touch Support
  handle.addEventListener("touchstart", () => { isDragging = true; });
  window.addEventListener("touchend", () => { isDragging = false; });
  window.addEventListener("touchmove", (e) => {
    if (isDragging && e.touches[0]) setSliderPosition(e.touches[0].clientX);
  });
}

// Benchmark Profile Switcher
function initBenchmarkTabs() {
  const btn1T = document.getElementById("tab-bench-1thread");
  const btnNT = document.getElementById("tab-bench-nthread");
  const profileDesc = document.getElementById("bench-profile-desc");

  if (!btn1T || !btnNT) return;

  btn1T.addEventListener("click", () => {
    btn1T.classList.add("active");
    btnNT.classList.remove("active");
    profileDesc.textContent = "1-Thread SIMD Isolation: Evaluates single-core vector efficiency without multi-core scheduling noise.";
    updateBenchmarkBars("1thread");
  });

  btnNT.addEventListener("click", () => {
    btnNT.classList.add("active");
    btn1T.classList.remove("active");
    profileDesc.textContent = "4-Thread Scaling: Measures parallel production pipeline throughput under factory inspection workloads.";
    updateBenchmarkBars("nthread");
  });
}

const BENCHMARK_DATA = {
  "1thread": [
    { name: "USAC_MAGSAC & Fusion", opt: "104.8 ms", base: "125.1 ms", speedup: "1.19x faster (-16.2%)", widthOpt: 70, widthBase: 84 },
    { name: "ORB Extraction", opt: "15.9 ms", base: "20.7 ms", speedup: "1.30x faster (-23.1%)", widthOpt: 45, widthBase: 58 },
    { name: "ArUco Detect & Metric", opt: "21.8 ms", base: "25.7 ms", speedup: "1.18x faster (-15.3%)", widthOpt: 52, widthBase: 61 },
    { name: "cv2.warpPerspective", opt: "21.3 ms", base: "24.9 ms", speedup: "1.17x faster (-14.6%)", widthOpt: 51, widthBase: 60 },
    { name: "Total Pipeline Latency", opt: "204.5 ms", base: "237.5 ms", speedup: "1.16x faster (-13.9%)", widthOpt: 75, widthBase: 88 }
  ],
  "nthread": [
    { name: "USAC_MAGSAC & Fusion", opt: "77.4 ms", base: "84.9 ms", speedup: "1.10x faster (-8.8%)", widthOpt: 62, widthBase: 68 },
    { name: "ORB Extraction", opt: "17.0 ms", base: "19.1 ms", speedup: "1.12x faster (-10.9%)", widthOpt: 44, widthBase: 49 },
    { name: "ArUco Detect & Metric", opt: "12.6 ms", base: "13.3 ms", speedup: "1.06x faster", widthOpt: 40, widthBase: 42 },
    { name: "cv2.warpPerspective", opt: "12.2 ms", base: "12.4 ms", speedup: "1.02x (Parity)", widthOpt: 39, widthBase: 40 },
    { name: "Total Pipeline Latency", opt: "157.9 ms", base: "167.6 ms", speedup: "1.06x faster (6.3 FPS)", widthOpt: 68, widthBase: 73 }
  ]
};

function updateBenchmarkBars(profile) {
  const container = document.getElementById("benchmark-bars-container");
  if (!container) return;

  const data = BENCHMARK_DATA[profile] || BENCHMARK_DATA["1thread"];
  container.innerHTML = data.map(item => `
    <div class="bar-row">
      <div class="bar-meta">
        <span style="color:#fff;">${item.name}</span>
        <span style="color:var(--accent-cyan);">${item.opt} <span style="color:var(--text-dim);">(vs ${item.base})</span> • ${item.speedup}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill cyan" style="width: ${item.widthOpt}%;"></div>
      </div>
    </div>
  `).join("");
}

// Initialise on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  // Specimen Pills click
  document.querySelectorAll(".specimen-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      updateSpecimenUI(pill.dataset.specimen);
    });
  });

  // View Mode Tabs click
  document.querySelectorAll(".mode-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentViewMode = tab.dataset.mode;
      updateSpecimenUI(currentSpecimenId);
    });
  });

  initSplitSlider();
  initBenchmarkTabs();
  updateSpecimenUI("specimen_06");
  updateBenchmarkBars("1thread");
});
