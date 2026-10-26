"""FastMCP server exposing active vision inspection tools over standard I/O (D7, Amendment 7).

Stateless Design Contract (Amendment 7):
Each tool call builds its own execution state purely from the arguments supplied to the
function call. No mutable global or session state is shared across requests, ensuring
safe concurrent invocation and zero cross-call contamination.
"""

import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from src.agent_orchestrator import BedrockMCPAgent
from src.vision_engine import PlanarActiveVisionEngine, load_image_exif_safe

mcp = FastMCP("SpecuLab-MCP")


@mcp.tool()
def rank_candidate_views(
    reference_path: str,
    candidate_paths: List[str],
    roi: Optional[List[int]] = None,
    glare_gate_pct: float = 2.0,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Stateless tool to evaluate and rank alternative candidate views against reference unhealed glare.

    Builds per-call state from arguments without mutating global variables (Amendment 7).
    """
    engine = PlanarActiveVisionEngine(config_path)
    agent = BedrockMCPAgent(config_path=config_path, engine=engine)

    resolved_roi = tuple(roi) if roi is not None else engine.DEFAULT_ROI

    # Rectify reference view and compute glare mask
    ref_bgr = load_image_exif_safe(reference_path)
    rect_ref, _ = engine.rectify_to_canonical_plane(ref_bgr)
    mask_ref, glare_pct, _, glare_meta = engine.compute_glare_mask(rect_ref, roi=resolved_roi)

    state = {
        "candidate_paths": [reference_path] + candidate_paths,
        "allowed_map": {os.path.abspath(p): p for p in [reference_path] + candidate_paths},
        "roi": resolved_roi,
        "glare_gate_pct": glare_gate_pct,
        "unhealed_mask": mask_ref,
        "ref_sharpness": glare_meta.get("glare_free_sharpness", 1.0),
        "optional_flags": engine.optional_features,
    }

    return agent._execute_tool("rank_candidate_views", {"candidate_paths": candidate_paths}, state)


@mcp.tool()
def fuse_candidate_view(
    reference_path: str,
    candidate_path: str,
    roi: Optional[List[int]] = None,
    rollback_previous_fusion: bool = False,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Stateless tool to fuse candidate view into reference canvas to heal specular glare.

    Enforces MAX_VIEW_ATTEMPTS = 2. Builds per-call state strictly from arguments.
    """
    engine = PlanarActiveVisionEngine(config_path)
    agent = BedrockMCPAgent(config_path=config_path, engine=engine)

    resolved_roi = tuple(roi) if roi is not None else engine.DEFAULT_ROI

    ref_bgr = load_image_exif_safe(reference_path)
    rect_ref, _ = engine.rectify_to_canonical_plane(ref_bgr)
    mask_ref, glare_pct, _, _ = engine.compute_glare_mask(rect_ref, roi=resolved_roi)

    state = {
        "roi": resolved_roi,
        "glare_gate_pct": float(engine.glare_cfg.get("default_gate_pct", 2.0)),
        "optional_flags": engine.optional_features,
        "acc_canvas": rect_ref,
        "unhealed_mask": mask_ref,
        "residual_glare_pct": glare_pct,
        "view_attempts": 0,
        "fused_paths": set(),
        "trace": [],
    }

    return agent._execute_tool(
        "fuse_candidate_view",
        {"candidate_path": candidate_path, "rollback_previous_fusion": rollback_previous_fusion},
        state,
    )


@mcp.tool()
def finalize_inspection_or_escalate(
    reference_path: str,
    fused_candidate_paths: Optional[List[str]] = None,
    decision_note: str = "MCP tool finalization",
    roi: Optional[List[int]] = None,
    glare_gate_pct: float = 2.0,
    morph_intensity_thresh: int = 30,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Stateless tool to finalize inspection with metrology or escalate unresolved occlusion.

    Builds state from input arguments and triggers Premature Finalize Guard if needed.
    """
    engine = PlanarActiveVisionEngine(config_path)
    agent = BedrockMCPAgent(config_path=config_path, engine=engine)

    resolved_roi = tuple(roi) if roi is not None else engine.DEFAULT_ROI

    ref_bgr = load_image_exif_safe(reference_path)
    rect_ref, _ = engine.rectify_to_canonical_plane(ref_bgr)
    mask_ref, glare_pct, _, _ = engine.compute_glare_mask(rect_ref, roi=resolved_roi)

    state = {
        "candidate_paths": [reference_path] + (fused_candidate_paths or []),
        "allowed_map": {
            os.path.abspath(p): p for p in [reference_path] + (fused_candidate_paths or [])
        },
        "roi": resolved_roi,
        "glare_gate_pct": glare_gate_pct,
        "morph_thresh": morph_intensity_thresh,
        "optional_flags": engine.optional_features,
        "acc_canvas": rect_ref,
        "unhealed_mask": mask_ref,
        "residual_glare_pct": glare_pct,
        "view_attempts": len(fused_candidate_paths) if fused_candidate_paths else 0,
        "fused_paths": set(fused_candidate_paths) if fused_candidate_paths else set(),
        "ranked_candidates": [],
        "trace": [],
    }

    # If candidates were provided, heal iteratively
    if fused_candidate_paths:
        for cpath in fused_candidate_paths:
            img_c = load_image_exif_safe(cpath)
            rect_c, _ = engine.rectify_to_canonical_plane(img_c)
            new_c, still_unh, res_g, _, _ = engine.heal_accumulated_canvas(
                state["acc_canvas"], state["unhealed_mask"], rect_c, roi=resolved_roi
            )
            state["acc_canvas"] = new_c
            state["unhealed_mask"] = still_unh
            state["residual_glare_pct"] = res_g

    return agent._execute_tool("finalize_inspection_or_escalate", {"decision_note": decision_note}, state)


if __name__ == "__main__":
    mcp.run()
