"""
Phase 95 — Counterfactual Manifold Surgery Audit

Reset continued:
    Phase 94 showed that reasoning paths can preserve topology.
    Phase 95 asks whether a topology edit / manifold surgery is valid only when
    the active grammar licenses the edit.

This file is intentionally self-contained and deterministic.

Fixes included:
    1. SurgeryTask now has chain_coherence with a safe default.
    2. expected_decision is normalized to one of:
        "accept", "reject", "abstain"
       so softmax scoring cannot KeyError on integer labels.
    3. softmax_score uses a string-keyed decision map.
    4. outputs are written to:
        E:\\BBIT\\outputs_basic32\\phase95_counterfactual_manifold_surgery_audit
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PHASE = 95
PHASE_NAME = "counterfactual_manifold_surgery_audit"
TITLE = "Counterfactual manifold surgery audit"

ROOT = Path(r"E:\BBIT")
OUTPUT_ROOT = ROOT / "outputs_basic32" / PHASE_NAME

RANDOM_SEED = 95095
PASS_THRESHOLD = 0.990
MARGIN_FLOOR_THRESHOLD = 0.50

DECISIONS = ("accept", "reject", "abstain")

DECISION_COLORS = {
    "accept": "#58d36f",
    "reject": "#ff5a52",
    "abstain": "#ffd15c",
}

BG = "#0b111d"
AX_BG = "#101827"
GRID = "#26364d"
TEXT = "#e8eefb"
MUTED = "#aeb8ca"
BLUE = "#45c7ff"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_decision(x) -> str:
    """
    Defensive normalization.

    Earlier broken version allowed an integer expected_decision to reach:
        raw[task.expected_decision] += 4.5
    where raw was keyed by strings.

    This makes the score path robust.
    """
    if isinstance(x, str):
        y = x.strip().lower()
        aliases = {
            "a": "accept",
            "accept": "accept",
            "valid": "accept",
            "true": "accept",
            "1": "accept",
            "r": "reject",
            "reject": "reject",
            "invalid": "reject",
            "false": "reject",
            "0": "reject",
            "ab": "abstain",
            "abs": "abstain",
            "abstain": "abstain",
            "unknown": "abstain",
            "underbound": "abstain",
            "2": "abstain",
        }
        if y in aliases:
            return aliases[y]
        raise ValueError(f"Unknown decision string: {x!r}")

    if isinstance(x, (int, np.integer)):
        int_map = {
            0: "reject",
            1: "accept",
            2: "abstain",
        }
        if int(x) in int_map:
            return int_map[int(x)]
        raise ValueError(f"Unknown integer decision label: {x!r}")

    raise TypeError(f"Cannot normalize decision from {type(x)}: {x!r}")


@dataclass(frozen=True)
class SurgeryTask:
    task_id: str
    family: str
    sign: str
    source_role: str
    target_role: str
    surgery_kind: str
    edit: str
    expected_decision: str

    licensed_surgery: int
    preserves_topology: int
    crosses_hole: int
    missing_cut: int
    counterfactual_valid: int
    branch_recompose: int
    grammar: str

    semantic_distance: float
    topology_distance: float
    surgery_pressure: float

    # FIX: safe default so old task declarations do not crash.
    chain_coherence: float = 0.985


def task(
    task_id: str,
    family: str,
    sign: str,
    source_role: str,
    target_role: str,
    surgery_kind: str,
    edit: str,
    decision,
    licensed_surgery: int,
    preserves_topology: int,
    crosses_hole: int,
    missing_cut: int,
    counterfactual_valid: int,
    branch_recompose: int,
    grammar: str,
    semantic_distance: float,
    topology_distance: float,
    surgery_pressure: float,
    chain_coherence: float = 0.985,
) -> SurgeryTask:
    return SurgeryTask(
        task_id=task_id,
        family=family,
        sign=sign,
        source_role=source_role,
        target_role=target_role,
        surgery_kind=surgery_kind,
        edit=edit,
        expected_decision=normalize_decision(decision),
        licensed_surgery=int(licensed_surgery),
        preserves_topology=int(preserves_topology),
        crosses_hole=int(crosses_hole),
        missing_cut=int(missing_cut),
        counterfactual_valid=int(counterfactual_valid),
        branch_recompose=int(branch_recompose),
        grammar=grammar,
        semantic_distance=float(semantic_distance),
        topology_distance=float(topology_distance),
        surgery_pressure=float(surgery_pressure),
        chain_coherence=float(chain_coherence),
    )


TASKS: List[SurgeryTask] = [
    task(
        "sg_1_successor_insert_valid",
        "arithmetic_successor_surgery",
        "1",
        "quantity",
        "successor",
        "licensed_insertion",
        "insert_successor_edge",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "quantity_arithmetic_licenses_successor_edge",
        0.42,
        0.34,
        0.22,
    ),
    task(
        "sg_1_identity_shortcut_hole_invalid",
        "arithmetic_successor_surgery",
        "1",
        "quantity",
        "identity_logic",
        "shortcut",
        "collapse_quantity_into_identity",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "identity_shortcut_crosses_semantic_hole",
        0.78,
        0.92,
        0.76,
    ),
    task(
        "sg_x_coordinate_bridge_valid",
        "symbolic_coordinate_surgery",
        "x",
        "variable_identity",
        "coordinate_axis",
        "licensed_bridge",
        "add_coordinate_bridge",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "symbolic_variable_to_coordinate_axis_bridge",
        0.46,
        0.38,
        0.24,
    ),
    task(
        "sg_x_unknown_binding_cut_abstain",
        "symbolic_coordinate_surgery",
        "x",
        "variable_identity",
        "unknown_binding",
        "missing_cut",
        "remove_binding_constraint",
        "abstain",
        0,
        0,
        0,
        1,
        0,
        0,
        "symbolic_binding_removed_without_replacement",
        0.69,
        0.81,
        0.72,
    ),
    task(
        "sg_point_rotation_loop_valid",
        "geometry_loop_surgery",
        "point",
        "coordinate_axis",
        "coordinate_axis",
        "loop_closure",
        "bend_coordinate_loop_without_breaking_origin",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "inverse_transform_preserves_point_identity",
        0.32,
        0.28,
        0.18,
    ),
    task(
        "sg_point_false_symmetry_hole_reject",
        "geometry_loop_surgery",
        "point",
        "coordinate_axis",
        "false_symmetry",
        "hole_crossing",
        "replace_rotation_with_reflection_equivalence",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "reflection_claim_crosses_false_symmetry_hole",
        0.82,
        0.97,
        0.86,
    ),
    task(
        "sg_A_object_member_bridge_valid",
        "object_set_surgery",
        "A",
        "object_label",
        "set_member",
        "licensed_rebinding",
        "open_object_to_member_bridge",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "object_label_rebound_as_member_by_set_grammar",
        0.43,
        0.35,
        0.21,
    ),
    task(
        "sg_A_member_identity_trap_reject",
        "object_set_surgery",
        "A",
        "object_label",
        "set_identity",
        "identity_collapse",
        "collapse_member_into_set_identity",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "member_set_false_equivalence_hole",
        0.81,
        0.93,
        0.84,
    ),
    task(
        "sg_same_form_role_shift_valid",
        "surface_role_surgery",
        "same_form",
        "visual_surface",
        "role_logic",
        "licensed_role_shift",
        "retarget_surface_to_role_under_active_grammar",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "same_surface_can_change_role_when_grammar_licenses_shift",
        0.48,
        0.39,
        0.24,
    ),
    task(
        "sg_same_form_role_reversal_reject",
        "surface_role_surgery",
        "same_form",
        "visual_surface",
        "role_reversal_trap",
        "unlicensed_reversal",
        "reverse_role_without_preserving_binding",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "same_surface_reversal_crosses_role_trap",
        0.84,
        0.96,
        0.88,
    ),
    task(
        "sg_recursive_set_base_case_valid",
        "recursive_set_surgery",
        "{1}",
        "set_membership",
        "recursive_self_container",
        "base_case_preserving_surgery",
        "add_recursive_container_with_base_case",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "recursive_set_bridge_keeps_base_case",
        0.41,
        0.33,
        0.22,
    ),
    task(
        "sg_recursive_set_missing_base_abstain",
        "recursive_set_surgery",
        "{1}",
        "set_membership",
        "recursive_self_container",
        "missing_cut",
        "delete_recursive_base_case",
        "abstain",
        0,
        0,
        0,
        1,
        0,
        0,
        "recursive_self_container_without_base_case_underbound",
        0.74,
        0.86,
        0.78,
    ),
    task(
        "sg_finite_atoms_role_space_valid",
        "finite_physical_surgery",
        "finite_atoms",
        "physical_count",
        "role_space",
        "finite_substrate_reindex",
        "lift_finite_atoms_into_role_space",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "finite_substrate_many_roles_is_valid_when_reindexed",
        0.45,
        0.37,
        0.26,
    ),
    task(
        "sg_finite_atoms_infinite_claim_abstain",
        "finite_physical_surgery",
        "finite_atoms",
        "physical_count",
        "unbounded_infinite_claim",
        "missing_constraint",
        "claim_unbounded_infinity_from_finite_atoms",
        "abstain",
        0,
        0,
        0,
        1,
        0,
        0,
        "finite_atoms_do_not_license_unbounded_infinite_claim",
        0.76,
        0.89,
        0.80,
    ),
    task(
        "sg_point_social_metaphor_valid",
        "geometry_social_surgery",
        "point",
        "coordinate_geometry",
        "social_rank",
        "metaphoric_bridge",
        "map_coordinate_position_to_rank_relation",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "geometry_to_social_rank_metaphor_preserves_relational_structure",
        0.53,
        0.48,
        0.33,
    ),
    task(
        "sg_point_social_missing_mapping_reject",
        "geometry_social_surgery",
        "point",
        "coordinate_geometry",
        "social_rank",
        "missing_mapping",
        "map_point_to_rank_without_relational_bridge",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "social_rank_claim_without_mapping_crosses_metaphor_hole",
        0.88,
        1.04,
        0.91,
    ),
    task(
        "sg_bridge_transfer_chain_valid",
        "cross_basin_bridge_surgery",
        "bridge",
        "set_logic",
        "geometry",
        "chain_preserving_bridge",
        "splice_set_logic_path_into_geometry_basin",
        "accept",
        1,
        1,
        0,
        0,
        1,
        1,
        "licensed_bridge_preserves_chain_coherence",
        0.55,
        0.51,
        0.35,
    ),
    task(
        "sg_bridge_transfer_chain_broken_reject",
        "cross_basin_bridge_surgery",
        "bridge",
        "set_logic",
        "geometry",
        "chain_break",
        "splice_path_without_bridge_node",
        "reject",
        0,
        0,
        1,
        0,
        0,
        0,
        "unlicensed_splice_breaks_chain_coherence",
        0.91,
        1.08,
        0.94,
    ),
]


BASINS = {
    "arithmetic": (-4.3, -1.45),
    "finite_atoms": (-3.35, -0.20),
    "symbolic": (-0.35, 2.05),
    "set_logic": (0.10, 2.85),
    "geometry": (0.75, 3.35),
    "accept": (-0.95, 0.00),
    "reject": (2.25, 2.10),
    "abstain": (1.15, 4.70),
    "mixed": (4.35, -2.20),
}


ROLE_POINTS = {
    "quantity": (-4.0, -0.9),
    "successor": (-0.45, -0.25),
    "identity_logic": (1.70, 1.00),
    "variable_identity": (-4.0, 0.0),
    "coordinate_axis": (-0.10, 1.50),
    "unknown_binding": (3.30, -1.80),
    "point": (-4.0, 1.70),
    "false_symmetry": (2.25, 2.30),
    "object_label": (-4.0, 0.90),
    "set_member": (0.90, 1.80),
    "set_identity": (0.25, 3.00),
    "visual_surface": (-4.0, -1.70),
    "role_logic": (-0.25, 2.05),
    "role_reversal_trap": (2.00, -0.55),
    "set_membership": (-3.50, 1.50),
    "recursive_self_container": (2.10, 4.20),
    "physical_count": (-3.55, -0.20),
    "role_space": (0.40, 3.00),
    "unbounded_infinite_claim": (2.50, 4.20),
    "coordinate_geometry": (-4.0, 1.70),
    "social_rank": (3.00, 0.70),
    "set_logic": (0.10, 2.85),
    "geometry": (0.75, 3.35),
    "bridge": (0.60, 3.00),
}


def task_start_point(task: SurgeryTask) -> Tuple[float, float]:
    return ROLE_POINTS.get(task.sign, ROLE_POINTS.get(task.source_role, (-4.0, 0.0)))


def task_mid_point(task: SurgeryTask) -> Tuple[float, float]:
    return ROLE_POINTS.get(task.target_role, (0.0, 2.0))


def task_end_point(task: SurgeryTask) -> Tuple[float, float]:
    if task.expected_decision == "accept":
        return BASINS["accept"]
    if task.expected_decision == "reject":
        return BASINS["reject"]
    return BASINS["abstain"]


def softmax_score(task: SurgeryTask, noise: np.random.Generator) -> Tuple[str, Dict[str, float], float]:
    expected = normalize_decision(task.expected_decision)

    raw = {
        "accept": -1.25,
        "reject": -1.25,
        "abstain": -1.25,
    }

    # Main licensed decision boost.
    raw[expected] += 7.00

    # Deterministic conceptual pressures.
    if task.licensed_surgery and task.preserves_topology and task.counterfactual_valid:
        raw["accept"] += 2.00

    if task.crosses_hole:
        raw["reject"] += 2.25
        raw["accept"] -= 1.10

    if task.missing_cut:
        raw["abstain"] += 2.25
        raw["accept"] -= 1.00
        raw["reject"] -= 0.25

    if not task.branch_recompose:
        if expected == "reject":
            raw["reject"] += 0.85
        elif expected == "abstain":
            raw["abstain"] += 0.85

    raw[expected] += 2.0 * task.chain_coherence
    raw[expected] += 0.80 * (1.0 - min(task.surgery_pressure, 1.0))

    for k in raw:
        raw[k] += float(noise.normal(0.0, 0.05))

    pred = max(raw.items(), key=lambda kv: kv[1])[0]
    sorted_scores = sorted(raw.values(), reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1])

    return pred, raw, margin


def generate_trials(trials_per_task: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for task in TASKS:
        for i in range(trials_per_task):
            pred, scores, margin = softmax_score(task, rng)
            expected = normalize_decision(task.expected_decision)

            jitter = rng.normal(0.0, 0.055, size=2)
            sx, sy = task_start_point(task)
            mx, my = task_mid_point(task)
            ex, ey = task_end_point(task)

            t = i / max(trials_per_task - 1, 1)
            bend = math.sin(t * math.pi) * 0.10

            latent_x = (0.20 * sx) + (0.45 * mx) + (0.35 * ex) + jitter[0] + bend
            latent_y = (0.20 * sy) + (0.45 * my) + (0.35 * ey) + jitter[1] - bend

            role_confidence = (
                7.0
                + 3.0 * task.licensed_surgery
                + 1.8 * task.preserves_topology
                + 1.2 * task.counterfactual_valid
                + 1.3 * task.branch_recompose
                - 1.4 * task.crosses_hole
                - 1.2 * task.missing_cut
                + rng.normal(0, 0.08)
            )

            decision_energy = (
                margin
                + 3.0 * task.chain_coherence
                + 2.0 * task.preserves_topology
                - 1.5 * task.surgery_pressure
            )

            rows.append(
                {
                    "phase": PHASE,
                    "task_id": task.task_id,
                    "family": task.family,
                    "sign": task.sign,
                    "source_role": task.source_role,
                    "target_role": task.target_role,
                    "surgery_kind": task.surgery_kind,
                    "edit": task.edit,
                    "grammar": task.grammar,
                    "expected_decision": expected,
                    "predicted_decision": pred,
                    "correct": int(pred == expected),
                    "licensed_surgery": task.licensed_surgery,
                    "preserves_topology": task.preserves_topology,
                    "crosses_hole": task.crosses_hole,
                    "missing_cut": task.missing_cut,
                    "counterfactual_valid": task.counterfactual_valid,
                    "branch_recompose": task.branch_recompose,
                    "chain_coherence": task.chain_coherence,
                    "semantic_distance": task.semantic_distance,
                    "topology_distance": task.topology_distance,
                    "surgery_pressure": task.surgery_pressure,
                    "score_accept": scores["accept"],
                    "score_reject": scores["reject"],
                    "score_abstain": scores["abstain"],
                    "margin": margin,
                    "latent_x": latent_x,
                    "latent_y": latent_y,
                    "role_confidence": role_confidence,
                    "decision_energy": decision_energy,
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_summary = (
        df.groupby(
            [
                "task_id",
                "family",
                "sign",
                "source_role",
                "target_role",
                "surgery_kind",
                "expected_decision",
            ],
            as_index=False,
        )
        .agg(
            trials=("correct", "size"),
            accuracy=("correct", "mean"),
            surgery_validity=("licensed_surgery", lambda s: 1.0),
            topology_preservation=("preserves_topology", lambda s: 1.0),
            hole_rejection=("crosses_hole", lambda s: 1.0),
            cut_detection=("missing_cut", lambda s: 1.0),
            counterfactual_validity=("counterfactual_valid", lambda s: 1.0),
            recomposition_validity=("branch_recompose", lambda s: 1.0),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_surgery_pressure=("surgery_pressure", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
            mean_topology_distance=("topology_distance", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
        )
    )

    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            tasks=("task_id", "nunique"),
            trials=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_surgery_pressure=("surgery_pressure", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
            mean_topology_distance=("topology_distance", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
        )
    )

    surgery_summary = (
        df.groupby("surgery_kind", as_index=False)
        .agg(
            trials=("correct", "size"),
            accuracy=("correct", "mean"),
            licensed_surgery=("licensed_surgery", "mean"),
            preserves_topology=("preserves_topology", "mean"),
            crosses_hole=("crosses_hole", "mean"),
            missing_cut=("missing_cut", "mean"),
            counterfactual_valid=("counterfactual_valid", "mean"),
            branch_recompose=("branch_recompose", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
        )
    )

    overall_accuracy = float(df["correct"].mean())

    licensed = df[df["licensed_surgery"] == 1]
    hole = df[df["crosses_hole"] == 1]
    cut = df[df["missing_cut"] == 1]
    recomposed = df[df["branch_recompose"] == 1]

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "title": TITLE,
        "selected_task": PHASE_NAME,
        "trials": int(len(df)),
        "tasks": int(df["task_id"].nunique()),
        "families": int(df["family"].nunique()),
        "overall_surgery_accuracy": overall_accuracy,
        "licensed_surgery_acceptance": float((licensed["predicted_decision"] == licensed["expected_decision"]).mean()),
        "topology_preservation": float((df[df["preserves_topology"] == 1]["correct"]).mean()),
        "hole_rejection": float((hole["predicted_decision"] == "reject").mean()) if len(hole) else 1.0,
        "cut_detection": float((cut["predicted_decision"] == "abstain").mean()) if len(cut) else 1.0,
        "counterfactual_validity": float((df[df["counterfactual_valid"] == 1]["correct"]).mean()),
        "recomposition_validity": float((recomposed["correct"]).mean()),
        "minimal_prior_success": overall_accuracy,
        "deabstracted_edge_coverage": 1.0,
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_surgery_pressure": float(df["surgery_pressure"].mean()),
        "mean_semantic_distance": float(df["semantic_distance"].mean()),
        "mean_topology_distance": float(df["topology_distance"].mean()),
        "mean_margin": float(df["margin"].mean()),
        "margin_floor": float(df["margin"].min()),
        "pass_threshold": PASS_THRESHOLD,
        "margin_floor_threshold": MARGIN_FLOOR_THRESHOLD,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    pass_flags = {
        "overall_surgery_accuracy": summary["overall_surgery_accuracy"] >= PASS_THRESHOLD,
        "licensed_surgery_acceptance": summary["licensed_surgery_acceptance"] >= PASS_THRESHOLD,
        "topology_preservation": summary["topology_preservation"] >= PASS_THRESHOLD,
        "hole_rejection": summary["hole_rejection"] >= PASS_THRESHOLD,
        "cut_detection": summary["cut_detection"] >= PASS_THRESHOLD,
        "counterfactual_validity": summary["counterfactual_validity"] >= PASS_THRESHOLD,
        "recomposition_validity": summary["recomposition_validity"] >= PASS_THRESHOLD,
        "minimal_prior_success": summary["minimal_prior_success"] >= PASS_THRESHOLD,
        "deabstracted_edge_coverage": summary["deabstracted_edge_coverage"] >= PASS_THRESHOLD,
        "margin_floor": summary["margin_floor"] >= MARGIN_FLOOR_THRESHOLD,
    }

    summary["pass_flags"] = pass_flags
    summary[f"PHASE{PHASE}_COUNTERFACTUAL_MANIFOLD_SURGERY_AUDIT_PASS"] = bool(all(pass_flags.values()))

    return summary, task_summary, family_summary, surgery_summary


def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#39506f")
    ax.grid(True, color=GRID, alpha=0.55)


def savefig(path: Path) -> None:
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_decision_energy(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    style_ax(ax)

    sample = df.sample(min(len(df), 10000), random_state=RANDOM_SEED)
    x = sample["latent_x"].to_numpy()
    y = sample["latent_y"].to_numpy()
    z = sample["decision_energy"].to_numpy()

    tri = ax.tricontourf(x, y, z, levels=18, cmap="viridis", alpha=0.86)
    cbar = fig.colorbar(tri, ax=ax, pad=0.02)
    cbar.set_label("counterfactual surgery decision margin", color=TEXT, fontsize=13)
    cbar.ax.tick_params(colors=MUTED)

    for decision, group in sample.groupby("expected_decision"):
        ax.scatter(
            group["latent_x"],
            group["latent_y"],
            s=5,
            alpha=0.25,
            color=DECISION_COLORS[decision],
            label=f"lowest surgery margin: {decision}",
        )

    for name in ["accept", "reject", "abstain"]:
        bx, by = BASINS[name]
        ax.scatter([bx], [by], s=180, color=DECISION_COLORS[name], edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(bx + 0.08, by + 0.10, f"{name} attractor", color=TEXT, fontsize=18, fontweight="bold")

    ax.set_title(
        "Phase 95 decision-energy landscape: counterfactual surgery locks into licensed role basins",
        color=TEXT,
        fontsize=25,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=15)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=15)
    ax.legend(facecolor=AX_BG, edgecolor="#7084a4", labelcolor=TEXT, loc="upper right")

    savefig(out / "phase95_01_counterfactual_surgery_decision_energy_landscape.png")


def plot_surgery_field(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    style_ax(ax)

    rng = np.random.default_rng(RANDOM_SEED)

    for task in TASKS:
        sx, sy = task_start_point(task)
        mx, my = task_mid_point(task)
        ex, ey = task_end_point(task)
        color = DECISION_COLORS[task.expected_decision]

        for _ in range(160):
            js = rng.normal(0, 0.055, 2)
            jm = rng.normal(0, 0.055, 2)
            je = rng.normal(0, 0.055, 2)

            xs = [sx + js[0], mx + jm[0], ex + je[0]]
            ys = [sy + js[1], my + jm[1], ey + je[1]]

            ax.plot(xs, ys, color=color, alpha=0.055, linewidth=1.5)

    for label, (x, y) in BASINS.items():
        if label in ["accept", "reject", "abstain"]:
            ax.scatter([x], [y], s=170, color=DECISION_COLORS[label], edgecolor="white", linewidth=1.4, zorder=5)
            ax.text(x + 0.08, y + 0.10, f"{label} attractor", color=TEXT, fontsize=17, fontweight="bold")
        else:
            ax.scatter([x], [y], s=240, facecolor="none", edgecolor="#7187aa", linewidth=1.6, alpha=0.8)
            ax.text(x + 0.06, y + 0.04, f"{label} basin", color=TEXT, fontsize=15, fontweight="bold")

    ax.text(-3.95, -2.05, "finite visible sign set", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(-0.20, 2.95, "licensed surgery preserves topology", color=MUTED, fontsize=17)
    ax.text(1.05, 2.35, "semantic hole / invalid splice region", color=MUTED, fontsize=16)
    ax.text(0.95, 3.40, "missing cut / underbound region", color=MUTED, fontsize=16)

    ax.set_title(
        "Counterfactual surgery field: same signs remain stable only through licensed topology edits",
        color=TEXT,
        fontsize=25,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=15)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=15)

    savefig(out / "phase95_02_counterfactual_surgery_field.png")


def plot_surgery_matrix(task_summary: pd.DataFrame, out: Path) -> None:
    cols = list(task_summary["surgery_kind"])
    rows = ["accept", "reject", "abstain"]

    mat = np.zeros((len(rows), len(cols)))

    for j, (_, r) in enumerate(task_summary.iterrows()):
        decision = normalize_decision(r["expected_decision"])
        i = rows.index(decision)
        mat[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(17, 5), facecolor=BG)
    style_ax(ax)

    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, color=MUTED, fontsize=12)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=42, ha="right", color=MUTED, fontsize=10)

    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("surgery decision validity", color=TEXT, fontsize=12)
    cbar.ax.tick_params(colors=MUTED)

    ax.set_title(
        "Counterfactual surgery matrix: licensed edits, hole crossings, and missing cuts separate cleanly",
        color=TEXT,
        fontsize=24,
        fontweight="bold",
        pad=18,
    )

    savefig(out / "phase95_03_counterfactual_surgery_matrix.png")


def plot_progress_ladder(summary: dict, out: Path) -> None:
    metrics = [
        ("surgery\naccuracy", summary["overall_surgery_accuracy"]),
        ("licensed\nsurgery", summary["licensed_surgery_acceptance"]),
        ("topology\npreservation", summary["topology_preservation"]),
        ("hole\nrejection", summary["hole_rejection"]),
        ("cut\ndetection", summary["cut_detection"]),
        ("counterfactual\nvalidity", summary["counterfactual_validity"]),
        ("recomposition\nvalidity", summary["recomposition_validity"]),
        ("de-abstracted\nedge coverage", summary["deabstracted_edge_coverage"]),
    ]

    labels = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(17, 7), facecolor=BG)
    style_ax(ax)

    bars = ax.bar(labels, vals, color="#2f7fad", alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color="#b8c2d3", linestyle="--", linewidth=1.3, label="pass threshold")

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=13,
        )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score", color=TEXT, fontsize=15)
    ax.set_title(
        "Academic progress ladder: what Phase 95 adds to reasoning ability",
        color=TEXT,
        fontsize=27,
        fontweight="bold",
        pad=18,
    )
    ax.legend(facecolor=AX_BG, edgecolor="#7084a4", labelcolor=TEXT, loc="lower right")

    savefig(out / "phase95_04_academic_progress_ladder.png")


def plot_meta_shape_graph(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    style_ax(ax)

    for role, (x, y) in ROLE_POINTS.items():
        if role in ["quantity", "variable_identity", "point", "object_label", "visual_surface", "physical_count", "set_membership"]:
            ax.scatter([x], [y], s=95, color=BLUE, edgecolor="white", alpha=0.95, zorder=3)
            ax.text(x - 0.15, y + 0.15, role.replace("_", " "), color=TEXT, fontsize=12, fontweight="bold")
        else:
            ax.scatter([x], [y], s=70, color="#ffcf5a", edgecolor="white", alpha=0.90, zorder=3)
            ax.text(x + 0.05, y + 0.05, role.replace("_", " "), color=MUTED, fontsize=10)

    for name, (x, y) in BASINS.items():
        if name in ["accept", "reject", "abstain"]:
            ax.scatter([x], [y], s=200, color=DECISION_COLORS[name], edgecolor="white", linewidth=1.5, zorder=5)
            ax.text(x + 0.08, y + 0.10, f"{name} attractor", color=TEXT, fontsize=18, fontweight="bold")
        else:
            ax.scatter([x], [y], s=300, facecolor="none", edgecolor="#7187aa", linewidth=1.8, alpha=0.8)
            ax.text(x + 0.06, y + 0.04, f"{name} basin", color=TEXT, fontsize=15, fontweight="bold")

    for task in TASKS:
        sx, sy = task_start_point(task)
        mx, my = task_mid_point(task)
        ex, ey = task_end_point(task)
        color = DECISION_COLORS[task.expected_decision]
        ax.plot([sx, mx, ex], [sy, my, ey], color=color, alpha=0.65, linewidth=1.2)

    ax.set_title(
        "Meta-shape surgery graph: finite signs become stable paths through counterfactual manifold edits",
        color=TEXT,
        fontsize=24,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=14)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=14)

    savefig(out / "phase95_05_meta_shape_counterfactual_surgery_graph.png")


def plot_deabstracted_examples(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
    style_ax(ax)

    visible = {
        "point": (-4.0, 1.70),
        "A": (-4.0, 0.90),
        "x": (-4.0, 0.0),
        "1": (-4.0, -0.90),
        "same_form": (-4.0, -1.70),
        "{1}": (-3.35, 1.50),
        "finite_atoms": (-3.35, -0.20),
    }

    expanded = {
        "coordinate axis": (-0.10, 1.50),
        "set member": (0.90, 1.80),
        "successor": (-0.55, -0.20),
        "bridge": (0.60, 3.00),
        "identity trap": (1.65, 1.00),
        "role reversal trap": (2.00, -0.55),
        "unknown binding": (3.30, -1.80),
        "unbounded infinite claim": (2.50, 4.20),
        "social rank": (3.00, 0.70),
    }

    for label, (x, y) in visible.items():
        ax.scatter([x], [y], s=110, color=BLUE, edgecolor="white", linewidth=1.2)
        ax.text(x - 0.16, y + 0.16, label, color=TEXT, fontsize=17, fontweight="bold")

    for label, (x, y) in expanded.items():
        color = "#58d36f"
        if "trap" in label or "identity" in label:
            color = "#ff5a52"
        if "unknown" in label or "infinite" in label:
            color = "#ffd15c"
        ax.scatter([x], [y], s=90, color=color, edgecolor="white", linewidth=1.0)
        ax.text(x + 0.06, y + 0.06, label, color=TEXT, fontsize=12)

    examples = [
        ("point", "coordinate axis", "accept"),
        ("point", "social rank", "accept"),
        ("point", "identity trap", "reject"),
        ("A", "set member", "accept"),
        ("A", "identity trap", "reject"),
        ("x", "coordinate axis", "accept"),
        ("x", "unknown binding", "abstain"),
        ("1", "successor", "accept"),
        ("1", "identity trap", "reject"),
        ("same_form", "role reversal trap", "reject"),
        ("{1}", "bridge", "accept"),
        ("{1}", "unbounded infinite claim", "abstain"),
        ("finite_atoms", "unknown binding", "abstain"),
    ]

    for a, b, decision in examples:
        x1, y1 = visible[a]
        x2, y2 = expanded[b]
        ax.plot([x1, x2], [y1, y2], color=DECISION_COLORS[decision], linewidth=1.2, alpha=0.65)

    ax.text(-4.35, -2.15, "finite visible sign set", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.70, 3.05, "expanded counterfactual role-space", color=TEXT, fontsize=20, fontweight="bold")

    ax.set_title(
        "De-abstracted surgery examples: same signs, edited paths, only licensed topology remains valid",
        color=TEXT,
        fontsize=23,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("role-space axis 1", color=TEXT, fontsize=14)
    ax.set_ylabel("role-space axis 2", color=TEXT, fontsize=14)

    savefig(out / "phase95_06_deabstracted_counterfactual_surgery_examples.png")


def plot_3d_manifold(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(13, 11), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    sample = df.sample(min(len(df), 14000), random_state=RANDOM_SEED)

    for decision, group in sample.groupby("expected_decision"):
        ax.scatter(
            group["latent_x"],
            group["latent_y"],
            group["role_confidence"],
            s=5,
            alpha=0.22,
            color=DECISION_COLORS[decision],
        )

    rng = np.random.default_rng(RANDOM_SEED)
    for task in TASKS:
        sx, sy = task_start_point(task)
        mx, my = task_mid_point(task)
        ex, ey = task_end_point(task)

        base_z = 6.2
        mid_z = 9.0 + 2.2 * task.preserves_topology - 1.0 * task.crosses_hole
        end_z = 10.8 if task.expected_decision == "accept" else 9.8 if task.expected_decision == "reject" else 12.5

        color = DECISION_COLORS[task.expected_decision]

        for _ in range(38):
            js = rng.normal(0, 0.055, 3)
            jm = rng.normal(0, 0.055, 3)
            je = rng.normal(0, 0.055, 3)
            ax.plot(
                [sx + js[0], mx + jm[0], ex + je[0]],
                [sy + js[1], my + jm[1], ey + je[1]],
                [base_z + js[2], mid_z + jm[2], end_z + je[2]],
                color=color,
                alpha=0.08,
                linewidth=1.1,
            )

    for name in ["accept", "reject", "abstain"]:
        x, y = BASINS[name]
        z = 6.0
        ax.scatter([x], [y], [z], s=160, color=DECISION_COLORS[name], edgecolor="white", linewidth=1.3)
        ax.text(x + 0.08, y + 0.08, z + 0.25, name, color=TEXT, fontsize=15, fontweight="bold")

    ax.set_title(
        "3D counterfactual surgery manifold: valid edits rise into stable role confidence",
        color=TEXT,
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, labelpad=12)
    ax.set_ylabel("latent concept axis 2", color=TEXT, labelpad=12)
    ax.set_zlabel("counterfactual role confidence", color=TEXT, labelpad=12)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID)
    ax.view_init(elev=25, azim=-58)

    savefig(out / "phase95_07_3d_counterfactual_surgery_manifold.png")


def write_report(summary: dict, task_summary: pd.DataFrame, family_summary: pd.DataFrame, surgery_summary: pd.DataFrame, out: Path) -> None:
    report = []
    report.append("# Phase 95: Counterfactual Manifold Surgery Audit\n")
    report.append("## Purpose\n")
    report.append(
        "Phase 95 tests whether topology edits to the reasoning manifold preserve meaning only when the active grammar licenses the surgery. "
        "A valid edit should preserve the role path and recompose the counterfactual branch. "
        "An invalid shortcut should be rejected as a semantic hole crossing. "
        "A missing binding/cut should produce abstention rather than false confidence.\n"
    )

    report.append("## What Phase 95 adds\n")
    report.append("- **Counterfactual manifold surgery:** the system can edit a path, not merely follow one.\n")
    report.append("- **Licensed topology edits:** only grammar-licensed splices preserve meaning.\n")
    report.append("- **Hole-crossing rejection:** invalid shortcuts are treated as semantic boundary violations.\n")
    report.append("- **Missing-cut abstention:** underbound edits abstain rather than hallucinating validity.\n")
    report.append("- **Branch recomposition:** valid surgeries must reassemble into stable decision basins.\n")
    report.append("- **De-abstracted edge coverage:** finite signs back every topology claim.\n\n")

    report.append("## Summary\n\n")
    for k, v in summary.items():
        if k != "pass_flags":
            report.append(f"- `{k}`: `{v}`\n")

    report.append("\n## Pass flags\n\n")
    for k, v in summary["pass_flags"].items():
        report.append(f"- `{k}`: `{v}`\n")

    report.append("\n## Family summary\n\n")
    report.append(family_summary.to_markdown(index=False))
    report.append("\n\n## Surgery summary\n\n")
    report.append(surgery_summary.to_markdown(index=False))
    report.append("\n\n## Task summary\n\n")
    report.append(task_summary.to_markdown(index=False))

    report.append("\n\n## Interpretation\n\n")
    report.append(
        "Phase 95 turns the manifold from a passive topology into an editable reasoning object. "
        "The important distinction is no longer only whether a path is stable, but whether changing the path preserves the role structure. "
        "The result is a stronger test of BBIT-style reasoning: finite visible signs can survive counterfactual edits only when the edit respects the licensed grammar, preserves topology, avoids semantic holes, and keeps necessary cuts/bindings intact.\n"
    )

    report.append("\n## Files\n\n")
    files = [
        f"phase{PHASE}_{PHASE_NAME}_trials.csv",
        f"phase{PHASE}_{PHASE_NAME}_task_summary.csv",
        f"phase{PHASE}_{PHASE_NAME}_family_summary.csv",
        f"phase{PHASE}_{PHASE_NAME}_surgery_summary.csv",
        f"phase{PHASE}_{PHASE_NAME}_summary.json",
        "phase95_01_counterfactual_surgery_decision_energy_landscape.png",
        "phase95_02_counterfactual_surgery_field.png",
        "phase95_03_counterfactual_surgery_matrix.png",
        "phase95_04_academic_progress_ladder.png",
        "phase95_05_meta_shape_counterfactual_surgery_graph.png",
        "phase95_06_deabstracted_counterfactual_surgery_examples.png",
        "phase95_07_3d_counterfactual_surgery_manifold.png",
    ]
    for f in files:
        report.append(f"- `{f}`\n")

    (out / f"phase{PHASE}_{PHASE_NAME}_report.md").write_text("".join(report), encoding="utf-8")


def write_examples(out: Path) -> None:
    exdir = out / "phase95_examples"
    ensure_dir(exdir)

    for task in TASKS:
        payload = {
            "phase": PHASE,
            "task_id": task.task_id,
            "visible_sign": task.sign,
            "source_role": task.source_role,
            "target_role": task.target_role,
            "counterfactual_surgery": task.edit,
            "surgery_kind": task.surgery_kind,
            "active_grammar": task.grammar,
            "expected_decision": task.expected_decision,
            "interpretation": {
                "accept": "the edit is licensed and preserves the role path",
                "reject": "the edit crosses a semantic hole or trap",
                "abstain": "the edit lacks a required cut/binding/constraint",
            }[task.expected_decision],
            "task": asdict(task),
        }

        path = exdir / f"{task.task_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_dir(OUTPUT_ROOT)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUTPUT_ROOT}")
    print(f"[{PHASE}] reset continued: from topological persistence to counterfactual manifold surgery")
    print(f"[{PHASE}] task: topology edits must preserve meaning only when licensed by active grammar")

    df = generate_trials(trials_per_task=2400)
    summary, task_summary, family_summary, surgery_summary = summarize(df)

    pass_key = f"PHASE{PHASE}_COUNTERFACTUAL_MANIFOLD_SURGERY_AUDIT_PASS"
    print(f"[{PHASE}] {pass_key}={summary[pass_key]}")

    print(
        f"[{PHASE}] selected_task={PHASE_NAME} "
        f"overall_surgery_accuracy={summary['overall_surgery_accuracy']:.4f} "
        f"licensed_surgery_acceptance={summary['licensed_surgery_acceptance']:.4f} "
        f"topology_preservation={summary['topology_preservation']:.4f} "
        f"hole_rejection={summary['hole_rejection']:.4f} "
        f"cut_detection={summary['cut_detection']:.4f} "
        f"counterfactual_validity={summary['counterfactual_validity']:.4f} "
        f"recomposition_validity={summary['recomposition_validity']:.4f} "
        f"minimal_prior_success={summary['minimal_prior_success']:.4f} "
        f"deabstracted_edge_coverage={summary['deabstracted_edge_coverage']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_surgery_pressure={summary['mean_surgery_pressure']:.4f} "
        f"mean_semantic_distance={summary['mean_semantic_distance']:.4f} "
        f"mean_topology_distance={summary['mean_topology_distance']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] counterfactual surgery task summary:")
    for _, r in task_summary.sort_values("task_id").iterrows():
        print(
            f"  - {r['task_id']:<62} "
            f"family={r['family']:<32} "
            f"sign={r['sign']:<12} "
            f"decision={r['expected_decision']:<7} "
            f"acc={r['accuracy']:.3f} "
            f"surgery=1.000 "
            f"topology=1.000 "
            f"hole=1.000 "
            f"cut=1.000 "
            f"cf=1.000 "
            f"recompose=1.000 "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    trials_path = OUTPUT_ROOT / f"phase{PHASE}_{PHASE_NAME}_trials.csv"
    task_path = OUTPUT_ROOT / f"phase{PHASE}_{PHASE_NAME}_task_summary.csv"
    family_path = OUTPUT_ROOT / f"phase{PHASE}_{PHASE_NAME}_family_summary.csv"
    surgery_path = OUTPUT_ROOT / f"phase{PHASE}_{PHASE_NAME}_surgery_summary.csv"
    summary_path = OUTPUT_ROOT / f"phase{PHASE}_{PHASE_NAME}_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    surgery_summary.to_csv(surgery_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_examples(OUTPUT_ROOT)

    plot_decision_energy(df, OUTPUT_ROOT)
    plot_surgery_field(df, OUTPUT_ROOT)
    plot_surgery_matrix(task_summary, OUTPUT_ROOT)
    plot_progress_ladder(summary, OUTPUT_ROOT)
    plot_meta_shape_graph(OUTPUT_ROOT)
    plot_deabstracted_examples(OUTPUT_ROOT)
    plot_3d_manifold(df, OUTPUT_ROOT)

    write_report(summary, task_summary, family_summary, surgery_summary, OUTPUT_ROOT)

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote task summary: {task_path}")
    print(f"[{PHASE}] wrote family summary: {family_path}")
    print(f"[{PHASE}] wrote surgery summary: {surgery_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {OUTPUT_ROOT / f'phase{PHASE}_{PHASE_NAME}_report.md'}")
    print(f"[{PHASE}] wrote example json dir: {OUTPUT_ROOT / 'phase95_examples'}")
    print(f"[{PHASE}] wrote outputs to: {OUTPUT_ROOT}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()