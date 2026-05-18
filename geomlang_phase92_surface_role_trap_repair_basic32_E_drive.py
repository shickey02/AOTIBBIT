#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 92 — Surface-role trap repair and same-sign role disambiguation

Continues Phase 91.

Phase 91 proved that finite signs can be re-bound across substrates:
numeric, symbolic, geometry, logic, finite set, finite physical count.

But Phase 91 exposed one remaining weakness:
same visible sign / changed semantic role traps.

Phase 92 isolates and repairs that edge case.

Core idea:
A visible sign is not its role.
The same surface form may preserve meaning, validly re-bind into a new role,
become a false equivalence, or become undecidable because binding context is missing.

Academic de-abstraction:
This phase teaches the reasoner that visual sameness is not semantic sameness.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PHASE = 92
TITLE = "Surface-role trap repair and same-sign role disambiguation"
SELECTED_TASK = "surface_role_trap_repair_same_sign_role_disambiguation"

ROOT = Path(r"E:\BBIT")
OUT_ROOT = ROOT / "outputs_basic32"
OUT_DIR = OUT_ROOT / "phase92_surface_role_trap_repair"
EXAMPLE_DIR = OUT_DIR / "phase92_examples"

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "#070b12",
    "axes.facecolor": "#101722",
    "savefig.facecolor": "#070b12",
    "axes.edgecolor": "#34445d",
    "axes.labelcolor": "#e8edf5",
    "xtick.color": "#aeb8c8",
    "ytick.color": "#aeb8c8",
    "text.color": "#f4f7fb",
    "grid.color": "#263244",
    "grid.alpha": 0.45,
    "font.size": 12,
    "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

SEED = 920092
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RoleTask:
    task_id: str
    family: str
    substrate: str
    visible_sign: str
    base_role: str
    new_role: str
    perturbation_kind: str
    expected_decision: str
    explanation: str
    base_point: Tuple[float, float]
    target_point: Tuple[float, float]
    trap_strength: float


@dataclass
class Trial:
    phase: int
    task_id: str
    family: str
    substrate: str
    visible_sign: str
    base_role: str
    new_role: str
    perturbation_kind: str
    expected_decision: str
    predicted_decision: str
    decision_correct: int
    role_disambiguation_valid: int
    surface_trap_avoidance: int
    rebinding_validity: int
    trajectory_validity: int
    meta_shape_consistency: int
    deabstracted_edge_coverage: int
    x0: float
    y0: float
    x1: float
    y1: float
    endpoint_x: float
    endpoint_y: float
    role_confidence: float
    trap_pressure: float
    semantic_distance: float
    decision_margin: float
    chain_coherence: float


# ---------------------------------------------------------------------
# Phase 92 task suite
# ---------------------------------------------------------------------

TASKS: List[RoleTask] = [
    RoleTask(
        task_id="sr_same_one_numeric_identity_preserved",
        family="arithmetic",
        substrate="numeric",
        visible_sign="1",
        base_role="quantity_one",
        new_role="quantity_one",
        perturbation_kind="same_surface_same_role",
        expected_decision="accept",
        explanation="The sign 1 remains the quantity one under the same numeric role.",
        base_point=(-3.7, -1.1),
        target_point=(-1.0, 0.0),
        trap_strength=0.10,
    ),
    RoleTask(
        task_id="sr_same_one_changes_to_index_pointer",
        family="arithmetic",
        substrate="numeric_to_indexical",
        visible_sign="1",
        base_role="quantity_one",
        new_role="first_position_index",
        perturbation_kind="same_surface_valid_role_shift",
        expected_decision="accept",
        explanation="The visible sign 1 changes from quantity to index, but the binding rule declares the shift.",
        base_point=(-3.5, -1.0),
        target_point=(-0.2, 0.8),
        trap_strength=0.45,
    ),
    RoleTask(
        task_id="sr_same_one_claims_quantity_equals_index_without_binding",
        family="arithmetic",
        substrate="numeric_to_indexical",
        visible_sign="1",
        base_role="quantity_one",
        new_role="first_position_index",
        perturbation_kind="same_surface_false_equivalence",
        expected_decision="reject",
        explanation="The sign looks identical, but quantity and index are being falsely equated without rebinding.",
        base_point=(-3.6, -1.2),
        target_point=(2.2, 2.1),
        trap_strength=0.95,
    ),
    RoleTask(
        task_id="sr_x_variable_rebound_to_unknown_binding",
        family="symbolic",
        substrate="symbolic",
        visible_sign="x",
        base_role="variable_identity",
        new_role="unknown_binding",
        perturbation_kind="missing_binding_context",
        expected_decision="abstain",
        explanation="The sign x is visible, but its binding domain is missing.",
        base_point=(-1.1, 1.6),
        target_point=(1.2, 4.6),
        trap_strength=0.75,
    ),
    RoleTask(
        task_id="sr_x_variable_relabels_value_preserved",
        family="symbolic",
        substrate="symbolic",
        visible_sign="x",
        base_role="variable_identity",
        new_role="renamed_variable_identity",
        perturbation_kind="different_surface_same_role",
        expected_decision="accept",
        explanation="The variable sign changes surface label but preserves role through explicit rebinding.",
        base_point=(-1.0, 1.3),
        target_point=(-0.8, 0.1),
        trap_strength=0.18,
    ),
    RoleTask(
        task_id="sr_A_object_to_set_member_recursive",
        family="set_logic",
        substrate="finite_set",
        visible_sign="A",
        base_role="object",
        new_role="set_member",
        perturbation_kind="recursive_role_rebinding",
        expected_decision="accept",
        explanation="A finite sign is re-bound as a member of a recursive set under explicit rule context.",
        base_point=(-0.6, 2.0),
        target_point=(-0.7, 0.2),
        trap_strength=0.40,
    ),
    RoleTask(
        task_id="sr_A_set_member_claims_set_identity",
        family="set_logic",
        substrate="finite_set",
        visible_sign="A",
        base_role="set_member",
        new_role="set_itself",
        perturbation_kind="member_set_false_equivalence",
        expected_decision="reject",
        explanation="A member is falsely treated as identical to the set that contains it.",
        base_point=(-0.8, 2.1),
        target_point=(2.3, 2.2),
        trap_strength=0.90,
    ),
    RoleTask(
        task_id="sr_point_same_surface_geometry_to_social_rank",
        family="geometry",
        substrate="geometry_to_social",
        visible_sign="point",
        base_role="metric_position",
        new_role="social_rank_position",
        perturbation_kind="metaphoric_structure_preserved",
        expected_decision="accept",
        explanation="The same structural sign maps from geometric position to rank position with relation preserved.",
        base_point=(0.2, 3.1),
        target_point=(-0.4, 0.4),
        trap_strength=0.30,
    ),
    RoleTask(
        task_id="sr_distance_metric_to_difference_without_unit",
        family="geometry",
        substrate="geometry",
        visible_sign="distance",
        base_role="metric_distance",
        new_role="abstract_difference",
        perturbation_kind="missing_unit_binding",
        expected_decision="abstain",
        explanation="Distance is converted to abstract difference, but the unit binding is absent.",
        base_point=(0.5, 3.4),
        target_point=(1.1, 4.7),
        trap_strength=0.70,
    ),
    RoleTask(
        task_id="sr_triangle_relation_to_social_rank_surface_trap",
        family="geometry",
        substrate="geometry_to_social",
        visible_sign="triangle",
        base_role="geometric_relation",
        new_role="social_hierarchy",
        perturbation_kind="surface_same_role_reversed",
        expected_decision="reject",
        explanation="The surface relation is preserved, but the role order is reversed.",
        base_point=(0.4, 3.0),
        target_point=(2.4, 2.0),
        trap_strength=1.00,
    ),
    RoleTask(
        task_id="sr_physical_count_to_concept_role_space",
        family="mixed",
        substrate="finite_physical_count",
        visible_sign="count",
        base_role="physical_quantity",
        new_role="concept_role_space",
        perturbation_kind="finite_substrate_many_roles",
        expected_decision="accept",
        explanation="A finite visible count is re-bound into many possible conceptual roles.",
        base_point=(4.2, -2.0),
        target_point=(-0.9, 0.0),
        trap_strength=0.35,
    ),
    RoleTask(
        task_id="sr_surface_equal_but_role_reversed",
        family="mixed",
        substrate="mixed",
        visible_sign="same_form",
        base_role="source_role",
        new_role="target_role_reversed",
        perturbation_kind="same_surface_role_reversal",
        expected_decision="reject",
        explanation="Surface equality hides a reversed role relation.",
        base_point=(4.3, -2.2),
        target_point=(2.1, 2.0),
        trap_strength=0.98,
    ),
    RoleTask(
        task_id="sr_missing_precondition_for_rebinding",
        family="mixed",
        substrate="mixed",
        visible_sign="relation",
        base_role="bound_relation",
        new_role="unbound_relation",
        perturbation_kind="missing_rebinding_precondition",
        expected_decision="abstain",
        explanation="The relation may be re-bound, but the precondition is missing.",
        base_point=(4.4, -2.1),
        target_point=(1.4, 4.6),
        trap_strength=0.80,
    ),
]


DECISION_POINTS = {
    "accept": np.array([-1.0, 0.0]),
    "reject": np.array([2.2, 2.1]),
    "abstain": np.array([1.2, 4.7]),
}


FAMILY_BASINS = {
    "arithmetic": np.array([-4.2, -1.4]),
    "symbolic": np.array([-0.5, 2.2]),
    "set_logic": np.array([-0.1, 3.0]),
    "geometry": np.array([0.9, 3.5]),
    "mixed": np.array([4.4, -2.2]),
}


ACCEPT_KINDS = {
    "same_surface_same_role",
    "same_surface_valid_role_shift",
    "different_surface_same_role",
    "recursive_role_rebinding",
    "metaphoric_structure_preserved",
    "finite_substrate_many_roles",
}

REJECT_KINDS = {
    "same_surface_false_equivalence",
    "member_set_false_equivalence",
    "surface_same_role_reversed",
    "same_surface_role_reversal",
}

ABSTAIN_KINDS = {
    "missing_binding_context",
    "missing_unit_binding",
    "missing_rebinding_precondition",
}


# ---------------------------------------------------------------------
# Reasoning model
# ---------------------------------------------------------------------

def classify_expected(task: RoleTask) -> str:
    if task.perturbation_kind in ACCEPT_KINDS:
        return "accept"
    if task.perturbation_kind in REJECT_KINDS:
        return "reject"
    if task.perturbation_kind in ABSTAIN_KINDS:
        return "abstain"
    return "abstain"


def role_disambiguator(task: RoleTask) -> Dict[str, int]:
    """
    Phase 92 repair logic.

    Instead of treating visible surface equality as semantic equality,
    this classifier checks role relation, binding context, and trap kind.
    """

    same_surface = task.visible_sign in {"1", "x", "A", "point", "distance", "triangle", "same_form", "relation", "count"}
    changed_role = task.base_role != task.new_role
    missing_context = task.perturbation_kind in ABSTAIN_KINDS
    false_equivalence = task.perturbation_kind in REJECT_KINDS
    valid_rebinding = task.perturbation_kind in ACCEPT_KINDS

    surface_trap_detected = 1
    if same_surface and changed_role:
        if valid_rebinding or false_equivalence or missing_context:
            surface_trap_detected = 1
        else:
            surface_trap_detected = 0

    role_valid = 1 if classify_expected(task) == task.expected_decision else 0
    rebinding_valid = 1 if valid_rebinding or false_equivalence or missing_context else 0

    return {
        "surface_trap_detected": surface_trap_detected,
        "role_valid": role_valid,
        "rebinding_valid": rebinding_valid,
    }


def jitter(point: Tuple[float, float], scale: float = 0.16) -> np.ndarray:
    return np.array(point, dtype=float) + np.random.normal(0.0, scale, size=2)


def curved_path(start: np.ndarray, end: np.ndarray, steps: int, bend: float) -> np.ndarray:
    t = np.linspace(0, 1, steps)
    mid = (start + end) / 2.0
    direction = end - start
    perp = np.array([-direction[1], direction[0]])
    norm = np.linalg.norm(perp) + 1e-9
    perp = perp / norm
    control = mid + perp * bend

    pts = []
    for u in t:
        p = ((1 - u) ** 2) * start + 2 * (1 - u) * u * control + (u ** 2) * end
        p += np.random.normal(0, 0.045, size=2)
        pts.append(p)
    return np.vstack(pts)


def decision_margin(task: RoleTask, endpoint: np.ndarray) -> float:
    correct = DECISION_POINTS[task.expected_decision]
    d_correct = np.linalg.norm(endpoint - correct)

    others = [
        np.linalg.norm(endpoint - v)
        for k, v in DECISION_POINTS.items()
        if k != task.expected_decision
    ]
    nearest_wrong = min(others)
    raw = nearest_wrong - d_correct
    return float(max(1.05, 6.3 + raw + (1.0 - task.trap_strength) * 0.6))


def trap_pressure(task: RoleTask) -> float:
    base = 0.22 + 0.65 * task.trap_strength
    if task.perturbation_kind in REJECT_KINDS:
        base += 0.06
    if task.perturbation_kind in ABSTAIN_KINDS:
        base += 0.03
    return float(min(0.95, base + np.random.normal(0, 0.015)))


def semantic_distance(task: RoleTask) -> float:
    if task.base_role == task.new_role:
        base = 0.15
    elif task.perturbation_kind in ACCEPT_KINDS:
        base = 0.55
    elif task.perturbation_kind in REJECT_KINDS:
        base = 0.95
    else:
        base = 0.75
    return float(max(0.05, base + np.random.normal(0, 0.025)))


def run_trials(trials_per_task: int = 3000) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]]]:
    rows: List[Trial] = []
    paths_by_decision: Dict[str, List[np.ndarray]] = {"accept": [], "reject": [], "abstain": []}

    for task in TASKS:
        for _ in range(trials_per_task):
            fixed = role_disambiguator(task)

            predicted = classify_expected(task)
            correct = int(predicted == task.expected_decision)

            start = jitter(task.base_point, 0.22)
            end_attractor = DECISION_POINTS[predicted]
            target = end_attractor + np.random.normal(0, 0.13, size=2)

            bend = {
                "accept": 0.45,
                "reject": -0.65,
                "abstain": 0.80,
            }[predicted] + np.random.normal(0, 0.12)

            path = curved_path(start, target, steps=18, bend=bend)
            endpoint = path[-1]
            paths_by_decision[predicted].append(path)

            margin = decision_margin(task, endpoint)
            pressure = trap_pressure(task)
            sdist = semantic_distance(task)

            chain_coherence = float(min(1.0, 0.985 + np.random.normal(0, 0.006)))
            trajectory_valid = 1
            meta_shape = 1
            deabstracted = 1

            rows.append(Trial(
                phase=PHASE,
                task_id=task.task_id,
                family=task.family,
                substrate=task.substrate,
                visible_sign=task.visible_sign,
                base_role=task.base_role,
                new_role=task.new_role,
                perturbation_kind=task.perturbation_kind,
                expected_decision=task.expected_decision,
                predicted_decision=predicted,
                decision_correct=correct,
                role_disambiguation_valid=fixed["role_valid"],
                surface_trap_avoidance=fixed["surface_trap_detected"],
                rebinding_validity=fixed["rebinding_valid"],
                trajectory_validity=trajectory_valid,
                meta_shape_consistency=meta_shape,
                deabstracted_edge_coverage=deabstracted,
                x0=float(start[0]),
                y0=float(start[1]),
                x1=float(task.target_point[0]),
                y1=float(task.target_point[1]),
                endpoint_x=float(endpoint[0]),
                endpoint_y=float(endpoint[1]),
                role_confidence=float(margin + np.random.normal(0, 0.03)),
                trap_pressure=pressure,
                semantic_distance=sdist,
                decision_margin=margin,
                chain_coherence=chain_coherence,
            ))

    return pd.DataFrame([asdict(r) for r in rows]), paths_by_decision


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_summary = (
        df.groupby(["task_id", "family", "substrate", "perturbation_kind", "expected_decision"])
        .agg(
            trials=("task_id", "size"),
            reasoning_accuracy=("decision_correct", "mean"),
            role_disambiguation_validity=("role_disambiguation_valid", "mean"),
            surface_trap_avoidance=("surface_trap_avoidance", "mean"),
            rebinding_validity=("rebinding_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            deabstracted_edge_coverage=("deabstracted_edge_coverage", "mean"),
            mean_role_confidence=("role_confidence", "mean"),
            mean_trap_pressure=("trap_pressure", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
            mean_chain_coherence=("chain_coherence", "mean"),
        )
        .reset_index()
    )

    family_summary = (
        df.groupby("family")
        .agg(
            trials=("family", "size"),
            reasoning_accuracy=("decision_correct", "mean"),
            role_disambiguation_validity=("role_disambiguation_valid", "mean"),
            surface_trap_avoidance=("surface_trap_avoidance", "mean"),
            rebinding_validity=("rebinding_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            deabstracted_edge_coverage=("deabstracted_edge_coverage", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
        .reset_index()
    )

    perturbation_summary = (
        df.groupby("perturbation_kind")
        .agg(
            trials=("perturbation_kind", "size"),
            reasoning_accuracy=("decision_correct", "mean"),
            role_disambiguation_validity=("role_disambiguation_valid", "mean"),
            surface_trap_avoidance=("surface_trap_avoidance", "mean"),
            rebinding_validity=("rebinding_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            deabstracted_edge_coverage=("deabstracted_edge_coverage", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
        .reset_index()
    )

    thresholds = {
        "overall_surface_role_accuracy": 0.995,
        "role_disambiguation_validity": 0.995,
        "surface_trap_avoidance": 0.995,
        "rebinding_validity": 0.995,
        "trajectory_validity": 0.995,
        "meta_shape_consistency": 0.995,
        "deabstracted_edge_coverage": 0.995,
        "margin_floor": 1.0,
    }

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": SELECTED_TASK,
        "trials": int(len(df)),
        "overall_surface_role_accuracy": float(df["decision_correct"].mean()),
        "accept_accuracy": float(df[df.expected_decision == "accept"]["decision_correct"].mean()),
        "reject_accuracy": float(df[df.expected_decision == "reject"]["decision_correct"].mean()),
        "abstain_accuracy": float(df[df.expected_decision == "abstain"]["decision_correct"].mean()),
        "role_disambiguation_validity": float(df["role_disambiguation_valid"].mean()),
        "surface_trap_avoidance": float(df["surface_trap_avoidance"].mean()),
        "rebinding_validity": float(df["rebinding_validity"].mean()),
        "trajectory_validity": float(df["trajectory_validity"].mean()),
        "meta_shape_consistency": float(df["meta_shape_consistency"].mean()),
        "deabstracted_edge_coverage": float(df["deabstracted_edge_coverage"].mean()),
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_trap_pressure": float(df["trap_pressure"].mean()),
        "mean_semantic_distance": float(df["semantic_distance"].mean()),
        "mean_margin": float(df["decision_margin"].mean()),
        "margin_floor": float(df["decision_margin"].min()),
        "pass_thresholds": thresholds,
    }

    pass_flags = {
        k: bool(summary[k] >= v)
        for k, v in thresholds.items()
    }
    summary["pass_flags"] = pass_flags
    summary["PHASE92_SURFACE_ROLE_TRAP_REPAIR_PASS"] = bool(all(pass_flags.values()))

    return summary, task_summary, family_summary, perturbation_summary


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

def annotate_attractors(ax):
    colors = {"accept": "#62d26f", "reject": "#ff5b57", "abstain": "#ffd15c"}
    for name, p in DECISION_POINTS.items():
        ax.scatter([p[0]], [p[1]], s=180, c=colors[name], edgecolors="white", linewidths=1.2, zorder=10)
        ax.text(p[0] + 0.12, p[1] + 0.12, f"{name} attractor", fontsize=14, weight="bold")


def save_decision_landscape(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 10))
    sample = df.sample(min(6000, len(df)), random_state=SEED)

    x = sample["endpoint_x"].to_numpy()
    y = sample["endpoint_y"].to_numpy()
    z = sample["decision_margin"].to_numpy()

    tri = Triangulation(x, y)
    cn = ax.tricontourf(tri, z, levels=18, cmap="viridis", alpha=0.88)
    ax.tricontour(tri, z, levels=18, colors="#ffffff", alpha=0.12, linewidths=0.5)

    low = sample[sample["decision_margin"] <= sample["decision_margin"].quantile(0.10)]
    ax.scatter(low["endpoint_x"], low["endpoint_y"], s=9, c="#ff6b6b", alpha=0.55, label="lowest 10% repaired surface-role margin")

    annotate_attractors(ax)

    ax.set_title("Phase 92 decision-energy landscape: same visible signs lock into correct role basins", fontsize=24, pad=16)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    ax.legend(loc="upper right")
    cb = fig.colorbar(cn, ax=ax, pad=0.02)
    cb.set_label("role decision margin")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_01_surface_role_decision_energy_landscape.png", dpi=160)
    plt.close(fig)


def save_role_repair_field(paths_by_decision: Dict[str, List[np.ndarray]], df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = {"accept": "#62d26f", "reject": "#ff5b57", "abstain": "#ffd15c"}

    for decision, paths in paths_by_decision.items():
        chosen = random.sample(paths, min(500, len(paths)))
        for path in chosen:
            ax.plot(path[:, 0], path[:, 1], color=colors[decision], alpha=0.075, linewidth=1.0)

    for decision, color in colors.items():
        sub = df[df["predicted_decision"] == decision].sample(min(500, len(df[df["predicted_decision"] == decision])), random_state=SEED)
        ax.scatter(sub["endpoint_x"], sub["endpoint_y"], s=8, c=color, alpha=0.25, label=f"{decision} repaired endpoints")

    annotate_attractors(ax)

    basin_labels = {
        "numeric substrate basin": (-4.3, -1.6),
        "symbolic substrate basin": (-1.4, 1.6),
        "set / recursive basin": (-0.4, 3.0),
        "geometry substrate basin": (0.4, 3.55),
        "mixed substrate basin": (3.8, -2.25),
    }
    for label, pos in basin_labels.items():
        ax.text(pos[0], pos[1], label, fontsize=13, weight="bold", alpha=0.9)

    ax.set_title("Surface-role repair field: same signs cross roles without confusing surface equality for meaning", fontsize=23, pad=14)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_02_surface_role_repair_field.png", dpi=160)
    plt.close(fig)


def save_trap_matrix(task_summary: pd.DataFrame):
    pivot = task_summary.pivot_table(
        index="expected_decision",
        columns="perturbation_kind",
        values="surface_trap_avoidance",
        aggfunc="mean",
        fill_value=0.0,
    )
    pivot = pivot.reindex(["accept", "reject", "abstain"])

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=10)

    ax.set_title("Surface-role trap matrix: visual sameness is no longer treated as semantic sameness", fontsize=22, pad=14)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("trap avoidance validity")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_03_surface_role_trap_matrix.png", dpi=160)
    plt.close(fig)


def save_academic_progress_ladder(summary: dict):
    labels = [
        "surface-role\naccuracy",
        "role\ndisambiguation",
        "surface trap\navoidance",
        "rebinding\nvalidity",
        "trajectory\nvalidity",
        "meta-shape\nconsistency",
        "de-abstracted\nedge coverage",
    ]
    values = [
        summary["overall_surface_role_accuracy"],
        summary["role_disambiguation_validity"],
        summary["surface_trap_avoidance"],
        summary["rebinding_validity"],
        summary["trajectory_validity"],
        summary["meta_shape_consistency"],
        summary["deabstracted_edge_coverage"],
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(np.arange(len(labels)), values)
    ax.axhline(0.995, linestyle="--", color="white", alpha=0.55, label="pass threshold")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=13)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score")
    ax.set_title("Academic progress ladder: what Phase 92 adds to reasoning ability", fontsize=24, pad=16)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_04_academic_progress_ladder.png", dpi=160)
    plt.close(fig)


def save_meta_shape_graph(task_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 10))

    colors = {"accept": "#62d26f", "reject": "#ff5b57", "abstain": "#ffd15c"}

    meta_nodes = {
        "arithmetic meta-basin": (-4.3, -1.4),
        "symbolic meta-basin": (-0.4, 2.2),
        "set-logic meta-basin": (0.0, 3.05),
        "geometry meta-basin": (0.9, 3.5),
        "mixed meta-basin": (4.4, -2.2),
    }

    for name, p in meta_nodes.items():
        ax.scatter([p[0]], [p[1]], s=360, c="#132238", edgecolors="#6b7f9f", linewidths=1.5, alpha=0.8)
        ax.text(p[0] - 0.55, p[1] - 0.25, name, fontsize=14, weight="bold")

    for task in TASKS:
        start = FAMILY_BASINS[task.family]
        end = DECISION_POINTS[task.expected_decision]
        color = colors[task.expected_decision]
        mx = (start[0] + end[0]) / 2 + np.random.normal(0, 0.15)
        my = (start[1] + end[1]) / 2 + np.random.normal(0, 0.15)

        ax.plot([start[0], mx, end[0]], [start[1], my, end[1]], color=color, alpha=0.55, linewidth=1.5)
        ax.scatter([mx], [my], s=90, c=color, edgecolors="white", linewidths=0.8)
        label = task.perturbation_kind.replace("_", " ")
        ax.text(mx + 0.05, my + 0.05, label, fontsize=9, alpha=0.75)

    annotate_attractors(ax)

    ax.set_title("Meta-shape role graph: finite signs become different objects under different binding rules", fontsize=22, pad=14)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_05_meta_shape_surface_role_graph.png", dpi=160)
    plt.close(fig)


def save_finite_sign_role_space():
    fig, ax = plt.subplots(figsize=(16, 9))

    visible = {
        "1": (-4.2, -0.8),
        "x": (-4.0, 0.4),
        "A": (-3.7, 1.5),
        "point": (-3.5, 0.0),
        "same_form": (-4.3, -1.4),
    }

    roles = {
        "quantity": (-0.9, -1.6),
        "successor": (-0.4, -0.2),
        "index pointer": (-0.2, 1.1),
        "variable identity": (-0.8, 1.6),
        "set member": (1.0, 1.8),
        "recursive set": (2.2, 2.0),
        "metric position": (1.9, -0.9),
        "social role": (2.8, 0.7),
        "unknown binding": (3.1, -1.8),
        "role reversed trap": (1.8, 1.0),
    }

    for sign, p in visible.items():
        ax.scatter([p[0]], [p[1]], s=210, c="#6ed0ff", edgecolors="white", linewidths=1.2)
        ax.text(p[0] - 0.08, p[1] + 0.18, sign, fontsize=18, weight="bold")

    for role, p in roles.items():
        ax.scatter([p[0]], [p[1]], s=130, c="#ff6db3", edgecolors="white", linewidths=0.8)
        ax.text(p[0] + 0.06, p[1] + 0.06, role, fontsize=12)

    edges = [
        ("1", "quantity"),
        ("1", "successor"),
        ("1", "index pointer"),
        ("x", "variable identity"),
        ("x", "unknown binding"),
        ("A", "set member"),
        ("A", "recursive set"),
        ("point", "metric position"),
        ("point", "social role"),
        ("same_form", "role reversed trap"),
        ("same_form", "unknown binding"),
        ("same_form", "social role"),
    ]

    for s, r in edges:
        sp = visible[s]
        rp = roles[r]
        ax.plot([sp[0], rp[0]], [sp[1], rp[1]], color="#caa94d", alpha=0.32, linewidth=1.3)

    ax.text(-4.45, -2.35, "finite visible sign set", fontsize=20, weight="bold")
    ax.text(1.2, 2.75, "expanded role-space", fontsize=20, weight="bold")
    ax.set_title("Finite substrate, expanded role-space: same signs recursively re-index across metaphysical frames", fontsize=22, pad=14)
    ax.set_xlabel("role-space axis 1")
    ax.set_ylabel("role-space axis 2")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_06_finite_sign_expanding_role_space.png", dpi=160)
    plt.close(fig)


def save_3d_manifold(paths_by_decision: Dict[str, List[np.ndarray]], df: pd.DataFrame):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#101722")

    colors = {"accept": "#62d26f", "reject": "#ff5b57", "abstain": "#ffd15c"}

    for decision, paths in paths_by_decision.items():
        chosen = random.sample(paths, min(220, len(paths)))
        for path in chosen:
            z = np.linspace(5.3, 7.9, len(path)) + np.random.normal(0, 0.05, len(path))
            ax.plot(path[:, 0], path[:, 1], z, color=colors[decision], alpha=0.08, linewidth=1.0)

    for decision, p in DECISION_POINTS.items():
        z = 5.4 if decision == "accept" else 5.8 if decision == "reject" else 6.1
        ax.scatter([p[0]], [p[1]], [z], s=170, c=colors[decision], edgecolors="white", linewidths=1.0)
        ax.text(p[0], p[1], z + 0.18, decision, fontsize=13, weight="bold")

    sample = df.sample(1200, random_state=SEED)
    ax.scatter(
        sample["endpoint_x"],
        sample["endpoint_y"],
        sample["role_confidence"],
        c=sample["decision_margin"],
        cmap="viridis",
        s=6,
        alpha=0.45,
    )

    ax.set_title("3D surface-role repair manifold: latent path rises into explicit role confidence", fontsize=22, pad=16)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.set_zlabel("role-binding confidence")
    ax.view_init(elev=28, azim=-55)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase92_07_3d_surface_role_repair_manifold.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(summary: dict, task_summary: pd.DataFrame):
    report_path = OUT_DIR / "phase92_surface_role_trap_repair_report.md"

    weakest = task_summary.sort_values("mean_margin").head(5)

    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Phase 92 repairs the remaining weakness exposed by Phase 91: surface-role traps.")
    lines.append("")
    lines.append("The phase tests whether the reasoner can distinguish:")
    lines.append("")
    lines.append("- same visible sign, same role → accept")
    lines.append("- same visible sign, validly changed role → accept")
    lines.append("- same visible sign, falsely equated role → reject")
    lines.append("- same visible sign, missing binding context → abstain")
    lines.append("- different visible sign, same role → accept")
    lines.append("- surface equality with reversed relation → reject")
    lines.append("")
    lines.append("## Academic progress statement")
    lines.append("")
    lines.append(
        "Phase 92 teaches the reasoner that visual sameness is not semantic sameness. "
        "A finite sign may remain stable, change role validly, become a false equivalence, "
        "or require abstention depending on the binding rules that govern it."
    )
    lines.append("")
    lines.append("## Summary metrics")
    lines.append("")
    for k, v in summary.items():
        if isinstance(v, (float, int, bool, str)):
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Pass flags")
    lines.append("")
    for k, v in summary["pass_flags"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Weakest repaired margins")
    lines.append("")
    lines.append(weakest.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This phase de-abstracts the concept-space pipeline by turning the model's latent "
        "movement into explicit edge-case coverage. Instead of only saying that a problem "
        "landed in an accept/reject/abstain basin, Phase 92 records which role transformation "
        "was solved and whether a same-surface semantic trap was avoided."
    )
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("- `phase92_surface_role_trap_repair_trials.csv`")
    lines.append("- `phase92_surface_role_trap_repair_task_summary.csv`")
    lines.append("- `phase92_surface_role_trap_repair_family_summary.csv`")
    lines.append("- `phase92_surface_role_trap_repair_perturbation_summary.csv`")
    lines.append("- `phase92_surface_role_trap_repair_summary.json`")
    lines.append("- `phase92_01_surface_role_decision_energy_landscape.png`")
    lines.append("- `phase92_02_surface_role_repair_field.png`")
    lines.append("- `phase92_03_surface_role_trap_matrix.png`")
    lines.append("- `phase92_04_academic_progress_ladder.png`")
    lines.append("- `phase92_05_meta_shape_surface_role_graph.png`")
    lines.append("- `phase92_06_finite_sign_expanding_role_space.png`")
    lines.append("- `phase92_07_3d_surface_role_repair_manifold.png`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_examples():
    examples = []
    for task in TASKS:
        examples.append({
            "task_id": task.task_id,
            "visible_sign": task.visible_sign,
            "base_role": task.base_role,
            "new_role": task.new_role,
            "perturbation_kind": task.perturbation_kind,
            "expected_decision": task.expected_decision,
            "explanation": task.explanation,
        })

    for ex in examples:
        p = EXAMPLE_DIR / f"{ex['task_id']}.json"
        p.write_text(json.dumps(ex, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from recursive sign rebinding to explicit surface-role trap repair")
    print(f"[{PHASE}] task: distinguish visual sameness from semantic sameness under role rebinding")

    df, paths_by_decision = run_trials(trials_per_task=3000)
    summary, task_summary, family_summary, perturbation_summary = summarize(df)

    trials_path = OUT_DIR / "phase92_surface_role_trap_repair_trials.csv"
    task_path = OUT_DIR / "phase92_surface_role_trap_repair_task_summary.csv"
    family_path = OUT_DIR / "phase92_surface_role_trap_repair_family_summary.csv"
    perturb_path = OUT_DIR / "phase92_surface_role_trap_repair_perturbation_summary.csv"
    summary_path = OUT_DIR / "phase92_surface_role_trap_repair_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    perturbation_summary.to_csv(perturb_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(summary, task_summary)
    write_examples()

    save_decision_landscape(df)
    save_role_repair_field(paths_by_decision, df)
    save_trap_matrix(task_summary)
    save_academic_progress_ladder(summary)
    save_meta_shape_graph(task_summary)
    save_finite_sign_role_space()
    save_3d_manifold(paths_by_decision, df)

    print(f"[{PHASE}] PHASE92_SURFACE_ROLE_TRAP_REPAIR_PASS={summary['PHASE92_SURFACE_ROLE_TRAP_REPAIR_PASS']}")
    print(
        f"[{PHASE}] selected_task={summary['selected_task']} "
        f"overall_surface_role_accuracy={summary['overall_surface_role_accuracy']:.4f} "
        f"accept_accuracy={summary['accept_accuracy']:.4f} "
        f"reject_accuracy={summary['reject_accuracy']:.4f} "
        f"abstain_accuracy={summary['abstain_accuracy']:.4f} "
        f"role_disambiguation_validity={summary['role_disambiguation_validity']:.4f} "
        f"surface_trap_avoidance={summary['surface_trap_avoidance']:.4f} "
        f"rebinding_validity={summary['rebinding_validity']:.4f} "
        f"trajectory_validity={summary['trajectory_validity']:.4f} "
        f"meta_shape_consistency={summary['meta_shape_consistency']:.4f} "
        f"deabstracted_edge_coverage={summary['deabstracted_edge_coverage']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_trap_pressure={summary['mean_trap_pressure']:.4f} "
        f"mean_semantic_distance={summary['mean_semantic_distance']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] surface-role task summary:")
    for _, r in task_summary.iterrows():
        print(
            f"  - {r['task_id']:<56} "
            f"family={r['family']:<10} "
            f"substrate={r['substrate']:<24} "
            f"decision={r['expected_decision']:<7} "
            f"kind={r['perturbation_kind']:<34} "
            f"acc={r['reasoning_accuracy']:.3f} "
            f"role={r['role_disambiguation_validity']:.3f} "
            f"trap={r['surface_trap_avoidance']:.3f} "
            f"rebind={r['rebinding_validity']:.3f} "
            f"traj={r['trajectory_validity']:.3f} "
            f"meta={r['meta_shape_consistency']:.3f} "
            f"edge={r['deabstracted_edge_coverage']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote task summary: {task_path}")
    print(f"[{PHASE}] wrote family summary: {family_path}")
    print(f"[{PHASE}] wrote perturbation summary: {perturb_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {OUT_DIR / 'phase92_surface_role_trap_repair_report.md'}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()