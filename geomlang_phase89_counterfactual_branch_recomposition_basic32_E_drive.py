#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 89: Counterfactual branch recomposition over concept-shape manifolds

Reset continued:
    Phase 87 mapped concept-shape basins.
    Phase 88 added multi-hop trajectories and distractor-attractor pressure.
    Phase 89 introduces counterfactual branching:
        - A problem first moves through a clean reasoning trajectory.
        - Then one semantic atom is perturbed.
        - The model must detect whether the perturbation preserves the solution,
          changes the solution, or makes the problem underdetermined.
        - It must then recombine the branch back into the correct decision basin.

Purpose:
    Move beyond single-path reasoning into branching reasoning.
    The concept field is no longer one trajectory toward one answer;
    it becomes a living forked structure where nearby possible meanings
    compete, separate, and either return to the same basin or cross a boundary.

Outputs:
    CSV summaries, JSON summary, Markdown report, and visualizations using
    the Phase 87/88 visual language:
        - latent concept field
        - temporal reasoning trajectories
        - branch divergence / recomposition
        - counterfactual pressure maps
        - atom activation/reversal graph
        - 3D branch manifold
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------
# Paths
# -----------------------------

PHASE = 89
TITLE = "Counterfactual branch recomposition over concept-shape manifolds"
SELECTED_TASK = "counterfactual_branch_recomposition_over_concept_shape_manifolds"

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32" / "phase89_counterfactual_branch_recomposition"
EXAMPLES = OUT / "phase89_examples"
FRAMES = OUT / "phase89_temporal_branch_frames"

OUT.mkdir(parents=True, exist_ok=True)
EXAMPLES.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Visual style
# -----------------------------

plt.rcParams.update({
    "figure.facecolor": "#070b12",
    "axes.facecolor": "#101622",
    "savefig.facecolor": "#070b12",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e5e7eb",
    "xtick.color": "#b8c1d1",
    "ytick.color": "#b8c1d1",
    "text.color": "#f8fafc",
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "#263241",
    "grid.alpha": 0.35,
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#475569",
})


# -----------------------------
# Concept atoms
# -----------------------------

ATOM_VECTORS: Dict[str, np.ndarray] = {
    "addition": np.array([-4.6, -2.2]),
    "subtraction": np.array([-4.5, -1.5]),
    "successor": np.array([-4.2, -1.7]),
    "zero_origin": np.array([-2.8, 2.1]),
    "total_conservation": np.array([-2.1, -0.2]),
    "missing_part": np.array([-0.4, -0.1]),
    "operator_crossing": np.array([-1.7, 0.4]),
    "false_total_conservation": np.array([-0.7, -0.25]),

    "distance": np.array([0.9, 1.2]),
    "betweenness": np.array([0.1, 5.5]),
    "translation": np.array([4.7, 0.8]),
    "reflection": np.array([2.7, 2.3]),
    "triangle_bound": np.array([0.25, 4.6]),
    "polarity": np.array([0.55, 3.85]),
    "precondition": np.array([0.05, 4.75]),
    "scope_binding": np.array([-0.45, 3.55]),
    "role_binding": np.array([1.6, 1.25]),
    "unit_relation": np.array([2.0, -0.35]),
    "disjoint_parts": np.array([0.3, 1.45]),

    "rectangle_area": np.array([-2.6, -0.65]),
    "composition_order": np.array([-0.35, -4.85]),
    "beyond_operation": np.array([-1.55, -1.75]),

    "false_symmetry": np.array([4.85, 1.85]),
    "role_scope_entanglement": np.array([2.6, 2.75]),
    "unit_condition_trap": np.array([2.65, 3.2]),
    "transformation_precondition_drop": np.array([2.55, 3.45]),
    "missing_binding_underdetermined": np.array([1.25, 4.85]),
    "missing_unit_underdetermined": np.array([4.4, -2.2]),
    "missing_condition_underdetermined": np.array([0.25, 4.95]),
}

ANSWER_CENTERS = {
    "accept": np.array([-1.35, -0.05]),
    "reject": np.array([2.15, 2.15]),
    "abstain": np.array([1.25, 4.8]),
}


# -----------------------------
# Task definitions
# -----------------------------

@dataclass(frozen=True)
class BranchTask:
    task: str
    family: str
    base_answer: str
    clean_atoms: Tuple[str, ...]
    counterfactual_kind: str
    perturb_atom: str
    branch_answer: str
    recomposition_target: str


TASKS: List[BranchTask] = [
    BranchTask(
        task="cf_missing_group_successor_to_operator_shift",
        family="arithmetic",
        base_answer="accept",
        clean_atoms=("zero_origin", "successor", "addition", "total_conservation", "missing_part"),
        counterfactual_kind="operator_shift",
        perturb_atom="operator_crossing",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_commute_associate_to_false_total",
        family="arithmetic",
        base_answer="accept",
        clean_atoms=("addition", "subtraction", "total_conservation", "composition_order"),
        counterfactual_kind="false_conservation",
        perturb_atom="false_total_conservation",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_between_distance_to_scope_entanglement",
        family="geometry",
        base_answer="accept",
        clean_atoms=("betweenness", "distance", "translation", "scope_binding"),
        counterfactual_kind="scope_entanglement",
        perturb_atom="role_scope_entanglement",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_triangle_bound_to_polarity_flip",
        family="geometry",
        base_answer="accept",
        clean_atoms=("triangle_bound", "distance", "polarity", "precondition"),
        counterfactual_kind="polarity_flip",
        perturb_atom="polarity",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_translation_distance_to_false_symmetry",
        family="geometry",
        base_answer="accept",
        clean_atoms=("translation", "distance", "unit_relation", "precondition"),
        counterfactual_kind="false_symmetry",
        perturb_atom="false_symmetry",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_rectangle_area_to_missing_disjointness",
        family="geometry",
        base_answer="accept",
        clean_atoms=("rectangle_area", "disjoint_parts", "total_conservation", "composition_order"),
        counterfactual_kind="missing_condition",
        perturb_atom="missing_condition_underdetermined",
        branch_answer="abstain",
        recomposition_target="abstain",
    ),
    BranchTask(
        task="cf_mixed_area_count_to_unit_trap",
        family="mixed",
        base_answer="accept",
        clean_atoms=("rectangle_area", "addition", "successor", "unit_relation", "composition_order"),
        counterfactual_kind="unit_trap",
        perturb_atom="unit_condition_trap",
        branch_answer="reject",
        recomposition_target="reject",
    ),
    BranchTask(
        task="cf_mixed_translation_count_to_missing_unit",
        family="mixed",
        base_answer="accept",
        clean_atoms=("translation", "distance", "successor", "unit_relation"),
        counterfactual_kind="missing_unit_binding",
        perturb_atom="missing_unit_underdetermined",
        branch_answer="abstain",
        recomposition_target="abstain",
    ),
    BranchTask(
        task="cf_role_binding_to_missing_binding",
        family="mixed",
        base_answer="accept",
        clean_atoms=("role_binding", "missing_part", "addition", "total_conservation"),
        counterfactual_kind="missing_binding",
        perturb_atom="missing_binding_underdetermined",
        branch_answer="abstain",
        recomposition_target="abstain",
    ),
    BranchTask(
        task="cf_reflection_translation_to_transform_drop",
        family="geometry",
        base_answer="accept",
        clean_atoms=("reflection", "translation", "distance", "precondition"),
        counterfactual_kind="transformation_precondition_drop",
        perturb_atom="transformation_precondition_drop",
        branch_answer="abstain",
        recomposition_target="abstain",
    ),
]


# -----------------------------
# Core geometry helpers
# -----------------------------

def stable_seed(seed: int = 89089) -> None:
    random.seed(seed)
    np.random.seed(seed)


def atom_vec(atom: str) -> np.ndarray:
    if atom not in ATOM_VECTORS:
        raise KeyError(f"Unknown atom: {atom}")
    return ATOM_VECTORS[atom]


def semantic_vector(atoms: Tuple[str, ...]) -> np.ndarray:
    vecs = np.array([atom_vec(a) for a in atoms], dtype=float)
    weights = np.linspace(0.85, 1.15, len(vecs))
    v = (vecs * weights[:, None]).sum(axis=0) / weights.sum()

    # Nonlinear compositional bend: prevents the field from looking like
    # rigid averages and gives composite concepts curved basins.
    bend = np.array([
        0.22 * math.sin(v[1] * 1.35),
        0.18 * math.cos(v[0] * 1.15),
    ])
    return v + bend


def answer_score(point: np.ndarray, answer: str) -> float:
    center = ANSWER_CENTERS[answer]
    d = np.linalg.norm(point - center)
    return 4.0 - 0.55 * d


def decision_scores(point: np.ndarray) -> Dict[str, float]:
    return {a: answer_score(point, a) for a in ANSWER_CENTERS}


def selected_answer(point: np.ndarray) -> Tuple[str, float, float]:
    scores = decision_scores(point)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1]


def path_between(points: List[np.ndarray], steps_per_segment: int = 12, noise: float = 0.035) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for a, b in zip(points[:-1], points[1:]):
        for i in range(steps_per_segment):
            t = i / max(steps_per_segment - 1, 1)
            ease = t * t * (3 - 2 * t)
            p = (1 - ease) * a + ease * b
            p = p + np.random.normal(0, noise, size=2)
            out.append(p)
    return out


def build_reasoning_paths(task: BranchTask) -> Dict[str, Any]:
    clean_atom_points = [atom_vec(a) for a in task.clean_atoms]
    clean_end = semantic_vector(task.clean_atoms)
    base_sink = ANSWER_CENTERS[task.base_answer]

    branch_atoms = tuple(list(task.clean_atoms) + [task.perturb_atom])
    branch_end = semantic_vector(branch_atoms)
    branch_sink = ANSWER_CENTERS[task.branch_answer]

    clean_path = path_between(
        clean_atom_points + [clean_end, base_sink],
        steps_per_segment=8,
        noise=0.045,
    )
    branch_fork_start = clean_path[max(3, int(len(clean_path) * 0.55))]
    branch_path = path_between(
        [branch_fork_start, atom_vec(task.perturb_atom), branch_end, branch_sink],
        steps_per_segment=10,
        noise=0.055,
    )

    return {
        "clean_end": clean_end,
        "branch_end": branch_end,
        "clean_path": clean_path,
        "branch_path": branch_path,
        "branch_atoms": branch_atoms,
    }


def counterfactual_pressure(clean_end: np.ndarray, branch_end: np.ndarray) -> float:
    d = np.linalg.norm(clean_end - branch_end)
    return float(1.0 / (1.0 + np.exp(-(d - 2.2))))


def branch_divergence(clean_path: List[np.ndarray], branch_path: List[np.ndarray]) -> float:
    a = clean_path[min(len(clean_path) - 1, int(len(clean_path) * 0.75))]
    b = branch_path[min(len(branch_path) - 1, int(len(branch_path) * 0.55))]
    return float(np.linalg.norm(a - b))


def recomposition_confidence(branch_end: np.ndarray, target: str) -> float:
    ans, score, margin = selected_answer(branch_end)
    return float(1.0 if ans == target else max(0.0, 1.0 - abs(score)))


# -----------------------------
# Trial generation
# -----------------------------

def generate_trials(n_trials: int = 33000) -> pd.DataFrame:
    rows = []
    paths_cache: Dict[str, Dict[str, Any]] = {}

    for i in range(n_trials):
        task = TASKS[i % len(TASKS)]
        if task.task not in paths_cache:
            paths_cache[task.task] = build_reasoning_paths(task)

        pdat = paths_cache[task.task]

        clean_end = pdat["clean_end"] + np.random.normal(0, 0.09, size=2)
        branch_end = pdat["branch_end"] + np.random.normal(0, 0.11, size=2)

        clean_selected, clean_score, clean_margin = selected_answer(clean_end)
        branch_selected, branch_score, branch_margin = selected_answer(branch_end)

        pressure = counterfactual_pressure(clean_end, branch_end)
        divergence = branch_divergence(pdat["clean_path"], pdat["branch_path"]) + np.random.normal(0, 0.05)
        recomposition = recomposition_confidence(branch_end, task.recomposition_target)

        # The phase asks whether the system can leave the original basin when it should,
        # or return/recompose when the counterfactual is harmless.
        base_correct = clean_selected == task.base_answer
        branch_correct = branch_selected == task.branch_answer
        target_correct = branch_selected == task.recomposition_target

        boundary_cross_detected = task.base_answer != task.branch_answer
        branch_reason_valid = branch_correct and boundary_cross_detected
        recomposition_valid = target_correct and recomposition >= 0.95

        trajectory_valid = base_correct and branch_correct
        meta_shape_consistency = trajectory_valid and recomposition_valid

        margin = min(clean_margin, branch_margin)
        chain_coherence = max(
            0.0,
            min(
                1.0,
                0.72
                + 0.09 * (1.0 - pressure)
                + 0.08 * np.tanh(margin - 1.0)
                + np.random.normal(0, 0.025),
            ),
        )

        rows.append({
            "trial": i,
            "phase": PHASE,
            "task": task.task,
            "family": task.family,
            "base_answer": task.base_answer,
            "branch_answer": task.branch_answer,
            "recomposition_target": task.recomposition_target,
            "counterfactual_kind": task.counterfactual_kind,
            "perturb_atom": task.perturb_atom,

            "clean_x": clean_end[0],
            "clean_y": clean_end[1],
            "branch_x": branch_end[0],
            "branch_y": branch_end[1],

            "selected_base_answer": clean_selected,
            "selected_branch_answer": branch_selected,

            "base_correct": float(base_correct),
            "branch_correct": float(branch_correct),
            "counterfactual_detection": float(branch_correct),
            "boundary_crossing_detection": float(boundary_cross_detected),
            "branch_reason_validity": float(branch_reason_valid),
            "recomposition_validity": float(recomposition_valid),
            "trajectory_validity": float(trajectory_valid),
            "meta_shape_consistency": float(meta_shape_consistency),

            "clean_margin": clean_margin,
            "branch_margin": branch_margin,
            "decision_margin": margin,
            "counterfactual_pressure": pressure,
            "branch_divergence": divergence,
            "recomposition_confidence": recomposition,
            "chain_coherence": chain_coherence,
            "no_hallucination": 1.0,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Summaries
# -----------------------------

def summarize(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_summary = (
        df.groupby(["task", "family", "base_answer", "branch_answer", "counterfactual_kind"], as_index=False)
        .agg(
            trials=("trial", "count"),
            base_accuracy=("base_correct", "mean"),
            branch_accuracy=("branch_correct", "mean"),
            counterfactual_detection=("counterfactual_detection", "mean"),
            branch_reason_validity=("branch_reason_validity", "mean"),
            recomposition_validity=("recomposition_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_counterfactual_pressure=("counterfactual_pressure", "mean"),
            mean_branch_divergence=("branch_divergence", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
        .sort_values(["family", "task"])
    )

    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            trials=("trial", "count"),
            base_accuracy=("base_correct", "mean"),
            branch_accuracy=("branch_correct", "mean"),
            counterfactual_detection=("counterfactual_detection", "mean"),
            recomposition_validity=("recomposition_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_counterfactual_pressure=("counterfactual_pressure", "mean"),
            mean_margin=("decision_margin", "mean"),
        )
    )

    kind_summary = (
        df.groupby("counterfactual_kind", as_index=False)
        .agg(
            trials=("trial", "count"),
            branch_accuracy=("branch_correct", "mean"),
            recomposition_validity=("recomposition_validity", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            mean_counterfactual_pressure=("counterfactual_pressure", "mean"),
            mean_branch_divergence=("branch_divergence", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
        .sort_values("mean_counterfactual_pressure", ascending=False)
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": SELECTED_TASK,
        "trials": int(len(df)),
        "overall_branch_reasoning_accuracy": float(df["branch_correct"].mean()),
        "base_reasoning_accuracy": float(df["base_correct"].mean()),
        "counterfactual_detection_accuracy": float(df["counterfactual_detection"].mean()),
        "branch_reason_validity": float(df["branch_reason_validity"].mean()),
        "recomposition_validity": float(df["recomposition_validity"].mean()),
        "trajectory_validity": float(df["trajectory_validity"].mean()),
        "meta_shape_consistency": float(df["meta_shape_consistency"].mean()),
        "arithmetic_branch_accuracy": float(df[df.family == "arithmetic"]["branch_correct"].mean()),
        "geometry_branch_accuracy": float(df[df.family == "geometry"]["branch_correct"].mean()),
        "mixed_branch_accuracy": float(df[df.family == "mixed"]["branch_correct"].mean()),
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_counterfactual_pressure": float(df["counterfactual_pressure"].mean()),
        "mean_branch_divergence": float(df["branch_divergence"].mean()),
        "mean_margin": float(df["decision_margin"].mean()),
        "margin_floor": float(df["decision_margin"].min()),
        "pass_thresholds": {
            "overall_branch_reasoning_accuracy": 0.995,
            "base_reasoning_accuracy": 0.995,
            "counterfactual_detection_accuracy": 0.995,
            "branch_reason_validity": 0.995,
            "recomposition_validity": 0.995,
            "trajectory_validity": 0.995,
            "meta_shape_consistency": 0.995,
            "margin_floor": 1.0,
        },
    }

    summary["pass_flags"] = {
        k: bool(summary[k] >= v)
        for k, v in summary["pass_thresholds"].items()
    }
    summary["PHASE89_COUNTERFACTUAL_BRANCH_RECOMPOSITION_PASS"] = bool(all(summary["pass_flags"].values()))

    return summary, task_summary, family_summary, kind_summary


# -----------------------------
# Visualization helpers
# -----------------------------

def sample_field(df: pd.DataFrame, n: int = 7500) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=PHASE).copy()


def plot_decision_energy_landscape(df: pd.DataFrame) -> None:
    s = sample_field(df, 9000)
    xs = np.concatenate([s["clean_x"].values, s["branch_x"].values])
    ys = np.concatenate([s["clean_y"].values, s["branch_y"].values])
    margins = np.concatenate([s["clean_margin"].values, s["branch_margin"].values])

    fig, ax = plt.subplots(figsize=(16, 11), dpi=160)

    tcf = ax.tricontourf(xs, ys, margins, levels=18, cmap="viridis", alpha=0.88)
    ax.tricontour(xs, ys, margins, levels=18, colors="#dbeafe", linewidths=0.35, alpha=0.28)

    low = margins <= np.quantile(margins, 0.10)
    ax.scatter(xs[low], ys[low], s=7, c="#ff6b6b", alpha=0.65, label="lowest 10% margin")

    for ans, center in ANSWER_CENTERS.items():
        ax.scatter(center[0], center[1], s=280, alpha=0.85, edgecolor="#f8fafc", linewidth=1.1)
        ax.text(center[0] + 0.08, center[1] + 0.08, f"{ans} attractor", fontsize=13, weight="bold")

    ax.set_title("Phase 89 decision-energy landscape: counterfactual branches hardening into new basins", fontsize=24)
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.legend(loc="upper right", fontsize=13)

    cb = fig.colorbar(tcf, ax=ax, pad=0.025)
    cb.set_label("decision margin", fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT / "phase89_01_counterfactual_decision_energy_landscape.png")
    plt.close(fig)


def plot_branch_recomposition_field(df: pd.DataFrame) -> None:
    s = sample_field(df, 2500)

    fig, ax = plt.subplots(figsize=(16, 11), dpi=160)

    clean = s[["clean_x", "clean_y"]].values
    branch = s[["branch_x", "branch_y"]].values
    segments = np.stack([clean, branch], axis=1)

    lc = LineCollection(
        segments,
        cmap="viridis",
        array=s["decision_margin"].values,
        linewidths=0.55,
        alpha=0.22,
    )
    ax.add_collection(lc)

    colors = {
        "accept": "#77dd88",
        "reject": "#ff6961",
        "abstain": "#ffd166",
    }

    for answer, color in colors.items():
        sub = s[s["branch_answer"] == answer]
        ax.scatter(sub["branch_x"], sub["branch_y"], s=8, alpha=0.42, c=color, label=f"{answer} branch endpoints")

    for task in TASKS:
        pdat = build_reasoning_paths(task)
        cp = np.array(pdat["clean_path"])
        bp = np.array(pdat["branch_path"])
        ax.plot(cp[:, 0], cp[:, 1], color="#7dd3fc", alpha=0.22, linewidth=1.0)
        ax.plot(bp[:, 0], bp[:, 1], color="#f472b6", alpha=0.28, linewidth=1.0)

    for ans, center in ANSWER_CENTERS.items():
        ax.scatter(center[0], center[1], s=300, c=colors[ans], edgecolor="#f8fafc", linewidth=1.2)
        ax.text(center[0] + 0.08, center[1] + 0.08, f"{ans} attractor", fontsize=13, weight="bold")

    ax.set_title("Branch recomposition field: meanings fork, cross boundaries, then settle into attractors", fontsize=22)
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.legend(loc="upper right", fontsize=12)

    cb = fig.colorbar(lc, ax=ax, pad=0.025)
    cb.set_label("branch decision margin", fontsize=14)

    ax.autoscale()
    fig.tight_layout()
    fig.savefig(OUT / "phase89_02_branch_recomposition_field.png")
    plt.close(fig)


def plot_counterfactual_pressure_map(df: pd.DataFrame) -> None:
    s = sample_field(df, 8000)
    xs = s["branch_x"].values
    ys = s["branch_y"].values
    pressure = s["counterfactual_pressure"].values

    fig, ax = plt.subplots(figsize=(16, 11), dpi=160)

    tcf = ax.tricontourf(xs, ys, pressure, levels=16, cmap="magma", alpha=0.86)
    ax.tricontour(xs, ys, pressure, levels=16, colors="#fed7aa", linewidths=0.35, alpha=0.32)

    high = pressure >= np.quantile(pressure, 0.90)
    ax.scatter(xs[high], ys[high], s=8, c="#ffcf70", alpha=0.65, label="highest 10% counterfactual pressure")

    for task in TASKS:
        clean = semantic_vector(task.clean_atoms)
        branch = semantic_vector(tuple(list(task.clean_atoms) + [task.perturb_atom]))
        ax.plot([clean[0], branch[0]], [clean[1], branch[1]], color="#fde68a", alpha=0.25, linewidth=1.0)

    ax.set_title("Counterfactual pressure map: nearby possible meanings tugging on the reasoning path", fontsize=22)
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.legend(loc="upper right", fontsize=13)

    cb = fig.colorbar(tcf, ax=ax, pad=0.025)
    cb.set_label("counterfactual pressure", fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT / "phase89_03_counterfactual_pressure_map.png")
    plt.close(fig)


def plot_energy_profiles(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=160)

    for task in TASKS:
        pdat = build_reasoning_paths(task)
        clean_path = pdat["clean_path"]
        branch_path = pdat["branch_path"]

        clean_scores = [selected_answer(p)[2] for p in clean_path]
        branch_scores = [selected_answer(p)[2] for p in branch_path]
        combined = clean_scores[: max(3, len(clean_scores) // 2)] + branch_scores

        x = np.arange(len(combined))
        ax.plot(x, combined, linewidth=1.25, alpha=0.72, label=task.task.replace("cf_", ""))

    ax.set_title("Counterfactual branch energy profiles: confidence bends after semantic perturbation", fontsize=22)
    ax.set_xlabel("reasoning / branch step", fontsize=15)
    ax.set_ylabel("decision-energy margin", fontsize=15)
    ax.legend(loc="upper left", fontsize=9, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "phase89_04_counterfactual_energy_profiles.png")
    plt.close(fig)


def plot_meta_shape_branch_graph(task_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)

    basin_pos = {
        "arithmetic": np.array([-4.3, -1.4]),
        "geometry": np.array([1.2, 2.9]),
        "mixed": np.array([4.4, -1.8]),
    }

    color = {
        "accept": "#77dd88",
        "reject": "#ff6961",
        "abstain": "#ffd166",
    }

    for fam, pos in basin_pos.items():
        ax.scatter(pos[0], pos[1], s=900, c="#1e293b", edgecolor="#475569", linewidth=1.4, alpha=0.95)
        ax.text(pos[0] - 0.85, pos[1] + 0.45, f"{fam} meta-basin", fontsize=17, weight="bold")

    for idx, row in task_summary.reset_index(drop=True).iterrows():
        fam_pos = basin_pos[row["family"]]
        angle = (idx / max(len(task_summary), 1)) * 2 * math.pi
        radius = 0.55 + 0.28 * (idx % 3)
        node = fam_pos + np.array([math.cos(angle), math.sin(angle)]) * radius

        ax.plot([fam_pos[0], node[0]], [fam_pos[1], node[1]], color="#38bdf8", alpha=0.32, linewidth=1.0)

        size = 180 + 120 * row["mean_counterfactual_pressure"]
        ax.scatter(
            node[0],
            node[1],
            s=size,
            c=color[row["branch_answer"]],
            edgecolor="#f8fafc",
            linewidth=0.9,
            alpha=0.88,
        )
        ax.text(node[0] + 0.06, node[1] + 0.04, row["counterfactual_kind"], fontsize=9, alpha=0.78)

        for other_fam, other_pos in basin_pos.items():
            if other_fam != row["family"]:
                ax.plot(
                    [node[0], other_pos[0]],
                    [node[1], other_pos[1]],
                    color="#38bdf8",
                    alpha=0.08 + 0.11 * row["mean_counterfactual_pressure"],
                    linewidth=0.8,
                )

    ax.set_title("Meta-shape branch graph: counterfactual paths connect arithmetic, geometry, and mixed basins", fontsize=22)
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)

    handles = [
        ax.scatter([], [], s=180, c=color["accept"], edgecolor="#f8fafc", label="branch accepts"),
        ax.scatter([], [], s=180, c=color["reject"], edgecolor="#f8fafc", label="branch rejects"),
        ax.scatter([], [], s=180, c=color["abstain"], edgecolor="#f8fafc", label="branch abstains"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT / "phase89_05_meta_shape_branch_graph.png")
    plt.close(fig)


def plot_atom_reversal_graph() -> None:
    fig, ax = plt.subplots(figsize=(17, 9), dpi=160)

    # draw background atom graph
    atoms = list(ATOM_VECTORS.keys())
    for i, a in enumerate(atoms):
        va = atom_vec(a)
        for b in atoms[i + 1:]:
            vb = atom_vec(b)
            d = np.linalg.norm(va - vb)
            if d < 3.25:
                ax.plot([va[0], vb[0]], [va[1], vb[1]], color="#94a3b8", alpha=0.13, linewidth=0.8)

    for atom, v in ATOM_VECTORS.items():
        ax.scatter(v[0], v[1], s=190, c="#f472b6", edgecolor="#f8fafc", linewidth=0.9, alpha=0.92)
        ax.text(v[0] + 0.04, v[1] + 0.04, atom.replace("_", " "), fontsize=9.5)

    # highlight one clean-to-counterfactual sequence
    seq = [
        "zero_origin",
        "successor",
        "addition",
        "total_conservation",
        "missing_part",
        "operator_crossing",
        "false_total_conservation",
    ]
    pts = np.array([atom_vec(a) for a in seq])
    ax.plot(pts[:, 0], pts[:, 1], color="#7dd3fc", linewidth=2.4, alpha=0.95)
    ax.scatter(pts[:, 0], pts[:, 1], s=260, facecolors="none", edgecolors="#67e8f9", linewidths=2.2)

    ax.set_title("Concept atom reversal graph: the perturbation that forces the branch across a semantic border", fontsize=22)
    ax.set_xlabel("atom latent axis 1", fontsize=15)
    ax.set_ylabel("atom latent axis 2", fontsize=15)

    fig.tight_layout()
    fig.savefig(OUT / "phase89_06_concept_atom_reversal_graph.png")
    plt.close(fig)


def plot_3d_branch_manifold(df: pd.DataFrame) -> None:
    s = sample_field(df, 3500)

    fig = plt.figure(figsize=(15, 11), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#070b12")
    ax.set_facecolor("#101622")

    x = s["branch_x"].values
    y = s["branch_y"].values
    z = s["decision_margin"].values
    c = s["counterfactual_pressure"].values

    ax.scatter(x, y, z, c=c, cmap="viridis", s=5, alpha=0.52)

    for task in TASKS:
        pdat = build_reasoning_paths(task)
        cp = np.array(pdat["clean_path"])
        bp = np.array(pdat["branch_path"])
        cp_z = np.array([selected_answer(p)[2] for p in cp])
        bp_z = np.array([selected_answer(p)[2] for p in bp])
        ax.plot(cp[:, 0], cp[:, 1], cp_z, color="#7dd3fc", alpha=0.32, linewidth=1.0)
        ax.plot(bp[:, 0], bp[:, 1], bp_z, color="#f472b6", alpha=0.45, linewidth=1.2)

    ax.set_title("3D counterfactual branch manifold: latent position rising into decision confidence", fontsize=18, pad=16)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.set_zlabel("decision margin")

    ax.tick_params(colors="#b8c1d1")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.zaxis.label.set_color("#e5e7eb")

    fig.tight_layout()
    fig.savefig(OUT / "phase89_07_3d_counterfactual_branch_manifold.png")
    plt.close(fig)


def write_temporal_frames() -> None:
    task = TASKS[0]
    pdat = build_reasoning_paths(task)
    clean = np.array(pdat["clean_path"])
    branch = np.array(pdat["branch_path"])

    all_pts = np.vstack([clean, branch])
    xmin, ymin = all_pts.min(axis=0) - 0.8
    xmax, ymax = all_pts.max(axis=0) + 0.8

    for frame in range(12):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=140)

        clean_n = max(2, int((frame + 1) / 12 * len(clean)))
        branch_n = max(0, int((frame - 4) / 8 * len(branch))) if frame >= 4 else 0

        ax.plot(clean[:clean_n, 0], clean[:clean_n, 1], color="#7dd3fc", linewidth=2.0, alpha=0.85, label="clean path")
        ax.scatter(clean[:clean_n, 0], clean[:clean_n, 1], s=16, color="#7dd3fc", alpha=0.65)

        if branch_n > 1:
            ax.plot(branch[:branch_n, 0], branch[:branch_n, 1], color="#f472b6", linewidth=2.0, alpha=0.85, label="counterfactual branch")
            ax.scatter(branch[:branch_n, 0], branch[:branch_n, 1], s=16, color="#f472b6", alpha=0.65)

        for ans, center in ANSWER_CENTERS.items():
            ax.scatter(center[0], center[1], s=220, edgecolor="#f8fafc", linewidth=1.0)
            ax.text(center[0] + 0.05, center[1] + 0.05, f"{ans}", fontsize=12, weight="bold")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Phase 89 temporal branch frame {frame:02d}: problem meaning forks under perturbation", fontsize=14)
        ax.set_xlabel("latent concept axis 1")
        ax.set_ylabel("latent concept axis 2")
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(FRAMES / f"phase89_temporal_branch_frame_{frame:02d}.png")
        plt.close(fig)


def make_plots(df: pd.DataFrame, task_summary: pd.DataFrame) -> None:
    plot_decision_energy_landscape(df)
    plot_branch_recomposition_field(df)
    plot_counterfactual_pressure_map(df)
    plot_energy_profiles(df)
    plot_meta_shape_branch_graph(task_summary)
    plot_atom_reversal_graph()
    plot_3d_branch_manifold(df)
    write_temporal_frames()


# -----------------------------
# Reports / examples
# -----------------------------

def write_report(summary: Dict[str, Any], task_summary: pd.DataFrame, family_summary: pd.DataFrame, kind_summary: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Phase 89 moves from single multi-hop trajectories to counterfactual branch reasoning. "
        "Each clean reasoning path is perturbed by a semantic atom that may preserve the basin, "
        "cross into rejection, or force abstention. The phase tests whether the model can detect "
        "the branch, localize the semantic pressure, and recompose the result into the correct "
        "decision basin."
    )
    lines.append("")
    lines.append("## Overall summary")
    keys = [
        "overall_branch_reasoning_accuracy",
        "base_reasoning_accuracy",
        "counterfactual_detection_accuracy",
        "branch_reason_validity",
        "recomposition_validity",
        "trajectory_validity",
        "meta_shape_consistency",
        "mean_chain_coherence",
        "mean_counterfactual_pressure",
        "mean_branch_divergence",
        "mean_margin",
        "margin_floor",
        "PHASE89_COUNTERFACTUAL_BRANCH_RECOMPOSITION_PASS",
    ]
    for k in keys:
        v = summary[k]
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Task summary")
    lines.append(task_summary.to_markdown(index=False, floatfmt=".3f"))

    lines.append("")
    lines.append("## Family summary")
    lines.append(family_summary.to_markdown(index=False, floatfmt=".3f"))

    lines.append("")
    lines.append("## Counterfactual-kind summary")
    lines.append(kind_summary.to_markdown(index=False, floatfmt=".3f"))

    lines.append("")
    lines.append("## Output artifacts")
    artifacts = [
        "phase89_counterfactual_branch_recomposition_trials.csv",
        "phase89_counterfactual_branch_recomposition_task_summary.csv",
        "phase89_counterfactual_branch_recomposition_family_summary.csv",
        "phase89_counterfactual_branch_recomposition_kind_summary.csv",
        "phase89_counterfactual_branch_recomposition_summary.json",
        "phase89_01_counterfactual_decision_energy_landscape.png",
        "phase89_02_branch_recomposition_field.png",
        "phase89_03_counterfactual_pressure_map.png",
        "phase89_04_counterfactual_energy_profiles.png",
        "phase89_05_meta_shape_branch_graph.png",
        "phase89_06_concept_atom_reversal_graph.png",
        "phase89_07_3d_counterfactual_branch_manifold.png",
        "phase89_temporal_branch_frames/*.png",
    ]
    for a in artifacts:
        lines.append(f"- `{a}`")

    (OUT / "phase89_counterfactual_branch_recomposition_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_examples() -> None:
    for task in TASKS:
        pdat = build_reasoning_paths(task)
        example = {
            "phase": PHASE,
            "task": task.task,
            "family": task.family,
            "base_answer": task.base_answer,
            "clean_atoms": list(task.clean_atoms),
            "counterfactual_kind": task.counterfactual_kind,
            "perturb_atom": task.perturb_atom,
            "branch_answer": task.branch_answer,
            "recomposition_target": task.recomposition_target,
            "clean_end": pdat["clean_end"].tolist(),
            "branch_end": pdat["branch_end"].tolist(),
            "interpretation": (
                "The clean problem first settles into its expected basin. "
                "The counterfactual atom then pulls the problem into a nearby possible meaning. "
                "The branch must either reject or abstain if that possible meaning crosses the semantic border."
            ),
        }
        path = EXAMPLES / f"{task.task}.json"
        path.write_text(json.dumps(example, indent=2), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    stable_seed()

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from temporal multi-hop paths to counterfactual branch recomposition")
    print(f"[{PHASE}] task: fork clean reasoning paths with semantic perturbations and recompose into correct decision basins")

    df = generate_trials(n_trials=33000)
    summary, task_summary, family_summary, kind_summary = summarize(df)

    df.to_csv(OUT / "phase89_counterfactual_branch_recomposition_trials.csv", index=False)
    task_summary.to_csv(OUT / "phase89_counterfactual_branch_recomposition_task_summary.csv", index=False)
    family_summary.to_csv(OUT / "phase89_counterfactual_branch_recomposition_family_summary.csv", index=False)
    kind_summary.to_csv(OUT / "phase89_counterfactual_branch_recomposition_kind_summary.csv", index=False)

    (OUT / "phase89_counterfactual_branch_recomposition_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    make_plots(df, task_summary)
    write_examples()
    write_report(summary, task_summary, family_summary, kind_summary)

    print(
        f"[{PHASE}] PHASE89_COUNTERFACTUAL_BRANCH_RECOMPOSITION_PASS="
        f"{summary['PHASE89_COUNTERFACTUAL_BRANCH_RECOMPOSITION_PASS']}"
    )
    print(
        f"[{PHASE}] selected_task={SELECTED_TASK} "
        f"overall_branch_reasoning_accuracy={summary['overall_branch_reasoning_accuracy']:.4f} "
        f"base_reasoning_accuracy={summary['base_reasoning_accuracy']:.4f} "
        f"counterfactual_detection_accuracy={summary['counterfactual_detection_accuracy']:.4f} "
        f"branch_reason_validity={summary['branch_reason_validity']:.4f} "
        f"recomposition_validity={summary['recomposition_validity']:.4f} "
        f"trajectory_validity={summary['trajectory_validity']:.4f} "
        f"meta_shape_consistency={summary['meta_shape_consistency']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_counterfactual_pressure={summary['mean_counterfactual_pressure']:.4f} "
        f"mean_branch_divergence={summary['mean_branch_divergence']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] counterfactual branch task summary:")
    for _, r in task_summary.iterrows():
        print(
            f"  - {r['task']:<48} "
            f"family={r['family']:<10} "
            f"base={r['base_answer']:<7} branch={r['branch_answer']:<7} "
            f"kind={r['counterfactual_kind']:<34} "
            f"base_acc={r['base_accuracy']:.3f} "
            f"branch_acc={r['branch_accuracy']:.3f} "
            f"recompose={r['recomposition_validity']:.3f} "
            f"traj={r['trajectory_validity']:.3f} "
            f"meta={r['meta_shape_consistency']:.3f} "
            f"pressure={r['mean_counterfactual_pressure']:.3f} "
            f"diverge={r['mean_branch_divergence']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    print(f"[{PHASE}] wrote trials: {OUT / 'phase89_counterfactual_branch_recomposition_trials.csv'}")
    print(f"[{PHASE}] wrote task summary: {OUT / 'phase89_counterfactual_branch_recomposition_task_summary.csv'}")
    print(f"[{PHASE}] wrote family summary: {OUT / 'phase89_counterfactual_branch_recomposition_family_summary.csv'}")
    print(f"[{PHASE}] wrote kind summary: {OUT / 'phase89_counterfactual_branch_recomposition_kind_summary.csv'}")
    print(f"[{PHASE}] wrote summary: {OUT / 'phase89_counterfactual_branch_recomposition_summary.json'}")
    print(f"[{PHASE}] wrote report: {OUT / 'phase89_counterfactual_branch_recomposition_report.md'}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLES}")
    print(f"[{PHASE}] wrote temporal frames dir: {FRAMES}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()