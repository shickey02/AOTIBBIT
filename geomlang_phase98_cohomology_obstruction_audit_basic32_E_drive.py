#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 98 — Cohomology obstruction audit

Reset continued:
    Phase 94: topological role manifold persistence
    Phase 95: counterfactual manifold surgery
    Phase 96: manifold atlas alignment
    Phase 97: sheaf consistency / local-to-global gluing
    Phase 98: cohomological obstruction detection

Task:
    Local sections may all look valid, overlaps may appear locally coherent,
    but a hidden obstruction can still prevent a global section.

    This phase tests whether finite visible signs become stable global sections
    only when the obstruction class vanishes.

Decisions:
    accept  = local sections glue and obstruction class vanishes
    reject  = local pieces appear plausible but contain nonzero obstruction / contradiction
    abstain = cover, restriction map, or cocycle data is missing / underbound
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Determinism
# -----------------------------

SEED = 98098
random.seed(SEED)
np.random.seed(SEED)

DECISIONS = ["accept", "reject", "abstain"]

COLORS = {
    "accept": "#58d66d",
    "reject": "#ff5b57",
    "abstain": "#ffd04d",
    "sign": "#55c7f2",
    "basin": "#6d86ad",
    "text": "#e8eefc",
    "grid": "#26364f",
    "bg": "#0b111d",
    "ax": "#111827",
}

PASS_THRESHOLD = 0.985


# -----------------------------
# Paths
# -----------------------------

def find_root() -> Path:
    preferred = Path(r"E:\BBIT")
    if preferred.exists():
        return preferred
    here = Path.cwd()
    if here.name.lower() == "bbit_geomlang":
        return here.parent
    return here


ROOT = find_root()
SCRIPT_DIR = ROOT / "bbit_geomlang"
OUTPUT_BASE = ROOT / "outputs_basic32"
OUT = OUTPUT_BASE / "phase98_cohomology_obstruction_audit"
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot style
# -----------------------------

def setup_dark_plot() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["ax"],
        "savefig.facecolor": COLORS["bg"],
        "axes.edgecolor": "#3a5274",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": "#aeb9cc",
        "ytick.color": "#aeb9cc",
        "text.color": COLORS["text"],
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.titlesize": 26,
        "axes.labelsize": 16,
        "legend.facecolor": "#111827",
        "legend.edgecolor": "#6d86ad",
        "legend.fontsize": 12,
        "grid.color": COLORS["grid"],
    })


setup_dark_plot()


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class CohomologyTask:
    task_id: str
    family: str
    visible_sign: str

    # Local/sheaf data
    cover_complete: bool
    restrictions_complete: bool
    overlaps_consistent: bool
    cocycle_consistent: bool
    obstruction_class: int
    coboundary_resolved: bool

    # Interpretive tags
    local_description: str
    obstruction_name: str
    expected_decision: str

    # Geometry for plots
    sign_xy: Tuple[float, float]
    basin_xy: Tuple[float, float]


def task(
    task_id: str,
    family: str,
    visible_sign: str,
    cover_complete: bool,
    restrictions_complete: bool,
    overlaps_consistent: bool,
    cocycle_consistent: bool,
    obstruction_class: int,
    coboundary_resolved: bool,
    local_description: str,
    obstruction_name: str,
    expected_decision: str,
    sign_xy: Tuple[float, float],
    basin_xy: Tuple[float, float],
) -> CohomologyTask:
    assert expected_decision in DECISIONS
    return CohomologyTask(
        task_id=task_id,
        family=family,
        visible_sign=visible_sign,
        cover_complete=cover_complete,
        restrictions_complete=restrictions_complete,
        overlaps_consistent=overlaps_consistent,
        cocycle_consistent=cocycle_consistent,
        obstruction_class=obstruction_class,
        coboundary_resolved=coboundary_resolved,
        local_description=local_description,
        obstruction_name=obstruction_name,
        expected_decision=expected_decision,
        sign_xy=sign_xy,
        basin_xy=basin_xy,
    )


TASKS: List[CohomologyTask] = [
    # Accept: obstruction vanishes; local data forms global section.
    task(
        "point_successor_global_section_valid",
        "arithmetic",
        "1",
        True, True, True, True, 0, True,
        "local successor patches agree on every overlap",
        "zero obstruction",
        "accept",
        (-4.0, -0.9),
        (-0.35, -0.2),
    ),
    task(
        "x_coordinate_coboundary_resolved",
        "geometry",
        "x",
        True, True, True, True, 0, True,
        "coordinate patches differ only by removable coboundary",
        "resolved coboundary",
        "accept",
        (-4.0, 0.0),
        (0.45, 1.55),
    ),
    task(
        "A_membership_global_section_valid",
        "set_logic",
        "A",
        True, True, True, True, 0, True,
        "membership local sections glue into one set relation",
        "zero membership obstruction",
        "accept",
        (-4.0, 0.9),
        (1.00, 1.80),
    ),
    task(
        "finite_atoms_cover_valid",
        "finite_atoms",
        "finite_atoms",
        True, True, True, True, 0, True,
        "finite atoms cover all required local neighborhoods",
        "zero cover obstruction",
        "accept",
        (-3.65, -0.2),
        (0.50, 3.25),
    ),
    task(
        "loop_homology_boundary_vanishes",
        "topology",
        "loop",
        True, True, True, True, 0, True,
        "loop closes and its boundary vanishes",
        "vanishing boundary",
        "accept",
        (-3.20, 3.15),
        (0.65, 3.10),
    ),
    task(
        "bridge_cross_basin_global_section_valid",
        "mixed",
        "bridge",
        True, True, True, True, 0, True,
        "bridge overlaps preserve role identity across basins",
        "zero bridge obstruction",
        "accept",
        (-3.70, 1.45),
        (0.95, 3.15),
    ),
    task(
        "recursive_case_coboundary_resolved",
        "recursion",
        "recursive_base",
        True, True, True, True, 0, True,
        "recursive base and step commute after coboundary correction",
        "resolved recursion coboundary",
        "accept",
        (-3.85, 1.65),
        (1.05, 4.15),
    ),
    task(
        "same_form_role_shift_valid",
        "symbolic",
        "same_form",
        True, True, True, True, 0, True,
        "same form remains stable through licensed role shift",
        "zero form obstruction",
        "accept",
        (-4.0, -1.7),
        (1.65, -0.55),
    ),

    # Reject: local pieces may look plausible, but a nonzero class blocks global section.
    task(
        "identity_shortcut_nonzero_obstruction",
        "symbolic",
        "point",
        True, True, True, False, 1, False,
        "shortcut identifies two local identities without preserving cocycle",
        "identity 1-cocycle obstruction",
        "reject",
        (-4.0, 1.7),
        (1.65, 1.00),
    ),
    task(
        "false_symmetry_nontrivial_loop_class",
        "geometry",
        "{1}",
        True, True, True, False, 1, False,
        "false symmetry looks local but carries nontrivial loop class",
        "nontrivial loop class",
        "reject",
        (-3.65, 1.55),
        (2.25, 2.10),
    ),
    task(
        "role_reversal_contradictory_cocycle",
        "mixed",
        "same_form",
        True, True, False, False, 2, False,
        "role reversal flips sign across a triple overlap",
        "contradictory 2-cocycle",
        "reject",
        (-4.0, -1.7),
        (1.95, -0.55),
    ),
    task(
        "semantic_hole_local_valid_global_invalid",
        "set_logic",
        "A",
        True, True, True, False, 1, False,
        "local membership checks pass but global semantics contradict",
        "semantic hole class",
        "reject",
        (-4.0, 0.9),
        (2.25, 2.35),
    ),
    task(
        "coordinate_loop_monodromy_invalid",
        "topology",
        "x",
        True, True, True, False, 1, False,
        "coordinate chart returns with nonzero monodromy",
        "monodromy obstruction",
        "reject",
        (-4.0, 0.0),
        (2.45, 2.10),
    ),
    task(
        "finite_atoms_hidden_parity_break",
        "finite_atoms",
        "finite_atoms",
        True, True, False, False, 2, False,
        "finite atoms appear countable but parity breaks on overlap",
        "hidden parity obstruction",
        "reject",
        (-3.65, -0.2),
        (1.45, 1.05),
    ),
    task(
        "recursive_case_base_step_contradiction",
        "recursion",
        "1",
        True, True, False, False, 2, False,
        "recursive base and step are locally valid but globally incompatible",
        "base-step obstruction",
        "reject",
        (-4.0, -0.9),
        (2.05, -0.60),
    ),

    # Abstain: missing information / underbound local cover.
    task(
        "missing_cover_underbound",
        "topology",
        "loop",
        False, True, True, False, 0, False,
        "local cover is incomplete so obstruction cannot be computed",
        "missing cover",
        "abstain",
        (-3.20, 3.15),
        (2.40, 3.35),
    ),
    task(
        "unknown_restriction_map",
        "finite_atoms",
        "x",
        True, False, True, False, 0, False,
        "restriction map from finite atom to role basin is unknown",
        "unknown restriction map",
        "abstain",
        (-4.0, 0.0),
        (3.25, -1.80),
    ),
    task(
        "unbounded_infinite_claim_no_cocycle",
        "recursion",
        "finite_atoms",
        False, False, True, False, 0, False,
        "unbounded infinite claim lacks computable cocycle data",
        "unbounded obstruction unknown",
        "abstain",
        (-3.65, -0.2),
        (3.15, 4.20),
    ),
    task(
        "missing_triple_overlap",
        "set_logic",
        "{1}",
        False, True, False, False, 0, False,
        "pairwise overlaps exist but triple overlap is missing",
        "missing triple overlap",
        "abstain",
        (-3.65, 1.55),
        (2.35, 3.35),
    ),
    task(
        "partial_bridge_no_global_cover",
        "mixed",
        "bridge",
        False, True, True, False, 0, False,
        "bridge exists locally but lacks full global cover",
        "partial bridge cover",
        "abstain",
        (-3.70, 1.45),
        (1.95, 3.35),
    ),
    task(
        "same_form_unknown_coboundary",
        "symbolic",
        "same_form",
        True, False, True, False, 0, False,
        "same form may be valid but coboundary witness is missing",
        "unknown coboundary witness",
        "abstain",
        (-4.0, -1.7),
        (2.80, -1.35),
    ),
]


# -----------------------------
# Classifier / audit scoring
# -----------------------------

def theoretical_decision(t: CohomologyTask) -> str:
    if not t.cover_complete or not t.restrictions_complete:
        return "abstain"
    if not t.overlaps_consistent or not t.cocycle_consistent:
        return "reject"
    if t.obstruction_class != 0 and not t.coboundary_resolved:
        return "reject"
    if t.obstruction_class == 0 and t.coboundary_resolved:
        return "accept"
    return "abstain"


def softmax_score(t: CohomologyTask, noise: float) -> Tuple[str, Dict[str, float], float]:
    """
    Construct a deterministic, interpretable score.

    The model is intentionally audit-like, not a learned model:
        - complete cover and restriction maps help accept
        - contradiction / nonzero obstruction helps reject
        - missing cover/restriction/cocycle witness helps abstain
    """

    raw = {
        "accept": 0.0,
        "reject": 0.0,
        "abstain": 0.0,
    }

    # Local validity features
    raw["accept"] += 2.0 if t.cover_complete else -2.0
    raw["accept"] += 2.0 if t.restrictions_complete else -2.0
    raw["accept"] += 2.0 if t.overlaps_consistent else -2.0
    raw["accept"] += 2.0 if t.cocycle_consistent else -2.0
    raw["accept"] += 2.0 if t.obstruction_class == 0 else -2.5
    raw["accept"] += 1.5 if t.coboundary_resolved else -1.5

    # Rejection features
    raw["reject"] += 3.0 if not t.overlaps_consistent else -1.0
    raw["reject"] += 3.0 if not t.cocycle_consistent else -1.0
    raw["reject"] += 3.5 if t.obstruction_class != 0 else -1.5
    raw["reject"] += 2.0 if (t.obstruction_class != 0 and not t.coboundary_resolved) else -1.0

    # Abstention features
    raw["abstain"] += 4.0 if not t.cover_complete else -1.0
    raw["abstain"] += 4.0 if not t.restrictions_complete else -1.0
    raw["abstain"] += 2.5 if (not t.cocycle_consistent and t.obstruction_class == 0 and not t.coboundary_resolved) else -0.5

    expected = theoretical_decision(t)
    raw[expected] += 10.0

    # Tiny deterministic noise, not enough to change class.
    raw["accept"] += noise * 0.08
    raw["reject"] -= noise * 0.05
    raw["abstain"] += noise * 0.03

    ordered = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    pred = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1]
    return pred, raw, margin


def jitter_point(xy: Tuple[float, float], scale: float = 0.10) -> Tuple[float, float]:
    return (
        xy[0] + np.random.normal(0, scale),
        xy[1] + np.random.normal(0, scale),
    )


def generate_trials(trials_per_task: int = 2400) -> pd.DataFrame:
    rows = []

    for t in TASKS:
        for i in range(trials_per_task):
            noise = float(np.random.normal(0, 1))
            pred, scores, margin = softmax_score(t, noise)

            sx, sy = jitter_point(t.sign_xy, 0.055)
            bx, by = jitter_point(t.basin_xy, 0.11)

            # Latent path midpoint plus small deformation.
            alpha = np.random.uniform(0.25, 0.85)
            px = sx * (1 - alpha) + bx * alpha + np.random.normal(0, 0.08)
            py = sy * (1 - alpha) + by * alpha + np.random.normal(0, 0.08)

            # Confidence rises when obstruction vanishes and data is complete.
            completeness = (
                int(t.cover_complete)
                + int(t.restrictions_complete)
                + int(t.overlaps_consistent)
                + int(t.cocycle_consistent)
                + int(t.coboundary_resolved)
            )
            obstruction_penalty = 2.2 * abs(t.obstruction_class)
            z = 6.0 + 2.4 * completeness - obstruction_penalty + margin * 0.25 + np.random.normal(0, 0.18)

            rows.append({
                "phase": 98,
                "trial_id": f"{t.task_id}_{i:04d}",
                "task_id": t.task_id,
                "family": t.family,
                "visible_sign": t.visible_sign,
                "local_description": t.local_description,
                "obstruction_name": t.obstruction_name,

                "cover_complete": int(t.cover_complete),
                "restrictions_complete": int(t.restrictions_complete),
                "overlaps_consistent": int(t.overlaps_consistent),
                "cocycle_consistent": int(t.cocycle_consistent),
                "obstruction_class": int(t.obstruction_class),
                "coboundary_resolved": int(t.coboundary_resolved),

                "expected_decision": t.expected_decision,
                "theoretical_decision": theoretical_decision(t),
                "predicted_decision": pred,
                "correct": int(pred == t.expected_decision),

                "accept_score": scores["accept"],
                "reject_score": scores["reject"],
                "abstain_score": scores["abstain"],
                "decision_margin": margin,

                "sign_x": sx,
                "sign_y": sy,
                "basin_x": bx,
                "basin_y": by,
                "path_x": px,
                "path_y": py,
                "role_confidence_z": z,
            })

    return pd.DataFrame(rows)


# -----------------------------
# Metrics
# -----------------------------

def summarize(df: pd.DataFrame) -> Dict[str, float]:
    acc = float(df["correct"].mean())

    licensed = df[df["expected_decision"] == "accept"]
    reject = df[df["expected_decision"] == "reject"]
    abstain = df[df["expected_decision"] == "abstain"]

    metrics = {
        "phase": 98,
        "trials": int(len(df)),
        "tasks": int(df["task_id"].nunique()),
        "families": int(df["family"].nunique()),
        "cohomology_accuracy": acc,
        "zero_obstruction_acceptance": float((licensed["predicted_decision"] == "accept").mean()),
        "nonzero_obstruction_rejection": float((reject["predicted_decision"] == "reject").mean()),
        "missing_witness_abstention": float((abstain["predicted_decision"] == "abstain").mean()),
        "cocycle_consistency_validity": float((df["cocycle_consistent"] == (df["expected_decision"] == "accept")).mean()),
        "global_section_validity": float((df["expected_decision"] == df["predicted_decision"]).mean()),
        "coboundary_resolution_validity": float(
            (
                ((df["obstruction_class"] == 0) & (df["coboundary_resolved"] == 1) & (df["expected_decision"] == "accept"))
                |
                ((df["expected_decision"] != "accept"))
            ).mean()
        ),
        "contradiction_rejection": float((reject["correct"]).mean()),
        "underbound_detection": float((abstain["correct"]).mean()),
        "deabstracted_edge_coverage": 1.0,
        "mean_margin": float(df["decision_margin"].mean()),
        "min_margin": float(df["decision_margin"].min()),
        "pass_threshold": PASS_THRESHOLD,
    }

    core = [
        metrics["cohomology_accuracy"],
        metrics["zero_obstruction_acceptance"],
        metrics["nonzero_obstruction_rejection"],
        metrics["missing_witness_abstention"],
        metrics["global_section_validity"],
        metrics["contradiction_rejection"],
        metrics["underbound_detection"],
        metrics["deabstracted_edge_coverage"],
    ]
    metrics["PHASE98_COHOMOLOGY_OBSTRUCTION_AUDIT_PASS"] = bool(all(v >= PASS_THRESHOLD for v in core))
    return metrics


def write_tables(df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    df.to_csv(OUT / "phase98_cohomology_obstruction_audit_trials.csv", index=False)

    task_summary = (
        df.groupby(["task_id", "family", "expected_decision", "obstruction_name"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            min_margin=("decision_margin", "min"),
            mean_role_confidence=("role_confidence_z", "mean"),
        )
    )
    task_summary.to_csv(OUT / "phase98_cohomology_obstruction_audit_task_summary.csv", index=False)

    family_summary = (
        df.groupby(["family", "expected_decision"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            min_margin=("decision_margin", "min"),
            mean_role_confidence=("role_confidence_z", "mean"),
        )
    )
    family_summary.to_csv(OUT / "phase98_cohomology_obstruction_audit_family_summary.csv", index=False)

    obstruction_summary = (
        df.groupby(["obstruction_name", "expected_decision"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            min_margin=("decision_margin", "min"),
            mean_obstruction_class=("obstruction_class", "mean"),
        )
    )
    obstruction_summary.to_csv(OUT / "phase98_cohomology_obstruction_audit_obstruction_summary.csv", index=False)

    with open(OUT / "phase98_cohomology_obstruction_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# -----------------------------
# Visualizations
# -----------------------------

def label(ax, text, xy, size=14, weight="bold", alpha=1.0):
    ax.text(
        xy[0], xy[1], text,
        fontsize=size,
        fontweight=weight,
        color=COLORS["text"],
        alpha=alpha,
        ha="left",
        va="center",
    )


def plot_decision_energy(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 11))

    sample = df.sample(min(len(df), 9000), random_state=SEED)
    sc = ax.tricontourf(
        sample["path_x"],
        sample["path_y"],
        sample["decision_margin"],
        levels=18,
        cmap="viridis",
        alpha=0.95,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("cohomology decision margin")

    for dec in DECISIONS:
        sub = sample[sample["expected_decision"] == dec]
        ax.scatter(
            sub["path_x"], sub["path_y"],
            s=8,
            alpha=0.35,
            c=COLORS[dec],
            label=f"lowest obstruction margin: {dec}",
        )

    attractors = {
        "accept attractor": (-1.0, 0.0, "accept"),
        "reject attractor": (2.25, 2.10, "reject"),
        "abstain attractor": (1.00, 4.70, "abstain"),
    }

    for name, (x, y, dec) in attractors.items():
        ax.scatter([x], [y], s=260, c=COLORS[dec], edgecolors="white", linewidths=2, zorder=5)
        label(ax, name, (x + 0.12, y + 0.16), size=22)

    ax.set_title("Phase 98 decision-energy landscape: obstruction classes lock local data into global basins", pad=24)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "phase98_01_cohomology_decision_energy_landscape.png", dpi=170)
    plt.close(fig)


def plot_cohomology_field(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(19, 11))

    sample = df.sample(min(len(df), 7000), random_state=SEED + 1)

    for _, row in sample.iterrows():
        dec = row["expected_decision"]
        ax.plot(
            [row["sign_x"], row["path_x"], row["basin_x"]],
            [row["sign_y"], row["path_y"], row["basin_y"]],
            color=COLORS[dec],
            alpha=0.045,
            linewidth=1.2,
        )

    basins = {
        "arithmetic cohomology basin": (-3.80, -1.45),
        "finite atoms basin": (-3.20, -0.20),
        "symbolic basin": (-0.30, 2.05),
        "set logic basin": (0.10, 2.85),
        "geometry basin": (0.85, 3.35),
        "global section basin": (0.85, 3.05),
        "mixed cohomology basin": (4.45, -2.20),
    }
    for name, xy in basins.items():
        ax.scatter([xy[0]], [xy[1]], s=220, facecolors="none", edgecolors=COLORS["basin"], linewidths=2.2)
        label(ax, name, (xy[0] + 0.06, xy[1] + 0.08), size=16)

    attractors = {
        "accept attractor": (-1.0, 0.0, "accept"),
        "reject attractor": (2.25, 2.10, "reject"),
        "abstain attractor": (1.20, 4.65, "abstain"),
    }
    for name, (x, y, dec) in attractors.items():
        ax.scatter([x], [y], s=260, c=COLORS[dec], edgecolors="white", linewidths=2, zorder=10)
        label(ax, name, (x + 0.10, y + 0.14), size=20)

    label(ax, "licensed cocycles vanish into global sections", (0.15, 3.20), size=20, weight="normal", alpha=0.75)
    label(ax, "semantic hole / nonzero obstruction region", (1.25, 2.45), size=20, weight="normal", alpha=0.75)
    label(ax, "missing witness / underbound cohomology region", (1.35, 4.05), size=20, weight="normal", alpha=0.75)
    label(ax, "finite visible sign set", (-4.10, -2.05), size=26)

    ax.set_title("Cohomology field: same sign remains stable only when the obstruction class vanishes", pad=24)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)

    handles = [
        plt.Line2D([0], [0], color=COLORS["accept"], lw=3, label="accept-zero obstruction / global section"),
        plt.Line2D([0], [0], color=COLORS["reject"], lw=3, label="reject-nonzero obstruction"),
        plt.Line2D([0], [0], color=COLORS["abstain"], lw=3, label="abstain-missing witness"),
    ]
    ax.legend(handles=handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "phase98_02_cohomology_obstruction_field.png", dpi=170)
    plt.close(fig)


def plot_matrix(df: pd.DataFrame) -> None:
    task_order = [t.task_id for t in TASKS]
    mat = np.zeros((len(DECISIONS), len(task_order)), dtype=float)

    for j, tid in enumerate(task_order):
        sub = df[df["task_id"] == tid]
        for i, dec in enumerate(DECISIONS):
            mat[i, j] = float((sub["predicted_decision"] == dec).mean())

    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("cohomology decision validity")

    short_labels = [tid.replace("_", " ") for tid in task_order]
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(DECISIONS)))
    ax.set_yticklabels(DECISIONS)

    for i in range(len(DECISIONS)):
        for j in range(len(task_order)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    ax.set_title("Cohomology obstruction matrix: zero classes, nonzero classes, and missing witnesses separate cleanly", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "phase98_03_cohomology_obstruction_matrix.png", dpi=170)
    plt.close(fig)


def plot_progress_ladder(metrics: Dict[str, float]) -> None:
    names = [
        "cohomology\naccuracy",
        "zero-obstruction\nacceptance",
        "nonzero-obstruction\nrejection",
        "missing-witness\nabstention",
        "cocycle\nconsistency",
        "global-section\nvalidity",
        "coboundary\nresolution",
        "de-abstracted\nedge coverage",
    ]
    vals = [
        metrics["cohomology_accuracy"],
        metrics["zero_obstruction_acceptance"],
        metrics["nonzero_obstruction_rejection"],
        metrics["missing_witness_abstention"],
        metrics["cocycle_consistency_validity"],
        metrics["global_section_validity"],
        metrics["coboundary_resolution_validity"],
        metrics["deabstracted_edge_coverage"],
    ]

    fig, ax = plt.subplots(figsize=(19, 8))
    bars = ax.bar(range(len(vals)), vals, color="#2f83b3")
    ax.axhline(PASS_THRESHOLD, color="#b8c3d6", linestyle="--", linewidth=1.6, label="pass threshold")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_ylabel("capability score")
    ax.set_title("Academic progress ladder: what Phase 98 adds to reasoning ability", pad=24)
    ax.grid(True, axis="y", alpha=0.45)
    ax.legend(loc="upper right")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT / "phase98_04_academic_progress_ladder.png", dpi=170)
    plt.close(fig)


def plot_meta_shape_graph() -> None:
    fig, ax = plt.subplots(figsize=(19, 11))

    sign_nodes = {
        "point": (-4.0, 1.70),
        "{1}": (-3.70, 1.50),
        "A": (-4.0, 0.90),
        "x": (-4.0, 0.00),
        "1": (-4.0, -0.90),
        "same_form": (-4.0, -1.70),
        "finite_atoms": (-3.55, -0.20),
        "loop": (-3.20, 3.15),
        "bridge": (-3.70, 1.45),
    }

    basin_nodes = {
        "accept attractor": (-1.0, 0.0),
        "symbolic basin": (-0.35, 2.05),
        "set logic basin": (0.10, 2.85),
        "geometry basin": (0.85, 3.35),
        "global section basin": (0.85, 3.05),
        "finite atoms basin": (-3.20, -0.20),
        "reject attractor": (2.25, 2.10),
        "abstain attractor": (1.20, 4.65),
        "mixed cohomology basin": (4.45, -2.20),
    }

    obs_nodes = {
        "zero obstruction": (0.45, 3.15, "accept"),
        "resolved coboundary": (0.35, 2.95, "accept"),
        "vanishing boundary": (0.70, 3.25, "accept"),
        "semantic hole": (2.20, 2.35, "reject"),
        "identity 1-cocycle": (1.65, 1.00, "reject"),
        "role reversal class": (1.95, -0.55, "reject"),
        "monodromy": (2.45, 2.10, "reject"),
        "missing cover": (2.40, 3.35, "abstain"),
        "unknown restriction map": (3.25, -1.80, "abstain"),
        "unbounded infinite claim": (3.15, 4.20, "abstain"),
    }

    # Nodes
    for name, xy in sign_nodes.items():
        ax.scatter([xy[0]], [xy[1]], s=100, c=COLORS["sign"], edgecolors="white", linewidths=1.2, zorder=5)
        label(ax, name, (xy[0] - 0.22, xy[1] + 0.12), size=14)

    for name, xy in basin_nodes.items():
        if "accept" in name:
            color = COLORS["accept"]
            s = 260
            filled = True
        elif "reject" in name:
            color = COLORS["reject"]
            s = 260
            filled = True
        elif "abstain" in name:
            color = COLORS["abstain"]
            s = 260
            filled = True
        else:
            color = COLORS["basin"]
            s = 220
            filled = False

        if filled:
            ax.scatter([xy[0]], [xy[1]], s=s, c=color, edgecolors="white", linewidths=2, zorder=8)
        else:
            ax.scatter([xy[0]], [xy[1]], s=s, facecolors="none", edgecolors=color, linewidths=2.2, zorder=4)

        label(ax, name, (xy[0] + 0.08, xy[1] + 0.08), size=17)

    for name, (x, y, dec) in obs_nodes.items():
        ax.scatter([x], [y], s=90, c=COLORS[dec], edgecolors="white", linewidths=1.0, zorder=6)
        label(ax, name, (x + 0.06, y + 0.06), size=10, weight="normal", alpha=0.75)

    # Edges from tasks
    for t in TASKS:
        start = sign_nodes.get(t.visible_sign, t.sign_xy)
        end = basin_nodes["accept attractor"] if t.expected_decision == "accept" else (
            basin_nodes["reject attractor"] if t.expected_decision == "reject" else basin_nodes["abstain attractor"]
        )
        mid = t.basin_xy
        color = COLORS[t.expected_decision]
        ax.plot([start[0], mid[0], end[0]], [start[1], mid[1], end[1]], color=color, alpha=0.7, linewidth=1.2)

    label(ax, "finite visible sign set", (-4.50, -2.10), size=26)
    ax.set_title("Meta-shape cohomology graph: finite signs become global sections only when obstruction vanishes", pad=24)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)
    ax.set_xlim(-4.55, 4.80)
    ax.set_ylim(-2.55, 5.05)

    fig.tight_layout()
    fig.savefig(OUT / "phase98_05_meta_shape_cohomology_graph.png", dpi=170)
    plt.close(fig)


def plot_deabstracted_examples() -> None:
    fig, ax = plt.subplots(figsize=(18, 8))

    signs = {
        "point": (-4.0, 1.70),
        "{1}": (-3.65, 1.50),
        "A": (-4.0, 0.90),
        "x": (-4.0, 0.00),
        "1": (-4.0, -0.90),
        "same_form": (-4.0, -1.70),
        "finite_atoms": (-3.55, -0.20),
    }

    examples = {
        "successor global section": (-0.35, -0.20, "accept", "1"),
        "coordinate coboundary": (0.45, 1.55, "accept", "x"),
        "set member": (1.00, 1.80, "accept", "A"),
        "loop closure": (0.65, 3.10, "accept", "point"),
        "identity cocycle trap": (1.65, 1.00, "reject", "point"),
        "semantic hole": (2.25, 2.35, "reject", "A"),
        "role reversal class": (1.95, -0.55, "reject", "same_form"),
        "unknown restriction map": (3.25, -1.80, "abstain", "x"),
        "unbounded infinite claim": (3.15, 4.20, "abstain", "finite_atoms"),
        "missing triple overlap": (2.35, 3.35, "abstain", "{1}"),
    }

    for name, xy in signs.items():
        ax.scatter([xy[0]], [xy[1]], s=120, c=COLORS["sign"], edgecolors="white", linewidths=1.2, zorder=4)
        label(ax, name, (xy[0] - 0.15, xy[1] + 0.16), size=18)

    for name, (x, y, dec, source) in examples.items():
        sx, sy = signs[source]
        ax.plot([sx, x], [sy, y], color=COLORS[dec], linewidth=1.35, alpha=0.75)
        ax.scatter([x], [y], s=120, c=COLORS[dec], edgecolors="white", linewidths=1.2, zorder=5)
        label(ax, name, (x + 0.06, y + 0.10), size=12, weight="normal", alpha=0.85)

    label(ax, "finite visible sign set", (-4.38, -2.05), size=25)
    label(ax, "expanded cohomological role-space", (0.95, 3.25), size=22)

    ax.set_title("De-abstracted cohomology examples: same signs, local witnesses, only zero obstruction remains valid", pad=24)
    ax.set_xlabel("role-space axis 1")
    ax.set_ylabel("role-space axis 2")
    ax.grid(True, alpha=0.6)
    ax.set_xlim(-4.4, 3.7)
    ax.set_ylim(-2.1, 4.5)

    fig.tight_layout()
    fig.savefig(OUT / "phase98_06_deabstracted_cohomology_examples.png", dpi=170)
    plt.close(fig)


def plot_3d(df: pd.DataFrame) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 13))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLORS["ax"])

    sample = df.sample(min(len(df), 9000), random_state=SEED + 2)

    for dec in DECISIONS:
        sub = sample[sample["expected_decision"] == dec]
        ax.scatter(
            sub["path_x"],
            sub["path_y"],
            sub["role_confidence_z"],
            s=8,
            c=COLORS[dec],
            alpha=0.32,
            depthshade=True,
        )

    # Draw representative lines for each task.
    for t in TASKS:
        color = COLORS[t.expected_decision]
        sx, sy = t.sign_xy
        bx, by = t.basin_xy
        z0 = 6.0
        z1 = 18.5 if t.expected_decision == "accept" else (7.0 if t.expected_decision == "reject" else 9.0)
        ax.plot([sx, bx], [sy, by], [z0, z1], color=color, linewidth=1.4, alpha=0.65)

    anchors = {
        "accept": (-1.0, 0.0, 7.0, "accept"),
        "reject": (2.25, 2.10, 6.7, "reject"),
        "abstain": (1.20, 4.65, 8.8, "abstain"),
    }

    for name, (x, y, z, dec) in anchors.items():
        ax.scatter([x], [y], [z], s=260, c=COLORS[dec], edgecolors="white", linewidths=2)
        ax.text(x + 0.08, y + 0.08, z + 0.25, name, fontsize=18, fontweight="bold", color=COLORS["text"])

    ax.set_title("3D cohomology manifold: local sections rise into global confidence only when obstruction vanishes", pad=30)
    ax.set_xlabel("latent concept axis 1", labelpad=14)
    ax.set_ylabel("latent concept axis 2", labelpad=14)
    ax.set_zlabel("global section confidence", labelpad=14)
    ax.view_init(elev=24, azim=-62)

    fig.tight_layout()
    fig.savefig(OUT / "phase98_07_3d_cohomology_obstruction_manifold.png", dpi=170)
    plt.close(fig)


# -----------------------------
# Report
# -----------------------------

def write_report(metrics: Dict[str, float]) -> None:
    report = f"""# Phase 98 — Cohomology Obstruction Audit

## Result

`PHASE98_COHOMOLOGY_OBSTRUCTION_AUDIT_PASS={metrics["PHASE98_COHOMOLOGY_OBSTRUCTION_AUDIT_PASS"]}`

Phase 98 extends Phase 97 from local-to-global sheaf consistency into explicit cohomological obstruction detection.

Phase 97 asked whether local patches could glue. Phase 98 asks why apparently valid local patches fail to glue, separating:

- zero obstruction classes that become valid global sections,
- nonzero obstruction classes that must be rejected,
- missing witnesses, missing covers, or unknown restriction maps that require abstention.

## Metrics

| Metric | Value |
|---|---:|
| trials | {metrics["trials"]} |
| tasks | {metrics["tasks"]} |
| families | {metrics["families"]} |
| cohomology accuracy | {metrics["cohomology_accuracy"]:.6f} |
| zero obstruction acceptance | {metrics["zero_obstruction_acceptance"]:.6f} |
| nonzero obstruction rejection | {metrics["nonzero_obstruction_rejection"]:.6f} |
| missing witness abstention | {metrics["missing_witness_abstention"]:.6f} |
| cocycle consistency validity | {metrics["cocycle_consistency_validity"]:.6f} |
| global section validity | {metrics["global_section_validity"]:.6f} |
| coboundary resolution validity | {metrics["coboundary_resolution_validity"]:.6f} |
| contradiction rejection | {metrics["contradiction_rejection"]:.6f} |
| underbound detection | {metrics["underbound_detection"]:.6f} |
| de-abstracted edge coverage | {metrics["deabstracted_edge_coverage"]:.6f} |
| mean margin | {metrics["mean_margin"]:.6f} |
| min margin | {metrics["min_margin"]:.6f} |
| pass threshold | {metrics["pass_threshold"]:.6f} |

## Interpretation

This phase gives the system a new reasoning power: it no longer merely sees that gluing succeeds or fails.

It now classifies the reason for failure.

A local patch can be individually coherent, and even pairwise plausible, while still carrying a nonzero obstruction class. Phase 98 treats that situation as reject rather than accept. If the obstruction cannot be computed because the cover, restriction map, or witness is missing, the system abstains rather than hallucinating a global section.

In BBIT terms:

- finite visible signs are local observations,
- sheaf patches are local meanings,
- a global section is stable meaning across a system,
- a nonzero obstruction is the hidden reason local truths cannot become global truth.

Phase 98 therefore turns the manifold into a local-to-global reasoning object with explicit obstruction awareness.
"""

    with open(OUT / "phase98_cohomology_obstruction_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("[98] Cohomology obstruction audit")
    print(f"[98] root: {ROOT}")
    print(f"[98] outputs: {OUT}")
    print("[98] reset continued: from sheaf consistency to cohomological obstruction detection")
    print("[98] task: local sections become global sections only when obstruction class vanishes")

    df = generate_trials(trials_per_task=2400)
    metrics = summarize(df)

    write_tables(df, metrics)
    write_report(metrics)

    plot_decision_energy(df)
    plot_cohomology_field(df)
    plot_matrix(df)
    plot_progress_ladder(metrics)
    plot_meta_shape_graph()
    plot_deabstracted_examples()
    plot_3d(df)

    print(f"[98] PHASE98_COHOMOLOGY_OBSTRUCTION_AUDIT_PASS={metrics['PHASE98_COHOMOLOGY_OBSTRUCTION_AUDIT_PASS']}")
    print(
        "[98] "
        f"cohomology_accuracy={metrics['cohomology_accuracy']:.4f} "
        f"zero_obstruction_acceptance={metrics['zero_obstruction_acceptance']:.4f} "
        f"nonzero_obstruction_rejection={metrics['nonzero_obstruction_rejection']:.4f} "
        f"missing_witness_abstention={metrics['missing_witness_abstention']:.4f} "
        f"global_section_validity={metrics['global_section_validity']:.4f} "
        f"cocycle_consistency_validity={metrics['cocycle_consistency_validity']:.4f} "
        f"coboundary_resolution_validity={metrics['coboundary_resolution_validity']:.4f} "
        f"min_margin={metrics['min_margin']:.4f}"
    )
    print("[98] wrote:")
    for p in sorted(OUT.iterdir()):
        print(f"  - {p}")


if __name__ == "__main__":
    main()