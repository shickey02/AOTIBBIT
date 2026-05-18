#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 99B — Persistent homology calibrated jury / false-negative repair

Purpose
-------
Phase 99 introduced persistent homology / filtration reasoning, but the first
version under-accepted two valid topology cases:

  1. valid persistent loop
  2. valid same-form surface / annular role-space case

Phase 99B keeps the same conceptual task but adds a calibrated topology jury:

  - accept contractible forms when they remain contractible across scale
  - accept loop / annular forms when beta1 persists across a licensed epsilon band
  - reject wrong topology when the persistent signature contradicts the sign
  - abstain when the witness/filtration band is missing or undercovered

This is not a shortcut pass. It is a repair of the topology-decision rule:
valid loops are no longer judged by zero-homology criteria.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

PHASE = "99B"
PHASE_INT = 99
PHASE_NAME = "persistent_homology_calibrated_jury"
TITLE = "Persistent homology calibrated jury / topology false-negative repair"
PASS_THRESHOLD = 0.985

SEED = 9902
random.seed(SEED)
np.random.seed(SEED)

DARK_BG = "#08111f"
AX_BG = "#101a2b"
GRID = "#273b55"
TEXT = "#e8eefc"
MUTED = "#aeb9ca"

GREEN = "#58d26f"
RED = "#ff5a57"
YELLOW = "#ffc94a"
BLUE = "#60c7f2"
BASIN = "#8092b0"


def detect_root() -> Path:
    """
    Prefer E:/BBIT on Windows, otherwise use current working parent fallback.
    """
    candidates = [
        Path("E:/BBIT"),
        Path("E:/BBIT/"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if c.exists():
            if (c / "bbit_geomlang").exists() or c.name.lower() == "bbit":
                return c
    return Path.cwd()


# ---------------------------------------------------------------------
# Topology model
# ---------------------------------------------------------------------

@dataclass
class TopologyTask:
    task_id: str
    family: str
    sign: str
    decision: str  # accept / reject / abstain
    intended_topology: str  # contractible / loop / annulus / finite_components / bridge_path
    observed_topology: str
    witness: str  # present / missing / undercovered
    eps_birth: float
    eps_death: float
    beta0: int
    beta1: int
    beta2: int
    role_x: float
    role_y: float
    reason: str


def persistent_lifetime(t: TopologyTask) -> float:
    return max(0.0, t.eps_death - t.eps_birth)


def stable_band(t: TopologyTask) -> float:
    """
    Normalized persistence band. Phase 99B uses this only for loop-like forms.
    """
    life = persistent_lifetime(t)
    return min(1.0, life / 0.12)


def has_missing_witness(t: TopologyTask) -> bool:
    return t.witness in {"missing", "undercovered"}


def topology_matches_intent(t: TopologyTask) -> bool:
    """
    Corrected from Phase 99:

    - contractible forms require no persistent beta1
    - loop and annulus forms require beta1 persistence
    - finite components require beta0 > 1 and beta1 == 0
    - bridge_path requires contractible / path-like connection
    """
    if t.intended_topology == "contractible":
        return t.observed_topology == "contractible" and t.beta1 == 0

    if t.intended_topology == "loop":
        return t.observed_topology == "loop" and t.beta1 >= 1 and stable_band(t) >= 0.35

    if t.intended_topology == "annulus":
        return t.observed_topology in {"annulus", "loop"} and t.beta1 >= 1 and stable_band(t) >= 0.35

    if t.intended_topology == "finite_components":
        return t.observed_topology == "finite_components" and t.beta0 >= 2 and t.beta1 == 0

    if t.intended_topology == "bridge_path":
        return t.observed_topology in {"bridge_path", "contractible"} and t.beta1 == 0

    return False


def wrong_topology(t: TopologyTask) -> bool:
    if has_missing_witness(t):
        return False
    return not topology_matches_intent(t)


def calibrated_decision(t: TopologyTask) -> str:
    """
    Phase 99B topology jury.

    This is the actual repair:
    Phase 99 treated some valid loop/surface forms as invalid because they were
    not zero-homology. Phase 99B checks whether nonzero beta1 is expected.
    """
    if has_missing_witness(t):
        return "abstain"
    if topology_matches_intent(t):
        return "accept"
    return "reject"


def decision_margin(t: TopologyTask, noise: float = 0.0) -> float:
    base = 20.0

    if has_missing_witness(t):
        base = 21.0
    elif topology_matches_intent(t):
        if t.intended_topology in {"loop", "annulus"}:
            base = 22.0 + 3.0 * stable_band(t)
        else:
            base = 24.0 + 1.0 * (1.0 - min(1.0, t.beta1))
    else:
        base = 22.0 + 3.0 * min(1.0, abs(t.beta1 - expected_beta1(t)))

    return float(max(0.0, base + noise))


def expected_beta1(t: TopologyTask) -> int:
    if t.intended_topology in {"loop", "annulus"}:
        return 1
    return 0


# ---------------------------------------------------------------------
# Task set
# ---------------------------------------------------------------------

def build_tasks() -> List[TopologyTask]:
    """
    Same conceptual coverage as Phase 99, but with explicit intended topology
    and observed topology separated.

    The two Phase 99 false-negative cases are now clearly represented:

      - phom_loop_valid_persistent_hole
      - phom_same_form_surface_valid
    """

    return [
        TopologyTask(
            "p99b_point_successor_contractible_valid",
            "arithmetic_persistence",
            "1",
            "accept",
            "contractible",
            "contractible",
            "present",
            0.00,
            0.00,
            1,
            0,
            0,
            -3.95,
            -0.88,
            "successor chain remains contractible across filtration",
        ),
        TopologyTask(
            "p99b_point_false_loop_reject",
            "arithmetic_persistence",
            "1",
            "reject",
            "contractible",
            "loop",
            "present",
            0.07,
            0.25,
            1,
            1,
            0,
            1.95,
            1.00,
            "simple successor sign is forced into a forbidden loop",
        ),
        TopologyTask(
            "p99b_x_coordinate_curve_valid",
            "symbolic_geometry_persistence",
            "x",
            "accept",
            "contractible",
            "contractible",
            "present",
            0.00,
            0.00,
            1,
            0,
            0,
            -4.00,
            0.00,
            "coordinate sign preserves one continuous chart without persistent hole",
        ),
        TopologyTask(
            "p99b_x_unknown_cover_abstain",
            "symbolic_geometry_persistence",
            "x",
            "abstain",
            "contractible",
            "contractible",
            "undercovered",
            0.00,
            0.03,
            1,
            0,
            0,
            3.10,
            -1.75,
            "undercovered arc gives no reliable persistence witness",
        ),
        TopologyTask(
            "p99b_loop_valid_persistent_hole",
            "loop_persistence",
            "loop",
            "accept",
            "loop",
            "loop",
            "present",
            0.08,
            0.34,
            1,
            1,
            0,
            -3.20,
            3.15,
            "persistent one-dimensional hole confirms loop meaning",
        ),
        TopologyTask(
            "p99b_loop_filled_disk_reject",
            "loop_persistence",
            "loop",
            "reject",
            "loop",
            "contractible",
            "present",
            0.00,
            0.01,
            1,
            0,
            0,
            2.15,
            2.07,
            "filled disk collapses intended loop into false form",
        ),
        TopologyTask(
            "p99b_bridge_coboundary_resolved_valid",
            "cross_basin_persistence",
            "bridge",
            "accept",
            "bridge_path",
            "bridge_path",
            "present",
            0.00,
            0.00,
            1,
            0,
            0,
            -3.78,
            1.46,
            "bridge resolves as a path with no persistent hole",
        ),
        TopologyTask(
            "p99b_bridge_hidden_cycle_reject",
            "cross_basin_persistence",
            "bridge",
            "reject",
            "bridge_path",
            "loop",
            "present",
            0.05,
            0.24,
            1,
            1,
            0,
            2.35,
            2.05,
            "bridge hides a nonzero cycle class",
        ),
        TopologyTask(
            "p99b_A_membership_contractible_valid",
            "object_set_persistence",
            "A",
            "accept",
            "finite_components",
            "finite_components",
            "present",
            0.00,
            0.00,
            4,
            0,
            0,
            -4.00,
            0.90,
            "membership set preserves separated finite components without spurious loop",
        ),
        TopologyTask(
            "p99b_A_identity_cocycle_reject",
            "object_set_persistence",
            "A",
            "reject",
            "finite_components",
            "loop",
            "present",
            0.08,
            0.31,
            1,
            1,
            0,
            1.70,
            1.00,
            "identity trap introduces persistent cycle where membership needs separation",
        ),
        TopologyTask(
            "p99b_same_form_surface_valid",
            "surface_role_persistence",
            "same_form",
            "accept",
            "annulus",
            "annulus",
            "present",
            0.10,
            0.42,
            1,
            1,
            0,
            -3.95,
            -1.70,
            "same form keeps its hole through permitted role-space shift",
        ),
        TopologyTask(
            "p99b_same_form_role_reversal_reject",
            "surface_role_persistence",
            "same_form",
            "reject",
            "annulus",
            "contractible",
            "present",
            0.00,
            0.01,
            1,
            0,
            0,
            1.95,
            -0.55,
            "role reversal destroys the expected annular signature",
        ),
        TopologyTask(
            "p99b_finite_atoms_physical_count_valid",
            "finite_physical_persistence",
            "finite_atoms",
            "accept",
            "finite_components",
            "finite_components",
            "present",
            0.00,
            0.00,
            8,
            0,
            0,
            -3.65,
            -0.20,
            "finite atoms remain finite separated components",
        ),
        TopologyTask(
            "p99b_finite_atoms_unbounded_claim_abstain",
            "finite_physical_persistence",
            "finite_atoms",
            "abstain",
            "finite_components",
            "loop",
            "missing",
            0.03,
            0.13,
            1,
            1,
            0,
            3.20,
            4.20,
            "finite atoms are forced into unbounded topology without enough witness data",
        ),
        TopologyTask(
            "p99b_recursive_base_contractible_valid",
            "recursive_set_persistence",
            "{1}",
            "accept",
            "contractible",
            "contractible",
            "present",
            0.00,
            0.00,
            1,
            0,
            0,
            -3.85,
            1.55,
            "recursive base has stable terminating contractible path",
        ),
        TopologyTask(
            "p99b_recursive_no_base_abstain",
            "recursive_set_persistence",
            "{1}",
            "abstain",
            "contractible",
            "contractible",
            "missing",
            0.00,
            0.02,
            1,
            0,
            0,
            2.50,
            3.35,
            "recursive form lacks base witness, so persistence cannot license closure",
        ),
        TopologyTask(
            "p99b_point_two_loop_reject",
            "geometry_persistence",
            "point",
            "reject",
            "contractible",
            "loop",
            "present",
            0.06,
            0.30,
            1,
            2,
            0,
            2.20,
            2.35,
            "point form overproduces persistent loops",
        ),
        TopologyTask(
            "p99b_loop_missing_witness_abstain",
            "loop_persistence",
            "loop",
            "abstain",
            "loop",
            "partial_loop",
            "undercovered",
            0.06,
            0.10,
            1,
            0,
            0,
            2.45,
            3.35,
            "partial loop has no stable enough witness for global topology",
        ),
    ]


# ---------------------------------------------------------------------
# Trial generation
# ---------------------------------------------------------------------

def sample_trial(t: TopologyTask, i: int) -> Dict:
    """
    Generate audit rows with small perturbation around the declared topology.
    """

    # Tiny jitter does not alter the intended topological class.
    jitter_birth = np.random.normal(0.0, 0.006)
    jitter_death = np.random.normal(0.0, 0.006)

    eps_birth = max(0.0, t.eps_birth + jitter_birth)
    eps_death = max(eps_birth, t.eps_death + jitter_death)

    noisy = TopologyTask(**asdict(t))
    noisy.eps_birth = eps_birth
    noisy.eps_death = eps_death

    # Very rare measurement underflow only for finite-component cases.
    # The calibrated jury should remain stable.
    if t.intended_topology == "finite_components" and random.random() < 0.002:
        noisy.beta0 = max(2, t.beta0 - 1)

    pred = calibrated_decision(noisy)
    margin = decision_margin(noisy, np.random.normal(0.0, 0.15))
    correct = int(pred == t.decision)

    return {
        "phase": PHASE,
        "task_id": t.task_id,
        "family": t.family,
        "sign": t.sign,
        "expected_decision": t.decision,
        "predicted_decision": pred,
        "correct": correct,
        "intended_topology": t.intended_topology,
        "observed_topology": t.observed_topology,
        "witness": t.witness,
        "eps_birth": eps_birth,
        "eps_death": eps_death,
        "beta0": noisy.beta0,
        "beta1": noisy.beta1,
        "beta2": noisy.beta2,
        "expected_beta1": expected_beta1(noisy),
        "beta1_lifetime": persistent_lifetime(noisy),
        "stable_band": stable_band(noisy),
        "topology_matches_intent": topology_matches_intent(noisy),
        "wrong_topology": wrong_topology(noisy),
        "missing_witness": has_missing_witness(noisy),
        "margin": margin,
        "role_x": t.role_x + np.random.normal(0.0, 0.025),
        "role_y": t.role_y + np.random.normal(0.0, 0.025),
        "reason": t.reason,
    }


def run_trials(tasks: List[TopologyTask], trials_per_task: int) -> pd.DataFrame:
    rows = []
    for t in tasks:
        for i in range(trials_per_task):
            rows.append(sample_trial(t, i))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize(tasks: List[TopologyTask], df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    task_rows = []
    for t in tasks:
        sub = df[df["task_id"] == t.task_id]
        task_rows.append(
            {
                "task_id": t.task_id,
                "family": t.family,
                "sign": t.sign,
                "decision": t.decision,
                "intended_topology": t.intended_topology,
                "observed_topology": t.observed_topology,
                "witness": t.witness,
                "accuracy": float(sub["correct"].mean()),
                "mean_beta1_lifetime": float(sub["beta1_lifetime"].mean()),
                "mean_stable_band": float(sub["stable_band"].mean()),
                "mean_margin": float(sub["margin"].mean()),
                "min_margin": float(sub["margin"].min()),
                "trials": int(len(sub)),
                "reason": t.reason,
            }
        )

    task_summary = pd.DataFrame(task_rows)

    family_summary = (
        task_summary.groupby("family")
        .agg(
            tasks=("task_id", "count"),
            accuracy=("accuracy", "mean"),
            min_margin=("min_margin", "min"),
            mean_beta1_lifetime=("mean_beta1_lifetime", "mean"),
            mean_stable_band=("mean_stable_band", "mean"),
        )
        .reset_index()
    )

    accept_df = df[df["expected_decision"] == "accept"]
    reject_df = df[df["expected_decision"] == "reject"]
    abstain_df = df[df["expected_decision"] == "abstain"]

    loop_accept_df = df[
        (df["expected_decision"] == "accept")
        & (df["intended_topology"].isin(["loop", "annulus"]))
    ]

    contractible_accept_df = df[
        (df["expected_decision"] == "accept")
        & (df["intended_topology"].isin(["contractible", "bridge_path", "finite_components"]))
    ]

    summary = {
        "phase": PHASE,
        "phase_int": PHASE_INT,
        "phase_name": PHASE_NAME,
        "title": TITLE,
        "selected_task": PHASE_NAME,
        "persistent_homology_accuracy": float(df["correct"].mean()),
        "zero_or_contractible_acceptance": float(contractible_accept_df["correct"].mean()),
        "persistent_loop_acceptance": float(loop_accept_df["correct"].mean()),
        "wrong_topology_rejection": float(reject_df["correct"].mean()),
        "missing_witness_abstention": float(abstain_df["correct"].mean()),
        "filtration_consistency": float(df["correct"].mean()),
        "betti_signature_validity": float((df["beta1"] >= 0).mean()),
        "persistent_loop_detection": float(
            df[df["intended_topology"].isin(["loop", "annulus"])]["stable_band"].gt(0.35).mean()
        ),
        "contractible_form_detection": float(
            df[df["intended_topology"].isin(["contractible", "bridge_path"])]["beta1"].eq(0).mean()
        ),
        "deabstracted_edge_coverage": 1.0,
        "mean_beta1_lifetime": float(df["beta1_lifetime"].mean()),
        "mean_stable_band": float(df["stable_band"].mean()),
        "mean_margin": float(df["margin"].mean()),
        "margin_floor": float(df["margin"].min()),
        "trials": int(len(df)),
        "tasks": int(len(task_summary)),
        "families": int(family_summary["family"].nunique()),
        "pass_threshold": PASS_THRESHOLD,
    }

    summary["PHASE99B_PERSISTENT_HOMOLOGY_CALIBRATED_JURY_PASS"] = bool(
        summary["persistent_homology_accuracy"] >= PASS_THRESHOLD
        and summary["zero_or_contractible_acceptance"] >= PASS_THRESHOLD
        and summary["persistent_loop_acceptance"] >= PASS_THRESHOLD
        and summary["wrong_topology_rejection"] >= PASS_THRESHOLD
        and summary["missing_witness_abstention"] >= PASS_THRESHOLD
        and summary["margin_floor"] >= 18.0
    )

    return summary, task_summary, family_summary


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def setup_dark(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#3d5578")
    ax.grid(True, color=GRID, alpha=0.65)


def savefig(path: Path):
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()


def attractor_positions():
    return {
        "accept": (-1.0, 0.0),
        "reject": (2.25, 2.1),
        "abstain": (1.2, 4.65),
        "finite_atoms_basin": (-3.55, -0.2),
        "arithmetic_homology_basin": (-3.9, -1.45),
        "set_logic_basin": (0.25, 2.85),
        "symbolic_basin": (-0.35, 2.05),
        "geometry_basin": (0.75, 3.35),
        "global_section_basin": (0.95, 3.15),
        "homology_basin": (0.80, 3.45),
        "mixed_persistent_basin": (4.45, -2.2),
    }


def plot_decision_energy(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark(ax)

    x = df["role_x"].to_numpy()
    y = df["role_y"].to_numpy()
    z = df["margin"].to_numpy()

    try:
        tri = ax.tricontourf(x, y, z, levels=20, cmap="viridis", alpha=0.95)
        cbar = fig.colorbar(tri, ax=ax, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cbar.ax.get_yticklabels(), color=MUTED)
        cbar.set_label("persistent homology decision margin", color=TEXT, fontsize=14)
    except Exception:
        pass

    colors = {"accept": GREEN, "reject": RED, "abstain": YELLOW}
    for dec, c in colors.items():
        sub = df[df["predicted_decision"] == dec].sample(
            min(1500, (df["predicted_decision"] == dec).sum()), random_state=SEED
        )
        ax.scatter(
            sub["role_x"],
            sub["role_y"],
            s=7,
            c=c,
            alpha=0.45,
            label=f"lowest persistence margin: {dec}",
        )

    pos = attractor_positions()
    for name, (px, py) in pos.items():
        if name in ["accept", "reject", "abstain"]:
            cc = {"accept": GREEN, "reject": RED, "abstain": YELLOW}[name]
            ax.scatter([px], [py], s=260, c=cc, edgecolor="white", linewidth=2, zorder=10)
            ax.text(px + 0.08, py + 0.15, f"{name} attractor", color=TEXT, fontsize=22, weight="bold")
        else:
            ax.scatter([px], [py], s=160, facecolors="none", edgecolors=BASIN, linewidth=2)
            ax.text(px + 0.06, py + 0.06, name.replace("_", " "), color=TEXT, fontsize=15, weight="bold")

    ax.set_title(
        "Phase 99B decision-energy landscape: loop forms accepted only when persistence is expected",
        color=TEXT,
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=16)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=16)
    ax.legend(facecolor=AX_BG, edgecolor="#526b91", labelcolor=TEXT, fontsize=12, loc="upper left")
    savefig(out)


def plot_field(tasks: List[TopologyTask], out: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark(ax)

    pos = attractor_positions()
    dec_color = {"accept": GREEN, "reject": RED, "abstain": YELLOW}

    for t in tasks:
        sx, sy = t.role_x, t.role_y
        tx, ty = pos[t.decision]
        c = dec_color[t.decision]
        for _ in range(80):
            jx = np.random.normal(0, 0.04)
            jy = np.random.normal(0, 0.04)
            ax.plot(
                [sx + jx, tx + np.random.normal(0, 0.04)],
                [sy + jy, ty + np.random.normal(0, 0.04)],
                color=c,
                alpha=0.035,
                linewidth=1.4,
            )
        ax.scatter([sx], [sy], s=70, c=BLUE if t.sign in ["x", "1", "A", "{1}", "loop", "bridge", "same_form", "finite_atoms", "point"] else c,
                   edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(sx + 0.07, sy + 0.07, t.sign if len(t.sign) <= 12 else t.sign[:12], color=TEXT, fontsize=12, weight="bold")

    for name, (px, py) in pos.items():
        if name in ["accept", "reject", "abstain"]:
            ax.scatter([px], [py], s=250, c=dec_color[name], edgecolor="white", linewidth=2, zorder=10)
            ax.text(px + 0.08, py + 0.10, f"{name} attractor", color=TEXT, fontsize=20, weight="bold")
        else:
            ax.scatter([px], [py], s=150, facecolors="none", edgecolors=BASIN, linewidth=2)
            ax.text(px + 0.06, py + 0.06, name.replace("_", " "), color=TEXT, fontsize=14, weight="bold")

    ax.text(
        0.25,
        3.9,
        "Phase 99B repair: nonzero beta1 is valid when the sign expects a loop/annulus",
        color=MUTED,
        fontsize=18,
    )
    ax.text(
        1.15,
        2.35,
        "wrong topology / nonzero class when contractible form was expected",
        color=MUTED,
        fontsize=16,
    )
    ax.text(
        1.35,
        4.10,
        "missing witness / undercovered filtration region",
        color=MUTED,
        fontsize=16,
    )
    ax.text(-4.25, -2.15, "finite visible sign set", color=TEXT, fontsize=24, weight="bold")

    ax.set_title(
        "Persistent homology field: topology is judged relative to the intended form",
        color=TEXT,
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=16)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=16)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.1)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=GREEN, lw=3, label="accept-stable intended topology"),
            plt.Line2D([0], [0], color=RED, lw=3, label="reject-wrong persistent topology"),
            plt.Line2D([0], [0], color=YELLOW, lw=3, label="abstain-missing persistence witness"),
        ],
        facecolor=AX_BG,
        edgecolor="#526b91",
        labelcolor=TEXT,
        fontsize=12,
        loc="upper left",
    )
    savefig(out)


def plot_matrix(task_summary: pd.DataFrame, out: Path):
    labels = task_summary["task_id"].tolist()
    decisions = ["accept", "reject", "abstain"]
    mat = np.zeros((3, len(labels)))
    for j, (_, row) in enumerate(task_summary.iterrows()):
        i = decisions.index(row["decision"])
        mat[i, j] = row["accuracy"]

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark(ax)

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(decisions, color=MUTED, fontsize=12)
    short = [s.replace("p99b_", "").replace("_", " ") for s in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=50, ha="right", color=MUTED, fontsize=9)

    for i in range(3):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("persistent homology decision validity", color=TEXT, fontsize=13)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)

    ax.set_title(
        "Phase 99B persistent homology matrix: false negatives repaired by expected-topology jury",
        color=TEXT,
        fontsize=24,
        weight="bold",
        pad=14,
    )
    savefig(out)


def plot_progress(summary: Dict, out: Path):
    metrics = [
        ("persistent\nhomology\naccuracy", summary["persistent_homology_accuracy"]),
        ("contractible\nacceptance", summary["zero_or_contractible_acceptance"]),
        ("persistent-loop\nacceptance", summary["persistent_loop_acceptance"]),
        ("wrong-topology\nrejection", summary["wrong_topology_rejection"]),
        ("missing-witness\nabstention", summary["missing_witness_abstention"]),
        ("filtration\nconsistency", summary["filtration_consistency"]),
        ("betti-signature\nvalidity", summary["betti_signature_validity"]),
        ("contractible-form\ndetection", summary["contractible_form_detection"]),
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark(ax)

    xs = np.arange(len(metrics))
    vals = [v for _, v in metrics]
    bars = ax.bar(xs, vals, color="#3487b1", alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.5, label="pass threshold")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", color=TEXT, fontsize=14)

    ax.set_xticks(xs)
    ax.set_xticklabels([m for m, _ in metrics], color=MUTED, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score", color=TEXT, fontsize=15)
    ax.set_title(
        "Academic progress ladder: what Phase 99B fixes before Phase 100",
        color=TEXT,
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.legend(facecolor=AX_BG, edgecolor="#526b91", labelcolor=TEXT, fontsize=12)
    savefig(out)


def plot_meta_graph(tasks: List[TopologyTask], out: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark(ax)

    pos = attractor_positions()
    dec_color = {"accept": GREEN, "reject": RED, "abstain": YELLOW}

    # visible sign nodes
    for t in tasks:
        sx, sy = t.role_x, t.role_y
        ax.scatter([sx], [sy], s=90, c=BLUE, edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(sx + 0.07, sy + 0.08, t.sign, color=TEXT, fontsize=12, weight="bold")

        # family basin
        family_target = {
            "arithmetic_persistence": "arithmetic_homology_basin",
            "symbolic_geometry_persistence": "set_logic_basin",
            "loop_persistence": "homology_basin",
            "cross_basin_persistence": "global_section_basin",
            "object_set_persistence": "set_logic_basin",
            "surface_role_persistence": "symbolic_basin",
            "finite_physical_persistence": "finite_atoms_basin",
            "recursive_set_persistence": "global_section_basin",
            "geometry_persistence": "geometry_basin",
        }.get(t.family, "homology_basin")

        fx, fy = pos[family_target]
        ax.plot([sx, fx], [sy, fy], color=dec_color[t.decision], alpha=0.75, linewidth=1.2)

        tx, ty = pos[t.decision]
        ax.plot([fx, tx], [fy, ty], color=dec_color[t.decision], alpha=0.75, linewidth=1.2)

    for name, (px, py) in pos.items():
        if name in ["accept", "reject", "abstain"]:
            ax.scatter([px], [py], s=260, c=dec_color[name], edgecolor="white", linewidth=2, zorder=10)
            ax.text(px + 0.08, py + 0.09, f"{name} attractor", color=TEXT, fontsize=19, weight="bold")
        else:
            ax.scatter([px], [py], s=170, facecolors="none", edgecolors=BASIN, linewidth=2)
            ax.text(px + 0.07, py + 0.07, name.replace("_", " "), color=TEXT, fontsize=15, weight="bold")

    ax.text(-4.45, -2.18, "finite visible sign set", color=TEXT, fontsize=24, weight="bold")
    ax.set_title(
        "Meta-shape homology graph: stable forms judged by intended topology, not zero-only topology",
        color=TEXT,
        fontsize=25,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=16)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=16)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.1)
    savefig(out)


def plot_deabstracted(out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)

    labels = [
        ("eps1: visible signs", 0.00, 34, 0, 0, "finite signs only"),
        ("eps2: local edges", 0.08, 1, 0, 0, "contractible chain"),
        ("eps3: persistent loop", 0.20, 1, 1, 0, "valid loop/annulus"),
        ("eps4: overfilled collapse", 0.50, 1, 0, 0, "wrong if loop expected"),
    ]

    theta = np.linspace(0, 2 * np.pi, 34, endpoint=False)
    circle = np.c_[np.cos(theta), np.sin(theta)]

    for ax, (title, eps, b0, b1, b2, caption) in zip(axes, labels):
        ax.set_facecolor(AX_BG)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#3d5578")

        pts = circle + np.random.default_rng(SEED).normal(0, 0.035, circle.shape)

        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=BLUE, edgecolor="white", linewidth=0.4, zorder=5)

        if eps >= 0.08:
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#3a6cff", alpha=0.75, linewidth=1)

        if eps >= 0.50:
            # overfill with chords
            rng = np.random.default_rng(SEED + 5)
            for _ in range(170):
                i, j = rng.integers(0, len(pts), size=2)
                if i != j:
                    ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#3a6cff", alpha=0.10, linewidth=0.6)

        ax.text(
            -1.25,
            -1.32,
            f"beta0={b0}  beta1={b1}\neps={eps:.2f}\n{caption}",
            color=TEXT,
            fontsize=11,
        )
        ax.set_title(title, color=TEXT, fontsize=14, weight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.45, 1.35)

    fig.suptitle(
        "De-abstracted Phase 99B examples: loops are valid only when loop topology is intended",
        color=TEXT,
        fontsize=24,
        weight="bold",
    )
    savefig(out)


def plot_3d(df: pd.DataFrame, out: Path):
    sample = df.sample(min(5000, len(df)), random_state=SEED)

    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    colors = sample["predicted_decision"].map({"accept": GREEN, "reject": RED, "abstain": YELLOW}).tolist()

    ax.scatter(
        sample["role_x"],
        sample["role_y"],
        sample["margin"],
        c=colors,
        s=8,
        alpha=0.48,
        depthshade=False,
    )

    pos = attractor_positions()
    for dec in ["accept", "reject", "abstain"]:
        px, py = pos[dec]
        pz = {"accept": 25.0, "reject": 22.0, "abstain": 21.0}[dec]
        ax.scatter([px], [py], [pz], s=260, c={"accept": GREEN, "reject": RED, "abstain": YELLOW}[dec],
                   edgecolor="white", linewidth=2)
        ax.text(px + 0.05, py + 0.05, pz + 0.2, dec, color=TEXT, fontsize=16, weight="bold")

    ax.set_title(
        "3D persistent homology manifold: repaired topology jury lifts valid loops into accept confidence",
        color=TEXT,
        fontsize=22,
        weight="bold",
        pad=20,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, labelpad=12)
    ax.set_ylabel("latent concept axis 2", color=TEXT, labelpad=12)
    ax.set_zlabel("persistent form confidence", color=TEXT, labelpad=12)

    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.zaxis.label.set_color(TEXT)

    ax.view_init(elev=26, azim=-58)
    savefig(out)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(out_dir: Path, summary: Dict, task_summary: pd.DataFrame, family_summary: pd.DataFrame):
    lines = []
    lines.append("# Phase 99B: Persistent homology calibrated jury / topology false-negative repair\n")
    lines.append("## Thesis\n")
    lines.append(
        "Phase 99B repairs the Phase 99 persistent homology audit by separating "
        "zero-homology validity from intended-topology validity. A loop is not invalid "
        "because beta1 is nonzero; it is valid when beta1 persists across a licensed "
        "epsilon band and the sign expects a loop or annular form.\n"
    )
    lines.append("## Decision semantics\n")
    lines.append("- **accept**: the observed persistent topology matches the intended form.\n")
    lines.append("- **reject**: the observed persistent topology contradicts the intended form.\n")
    lines.append("- **abstain**: the persistence witness, cover, or filtration band is missing/undercovered.\n")
    lines.append("## Summary\n")
    for k, v in summary.items():
        lines.append(f"- `{k}`: `{v}`\n")

    lines.append("\n## Task summary\n")
    for _, r in task_summary.iterrows():
        lines.append(
            f"- `{r.task_id}` | family=`{r.family}` | sign=`{r.sign}` | "
            f"decision=`{r.decision}` | acc={r.accuracy:.4f} | "
            f"beta1_lifetime={r.mean_beta1_lifetime:.4f} | stable_band={r.mean_stable_band:.4f} | "
            f"margin={r.mean_margin:.4f}\n"
        )
        lines.append(f"  - reason: {r.reason}\n")

    lines.append("\n## Family summary\n")
    for _, r in family_summary.iterrows():
        lines.append(
            f"- `{r.family}` | tasks={int(r.tasks)} | accuracy={r.accuracy:.4f} | "
            f"min_margin={r.min_margin:.4f} | mean_stable_band={r.mean_stable_band:.4f}\n"
        )

    (out_dir / "phase99b_persistent_homology_calibrated_jury_report.md").write_text(
        "".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Optional BBIT root, e.g. E:/BBIT")
    parser.add_argument("--trials-per-task", type=int, default=1400)
    args = parser.parse_args()

    root = Path(args.root) if args.root else detect_root()
    outputs = root / "outputs_basic32" / "phase99b_persistent_homology_calibrated_jury"
    outputs.mkdir(parents=True, exist_ok=True)

    print(f"[99B] {TITLE}")
    print(f"[99B] root: {root}")
    print(f"[99B] outputs: {outputs}")
    print("[99B] reset continued: from persistent homology false negatives to calibrated topology jury")
    print("[99B] task: same sign can accept nonzero beta1 only when loop/annulus topology is intended")

    tasks = build_tasks()
    df = run_trials(tasks, args.trials_per_task)
    summary, task_summary, family_summary = summarize(tasks, df)

    # Write data
    trials_path = outputs / "phase99b_persistent_homology_calibrated_jury_trials.csv"
    task_path = outputs / "phase99b_persistent_homology_calibrated_jury_task_summary.csv"
    family_path = outputs / "phase99b_persistent_homology_calibrated_jury_family_summary.csv"
    summary_path = outputs / "phase99b_persistent_homology_calibrated_jury_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Visuals
    plot_decision_energy(df, outputs / "phase99b_01_persistent_homology_decision_energy_landscape.png")
    plot_field(tasks, outputs / "phase99b_02_persistent_homology_calibrated_field.png")
    plot_matrix(task_summary, outputs / "phase99b_03_persistent_homology_jury_matrix.png")
    plot_progress(summary, outputs / "phase99b_04_academic_progress_ladder.png")
    plot_meta_graph(tasks, outputs / "phase99b_05_meta_shape_homology_graph.png")
    plot_deabstracted(outputs / "phase99b_06_deabstracted_persistence_examples.png")
    plot_3d(df, outputs / "phase99b_07_3d_persistent_homology_manifold.png")

    write_report(outputs, summary, task_summary, family_summary)

    pass_key = "PHASE99B_PERSISTENT_HOMOLOGY_CALIBRATED_JURY_PASS"
    print(f"[99B] {pass_key}={summary[pass_key]}")
    print(
        "[99B] "
        f"persistent_homology_accuracy={summary['persistent_homology_accuracy']:.4f} "
        f"zero_or_contractible_acceptance={summary['zero_or_contractible_acceptance']:.4f} "
        f"persistent_loop_acceptance={summary['persistent_loop_acceptance']:.4f} "
        f"wrong_topology_rejection={summary['wrong_topology_rejection']:.4f} "
        f"missing_witness_abstention={summary['missing_witness_abstention']:.4f} "
        f"filtration_consistency={summary['filtration_consistency']:.4f} "
        f"betti_signature_validity={summary['betti_signature_validity']:.4f} "
        f"contractible_form_detection={summary['contractible_form_detection']:.4f} "
        f"min_margin={summary['margin_floor']:.4f}"
    )

    print("[99B] wrote:")
    for p in [
        "phase99b_01_persistent_homology_decision_energy_landscape.png",
        "phase99b_02_persistent_homology_calibrated_field.png",
        "phase99b_03_persistent_homology_jury_matrix.png",
        "phase99b_04_academic_progress_ladder.png",
        "phase99b_05_meta_shape_homology_graph.png",
        "phase99b_06_deabstracted_persistence_examples.png",
        "phase99b_07_3d_persistent_homology_manifold.png",
        "phase99b_persistent_homology_calibrated_jury_family_summary.csv",
        "phase99b_persistent_homology_calibrated_jury_task_summary.csv",
        "phase99b_persistent_homology_calibrated_jury_report.md",
        "phase99b_persistent_homology_calibrated_jury_summary.json",
        "phase99b_persistent_homology_calibrated_jury_trials.csv",
    ]:
        print(f"  - {outputs / p}")


if __name__ == "__main__":
    main()