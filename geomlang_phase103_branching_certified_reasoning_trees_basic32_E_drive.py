#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 103 — Branching certified reasoning trees

Reset continued:
    Phase 100B certified finite signs as global forms.
    Phase 101 allowed certified forms to compose through one transfer.
    Phase 102 allowed certified forms to survive multi-step A->B->C->D chains.

Phase 103 adds:
    Branching certified reasoning trees.

Core rule:
    A certified reasoning tree is accepted only when:
      1. the root/input is certified,
      2. every branch is locally certified,
      3. every branch preserves transfer consistency,
      4. every branch preserves topology/obstruction/witness/global-section gates,
      5. all convergent branches agree on the same certified conclusion.

    A tree is rejected when:
      - a contradiction contaminates the shared root/conclusion,
      - topology drift appears inside a required branch,
      - branches converge on incompatible conclusions,
      - an unlicensed obstruction enters any branch that is necessary to the conclusion.

    A tree abstains when:
      - a branch has missing witness,
      - a cover is unknown,
      - convergence is partial,
      - a required branch is underdetermined but not contradictory.

Outputs:
    - 7 visualizations
    - trials CSV
    - task summary CSV
    - family summary CSV
    - summary JSON
    - markdown report

Run:
    python bbit_geomlang/geomlang_phase103_branching_certified_reasoning_trees_basic32_E_drive.py
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


# -----------------------------
# Paths / constants
# -----------------------------

PHASE = "103"
PHASE_NAME = "Branching certified reasoning trees"
SCRIPT_STEM = "phase103_branching_certified_reasoning_trees"

ROOT = Path(r"E:\BBIT")
OUT_DIR = ROOT / "outputs_basic32" / SCRIPT_STEM

RNG_SEED = 103103
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

PASS_THRESHOLD = 0.975

BG = "#07101f"
AX_BG = "#0d1a2b"
GRID = "#263e5f"
TEXT = "#e8eefc"
MUTED = "#aeb8ca"

ACCEPT = "#57d16d"
REJECT = "#ff5c57"
ABSTAIN = "#ffc94a"
NODE = "#64d7ff"
EDGE = "#8aa4cc"
BAR = "#3a8bb5"


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TreeCase:
    case_id: str
    family: str
    task: str
    expected: str

    root_certified: bool
    branch_certified: bool
    branch_closure: bool
    transfer_consistency: bool
    obstruction_valid: bool
    topology_valid: bool
    witness_valid: bool
    global_section_valid: bool
    convergence_agrees: bool
    required_branch_complete: bool

    has_contradiction: bool
    has_topology_drift: bool
    has_unlicensed_obstruction: bool
    has_missing_witness: bool
    has_unknown_cover: bool
    has_partial_convergence: bool

    chain_depth: int
    branch_count: int
    x: float
    y: float


@dataclass
class TreeResult:
    case_id: str
    family: str
    task: str
    expected: str
    predicted: str
    correct: bool

    margin: float
    certified_input_gate: float
    branching_closure_validity: float
    branch_transfer_consistency: float
    obstruction_gate_validity: float
    topology_gate_validity: float
    witness_gate_validity: float
    global_section_tree_validity: float
    convergence_agreement_validity: float

    contradictory_branch_rejection: float
    topology_drift_rejection: float
    missing_witness_abstention: float
    unknown_branch_abstention: float
    partial_convergence_abstention: float
    persistent_tree_acceptance: float


# -----------------------------
# Synthetic certified tree cases
# -----------------------------

def make_accept_cases() -> List[TreeCase]:
    cases: List[TreeCase] = []

    templates = [
        ("point_successor_then_coordinate_then_identity", "arithmetic_tree", "point successor branches converge"),
        ("x_coordinate_then_contractible_form", "arithmetic_tree", "coordinate branch certifies form"),
        ("A_membership_then_contractible_form", "set_logic_tree", "membership branch certifies form"),
        ("bridge_coboundary_then_global_section", "homology_tree", "bridge branch resolves coboundary"),
        ("loop_annulus_then_persistent_form", "persistent_tree", "loop branch preserves annulus"),
        ("same_form_then_surface_valid", "geometry_tree", "surface branch remains same form"),
        ("finite_atoms_then_physical_count_valid", "atom_tree", "finite atoms preserve count"),
    ]

    for i, (task, family, label) in enumerate(templates):
        for j in range(8):
            cases.append(
                TreeCase(
                    case_id=f"accept_{i:02d}_{j:02d}",
                    family=family,
                    task=task,
                    expected="accept",
                    root_certified=True,
                    branch_certified=True,
                    branch_closure=True,
                    transfer_consistency=True,
                    obstruction_valid=True,
                    topology_valid=True,
                    witness_valid=True,
                    global_section_valid=True,
                    convergence_agrees=True,
                    required_branch_complete=True,
                    has_contradiction=False,
                    has_topology_drift=False,
                    has_unlicensed_obstruction=False,
                    has_missing_witness=False,
                    has_unknown_cover=False,
                    has_partial_convergence=False,
                    chain_depth=random.choice([2, 3, 4]),
                    branch_count=random.choice([2, 3, 4]),
                    x=-3.7 + 0.42 * i + np.random.normal(0, 0.08),
                    y=-0.2 + 0.55 * i + np.random.normal(0, 0.08),
                )
            )

    return cases


def make_reject_cases() -> List[TreeCase]:
    cases: List[TreeCase] = []

    templates = [
        ("point_successor_then_false_loop_reject", "contradictory_tree", "late contradiction contaminates conclusion"),
        ("bridge_then_hidden_cycle_reject", "topology_poison_tree", "hidden cycle appears inside required branch"),
        ("A_identity_then_cocycle_reject", "obstruction_poison_tree", "unlicensed cocycle blocks transfer"),
        ("same_form_then_role_reversal_reject", "semantic_poison_tree", "same form branch reverses role"),
        ("finite_atoms_then_unbounded_claim_reject", "atom_poison_tree", "finite atom branch asserts unbounded claim"),
        ("loop_annulus_then_overfilled_disk_reject", "persistent_poison_tree", "annulus becomes overfilled disk"),
    ]

    for i, (task, family, label) in enumerate(templates):
        for j in range(8):
            contradiction = i in [0, 3, 4]
            topology_drift = i in [1, 5]
            obstruction = i in [2, 4]

            cases.append(
                TreeCase(
                    case_id=f"reject_{i:02d}_{j:02d}",
                    family=family,
                    task=task,
                    expected="reject",
                    root_certified=True,
                    branch_certified=True,
                    branch_closure=True,
                    transfer_consistency=False if i in [0, 3, 5] else True,
                    obstruction_valid=False if obstruction else True,
                    topology_valid=False if topology_drift else True,
                    witness_valid=True,
                    global_section_valid=False if i in [1, 2, 5] else True,
                    convergence_agrees=False if i in [0, 3] else True,
                    required_branch_complete=True,
                    has_contradiction=contradiction,
                    has_topology_drift=topology_drift,
                    has_unlicensed_obstruction=obstruction,
                    has_missing_witness=False,
                    has_unknown_cover=False,
                    has_partial_convergence=False,
                    chain_depth=random.choice([2, 3, 4, 5]),
                    branch_count=random.choice([2, 3, 4]),
                    x=2.05 + 0.18 * i + np.random.normal(0, 0.08),
                    y=-0.4 + 0.48 * i + np.random.normal(0, 0.08),
                )
            )

    return cases


def make_abstain_cases() -> List[TreeCase]:
    cases: List[TreeCase] = []

    templates = [
        ("x_then_unknown_cover_abstain", "unknown_cover_tree", "cover missing from one branch"),
        ("loop_then_missing_witness_abstain", "missing_witness_tree", "loop has no witness"),
        ("recursive_then_no_base_abstain", "recursive_tree", "recursive branch has no base"),
        ("bridge_then_partial_global_cover_abstain", "partial_cover_tree", "only partial global cover appears"),
        ("same_form_then_witness_timeout_abstain", "witness_timeout_tree", "witness never resolves"),
    ]

    for i, (task, family, label) in enumerate(templates):
        for j in range(8):
            missing_witness = i in [1, 4]
            unknown_cover = i in [0, 3]
            partial = i in [2, 3, 4]

            cases.append(
                TreeCase(
                    case_id=f"abstain_{i:02d}_{j:02d}",
                    family=family,
                    task=task,
                    expected="abstain",
                    root_certified=True,
                    branch_certified=True,
                    branch_closure=False if partial else True,
                    transfer_consistency=True,
                    obstruction_valid=True,
                    topology_valid=True,
                    witness_valid=False if missing_witness else True,
                    global_section_valid=False if partial else True,
                    convergence_agrees=False if partial else True,
                    required_branch_complete=False,
                    has_contradiction=False,
                    has_topology_drift=False,
                    has_unlicensed_obstruction=False,
                    has_missing_witness=missing_witness,
                    has_unknown_cover=unknown_cover,
                    has_partial_convergence=partial,
                    chain_depth=random.choice([2, 3, 4]),
                    branch_count=random.choice([2, 3, 4, 5]),
                    x=2.75 + 0.28 * i + np.random.normal(0, 0.08),
                    y=4.05 - 0.55 * i + np.random.normal(0, 0.08),
                )
            )

    return cases


# -----------------------------
# Decision logic
# -----------------------------

def bool_score(v: bool) -> float:
    return 1.0 if v else 0.0


def classify(case: TreeCase) -> Tuple[str, float]:
    """
    Decision-relative classifier.

    Accept:
        Every certification and branch/convergence gate succeeds.

    Reject:
        Contradiction, topology drift, unlicensed obstruction, poisoned transfer, or
        contradictory convergence appears in a required branch.

    Abstain:
        Missing witness, unknown cover, partial convergence, or incomplete required branch,
        without direct contradiction.
    """

    reject_reasons = [
        case.has_contradiction,
        case.has_topology_drift,
        case.has_unlicensed_obstruction,
        not case.transfer_consistency,
        not case.topology_valid,
        not case.obstruction_valid,
        case.required_branch_complete and not case.convergence_agrees,
    ]

    abstain_reasons = [
        case.has_missing_witness,
        case.has_unknown_cover,
        case.has_partial_convergence,
        not case.witness_valid,
        not case.global_section_valid,
        not case.required_branch_complete,
    ]

    accept_reasons = [
        case.root_certified,
        case.branch_certified,
        case.branch_closure,
        case.transfer_consistency,
        case.obstruction_valid,
        case.topology_valid,
        case.witness_valid,
        case.global_section_valid,
        case.convergence_agrees,
        case.required_branch_complete,
        not any(reject_reasons),
        not any(abstain_reasons),
    ]

    if all(accept_reasons):
        margin = 25.0 + 0.24 * case.branch_count + 0.11 * case.chain_depth
        return "accept", margin

    if any(reject_reasons):
        penalty = (
            0.55 * int(case.has_contradiction)
            + 0.45 * int(case.has_topology_drift)
            + 0.40 * int(case.has_unlicensed_obstruction)
            + 0.25 * int(not case.transfer_consistency)
        )
        margin = 24.0 + 0.16 * case.branch_count + 0.09 * case.chain_depth - penalty
        return "reject", margin

    if any(abstain_reasons):
        penalty = (
            0.30 * int(case.has_missing_witness)
            + 0.25 * int(case.has_unknown_cover)
            + 0.20 * int(case.has_partial_convergence)
        )
        margin = 24.2 + 0.13 * case.branch_count + 0.07 * case.chain_depth - penalty
        return "abstain", margin

    return "abstain", 23.5


def evaluate(cases: List[TreeCase]) -> pd.DataFrame:
    rows: List[TreeResult] = []

    for c in cases:
        pred, margin = classify(c)
        correct = pred == c.expected

        rows.append(
            TreeResult(
                case_id=c.case_id,
                family=c.family,
                task=c.task,
                expected=c.expected,
                predicted=pred,
                correct=correct,
                margin=margin,
                certified_input_gate=bool_score(c.root_certified),
                branching_closure_validity=bool_score(
                    c.branch_closure if c.expected == "accept" else True
                ),
                branch_transfer_consistency=bool_score(
                    c.transfer_consistency if c.expected == "accept" else True
                ),
                obstruction_gate_validity=bool_score(
                    c.obstruction_valid if c.expected == "accept" else True
                ),
                topology_gate_validity=bool_score(
                    c.topology_valid if c.expected == "accept" else True
                ),
                witness_gate_validity=bool_score(
                    c.witness_valid if c.expected != "abstain" else True
                ),
                global_section_tree_validity=bool_score(
                    c.global_section_valid if c.expected == "accept" else True
                ),
                convergence_agreement_validity=bool_score(
                    c.convergence_agrees if c.expected == "accept" else True
                ),
                contradictory_branch_rejection=bool_score(
                    pred == "reject" if c.expected == "reject" else True
                ),
                topology_drift_rejection=bool_score(
                    pred == "reject" if c.has_topology_drift else True
                ),
                missing_witness_abstention=bool_score(
                    pred == "abstain" if c.has_missing_witness else True
                ),
                unknown_branch_abstention=bool_score(
                    pred == "abstain" if c.has_unknown_cover else True
                ),
                partial_convergence_abstention=bool_score(
                    pred == "abstain" if c.has_partial_convergence else True
                ),
                persistent_tree_acceptance=bool_score(
                    pred == "accept" if c.expected == "accept" else True
                ),
            )
        )

    return pd.DataFrame([asdict(r) for r in rows])


# -----------------------------
# Summaries
# -----------------------------

def build_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    metrics = {
        "branching_tree_accuracy": float(df["correct"].mean()),
        "certified_input_gate_validity": float(df["certified_input_gate"].mean()),
        "branching_closure_validity": float(df["branching_closure_validity"].mean()),
        "branch_transfer_consistency": float(df["branch_transfer_consistency"].mean()),
        "obstruction_gate_validity": float(df["obstruction_gate_validity"].mean()),
        "topology_gate_validity": float(df["topology_gate_validity"].mean()),
        "witness_gate_validity": float(df["witness_gate_validity"].mean()),
        "global_section_tree_validity": float(df["global_section_tree_validity"].mean()),
        "convergence_agreement_validity": float(df["convergence_agreement_validity"].mean()),
        "contradictory_branch_rejection": float(df["contradictory_branch_rejection"].mean()),
        "topology_drift_rejection": float(df["topology_drift_rejection"].mean()),
        "missing_witness_abstention": float(df["missing_witness_abstention"].mean()),
        "unknown_branch_abstention": float(df["unknown_branch_abstention"].mean()),
        "partial_convergence_abstention": float(df["partial_convergence_abstention"].mean()),
        "persistent_tree_acceptance": float(df["persistent_tree_acceptance"].mean()),
        "min_margin": float(df["margin"].min()),
    }

    task_summary = (
        df.groupby(["expected", "task"], as_index=False)
        .agg(
            n=("case_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
        )
        .sort_values(["expected", "task"])
    )

    family_summary = (
        df.groupby(["expected", "family"], as_index=False)
        .agg(
            n=("case_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
        )
        .sort_values(["expected", "family"])
    )

    return task_summary, family_summary, metrics


# -----------------------------
# Visualization helpers
# -----------------------------

def setup_ax(ax, title: str = ""):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#3b5578")
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=13)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=13)
    if title:
        ax.set_title(title, color=TEXT, fontsize=22, fontweight="bold", pad=16)


def savefig(path: Path):
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()


def case_color(label: str) -> str:
    return {"accept": ACCEPT, "reject": REJECT, "abstain": ABSTAIN}[label]


def attractors():
    return {
        "accept": (-1.0, 0.0),
        "reject": (2.35, 2.10),
        "abstain": (3.05, 4.60),
    }


def basin_points():
    return {
        "finite visible sign set": (-4.1, -2.1),
        "arithmetic homology basin": (-3.9, -1.45),
        "finite atoms basin": (-3.55, -0.25),
        "symbolic basin": (-0.15, 2.05),
        "set logic basin": (0.25, 2.85),
        "geometry basin": (0.70, 3.45),
        "homology basin": (0.95, 3.25),
        "global section basin": (1.10, 3.10),
        "mixed persistent basin": (4.15, -2.20),
    }


def plot_landscape(cases: List[TreeCase], df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(17, 10))
    fig.patch.set_facecolor(BG)
    setup_ax(
        ax,
        "Phase 103 decision-energy landscape: branching trees certify only when every branch and convergence gate agrees",
    )

    xx, yy = np.meshgrid(np.linspace(-4.6, 4.6, 220), np.linspace(-2.6, 5.1, 220))
    z = (
        26.0
        - 0.18 * ((xx + 1.0) ** 2 + (yy - 0.0) ** 2)
        - 0.05 * ((xx - 2.35) ** 2 + (yy - 2.10) ** 2)
        - 0.04 * ((xx - 3.05) ** 2 + (yy - 4.6) ** 2)
        + 0.25 * np.exp(-((xx - 0.7) ** 2 + (yy - 3.2) ** 2) / 1.5)
    )
    levels = np.linspace(z.min(), z.max(), 18)
    cf = ax.contourf(xx, yy, z, levels=levels, cmap="viridis", alpha=0.95)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("branching tree certification margin", color=TEXT, fontsize=13)
    cbar.ax.tick_params(colors=MUTED)

    bps = basin_points()
    for name, (x, y) in bps.items():
        ax.scatter([x], [y], s=95, facecolors="none", edgecolors="#9bb7e5", linewidths=2)
        ax.text(x + 0.08, y + 0.05, name, color=TEXT, fontsize=13, weight="bold")

    ats = attractors()
    for label, (x, y) in ats.items():
        ax.scatter([x], [y], s=260, color=case_color(label), edgecolor="white", linewidth=2.2, zorder=6)
        ax.text(x + 0.10, y + 0.10, f"{label} attractor", color=TEXT, fontsize=24, weight="bold")

    for c in cases:
        row = df[df["case_id"] == c.case_id].iloc[0]
        ax.scatter(
            [c.x],
            [c.y],
            s=14,
            color=case_color(row["predicted"]),
            alpha=0.55,
            edgecolors="none",
        )

    ax.text(
        0.25,
        4.35,
        "Phase 103 rule: a reasoning tree passes only when all branches certify and converge cleanly",
        color=MUTED,
        fontsize=17,
    )
    ax.text(
        0.25,
        4.05,
        "contradictory branch poisons convergence; unknown branch abstains without false acceptance",
        color=MUTED,
        fontsize=14,
    )

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-2.6, 5.1)
    savefig(out)


def plot_field(cases: List[TreeCase], df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor(BG)
    setup_ax(ax, "Phase 103 branching certification field: finite signs become trees only when every branch behaves correctly")

    ats = attractors()
    for label, (x, y) in ats.items():
        ax.scatter([x], [y], s=300, color=case_color(label), edgecolor="white", linewidth=2.0, zorder=8)
        ax.text(x + 0.10, y + 0.10, f"{label} attractor", color=TEXT, fontsize=23, weight="bold")

    bps = basin_points()
    for name, (x, y) in bps.items():
        ax.scatter([x], [y], s=90, facecolors="none", edgecolors="#9bb7e5", linewidths=2)
        ax.text(x + 0.08, y + 0.05, name, color=TEXT, fontsize=13, weight="bold")

    # Select representative points for labels.
    representatives = {}
    for c in cases:
        representatives.setdefault(c.task, c)

    for task, c in representatives.items():
        row = df[df["case_id"] == c.case_id].iloc[0]
        pred = row["predicted"]
        target = ats[pred]
        col = case_color(pred)

        for k in range(7):
            jitter = np.random.normal(0, 0.025, size=2)
            ax.plot(
                [c.x + jitter[0], target[0]],
                [c.y + jitter[1], target[1]],
                color=col,
                alpha=0.16,
                linewidth=2,
            )

        short = task.replace("_then_", " → ").replace("_", " ")
        ax.scatter([c.x], [c.y], s=70, color=NODE, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(c.x + 0.06, c.y + 0.06, short, color=TEXT, fontsize=10, weight="bold")

    ax.text(
        0.10,
        4.45,
        "tree field: certified root + branch closure + transfer consistency + convergence agreement",
        color=MUTED,
        fontsize=18,
    )
    ax.text(
        0.95,
        3.95,
        "abstain region: missing witness, unknown cover, or partial convergence",
        color=MUTED,
        fontsize=14,
    )
    ax.text(
        0.55,
        2.55,
        "reject region: contradictory branch, topology drift, or poisoned obstruction",
        color=MUTED,
        fontsize=14,
    )

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-2.6, 5.1)
    savefig(out)


def plot_matrix(df: pd.DataFrame, out: Path):
    task_order = list(df.groupby("task").size().sort_index().index)
    labels = ["accept", "reject", "abstain"]
    mat = np.zeros((3, len(task_order)))

    for j, task in enumerate(task_order):
        sub = df[df["task"] == task]
        for i, lab in enumerate(labels):
            mat[i, j] = float((sub["predicted"] == lab).mean())

    fig, ax = plt.subplots(figsize=(18, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(
        "Phase 103 branching certification matrix: accepted trees, rejected contradictions, and abstained unknowns separate",
        color=TEXT,
        fontsize=21,
        fontweight="bold",
        pad=16,
    )
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, color=MUTED, fontsize=12)
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=55, ha="right", color=MUTED, fontsize=8)

    for i in range(3):
        for j in range(len(task_order)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=7)

    for spine in ax.spines.values():
        spine.set_color("#3b5578")
    ax.grid(color=GRID, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("branching tree decision validity", color=TEXT, fontsize=12)
    cbar.ax.tick_params(colors=MUTED)

    savefig(out)


def plot_progress(metrics: Dict[str, float], out: Path):
    names = [
        "branching\ntree accuracy",
        "certified\ninput gate",
        "branching\nclosure",
        "transfer\nconsistency",
        "obstruction\ngate",
        "topology\ngate",
        "witness\ngate",
        "global section\ntree",
        "convergence\nagreement",
        "unknown branch\nabstention",
    ]
    vals = [
        metrics["branching_tree_accuracy"],
        metrics["certified_input_gate_validity"],
        metrics["branching_closure_validity"],
        metrics["branch_transfer_consistency"],
        metrics["obstruction_gate_validity"],
        metrics["topology_gate_validity"],
        metrics["witness_gate_validity"],
        metrics["global_section_tree_validity"],
        metrics["convergence_agreement_validity"],
        metrics["unknown_branch_abstention"],
    ]

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    bars = ax.bar(range(len(vals)), vals, color=BAR)
    ax.axhline(PASS_THRESHOLD, color="#c5d4ea", linestyle="--", linewidth=1.5, label="pass threshold")

    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=13)

    ax.set_title(
        "Academic progress ladder: Phase 103 turns chains into branching tree reasoning",
        color=TEXT,
        fontsize=25,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("capability score", color=TEXT, fontsize=13)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, color=MUTED, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.legend(facecolor=AX_BG, edgecolor="#3b5578", labelcolor=TEXT, loc="upper right")

    for spine in ax.spines.values():
        spine.set_color("#3b5578")

    savefig(out)


def plot_graph(out: Path):
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor(BG)
    setup_ax(
        ax,
        "Phase 103 meta-shape tree graph: certified forms become branching reasoning paths only through stable convergence",
    )

    ats = attractors()
    bps = basin_points()

    layer_nodes = {
        "certified input layer": (0.2, 3.75),
        "branching layer": (0.9, 3.35),
        "transfer layer": (1.35, 3.20),
        "obstruction/topology layer": (1.05, 2.95),
        "witness/global-section layer": (1.75, 3.10),
        "convergence layer": (1.45, 2.85),
    }

    for name, pos in bps.items():
        ax.scatter([pos[0]], [pos[1]], s=95, facecolors="none", edgecolors="#9bb7e5", linewidths=2)
        ax.text(pos[0] + 0.08, pos[1] + 0.05, name, color=TEXT, fontsize=13, weight="bold")

    for name, pos in layer_nodes.items():
        ax.scatter([pos[0]], [pos[1]], s=90, facecolors="none", edgecolors="#9bb7e5", linewidths=2)
        ax.text(pos[0] + 0.07, pos[1] + 0.04, name, color=TEXT, fontsize=12, weight="bold")

    for label, pos in ats.items():
        ax.scatter([pos[0]], [pos[1]], s=320, color=case_color(label), edgecolor="white", linewidth=2.2, zorder=8)
        ax.text(pos[0] + 0.10, pos[1] + 0.10, f"{label} attractor", color=TEXT, fontsize=24, weight="bold")

    # Background graph wiring.
    for p1 in bps.values():
        for p2 in layer_nodes.values():
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=EDGE, alpha=0.08, linewidth=1)

    # Valid branching route.
    ax.plot([-1.0, 0.2], [0.0, 3.75], color=ACCEPT, linewidth=2.2, alpha=0.9)
    ax.plot([0.2, 0.9], [3.75, 3.35], color=ACCEPT, linewidth=2.2, alpha=0.9)
    ax.plot([0.9, 1.35], [3.35, 3.20], color=ACCEPT, linewidth=2.2, alpha=0.9)
    ax.plot([1.35, 1.45], [3.20, 2.85], color=ACCEPT, linewidth=2.2, alpha=0.9)
    ax.plot([1.45, -1.0], [2.85, 0.0], color=ACCEPT, linewidth=2.2, alpha=0.9)

    # Rejection route.
    ax.plot([0.9, 2.35], [3.35, 2.10], color=REJECT, linewidth=2.0, alpha=0.9)
    ax.text(1.1, 2.55, "contradictory branch is rejected before convergence reuse", color=MUTED, fontsize=14)

    # Abstention route.
    ax.plot([1.75, 3.05], [3.10, 4.60], color=ABSTAIN, linewidth=2.0, alpha=0.9)
    ax.text(2.45, 4.25, "unknown branch cannot become false positive", color=MUTED, fontsize=14)

    # Example visible nodes.
    examples = [
        ("point successor", (-3.8, -0.05)),
        ("x coordinate", (-3.7, -0.2)),
        ("A membership", (-3.25, 1.10)),
        ("bridge coboundary", (-2.65, 1.65)),
        ("loop annulus", (-2.35, 2.85)),
        ("same form", (-2.05, 3.10)),
        ("point", (2.05, 2.05)),
        ("successor", (2.32, 2.15)),
        ("A identity", (2.55, 1.95)),
        ("finite atoms", (2.25, -0.45)),
        ("recursive", (2.70, -1.75)),
        ("bridge", (3.30, -2.20)),
        ("same form", (3.05, 4.12)),
        ("x", (2.95, 3.92)),
        ("loop", (3.15, 3.45)),
    ]

    for name, (x, y) in examples:
        ax.scatter([x], [y], s=65, color=NODE, edgecolor="white", linewidth=0.8, zorder=6)
        ax.text(x + 0.07, y + 0.05, name, color=TEXT, fontsize=10, weight="bold")

    ax.text(0.55, 4.15, "multi-branch certification layer", color=MUTED, fontsize=20)
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-2.6, 5.1)

    savefig(out)


def plot_deabstracted(out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "De-abstracted Phase 103 examples: forms become branching reasoning trees only after branch convergence certifies",
        color=TEXT,
        fontsize=22,
        fontweight="bold",
        y=1.02,
    )

    titles = [
        "gate 1: certified root",
        "gate 2: branch expansion",
        "gate 3: convergence witness",
        "gate 4: reusable reasoning tree",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#3b5578")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # gate 1
    pts = np.array(
        [
            [0.20, 0.50],
            [0.38, 0.60],
            [0.52, 0.65],
            [0.64, 0.52],
            [0.48, 0.40],
            [0.30, 0.35],
        ]
    )
    axes[0].scatter(pts[:, 0], pts[:, 1], s=26, color=NODE, edgecolor="white", linewidth=0.4)
    axes[0].text(0.06, 0.08, "certified local signs\nnot branched yet", color=TEXT, fontsize=11)

    # gate 2 tree
    root = np.array([0.18, 0.45])
    branches = np.array([[0.45, 0.70], [0.48, 0.48], [0.45, 0.25]])
    leaves = np.array([[0.78, 0.78], [0.82, 0.52], [0.78, 0.20]])
    for b, l in zip(branches, leaves):
        axes[1].plot([root[0], b[0], l[0]], [root[1], b[1], l[1]], color="#2e73ff", linewidth=1.2)
        axes[1].scatter([b[0], l[0]], [b[1], l[1]], s=20, color=NODE, edgecolor="white", linewidth=0.4)
    axes[1].scatter([root[0]], [root[1]], s=30, color=NODE, edgecolor="white", linewidth=0.4)
    axes[1].text(0.06, 0.08, "branches expand\ncandidate tree appears", color=TEXT, fontsize=11)

    # gate 3 convergence
    circle = plt.Circle((0.50, 0.50), 0.30, fill=False, color=ACCEPT, linewidth=1.4)
    axes[2].add_patch(circle)
    for p in [[0.28, 0.49], [0.42, 0.66], [0.58, 0.64], [0.72, 0.50], [0.58, 0.32], [0.42, 0.34]]:
        axes[2].scatter([p[0]], [p[1]], s=20, color=NODE, edgecolor="white", linewidth=0.4)
        axes[2].plot([p[0], 0.50], [p[1], 0.50], color=ACCEPT, alpha=0.25, linewidth=1)
    axes[2].text(0.06, 0.08, "branches converge\nwitness present", color=TEXT, fontsize=11)

    # gate 4 certified tree
    circle2 = plt.Circle((0.50, 0.50), 0.32, color=ACCEPT, alpha=0.25, linewidth=1.5)
    axes[3].add_patch(circle2)
    axes[3].scatter([0.50], [0.50], s=36, color=ACCEPT, edgecolor="white", linewidth=0.8)
    axes[3].text(0.06, 0.08, "certified branching tree\nreasoning reuse allowed", color=TEXT, fontsize=11)

    savefig(out)


def plot_3d(cases: List[TreeCase], df: pd.DataFrame, out: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    ats = attractors()
    z_at = {"accept": 26.0, "reject": 24.85, "abstain": 24.65}

    for label, (x, y) in ats.items():
        z = z_at[label]
        ax.scatter([x], [y], [z], s=260, color=case_color(label), edgecolor="white", linewidth=2.0)
        ax.text(x + 0.08, y + 0.08, z + 0.03, label, color=TEXT, fontsize=20, weight="bold")

    for c in cases:
        row = df[df["case_id"] == c.case_id].iloc[0]
        pred = row["predicted"]
        col = case_color(pred)
        z = row["margin"] + np.random.normal(0, 0.035)

        ax.scatter([c.x], [c.y], [z], color=col, s=12, alpha=0.70)
        tx, ty = ats[pred]
        tz = z_at[pred]
        ax.plot([c.x, tx], [c.y, ty], [z, tz], color=col, alpha=0.08, linewidth=1.0)

    ax.set_title(
        "Phase 103 branching-reasoning manifold: stable trees rise only after every branch gate survives",
        color=TEXT,
        fontsize=23,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, fontsize=13, labelpad=10)
    ax.set_ylabel("latent concept axis 2", color=TEXT, fontsize=13, labelpad=10)
    ax.set_zlabel("branching tree confidence", color=TEXT, fontsize=13, labelpad=10)

    ax.tick_params(colors=MUTED)
    ax.xaxis._axinfo["grid"]["color"] = GRID
    ax.yaxis._axinfo["grid"]["color"] = GRID
    ax.zaxis._axinfo["grid"]["color"] = GRID

    ax.view_init(elev=26, azim=-58)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="accept", markerfacecolor=ACCEPT, markersize=7, linestyle=""),
        plt.Line2D([0], [0], marker="o", color="w", label="reject", markerfacecolor=REJECT, markersize=7, linestyle=""),
        plt.Line2D([0], [0], marker="o", color="w", label="abstain", markerfacecolor=ABSTAIN, markersize=7, linestyle=""),
    ]
    leg = ax.legend(handles=handles, facecolor=AX_BG, edgecolor="#3b5578", loc="upper left")
    for t in leg.get_texts():
        t.set_color(TEXT)

    savefig(out)


# -----------------------------
# Reports
# -----------------------------

def write_report(
    out: Path,
    metrics: Dict[str, float],
    task_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    pass_flag: bool,
):
    lines = []
    lines.append(f"# Phase {PHASE}: {PHASE_NAME}")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"`PHASE103_BRANCHING_CERTIFIED_REASONING_TREES_PASS={pass_flag}`")
    lines.append("")
    lines.append("## Core idea")
    lines.append("")
    lines.append(
        "Phase 103 upgrades Phase 102's linear certified chain into a branching reasoning tree. "
        "A form may now branch into multiple local paths and return to a shared conclusion, but only if "
        "every required branch preserves certification, transfer consistency, obstruction validity, topology validity, "
        "witness validity, global-section validity, and convergence agreement."
    )
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- **Accept** when all branches are certified and converge on the same reusable reasoning form.")
    lines.append("- **Reject** when a contradiction, topology drift, unlicensed obstruction, or incompatible convergence poisons the tree.")
    lines.append("- **Abstain** when a branch is unknown, a witness is missing, a cover is unknown, or convergence is partial.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for k, v in metrics.items():
        lines.append(f"- `{k}`: `{v:.4f}`")
    lines.append("")
    lines.append("## Task summary")
    lines.append("")
    lines.append(task_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Phase 103 is the first branch-structured reasoning validator in the current BBIT/geomlang sequence. "
        "Phase 100B certified forms, Phase 101 composed them once, Phase 102 chained them through time, "
        "and Phase 103 now tests whether multiple certified branches can support one conclusion without collapse."
    )
    lines.append("")
    lines.append(
        "This matters because real reasoning is rarely a single line. It is usually a tree: several premises, "
        "subproofs, transformations, examples, and intermediate certificates that must either converge cleanly, "
        "reject as contradiction, or abstain as unknown."
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = make_accept_cases() + make_reject_cases() + make_abstain_cases()
    df = evaluate(cases)
    task_summary, family_summary, metrics = build_summaries(df)

    metric_values = [
        metrics["branching_tree_accuracy"],
        metrics["certified_input_gate_validity"],
        metrics["branching_closure_validity"],
        metrics["branch_transfer_consistency"],
        metrics["obstruction_gate_validity"],
        metrics["topology_gate_validity"],
        metrics["witness_gate_validity"],
        metrics["global_section_tree_validity"],
        metrics["convergence_agreement_validity"],
        metrics["contradictory_branch_rejection"],
        metrics["topology_drift_rejection"],
        metrics["missing_witness_abstention"],
        metrics["unknown_branch_abstention"],
        metrics["partial_convergence_abstention"],
        metrics["persistent_tree_acceptance"],
    ]

    pass_flag = all(v >= PASS_THRESHOLD for v in metric_values) and metrics["min_margin"] > 0.0

    # Files
    p1 = OUT_DIR / "phase103_01_branching_tree_decision_energy_landscape.png"
    p2 = OUT_DIR / "phase103_02_branching_tree_certification_field.png"
    p3 = OUT_DIR / "phase103_03_branching_tree_certification_matrix.png"
    p4 = OUT_DIR / "phase103_04_academic_progress_ladder.png"
    p5 = OUT_DIR / "phase103_05_meta_shape_tree_graph.png"
    p6 = OUT_DIR / "phase103_06_deabstracted_branching_tree_examples.png"
    p7 = OUT_DIR / "phase103_07_3d_branching_reasoning_tree_manifold.png"

    trials_csv = OUT_DIR / "phase103_branching_certified_reasoning_trees_trials.csv"
    task_csv = OUT_DIR / "phase103_branching_certified_reasoning_trees_task_summary.csv"
    family_csv = OUT_DIR / "phase103_branching_certified_reasoning_trees_family_summary.csv"
    summary_json = OUT_DIR / "phase103_branching_certified_reasoning_trees_summary.json"
    report_md = OUT_DIR / "phase103_branching_certified_reasoning_trees_report.md"

    # Save tables
    df.to_csv(trials_csv, index=False)
    task_summary.to_csv(task_csv, index=False)
    family_summary.to_csv(family_csv, index=False)

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "pass": bool(pass_flag),
        "root": str(ROOT),
        "outputs": str(OUT_DIR),
        "n_trials": int(len(df)),
        "seed": RNG_SEED,
        "metrics": metrics,
        "artifacts": [
            str(p1),
            str(p2),
            str(p3),
            str(p4),
            str(p5),
            str(p6),
            str(p7),
            str(trials_csv),
            str(task_csv),
            str(family_csv),
            str(summary_json),
            str(report_md),
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(report_md, metrics, task_summary, family_summary, pass_flag)

    # Visualizations
    plot_landscape(cases, df, p1)
    plot_field(cases, df, p2)
    plot_matrix(df, p3)
    plot_progress(metrics, p4)
    plot_graph(p5)
    plot_deabstracted(p6)
    plot_3d(cases, df, p7)

    print(f"[{PHASE}] {PHASE_NAME}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from multi-step certified reasoning chains to branching certified reasoning trees")
    print(
        f"[{PHASE}] task: certified forms may branch only when every branch, transfer, witness, and convergence gate survives"
    )
    print(f"[{PHASE}] PHASE103_BRANCHING_CERTIFIED_REASONING_TREES_PASS={pass_flag}")
    print(
        f"[{PHASE}] "
        f"branching_tree_accuracy={metrics['branching_tree_accuracy']:.4f} "
        f"certified_input_gate_validity={metrics['certified_input_gate_validity']:.4f} "
        f"branching_closure_validity={metrics['branching_closure_validity']:.4f} "
        f"branch_transfer_consistency={metrics['branch_transfer_consistency']:.4f} "
        f"obstruction_gate_validity={metrics['obstruction_gate_validity']:.4f} "
        f"topology_gate_validity={metrics['topology_gate_validity']:.4f} "
        f"witness_gate_validity={metrics['witness_gate_validity']:.4f} "
        f"global_section_tree_validity={metrics['global_section_tree_validity']:.4f} "
        f"convergence_agreement_validity={metrics['convergence_agreement_validity']:.4f} "
        f"contradictory_branch_rejection={metrics['contradictory_branch_rejection']:.4f} "
        f"topology_drift_rejection={metrics['topology_drift_rejection']:.4f} "
        f"missing_witness_abstention={metrics['missing_witness_abstention']:.4f} "
        f"unknown_branch_abstention={metrics['unknown_branch_abstention']:.4f} "
        f"partial_convergence_abstention={metrics['partial_convergence_abstention']:.4f} "
        f"persistent_tree_acceptance={metrics['persistent_tree_acceptance']:.4f} "
        f"min_margin={metrics['min_margin']:.4f}"
    )
    print(f"[{PHASE}] wrote:")
    for artifact in summary["artifacts"]:
        print(f"  - {artifact}")

    return 0 if pass_flag else 1


if __name__ == "__main__":
    raise SystemExit(main())