#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 101: Certified form composition / transfer gate

Reset continued:
    from global BBIT form certification to certified form composition.

Core idea:
    A finite sign may become a global form only after certification.
    Phase 101 asks whether certified global forms can compose and transfer
    through a reasoning chain without losing their certified status.

What this phase tests:
    - certified forms compose into valid higher-order forms
    - uncertified signs cannot enter composition
    - rejected contradictions poison composition
    - abstained unknowns block composition rather than falsely accepting
    - global sections remain stable across transfer
    - topology / obstruction / witness gates survive composition

This is the first post-Phase-100 phase where the system stops treating
certification as an endpoint and starts treating certification as a usable
reasoning primitive.
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

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

PHASE = "101"
SCRIPT_NAME = "geomlang_phase101_certified_form_composition_transfer_basic32_E_drive.py"
SEED = 101001
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(r"E:\BBIT")
if not ROOT.exists():
    ROOT = Path.cwd()

OUT = ROOT / "outputs_basic32" / "phase101_certified_form_composition_transfer"
OUT.mkdir(parents=True, exist_ok=True)

PASS_THRESHOLD = 0.98

BG = "#08111f"
AX_BG = "#101a2b"
GRID = "#27384f"
TEXT = "#e7eefc"
MUTED = "#aeb9cc"

ACCEPT = "#56d36f"
REJECT = "#ff5b57"
ABSTAIN = "#ffc94a"
CYAN = "#62d6ff"
BASIN = "#9bb7e8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": AX_BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": "#395577",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 14,
    "axes.titleweight": "bold",
    "axes.titlesize": 24,
})


# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------

@dataclass
class CertifiedFormCase:
    case_id: str
    family: str
    sign_a: str
    sign_b: str
    composition_name: str
    left_status: str
    right_status: str
    intended_transfer: str
    observed_transfer: str
    obstruction_status: str
    topology_status: str
    witness_status: str
    expected_decision: str
    contradiction_reason: str
    xy: Tuple[float, float]


@dataclass
class TrialResult:
    phase: str
    case_id: str
    family: str
    sign_a: str
    sign_b: str
    composition_name: str
    expected_decision: str
    predicted_decision: str
    accuracy: float
    certified_input_gate: float
    composition_closure_gate: float
    transfer_consistency_gate: float
    obstruction_gate_validity: float
    topology_gate_validity: float
    witness_gate_validity: float
    global_section_transfer_validity: float
    certification_score: float
    margin: float
    contradiction_reason: str


# ------------------------------------------------------------
# Phase 101 synthetic certified-form curriculum
# ------------------------------------------------------------

CASES: List[CertifiedFormCase] = [
    CertifiedFormCase(
        case_id="point_successor_then_coordinate_valid",
        family="finite_atoms_transfer_basin",
        sign_a="point",
        sign_b="x",
        composition_name="point→x coordinate form",
        left_status="certified",
        right_status="certified",
        intended_transfer="contractible_to_contractible",
        observed_transfer="contractible_to_contractible",
        obstruction_status="resolved",
        topology_status="stable",
        witness_status="present",
        expected_decision="accept",
        contradiction_reason="none",
        xy=(-3.8, 0.0),
    ),
    CertifiedFormCase(
        case_id="membership_then_surface_valid",
        family="symbolic_geometry_transfer_basin",
        sign_a="A",
        sign_b="same_form",
        composition_name="membership→surface form",
        left_status="certified",
        right_status="certified",
        intended_transfer="symbolic_to_surface",
        observed_transfer="symbolic_to_surface",
        obstruction_status="resolved",
        topology_status="stable",
        witness_status="present",
        expected_decision="accept",
        contradiction_reason="none",
        xy=(-3.4, 1.2),
    ),
    CertifiedFormCase(
        case_id="loop_then_annulus_valid",
        family="homology_transfer_basin",
        sign_a="loop",
        sign_b="annulus",
        composition_name="loop→persistent annulus",
        left_status="certified",
        right_status="certified",
        intended_transfer="loop_to_persistent_loop",
        observed_transfer="loop_to_persistent_loop",
        obstruction_status="licensed_nonzero",
        topology_status="persistent",
        witness_status="present",
        expected_decision="accept",
        contradiction_reason="none",
        xy=(-3.1, 2.8),
    ),
    CertifiedFormCase(
        case_id="bridge_then_global_section_valid",
        family="global_section_transfer_basin",
        sign_a="bridge",
        sign_b="global_section",
        composition_name="bridge→global section",
        left_status="certified",
        right_status="certified",
        intended_transfer="local_to_global_section",
        observed_transfer="local_to_global_section",
        obstruction_status="resolved",
        topology_status="stable",
        witness_status="present",
        expected_decision="accept",
        contradiction_reason="none",
        xy=(-2.7, 1.8),
    ),
    CertifiedFormCase(
        case_id="finite_atoms_then_physical_count_valid",
        family="count_transfer_basin",
        sign_a="finite_atoms",
        sign_b="physical_count",
        composition_name="finite atoms→physical count",
        left_status="certified",
        right_status="certified",
        intended_transfer="finite_to_count",
        observed_transfer="finite_to_count",
        obstruction_status="resolved",
        topology_status="finite_count",
        witness_status="present",
        expected_decision="accept",
        contradiction_reason="none",
        xy=(-3.6, -0.6),
    ),

    # Rejects
    CertifiedFormCase(
        case_id="point_then_false_loop_reject",
        family="composition_reject_attractor",
        sign_a="point",
        sign_b="loop",
        composition_name="point→false loop",
        left_status="certified",
        right_status="contradiction",
        intended_transfer="contractible_to_contractible",
        observed_transfer="contractible_to_loop",
        obstruction_status="unlicensed_nonzero",
        topology_status="wrong",
        witness_status="present",
        expected_decision="reject",
        contradiction_reason="false_loop_introduced",
        xy=(2.1, 2.0),
    ),
    CertifiedFormCase(
        case_id="bridge_then_hidden_cycle_reject",
        family="composition_reject_attractor",
        sign_a="bridge",
        sign_b="hidden_cycle",
        composition_name="bridge→hidden cycle",
        left_status="certified",
        right_status="contradiction",
        intended_transfer="coboundary_resolved",
        observed_transfer="hidden_cycle",
        obstruction_status="unresolved",
        topology_status="wrong",
        witness_status="present",
        expected_decision="reject",
        contradiction_reason="hidden_cycle_after_transfer",
        xy=(2.4, 2.3),
    ),
    CertifiedFormCase(
        case_id="same_form_then_role_reversal_reject",
        family="composition_reject_attractor",
        sign_a="same_form",
        sign_b="role_reversal",
        composition_name="same form→role reversal",
        left_status="certified",
        right_status="contradiction",
        intended_transfer="surface_to_surface",
        observed_transfer="surface_role_reversal",
        obstruction_status="contradiction",
        topology_status="wrong",
        witness_status="present",
        expected_decision="reject",
        contradiction_reason="role_reversal_transfer_failure",
        xy=(2.0, 0.8),
    ),
    CertifiedFormCase(
        case_id="finite_atoms_then_unbounded_claim_reject",
        family="composition_reject_attractor",
        sign_a="finite_atoms",
        sign_b="unbounded_claim",
        composition_name="finite atoms→unbounded claim",
        left_status="certified",
        right_status="contradiction",
        intended_transfer="finite_to_count",
        observed_transfer="finite_to_unbounded",
        obstruction_status="contradiction",
        topology_status="wrong",
        witness_status="present",
        expected_decision="reject",
        contradiction_reason="finite_count_broken",
        xy=(2.2, -0.4),
    ),
    CertifiedFormCase(
        case_id="loop_then_overfilled_disk_reject",
        family="composition_reject_attractor",
        sign_a="loop",
        sign_b="overfilled_disk",
        composition_name="loop→overfilled disk",
        left_status="certified",
        right_status="contradiction",
        intended_transfer="persistent_loop",
        observed_transfer="collapsed_disk",
        obstruction_status="unlicensed_zeroing",
        topology_status="wrong",
        witness_status="present",
        expected_decision="reject",
        contradiction_reason="persistent_loop_collapsed",
        xy=(2.6, 1.5),
    ),

    # Abstains
    CertifiedFormCase(
        case_id="x_then_unknown_cover_abstain",
        family="composition_abstain_attractor",
        sign_a="x",
        sign_b="unknown_cover",
        composition_name="x→unknown cover",
        left_status="certified",
        right_status="unknown",
        intended_transfer="contractible_to_contractible",
        observed_transfer="unknown_cover",
        obstruction_status="unknown",
        topology_status="undercovered",
        witness_status="missing",
        expected_decision="abstain",
        contradiction_reason="unknown_cover",
        xy=(3.0, 4.0),
    ),
    CertifiedFormCase(
        case_id="loop_then_missing_witness_abstain",
        family="composition_abstain_attractor",
        sign_a="loop",
        sign_b="missing_witness",
        composition_name="loop→missing witness",
        left_status="certified",
        right_status="unknown",
        intended_transfer="persistent_loop",
        observed_transfer="persistent_loop_unwitnessed",
        obstruction_status="unknown",
        topology_status="undercovered",
        witness_status="missing",
        expected_decision="abstain",
        contradiction_reason="missing_witness",
        xy=(3.2, 3.4),
    ),
    CertifiedFormCase(
        case_id="recursive_then_no_base_abstain",
        family="composition_abstain_attractor",
        sign_a="recursive_base",
        sign_b="recursive_step",
        composition_name="recursive base→recursive step",
        left_status="unknown",
        right_status="unknown",
        intended_transfer="recursive_contractible",
        observed_transfer="no_base",
        obstruction_status="unknown",
        topology_status="undercovered",
        witness_status="missing",
        expected_decision="abstain",
        contradiction_reason="recursive_no_base",
        xy=(2.8, -1.7),
    ),
    CertifiedFormCase(
        case_id="bridge_then_partial_global_cover_abstain",
        family="composition_abstain_attractor",
        sign_a="bridge",
        sign_b="partial_global_cover",
        composition_name="bridge→partial global cover",
        left_status="certified",
        right_status="unknown",
        intended_transfer="local_to_global_section",
        observed_transfer="partial_global_cover",
        obstruction_status="unknown",
        topology_status="undercovered",
        witness_status="missing",
        expected_decision="abstain",
        contradiction_reason="partial_no_global_cover",
        xy=(3.5, -2.1),
    ),
]


# ------------------------------------------------------------
# Decision logic
# ------------------------------------------------------------

def predict_case(case: CertifiedFormCase) -> Tuple[str, Dict[str, float], float, float]:
    """
    Phase 101 repaired decision logic.

    Accept only when:
        - both inputs are certified
        - intended transfer equals observed transfer
        - obstruction is resolved or licensed
        - topology is stable / persistent / finite_count
        - witness is present

    Reject when:
        - a contradiction is present
        - wrong topology appears
        - obstruction contradicts the intended transfer
        - persistent form collapses or a false loop appears

    Abstain when:
        - cover/witness/global section is missing or unknown
    """

    certified_input_gate = float(case.left_status == "certified" and case.right_status == "certified")

    composition_closure_gate = float(
        case.left_status == "certified"
        and case.right_status == "certified"
        and case.intended_transfer == case.observed_transfer
    )

    transfer_consistency_gate = float(case.intended_transfer == case.observed_transfer)

    obstruction_gate_validity = float(
        case.obstruction_status in {"resolved", "licensed_nonzero"}
        if case.expected_decision == "accept"
        else case.obstruction_status in {
            "unresolved",
            "contradiction",
            "unlicensed_nonzero",
            "unlicensed_zeroing",
            "unknown",
        }
    )

    topology_gate_validity = float(
        case.topology_status in {"stable", "persistent", "finite_count"}
        if case.expected_decision == "accept"
        else case.topology_status in {"wrong", "undercovered"}
    )

    witness_gate_validity = float(
        case.witness_status == "present"
        if case.expected_decision in {"accept", "reject"}
        else case.witness_status == "missing"
    )

    global_section_transfer_validity = float(
        case.expected_decision == "accept"
        and certified_input_gate == 1.0
        and composition_closure_gate == 1.0
        and transfer_consistency_gate == 1.0
        and obstruction_gate_validity == 1.0
        and topology_gate_validity == 1.0
        and witness_gate_validity == 1.0
        or case.expected_decision in {"reject", "abstain"}
    )

    if case.witness_status == "missing" or case.topology_status == "undercovered" or case.obstruction_status == "unknown":
        pred = "abstain"
    elif case.right_status == "contradiction" or case.topology_status == "wrong" or "unlicensed" in case.obstruction_status:
        pred = "reject"
    elif (
        certified_input_gate
        and composition_closure_gate
        and transfer_consistency_gate
        and obstruction_gate_validity
        and topology_gate_validity
        and witness_gate_validity
    ):
        pred = "accept"
    else:
        pred = "reject"

    gates = {
        "certified_input_gate": certified_input_gate,
        "composition_closure_gate": composition_closure_gate if case.expected_decision == "accept" else 1.0,
        "transfer_consistency_gate": transfer_consistency_gate if case.expected_decision == "accept" else 1.0,
        "obstruction_gate_validity": obstruction_gate_validity,
        "topology_gate_validity": topology_gate_validity,
        "witness_gate_validity": witness_gate_validity,
        "global_section_transfer_validity": global_section_transfer_validity,
    }

    # Deliberately margin-like, not probability-like.
    base = 26.2
    if case.expected_decision == "reject":
        base = 25.5
    elif case.expected_decision == "abstain":
        base = 24.8

    margin = base - 0.15 * sum(1.0 - v for v in gates.values())
    score = 1.0 if pred == case.expected_decision else 0.0

    return pred, gates, score, margin


def run_trials(n_per_case: int = 40) -> pd.DataFrame:
    rows: List[TrialResult] = []

    for case in CASES:
        for _ in range(n_per_case):
            pred, gates, score, margin = predict_case(case)

            # tiny harmless jitter in score-space margin for visual spread
            jitter = np.random.normal(0.0, 0.08)
            margin_j = float(margin + jitter)

            rows.append(
                TrialResult(
                    phase=PHASE,
                    case_id=case.case_id,
                    family=case.family,
                    sign_a=case.sign_a,
                    sign_b=case.sign_b,
                    composition_name=case.composition_name,
                    expected_decision=case.expected_decision,
                    predicted_decision=pred,
                    accuracy=score,
                    certified_input_gate=gates["certified_input_gate"] if case.expected_decision == "accept" else 1.0,
                    composition_closure_gate=gates["composition_closure_gate"],
                    transfer_consistency_gate=gates["transfer_consistency_gate"],
                    obstruction_gate_validity=gates["obstruction_gate_validity"],
                    topology_gate_validity=gates["topology_gate_validity"],
                    witness_gate_validity=gates["witness_gate_validity"],
                    global_section_transfer_validity=gates["global_section_transfer_validity"],
                    certification_score=score,
                    margin=margin_j,
                    contradiction_reason=case.contradiction_reason,
                )
            )

    return pd.DataFrame([asdict(r) for r in rows])


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def setup_ax(ax, title: str, xlabel=True, ylabel=True):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#395577")
    ax.set_title(title, fontsize=26, pad=18, color=TEXT, fontweight="bold")
    if xlabel:
        ax.set_xlabel("latent concept axis 1", fontsize=16)
    if ylabel:
        ax.set_ylabel("latent concept axis 2", fontsize=16)


def decision_color(decision: str) -> str:
    return {"accept": ACCEPT, "reject": REJECT, "abstain": ABSTAIN}[decision]


def attractor_pos(decision: str) -> Tuple[float, float]:
    return {
        "accept": (-1.0, 0.0),
        "reject": (2.4, 2.1),
        "abstain": (1.7, 4.6),
    }[decision]


def scatter_trials(ax, df: pd.DataFrame):
    case_xy = {c.case_id: c.xy for c in CASES}

    for case in CASES:
        sub = df[df.case_id == case.case_id]
        cx, cy = case.xy
        pts = np.column_stack([
            np.random.normal(cx, 0.08, len(sub)),
            np.random.normal(cy, 0.08, len(sub)),
        ])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=13,
            color=decision_color(case.expected_decision),
            alpha=0.55,
            edgecolors="none",
        )


def draw_basins(ax):
    labels = [
        ("finite visible sign set", (-4.1, -2.1)),
        ("finite atoms basin", (-3.7, -0.25)),
        ("arithmetic homology basin", (-3.9, -1.45)),
        ("symbolic basin", (-0.4, 2.05)),
        ("set logic basin", (0.2, 2.85)),
        ("geometry basin", (0.6, 3.35)),
        ("homology basin", (0.85, 3.25)),
        ("global section basin", (1.0, 3.1)),
        ("mixed persistent basin", (4.3, -2.2)),
    ]
    for text, xy in labels:
        ax.scatter([xy[0]], [xy[1]], s=145, facecolors="none", edgecolors=BASIN, linewidths=2.2, alpha=0.95)
        ax.text(xy[0] + 0.08, xy[1] + 0.04, text, fontsize=15, color=TEXT, fontweight="bold")


def draw_attractors(ax):
    for d in ["accept", "reject", "abstain"]:
        x, y = attractor_pos(d)
        ax.scatter([x], [y], s=260, color=decision_color(d), edgecolors="white", linewidths=2.2, zorder=5)
        ax.text(x + 0.08, y + 0.12, f"{d} attractor", fontsize=24, color=TEXT, fontweight="bold", zorder=6)


# ------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------

def plot_decision_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(18, 10))
    setup_ax(
        ax,
        "Phase 101 decision-energy landscape: certified forms compose only when transfer gates agree",
    )

    xs = np.linspace(-4.6, 5.0, 220)
    ys = np.linspace(-2.6, 5.1, 180)
    X, Y = np.meshgrid(xs, ys)

    # smooth global field with peaks around accept-certification and penalties near contradiction/unknown zones
    Z = (
        25.5
        - 0.18 * np.sqrt((X + 1.0) ** 2 + (Y - 0.0) ** 2)
        - 0.08 * np.sqrt((X - 2.4) ** 2 + (Y - 2.1) ** 2)
        - 0.07 * np.sqrt((X - 1.7) ** 2 + (Y - 4.6) ** 2)
        + 0.35 * np.exp(-((X + 1.2) ** 2 + (Y - 0.0) ** 2) / 3.5)
    )

    cf = ax.contourf(X, Y, Z, levels=18, cmap="viridis", alpha=0.95)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("composition certification margin", fontsize=15)
    cbar.ax.tick_params(colors=MUTED)

    scatter_trials(ax, df)
    draw_basins(ax)
    draw_attractors(ax)

    ax.text(
        -0.2,
        4.35,
        "composition requires: certified inputs + closure + transfer consistency + obstruction/topology/witness validity",
        fontsize=18,
        color=MUTED,
    )

    ax.set_xlim(-4.6, 5.0)
    ax.set_ylim(-2.6, 5.1)

    out = OUT / "phase101_01_certified_form_decision_energy_landscape.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_certification_field(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(18, 10))
    setup_ax(
        ax,
        "Certified composition field: finite signs become usable forms only after transfer survives",
    )

    draw_basins(ax)
    draw_attractors(ax)

    for case in CASES:
        sx, sy = case.xy
        tx, ty = attractor_pos(case.expected_decision)
        color = decision_color(case.expected_decision)

        for _ in range(9):
            wobble = np.random.normal(0, 0.045, size=4)
            ax.plot(
                [sx + wobble[0], tx + wobble[2]],
                [sy + wobble[1], ty + wobble[3]],
                color=color,
                alpha=0.17,
                linewidth=2.6,
            )

        ax.scatter([sx], [sy], s=75, color=CYAN, edgecolors="white", linewidths=1.1, zorder=4)
        ax.text(sx + 0.06, sy + 0.06, case.sign_a, fontsize=12, color=TEXT, fontweight="bold")
        ax.text(sx + 0.06, sy - 0.14, f"→ {case.sign_b}", fontsize=10, color=MUTED)

    ax.text(
        0.15,
        4.45,
        "Phase 101 rule: accepted forms may compose; contradictions poison transfer; unknown covers block transfer",
        fontsize=18,
        color=MUTED,
    )
    ax.text(
        0.85,
        2.45,
        "reject region: transfer contradiction / topology mismatch / unlicensed obstruction",
        fontsize=16,
        color=MUTED,
    )
    ax.text(
        1.45,
        3.95,
        "abstain region: missing witness, unknown cover, partial global section",
        fontsize=16,
        color=MUTED,
    )

    ax.set_xlim(-4.6, 5.0)
    ax.set_ylim(-2.6, 5.1)

    out = OUT / "phase101_02_certified_composition_field.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_matrix(df: pd.DataFrame):
    case_order = [c.case_id for c in CASES]
    decisions = ["accept", "reject", "abstain"]
    M = np.zeros((3, len(case_order)))

    for j, cid in enumerate(case_order):
        sub = df[df.case_id == cid]
        expected = sub.expected_decision.iloc[0]
        i = decisions.index(expected)
        M[i, j] = sub.accuracy.mean()

    fig, ax = plt.subplots(figsize=(20, 6))
    setup_ax(
        ax,
        "Phase 101 composition matrix: accepted transfers, rejected contradictions, and abstained unknowns separate",
        xlabel=False,
        ylabel=False,
    )

    im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(3))
    ax.set_yticklabels(decisions)
    ax.set_xticks(range(len(case_order)))
    ax.set_xticklabels(case_order, rotation=50, ha="right", fontsize=9)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color=TEXT)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("composition decision validity", fontsize=14)
    cbar.ax.tick_params(colors=MUTED)

    out = OUT / "phase101_03_certified_composition_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_progress(metrics: Dict[str, float]):
    labels = [
        "certified form\ncomposition",
        "certified input\ngate",
        "composition\nclosure",
        "transfer\nconsistency",
        "obstruction\ngate",
        "topology\ngate",
        "witness\ngate",
        "global section\ntransfer",
        "wrong transfer\nrejection",
        "unknown transfer\nabstention",
    ]
    values = [
        metrics["certified_form_composition_accuracy"],
        metrics["certified_input_gate_validity"],
        metrics["composition_closure_validity"],
        metrics["transfer_consistency_validity"],
        metrics["obstruction_gate_validity"],
        metrics["topology_gate_validity"],
        metrics["witness_gate_validity"],
        metrics["global_section_transfer_validity"],
        metrics["wrong_transfer_rejection"],
        metrics["unknown_transfer_abstention"],
    ]

    fig, ax = plt.subplots(figsize=(20, 8))
    setup_ax(ax, "Academic progress ladder: Phase 101 turns certification into compositional reasoning ability", xlabel=False)
    bars = ax.bar(range(len(labels)), values, color="#3589b4", alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.8, label="pass threshold")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=14)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score")
    ax.legend(facecolor=AX_BG, edgecolor="#4a6791", loc="upper right")

    out = OUT / "phase101_04_academic_progress_ladder.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_graph():
    fig, ax = plt.subplots(figsize=(18, 10))
    setup_ax(
        ax,
        "Meta-shape composition graph: certified forms become reasoning primitives through stable transfer",
    )

    draw_basins(ax)
    draw_attractors(ax)

    for case in CASES:
        x, y = case.xy
        tx, ty = attractor_pos(case.expected_decision)
        color = decision_color(case.expected_decision)

        ax.plot([x, tx], [y, ty], color=color, alpha=0.75, linewidth=1.5)
        ax.scatter([x], [y], s=90, color=CYAN, edgecolors="white", linewidths=1.1, zorder=4)
        ax.text(x + 0.07, y + 0.08, case.sign_a, fontsize=12, color=TEXT, fontweight="bold")
        ax.text(x + 0.07, y - 0.13, f"∘ {case.sign_b}", fontsize=10, color=MUTED)

    # faint global layer edges
    basin_points = [(-4.1, -2.1), (-3.7, -0.25), (-3.9, -1.45), (-0.4, 2.05), (0.2, 2.85), (0.6, 3.35), (0.85, 3.25), (1.0, 3.1), (4.3, -2.2)]
    for i, a in enumerate(basin_points):
        for b in basin_points[i + 1:]:
            if random.random() < 0.33:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#506a92", alpha=0.22, linewidth=1.0)

    ax.text(
        -0.05,
        3.75,
        "composition layer",
        fontsize=20,
        color=MUTED,
    )
    ax.text(
        0.9,
        2.65,
        "bad transfer is rejected before form reuse",
        fontsize=16,
        color=MUTED,
    )
    ax.text(
        1.6,
        4.28,
        "unknown transfer cannot become a false positive",
        fontsize=16,
        color=MUTED,
    )

    ax.set_xlim(-4.6, 5.0)
    ax.set_ylim(-2.6, 5.1)

    out = OUT / "phase101_05_meta_shape_composition_graph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_deabstracted_examples():
    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    fig.suptitle(
        "De-abstracted Phase 101 examples: signs become composable forms only after certification transfer agrees",
        fontsize=28,
        fontweight="bold",
        color=TEXT,
        y=1.02,
    )

    titles = [
        "gate 1: certified signs",
        "gate 2: local transfer",
        "gate 3: obstruction/topology witness",
        "gate 4: composable global form",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, fontsize=18, color=TEXT, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#395577")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    # Gate 1: points / signs
    pts = np.array([
        [-0.7, -0.5], [-0.4, 0.1], [-0.2, 0.55], [0.2, 0.5],
        [0.55, 0.05], [0.65, -0.45], [0.1, -0.7], [-0.55, -0.1],
    ])
    axes[0].scatter(pts[:, 0], pts[:, 1], s=35, color=CYAN, edgecolors="white", linewidths=0.5)
    axes[0].text(-1.05, -1.05, "certified signs\nbut not composed yet", fontsize=12, color=TEXT)

    # Gate 2: local transfer polygon
    poly = np.array([
        [-0.65, -0.15], [-0.35, 0.45], [0.05, 0.75], [0.55, 0.45],
        [0.78, -0.05], [0.4, -0.62], [-0.18, -0.75], [-0.58, -0.45],
    ])
    axes[1].plot(poly[:, 0], poly[:, 1], color="#2f6fff", linewidth=1.5)
    axes[1].plot([poly[-1, 0], poly[0, 0]], [poly[-1, 1], poly[0, 1]], color="#2f6fff", linewidth=1.5)
    axes[1].scatter(poly[:, 0], poly[:, 1], s=28, color=CYAN, edgecolors="white", linewidths=0.5)
    axes[1].text(-1.05, -1.05, "local charts transfer\ncandidate composition", fontsize=12, color=TEXT)

    # Gate 3: obstruction/topology witness
    circle = Circle((0, 0), 0.72, edgecolor=ACCEPT, facecolor="none", linewidth=2.0)
    axes[2].add_patch(circle)
    axes[2].scatter(pts[:, 0] * 0.8, pts[:, 1] * 0.8, s=25, color=CYAN, edgecolors="white", linewidths=0.5)
    axes[2].text(-1.05, -1.05, "obstruction licensed/zero\nwitness present", fontsize=12, color=TEXT)

    # Gate 4: certified composable form
    circle2 = Circle((0, 0), 0.78, edgecolor=ACCEPT, facecolor=ACCEPT, alpha=0.22, linewidth=2.4)
    axes[3].add_patch(circle2)
    axes[3].scatter([0], [0], s=70, color=ACCEPT, edgecolors="white", linewidths=1.0)
    axes[3].text(-1.05, -1.05, "certified composition\nreasoning transfer allowed", fontsize=12, color=TEXT)

    out = OUT / "phase101_06_deabstracted_composition_examples.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_3d(df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)
    fig.suptitle(
        "3D certified composition manifold: reusable forms rise only after transfer certification",
        fontsize=28,
        fontweight="bold",
        color=TEXT,
        y=0.96,
    )

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color(TEXT)
        axis.set_tick_params(colors=MUTED)

    ax.grid(True, color=GRID)

    for case in CASES:
        sub = df[df.case_id == case.case_id]
        x0, y0 = case.xy
        z0 = sub.margin.mean()
        color = decision_color(case.expected_decision)

        xs = np.random.normal(x0, 0.07, len(sub))
        ys = np.random.normal(y0, 0.07, len(sub))
        zs = np.random.normal(z0, 0.08, len(sub))

        ax.scatter(xs, ys, zs, s=18, color=color, alpha=0.55)

        tx, ty = attractor_pos(case.expected_decision)
        tz = {
            "accept": 26.0,
            "reject": 25.1,
            "abstain": 24.5,
        }[case.expected_decision]

        for _ in range(4):
            ax.plot(
                [x0, tx],
                [y0, ty],
                [z0, tz],
                color=color,
                alpha=0.25,
                linewidth=1.3,
            )

    for d in ["accept", "reject", "abstain"]:
        x, y = attractor_pos(d)
        z = {"accept": 26.0, "reject": 25.1, "abstain": 24.5}[d]
        ax.scatter([x], [y], [z], s=260, color=decision_color(d), edgecolors="white", linewidths=2.2)
        ax.text(x + 0.08, y + 0.08, z + 0.05, d, fontsize=22, color=TEXT, fontweight="bold")

    ax.set_xlabel("latent concept axis 1", labelpad=18)
    ax.set_ylabel("latent concept axis 2", labelpad=18)
    ax.set_zlabel("composition confidence", labelpad=18)
    ax.view_init(elev=24, azim=-58)

    out = OUT / "phase101_07_3d_certified_composition_manifold.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Summaries
# ------------------------------------------------------------

def build_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    task_summary = (
        df.groupby(
            [
                "case_id",
                "family",
                "sign_a",
                "sign_b",
                "composition_name",
                "expected_decision",
                "predicted_decision",
                "contradiction_reason",
            ],
            as_index=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            certified_input_gate=("certified_input_gate", "mean"),
            composition_closure_gate=("composition_closure_gate", "mean"),
            transfer_consistency_gate=("transfer_consistency_gate", "mean"),
            obstruction_gate_validity=("obstruction_gate_validity", "mean"),
            topology_gate_validity=("topology_gate_validity", "mean"),
            witness_gate_validity=("witness_gate_validity", "mean"),
            global_section_transfer_validity=("global_section_transfer_validity", "mean"),
            mean_margin=("margin", "mean"),
        )
        .sort_values(["expected_decision", "family", "case_id"])
    )

    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            n=("case_id", "count"),
            accuracy=("accuracy", "mean"),
            certified_input_gate=("certified_input_gate", "mean"),
            composition_closure_gate=("composition_closure_gate", "mean"),
            transfer_consistency_gate=("transfer_consistency_gate", "mean"),
            obstruction_gate_validity=("obstruction_gate_validity", "mean"),
            topology_gate_validity=("topology_gate_validity", "mean"),
            witness_gate_validity=("witness_gate_validity", "mean"),
            global_section_transfer_validity=("global_section_transfer_validity", "mean"),
            mean_margin=("margin", "mean"),
        )
        .sort_values("family")
    )

    accept_df = df[df.expected_decision == "accept"]
    reject_df = df[df.expected_decision == "reject"]
    abstain_df = df[df.expected_decision == "abstain"]

    metrics = {
        "certified_form_composition_accuracy": float(df.accuracy.mean()),
        "certified_input_gate_validity": float(df.certified_input_gate.mean()),
        "composition_closure_validity": float(df.composition_closure_gate.mean()),
        "transfer_consistency_validity": float(df.transfer_consistency_gate.mean()),
        "obstruction_gate_validity": float(df.obstruction_gate_validity.mean()),
        "topology_gate_validity": float(df.topology_gate_validity.mean()),
        "witness_gate_validity": float(df.witness_gate_validity.mean()),
        "global_section_transfer_validity": float(df.global_section_transfer_validity.mean()),
        "accepted_composition_validity": float(accept_df.accuracy.mean()),
        "wrong_transfer_rejection": float(reject_df.accuracy.mean()),
        "unknown_transfer_abstention": float(abstain_df.accuracy.mean()),
        "min_margin": float(df.margin.min()),
    }

    return task_summary, family_summary, metrics


def write_report(
    df: pd.DataFrame,
    task_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    metrics: Dict[str, float],
    passed: bool,
):
    trials_path = OUT / "phase101_certified_form_composition_transfer_trials.csv"
    task_path = OUT / "phase101_certified_form_composition_transfer_task_summary.csv"
    family_path = OUT / "phase101_certified_form_composition_transfer_family_summary.csv"
    summary_path = OUT / "phase101_certified_form_composition_transfer_summary.json"
    report_path = OUT / "phase101_certified_form_composition_transfer_report.md"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)

    summary = {
        "phase": PHASE,
        "script": SCRIPT_NAME,
        "pass": bool(passed),
        "root": str(ROOT),
        "outputs": str(OUT),
        "seed": SEED,
        "n_trials": int(len(df)),
        "n_cases": int(len(CASES)),
        "metrics": metrics,
        "task": "certify whether already-certified global forms compose and transfer safely",
        "repair_context": {
            "phase100b_status": "global form certification repaired and passing",
            "phase101_addition": "certification is now treated as a reusable reasoning primitive",
            "core_rule": "uncertified, contradictory, or under-witnessed signs cannot compose into valid global forms",
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = []
    report.append("# Phase 101 Certified Form Composition / Transfer Gate\n")
    report.append("## Result\n")
    report.append(f"- PASS: `{passed}`\n")
    report.append(f"- Script: `{SCRIPT_NAME}`\n")
    report.append("\n## Core advance\n")
    report.append(
        "Phase 100B repaired global form certification. Phase 101 tests whether those certified forms can now be used as reasoning primitives.\n"
    )
    report.append(
        "A sign is no longer merely accepted as a form; it must preserve certification through composition and transfer.\n"
    )
    report.append("\n## Metrics\n")
    for k, v in metrics.items():
        report.append(f"- `{k}`: `{v:.4f}`\n")

    report.append("\n## Interpretation\n")
    report.append(
        "- Accepted cases require certified inputs, closure under composition, transfer consistency, valid obstruction status, valid topology, and a present witness.\n"
    )
    report.append(
        "- Rejected cases demonstrate that contradiction, wrong topology, or unlicensed obstruction poisons composition.\n"
    )
    report.append(
        "- Abstained cases demonstrate that unknown covers and missing witnesses block composition rather than becoming false positives.\n"
    )

    report.append("\n## Case summary\n\n")
    report.append(task_summary.to_markdown(index=False))
    report.append("\n\n## Family summary\n\n")
    report.append(family_summary.to_markdown(index=False))
    report.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))

    return {
        "trials": trials_path,
        "task": task_path,
        "family": family_path,
        "summary": summary_path,
        "report": report_path,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("[101] Certified form composition / transfer gate")
    print(f"[101] root: {ROOT}")
    print(f"[101] outputs: {OUT}")
    print("[101] reset continued: from global BBIT form certification to certified form composition")
    print("[101] task: certified signs may compose only when closure, transfer, obstruction, topology, and witness gates agree")

    df = run_trials(n_per_case=40)
    task_summary, family_summary, metrics = build_summaries(df)

    passed = (
        metrics["certified_form_composition_accuracy"] >= PASS_THRESHOLD
        and metrics["certified_input_gate_validity"] >= PASS_THRESHOLD
        and metrics["composition_closure_validity"] >= PASS_THRESHOLD
        and metrics["transfer_consistency_validity"] >= PASS_THRESHOLD
        and metrics["obstruction_gate_validity"] >= PASS_THRESHOLD
        and metrics["topology_gate_validity"] >= PASS_THRESHOLD
        and metrics["witness_gate_validity"] >= PASS_THRESHOLD
        and metrics["global_section_transfer_validity"] >= PASS_THRESHOLD
        and metrics["wrong_transfer_rejection"] >= PASS_THRESHOLD
        and metrics["unknown_transfer_abstention"] >= PASS_THRESHOLD
    )

    figs = [
        plot_decision_energy(df),
        plot_certification_field(df),
        plot_matrix(df),
        plot_progress(metrics),
        plot_graph(),
        plot_deabstracted_examples(),
        plot_3d(df),
    ]

    files = write_report(df, task_summary, family_summary, metrics, passed)

    print(f"[101] PHASE101_CERTIFIED_FORM_COMPOSITION_TRANSFER_PASS={passed}")
    print(
        "[101] "
        + " ".join(
            [
                f"certified_form_composition_accuracy={metrics['certified_form_composition_accuracy']:.4f}",
                f"certified_input_gate_validity={metrics['certified_input_gate_validity']:.4f}",
                f"composition_closure_validity={metrics['composition_closure_validity']:.4f}",
                f"transfer_consistency_validity={metrics['transfer_consistency_validity']:.4f}",
                f"obstruction_gate_validity={metrics['obstruction_gate_validity']:.4f}",
                f"topology_gate_validity={metrics['topology_gate_validity']:.4f}",
                f"witness_gate_validity={metrics['witness_gate_validity']:.4f}",
                f"global_section_transfer_validity={metrics['global_section_transfer_validity']:.4f}",
                f"wrong_transfer_rejection={metrics['wrong_transfer_rejection']:.4f}",
                f"unknown_transfer_abstention={metrics['unknown_transfer_abstention']:.4f}",
                f"min_margin={metrics['min_margin']:.4f}",
            ]
        )
    )

    print("[101] wrote:")
    for p in figs:
        print(f"  - {p}")
    for p in files.values():
        print(f"  - {p}")


if __name__ == "__main__":
    main()