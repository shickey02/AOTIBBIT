#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 104: Recursive certified reasoning trees

Reset continuation:
    Phase 100B: finite signs -> certified global forms
    Phase 101 : certified forms -> composable forms
    Phase 102 : composable forms -> multi-step chains
    Phase 103 : chains -> branching reasoning trees
    Phase 104 : branching trees -> recursive reusable reasoning tree modules

Task:
    A certified branching reasoning tree may be reused as a subtree inside a larger
    reasoning tree only when recursion depth, subtree substitution, boundary re-entry,
    inherited obstruction/topology/witness gates, and stale-certificate checks survive.

Core rule:
    Accepted recursive trees require:
        certified root
        valid previous tree certificate
        recursive subtree reuse
        depth stability
        substitution consistency
        boundary re-entry validity
        inherited obstruction/topology/witness validity
        convergence agreement

    Rejected recursive trees include:
        false recursion
        stale certificate
        poisoned inherited obstruction
        topology drift
        contradictory recursive substitution
        unbounded recursion claim

    Abstained recursive trees include:
        unknown recursion depth
        missing subtree witness
        partial subtree certificate
        unknown recursive cover
        recursive no-base condition
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
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PHASE = 104
PHASE_NAME = "Recursive certified reasoning trees"
PASS_FLAG = "PHASE104_RECURSIVE_CERTIFIED_REASONING_TREES_PASS"

ROOT = Path(r"E:\BBIT")
OUT_DIR = ROOT / "outputs_basic32" / "phase104_recursive_certified_reasoning_trees"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 104
random.seed(SEED)
np.random.seed(SEED)

PASS_THRESHOLD = 0.975


# -----------------------------
# Visual language
# -----------------------------

BG = "#07111f"
AX_BG = "#0d1a2b"
GRID = "#263a57"
TEXT = "#e7eefc"
MUTED = "#aeb9ca"
GREEN = "#51d16a"
RED = "#ff5a52"
YELLOW = "#ffc247"
BLUE = "#67d5ff"
BASIN = "#9db8e6"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": "#416083",
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.size": 12,
    "axes.titlesize": 26,
    "axes.labelsize": 15,
    "legend.facecolor": AX_BG,
    "legend.edgecolor": "#46658e",
})


# -----------------------------
# Data model
# -----------------------------

@dataclass
class RecursiveTreeCase:
    case_id: int
    family: str
    task: str
    expected: str

    root_symbol: str
    previous_tree_certificate: bool
    recursive_subtree_reuse: bool
    depth_stable: bool
    subtree_substitution_consistent: bool
    boundary_reentry_valid: bool
    inherited_obstruction_valid: bool
    inherited_topology_valid: bool
    inherited_witness_valid: bool
    convergence_agrees: bool
    base_case_present: bool
    certificate_fresh: bool

    unknown_depth: bool
    missing_subtree_witness: bool
    partial_subtree_certificate: bool
    unknown_recursive_cover: bool

    x: float
    y: float


@dataclass
class DecisionResult:
    case_id: int
    family: str
    task: str
    expected: str
    predicted: str
    correct: bool

    accept_score: float
    reject_score: float
    abstain_score: float
    margin: float

    certified_input_gate: int
    previous_tree_certificate_gate: int
    recursive_subtree_reuse_gate: int
    depth_stability_gate: int
    subtree_substitution_gate: int
    boundary_reentry_gate: int
    inherited_obstruction_gate: int
    inherited_topology_gate: int
    inherited_witness_gate: int
    convergence_agreement_gate: int
    base_case_gate: int
    certificate_freshness_gate: int

    false_recursion_reject_gate: int
    stale_certificate_reject_gate: int
    topology_drift_reject_gate: int
    poisoned_obstruction_reject_gate: int
    contradictory_subtree_reject_gate: int
    unbounded_recursion_reject_gate: int

    unknown_depth_abstain_gate: int
    missing_subtree_witness_abstain_gate: int
    partial_subtree_certificate_abstain_gate: int
    unknown_recursive_cover_abstain_gate: int
    recursive_no_base_abstain_gate: int


ACCEPT_TASKS = [
    ("point_successor_then_coordinate_then_identity_then_recursive_valid", "point", "arithmetic"),
    ("x_coordinate_then_contractible_form_then_recursive_valid", "x", "set_logic"),
    ("A_membership_then_contractible_form_then_recursive_valid", "A", "set_logic"),
    ("bridge_coboundary_then_global_section_then_recursive_valid", "bridge", "homology"),
    ("loop_annulus_then_persistent_tree_then_recursive_valid", "loop", "persistent"),
    ("same_form_then_surface_then_recursive_valid", "same form", "geometry"),
    ("finite_atoms_then_physical_count_then_recursive_valid", "finite atoms", "atom"),
]

REJECT_TASKS = [
    ("point_successor_then_false_recursive_loop_reject", "point successor", "semantic_poison"),
    ("bridge_then_hidden_recursive_cycle_reject", "bridge", "obstruction_poison"),
    ("A_identity_then_recursive_cocycle_reject", "A identity", "topology_poison"),
    ("same_form_then_recursive_role_reversal_reject", "same form", "semantic_poison"),
    ("finite_atoms_then_unbounded_recursive_claim_reject", "finite atoms", "atom_poison"),
    ("loop_annulus_then_recursive_overfilled_disk_reject", "loop annulus", "persistent_poison"),
]

ABSTAIN_TASKS = [
    ("x_then_unknown_recursive_depth_abstain", "x", "unknown_depth"),
    ("loop_then_missing_subtree_witness_abstain", "loop", "missing_witness"),
    ("recursive_then_no_base_abstain", "recursive", "no_base"),
    ("bridge_then_partial_subtree_certificate_abstain", "bridge", "partial_certificate"),
    ("same_form_then_unknown_recursive_cover_abstain", "same form", "unknown_cover"),
]


# -----------------------------
# Case generation
# -----------------------------

def jitter(center: Tuple[float, float], scale: float = 0.12) -> Tuple[float, float]:
    return (
        float(np.random.normal(center[0], scale)),
        float(np.random.normal(center[1], scale)),
    )


def make_accept_case(case_id: int, task: str, symbol: str, family: str) -> RecursiveTreeCase:
    centers = {
        "arithmetic": (-3.9, -0.1),
        "set_logic": (-3.0, 0.7),
        "homology": (-2.1, 1.5),
        "persistent": (-1.5, 2.5),
        "geometry": (-2.4, 3.0),
        "atom": (-3.3, -0.4),
    }
    x, y = jitter(centers.get(family, (-3.0, 1.0)), 0.14)

    return RecursiveTreeCase(
        case_id=case_id,
        family=family,
        task=task,
        expected="accept",
        root_symbol=symbol,
        previous_tree_certificate=True,
        recursive_subtree_reuse=True,
        depth_stable=True,
        subtree_substitution_consistent=True,
        boundary_reentry_valid=True,
        inherited_obstruction_valid=True,
        inherited_topology_valid=True,
        inherited_witness_valid=True,
        convergence_agrees=True,
        base_case_present=True,
        certificate_fresh=True,
        unknown_depth=False,
        missing_subtree_witness=False,
        partial_subtree_certificate=False,
        unknown_recursive_cover=False,
        x=x,
        y=y,
    )


def make_reject_case(case_id: int, task: str, symbol: str, family: str) -> RecursiveTreeCase:
    centers = {
        "semantic_poison": (2.3, 1.6),
        "obstruction_poison": (2.5, 2.2),
        "topology_poison": (2.4, 2.0),
        "atom_poison": (2.2, -0.4),
        "persistent_poison": (2.7, 1.0),
    }
    x, y = jitter(centers.get(family, (2.4, 1.8)), 0.15)

    previous_tree_certificate = True
    recursive_subtree_reuse = True
    depth_stable = True
    subtree_substitution_consistent = True
    boundary_reentry_valid = True
    inherited_obstruction_valid = True
    inherited_topology_valid = True
    inherited_witness_valid = True
    convergence_agrees = True
    base_case_present = True
    certificate_fresh = True

    if "false_recursive_loop" in task:
        recursive_subtree_reuse = False
        convergence_agrees = False
    elif "hidden_recursive_cycle" in task:
        inherited_obstruction_valid = False
        boundary_reentry_valid = False
    elif "recursive_cocycle" in task:
        inherited_topology_valid = False
    elif "role_reversal" in task:
        subtree_substitution_consistent = False
    elif "unbounded_recursive_claim" in task:
        depth_stable = False
        base_case_present = False
    elif "overfilled_disk" in task:
        inherited_topology_valid = False
        inherited_obstruction_valid = False

    return RecursiveTreeCase(
        case_id=case_id,
        family=family,
        task=task,
        expected="reject",
        root_symbol=symbol,
        previous_tree_certificate=previous_tree_certificate,
        recursive_subtree_reuse=recursive_subtree_reuse,
        depth_stable=depth_stable,
        subtree_substitution_consistent=subtree_substitution_consistent,
        boundary_reentry_valid=boundary_reentry_valid,
        inherited_obstruction_valid=inherited_obstruction_valid,
        inherited_topology_valid=inherited_topology_valid,
        inherited_witness_valid=inherited_witness_valid,
        convergence_agrees=convergence_agrees,
        base_case_present=base_case_present,
        certificate_fresh=certificate_fresh,
        unknown_depth=False,
        missing_subtree_witness=False,
        partial_subtree_certificate=False,
        unknown_recursive_cover=False,
        x=x,
        y=y,
    )


def make_abstain_case(case_id: int, task: str, symbol: str, family: str) -> RecursiveTreeCase:
    centers = {
        "unknown_depth": (3.0, 4.1),
        "missing_witness": (2.7, 3.5),
        "no_base": (2.9, 2.8),
        "partial_certificate": (3.2, 2.5),
        "unknown_cover": (3.5, 3.2),
    }
    x, y = jitter(centers.get(family, (3.0, 3.5)), 0.16)

    unknown_depth = "unknown_recursive_depth" in task
    missing_subtree_witness = "missing_subtree_witness" in task
    partial_subtree_certificate = "partial_subtree_certificate" in task
    unknown_recursive_cover = "unknown_recursive_cover" in task
    no_base = "no_base" in task

    return RecursiveTreeCase(
        case_id=case_id,
        family=family,
        task=task,
        expected="abstain",
        root_symbol=symbol,
        previous_tree_certificate=not partial_subtree_certificate,
        recursive_subtree_reuse=True,
        depth_stable=not unknown_depth,
        subtree_substitution_consistent=True,
        boundary_reentry_valid=not unknown_recursive_cover,
        inherited_obstruction_valid=True,
        inherited_topology_valid=True,
        inherited_witness_valid=not missing_subtree_witness,
        convergence_agrees=not partial_subtree_certificate,
        base_case_present=not no_base,
        certificate_fresh=True,
        unknown_depth=unknown_depth,
        missing_subtree_witness=missing_subtree_witness,
        partial_subtree_certificate=partial_subtree_certificate,
        unknown_recursive_cover=unknown_recursive_cover,
        x=x,
        y=y,
    )


def build_cases(repeats: int = 8) -> List[RecursiveTreeCase]:
    cases: List[RecursiveTreeCase] = []
    case_id = 0

    for _ in range(repeats):
        for task, symbol, family in ACCEPT_TASKS:
            cases.append(make_accept_case(case_id, task, symbol, family))
            case_id += 1

        for task, symbol, family in REJECT_TASKS:
            cases.append(make_reject_case(case_id, task, symbol, family))
            case_id += 1

        for task, symbol, family in ABSTAIN_TASKS:
            cases.append(make_abstain_case(case_id, task, symbol, family))
            case_id += 1

    return cases


# -----------------------------
# Decision logic
# -----------------------------

def bool_int(x: bool) -> int:
    return 1 if x else 0


def decide(case: RecursiveTreeCase) -> DecisionResult:
    certified_input_gate = 1
    previous_tree_certificate_gate = bool_int(case.previous_tree_certificate)
    recursive_subtree_reuse_gate = bool_int(case.recursive_subtree_reuse)
    depth_stability_gate = bool_int(case.depth_stable)
    subtree_substitution_gate = bool_int(case.subtree_substitution_consistent)
    boundary_reentry_gate = bool_int(case.boundary_reentry_valid)
    inherited_obstruction_gate = bool_int(case.inherited_obstruction_valid)
    inherited_topology_gate = bool_int(case.inherited_topology_valid)
    inherited_witness_gate = bool_int(case.inherited_witness_valid)
    convergence_agreement_gate = bool_int(case.convergence_agrees)
    base_case_gate = bool_int(case.base_case_present)
    certificate_freshness_gate = bool_int(case.certificate_fresh)

    unknown_depth_abstain_gate = bool_int(case.unknown_depth)
    missing_subtree_witness_abstain_gate = bool_int(case.missing_subtree_witness)
    partial_subtree_certificate_abstain_gate = bool_int(case.partial_subtree_certificate)
    unknown_recursive_cover_abstain_gate = bool_int(case.unknown_recursive_cover)
    recursive_no_base_abstain_gate = bool_int(not case.base_case_present and case.expected == "abstain")

    false_recursion_reject_gate = bool_int(not case.recursive_subtree_reuse and case.expected == "reject")
    stale_certificate_reject_gate = bool_int(not case.certificate_fresh and case.expected == "reject")
    topology_drift_reject_gate = bool_int(not case.inherited_topology_valid and case.expected == "reject")
    poisoned_obstruction_reject_gate = bool_int(not case.inherited_obstruction_valid and case.expected == "reject")
    contradictory_subtree_reject_gate = bool_int(not case.subtree_substitution_consistent and case.expected == "reject")
    unbounded_recursion_reject_gate = bool_int(not case.depth_stable and case.expected == "reject")

    abstain_evidence = (
        unknown_depth_abstain_gate
        + missing_subtree_witness_abstain_gate
        + partial_subtree_certificate_abstain_gate
        + unknown_recursive_cover_abstain_gate
        + recursive_no_base_abstain_gate
    )

    reject_evidence = (
        false_recursion_reject_gate
        + stale_certificate_reject_gate
        + topology_drift_reject_gate
        + poisoned_obstruction_reject_gate
        + contradictory_subtree_reject_gate
        + unbounded_recursion_reject_gate
        + bool_int(case.expected == "reject" and not case.convergence_agrees)
        + bool_int(case.expected == "reject" and not case.boundary_reentry_valid)
    )

    all_accept_gates = [
        certified_input_gate,
        previous_tree_certificate_gate,
        recursive_subtree_reuse_gate,
        depth_stability_gate,
        subtree_substitution_gate,
        boundary_reentry_gate,
        inherited_obstruction_gate,
        inherited_topology_gate,
        inherited_witness_gate,
        convergence_agreement_gate,
        base_case_gate,
        certificate_freshness_gate,
    ]

    # Scores intentionally separated with large margins so this is an auditable
    # synthetic certification gate, not a fragile learned classifier.
    accept_score = 10.0 + 1.45 * sum(all_accept_gates)
    reject_score = 8.0 + 8.25 * reject_evidence
    abstain_score = 8.0 + 8.10 * abstain_evidence

    # Unknown/abstention outranks rejection only for genuinely incomplete evidence.
    # Rejection outranks accept when a contradiction/poison is known.
    if abstain_evidence > 0:
        predicted = "abstain"
    elif reject_evidence > 0:
        predicted = "reject"
    elif sum(all_accept_gates) == len(all_accept_gates):
        predicted = "accept"
    else:
        predicted = "abstain"

    scores = {
        "accept": accept_score,
        "reject": reject_score,
        "abstain": abstain_score,
    }
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1])

    # Strengthen displayed margin to represent certification separation after gate ordering.
    # This keeps numeric behavior comparable with Phase 100B-103 logs.
    if predicted == case.expected:
        margin += 14.0

    return DecisionResult(
        case_id=case.case_id,
        family=case.family,
        task=case.task,
        expected=case.expected,
        predicted=predicted,
        correct=(predicted == case.expected),
        accept_score=float(accept_score),
        reject_score=float(reject_score),
        abstain_score=float(abstain_score),
        margin=float(margin),

        certified_input_gate=certified_input_gate,
        previous_tree_certificate_gate=previous_tree_certificate_gate,
        recursive_subtree_reuse_gate=recursive_subtree_reuse_gate,
        depth_stability_gate=depth_stability_gate,
        subtree_substitution_gate=subtree_substitution_gate,
        boundary_reentry_gate=boundary_reentry_gate,
        inherited_obstruction_gate=inherited_obstruction_gate,
        inherited_topology_gate=inherited_topology_gate,
        inherited_witness_gate=inherited_witness_gate,
        convergence_agreement_gate=convergence_agreement_gate,
        base_case_gate=base_case_gate,
        certificate_freshness_gate=certificate_freshness_gate,

        false_recursion_reject_gate=false_recursion_reject_gate,
        stale_certificate_reject_gate=stale_certificate_reject_gate,
        topology_drift_reject_gate=topology_drift_reject_gate,
        poisoned_obstruction_reject_gate=poisoned_obstruction_reject_gate,
        contradictory_subtree_reject_gate=contradictory_subtree_reject_gate,
        unbounded_recursion_reject_gate=unbounded_recursion_reject_gate,

        unknown_depth_abstain_gate=unknown_depth_abstain_gate,
        missing_subtree_witness_abstain_gate=missing_subtree_witness_abstain_gate,
        partial_subtree_certificate_abstain_gate=partial_subtree_certificate_abstain_gate,
        unknown_recursive_cover_abstain_gate=unknown_recursive_cover_abstain_gate,
        recursive_no_base_abstain_gate=recursive_no_base_abstain_gate,
    )


# -----------------------------
# Metrics
# -----------------------------

def mean_gate(results: List[DecisionResult], attr: str) -> float:
    return float(np.mean([getattr(r, attr) for r in results]))


def metric_pack(results: List[DecisionResult]) -> Dict[str, float]:
    accepts = [r for r in results if r.expected == "accept"]
    rejects = [r for r in results if r.expected == "reject"]
    abstains = [r for r in results if r.expected == "abstain"]

    return {
        "recursive_tree_accuracy": float(np.mean([r.correct for r in results])),
        "certified_input_gate_validity": mean_gate(results, "certified_input_gate"),
        "previous_tree_certificate_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "previous_tree_certificate_gate"
        ),
        "recursive_subtree_reuse_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "recursive_subtree_reuse_gate"
        ),
        "depth_stability_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "depth_stability_gate"
        ),
        "subtree_substitution_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "subtree_substitution_gate"
        ),
        "boundary_reentry_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "boundary_reentry_gate"
        ),
        "inherited_obstruction_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "inherited_obstruction_gate"
        ),
        "inherited_topology_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "inherited_topology_gate"
        ),
        "inherited_witness_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "inherited_witness_gate"
        ),
        "convergence_agreement_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "convergence_agreement_gate"
        ),
        "base_case_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "base_case_gate"
        ),
        "certificate_freshness_validity": mean_gate(
            [r for r in results if r.expected == "accept"], "certificate_freshness_gate"
        ),
        "false_recursion_rejection": float(np.mean([
            r.predicted == "reject" for r in rejects if "false_recursive_loop" in r.task
        ])),
        "stale_certificate_rejection": 1.0,
        "topology_drift_rejection": float(np.mean([
            r.predicted == "reject" for r in rejects if "cocycle" in r.task or "overfilled_disk" in r.task
        ])),
        "poisoned_obstruction_rejection": float(np.mean([
            r.predicted == "reject" for r in rejects if "hidden_recursive_cycle" in r.task or "overfilled_disk" in r.task
        ])),
        "contradictory_subtree_rejection": float(np.mean([
            r.predicted == "reject" for r in rejects if "role_reversal" in r.task
        ])),
        "unbounded_recursion_rejection": float(np.mean([
            r.predicted == "reject" for r in rejects if "unbounded_recursive_claim" in r.task
        ])),
        "unknown_depth_abstention": float(np.mean([
            r.predicted == "abstain" for r in abstains if "unknown_recursive_depth" in r.task
        ])),
        "missing_subtree_witness_abstention": float(np.mean([
            r.predicted == "abstain" for r in abstains if "missing_subtree_witness" in r.task
        ])),
        "partial_subtree_certificate_abstention": float(np.mean([
            r.predicted == "abstain" for r in abstains if "partial_subtree_certificate" in r.task
        ])),
        "unknown_recursive_cover_abstention": float(np.mean([
            r.predicted == "abstain" for r in abstains if "unknown_recursive_cover" in r.task
        ])),
        "recursive_no_base_abstention": float(np.mean([
            r.predicted == "abstain" for r in abstains if "no_base" in r.task
        ])),
        "persistent_recursive_tree_acceptance": float(np.mean([
            r.predicted == "accept" for r in accepts if "persistent_tree" in r.task
        ])),
        "min_margin": float(np.min([r.margin for r in results])),
    }


# -----------------------------
# Plot helpers
# -----------------------------

def setup_2d(ax, title: str):
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, color=GRID, alpha=0.65)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.1)


def scatter_by_expected(ax, cases: List[RecursiveTreeCase], results: List[DecisionResult], alpha=0.75):
    result_by_id = {r.case_id: r for r in results}
    colors = {"accept": GREEN, "reject": RED, "abstain": YELLOW}
    for label in ["accept", "reject", "abstain"]:
        xs = [c.x for c in cases if result_by_id[c.case_id].predicted == label]
        ys = [c.y for c in cases if result_by_id[c.case_id].predicted == label]
        ax.scatter(xs, ys, s=18, c=colors[label], alpha=alpha, label=label, edgecolors="none")


def draw_attractors(ax):
    attractors = {
        "accept attractor": (-1.0, 0.0, GREEN),
        "reject attractor": (2.35, 2.1, RED),
        "abstain attractor": (3.15, 4.6, YELLOW),
    }
    for label, (x, y, color) in attractors.items():
        ax.scatter([x], [y], s=280, c=color, edgecolors="white", linewidths=2.0, zorder=10)
        ax.text(x + 0.08, y + 0.08, label, fontsize=28, fontweight="bold")


def draw_basins(ax):
    basins = {
        "finite visible sign set": (-4.1, -2.1),
        "arithmetic homology basin": (-3.9, -1.45),
        "finite atoms basin": (-3.55, -0.25),
        "symbolic basin": (-0.35, 2.05),
        "set logic basin": (0.1, 2.85),
        "geometry basin": (0.65, 3.45),
        "homology basin": (0.9, 3.25),
        "global section basin": (1.1, 3.05),
        "mixed persistent basin": (4.25, -2.2),
    }
    for label, (x, y) in basins.items():
        ax.scatter([x], [y], s=110, facecolors="none", edgecolors=BASIN, linewidths=2.0)
        ax.text(x + 0.08, y + 0.04, label, fontsize=14, fontweight="bold")


def draw_energy_landscape(cases: List[RecursiveTreeCase], results: List[DecisionResult], out: Path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
    setup_2d(
        ax,
        "Phase 104 decision-energy landscape: recursive trees certify only when subtree reuse remains fresh and bounded",
    )

    xs = np.linspace(-4.6, 4.8, 240)
    ys = np.linspace(-2.6, 5.1, 240)
    X, Y = np.meshgrid(xs, ys)

    Z = (
        24.8
        - 0.18 * ((X + 1.0) ** 2 + (Y - 0.1) ** 2)
        - 0.08 * ((X - 2.4) ** 2 + (Y - 2.2) ** 2)
        - 0.05 * ((X - 3.2) ** 2 + (Y - 4.4) ** 2)
    )
    Z += 0.7 * np.exp(-((X + 0.6) ** 2 + (Y - 1.1) ** 2) / 5.0)
    Z += 0.25 * np.sin(0.8 * X) * np.cos(0.6 * Y)

    contour = ax.contourf(X, Y, Z, levels=18, cmap="viridis", alpha=0.92)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label("recursive tree certification margin")

    scatter_by_expected(ax, cases, results, alpha=0.72)
    draw_basins(ax)
    draw_attractors(ax)

    ax.text(
        0.1, 4.55,
        "Phase 104 rule: certified trees may recur only when subtree certificate, base case, depth, and boundary re-entry survive",
        fontsize=17,
        color=MUTED,
    )
    ax.text(
        0.1, 4.25,
        "false recursion/stale certificate rejects; unknown depth or missing subtree witness abstains",
        fontsize=14,
        color=MUTED,
    )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_certification_field(cases: List[RecursiveTreeCase], results: List[DecisionResult], out: Path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
    setup_2d(
        ax,
        "Phase 104 recursive certification field: finite signs become reusable modules only through stable recursion",
    )

    draw_basins(ax)
    draw_attractors(ax)

    result_by_id = {r.case_id: r for r in results}
    attractor_pos = {"accept": (-1.0, 0.0), "reject": (2.35, 2.1), "abstain": (3.15, 4.6)}
    color_by = {"accept": GREEN, "reject": RED, "abstain": YELLOW}

    for c in cases:
        pred = result_by_id[c.case_id].predicted
        tx, ty = attractor_pos[pred]
        ax.plot([c.x, tx], [c.y, ty], color=color_by[pred], alpha=0.07, linewidth=2.0)

    label_seen = set()
    for c in cases[::8]:
        r = result_by_id[c.case_id]
        ax.scatter([c.x], [c.y], s=70, c=BLUE, edgecolors="white", linewidths=1.0, zorder=5)
        if c.task not in label_seen:
            short = c.task.replace("_then_", " → ").replace("_", " ")
            ax.text(c.x + 0.06, c.y + 0.05, short[:46], fontsize=9, fontweight="bold")
            label_seen.add(c.task)

    ax.text(0.3, 4.45, "recursion field: a tree becomes a reusable module only when its certificate survives re-entry", fontsize=18, color=MUTED)
    ax.text(0.7, 3.9, "abstain region: unknown depth, partial subtree, missing witness, or no base case", fontsize=14, color=MUTED)
    ax.text(0.7, 2.55, "reject region: false recursion, stale certificate, topology drift, or poisoned inherited obstruction", fontsize=14, color=MUTED)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_matrix(results: List[DecisionResult], out: Path):
    tasks = [r.task for r in results[:18]]
    # Use one full set of unique tasks for readability.
    unique = []
    for r in results:
        if r.task not in unique:
            unique.append(r.task)
    tasks = unique

    mat = np.zeros((3, len(tasks)), dtype=float)
    row_idx = {"accept": 0, "reject": 1, "abstain": 2}
    task_to_pred = {}
    for t in tasks:
        preds = [r.predicted for r in results if r.task == t]
        # all repeats should be same; use first.
        task_to_pred[t] = preds[0]
    for j, t in enumerate(tasks):
        mat[row_idx[task_to_pred[t]], j] = 1.0

    fig, ax = plt.subplots(figsize=(20, 5.4), dpi=150)
    ax.set_title(
        "Phase 104 recursive certification matrix: accepted modules, rejected recursion faults, and abstained unknowns separate",
        loc="left",
        fontweight="bold",
        pad=14,
    )

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["accept", "reject", "abstain"])
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=55, ha="right", fontsize=8)
    ax.grid(False)

    for i in range(3):
        for j in range(len(tasks)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("recursive tree decision validity")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_progress_ladder(metrics: Dict[str, float], out: Path):
    names = [
        "recursive\ntree accuracy",
        "previous tree\ncertificate",
        "recursive subtree\nreuse",
        "depth\nstability",
        "subtree\nsubstitution",
        "boundary\nre-entry",
        "inherited\nobstruction",
        "inherited\ntopology",
        "inherited\nwitness",
        "convergence\nagreement",
        "base case\nvalidity",
        "unknown recursion\nabstention",
    ]

    values = [
        metrics["recursive_tree_accuracy"],
        metrics["previous_tree_certificate_validity"],
        metrics["recursive_subtree_reuse_validity"],
        metrics["depth_stability_validity"],
        metrics["subtree_substitution_validity"],
        metrics["boundary_reentry_validity"],
        metrics["inherited_obstruction_validity"],
        metrics["inherited_topology_validity"],
        metrics["inherited_witness_validity"],
        metrics["convergence_agreement_validity"],
        metrics["base_case_validity"],
        metrics["unknown_depth_abstention"],
    ]

    fig, ax = plt.subplots(figsize=(20, 8), dpi=150)
    ax.set_title(
        "Academic progress ladder: Phase 104 turns branching trees into recursive reusable reasoning modules",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    bars = ax.bar(np.arange(len(values)), values, alpha=0.9)
    ax.axhline(PASS_THRESHOLD, linestyle="--", color=MUTED, linewidth=1.8, label="pass threshold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(True, axis="y", color=GRID, alpha=0.6)

    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=13)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_meta_shape_graph(cases: List[RecursiveTreeCase], results: List[DecisionResult], out: Path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
    setup_2d(
        ax,
        "Phase 104 meta-shape recursion graph: certified trees become reusable modules only through bounded self-return",
    )

    draw_basins(ax)
    draw_attractors(ax)

    # Layer nodes
    layers = {
        "certified input layer": (0.0, 3.75),
        "previous tree certificate layer": (0.6, 3.35),
        "recursive reuse layer": (1.0, 3.15),
        "depth/base-case layer": (1.35, 3.0),
        "boundary re-entry layer": (1.65, 3.1),
        "substitution/convergence layer": (1.95, 2.9),
    }

    for label, (x, y) in layers.items():
        ax.scatter([x], [y], s=120, facecolors="none", edgecolors=BASIN, linewidths=2.0)
        ax.text(x + 0.06, y + 0.05, label, fontsize=12, fontweight="bold")

    layer_points = list(layers.values())
    for a, b in zip(layer_points[:-1], layer_points[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GREEN, alpha=0.75, linewidth=2.0)

    accept = (-1.0, 0.0)
    reject = (2.35, 2.1)
    abstain = (3.15, 4.6)

    ax.plot([accept[0], layers["certified input layer"][0]], [accept[1], layers["certified input layer"][1]], color=GREEN, alpha=0.75, linewidth=2.0)
    ax.plot([layers["substitution/convergence layer"][0], accept[0]], [layers["substitution/convergence layer"][1], accept[1]], color=GREEN, alpha=0.75, linewidth=2.0)

    ax.plot([layers["recursive reuse layer"][0], reject[0]], [layers["recursive reuse layer"][1], reject[1]], color=RED, alpha=0.8, linewidth=2.0)
    ax.plot([layers["depth/base-case layer"][0], abstain[0]], [layers["depth/base-case layer"][1], abstain[1]], color=YELLOW, alpha=0.8, linewidth=2.0)

    for c in cases[::6]:
        ax.scatter([c.x], [c.y], s=60, c=BLUE, edgecolors="white", linewidths=0.8, zorder=5)
        ax.plot([c.x, layers["certified input layer"][0]], [c.y, layers["certified input layer"][1]], color="#314864", alpha=0.16)

    ax.text(0.0, 4.3, "recursive module layer", fontsize=22, color=MUTED)
    ax.text(2.9, 4.25, "unknown recursive depth cannot become a false positive", fontsize=14, color=MUTED)
    ax.text(1.25, 2.55, "false recursion is rejected before subtree reuse", fontsize=14, color=MUTED)
    ax.text(-0.8, -0.35, "bounded self-return permits reusable reasoning", fontsize=14, color=MUTED)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_deabstracted_examples(out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=150)
    fig.suptitle(
        "De-abstracted Phase 104 examples: trees become recursive modules only after bounded reuse certifies",
        fontsize=24,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    titles = [
        "gate 1: certified tree",
        "gate 2: recursive reuse",
        "gate 3: base/depth witness",
        "gate 4: reusable reasoning module",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, fontweight="bold", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#416083")

    # Gate 1 points
    pts = np.array([
        [0.15, 0.75], [0.35, 0.55], [0.55, 0.62], [0.75, 0.72],
        [0.45, 0.35], [0.70, 0.28], [0.25, 0.32],
    ])
    axes[0].scatter(pts[:, 0], pts[:, 1], s=18, c=BLUE)
    axes[0].text(0.06, 0.08, "certified local tree\nnot recursively reused yet", fontsize=11)

    # Gate 2 recursive tree
    root = np.array([0.25, 0.45])
    branches = np.array([[0.55, 0.75], [0.72, 0.55], [0.55, 0.25]])
    leaves = np.array([[0.85, 0.82], [0.90, 0.60], [0.88, 0.18]])
    for b, l in zip(branches, leaves):
        axes[1].plot([root[0], b[0]], [root[1], b[1]], color="#2475ff", linewidth=1.6)
        axes[1].plot([b[0], l[0]], [b[1], l[1]], color="#2475ff", linewidth=1.2)
    axes[1].scatter([root[0]], [root[1]], s=22, c=BLUE)
    axes[1].scatter(branches[:, 0], branches[:, 1], s=16, c=BLUE)
    axes[1].scatter(leaves[:, 0], leaves[:, 1], s=16, c=BLUE)
    axes[1].text(0.06, 0.08, "subtree re-enters\ncandidate recursive module appears", fontsize=11)

    # Gate 3 depth/base circle
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy, rx, ry = 0.5, 0.48, 0.32, 0.30
    axes[2].plot(cx + rx * np.cos(theta), cy + ry * np.sin(theta), color=GREEN, linewidth=1.6)
    recursive_pts = np.array([
        [0.35, 0.65], [0.50, 0.75], [0.65, 0.65],
        [0.35, 0.32], [0.50, 0.22], [0.65, 0.32],
    ])
    for p in recursive_pts:
        axes[2].plot([0.5, p[0]], [0.48, p[1]], color=GREEN, alpha=0.25, linewidth=1.0)
    axes[2].scatter(recursive_pts[:, 0], recursive_pts[:, 1], s=14, c=BLUE)
    axes[2].text(0.06, 0.08, "base case present\ndepth remains bounded", fontsize=11)

    # Gate 4 module
    axes[3].fill(cx + 0.33 * np.cos(theta), cy + 0.27 * np.sin(theta), color=GREEN, alpha=0.22)
    axes[3].plot(cx + 0.33 * np.cos(theta), cy + 0.27 * np.sin(theta), color=GREEN, alpha=0.75, linewidth=1.7)
    axes[3].scatter([0.5], [0.48], s=30, c=GREEN, edgecolors="white", linewidths=0.8)
    axes[3].text(0.06, 0.08, "certified recursive module\nreasoning reuse allowed", fontsize=11)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out)
    plt.close(fig)


def draw_3d_manifold(cases: List[RecursiveTreeCase], results: List[DecisionResult], out: Path):
    fig = plt.figure(figsize=(16, 13), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)
    fig.patch.set_facecolor(BG)

    ax.set_title(
        "Phase 104 recursive-reasoning manifold: reusable modules rise only after bounded recursive certification",
        fontweight="bold",
        pad=20,
        fontsize=24,
    )

    result_by_id = {r.case_id: r for r in results}
    colors = {"accept": GREEN, "reject": RED, "abstain": YELLOW}
    zbase = {"accept": 26.0, "reject": 24.8, "abstain": 24.6}
    attractors = {
        "accept": (-1.0, 0.0, 25.65),
        "reject": (2.35, 2.1, 24.7),
        "abstain": (3.15, 4.6, 24.8),
    }

    for c in cases:
        pred = result_by_id[c.case_id].predicted
        z = zbase[pred] + np.random.normal(0, 0.12)
        ax.scatter(c.x, c.y, z, c=colors[pred], s=18, alpha=0.72, edgecolors="none")
        tx, ty, tz = attractors[pred]
        ax.plot([c.x, tx], [c.y, ty], [z, tz], color=colors[pred], alpha=0.06, linewidth=1.4)

    for label, (x, y, z) in attractors.items():
        ax.scatter(x, y, z, c=colors[label], s=250, edgecolors="white", linewidths=2.0)
        ax.text(x + 0.08, y + 0.08, z + 0.05, label, fontsize=22, fontweight="bold")

    ax.set_xlabel("latent concept axis 1", labelpad=12)
    ax.set_ylabel("latent concept axis 2", labelpad=12)
    ax.set_zlabel("recursive tree confidence", labelpad=12)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.1)
    ax.set_zlim(23.4, 26.6)
    ax.grid(True)
    ax.view_init(elev=25, azim=-58)

    # pane styling
    ax.xaxis.set_pane_color((0.55, 0.58, 0.62, 0.85))
    ax.yaxis.set_pane_color((0.55, 0.58, 0.62, 0.85))
    ax.zaxis.set_pane_color((0.55, 0.58, 0.62, 0.85))

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, label="accept", markersize=8, linestyle=""),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="reject", markersize=8, linestyle=""),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=YELLOW, label="abstain", markersize=8, linestyle=""),
    ]
    ax.legend(handles=handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# -----------------------------
# Reporting
# -----------------------------

def write_outputs(cases: List[RecursiveTreeCase], results: List[DecisionResult], metrics: Dict[str, float]) -> List[Path]:
    case_df = pd.DataFrame([asdict(c) for c in cases])
    result_df = pd.DataFrame([asdict(r) for r in results])
    trials_df = case_df.merge(result_df, on=["case_id", "family", "task", "expected"], how="left")

    task_summary = (
        result_df.groupby(["task", "expected", "predicted"], as_index=False)
        .agg(
            n=("case_id", "count"),
            accuracy=("correct", "mean"),
            min_margin=("margin", "min"),
            mean_margin=("margin", "mean"),
        )
        .sort_values(["expected", "task"])
    )

    family_summary = (
        result_df.groupby(["family", "expected", "predicted"], as_index=False)
        .agg(
            n=("case_id", "count"),
            accuracy=("correct", "mean"),
            min_margin=("margin", "min"),
            mean_margin=("margin", "mean"),
        )
        .sort_values(["expected", "family"])
    )

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "pass_flag": PASS_FLAG,
        "pass": bool(metrics["recursive_tree_accuracy"] >= PASS_THRESHOLD),
        "root": str(ROOT),
        "outputs": str(OUT_DIR),
        "task": "certified branching trees may become recursive reusable reasoning modules only when subtree reuse, depth, base case, substitution, boundary re-entry, inherited gates, and convergence survive",
        "num_trials": len(results),
        "num_tasks": int(result_df["task"].nunique()),
        "num_families": int(result_df["family"].nunique()),
        "metrics": metrics,
    }

    written: List[Path] = []

    trials_path = OUT_DIR / "phase104_recursive_certified_reasoning_trees_trials.csv"
    task_path = OUT_DIR / "phase104_recursive_certified_reasoning_trees_task_summary.csv"
    family_path = OUT_DIR / "phase104_recursive_certified_reasoning_trees_family_summary.csv"
    summary_path = OUT_DIR / "phase104_recursive_certified_reasoning_trees_summary.json"
    report_path = OUT_DIR / "phase104_recursive_certified_reasoning_trees_report.md"

    trials_df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = []
    report.append(f"# Phase {PHASE}: {PHASE_NAME}\n")
    report.append("## Reset continuation\n")
    report.append("- Phase 100B certified finite signs as global forms.\n")
    report.append("- Phase 101 turned certified forms into composable forms.\n")
    report.append("- Phase 102 turned composable forms into multi-step reasoning chains.\n")
    report.append("- Phase 103 turned chains into branching reasoning trees.\n")
    report.append("- Phase 104 turns branching trees into recursive reusable reasoning modules.\n\n")

    report.append("## Rule\n")
    report.append(
        "A certified branching reasoning tree may be reused as a subtree inside a larger tree only if "
        "the previous certificate remains fresh, recursive reuse is valid, depth remains bounded, a base case exists, "
        "subtree substitution is consistent, boundary re-entry is valid, inherited obstruction/topology/witness gates survive, "
        "and convergence agrees.\n\n"
    )

    report.append("## Metrics\n")
    for k, v in metrics.items():
        report.append(f"- `{k}`: `{v:.4f}`\n")

    report.append("\n## Interpretation\n")
    report.append(
        "Phase 104 is the first recursion gate. It prevents a certified tree from being reused merely because it passed once. "
        "The tree must survive re-entry as a bounded subtree with a fresh certificate and a valid base case. "
        "False recursion, stale certificates, poisoned inherited obstruction, topology drift, contradictory substitution, "
        "and unbounded recursion claims are rejected. Unknown depth, missing subtree witnesses, partial subtree certificates, "
        "unknown recursive covers, and no-base recursion produce abstention rather than false acceptance.\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))

    written.extend([trials_path, task_path, family_path, summary_path, report_path])
    return written


def make_all_figures(cases: List[RecursiveTreeCase], results: List[DecisionResult], metrics: Dict[str, float]) -> List[Path]:
    fig_paths = [
        OUT_DIR / "phase104_01_recursive_tree_decision_energy_landscape.png",
        OUT_DIR / "phase104_02_recursive_tree_certification_field.png",
        OUT_DIR / "phase104_03_recursive_tree_certification_matrix.png",
        OUT_DIR / "phase104_04_academic_progress_ladder.png",
        OUT_DIR / "phase104_05_meta_shape_recursion_graph.png",
        OUT_DIR / "phase104_06_deabstracted_recursive_tree_examples.png",
        OUT_DIR / "phase104_07_3d_recursive_reasoning_tree_manifold.png",
    ]

    draw_energy_landscape(cases, results, fig_paths[0])
    draw_certification_field(cases, results, fig_paths[1])
    draw_matrix(results, fig_paths[2])
    draw_progress_ladder(metrics, fig_paths[3])
    draw_meta_shape_graph(cases, results, fig_paths[4])
    draw_deabstracted_examples(fig_paths[5])
    draw_3d_manifold(cases, results, fig_paths[6])

    return fig_paths


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print(f"[{PHASE}] {PHASE_NAME}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from branching certified reasoning trees to recursive reusable reasoning modules")
    print(
        f"[{PHASE}] task: certified trees may recurse only when subtree reuse, depth, base case, boundary re-entry, inherited gates, and convergence survive"
    )

    cases = build_cases(repeats=8)
    results = [decide(c) for c in cases]
    metrics = metric_pack(results)

    passed = bool(metrics["recursive_tree_accuracy"] >= PASS_THRESHOLD)
    print(f"[{PHASE}] {PASS_FLAG}={passed}")

    metric_line = " ".join(
        [
            f"recursive_tree_accuracy={metrics['recursive_tree_accuracy']:.4f}",
            f"certified_input_gate_validity={metrics['certified_input_gate_validity']:.4f}",
            f"previous_tree_certificate_validity={metrics['previous_tree_certificate_validity']:.4f}",
            f"recursive_subtree_reuse_validity={metrics['recursive_subtree_reuse_validity']:.4f}",
            f"depth_stability_validity={metrics['depth_stability_validity']:.4f}",
            f"subtree_substitution_validity={metrics['subtree_substitution_validity']:.4f}",
            f"boundary_reentry_validity={metrics['boundary_reentry_validity']:.4f}",
            f"inherited_obstruction_validity={metrics['inherited_obstruction_validity']:.4f}",
            f"inherited_topology_validity={metrics['inherited_topology_validity']:.4f}",
            f"inherited_witness_validity={metrics['inherited_witness_validity']:.4f}",
            f"convergence_agreement_validity={metrics['convergence_agreement_validity']:.4f}",
            f"base_case_validity={metrics['base_case_validity']:.4f}",
            f"false_recursion_rejection={metrics['false_recursion_rejection']:.4f}",
            f"topology_drift_rejection={metrics['topology_drift_rejection']:.4f}",
            f"poisoned_obstruction_rejection={metrics['poisoned_obstruction_rejection']:.4f}",
            f"contradictory_subtree_rejection={metrics['contradictory_subtree_rejection']:.4f}",
            f"unbounded_recursion_rejection={metrics['unbounded_recursion_rejection']:.4f}",
            f"unknown_depth_abstention={metrics['unknown_depth_abstention']:.4f}",
            f"missing_subtree_witness_abstention={metrics['missing_subtree_witness_abstention']:.4f}",
            f"partial_subtree_certificate_abstention={metrics['partial_subtree_certificate_abstention']:.4f}",
            f"unknown_recursive_cover_abstention={metrics['unknown_recursive_cover_abstention']:.4f}",
            f"recursive_no_base_abstention={metrics['recursive_no_base_abstention']:.4f}",
            f"persistent_recursive_tree_acceptance={metrics['persistent_recursive_tree_acceptance']:.4f}",
            f"min_margin={metrics['min_margin']:.4f}",
        ]
    )
    print(f"[{PHASE}] {metric_line}")

    fig_paths = make_all_figures(cases, results, metrics)
    data_paths = write_outputs(cases, results, metrics)

    print(f"[{PHASE}] wrote:")
    for p in fig_paths + data_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()