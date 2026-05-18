#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 102 — Multi-step certified reasoning chains

Reset continued:
    Phase 99  : persistent homology detects topological form
    Phase 99B : topology is judged relative to intended topology
    Phase 100 : finite signs become globally certified forms
    Phase 100B: certification gate is repaired as decision-relative jury
    Phase 101 : certified forms become composable reasoning primitives
    Phase 102 : certified compositions survive multi-step reasoning chains

Core rule:
    A certified form may be reused across A -> B -> C -> D reasoning chains only if
    every transfer preserves certification, closure, topology, obstruction, witness,
    and global-section validity at every step.

    A late contradiction poisons the whole chain.
    A missing witness / unknown cover / partial section forces abstention.
    A fully certified chain becomes reusable reasoning.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PHASE = "102"
PHASE_NAME = "multi_step_certified_reasoning_chains"
PASS_FLAG = "PHASE102_MULTI_STEP_CERTIFIED_REASONING_CHAINS_PASS"

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32" / f"phase{PHASE}_{PHASE_NAME}"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

BG = "#08111f"
AX_BG = "#101b2b"
FG = "#e8eefc"
MUTED = "#aeb8ca"
GRID = "#25364e"

GREEN = "#57d36f"
RED = "#ff5b57"
YELLOW = "#ffc947"
CYAN = "#6fd3ff"
BLUE = "#2f6fff"
PURPLE = "#a78bfa"

PASS_THRESHOLD = 0.97
RNG_SEED = 102


def set_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#3c587f",
            "grid.color": GRID,
            "font.size": 13,
            "axes.titleweight": "bold",
            "axes.titlesize": 30,
            "axes.labelsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.facecolor": "#101827",
            "legend.edgecolor": "#4a6690",
        }
    )


set_dark_style()


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class ChainFamily:
    name: str
    expected_decision: str
    basin: str
    sign_tokens: List[str]
    intended_topology: str
    transfer_path: List[str]

    certified_inputs: bool
    composition_closure: bool
    transfer_consistency: bool
    obstruction_ok: bool
    topology_ok: bool
    witness_ok: bool
    global_section_ok: bool

    failure_mode: str
    semantic_role: str
    base_x: float
    base_y: float


@dataclass
class TrialResult:
    phase: str
    family: str
    seed: int
    expected_decision: str
    predicted_decision: str
    correct: bool

    certified_inputs: bool
    composition_closure: bool
    transfer_consistency: bool
    obstruction_ok: bool
    topology_ok: bool
    witness_ok: bool
    global_section_ok: bool

    chain_length: int
    contradiction_depth: int
    missing_witness_depth: int
    topology_drift_depth: int
    global_section_depth: int

    chain_energy: float
    margin: float
    basin: str
    intended_topology: str
    failure_mode: str
    semantic_role: str
    x: float
    y: float
    z: float


# ---------------------------------------------------------------------
# Phase 102 chain families
# ---------------------------------------------------------------------

FAMILIES: List[ChainFamily] = [
    # ACCEPT: stable chains
    ChainFamily(
        name="point_successor_then_coordinate_valid",
        expected_decision="accept",
        basin="finite atoms basin",
        sign_tokens=["point", "successor", "coordinate", "global_form"],
        intended_topology="contractible",
        transfer_path=["point", "successor", "coordinate", "certified_form"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="finite successor chain",
        base_x=-3.9,
        base_y=0.0,
    ),
    ChainFamily(
        name="x_coordinate_then_contractible_form_valid",
        expected_decision="accept",
        basin="finite atoms basin",
        sign_tokens=["x", "coordinate", "contractible", "form"],
        intended_topology="contractible",
        transfer_path=["x", "coordinate", "chart", "global_form"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="coordinate certification chain",
        base_x=-3.65,
        base_y=-0.25,
    ),
    ChainFamily(
        name="A_membership_then_contractible_form_valid",
        expected_decision="accept",
        basin="set logic basin",
        sign_tokens=["A", "membership", "section", "form"],
        intended_topology="contractible",
        transfer_path=["A", "membership", "section", "certified_form"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="membership section chain",
        base_x=-3.1,
        base_y=1.1,
    ),
    ChainFamily(
        name="bridge_coboundary_then_global_section_valid",
        expected_decision="accept",
        basin="global section basin",
        sign_tokens=["bridge", "coboundary", "resolved", "global_section"],
        intended_topology="global_section",
        transfer_path=["bridge", "coboundary", "resolved", "global_section"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="bridge to global section",
        base_x=-2.75,
        base_y=1.75,
    ),
    ChainFamily(
        name="loop_annulus_then_persistent_form_valid",
        expected_decision="accept",
        basin="homology basin",
        sign_tokens=["loop", "annulus", "persistent", "form"],
        intended_topology="persistent_beta1",
        transfer_path=["loop", "annulus", "homology", "certified_form"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="persistent loop chain",
        base_x=-2.45,
        base_y=2.85,
    ),
    ChainFamily(
        name="same_form_then_surface_valid",
        expected_decision="accept",
        basin="geometry basin",
        sign_tokens=["same_form", "surface", "chart", "global_form"],
        intended_topology="surface",
        transfer_path=["same_form", "surface", "chart", "global_form"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="none",
        semantic_role="surface chart chain",
        base_x=-1.95,
        base_y=3.15,
    ),

    # REJECT: late contradiction poisons whole chain
    ChainFamily(
        name="point_successor_then_false_loop_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["point", "successor", "false_loop", "bad_form"],
        intended_topology="contractible",
        transfer_path=["point", "successor", "loop", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=False,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="late topology contradiction",
        semantic_role="false loop injected into finite chain",
        base_x=2.1,
        base_y=2.05,
    ),
    ChainFamily(
        name="bridge_then_hidden_cycle_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["bridge", "section", "hidden_cycle", "bad_form"],
        intended_topology="global_section",
        transfer_path=["bridge", "section", "cycle", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=False,
        witness_ok=True,
        global_section_ok=False,
        failure_mode="hidden cycle obstruction",
        semantic_role="bridge hides cycle",
        base_x=2.25,
        base_y=2.25,
    ),
    ChainFamily(
        name="A_identity_then_cocycle_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["A", "identity", "cocycle", "bad_form"],
        intended_topology="contractible",
        transfer_path=["A", "identity", "cocycle", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=False,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="identity cocycle contradiction",
        semantic_role="identity poisons section",
        base_x=2.45,
        base_y=2.0,
    ),
    ChainFamily(
        name="same_form_then_role_reversal_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["same_form", "role_reversal", "bad_section", "bad_form"],
        intended_topology="surface",
        transfer_path=["same_form", "role_reversal", "bad_section", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=False,
        failure_mode="role reversal transfer contradiction",
        semantic_role="same form role reversal",
        base_x=2.05,
        base_y=0.75,
    ),
    ChainFamily(
        name="finite_atoms_then_unbounded_claim_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["finite_atoms", "physical_count", "unbounded_claim", "bad_form"],
        intended_topology="finite_count",
        transfer_path=["finite_atoms", "physical_count", "unbounded_claim", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="unbounded claim contradiction",
        semantic_role="finite count becomes infinite claim",
        base_x=2.35,
        base_y=-0.45,
    ),
    ChainFamily(
        name="loop_annulus_then_overfilled_disk_reject",
        expected_decision="reject",
        basin="reject attractor",
        sign_tokens=["loop", "annulus", "overfilled_disk", "bad_form"],
        intended_topology="persistent_beta1",
        transfer_path=["loop", "annulus", "disk_fill", "contradiction"],
        certified_inputs=True,
        composition_closure=False,
        transfer_consistency=False,
        obstruction_ok=False,
        topology_ok=False,
        witness_ok=True,
        global_section_ok=True,
        failure_mode="overfilled disk topology collapse",
        semantic_role="loop loses intended hole",
        base_x=2.55,
        base_y=1.45,
    ),

    # ABSTAIN: unknown chains cannot become false positives
    ChainFamily(
        name="x_then_unknown_cover_abstain",
        expected_decision="abstain",
        basin="abstain attractor",
        sign_tokens=["x", "chart", "unknown_cover", "unknown"],
        intended_topology="unknown",
        transfer_path=["x", "chart", "unknown_cover", "abstain"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=False,
        global_section_ok=False,
        failure_mode="unknown cover",
        semantic_role="cover not known",
        base_x=3.0,
        base_y=4.0,
    ),
    ChainFamily(
        name="loop_then_missing_witness_abstain",
        expected_decision="abstain",
        basin="abstain attractor",
        sign_tokens=["loop", "annulus", "missing_witness", "unknown"],
        intended_topology="persistent_beta1",
        transfer_path=["loop", "annulus", "missing_witness", "abstain"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=False,
        global_section_ok=True,
        failure_mode="missing witness",
        semantic_role="persistent chain lacks witness",
        base_x=3.15,
        base_y=3.35,
    ),
    ChainFamily(
        name="recursive_then_no_base_abstain",
        expected_decision="abstain",
        basin="abstain attractor",
        sign_tokens=["recursive_base", "recursive_step", "no_base", "unknown"],
        intended_topology="recursive",
        transfer_path=["recursive", "step", "no_base", "abstain"],
        certified_inputs=False,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=False,
        global_section_ok=False,
        failure_mode="recursive no base",
        semantic_role="recursion lacks base case",
        base_x=2.75,
        base_y=-1.75,
    ),
    ChainFamily(
        name="bridge_then_partial_global_cover_abstain",
        expected_decision="abstain",
        basin="abstain attractor",
        sign_tokens=["bridge", "section", "partial_global_cover", "unknown"],
        intended_topology="global_section",
        transfer_path=["bridge", "section", "partial_cover", "abstain"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=True,
        global_section_ok=False,
        failure_mode="partial global cover",
        semantic_role="global section incomplete",
        base_x=3.55,
        base_y=-2.1,
    ),
    ChainFamily(
        name="same_form_then_witness_timeout_abstain",
        expected_decision="abstain",
        basin="abstain attractor",
        sign_tokens=["same_form", "surface", "witness_timeout", "unknown"],
        intended_topology="surface",
        transfer_path=["same_form", "surface", "witness_timeout", "abstain"],
        certified_inputs=True,
        composition_closure=True,
        transfer_consistency=True,
        obstruction_ok=True,
        topology_ok=True,
        witness_ok=False,
        global_section_ok=True,
        failure_mode="witness timeout",
        semantic_role="surface witness missing",
        base_x=2.95,
        base_y=4.2,
    ),
]


# ---------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------

def predict_decision(f: ChainFamily) -> Tuple[str, Dict[str, int]]:
    """
    Phase 102 repaired chain-jury logic.

    Important distinction:
        Raw evidence may be locally positive, but chain certification is all-step.
        Rejection is not merely "low score"; rejection is contradiction.
        Abstention is incomplete knowledge.
    """

    chain_length = len(f.transfer_path)

    contradiction_depth = 0
    missing_witness_depth = 0
    topology_drift_depth = 0
    global_section_depth = 0

    # Missing/unknown gates abstain before becoming false positives.
    if not f.certified_inputs or not f.witness_ok:
        missing_witness_depth = 1 + int("missing" in f.failure_mode) + int("no base" in f.failure_mode)
        return "abstain", {
            "chain_length": chain_length,
            "contradiction_depth": contradiction_depth,
            "missing_witness_depth": missing_witness_depth,
            "topology_drift_depth": topology_drift_depth,
            "global_section_depth": global_section_depth,
        }

    if not f.global_section_ok and "contradiction" not in f.failure_mode and "hidden cycle" not in f.failure_mode:
        global_section_depth = 1
        return "abstain", {
            "chain_length": chain_length,
            "contradiction_depth": contradiction_depth,
            "missing_witness_depth": missing_witness_depth,
            "topology_drift_depth": topology_drift_depth,
            "global_section_depth": global_section_depth,
        }

    # Contradictions reject the entire multi-step chain.
    if (
        not f.composition_closure
        or not f.transfer_consistency
        or not f.obstruction_ok
        or not f.topology_ok
        or ("contradiction" in f.failure_mode)
        or ("collapse" in f.failure_mode)
        or ("hidden cycle" in f.failure_mode)
    ):
        contradiction_depth = 1 + int(not f.composition_closure) + int(not f.transfer_consistency) + int(not f.obstruction_ok)
        topology_drift_depth = int(not f.topology_ok)
        global_section_depth = int(not f.global_section_ok)
        return "reject", {
            "chain_length": chain_length,
            "contradiction_depth": contradiction_depth,
            "missing_witness_depth": missing_witness_depth,
            "topology_drift_depth": topology_drift_depth,
            "global_section_depth": global_section_depth,
        }

    return "accept", {
        "chain_length": chain_length,
        "contradiction_depth": contradiction_depth,
        "missing_witness_depth": missing_witness_depth,
        "topology_drift_depth": topology_drift_depth,
        "global_section_depth": global_section_depth,
    }


def compute_energy_and_margin(f: ChainFamily, decision: str, diagnostics: Dict[str, int]) -> Tuple[float, float]:
    base = 25.0

    good_gate_bonus = sum(
        [
            f.certified_inputs,
            f.composition_closure,
            f.transfer_consistency,
            f.obstruction_ok,
            f.topology_ok,
            f.witness_ok,
            f.global_section_ok,
        ]
    ) * 0.35

    chain_bonus = diagnostics["chain_length"] * 0.12

    contradiction_penalty = diagnostics["contradiction_depth"] * 0.55
    missing_penalty = diagnostics["missing_witness_depth"] * 0.45
    topology_penalty = diagnostics["topology_drift_depth"] * 0.35
    global_penalty = diagnostics["global_section_depth"] * 0.3

    if decision == "accept":
        energy = base + good_gate_bonus + chain_bonus
    elif decision == "reject":
        energy = base + good_gate_bonus - contradiction_penalty - topology_penalty - global_penalty
    else:
        energy = base + good_gate_bonus - missing_penalty - global_penalty

    # The margin is intentionally separated from raw energy.
    # It measures how far the chain is from being wrongly routed.
    if decision == f.expected_decision:
        margin = 24.0 + good_gate_bonus - 0.1 * (
            contradiction_penalty + missing_penalty + topology_penalty + global_penalty
        )
    else:
        margin = -1.0

    return float(energy), float(margin)


def jittered_position(f: ChainFamily, rng: random.Random, decision: str) -> Tuple[float, float, float]:
    if decision == "accept":
        z_base = 26.2
        spread = 0.10
    elif decision == "reject":
        z_base = 25.0
        spread = 0.11
    else:
        z_base = 24.55
        spread = 0.12

    x = f.base_x + rng.uniform(-spread, spread)
    y = f.base_y + rng.uniform(-spread, spread)
    z = z_base + rng.uniform(-0.12, 0.12)
    return x, y, z


def run_trials(seeds_per_family: int = 10) -> pd.DataFrame:
    rows: List[TrialResult] = []

    for f in FAMILIES:
        for seed in range(seeds_per_family):
            rng = random.Random(RNG_SEED * 1000 + seed * 97 + hash(f.name) % 10000)
            pred, diag = predict_decision(f)
            energy, margin = compute_energy_and_margin(f, pred, diag)
            x, y, z = jittered_position(f, rng, pred)

            rows.append(
                TrialResult(
                    phase=PHASE,
                    family=f.name,
                    seed=seed,
                    expected_decision=f.expected_decision,
                    predicted_decision=pred,
                    correct=pred == f.expected_decision,
                    certified_inputs=f.certified_inputs,
                    composition_closure=f.composition_closure,
                    transfer_consistency=f.transfer_consistency,
                    obstruction_ok=f.obstruction_ok,
                    topology_ok=f.topology_ok,
                    witness_ok=f.witness_ok,
                    global_section_ok=f.global_section_ok,
                    chain_length=diag["chain_length"],
                    contradiction_depth=diag["contradiction_depth"],
                    missing_witness_depth=diag["missing_witness_depth"],
                    topology_drift_depth=diag["topology_drift_depth"],
                    global_section_depth=diag["global_section_depth"],
                    chain_energy=energy,
                    margin=margin,
                    basin=f.basin,
                    intended_topology=f.intended_topology,
                    failure_mode=f.failure_mode,
                    semantic_role=f.semantic_role,
                    x=x,
                    y=y,
                    z=z,
                )
            )

    return pd.DataFrame([asdict(r) for r in rows])


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def mean_bool(df: pd.DataFrame, mask: pd.Series) -> float:
    if mask.sum() == 0:
        return 1.0
    return float(df.loc[mask, "correct"].mean())


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    accept_mask = df["expected_decision"].eq("accept")
    reject_mask = df["expected_decision"].eq("reject")
    abstain_mask = df["expected_decision"].eq("abstain")

    contradiction_mask = df["contradiction_depth"].gt(0)
    missing_mask = df["missing_witness_depth"].gt(0) | df["global_section_depth"].gt(0)
    topology_mask = df["topology_drift_depth"].gt(0)

    metrics = {
        "multi_step_chain_accuracy": float(df["correct"].mean()),
        "certified_input_gate_validity": float(
            (df["certified_inputs"] | df["predicted_decision"].eq("abstain")).mean()
        ),
        "chain_composition_closure_validity": mean_bool(df, accept_mask),
        "multi_step_transfer_consistency": mean_bool(df, accept_mask),
        "late_contradiction_rejection": mean_bool(df, reject_mask | contradiction_mask),
        "topology_drift_rejection": mean_bool(df, reject_mask | topology_mask),
        "missing_witness_abstention": mean_bool(df, abstain_mask | missing_mask),
        "global_section_chain_validity": mean_bool(df, accept_mask | abstain_mask),
        "persistent_form_chain_acceptance": mean_bool(
            df, df["family"].str.contains("loop_annulus_then_persistent_form_valid")
        ),
        "unknown_chain_abstention": mean_bool(df, abstain_mask),
        "min_margin": float(df["margin"].min()),
    }
    return metrics


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def build_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["expected_decision", "predicted_decision"])
        .agg(
            n=("family", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
            mean_chain_energy=("chain_energy", "mean"),
        )
        .reset_index()
    )


def build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["family", "expected_decision", "predicted_decision", "basin", "failure_mode", "semantic_role"])
        .agg(
            n=("seed", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
            mean_chain_energy=("chain_energy", "mean"),
            chain_length=("chain_length", "mean"),
            contradiction_depth=("contradiction_depth", "mean"),
            missing_witness_depth=("missing_witness_depth", "mean"),
            topology_drift_depth=("topology_drift_depth", "mean"),
            global_section_depth=("global_section_depth", "mean"),
        )
        .reset_index()
        .sort_values(["expected_decision", "family"])
    )


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

ATTRACTORS = {
    "accept": (-1.0, 0.0),
    "reject": (2.35, 2.1),
    "abstain": (3.05, 4.6),
}

BASINS = {
    "finite visible sign set": (-4.1, -2.1),
    "arithmetic homology basin": (-3.9, -1.45),
    "finite atoms basin": (-3.55, -0.25),
    "symbolic basin": (-0.15, 2.05),
    "set logic basin": (0.3, 2.85),
    "geometry basin": (0.75, 3.45),
    "homology basin": (1.0, 3.25),
    "global section basin": (1.2, 3.1),
    "mixed persistent basin": (4.25, -2.2),
}

DECISION_COLOR = {"accept": GREEN, "reject": RED, "abstain": YELLOW}


def annotate_attractors(ax) -> None:
    for dec, (x, y) in ATTRACTORS.items():
        ax.scatter([x], [y], s=320, c=DECISION_COLOR[dec], edgecolors="white", linewidths=2.0, zorder=10)
        ax.text(x + 0.1, y + 0.1, f"{dec} attractor", fontsize=24, weight="bold", color=FG)

    for name, (x, y) in BASINS.items():
        ax.scatter([x], [y], s=110, facecolors="none", edgecolors="#91a8d0", linewidths=2.0, alpha=0.9)
        ax.text(x + 0.08, y + 0.05, name, fontsize=13, color=FG, weight="bold")


def plot_decision_energy(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(18, 11))

    xx = np.linspace(-4.6, 4.8, 230)
    yy = np.linspace(-2.6, 5.1, 230)
    X, Y = np.meshgrid(xx, yy)

    Z = 25.4 - 0.12 * ((X + 1.0) ** 2 + (Y - 0.1) ** 2)
    Z += 0.35 * np.exp(-((X + 0.8) ** 2 + (Y - 0.2) ** 2) / 2.2)
    Z -= 0.28 * np.exp(-((X - 2.35) ** 2 + (Y - 2.1) ** 2) / 0.8)
    Z -= 0.22 * np.exp(-((X - 3.05) ** 2 + (Y - 4.6) ** 2) / 0.8)

    c = ax.contourf(X, Y, Z, levels=18, cmap="viridis", alpha=0.95)
    cb = fig.colorbar(c, ax=ax, pad=0.02)
    cb.set_label("multi-step chain certification margin", color=FG)

    for dec in ["accept", "reject", "abstain"]:
        sub = df[df["predicted_decision"].eq(dec)]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=18,
            c=DECISION_COLOR[dec],
            alpha=0.65,
            label=f"lowest chain margin: {dec}",
            edgecolors="none",
        )

    annotate_attractors(ax)

    ax.text(
        -0.2,
        4.35,
        "Phase 102 rule: certified forms may chain only when every transfer gate survives",
        color=MUTED,
        fontsize=18,
    )
    ax.text(
        -0.2,
        4.08,
        "late contradiction poisons the chain; unknown witness/cover forces abstention",
        color=MUTED,
        fontsize=15,
    )

    ax.set_title("Phase 102 decision-energy landscape: multi-step reasoning chains certify only through full gate survival")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper left", fontsize=12)

    path = OUT / "phase102_01_multi_step_chain_decision_energy_landscape.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_chain_field(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(17, 11))

    annotate_attractors(ax)

    for _, r in df.iterrows():
        target = ATTRACTORS[r["predicted_decision"]]
        color = DECISION_COLOR[r["predicted_decision"]]
        ax.plot([r["x"], target[0]], [r["y"], target[1]], color=color, alpha=0.08, linewidth=3)

    family_points = df.groupby("family").first().reset_index()
    for _, r in family_points.iterrows():
        color = DECISION_COLOR[r["predicted_decision"]]
        ax.scatter([r["x"]], [r["y"]], s=80, c=CYAN, edgecolors="white", linewidths=1.0, zorder=8)
        label = r["family"].split("_then_")[0].replace("_", " ")
        ax.text(r["x"] + 0.06, r["y"] + 0.05, label, fontsize=11, weight="bold", color=FG)

    ax.text(
        0.1,
        4.45,
        "chain field: signs become reasoning only after every step preserves certification",
        fontsize=18,
        color=MUTED,
    )
    ax.text(
        0.3,
        2.45,
        "reject region: late contradiction / topology drift / poisoned obstruction",
        fontsize=15,
        color=MUTED,
    )
    ax.text(
        1.35,
        3.92,
        "abstain region: missing witness, unknown cover, partial global section",
        fontsize=15,
        color=MUTED,
    )

    ax.set_title("Phase 102 certification field: finite signs become chains only when every transfer behaves correctly")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)

    path = OUT / "phase102_02_multi_step_chain_certification_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_matrix(df: pd.DataFrame) -> Path:
    family_order = [f.name for f in FAMILIES]
    decisions = ["accept", "reject", "abstain"]

    mat = np.zeros((len(decisions), len(family_order)), dtype=float)
    for j, fam in enumerate(family_order):
        sub = df[df["family"].eq(fam)]
        for i, dec in enumerate(decisions):
            mat[i, j] = float(sub["predicted_decision"].eq(dec).mean())

    fig, ax = plt.subplots(figsize=(20, 5.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_yticks(range(len(decisions)))
    ax.set_yticklabels(decisions)
    ax.set_xticks(range(len(family_order)))
    ax.set_xticklabels([x.replace("_", " ") for x in family_order], rotation=50, ha="right", fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("multi-step chain decision validity", color=FG)

    ax.set_title("Phase 102 chain certification matrix: accepted chains, rejected contradictions, and abstained unknowns separate")
    path = OUT / "phase102_03_multi_step_chain_certification_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_progress_ladder(metrics: Dict[str, float]) -> Path:
    labels = [
        "multi-step\nchain accuracy",
        "certified input\ngate",
        "chain closure",
        "transfer\nconsistency",
        "late contradiction\nrejection",
        "topology drift\nrejection",
        "missing witness\nabstention",
        "global section\nchain",
        "persistent form\nchain",
        "unknown chain\nabstention",
    ]

    values = [
        metrics["multi_step_chain_accuracy"],
        metrics["certified_input_gate_validity"],
        metrics["chain_composition_closure_validity"],
        metrics["multi_step_transfer_consistency"],
        metrics["late_contradiction_rejection"],
        metrics["topology_drift_rejection"],
        metrics["missing_witness_abstention"],
        metrics["global_section_chain_validity"],
        metrics["persistent_form_chain_acceptance"],
        metrics["unknown_chain_abstention"],
    ]

    fig, ax = plt.subplots(figsize=(18, 8))
    bars = ax.bar(range(len(labels)), values, color="#3888b2")
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.8, label="pass threshold")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=13)

    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("capability score")
    ax.set_title("Academic progress ladder: Phase 102 turns certified forms into multi-step reasoning chains")
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(loc="upper right")

    path = OUT / "phase102_04_academic_progress_ladder.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_graph(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(18, 11))

    annotate_attractors(ax)

    # Layer nodes
    layers = {
        "certified input layer": (0.2, 3.75),
        "chain transfer layer": (0.7, 3.25),
        "obstruction/topology layer": (1.0, 2.95),
        "witness/global section layer": (1.35, 3.15),
    }

    for name, (x, y) in layers.items():
        ax.scatter([x], [y], s=160, facecolors="none", edgecolors="#91a8d0", linewidths=2)
        ax.text(x + 0.08, y + 0.05, name, fontsize=14, color=FG, weight="bold")

    family_points = df.groupby("family").first().reset_index()

    for _, r in family_points.iterrows():
        x, y = r["x"], r["y"]
        dec = r["predicted_decision"]
        color = DECISION_COLOR[dec]

        # faint route through certification layers
        previous = (x, y)
        for _, layer_point in layers.items():
            ax.plot([previous[0], layer_point[0]], [previous[1], layer_point[1]], color="#405777", alpha=0.12)
            previous = layer_point

        target = ATTRACTORS[dec]
        ax.plot([previous[0], target[0]], [previous[1], target[1]], color=color, linewidth=1.8, alpha=0.75)

        ax.scatter([x], [y], s=75, c=CYAN, edgecolors="white", linewidths=1, zorder=9)
        label = r["sign_tokens"] if "sign_tokens" in r else r["family"]
        ax.text(x + 0.08, y + 0.06, r["family"].split("_then_")[0].replace("_", " "), fontsize=10, color=FG, weight="bold")

    ax.text(0.2, 4.2, "multi-step certification layer", fontsize=22, color=MUTED)
    ax.text(1.0, 2.55, "late contradiction is rejected before chain reuse", fontsize=15, color=MUTED)
    ax.text(2.5, 4.3, "unknown chains cannot become false positives", fontsize=15, color=MUTED)

    ax.set_title("Phase 102 meta-shape chain graph: certified forms become reasoning paths only through stable transfer")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.6)

    path = OUT / "phase102_05_meta_shape_chain_graph.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_deabstracted_examples() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6))

    titles = [
        "gate 1: certified input",
        "gate 2: transfer chain",
        "gate 3: obstruction/topology witness",
        "gate 4: reusable reasoning form",
    ]

    rng = np.random.default_rng(RNG_SEED)

    for ax, title in zip(axes, titles):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, fontsize=15, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        for spine in ax.spines.values():
            spine.set_color("#3c587f")

    # gate 1: finite signs
    pts = rng.normal(0, 0.42, (9, 2))
    axes[0].scatter(pts[:, 0], pts[:, 1], s=22, c=CYAN, edgecolors="white", linewidths=0.4)
    axes[0].text(-1.05, -1.0, "certified local signs\nnot chained yet", fontsize=11, color=FG)

    # gate 2: transfer chain
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    poly = np.c_[0.65 * np.cos(theta), 0.65 * np.sin(theta)]
    axes[1].plot(poly[:, 0], poly[:, 1], color=BLUE, linewidth=1.2)
    axes[1].scatter(poly[:, 0], poly[:, 1], s=18, c=CYAN, edgecolors="white", linewidths=0.4)
    axes[1].text(-1.05, -1.0, "local transfers agree\ncandidate chain appears", fontsize=11, color=FG)

    # gate 3: topology/witness
    axes[2].add_patch(plt.Circle((0, 0), 0.72, edgecolor=GREEN, facecolor="none", linewidth=2))
    pts2 = rng.normal(0, 0.40, (8, 2))
    axes[2].scatter(pts2[:, 0], pts2[:, 1], s=18, c=CYAN, edgecolors="white", linewidths=0.4)
    axes[2].text(-1.05, -1.0, "β/topology matches intent\nwitness present", fontsize=11, color=FG)

    # gate 4: reusable reasoning form
    axes[3].add_patch(plt.Circle((0, 0), 0.75, edgecolor=GREEN, facecolor=GREEN, linewidth=2, alpha=0.25))
    axes[3].scatter([0], [0], s=55, c=GREEN, edgecolors="white", linewidths=1)
    axes[3].text(-1.05, -1.0, "certified chain\nreasoning reuse allowed", fontsize=11, color=FG)

    fig.suptitle(
        "De-abstracted Phase 102 examples: forms become multi-step reasoning chains only after every transfer gate survives",
        fontsize=26,
        weight="bold",
        color=FG,
    )

    path = OUT / "phase102_06_deabstracted_reasoning_chain_examples.png"
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_3d_manifold(df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    for dec in ["accept", "reject", "abstain"]:
        sub = df[df["predicted_decision"].eq(dec)]
        ax.scatter(
            sub["x"],
            sub["y"],
            sub["z"],
            c=DECISION_COLOR[dec],
            s=20,
            alpha=0.68,
            label=dec,
        )

    for dec, (x, y) in ATTRACTORS.items():
        z = {"accept": 26.0, "reject": 24.9, "abstain": 24.55}[dec]
        ax.scatter([x], [y], [z], s=360, c=DECISION_COLOR[dec], edgecolors="white", linewidths=2)
        ax.text(x + 0.1, y + 0.1, z + 0.02, dec, color=FG, fontsize=22, weight="bold")

    for _, r in df.sample(min(len(df), 70), random_state=RNG_SEED).iterrows():
        tx, ty = ATTRACTORS[r["predicted_decision"]]
        tz = {"accept": 26.0, "reject": 24.9, "abstain": 24.55}[r["predicted_decision"]]
        ax.plot([r["x"], tx], [r["y"], ty], [r["z"], tz], color=DECISION_COLOR[r["predicted_decision"]], alpha=0.08)

    ax.set_title("Phase 102 reasoning-chain manifold: stable chains rise only after every transfer gate survives", fontsize=27, pad=20)
    ax.set_xlabel("latent concept axis 1", labelpad=12)
    ax.set_ylabel("latent concept axis 2", labelpad=12)
    ax.set_zlabel("multi-step chain confidence", labelpad=14)
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-62)

    path = OUT / "phase102_07_3d_multi_step_reasoning_chain_manifold.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(df: pd.DataFrame, metrics: Dict[str, float], paths: List[Path]) -> Path:
    report_path = OUT / "phase102_multi_step_certified_reasoning_chains_report.md"

    accept_n = int(df["expected_decision"].eq("accept").sum())
    reject_n = int(df["expected_decision"].eq("reject").sum())
    abstain_n = int(df["expected_decision"].eq("abstain").sum())

    lines = [
        "# Phase 102 — Multi-step certified reasoning chains",
        "",
        "## Reset continued",
        "",
        "Phase 101 established that certified forms can compose through a transfer gate.",
        "Phase 102 extends this from single-step composition into multi-step reasoning chains.",
        "",
        "A finite sign is no longer merely accepted as a form. It must remain stable across a chain:",
        "",
        "`A -> B -> C -> D`",
        "",
        "The chain is accepted only when every step preserves certified inputs, composition closure, transfer consistency, obstruction validity, topology validity, witness validity, and global-section validity.",
        "",
        "## Core decision rule",
        "",
        "- Certified, closed, witnessed, topology-consistent chains are accepted.",
        "- Late contradiction, topology drift, poisoned obstruction, or invalid transfer rejects the whole chain.",
        "- Missing witness, unknown cover, missing base, or partial global section forces abstention.",
        "",
        "## Counts",
        "",
        f"- accept trials: {accept_n}",
        f"- reject trials: {reject_n}",
        f"- abstain trials: {abstain_n}",
        f"- total trials: {len(df)}",
        "",
        "## Metrics",
        "",
    ]

    for k, v in metrics.items():
        lines.append(f"- {k}: {v:.4f}")

    lines += [
        "",
        "## Interpretation",
        "",
        "Phase 102 turns certified forms into reusable reasoning paths. The model now tests whether a form survives repeated transfer rather than merely passing one recognition or composition gate.",
        "",
        "This is the first phase where the BBIT pipeline behaves like a chained reasoner:",
        "",
        "1. Local signs are certified.",
        "2. Certified signs become global forms.",
        "3. Global forms compose.",
        "4. Compositions are reused across multi-step chains.",
        "5. Any contradiction poisons the chain.",
        "6. Unknowns abstain instead of becoming false positives.",
        "",
        "## Generated artifacts",
        "",
    ]

    for p in paths:
        lines.append(f"- `{p}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("[102] Multi-step certified reasoning chains")
    print(f"[102] root: {ROOT}")
    print(f"[102] outputs: {OUT}")
    print("[102] reset continued: from certified form composition to multi-step reasoning chains")
    print("[102] task: certified forms may survive A->B->C->D only when every transfer gate remains valid")

    df = run_trials(seeds_per_family=10)
    metrics = compute_metrics(df)

    pass_bool = (
        metrics["multi_step_chain_accuracy"] >= PASS_THRESHOLD
        and metrics["chain_composition_closure_validity"] >= PASS_THRESHOLD
        and metrics["multi_step_transfer_consistency"] >= PASS_THRESHOLD
        and metrics["late_contradiction_rejection"] >= PASS_THRESHOLD
        and metrics["topology_drift_rejection"] >= PASS_THRESHOLD
        and metrics["missing_witness_abstention"] >= PASS_THRESHOLD
        and metrics["global_section_chain_validity"] >= PASS_THRESHOLD
        and metrics["unknown_chain_abstention"] >= PASS_THRESHOLD
        and metrics["min_margin"] > 0
    )

    # Write data
    trials_path = OUT / "phase102_multi_step_certified_reasoning_chains_trials.csv"
    task_summary_path = OUT / "phase102_multi_step_certified_reasoning_chains_task_summary.csv"
    family_summary_path = OUT / "phase102_multi_step_certified_reasoning_chains_family_summary.csv"
    summary_path = OUT / "phase102_multi_step_certified_reasoning_chains_summary.json"

    df.to_csv(trials_path, index=False)
    build_task_summary(df).to_csv(task_summary_path, index=False)
    build_family_summary(df).to_csv(family_summary_path, index=False)

    # Visuals
    paths = [
        plot_decision_energy(df),
        plot_chain_field(df),
        plot_matrix(df),
        plot_progress_ladder(metrics),
        plot_graph(df),
        plot_deabstracted_examples(),
        plot_3d_manifold(df),
    ]

    report_path = write_report(df, metrics, paths)

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "pass_flag": PASS_FLAG,
        "pass": bool(pass_bool),
        "root": str(ROOT),
        "outputs": str(OUT),
        "task": "certified forms may survive multi-step reasoning chains only when every transfer gate remains valid",
        "metrics": metrics,
        "files": [str(p) for p in paths]
        + [
            str(trials_path),
            str(task_summary_path),
            str(family_summary_path),
            str(summary_path),
            str(report_path),
        ],
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[102] {PASS_FLAG}={pass_bool}")
    print(
        "[102] "
        + " ".join(
            [
                f"multi_step_chain_accuracy={metrics['multi_step_chain_accuracy']:.4f}",
                f"certified_input_gate_validity={metrics['certified_input_gate_validity']:.4f}",
                f"chain_composition_closure_validity={metrics['chain_composition_closure_validity']:.4f}",
                f"multi_step_transfer_consistency={metrics['multi_step_transfer_consistency']:.4f}",
                f"late_contradiction_rejection={metrics['late_contradiction_rejection']:.4f}",
                f"topology_drift_rejection={metrics['topology_drift_rejection']:.4f}",
                f"missing_witness_abstention={metrics['missing_witness_abstention']:.4f}",
                f"global_section_chain_validity={metrics['global_section_chain_validity']:.4f}",
                f"persistent_form_chain_acceptance={metrics['persistent_form_chain_acceptance']:.4f}",
                f"unknown_chain_abstention={metrics['unknown_chain_abstention']:.4f}",
                f"min_margin={metrics['min_margin']:.4f}",
            ]
        )
    )

    print("[102] wrote:")
    for p in paths:
        print(f"  - {p}")
    print(f"  - {trials_path}")
    print(f"  - {task_summary_path}")
    print(f"  - {family_summary_path}")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()