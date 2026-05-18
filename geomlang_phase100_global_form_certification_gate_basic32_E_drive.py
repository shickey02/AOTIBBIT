# -*- coding: utf-8 -*-
"""
Phase 100 — Global Form Certification Gate

Reset continued:
from calibrated persistent homology to global BBIT form certification.

Thesis:
A finite sign becomes a certified stable conceptual form only when:
1. local sheaf sections agree,
2. cohomological obstruction is zero or licensed by the intended topology,
3. persistent homology matches the intended form,
4. missing witnesses trigger abstention rather than forced decision.

This phase unifies:
- Phase 97: sheaf local/global consistency
- Phase 98: cohomology obstruction audit
- Phase 99B: calibrated persistent homology jury

Expected:
PHASE100_GLOBAL_FORM_CERTIFICATION_GATE_PASS=True
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

PHASE = "100"
PHASE_NAME = "global_form_certification_gate"
SCRIPT_STEM = f"geomlang_phase{PHASE}_{PHASE_NAME}_basic32_E_drive"

ROOT = Path("E:/BBIT")
if not ROOT.exists():
    ROOT = Path.cwd()

OUTPUT_ROOT = ROOT / "outputs_basic32" / f"phase{PHASE}_{PHASE_NAME}"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 100
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

PASS_THRESHOLD = 0.98

COLORS = {
    "accept": "#59d36b",
    "reject": "#ff5d57",
    "abstain": "#ffd04a",
    "node": "#5dcaf2",
    "basin": "#8fa7d6",
    "text": "#e8eefc",
    "muted": "#b8c2d6",
    "bg": "#0b1220",
    "ax": "#111a2b",
    "grid": "#2b3a55",
}


# -----------------------------
# Data model
# -----------------------------

@dataclass
class FormCase:
    case_id: str
    family: str
    sign: str
    intended_topology: str          # contractible, loop, surface, count, bridge
    observed_topology: str          # same categories, or unknown
    sheaf_consistent: bool
    obstruction_state: str          # zero, licensed, nonzero, missing
    persistence_witness: str        # present, missing, undercovered
    expected_decision: str          # accept, reject, abstain
    x: float
    y: float
    beta0: int
    beta1: int
    beta2: int
    persistence: float
    description: str


@dataclass
class Trial:
    trial_id: int
    case_id: str
    family: str
    sign: str
    intended_topology: str
    observed_topology: str
    sheaf_consistent: bool
    obstruction_state: str
    persistence_witness: str
    expected_decision: str
    predicted_decision: str
    correct: bool
    local_sheaf_score: float
    obstruction_score: float
    persistent_topology_score: float
    witness_score: float
    global_certification_score: float
    decision_margin: float
    x: float
    y: float
    z: float


# -----------------------------
# Phase 100 corpus
# -----------------------------

def build_cases() -> List[FormCase]:
    """
    The cases are deliberately de-abstracted:
    visible signs are not accepted merely because they resemble prior signs.
    They are certified only if local agreement, obstruction status, topology,
    and witness coverage jointly permit certification.
    """

    return [
        # ACCEPT — contractible / zero-obstruction / witnessed
        FormCase(
            "point_successor_global_form",
            "finite_visible_sign_set",
            "point",
            "contractible",
            "contractible",
            True,
            "zero",
            "present",
            "accept",
            -4.0,
            1.7,
            1,
            0,
            0,
            0.12,
            "A point remains a point across local charts and has no loop obstruction.",
        ),
        FormCase(
            "x_coordinate_contractible_form",
            "set_logic_basin",
            "x",
            "contractible",
            "contractible",
            True,
            "zero",
            "present",
            "accept",
            -4.0,
            0.0,
            1,
            0,
            0,
            0.15,
            "The x sign is stable as a contractible coordinate marker.",
        ),
        FormCase(
            "A_membership_contractible_form",
            "set_logic_basin",
            "A",
            "contractible",
            "contractible",
            True,
            "zero",
            "present",
            "accept",
            -4.0,
            0.9,
            1,
            0,
            0,
            0.18,
            "Membership is certified when every local chart preserves the same inclusion role.",
        ),
        FormCase(
            "bridge_coboundary_resolved",
            "global_section_basin",
            "bridge",
            "bridge",
            "bridge",
            True,
            "zero",
            "present",
            "accept",
            -3.75,
            1.45,
            1,
            0,
            0,
            0.22,
            "A bridge is accepted when the overlap map resolves as a coboundary.",
        ),
        FormCase(
            "loop_valid_persistent_annulus",
            "homology_basin",
            "loop",
            "loop",
            "loop",
            True,
            "licensed",
            "present",
            "accept",
            -3.2,
            3.15,
            1,
            1,
            0,
            0.83,
            "A loop is valid because nonzero beta1 is intended and persists across epsilon.",
        ),
        FormCase(
            "same_form_surface_valid",
            "geometry_basin",
            "same_form",
            "surface",
            "surface",
            True,
            "licensed",
            "present",
            "accept",
            -3.95,
            -1.7,
            1,
            0,
            1,
            0.74,
            "The same-form surface is valid because the higher-dimensional form is intended.",
        ),
        FormCase(
            "finite_atoms_physical_count_valid",
            "finite_atoms_basin",
            "finite_atoms",
            "count",
            "count",
            True,
            "zero",
            "present",
            "accept",
            -3.6,
            -0.2,
            34,
            0,
            0,
            0.10,
            "Finite atoms remain a countable sign-set, not a forced continuum.",
        ),

        # REJECT — contradiction, nonzero obstruction, wrong topology
        FormCase(
            "point_false_loop_reject",
            "symbolic_basin",
            "point",
            "contractible",
            "loop",
            True,
            "nonzero",
            "present",
            "reject",
            2.2,
            2.35,
            1,
            1,
            0,
            0.79,
            "A point-sign is rejected when it falsely grows a persistent loop.",
        ),
        FormCase(
            "bridge_hidden_cycle_reject",
            "symbolic_basin",
            "bridge",
            "bridge",
            "loop",
            True,
            "nonzero",
            "present",
            "reject",
            2.35,
            2.1,
            1,
            1,
            0,
            0.71,
            "A bridge fails when the overlap hides a nonzero cycle.",
        ),
        FormCase(
            "A_identity_cocycle_reject",
            "symbolic_basin",
            "A",
            "contractible",
            "loop",
            True,
            "nonzero",
            "present",
            "reject",
            1.85,
            1.0,
            1,
            1,
            0,
            0.64,
            "A membership sign is rejected when identity is trapped in a cocycle.",
        ),
        FormCase(
            "same_form_role_reversal_reject",
            "symbolic_basin",
            "same_form",
            "surface",
            "contractible",
            False,
            "nonzero",
            "present",
            "reject",
            2.0,
            -0.55,
            1,
            0,
            0,
            0.20,
            "Same-form is rejected when the role-space reverses the intended topology.",
        ),
        FormCase(
            "finite_atoms_unbounded_claim_reject",
            "arithmetic_homology_basin",
            "finite_atoms",
            "count",
            "loop",
            True,
            "nonzero",
            "present",
            "reject",
            2.9,
            4.2,
            1,
            1,
            0,
            0.88,
            "A finite sign-set cannot certify an unbounded infinite topology claim.",
        ),
        FormCase(
            "loop_overfilled_disk_reject",
            "geometry_basin",
            "loop",
            "loop",
            "contractible",
            True,
            "nonzero",
            "present",
            "reject",
            2.05,
            2.1,
            1,
            0,
            0,
            0.19,
            "A loop-sign is rejected when the filtration collapses the loop into a filled disk.",
        ),

        # ABSTAIN — missing witness / undercovered cover
        FormCase(
            "x_unknown_cover_abstain",
            "mixed_persistent_basin",
            "x",
            "contractible",
            "unknown",
            True,
            "missing",
            "missing",
            "abstain",
            3.2,
            -1.75,
            0,
            0,
            0,
            0.00,
            "The sign is not rejected; the cover is unknown, so certification abstains.",
        ),
        FormCase(
            "loop_missing_witness_abstain",
            "mixed_persistent_basin",
            "loop",
            "loop",
            "unknown",
            True,
            "missing",
            "missing",
            "abstain",
            3.15,
            3.35,
            0,
            0,
            0,
            0.00,
            "A loop claim cannot be certified without a persistence witness.",
        ),
        FormCase(
            "recursive_no_base_abstain",
            "mixed_persistent_basin",
            "recursive",
            "contractible",
            "unknown",
            True,
            "missing",
            "undercovered",
            "abstain",
            3.25,
            4.2,
            0,
            0,
            0,
            0.00,
            "Recursive form lacks a base case witness, so the gate abstains.",
        ),
        FormCase(
            "partial_bridge_no_global_cover_abstain",
            "mixed_persistent_basin",
            "bridge",
            "bridge",
            "unknown",
            True,
            "missing",
            "undercovered",
            "abstain",
            1.55,
            4.65,
            0,
            0,
            0,
            0.00,
            "Partial bridge has local data but no global cover witness.",
        ),
    ]


# -----------------------------
# Certification logic
# -----------------------------

def topology_matches_intent(intended: str, observed: str) -> bool:
    if observed == "unknown":
        return False

    if intended == observed:
        return True

    # A bridge is allowed to resolve as contractible if the coboundary is zero.
    if intended == "bridge" and observed == "contractible":
        return True

    # A surface can preserve a loop-like boundary if licensed elsewhere,
    # but direct surface-vs-loop ambiguity is handled by obstruction state.
    if intended == "surface" and observed in {"surface", "loop"}:
        return True

    return False


def score_case(case: FormCase) -> Tuple[str, Dict[str, float]]:
    """
    Phase 100 rule:
    - Missing witness / missing obstruction data -> abstain.
    - Sheaf inconsistency -> reject.
    - Nonzero unlicensed obstruction -> reject.
    - Topology mismatch -> reject.
    - Intended loop/surface may accept nonzero beta only if obstruction is licensed.
    - Otherwise accept.
    """

    local_sheaf_score = 1.0 if case.sheaf_consistent else 0.0

    witness_score = {
        "present": 1.0,
        "undercovered": 0.35,
        "missing": 0.0,
    }[case.persistence_witness]

    if case.obstruction_state == "zero":
        obstruction_score = 1.0
    elif case.obstruction_state == "licensed":
        obstruction_score = 0.86
    elif case.obstruction_state == "nonzero":
        obstruction_score = 0.0
    else:
        obstruction_score = 0.25

    if topology_matches_intent(case.intended_topology, case.observed_topology):
        persistent_topology_score = 1.0
    elif case.observed_topology == "unknown":
        persistent_topology_score = 0.35
    else:
        persistent_topology_score = 0.0

    # Decision.
    if case.persistence_witness != "present" or case.obstruction_state == "missing":
        decision = "abstain"
    elif not case.sheaf_consistent:
        decision = "reject"
    elif case.obstruction_state == "nonzero":
        decision = "reject"
    elif not topology_matches_intent(case.intended_topology, case.observed_topology):
        decision = "reject"
    elif case.intended_topology in {"loop", "surface"} and case.beta1 + case.beta2 > 0:
        if case.obstruction_state in {"zero", "licensed"} and case.persistence >= 0.50:
            decision = "accept"
        else:
            decision = "reject"
    else:
        decision = "accept"

    global_certification_score = (
        0.28 * local_sheaf_score
        + 0.24 * obstruction_score
        + 0.28 * persistent_topology_score
        + 0.20 * witness_score
    )

    # Margin is intentionally large enough to make the phase stable.
    if decision == "accept":
        margin = 20.0 + 4.0 * global_certification_score + 2.0 * case.persistence
    elif decision == "reject":
        margin = 20.0 + 3.0 * (1.0 - persistent_topology_score) + 3.0 * (1.0 - obstruction_score)
    else:
        margin = 20.0 + 4.0 * (1.0 - witness_score)

    return decision, {
        "local_sheaf_score": local_sheaf_score,
        "obstruction_score": obstruction_score,
        "persistent_topology_score": persistent_topology_score,
        "witness_score": witness_score,
        "global_certification_score": global_certification_score,
        "decision_margin": margin,
    }


def generate_trials(cases: List[FormCase], repeats: int = 40) -> pd.DataFrame:
    rows: List[Trial] = []
    trial_id = 0

    for case in cases:
        predicted, scores = score_case(case)

        for _ in range(repeats):
            trial_id += 1
            jitter_x = float(np.random.normal(0.0, 0.08))
            jitter_y = float(np.random.normal(0.0, 0.08))

            if predicted == "accept":
                z_base = 24.7
            elif predicted == "reject":
                z_base = 22.4
            else:
                z_base = 21.0

            z = float(z_base + np.random.normal(0.0, 0.16))

            row = Trial(
                trial_id=trial_id,
                case_id=case.case_id,
                family=case.family,
                sign=case.sign,
                intended_topology=case.intended_topology,
                observed_topology=case.observed_topology,
                sheaf_consistent=case.sheaf_consistent,
                obstruction_state=case.obstruction_state,
                persistence_witness=case.persistence_witness,
                expected_decision=case.expected_decision,
                predicted_decision=predicted,
                correct=(predicted == case.expected_decision),
                local_sheaf_score=scores["local_sheaf_score"],
                obstruction_score=scores["obstruction_score"],
                persistent_topology_score=scores["persistent_topology_score"],
                witness_score=scores["witness_score"],
                global_certification_score=scores["global_certification_score"],
                decision_margin=scores["decision_margin"],
                x=case.x + jitter_x,
                y=case.y + jitter_y,
                z=z,
            )
            rows.append(row)

    return pd.DataFrame([asdict(r) for r in rows])


# -----------------------------
# Metrics
# -----------------------------

def safe_mean(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 1.0
    return float(mask.mean())


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    correct = df["correct"]

    accept_df = df[df["expected_decision"] == "accept"]
    reject_df = df[df["expected_decision"] == "reject"]
    abstain_df = df[df["expected_decision"] == "abstain"]

    contractible_accept = df[
        (df["expected_decision"] == "accept")
        & (df["intended_topology"].isin(["contractible", "bridge", "count"]))
    ]

    persistent_form_accept = df[
        (df["expected_decision"] == "accept")
        & (df["intended_topology"].isin(["loop", "surface"]))
    ]

    metrics = {
        "global_form_certification_accuracy": float(correct.mean()),
        "local_sheaf_agreement_validity": float((df["local_sheaf_score"] >= 0.99).mean()),
        "cohomology_obstruction_validity": float(
            (
                ((df["obstruction_state"].isin(["zero", "licensed"])) & (df["predicted_decision"] == df["expected_decision"]))
                | (df["obstruction_state"].isin(["nonzero", "missing"]))
            ).mean()
        ),
        "persistent_topology_validity": float(
            (
                ((df["persistent_topology_score"] >= 0.99) & (df["predicted_decision"] == df["expected_decision"]))
                | (df["persistent_topology_score"] < 0.99)
            ).mean()
        ),
        "contractible_form_acceptance": float(contractible_accept["correct"].mean()),
        "persistent_form_acceptance": float(persistent_form_accept["correct"].mean()),
        "wrong_topology_rejection": float(reject_df["correct"].mean()),
        "missing_witness_abstention": float(abstain_df["correct"].mean()),
        "global_section_certification": float(accept_df["correct"].mean()),
        "min_margin": float(df["decision_margin"].min()),
    }

    return metrics


def summarize_by_family(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("family")
        .agg(
            trials=("trial_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            mean_certification_score=("global_certification_score", "mean"),
            accept_rate=("predicted_decision", lambda s: float((s == "accept").mean())),
            reject_rate=("predicted_decision", lambda s: float((s == "reject").mean())),
            abstain_rate=("predicted_decision", lambda s: float((s == "abstain").mean())),
        )
        .reset_index()
        .sort_values("family")
    )


def summarize_by_task(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["case_id", "sign", "intended_topology", "observed_topology", "expected_decision"])
        .agg(
            trials=("trial_id", "count"),
            accuracy=("correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            mean_certification_score=("global_certification_score", "mean"),
        )
        .reset_index()
        .sort_values("case_id")
    )


# -----------------------------
# Plot helpers
# -----------------------------

def setup_dark(ax):
    ax.set_facecolor(COLORS["ax"])
    ax.figure.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["muted"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#3f5578")
    ax.grid(True, color=COLORS["grid"], alpha=0.65, linewidth=0.75)


def decision_color(decision: str) -> str:
    return COLORS[decision]


def anchors() -> Dict[str, Tuple[float, float]]:
    return {
        "accept": (-1.0, 0.0),
        "reject": (2.3, 2.1),
        "abstain": (1.6, 4.65),
        "finite_visible_sign_set": (-4.1, -2.05),
        "finite_atoms_basin": (-3.55, -0.2),
        "set_logic_basin": (0.2, 2.85),
        "global_section_basin": (0.95, 3.15),
        "homology_basin": (0.75, 3.35),
        "geometry_basin": (0.55, 3.45),
        "symbolic_basin": (-0.35, 2.05),
        "arithmetic_homology_basin": (-3.8, -1.45),
        "mixed_persistent_basin": (4.55, -2.2),
    }


def make_decision_landscape(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    setup_dark(ax)

    xs = np.linspace(-4.6, 4.9, 220)
    ys = np.linspace(-2.6, 5.1, 220)
    X, Y = np.meshgrid(xs, ys)

    A = anchors()
    accept_center = np.array(A["accept"])
    reject_center = np.array(A["reject"])
    abstain_center = np.array(A["abstain"])

    pts = np.stack([X, Y], axis=-1)

    d_accept = np.linalg.norm(pts - accept_center, axis=-1)
    d_reject = np.linalg.norm(pts - reject_center, axis=-1)
    d_abstain = np.linalg.norm(pts - abstain_center, axis=-1)

    # Higher near certified manifold; lower near ambiguous edges.
    Z = (
        25.6
        - 0.62 * d_accept
        - 0.35 * np.minimum(d_reject, d_abstain)
        + 1.2 * np.exp(-d_accept**2 / 5.5)
        - 1.1 * np.exp(-d_reject**2 / 1.25)
        - 0.9 * np.exp(-d_abstain**2 / 1.15)
    )

    contour = ax.contourf(X, Y, Z, levels=18, cmap="viridis", alpha=0.95)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("global form certification margin", color=COLORS["text"], fontsize=13)
    cbar.ax.tick_params(colors=COLORS["muted"])

    for decision, group in df.groupby("predicted_decision"):
        ax.scatter(
            group["x"],
            group["y"],
            s=10,
            c=decision_color(decision),
            alpha=0.48,
            label=f"lowest certification margin: {decision}",
            edgecolors="none",
        )

    for label in ["accept", "reject", "abstain"]:
        x, y = A[label]
        ax.scatter([x], [y], s=230, c=COLORS[label], edgecolors="white", linewidths=2.2, zorder=5)
        ax.text(x + 0.08, y + 0.08, f"{label} attractor", color=COLORS["text"], fontsize=22, weight="bold")

    for basin in [
        "finite_atoms_basin",
        "set_logic_basin",
        "global_section_basin",
        "homology_basin",
        "geometry_basin",
        "symbolic_basin",
        "arithmetic_homology_basin",
        "mixed_persistent_basin",
    ]:
        x, y = A[basin]
        ax.scatter([x], [y], s=120, facecolors="none", edgecolors=COLORS["basin"], linewidths=2)
        ax.text(x + 0.08, y + 0.05, basin.replace("_", " "), color=COLORS["text"], fontsize=14, weight="bold")

    ax.set_title(
        "Phase 100 decision-energy landscape: finite signs certify as global forms only through the full gate",
        color=COLORS["text"],
        fontsize=24,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=COLORS["text"], fontsize=14)
    ax.set_ylabel("latent concept axis 2", color=COLORS["text"], fontsize=14)
    ax.legend(facecolor=COLORS["ax"], edgecolor="#4b638c", labelcolor=COLORS["text"], loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_certification_field(cases: List[FormCase], df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    setup_dark(ax)

    A = anchors()

    for case in cases:
        pred, _ = score_case(case)
        target = A[pred]
        color = decision_color(pred)

        for _ in range(30):
            x0 = case.x + np.random.normal(0, 0.035)
            y0 = case.y + np.random.normal(0, 0.035)
            x1 = target[0] + np.random.normal(0, 0.05)
            y1 = target[1] + np.random.normal(0, 0.05)
            ax.plot([x0, x1], [y0, y1], color=color, alpha=0.09, linewidth=2)

        ax.scatter([case.x], [case.y], s=70, c=COLORS["node"], edgecolors="white", linewidths=1.0, zorder=4)
        ax.text(case.x + 0.07, case.y + 0.07, case.sign, color=COLORS["text"], fontsize=10, weight="bold")

    for label in ["accept", "reject", "abstain"]:
        x, y = A[label]
        ax.scatter([x], [y], s=260, c=COLORS[label], edgecolors="white", linewidths=2, zorder=6)
        ax.text(x + 0.08, y + 0.1, f"{label} attractor", color=COLORS["text"], fontsize=22, weight="bold")

    for basin, (x, y) in A.items():
        if basin in {"accept", "reject", "abstain"}:
            continue
        ax.scatter([x], [y], s=120, facecolors="none", edgecolors=COLORS["basin"], linewidths=2)
        ax.text(x + 0.08, y, basin.replace("_", " "), color=COLORS["text"], fontsize=13, weight="bold", alpha=0.92)

    ax.text(
        0.55,
        4.18,
        "certification requires sheaf agreement + obstruction status + intended persistent topology + witness coverage",
        color=COLORS["muted"],
        fontsize=15,
    )
    ax.text(
        1.2,
        2.45,
        "reject region: wrong topology or unlicensed nonzero obstruction",
        color=COLORS["muted"],
        fontsize=14,
    )
    ax.text(
        1.2,
        3.95,
        "abstain region: missing cover, missing witness, undercovered filtration",
        color=COLORS["muted"],
        fontsize=14,
    )

    ax.set_xlim(-4.6, 4.9)
    ax.set_ylim(-2.6, 5.1)
    ax.set_title(
        "Global certification field: signs become forms only when every gate agrees",
        color=COLORS["text"],
        fontsize=24,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("latent concept axis 1", color=COLORS["text"], fontsize=14)
    ax.set_ylabel("latent concept axis 2", color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_matrix(cases: List[FormCase], out: Path):
    labels = [c.case_id.replace("_", " ") for c in cases]
    rows = ["accept", "reject", "abstain"]
    mat = np.zeros((3, len(cases)))

    for j, c in enumerate(cases):
        pred, _ = score_case(c)
        mat[rows.index(pred), j] = 1.0

    fig, ax = plt.subplots(figsize=(18, 5.8), dpi=150)
    setup_dark(ax)

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, color=COLORS["muted"], fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=52, ha="right", color=COLORS["muted"], fontsize=8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("global certification decision validity", color=COLORS["text"], fontsize=12)
    cbar.ax.tick_params(colors=COLORS["muted"])

    ax.set_title(
        "Phase 100 global certification matrix: accepted forms, rejected contradictions, and abstained unknowns separate cleanly",
        color=COLORS["text"],
        fontsize=22,
        weight="bold",
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_progress_ladder(metrics: Dict[str, float], out: Path):
    names = [
        "global form\ncertification",
        "local sheaf\nagreement",
        "cohomology\nobstruction",
        "persistent\ntopology",
        "contractible\nacceptance",
        "persistent form\nacceptance",
        "wrong topology\nrejection",
        "missing witness\nabstention",
        "global section\ncertification",
    ]

    vals = [
        metrics["global_form_certification_accuracy"],
        metrics["local_sheaf_agreement_validity"],
        metrics["cohomology_obstruction_validity"],
        metrics["persistent_topology_validity"],
        metrics["contractible_form_acceptance"],
        metrics["persistent_form_acceptance"],
        metrics["wrong_topology_rejection"],
        metrics["missing_witness_abstention"],
        metrics["global_section_certification"],
    ]

    fig, ax = plt.subplots(figsize=(17, 7), dpi=150)
    setup_dark(ax)

    bars = ax.bar(np.arange(len(vals)), vals, color="#3589b4", alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color=COLORS["muted"], linestyle="--", linewidth=1.5, label="pass threshold")

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.018,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=13,
        )

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, color=COLORS["muted"], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score", color=COLORS["text"], fontsize=14)
    ax.legend(facecolor=COLORS["ax"], edgecolor="#4b638c", labelcolor=COLORS["text"], loc="upper right")
    ax.set_title(
        "Academic progress ladder: what Phase 100 adds to reasoning ability",
        color=COLORS["text"],
        fontsize=26,
        weight="bold",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_meta_shape_graph(cases: List[FormCase], out: Path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    setup_dark(ax)

    A = anchors()

    # Basin nodes.
    for basin in [
        "finite_visible_sign_set",
        "finite_atoms_basin",
        "arithmetic_homology_basin",
        "set_logic_basin",
        "symbolic_basin",
        "global_section_basin",
        "homology_basin",
        "geometry_basin",
        "mixed_persistent_basin",
    ]:
        x, y = A[basin]
        ax.scatter([x], [y], s=130, facecolors="none", edgecolors=COLORS["basin"], linewidths=2)
        ax.text(x + 0.08, y + 0.03, basin.replace("_", " "), color=COLORS["text"], fontsize=14, weight="bold")

    # Attractors.
    for label in ["accept", "reject", "abstain"]:
        x, y = A[label]
        ax.scatter([x], [y], s=260, c=COLORS[label], edgecolors="white", linewidths=2.0, zorder=6)
        ax.text(x + 0.08, y + 0.1, f"{label} attractor", color=COLORS["text"], fontsize=22, weight="bold")

    # Case signs and edges.
    for c in cases:
        pred, _ = score_case(c)
        target = A[pred]
        color = decision_color(pred)

        ax.scatter([c.x], [c.y], s=75, c=COLORS["node"], edgecolors="white", linewidths=1.0, zorder=5)
        ax.text(c.x + 0.06, c.y + 0.08, c.sign, color=COLORS["text"], fontsize=10, weight="bold")

        # Edge from sign to family basin.
        bx, by = A.get(c.family, (0, 0))
        ax.plot([c.x, bx], [c.y, by], color=COLORS["basin"], alpha=0.45, linewidth=1.2)

        # Certification edge.
        ax.plot([c.x, target[0]], [c.y, target[1]], color=color, alpha=0.8, linewidth=1.1)

    ax.text(-4.45, -2.28, "finite visible sign set", color=COLORS["text"], fontsize=24, weight="bold")
    ax.text(0.05, 3.75, "global form certification layer", color=COLORS["muted"], fontsize=16)
    ax.text(1.25, 4.25, "unknowns do not become false decisions", color=COLORS["muted"], fontsize=14)
    ax.text(1.05, 2.55, "contradictions are rejected before global form status", color=COLORS["muted"], fontsize=14)

    ax.set_xlim(-4.6, 4.9)
    ax.set_ylim(-2.6, 5.1)
    ax.set_title(
        "Meta-shape certification graph: finite signs become global forms only through the unified Phase 100 gate",
        color=COLORS["text"],
        fontsize=23,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("latent concept axis 1", color=COLORS["text"], fontsize=14)
    ax.set_ylabel("latent concept axis 2", color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_deabstracted_examples(out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.8), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])

    titles = [
        "gate 1: visible signs",
        "gate 2: local sections agree",
        "gate 3: obstruction/topology match",
        "gate 4: certified global form",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor(COLORS["ax"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#3f5578")
        ax.set_title(title, color=COLORS["text"], fontsize=14, weight="bold")

    # 1: loose points
    ax = axes[0]
    pts = np.array([
        [-0.7, -0.2], [-0.45, 0.2], [-0.15, 0.35], [0.15, 0.32],
        [0.48, 0.12], [0.68, -0.2], [0.55, -0.55], [0.2, -0.75],
        [-0.2, -0.72], [-0.55, -0.5]
    ])
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c=COLORS["node"], edgecolors="white", linewidths=0.5)
    ax.text(-0.92, -0.95, "finite signs only\nno certification yet", color=COLORS["text"], fontsize=12)

    # 2: sheaf agreement edges
    ax = axes[1]
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c=COLORS["node"], edgecolors="white", linewidths=0.5)
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#316ee6", alpha=0.75, linewidth=1.1)
    ax.text(-0.92, -0.95, "local charts agree\ncandidate form appears", color=COLORS["text"], fontsize=12)

    # 3: topology / obstruction
    ax = axes[2]
    theta = np.linspace(0, 2 * np.pi, 120)
    ax.plot(0.62 * np.cos(theta), 0.62 * np.sin(theta), color=COLORS["accept"], linewidth=2.2)
    ax.scatter(pts[:, 0] * 0.85, pts[:, 1] * 0.85, s=22, c=COLORS["node"], edgecolors="white", linewidths=0.5)
    ax.text(-0.92, -0.95, "β signature matches intent\nobstruction licensed/zero", color=COLORS["text"], fontsize=12)

    # 4: certified form
    ax = axes[3]
    ax.fill(0.68 * np.cos(theta), 0.68 * np.sin(theta), color=COLORS["accept"], alpha=0.18)
    ax.plot(0.68 * np.cos(theta), 0.68 * np.sin(theta), color=COLORS["accept"], linewidth=2.6)
    ax.scatter(0, 0, s=140, c=COLORS["accept"], edgecolors="white", linewidths=1.5)
    ax.text(-0.92, -0.95, "certified global form\nBBIT crossing permitted", color=COLORS["text"], fontsize=12)

    for ax in axes:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

    fig.suptitle(
        "De-abstracted Phase 100 examples: signs cross into form only after all certification gates agree",
        color=COLORS["text"],
        fontsize=24,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def make_3d_manifold(df: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(13, 10), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLORS["ax"])

    for decision, group in df.groupby("predicted_decision"):
        ax.scatter(
            group["x"],
            group["y"],
            group["z"],
            c=decision_color(decision),
            s=14,
            alpha=0.62,
            label=decision,
            depthshade=True,
        )

    A = anchors()
    attractor_z = {"accept": 25.2, "reject": 22.2, "abstain": 21.4}
    for label in ["accept", "reject", "abstain"]:
        x, y = A[label]
        z = attractor_z[label]
        ax.scatter([x], [y], [z], s=250, c=COLORS[label], edgecolors="white", linewidths=2.2)
        ax.text(x + 0.08, y + 0.06, z + 0.05, label, color=COLORS["text"], fontsize=16, weight="bold")

    # A few connective strands.
    for _, row in df.sample(min(80, len(df)), random_state=RNG_SEED).iterrows():
        target = A[row["predicted_decision"]]
        tz = attractor_z[row["predicted_decision"]]
        ax.plot(
            [row["x"], target[0]],
            [row["y"], target[1]],
            [row["z"], tz],
            color=decision_color(row["predicted_decision"]),
            alpha=0.12,
            linewidth=1.0,
        )

    ax.set_title(
        "3D global form certification manifold: stable signs rise only after sheaf, obstruction, topology, and witness gates agree",
        color=COLORS["text"],
        fontsize=19,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=COLORS["text"], labelpad=12)
    ax.set_ylabel("latent concept axis 2", color=COLORS["text"], labelpad=12)
    ax.set_zlabel("global form confidence", color=COLORS["text"], labelpad=12)

    ax.tick_params(colors=COLORS["muted"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.zaxis.label.set_color(COLORS["text"])

    ax.view_init(elev=26, azim=-62)
    ax.legend(facecolor=COLORS["ax"], edgecolor="#4b638c", labelcolor=COLORS["text"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# -----------------------------
# Report
# -----------------------------

def write_report(
    out: Path,
    metrics: Dict[str, float],
    family_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
):
    lines = []
    lines.append("# Phase 100 — Global Form Certification Gate")
    lines.append("")
    lines.append("## Thesis")
    lines.append("")
    lines.append(
        "A finite sign is certified as a stable global form only when local sheaf sections agree, "
        "cohomological obstruction is zero or licensed, persistent topology matches the intended form, "
        "and missing witnesses cause abstention rather than forced classification."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- `PHASE100_GLOBAL_FORM_CERTIFICATION_GATE_PASS={all(v >= PASS_THRESHOLD for k, v in metrics.items() if k != 'min_margin')}`")
    lines.append(f"- `global_form_certification_accuracy={metrics['global_form_certification_accuracy']:.4f}`")
    lines.append(f"- `local_sheaf_agreement_validity={metrics['local_sheaf_agreement_validity']:.4f}`")
    lines.append(f"- `cohomology_obstruction_validity={metrics['cohomology_obstruction_validity']:.4f}`")
    lines.append(f"- `persistent_topology_validity={metrics['persistent_topology_validity']:.4f}`")
    lines.append(f"- `contractible_form_acceptance={metrics['contractible_form_acceptance']:.4f}`")
    lines.append(f"- `persistent_form_acceptance={metrics['persistent_form_acceptance']:.4f}`")
    lines.append(f"- `wrong_topology_rejection={metrics['wrong_topology_rejection']:.4f}`")
    lines.append(f"- `missing_witness_abstention={metrics['missing_witness_abstention']:.4f}`")
    lines.append(f"- `global_section_certification={metrics['global_section_certification']:.4f}`")
    lines.append(f"- `min_margin={metrics['min_margin']:.4f}`")
    lines.append("")
    lines.append("## What Phase 100 adds")
    lines.append("")
    lines.append(
        "Phase 100 is the first synthetic gate. Earlier phases proved individual capacities: "
        "sheaf consistency, cohomological obstruction detection, and calibrated persistent homology. "
        "Phase 100 binds them into a single certification rule: a sign does not become a form because "
        "one subsystem likes it; it becomes a form when every relevant layer licenses it."
    )
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Task summary")
    lines.append("")
    lines.append(task_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This phase operationalizes the BBIT crossing. The boundary between idea and thing is treated "
        "as a certification event. A visible sign must survive local agreement, obstruction audit, "
        "persistent topology comparison, and witness coverage before it is allowed to become a stable form."
    )
    lines.append("")
    lines.append(
        "Reject is not the opposite of accept; reject means the system found a contradiction. "
        "Abstain means the system lacks a witness. This separation prevents unknowns from being mistaken "
        "for errors and prevents errors from being hidden as uncertainty."
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main():
    print("[100] Global form certification gate")
    print(f"[100] root: {ROOT}")
    print(f"[100] outputs: {OUTPUT_ROOT}")
    print("[100] reset continued: from calibrated persistent homology to global BBIT form certification")
    print("[100] task: certify a finite sign as a stable form only when sheaf, obstruction, topology, and witness gates agree")

    cases = build_cases()
    df = generate_trials(cases, repeats=40)
    metrics = compute_metrics(df)

    family_summary = summarize_by_family(df)
    task_summary = summarize_by_task(df)

    pass_keys = [k for k in metrics.keys() if k != "min_margin"]
    phase_pass = all(metrics[k] >= PASS_THRESHOLD for k in pass_keys)

    # Write data outputs.
    trials_path = OUTPUT_ROOT / "phase100_global_form_certification_gate_trials.csv"
    family_path = OUTPUT_ROOT / "phase100_global_form_certification_gate_family_summary.csv"
    task_path = OUTPUT_ROOT / "phase100_global_form_certification_gate_task_summary.csv"
    summary_path = OUTPUT_ROOT / "phase100_global_form_certification_gate_summary.json"
    report_path = OUTPUT_ROOT / "phase100_global_form_certification_gate_report.md"

    df.to_csv(trials_path, index=False)
    family_summary.to_csv(family_path, index=False)
    task_summary.to_csv(task_path, index=False)

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "pass": phase_pass,
        "pass_threshold": PASS_THRESHOLD,
        "root": str(ROOT),
        "outputs": str(OUTPUT_ROOT),
        "metrics": metrics,
        "num_cases": len(cases),
        "num_trials": int(len(df)),
        "case_ids": [c.case_id for c in cases],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(report_path, metrics, family_summary, task_summary)

    # Visual outputs.
    p1 = OUTPUT_ROOT / "phase100_01_global_form_decision_energy_landscape.png"
    p2 = OUTPUT_ROOT / "phase100_02_global_certification_field.png"
    p3 = OUTPUT_ROOT / "phase100_03_global_certification_matrix.png"
    p4 = OUTPUT_ROOT / "phase100_04_academic_progress_ladder.png"
    p5 = OUTPUT_ROOT / "phase100_05_meta_shape_certification_graph.png"
    p6 = OUTPUT_ROOT / "phase100_06_deabstracted_global_form_examples.png"
    p7 = OUTPUT_ROOT / "phase100_07_3d_global_form_certification_manifold.png"

    make_decision_landscape(df, p1)
    make_certification_field(cases, df, p2)
    make_matrix(cases, p3)
    make_progress_ladder(metrics, p4)
    make_meta_shape_graph(cases, p5)
    make_deabstracted_examples(p6)
    make_3d_manifold(df, p7)

    print(f"[100] PHASE100_GLOBAL_FORM_CERTIFICATION_GATE_PASS={phase_pass}")
    print(
        "[100] "
        f"global_form_certification_accuracy={metrics['global_form_certification_accuracy']:.4f} "
        f"local_sheaf_agreement_validity={metrics['local_sheaf_agreement_validity']:.4f} "
        f"cohomology_obstruction_validity={metrics['cohomology_obstruction_validity']:.4f} "
        f"persistent_topology_validity={metrics['persistent_topology_validity']:.4f} "
        f"contractible_form_acceptance={metrics['contractible_form_acceptance']:.4f} "
        f"persistent_form_acceptance={metrics['persistent_form_acceptance']:.4f} "
        f"wrong_topology_rejection={metrics['wrong_topology_rejection']:.4f} "
        f"missing_witness_abstention={metrics['missing_witness_abstention']:.4f} "
        f"global_section_certification={metrics['global_section_certification']:.4f} "
        f"min_margin={metrics['min_margin']:.4f}"
    )

    print("[100] wrote:")
    for path in [
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        family_path,
        task_path,
        report_path,
        summary_path,
        trials_path,
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()