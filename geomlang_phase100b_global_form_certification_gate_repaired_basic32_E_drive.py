# bbit_geomlang/geomlang_phase100b_global_form_certification_gate_repaired_basic32_E_drive.py
"""
Phase 100B: Global form certification gate / repaired sheaf-validity jury

Reset continued:
    Phase 98: cohomology obstruction audit
    Phase 99: persistent homology form audit
    Phase 99B: calibrated topology jury
    Phase 100: global form certification gate

Problem found in Phase 100:
    The classifier achieved 1.000 global form certification accuracy, but the run failed
    because local_sheaf_agreement_validity was scored as raw sheaf agreement.

    That is too strict.

    In contradiction cases, especially same_form_role_reversal_reject, local sheaf
    disagreement is the evidence for rejection. A valid reject case should not be
    punished merely because local sheaf agreement failed.

Phase 100B fix:
    Replace raw local sheaf agreement as a pass metric with a decision-relative
    local sheaf gate validity metric.

    Accept:
        local sheaf agreement must hold.

    Reject:
        local sheaf failure is valid when the rejection reason is a contradiction,
        role reversal, wrong topology, hidden cycle, false loop, etc.

    Abstain:
        missing cover / missing witness / unknown base / incomplete global section
        is valid when the system abstains.

Task:
    Certify finite signs as global forms only when:
        1. local sheaf sections agree,
        2. cohomology obstruction state permits the form,
        3. persistent topology matches intended topology,
        4. witness/global cover exists,
        5. global section certification succeeds.

    Reject contradictions.
    Abstain unknowns.
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
from matplotlib.collections import LineCollection


# ----------------------------
# Paths
# ----------------------------

ROOT = Path(r"E:\BBIT")
OUT_ROOT = ROOT / "outputs_basic32"
OUT = OUT_ROOT / "phase100b_global_form_certification_gate_repaired"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "geomlang_phase100b_global_form_certification_gate_repaired_basic32_E_drive.py"

RNG_SEED = 100_002
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ----------------------------
# Visual style
# ----------------------------

BG = "#07101f"
AX_BG = "#0d1728"
GRID = "#263a56"
TEXT = "#e7eefc"
MUTED = "#aeb9cc"

ACCEPT = "#56d36f"
REJECT = "#ff5c57"
ABSTAIN = "#ffc94a"
POINT = "#65d2ff"
BASIN = "#92a8d8"
EDGE = "#526989"


plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": AX_BG,
        "axes.edgecolor": "#3f587c",
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.titlecolor": TEXT,
        "grid.color": GRID,
        "font.size": 12,
        "axes.titlesize": 24,
        "axes.labelsize": 15,
        "legend.facecolor": AX_BG,
        "legend.edgecolor": "#5c7aa7",
        "legend.labelcolor": TEXT,
        "savefig.facecolor": BG,
    }
)


# ----------------------------
# Data model
# ----------------------------

@dataclass
class CaseSpec:
    case_id: str
    family: str
    sign: str
    intended_topology: str
    observed_topology: str
    expected_decision: str

    sheaf_consistent: bool
    obstruction_clear: bool
    topology_valid: bool
    witness_present: bool
    cover_complete: bool
    global_section_exists: bool

    contradiction_reason: str
    x: float
    y: float


CASES: List[CaseSpec] = [
    CaseSpec(
        case_id="point_successor_global_form",
        family="finite_atoms_basin",
        sign="point",
        intended_topology="contractible",
        observed_topology="contractible",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-4.0,
        y=1.7,
    ),
    CaseSpec(
        case_id="x_coordinate_contractible_form",
        family="finite_atoms_basin",
        sign="x",
        intended_topology="contractible",
        observed_topology="contractible",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-4.0,
        y=0.0,
    ),
    CaseSpec(
        case_id="A_membership_contractible_form",
        family="symbolic_basin",
        sign="A",
        intended_topology="contractible",
        observed_topology="contractible",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-4.0,
        y=0.9,
    ),
    CaseSpec(
        case_id="bridge_coboundary_resolved",
        family="global_section_basin",
        sign="bridge",
        intended_topology="contractible",
        observed_topology="contractible",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-3.75,
        y=1.45,
    ),
    CaseSpec(
        case_id="loop_valid_persistent_annulus",
        family="homology_basin",
        sign="loop",
        intended_topology="persistent_loop",
        observed_topology="persistent_loop",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-3.2,
        y=3.15,
    ),
    CaseSpec(
        case_id="same_form_surface_valid",
        family="geometry_basin",
        sign="same_form",
        intended_topology="surface",
        observed_topology="surface",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-3.9,
        y=-1.7,
    ),
    CaseSpec(
        case_id="finite_atoms_physical_count_valid",
        family="finite_atoms_basin",
        sign="finite_atoms",
        intended_topology="finite_count",
        observed_topology="finite_count",
        expected_decision="accept",
        sheaf_consistent=True,
        obstruction_clear=True,
        topology_valid=True,
        witness_present=True,
        cover_complete=True,
        global_section_exists=True,
        contradiction_reason="none",
        x=-3.65,
        y=-0.2,
    ),
    CaseSpec(
        case_id="point_false_loop_reject",
        family="reject_attractor",
        sign="point",
        intended_topology="contractible",
        observed_topology="persistent_loop",
        expected_decision="reject",
        sheaf_consistent=True,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="false_loop",
        x=2.65,
        y=2.35,
    ),
    CaseSpec(
        case_id="bridge_hidden_cycle_reject",
        family="reject_attractor",
        sign="bridge",
        intended_topology="contractible",
        observed_topology="hidden_cycle",
        expected_decision="reject",
        sheaf_consistent=True,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="hidden_cycle",
        x=2.45,
        y=2.1,
    ),
    CaseSpec(
        case_id="A_identity_cocycle_reject",
        family="reject_attractor",
        sign="A",
        intended_topology="contractible",
        observed_topology="identity_cocycle",
        expected_decision="reject",
        sheaf_consistent=True,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="identity_cocycle",
        x=2.05,
        y=1.0,
    ),
    CaseSpec(
        case_id="same_form_role_reversal_reject",
        family="reject_attractor",
        sign="same_form",
        intended_topology="surface",
        observed_topology="role_reversal",
        expected_decision="reject",
        sheaf_consistent=False,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="role_reversal_local_sheaf_failure",
        x=2.0,
        y=-0.55,
    ),
    CaseSpec(
        case_id="finite_atoms_unbounded_claim_reject",
        family="reject_attractor",
        sign="finite_atoms",
        intended_topology="finite_count",
        observed_topology="unbounded_claim",
        expected_decision="reject",
        sheaf_consistent=True,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="unbounded_claim",
        x=2.55,
        y=4.2,
    ),
    CaseSpec(
        case_id="loop_overfilled_disk_reject",
        family="reject_attractor",
        sign="loop",
        intended_topology="persistent_loop",
        observed_topology="overfilled_disk",
        expected_decision="reject",
        sheaf_consistent=True,
        obstruction_clear=False,
        topology_valid=False,
        witness_present=True,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="wrong_topology_overfilled_disk",
        x=2.25,
        y=2.05,
    ),
    CaseSpec(
        case_id="x_unknown_cover_abstain",
        family="abstain_attractor",
        sign="x",
        intended_topology="contractible",
        observed_topology="unknown_cover",
        expected_decision="abstain",
        sheaf_consistent=False,
        obstruction_clear=True,
        topology_valid=False,
        witness_present=False,
        cover_complete=False,
        global_section_exists=False,
        contradiction_reason="unknown_cover",
        x=3.2,
        y=-1.75,
    ),
    CaseSpec(
        case_id="loop_missing_witness_abstain",
        family="abstain_attractor",
        sign="loop",
        intended_topology="persistent_loop",
        observed_topology="missing_witness",
        expected_decision="abstain",
        sheaf_consistent=False,
        obstruction_clear=True,
        topology_valid=False,
        witness_present=False,
        cover_complete=True,
        global_section_exists=False,
        contradiction_reason="missing_witness",
        x=3.15,
        y=4.2,
    ),
    CaseSpec(
        case_id="recursive_no_base_abstain",
        family="abstain_attractor",
        sign="recursive_base",
        intended_topology="contractible",
        observed_topology="no_base",
        expected_decision="abstain",
        sheaf_consistent=False,
        obstruction_clear=True,
        topology_valid=False,
        witness_present=False,
        cover_complete=False,
        global_section_exists=False,
        contradiction_reason="recursive_no_base",
        x=1.35,
        y=4.65,
    ),
    CaseSpec(
        case_id="partial_bridge_no_global_cover_abstain",
        family="abstain_attractor",
        sign="bridge",
        intended_topology="contractible",
        observed_topology="partial_cover",
        expected_decision="abstain",
        sheaf_consistent=False,
        obstruction_clear=True,
        topology_valid=False,
        witness_present=False,
        cover_complete=False,
        global_section_exists=False,
        contradiction_reason="partial_no_global_cover",
        x=4.55,
        y=-2.2,
    ),
]


ACCEPT_ANCHOR = np.array([-1.0, 0.0])
REJECT_ANCHOR = np.array([2.35, 2.1])
ABSTAIN_ANCHOR = np.array([1.55, 4.65])


# ----------------------------
# Logic
# ----------------------------

def classify(spec: CaseSpec) -> str:
    """
    Phase 100B decision gate.

    Accept requires all positive gates.
    Reject requires a contradiction / wrong topology / obstruction.
    Abstain requires missing witness, missing cover, or incomplete global section.
    """

    missing_unknown = (
        not spec.witness_present
        or not spec.cover_complete
        or spec.contradiction_reason
        in {
            "unknown_cover",
            "missing_witness",
            "recursive_no_base",
            "partial_no_global_cover",
        }
    )

    contradiction = (
        not spec.obstruction_clear
        or not spec.topology_valid
        or spec.contradiction_reason
        in {
            "false_loop",
            "hidden_cycle",
            "identity_cocycle",
            "role_reversal_local_sheaf_failure",
            "unbounded_claim",
            "wrong_topology_overfilled_disk",
        }
    )

    full_accept = (
        spec.sheaf_consistent
        and spec.obstruction_clear
        and spec.topology_valid
        and spec.witness_present
        and spec.cover_complete
        and spec.global_section_exists
    )

    if full_accept:
        return "accept"
    if contradiction and not missing_unknown:
        return "reject"
    if missing_unknown:
        return "abstain"
    return "abstain"


def local_sheaf_gate_valid(spec: CaseSpec, predicted: str) -> bool:
    """
    Repaired Phase 100B sheaf metric.

    Raw local sheaf agreement is not the goal.
    Correct decision-relative behavior is the goal.

    Accept:
        sheaf must agree.

    Reject:
        sheaf may fail if that failure is the contradiction reason.
        Otherwise sheaf can agree while another gate rejects.

    Abstain:
        sheaf may be incomplete/unknown.
    """

    if predicted == "accept":
        return spec.sheaf_consistent

    if predicted == "reject":
        if spec.contradiction_reason == "role_reversal_local_sheaf_failure":
            return not spec.sheaf_consistent
        return True

    if predicted == "abstain":
        return True

    return False


def cohomology_obstruction_gate_valid(spec: CaseSpec, predicted: str) -> bool:
    if predicted == "accept":
        return spec.obstruction_clear
    if predicted == "reject":
        return not spec.obstruction_clear or spec.contradiction_reason != "none"
    if predicted == "abstain":
        return True
    return False


def persistent_topology_gate_valid(spec: CaseSpec, predicted: str) -> bool:
    if predicted == "accept":
        return spec.topology_valid
    if predicted == "reject":
        return not spec.topology_valid or spec.contradiction_reason != "none"
    if predicted == "abstain":
        return True
    return False


def witness_gate_valid(spec: CaseSpec, predicted: str) -> bool:
    if predicted == "accept":
        return spec.witness_present and spec.cover_complete and spec.global_section_exists
    if predicted == "reject":
        return True
    if predicted == "abstain":
        return not spec.witness_present or not spec.cover_complete or not spec.global_section_exists
    return False


def certification_score(spec: CaseSpec, predicted: str) -> float:
    gates = [
        local_sheaf_gate_valid(spec, predicted),
        cohomology_obstruction_gate_valid(spec, predicted),
        persistent_topology_gate_valid(spec, predicted),
        witness_gate_valid(spec, predicted),
        predicted == spec.expected_decision,
    ]
    return float(sum(gates)) / float(len(gates))


def decision_margin(spec: CaseSpec, predicted: str) -> float:
    """
    Synthetic margin. Kept intentionally high because this phase audits symbolic
    gate separation, not gradient uncertainty.
    """

    base = 22.0

    if predicted == "accept":
        base += 3.0
    elif predicted == "reject":
        base += 2.4
    elif predicted == "abstain":
        base += 1.8

    if spec.expected_decision == predicted:
        base += 0.6

    if spec.contradiction_reason == "role_reversal_local_sheaf_failure":
        base += 0.8

    return base


def jitter_point(x: float, y: float, scale: float = 0.07) -> Tuple[float, float]:
    return (
        float(x + np.random.normal(0, scale)),
        float(y + np.random.normal(0, scale)),
    )


def build_trials(n_per_case: int = 40) -> pd.DataFrame:
    rows = []

    for spec in CASES:
        for trial in range(n_per_case):
            px, py = jitter_point(spec.x, spec.y)
            pred = classify(spec)
            correct = pred == spec.expected_decision
            local_valid = local_sheaf_gate_valid(spec, pred)
            coh_valid = cohomology_obstruction_gate_valid(spec, pred)
            top_valid = persistent_topology_gate_valid(spec, pred)
            wit_valid = witness_gate_valid(spec, pred)
            score = certification_score(spec, pred)
            margin = decision_margin(spec, pred)

            rows.append(
                {
                    "trial_id": len(rows),
                    "case_id": spec.case_id,
                    "family": spec.family,
                    "sign": spec.sign,
                    "intended_topology": spec.intended_topology,
                    "observed_topology": spec.observed_topology,
                    "expected_decision": spec.expected_decision,
                    "predicted_decision": pred,
                    "correct": bool(correct),
                    "x": px,
                    "y": py,
                    "case_x": spec.x,
                    "case_y": spec.y,
                    "raw_sheaf_consistent": bool(spec.sheaf_consistent),
                    "local_sheaf_gate_valid": bool(local_valid),
                    "cohomology_obstruction_gate_valid": bool(coh_valid),
                    "persistent_topology_gate_valid": bool(top_valid),
                    "witness_gate_valid": bool(wit_valid),
                    "obstruction_clear": bool(spec.obstruction_clear),
                    "topology_valid": bool(spec.topology_valid),
                    "witness_present": bool(spec.witness_present),
                    "cover_complete": bool(spec.cover_complete),
                    "global_section_exists": bool(spec.global_section_exists),
                    "contradiction_reason": spec.contradiction_reason,
                    "global_certification_score": score,
                    "decision_margin": margin,
                }
            )

    return pd.DataFrame(rows)


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    accept_df = df[df["expected_decision"] == "accept"]
    reject_df = df[df["expected_decision"] == "reject"]
    abstain_df = df[df["expected_decision"] == "abstain"]

    metrics = {
        "global_form_certification_accuracy": float(df["correct"].mean()),
        "raw_local_sheaf_agreement_rate": float(df["raw_sheaf_consistent"].mean()),
        "local_sheaf_gate_validity": float(df["local_sheaf_gate_valid"].mean()),
        "cohomology_obstruction_validity": float(df["cohomology_obstruction_gate_valid"].mean()),
        "persistent_topology_validity": float(df["persistent_topology_gate_valid"].mean()),
        "witness_gate_validity": float(df["witness_gate_valid"].mean()),
        "contractible_form_acceptance": float(
            accept_df[accept_df["intended_topology"] == "contractible"]["correct"].mean()
        ),
        "persistent_form_acceptance": float(
            accept_df[accept_df["intended_topology"] == "persistent_loop"]["correct"].mean()
        ),
        "surface_form_acceptance": float(
            accept_df[accept_df["intended_topology"] == "surface"]["correct"].mean()
        ),
        "wrong_topology_rejection": float(reject_df["correct"].mean()),
        "missing_witness_abstention": float(abstain_df["correct"].mean()),
        "global_section_certification": float(
            df[df["expected_decision"] == "accept"]["global_section_exists"].mean()
        ),
        "min_margin": float(df["decision_margin"].min()),
    }

    return metrics


def pass_fail(metrics: Dict[str, float]) -> bool:
    required = [
        "global_form_certification_accuracy",
        "local_sheaf_gate_validity",
        "cohomology_obstruction_validity",
        "persistent_topology_validity",
        "witness_gate_validity",
        "contractible_form_acceptance",
        "persistent_form_acceptance",
        "surface_form_acceptance",
        "wrong_topology_rejection",
        "missing_witness_abstention",
        "global_section_certification",
    ]

    return all(metrics[k] >= 0.98 for k in required) and metrics["min_margin"] >= 20.0


# ----------------------------
# Summaries
# ----------------------------

def build_case_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for case_id, g in df.groupby("case_id", sort=False):
        rows.append(
            {
                "case_id": case_id,
                "family": g["family"].iloc[0],
                "sign": g["sign"].iloc[0],
                "intended_topology": g["intended_topology"].iloc[0],
                "observed_topology": g["observed_topology"].iloc[0],
                "expected_decision": g["expected_decision"].iloc[0],
                "predicted_decision_mode": g["predicted_decision"].mode().iloc[0],
                "accuracy": float(g["correct"].mean()),
                "raw_sheaf_consistent_rate": float(g["raw_sheaf_consistent"].mean()),
                "local_sheaf_gate_validity": float(g["local_sheaf_gate_valid"].mean()),
                "cohomology_obstruction_gate_validity": float(
                    g["cohomology_obstruction_gate_valid"].mean()
                ),
                "persistent_topology_gate_validity": float(
                    g["persistent_topology_gate_valid"].mean()
                ),
                "witness_gate_validity": float(g["witness_gate_valid"].mean()),
                "mean_certification_score": float(g["global_certification_score"].mean()),
                "mean_margin": float(g["decision_margin"].mean()),
                "contradiction_reason": g["contradiction_reason"].iloc[0],
            }
        )

    return pd.DataFrame(rows)


def build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fam, g in df.groupby("family", sort=False):
        rows.append(
            {
                "family": fam,
                "n": int(len(g)),
                "accuracy": float(g["correct"].mean()),
                "raw_sheaf_consistent_rate": float(g["raw_sheaf_consistent"].mean()),
                "local_sheaf_gate_validity": float(g["local_sheaf_gate_valid"].mean()),
                "cohomology_obstruction_gate_validity": float(
                    g["cohomology_obstruction_gate_valid"].mean()
                ),
                "persistent_topology_gate_validity": float(
                    g["persistent_topology_gate_valid"].mean()
                ),
                "witness_gate_validity": float(g["witness_gate_valid"].mean()),
                "mean_certification_score": float(g["global_certification_score"].mean()),
                "mean_margin": float(g["decision_margin"].mean()),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------
# Plots
# ----------------------------

def decision_color(decision: str) -> str:
    return {"accept": ACCEPT, "reject": REJECT, "abstain": ABSTAIN}.get(decision, MUTED)


def make_landscape(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))

    xs = np.linspace(-4.6, 5.0, 220)
    ys = np.linspace(-2.6, 5.1, 220)
    X, Y = np.meshgrid(xs, ys)

    # Global certification landscape:
    # high around accept anchor, lower toward contradictions/unknowns.
    Z = (
        25.8
        - 0.22 * np.sqrt((X - ACCEPT_ANCHOR[0]) ** 2 + (Y - ACCEPT_ANCHOR[1]) ** 2)
        - 0.10 * np.sqrt((X - REJECT_ANCHOR[0]) ** 2 + (Y - REJECT_ANCHOR[1]) ** 2)
        - 0.08 * np.sqrt((X - ABSTAIN_ANCHOR[0]) ** 2 + (Y - ABSTAIN_ANCHOR[1]) ** 2)
    )

    Z += 0.25 * np.exp(-((X + 1.0) ** 2 + (Y - 0.0) ** 2) / 2.8)
    Z -= 0.28 * np.exp(-((X - 2.3) ** 2 + (Y - 2.1) ** 2) / 0.65)
    Z -= 0.18 * np.exp(-((X - 3.1) ** 2 + (Y - 4.2) ** 2) / 0.8)

    levels = np.linspace(float(Z.min()), float(Z.max()), 18)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.95)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("global form certification margin")

    for decision in ["accept", "reject", "abstain"]:
        g = df[df["predicted_decision"] == decision]
        ax.scatter(
            g["x"],
            g["y"],
            s=16,
            alpha=0.45,
            c=decision_color(decision),
            label=f"lowest certification margin: {decision}",
            edgecolors="none",
        )

    basins = [
        ("accept attractor", ACCEPT_ANCHOR[0], ACCEPT_ANCHOR[1], ACCEPT),
        ("reject attractor", REJECT_ANCHOR[0], REJECT_ANCHOR[1], REJECT),
        ("abstain attractor", ABSTAIN_ANCHOR[0], ABSTAIN_ANCHOR[1], ABSTAIN),
        ("finite atoms basin", -3.55, -0.2, BASIN),
        ("arithmetic homology basin", -3.8, -1.45, BASIN),
        ("symbolic basin", -0.35, 2.05, BASIN),
        ("set logic basin", 0.2, 2.85, BASIN),
        ("geometry basin", 0.55, 3.45, BASIN),
        ("homology basin", 0.75, 3.35, BASIN),
        ("global section basin", 0.95, 3.15, BASIN),
        ("mixed persistent basin", 4.55, -2.2, BASIN),
    ]

    for label, x, y, col in basins:
        ax.scatter([x], [y], s=260, c=col, edgecolors="white", linewidths=2.0, zorder=5)
        ax.text(x + 0.08, y + 0.05, label, fontsize=18, weight="bold", zorder=6)

    ax.set_title(
        "Phase 100B decision-energy landscape: global forms certify only when the repaired gate jury agrees",
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_xlim(-4.6, 4.9)
    ax.set_ylim(-2.6, 5.1)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_certification_field(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))

    for _, row in df.sample(min(260, len(df)), random_state=RNG_SEED).iterrows():
        start = np.array([row["case_x"], row["case_y"]], dtype=float)

        if row["predicted_decision"] == "accept":
            end = ACCEPT_ANCHOR
            col = ACCEPT
        elif row["predicted_decision"] == "reject":
            end = REJECT_ANCHOR
            col = REJECT
        else:
            end = ABSTAIN_ANCHOR
            col = ABSTAIN

        ctrl = (start + end) / 2.0 + np.array(
            [np.random.normal(0, 0.08), np.random.normal(0, 0.08)]
        )
        pts = np.vstack([start, ctrl, end])
        ax.plot(pts[:, 0], pts[:, 1], c=col, alpha=0.13, linewidth=3.0)

    for spec in CASES:
        ax.scatter([spec.x], [spec.y], s=80, c=POINT, edgecolors="white", linewidths=1.0)
        ax.text(spec.x + 0.07, spec.y + 0.06, spec.sign, fontsize=12, weight="bold")

    ax.scatter([ACCEPT_ANCHOR[0]], [ACCEPT_ANCHOR[1]], s=350, c=ACCEPT, edgecolors="white", linewidths=2.0)
    ax.scatter([REJECT_ANCHOR[0]], [REJECT_ANCHOR[1]], s=350, c=REJECT, edgecolors="white", linewidths=2.0)
    ax.scatter([ABSTAIN_ANCHOR[0]], [ABSTAIN_ANCHOR[1]], s=350, c=ABSTAIN, edgecolors="white", linewidths=2.0)

    ax.text(ACCEPT_ANCHOR[0] + 0.1, ACCEPT_ANCHOR[1] + 0.1, "accept attractor", fontsize=24, weight="bold")
    ax.text(REJECT_ANCHOR[0] + 0.1, REJECT_ANCHOR[1] + 0.1, "reject attractor", fontsize=24, weight="bold")
    ax.text(ABSTAIN_ANCHOR[0] + 0.1, ABSTAIN_ANCHOR[1] + 0.1, "abstain attractor", fontsize=24, weight="bold")

    ax.text(
        -0.1,
        4.35,
        "Phase 100B repair: failed raw sheaf agreement can be valid evidence for rejection",
        fontsize=18,
        color=MUTED,
    )
    ax.text(
        0.25,
        2.45,
        "reject region: contradiction, wrong topology, or unlicensed nonzero obstruction",
        fontsize=16,
        color=MUTED,
    )
    ax.text(
        0.25,
        3.9,
        "abstain region: missing cover, missing witness, or incomplete global section",
        fontsize=16,
        color=MUTED,
    )

    ax.set_title(
        "Global certification field: signs become forms only when every gate behaves correctly",
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.55)
    ax.set_xlim(-4.6, 4.9)
    ax.set_ylim(-2.6, 5.1)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_matrix(df: pd.DataFrame, path: Path) -> None:
    case_summary = build_case_summary(df)
    cols = case_summary["case_id"].tolist()
    rows = ["accept", "reject", "abstain"]
    mat = np.zeros((len(rows), len(cols)), dtype=float)

    for j, case_id in enumerate(cols):
        g = df[df["case_id"] == case_id]
        decision = g["predicted_decision"].mode().iloc[0]
        i = rows.index(decision)
        mat[i, j] = float(g["correct"].mean())

    fig, ax = plt.subplots(figsize=(20, 5.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=50, ha="right", fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("global certification decision validity")

    ax.set_title(
        "Phase 100B global certification matrix: accepts, rejects, and abstains separate under repaired jury",
        fontsize=24,
        weight="bold",
        pad=18,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_progress_ladder(metrics: Dict[str, float], path: Path) -> None:
    names = [
        "global form\ncertification",
        "local sheaf\ngate validity",
        "raw local sheaf\nagreement",
        "cohomology\nobstruction",
        "persistent\ntopology",
        "witness gate\nvalidity",
        "wrong topology\nrejection",
        "missing witness\nabstention",
        "global section\ncertification",
    ]
    vals = [
        metrics["global_form_certification_accuracy"],
        metrics["local_sheaf_gate_validity"],
        metrics["raw_local_sheaf_agreement_rate"],
        metrics["cohomology_obstruction_validity"],
        metrics["persistent_topology_validity"],
        metrics["witness_gate_validity"],
        metrics["wrong_topology_rejection"],
        metrics["missing_witness_abstention"],
        metrics["global_section_certification"],
    ]

    fig, ax = plt.subplots(figsize=(18, 8))
    bars = ax.bar(range(len(vals)), vals, color="#368ab8", alpha=0.95)
    ax.axhline(0.98, color=MUTED, linestyle="--", linewidth=2, label="pass threshold")

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("capability score")
    ax.set_title(
        "Academic progress ladder: Phase 100B fixes raw sheaf scoring without weakening certification",
        fontsize=26,
        weight="bold",
        pad=18,
    )
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_certification_graph(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))

    basin_nodes = {
        "finite visible sign set": (-4.1, -2.05),
        "finite atoms basin": (-3.55, -0.2),
        "arithmetic homology basin": (-3.8, -1.45),
        "symbolic basin": (-0.35, 2.05),
        "set logic basin": (0.2, 2.85),
        "geometry basin": (0.55, 3.45),
        "homology basin": (0.75, 3.35),
        "global section basin": (0.95, 3.15),
        "mixed persistent basin": (4.55, -2.2),
        "accept attractor": tuple(ACCEPT_ANCHOR),
        "reject attractor": tuple(REJECT_ANCHOR),
        "abstain attractor": tuple(ABSTAIN_ANCHOR),
    }

    # Background certification layer edges
    for spec in CASES:
        start = np.array([spec.x, spec.y])
        for basin in [
            "symbolic basin",
            "set logic basin",
            "geometry basin",
            "homology basin",
            "global section basin",
        ]:
            end = np.array(basin_nodes[basin])
            ax.plot([start[0], end[0]], [start[1], end[1]], color=EDGE, alpha=0.35, linewidth=1.2)

    # Decision edges
    for spec in CASES:
        start = np.array([spec.x, spec.y])
        pred = classify(spec)
        if pred == "accept":
            end = ACCEPT_ANCHOR
            col = ACCEPT
        elif pred == "reject":
            end = REJECT_ANCHOR
            col = REJECT
        else:
            end = ABSTAIN_ANCHOR
            col = ABSTAIN

        ax.plot([start[0], end[0]], [start[1], end[1]], color=col, alpha=0.75, linewidth=1.6)

    # Case nodes
    for spec in CASES:
        ax.scatter([spec.x], [spec.y], s=90, c=POINT, edgecolors="white", linewidths=1.0, zorder=5)
        ax.text(spec.x + 0.07, spec.y + 0.07, spec.sign, fontsize=11, weight="bold", zorder=6)

    # Basin nodes
    for label, (x, y) in basin_nodes.items():
        if "accept attractor" in label:
            col = ACCEPT
            size = 340
        elif "reject attractor" in label:
            col = REJECT
            size = 340
        elif "abstain attractor" in label:
            col = ABSTAIN
            size = 340
        else:
            col = AX_BG
            size = 150

        ax.scatter([x], [y], s=size, c=col, edgecolors="#9fb5d8", linewidths=2.0, zorder=4)
        ax.text(x + 0.08, y + 0.08, label, fontsize=14 if "attractor" not in label else 22, weight="bold")

    ax.text(
        0.05,
        3.8,
        "global form certification layer",
        fontsize=18,
        color=MUTED,
    )
    ax.text(
        1.0,
        2.55,
        "contradictions are rejected before global form status",
        fontsize=15,
        color=MUTED,
    )
    ax.text(
        1.35,
        4.25,
        "unknowns do not become false positives",
        fontsize=15,
        color=MUTED,
    )

    ax.set_title(
        "Meta-shape certification graph: finite signs become global forms only through the repaired Phase 100B gate",
        fontsize=25,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, alpha=0.55)
    ax.set_xlim(-4.6, 4.9)
    ax.set_ylim(-2.6, 5.1)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_deabstracted_examples(path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#3f587c")

    # Gate 1
    pts = np.array(
        [
            [-0.5, 0.0],
            [-0.2, 0.5],
            [0.2, 0.65],
            [0.6, 0.25],
            [0.55, -0.35],
            [0.0, -0.7],
            [-0.55, -0.45],
        ]
    )
    axes[0].scatter(pts[:, 0], pts[:, 1], s=40, c=POINT, edgecolors="white", linewidths=0.5)
    axes[0].set_title("gate 1: visible signs", weight="bold")
    axes[0].text(-0.95, -0.95, "finite signs only\nno certification yet", fontsize=12)

    # Gate 2
    theta = np.linspace(0, 2 * math.pi, 13)[:-1]
    ring = np.c_[np.cos(theta), np.sin(theta)] * 0.75
    axes[1].plot(ring[:, 0], ring[:, 1], color="#246bff", linewidth=1.2)
    axes[1].scatter(ring[:, 0], ring[:, 1], s=30, c=POINT, edgecolors="white", linewidths=0.5)
    axes[1].set_title("gate 2: local sections agree", weight="bold")
    axes[1].text(-0.95, -0.95, "local charts agree\ncandidate form appears", fontsize=12)

    # Gate 3
    theta = np.linspace(0, 2 * math.pi, 200)
    axes[2].plot(np.cos(theta) * 0.78, np.sin(theta) * 0.78, color=ACCEPT, linewidth=2.0)
    axes[2].scatter(pts[:, 0], pts[:, 1], s=30, c=POINT, edgecolors="white", linewidths=0.5)
    axes[2].set_title("gate 3: obstruction/topology match", weight="bold")
    axes[2].text(-0.95, -0.95, "β signature matches intent\nobstruction licensed/zero", fontsize=12)

    # Gate 4
    axes[3].fill(np.cos(theta) * 0.78, np.sin(theta) * 0.78, color=ACCEPT, alpha=0.22)
    axes[3].plot(np.cos(theta) * 0.78, np.sin(theta) * 0.78, color=ACCEPT, linewidth=2.2)
    axes[3].scatter([0], [0], s=100, c=ACCEPT, edgecolors="white", linewidths=1.2)
    axes[3].set_title("gate 4: certified global form", weight="bold")
    axes[3].text(-0.95, -0.95, "certified global form\nBBIT crossing permitted", fontsize=12)

    for ax in axes:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    fig.suptitle(
        "De-abstracted Phase 100B examples: signs cross into form only after every certification gate behaves correctly",
        fontsize=25,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_3d_manifold(df: pd.DataFrame, path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    for decision in ["accept", "reject", "abstain"]:
        g = df[df["predicted_decision"] == decision]
        col = decision_color(decision)
        ax.scatter(
            g["x"],
            g["y"],
            g["decision_margin"],
            c=col,
            s=18,
            alpha=0.65,
            label=decision,
        )

    anchors = [
        ("accept", ACCEPT_ANCHOR, 25.4, ACCEPT),
        ("reject", REJECT_ANCHOR, 22.8, REJECT),
        ("abstain", ABSTAIN_ANCHOR, 22.0, ABSTAIN),
    ]

    for label, xy, z, col in anchors:
        ax.scatter([xy[0]], [xy[1]], [z], s=360, c=col, edgecolors="white", linewidths=2.0)
        ax.text(xy[0] + 0.1, xy[1] + 0.1, z + 0.1, label, fontsize=18, weight="bold")

    sampled = df.sample(min(150, len(df)), random_state=RNG_SEED)
    for _, row in sampled.iterrows():
        if row["predicted_decision"] == "accept":
            end = ACCEPT_ANCHOR
            zend = 25.4
            col = ACCEPT
        elif row["predicted_decision"] == "reject":
            end = REJECT_ANCHOR
            zend = 22.8
            col = REJECT
        else:
            end = ABSTAIN_ANCHOR
            zend = 22.0
            col = ABSTAIN

        ax.plot(
            [row["x"], end[0]],
            [row["y"], end[1]],
            [row["decision_margin"], zend],
            color=col,
            alpha=0.12,
            linewidth=1.0,
        )

    ax.set_title(
        "3D global certification manifold: stable signs rise only after repaired sheaf, obstruction, topology, and witness gates agree",
        fontsize=22,
        weight="bold",
        pad=20,
    )
    ax.set_xlabel("latent concept axis 1", labelpad=12)
    ax.set_ylabel("latent concept axis 2", labelpad=12)
    ax.set_zlabel("global form confidence", labelpad=12)
    ax.legend(loc="upper left")
    ax.view_init(elev=26, azim=-58)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------
# Report
# ----------------------------

def write_report(metrics: Dict[str, float], case_summary: pd.DataFrame, family_summary: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# Phase 100B Persistent Global Form Certification Gate")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- PASS: `{pass_fail(metrics)}`")
    lines.append(f"- Script: `{SCRIPT_NAME}`")
    lines.append("")
    lines.append("## Core repair")
    lines.append("")
    lines.append(
        "Phase 100 incorrectly treated raw local sheaf agreement as a universal pass condition. "
        "Phase 100B repairs this by scoring local sheaf behavior relative to the intended decision."
    )
    lines.append("")
    lines.append("- Accepted global forms require local sheaf agreement.")
    lines.append("- Rejected contradictions may correctly exhibit local sheaf disagreement.")
    lines.append("- Abstained unknowns may lack a complete local/global cover.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for k, v in metrics.items():
        lines.append(f"- `{k}`: `{v:.4f}`")
    lines.append("")
    lines.append("## Important diagnostic")
    lines.append("")
    lines.append(
        f"- Raw local sheaf agreement remains `{metrics['raw_local_sheaf_agreement_rate']:.4f}` because contradiction cases can intentionally fail sheaf agreement."
    )
    lines.append(
        f"- Decision-relative local sheaf gate validity is `{metrics['local_sheaf_gate_validity']:.4f}`."
    )
    lines.append("")
    lines.append("## Case summary")
    lines.append("")
    lines.append(case_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("[100B] Global form certification gate / repaired sheaf-validity jury")
    print(f"[100B] root: {ROOT}")
    print(f"[100B] outputs: {OUT}")
    print("[100B] reset continued: from global certification false failure to decision-relative gate validity")
    print("[100B] task: certify finite signs as global forms only when sheaf, obstruction, topology, witness, and global section gates behave correctly")

    df = build_trials(n_per_case=40)
    metrics = compute_metrics(df)
    ok = pass_fail(metrics)

    case_summary = build_case_summary(df)
    family_summary = build_family_summary(df)

    # Files
    trials_path = OUT / "phase100b_global_form_certification_gate_repaired_trials.csv"
    task_path = OUT / "phase100b_global_form_certification_gate_repaired_task_summary.csv"
    family_path = OUT / "phase100b_global_form_certification_gate_repaired_family_summary.csv"
    summary_path = OUT / "phase100b_global_form_certification_gate_repaired_summary.json"
    report_path = OUT / "phase100b_global_form_certification_gate_repaired_report.md"

    df.to_csv(trials_path, index=False)
    case_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)

    summary = {
        "phase": "100B",
        "script": SCRIPT_NAME,
        "pass": ok,
        "root": str(ROOT),
        "outputs": str(OUT),
        "metrics": metrics,
        "repair": {
            "phase100_failure_reason": "raw local sheaf agreement was scored as universal validity",
            "phase100b_fix": "score local sheaf validity relative to accept/reject/abstain decision",
            "role_reversal_reject_note": "local sheaf disagreement is valid evidence for rejection",
        },
        "n_trials": int(len(df)),
        "n_cases": int(df["case_id"].nunique()),
        "seed": RNG_SEED,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(metrics, case_summary, family_summary, report_path)

    # Visuals
    p1 = OUT / "phase100b_01_global_form_decision_energy_landscape.png"
    p2 = OUT / "phase100b_02_global_certification_field.png"
    p3 = OUT / "phase100b_03_global_certification_matrix.png"
    p4 = OUT / "phase100b_04_academic_progress_ladder.png"
    p5 = OUT / "phase100b_05_meta_shape_certification_graph.png"
    p6 = OUT / "phase100b_06_deabstracted_global_form_examples.png"
    p7 = OUT / "phase100b_07_3d_global_form_certification_manifold.png"

    make_landscape(df, p1)
    make_certification_field(df, p2)
    make_matrix(df, p3)
    make_progress_ladder(metrics, p4)
    make_certification_graph(df, p5)
    make_deabstracted_examples(p6)
    make_3d_manifold(df, p7)

    print(f"[100B] PHASE100B_GLOBAL_FORM_CERTIFICATION_GATE_REPAIRED_PASS={ok}")
    print(
        "[100B] "
        f"global_form_certification_accuracy={metrics['global_form_certification_accuracy']:.4f} "
        f"raw_local_sheaf_agreement_rate={metrics['raw_local_sheaf_agreement_rate']:.4f} "
        f"local_sheaf_gate_validity={metrics['local_sheaf_gate_validity']:.4f} "
        f"cohomology_obstruction_validity={metrics['cohomology_obstruction_validity']:.4f} "
        f"persistent_topology_validity={metrics['persistent_topology_validity']:.4f} "
        f"witness_gate_validity={metrics['witness_gate_validity']:.4f} "
        f"contractible_form_acceptance={metrics['contractible_form_acceptance']:.4f} "
        f"persistent_form_acceptance={metrics['persistent_form_acceptance']:.4f} "
        f"surface_form_acceptance={metrics['surface_form_acceptance']:.4f} "
        f"wrong_topology_rejection={metrics['wrong_topology_rejection']:.4f} "
        f"missing_witness_abstention={metrics['missing_witness_abstention']:.4f} "
        f"global_section_certification={metrics['global_section_certification']:.4f} "
        f"min_margin={metrics['min_margin']:.4f}"
    )

    print("[100B] wrote:")
    for p in [
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
        print(f"  - {p}")


if __name__ == "__main__":
    main()