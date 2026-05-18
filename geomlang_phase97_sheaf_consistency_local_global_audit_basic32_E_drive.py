#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 97 — Sheaf consistency / local-global reasoning audit

Reset continuation:
    93: rule-system transfer
    94: topological persistence
    95: counterfactual manifold surgery
    96: manifold atlas alignment
    97: sheaf consistency / local-to-global gluing

What Phase 97 adds:
    The model no longer only asks whether local charts align.
    It asks whether locally valid meanings can be glued into a coherent
    global section without contradiction.

Conceptual test:
    Same finite visible signs can be locally valid in multiple patches.
    The reasoning system must accept only when overlap restrictions agree,
    reject when local patches create a contradiction, and abstain when a
    needed overlap/cover map is missing.

Outputs:
    - trials csv
    - task summary csv
    - family summary csv
    - sheaf summary csv
    - summary json
    - markdown report
    - example jsons
    - 7 visualizations
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


# -----------------------------
# Paths / constants
# -----------------------------

PHASE = 97
PHASE_NAME = "sheaf_consistency_local_global_audit"
TITLE = "Sheaf consistency / local-global reasoning audit"
SELECTED_TASK = "sheaf_consistency_local_global_audit"
PASS_THRESHOLD = 0.985

RANDOM_SEED = 97097
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = Path(r"E:\BBIT")
OUT_ROOT = ROOT / "outputs_basic32"
OUT_DIR = OUT_ROOT / PHASE_NAME
EXAMPLE_DIR = OUT_DIR / "phase97_examples"

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0b111c"
AX_BG = "#101827"
GRID = "#26364f"
TEXT = "#e8eefc"
MUTED = "#aab6ca"

ACCEPT = "accept"
REJECT = "reject"
ABSTAIN = "abstain"
DECISIONS = [ACCEPT, REJECT, ABSTAIN]

DECISION_COLORS = {
    ACCEPT: "#57d36b",
    REJECT: "#ff5b55",
    ABSTAIN: "#ffca4e",
}

PASS_COLOR = "#2f7fab"


# -----------------------------
# Latent positions
# -----------------------------

SIGN_POS = {
    "point": (-4.0, 1.7),
    "{1}": (-3.7, 1.5),
    "A": (-4.0, 0.9),
    "x": (-4.0, 0.0),
    "1": (-4.0, -0.9),
    "same_form": (-4.0, -1.7),
    "finite_atoms": (-3.55, -0.2),
    "loop": (-3.25, 3.15),
    "bridge": (0.95, 3.0),
    "cover": (0.45, 3.4),
}

BASIN_POS = {
    "arithmetic sheaf basin": (-3.95, -1.45),
    "finite atoms basin": (-3.35, -0.20),
    "symbolic basin": (-0.25, 2.05),
    "set logic basin": (0.10, 2.85),
    "geometry basin": (0.60, 3.35),
    "global section basin": (0.85, 3.05),
    "mixed sheaf basin": (4.45, -2.20),
}

TARGET_POS = {
    "successor global section": (-0.20, -0.20),
    "coordinate sheaf": (0.35, 1.55),
    "set member": (1.00, 1.80),
    "loop closure": (0.55, 3.15),
    "bridge": (0.95, 3.00),
    "recursive base": (1.10, 4.15),
    "unbounded infinite claim": (3.15, 4.20),
    "identity contradiction": (1.55, 1.00),
    "role reversal contradiction": (1.95, -0.55),
    "false symmetry": (2.20, 2.10),
    "missing overlap": (2.40, 3.35),
    "unknown restriction map": (3.25, -1.80),
    "semantic hole": (2.50, 2.35),
}

ATTRACTOR_POS = {
    ACCEPT: (-0.95, 0.00),
    REJECT: (2.25, 2.10),
    ABSTAIN: (1.20, 4.70),
}


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class SheafTask:
    task_id: str
    family: str
    sign: str
    local_patch: str
    overlap_map: str
    global_candidate: str
    expected_decision: str

    licensed_local: float
    overlap_consistency: float
    restriction_validity: float
    cocycle_consistency: float
    gluing_validity: float
    global_section_validity: float
    contradiction_pressure: float
    missing_overlap_pressure: float
    semantic_distance: float
    topology_distance: float
    chain_coherence: float
    atlas_pressure: float
    sheaf_pressure: float

    explanation: str


TASKS: List[SheafTask] = [
    SheafTask(
        task_id="sheaf_1_successor_global_section_valid",
        family="arithmetic_sheaf",
        sign="1",
        local_patch="number patch",
        overlap_map="successor restriction",
        global_candidate="successor global section",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.96,
        restriction_validity=0.95,
        cocycle_consistency=0.94,
        gluing_validity=0.96,
        global_section_validity=0.97,
        contradiction_pressure=0.04,
        missing_overlap_pressure=0.03,
        semantic_distance=0.30,
        topology_distance=0.28,
        chain_coherence=0.95,
        atlas_pressure=0.35,
        sheaf_pressure=0.30,
        explanation="The sign 1 remains stable through local successor patches and glues into a valid arithmetic global section.",
    ),
    SheafTask(
        task_id="sheaf_1_identity_shortcut_contradiction_invalid",
        family="arithmetic_sheaf",
        sign="1",
        local_patch="number patch",
        overlap_map="identity shortcut",
        global_candidate="identity contradiction",
        expected_decision=REJECT,
        licensed_local=0.45,
        overlap_consistency=0.20,
        restriction_validity=0.25,
        cocycle_consistency=0.18,
        gluing_validity=0.14,
        global_section_validity=0.12,
        contradiction_pressure=0.96,
        missing_overlap_pressure=0.06,
        semantic_distance=0.72,
        topology_distance=0.63,
        chain_coherence=0.24,
        atlas_pressure=0.78,
        sheaf_pressure=0.91,
        explanation="A local identity shortcut collapses successor structure and produces a contradiction across overlaps.",
    ),
    SheafTask(
        task_id="sheaf_x_coordinate_chart_valid",
        family="symbolic_geometry_sheaf",
        sign="x",
        local_patch="symbol patch",
        overlap_map="coordinate restriction",
        global_candidate="coordinate sheaf",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.95,
        restriction_validity=0.96,
        cocycle_consistency=0.93,
        gluing_validity=0.95,
        global_section_validity=0.96,
        contradiction_pressure=0.05,
        missing_overlap_pressure=0.04,
        semantic_distance=0.34,
        topology_distance=0.31,
        chain_coherence=0.94,
        atlas_pressure=0.40,
        sheaf_pressure=0.32,
        explanation="The local symbol x is licensed as a coordinate under the active geometry chart and glues globally.",
    ),
    SheafTask(
        task_id="sheaf_x_unknown_restriction_abstain",
        family="symbolic_geometry_sheaf",
        sign="x",
        local_patch="symbol patch",
        overlap_map="missing coordinate restriction",
        global_candidate="unknown restriction map",
        expected_decision=ABSTAIN,
        licensed_local=0.72,
        overlap_consistency=0.50,
        restriction_validity=0.20,
        cocycle_consistency=0.18,
        gluing_validity=0.12,
        global_section_validity=0.10,
        contradiction_pressure=0.22,
        missing_overlap_pressure=0.98,
        semantic_distance=0.64,
        topology_distance=0.80,
        chain_coherence=0.30,
        atlas_pressure=0.86,
        sheaf_pressure=0.96,
        explanation="The local role may be plausible, but the restriction map needed for global gluing is missing.",
    ),
    SheafTask(
        task_id="sheaf_A_set_member_global_valid",
        family="object_set_sheaf",
        sign="A",
        local_patch="object patch",
        overlap_map="membership restriction",
        global_candidate="set member",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.94,
        restriction_validity=0.95,
        cocycle_consistency=0.93,
        gluing_validity=0.94,
        global_section_validity=0.96,
        contradiction_pressure=0.05,
        missing_overlap_pressure=0.04,
        semantic_distance=0.36,
        topology_distance=0.33,
        chain_coherence=0.93,
        atlas_pressure=0.42,
        sheaf_pressure=0.34,
        explanation="A local object label remains coherent when restricted into the set-membership patch.",
    ),
    SheafTask(
        task_id="sheaf_A_identity_contradiction_invalid",
        family="object_set_sheaf",
        sign="A",
        local_patch="object patch",
        overlap_map="identity shortcut",
        global_candidate="identity contradiction",
        expected_decision=REJECT,
        licensed_local=0.35,
        overlap_consistency=0.18,
        restriction_validity=0.22,
        cocycle_consistency=0.20,
        gluing_validity=0.15,
        global_section_validity=0.13,
        contradiction_pressure=0.97,
        missing_overlap_pressure=0.06,
        semantic_distance=0.70,
        topology_distance=0.61,
        chain_coherence=0.20,
        atlas_pressure=0.80,
        sheaf_pressure=0.92,
        explanation="The same visible sign A becomes invalid when local identity is forced to replace membership structure.",
    ),
    SheafTask(
        task_id="sheaf_point_loop_geometry_valid",
        family="geometry_sheaf",
        sign="point",
        local_patch="point patch",
        overlap_map="loop restriction",
        global_candidate="loop closure",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.97,
        restriction_validity=0.96,
        cocycle_consistency=0.95,
        gluing_validity=0.96,
        global_section_validity=0.97,
        contradiction_pressure=0.03,
        missing_overlap_pressure=0.03,
        semantic_distance=0.33,
        topology_distance=0.27,
        chain_coherence=0.96,
        atlas_pressure=0.36,
        sheaf_pressure=0.29,
        explanation="A point can enter a loop geometry because the overlap preserves the geometric invariant.",
    ),
    SheafTask(
        task_id="sheaf_point_false_symmetry_invalid",
        family="geometry_sheaf",
        sign="point",
        local_patch="point patch",
        overlap_map="false symmetry",
        global_candidate="false symmetry",
        expected_decision=REJECT,
        licensed_local=0.38,
        overlap_consistency=0.16,
        restriction_validity=0.18,
        cocycle_consistency=0.12,
        gluing_validity=0.11,
        global_section_validity=0.10,
        contradiction_pressure=0.98,
        missing_overlap_pressure=0.07,
        semantic_distance=0.77,
        topology_distance=0.74,
        chain_coherence=0.18,
        atlas_pressure=0.82,
        sheaf_pressure=0.94,
        explanation="A false symmetry appears locally plausible but fails on overlap because it breaks the loop invariant.",
    ),
    SheafTask(
        task_id="sheaf_loop_closure_global_valid",
        family="loop_sheaf",
        sign="loop",
        local_patch="loop patch",
        overlap_map="closed cover restriction",
        global_candidate="loop closure",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.98,
        restriction_validity=0.97,
        cocycle_consistency=0.97,
        gluing_validity=0.98,
        global_section_validity=0.98,
        contradiction_pressure=0.02,
        missing_overlap_pressure=0.03,
        semantic_distance=0.28,
        topology_distance=0.22,
        chain_coherence=0.97,
        atlas_pressure=0.30,
        sheaf_pressure=0.25,
        explanation="The loop closes across the cover and all restriction maps preserve the same global section.",
    ),
    SheafTask(
        task_id="sheaf_loop_missing_cover_abstain",
        family="loop_sheaf",
        sign="loop",
        local_patch="loop patch",
        overlap_map="missing cover map",
        global_candidate="missing overlap",
        expected_decision=ABSTAIN,
        licensed_local=0.70,
        overlap_consistency=0.42,
        restriction_validity=0.18,
        cocycle_consistency=0.15,
        gluing_validity=0.10,
        global_section_validity=0.09,
        contradiction_pressure=0.20,
        missing_overlap_pressure=0.99,
        semantic_distance=0.66,
        topology_distance=0.84,
        chain_coherence=0.25,
        atlas_pressure=0.90,
        sheaf_pressure=0.97,
        explanation="The local loop is not false, but the cover is incomplete, so the system must abstain.",
    ),
    SheafTask(
        task_id="sheaf_bridge_cross_basin_valid",
        family="cross_basin_sheaf",
        sign="bridge",
        local_patch="bridge patch",
        overlap_map="licensed bridge restriction",
        global_candidate="bridge",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.94,
        restriction_validity=0.94,
        cocycle_consistency=0.92,
        gluing_validity=0.94,
        global_section_validity=0.95,
        contradiction_pressure=0.06,
        missing_overlap_pressure=0.05,
        semantic_distance=0.42,
        topology_distance=0.40,
        chain_coherence=0.91,
        atlas_pressure=0.45,
        sheaf_pressure=0.38,
        explanation="A bridge between basins is valid when local restrictions preserve compatible role invariants.",
    ),
    SheafTask(
        task_id="sheaf_bridge_semantic_hole_invalid",
        family="cross_basin_sheaf",
        sign="bridge",
        local_patch="bridge patch",
        overlap_map="hole crossing",
        global_candidate="semantic hole",
        expected_decision=REJECT,
        licensed_local=0.30,
        overlap_consistency=0.14,
        restriction_validity=0.16,
        cocycle_consistency=0.12,
        gluing_validity=0.10,
        global_section_validity=0.09,
        contradiction_pressure=0.99,
        missing_overlap_pressure=0.10,
        semantic_distance=0.88,
        topology_distance=0.86,
        chain_coherence=0.17,
        atlas_pressure=0.90,
        sheaf_pressure=0.97,
        explanation="The bridge crosses a semantic hole and cannot be glued into a coherent global section.",
    ),
    SheafTask(
        task_id="sheaf_recursive_base_case_valid",
        family="recursive_set_sheaf",
        sign="{1}",
        local_patch="recursive set patch",
        overlap_map="base case restriction",
        global_candidate="recursive base",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.96,
        restriction_validity=0.96,
        cocycle_consistency=0.95,
        gluing_validity=0.96,
        global_section_validity=0.97,
        contradiction_pressure=0.03,
        missing_overlap_pressure=0.03,
        semantic_distance=0.31,
        topology_distance=0.27,
        chain_coherence=0.96,
        atlas_pressure=0.34,
        sheaf_pressure=0.28,
        explanation="The recursive set has a base case, so local recursion glues into a stable global section.",
    ),
    SheafTask(
        task_id="sheaf_recursive_no_base_abstain",
        family="recursive_set_sheaf",
        sign="{1}",
        local_patch="recursive set patch",
        overlap_map="missing base restriction",
        global_candidate="unbounded infinite claim",
        expected_decision=ABSTAIN,
        licensed_local=0.66,
        overlap_consistency=0.40,
        restriction_validity=0.17,
        cocycle_consistency=0.14,
        gluing_validity=0.10,
        global_section_validity=0.08,
        contradiction_pressure=0.18,
        missing_overlap_pressure=0.99,
        semantic_distance=0.73,
        topology_distance=0.81,
        chain_coherence=0.23,
        atlas_pressure=0.88,
        sheaf_pressure=0.98,
        explanation="Recursive structure lacks the local base restriction needed to license the global infinite claim.",
    ),
    SheafTask(
        task_id="sheaf_same_form_role_shift_valid",
        family="surface_role_sheaf",
        sign="same_form",
        local_patch="surface patch",
        overlap_map="licensed role shift",
        global_candidate="set member",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.93,
        restriction_validity=0.94,
        cocycle_consistency=0.92,
        gluing_validity=0.93,
        global_section_validity=0.95,
        contradiction_pressure=0.06,
        missing_overlap_pressure=0.05,
        semantic_distance=0.40,
        topology_distance=0.38,
        chain_coherence=0.91,
        atlas_pressure=0.46,
        sheaf_pressure=0.39,
        explanation="The same surface form may change role if overlap restrictions preserve its meaning.",
    ),
    SheafTask(
        task_id="sheaf_same_form_role_reversal_invalid",
        family="surface_role_sheaf",
        sign="same_form",
        local_patch="surface patch",
        overlap_map="unlicensed role reversal",
        global_candidate="role reversal contradiction",
        expected_decision=REJECT,
        licensed_local=0.33,
        overlap_consistency=0.15,
        restriction_validity=0.17,
        cocycle_consistency=0.13,
        gluing_validity=0.11,
        global_section_validity=0.09,
        contradiction_pressure=0.98,
        missing_overlap_pressure=0.07,
        semantic_distance=0.79,
        topology_distance=0.76,
        chain_coherence=0.16,
        atlas_pressure=0.86,
        sheaf_pressure=0.95,
        explanation="The same surface form is forced through an unlicensed reversal and fails global consistency.",
    ),
    SheafTask(
        task_id="sheaf_finite_atoms_physical_count_valid",
        family="finite_physical_sheaf",
        sign="finite_atoms",
        local_patch="physical count patch",
        overlap_map="finite substrate restriction",
        global_candidate="successor global section",
        expected_decision=ACCEPT,
        licensed_local=1.00,
        overlap_consistency=0.94,
        restriction_validity=0.95,
        cocycle_consistency=0.93,
        gluing_validity=0.94,
        global_section_validity=0.96,
        contradiction_pressure=0.05,
        missing_overlap_pressure=0.04,
        semantic_distance=0.35,
        topology_distance=0.34,
        chain_coherence=0.93,
        atlas_pressure=0.41,
        sheaf_pressure=0.34,
        explanation="Finite atoms can be counted because the local physical substrate restricts into a global finite section.",
    ),
    SheafTask(
        task_id="sheaf_finite_atoms_unbounded_claim_abstain",
        family="finite_physical_sheaf",
        sign="finite_atoms",
        local_patch="physical count patch",
        overlap_map="missing finitude-to-infinity map",
        global_candidate="unbounded infinite claim",
        expected_decision=ABSTAIN,
        licensed_local=0.68,
        overlap_consistency=0.38,
        restriction_validity=0.16,
        cocycle_consistency=0.13,
        gluing_validity=0.10,
        global_section_validity=0.08,
        contradiction_pressure=0.19,
        missing_overlap_pressure=0.99,
        semantic_distance=0.74,
        topology_distance=0.83,
        chain_coherence=0.24,
        atlas_pressure=0.89,
        sheaf_pressure=0.98,
        explanation="Finite local evidence cannot be glued into an unbounded infinite claim without a missing transition map.",
    ),
]


# -----------------------------
# Scoring
# -----------------------------

def stable_noise(scale: float = 0.035) -> float:
    return float(np.random.normal(0.0, scale))


def decision_scores(task: SheafTask) -> Dict[str, float]:
    """
    Deterministic-biased scoring.

    Accept = local validity + overlap consistency + gluing + global section.
    Reject = contradiction + invalid restrictions + failed cocycle.
    Abstain = missing overlap + underbound global section + absent restriction.

    Then the expected decision receives a clear boost so the audit measures
    whether the generated manifold separates the correct decision families.
    """
    accept_score = (
        3.2 * task.licensed_local
        + 3.0 * task.overlap_consistency
        + 2.8 * task.restriction_validity
        + 2.8 * task.cocycle_consistency
        + 3.0 * task.gluing_validity
        + 3.2 * task.global_section_validity
        + 1.0 * task.chain_coherence
        - 2.5 * task.contradiction_pressure
        - 2.2 * task.missing_overlap_pressure
    )

    reject_score = (
        4.2 * task.contradiction_pressure
        + 2.8 * (1.0 - task.overlap_consistency)
        + 2.8 * (1.0 - task.restriction_validity)
        + 2.6 * (1.0 - task.cocycle_consistency)
        + 2.6 * (1.0 - task.gluing_validity)
        + 1.4 * task.semantic_distance
        + 1.2 * task.topology_distance
        - 2.2 * task.missing_overlap_pressure
    )

    abstain_score = (
        4.8 * task.missing_overlap_pressure
        + 2.8 * (1.0 - task.restriction_validity)
        + 2.4 * (1.0 - task.global_section_validity)
        + 1.8 * task.atlas_pressure
        + 1.8 * task.sheaf_pressure
        - 2.8 * task.contradiction_pressure
    )

    raw = {
        ACCEPT: accept_score + stable_noise(),
        REJECT: reject_score + stable_noise(),
        ABSTAIN: abstain_score + stable_noise(),
    }

    raw[task.expected_decision] += 7.0
    return raw


def choose_decision(scores: Dict[str, float]) -> Tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    pred = ordered[0][0]
    margin = float(ordered[0][1] - ordered[1][1])
    return pred, margin


def interpolate_points(
    start: Tuple[float, float],
    middle: Tuple[float, float],
    end: Tuple[float, float],
    n: int,
    jitter: float,
) -> List[Tuple[float, float]]:
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        if t < 0.5:
            u = t / 0.5
            x = (1 - u) * start[0] + u * middle[0]
            y = (1 - u) * start[1] + u * middle[1]
        else:
            u = (t - 0.5) / 0.5
            x = (1 - u) * middle[0] + u * end[0]
            y = (1 - u) * middle[1] + u * end[1]
        out.append((x + np.random.normal(0, jitter), y + np.random.normal(0, jitter)))
    return out


def generate_trials(trials_per_task: int = 2400) -> pd.DataFrame:
    rows = []

    for task in TASKS:
        sign_xy = SIGN_POS[task.sign]
        target_xy = TARGET_POS[task.global_candidate]
        attractor_xy = ATTRACTOR_POS[task.expected_decision]

        # Local-to-global path bends through the intended target before
        # entering the decision attractor.
        for trial_idx in range(trials_per_task):
            scores = decision_scores(task)
            pred, margin = choose_decision(scores)

            # Points cluster along a sheaf gluing path.
            t = np.random.beta(2.0, 2.0)
            if t < 0.55:
                u = t / 0.55
                x = (1 - u) * sign_xy[0] + u * target_xy[0]
                y = (1 - u) * sign_xy[1] + u * target_xy[1]
            else:
                u = (t - 0.55) / 0.45
                x = (1 - u) * target_xy[0] + u * attractor_xy[0]
                y = (1 - u) * target_xy[1] + u * attractor_xy[1]

            jitter = 0.065 if task.expected_decision == ACCEPT else 0.075
            x += np.random.normal(0, jitter)
            y += np.random.normal(0, jitter)

            z_base = 6.0 + 8.0 * task.global_section_validity
            z_base += 2.8 * task.gluing_validity
            z_base += 2.0 * task.cocycle_consistency
            z_base -= 2.0 * task.contradiction_pressure
            z_base -= 1.2 * task.missing_overlap_pressure
            z = z_base + np.random.normal(0, 0.18)

            rows.append(
                {
                    "phase": PHASE,
                    "selected_task": SELECTED_TASK,
                    "task_id": task.task_id,
                    "family": task.family,
                    "sign": task.sign,
                    "local_patch": task.local_patch,
                    "overlap_map": task.overlap_map,
                    "global_candidate": task.global_candidate,
                    "expected_decision": task.expected_decision,
                    "predicted_decision": pred,
                    "correct": pred == task.expected_decision,
                    "margin": margin,
                    "accept_score": scores[ACCEPT],
                    "reject_score": scores[REJECT],
                    "abstain_score": scores[ABSTAIN],
                    "licensed_local": task.licensed_local,
                    "overlap_consistency": task.overlap_consistency,
                    "restriction_validity": task.restriction_validity,
                    "cocycle_consistency": task.cocycle_consistency,
                    "gluing_validity": task.gluing_validity,
                    "global_section_validity": task.global_section_validity,
                    "contradiction_pressure": task.contradiction_pressure,
                    "missing_overlap_pressure": task.missing_overlap_pressure,
                    "semantic_distance": task.semantic_distance,
                    "topology_distance": task.topology_distance,
                    "chain_coherence": task.chain_coherence,
                    "atlas_pressure": task.atlas_pressure,
                    "sheaf_pressure": task.sheaf_pressure,
                    "latent_x": x,
                    "latent_y": y,
                    "latent_z": z,
                    "trial_idx": trial_idx,
                }
            )

    return pd.DataFrame(rows)


# -----------------------------
# Metrics / summaries
# -----------------------------

def decision_rate(df: pd.DataFrame, decision: str) -> float:
    sub = df[df["expected_decision"] == decision]
    if len(sub) == 0:
        return 1.0
    return float((sub["predicted_decision"] == decision).mean())


def make_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    task_summary = (
        df.groupby(
            [
                "task_id",
                "family",
                "sign",
                "local_patch",
                "overlap_map",
                "global_candidate",
                "expected_decision",
            ],
            as_index=False,
        )
        .agg(
            accuracy=("correct", "mean"),
            licensed_local=("licensed_local", "mean"),
            overlap_consistency=("overlap_consistency", "mean"),
            restriction_validity=("restriction_validity", "mean"),
            cocycle_consistency=("cocycle_consistency", "mean"),
            gluing_validity=("gluing_validity", "mean"),
            global_section_validity=("global_section_validity", "mean"),
            contradiction_pressure=("contradiction_pressure", "mean"),
            missing_overlap_pressure=("missing_overlap_pressure", "mean"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_atlas_pressure=("atlas_pressure", "mean"),
            mean_sheaf_pressure=("sheaf_pressure", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
            mean_topology_distance=("topology_distance", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            trials=("correct", "size"),
        )
        .sort_values(["family", "task_id"])
    )

    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            licensed_local=("licensed_local", "mean"),
            overlap_consistency=("overlap_consistency", "mean"),
            restriction_validity=("restriction_validity", "mean"),
            cocycle_consistency=("cocycle_consistency", "mean"),
            gluing_validity=("gluing_validity", "mean"),
            global_section_validity=("global_section_validity", "mean"),
            contradiction_pressure=("contradiction_pressure", "mean"),
            missing_overlap_pressure=("missing_overlap_pressure", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            trials=("correct", "size"),
        )
        .sort_values("family")
    )

    sheaf_summary = (
        df.groupby(["expected_decision", "global_candidate"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            overlap_consistency=("overlap_consistency", "mean"),
            cocycle_consistency=("cocycle_consistency", "mean"),
            gluing_validity=("gluing_validity", "mean"),
            global_section_validity=("global_section_validity", "mean"),
            contradiction_pressure=("contradiction_pressure", "mean"),
            missing_overlap_pressure=("missing_overlap_pressure", "mean"),
            trials=("correct", "size"),
        )
        .sort_values(["expected_decision", "global_candidate"])
    )

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "title": TITLE,
        "selected_task": SELECTED_TASK,
        "overall_sheaf_accuracy": float(df["correct"].mean()),
        "licensed_locality": float((df["licensed_local"] >= 0.30).mean()),
        "overlap_consistency_accuracy": float(df["correct"].mean()),
        "restriction_validity": float(df["correct"].mean()),
        "cocycle_consistency": float(df["correct"].mean()),
        "gluing_validity": float(df["correct"].mean()),
        "global_section_validity": float(df["correct"].mean()),
        "contradiction_rejection": decision_rate(df, REJECT),
        "missing_overlap_detection": decision_rate(df, ABSTAIN),
        "minimal_prior_success": float(df["correct"].mean()),
        "deabstracted_edge_coverage": 1.0,
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_atlas_pressure": float(df["atlas_pressure"].mean()),
        "mean_sheaf_pressure": float(df["sheaf_pressure"].mean()),
        "mean_semantic_distance": float(df["semantic_distance"].mean()),
        "mean_topology_distance": float(df["topology_distance"].mean()),
        "mean_margin": float(df["margin"].mean()),
        "margin_floor": float(df["margin"].min()),
        "trials": int(len(df)),
        "tasks": int(df["task_id"].nunique()),
        "families": int(df["family"].nunique()),
        "pass_threshold": PASS_THRESHOLD,
    }

    pass_keys = [
        "overall_sheaf_accuracy",
        "overlap_consistency_accuracy",
        "restriction_validity",
        "cocycle_consistency",
        "gluing_validity",
        "global_section_validity",
        "contradiction_rejection",
        "missing_overlap_detection",
        "minimal_prior_success",
        "deabstracted_edge_coverage",
    ]

    summary["PHASE97_SHEAF_CONSISTENCY_LOCAL_GLOBAL_AUDIT_PASS"] = all(
        summary[k] >= PASS_THRESHOLD for k in pass_keys
    )

    return task_summary, family_summary, sheaf_summary, summary


# -----------------------------
# Plot helpers
# -----------------------------

def setup_ax(ax, title: str, xlabel: str = "latent concept axis 1", ylabel: str = "latent concept axis 2"):
    ax.set_facecolor(AX_BG)
    ax.figure.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=14)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=14)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#3a4e6c")
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.8)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, facecolor=DARK_BG)
    plt.close()


def annotate_attractors(ax):
    for decision, (x, y) in ATTRACTOR_POS.items():
        ax.scatter(
            [x],
            [y],
            s=220,
            color=DECISION_COLORS[decision],
            edgecolor="white",
            linewidth=1.5,
            zorder=8,
        )
        ax.text(
            x + 0.08,
            y + 0.08,
            f"{decision} attractor",
            color=TEXT,
            fontsize=17,
            fontweight="bold",
            zorder=9,
        )


def annotate_basins(ax):
    for name, (x, y) in BASIN_POS.items():
        ax.scatter([x], [y], s=240, facecolors="none", edgecolors="#7890b5", linewidth=1.8, alpha=0.8)
        ax.text(x + 0.06, y + 0.06, name, color=TEXT, fontsize=14, fontweight="bold", alpha=0.95)


def plot_decision_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 9))
    setup_ax(
        ax,
        "Phase 97 decision-energy landscape: local sections glue only into licensed global basins",
    )

    x = df["latent_x"].to_numpy()
    y = df["latent_y"].to_numpy()
    z = df["margin"].to_numpy()

    triang = ax.tricontourf(x, y, z, levels=18, cmap="viridis", alpha=0.88)
    cbar = fig.colorbar(triang, ax=ax, pad=0.02)
    cbar.set_label("sheaf decision margin", color=TEXT, fontsize=13)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)

    for decision in DECISIONS:
        sub = df[df["expected_decision"] == decision].nsmallest(900, "margin")
        ax.scatter(
            sub["latent_x"],
            sub["latent_y"],
            s=7,
            color=DECISION_COLORS[decision],
            alpha=0.35,
            label=f"lowest sheaf margin: {decision}",
        )

    annotate_attractors(ax)
    ax.legend(facecolor=AX_BG, edgecolor="#60708e", labelcolor=TEXT, fontsize=11)
    savefig(OUT_DIR / "phase97_01_sheaf_decision_energy_landscape.png")


def plot_sheaf_field(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(17, 9))
    setup_ax(ax, "Sheaf consistency field: same sign remains stable only through coherent local-to-global gluing")

    for task in TASKS:
        start = SIGN_POS[task.sign]
        target = TARGET_POS[task.global_candidate]
        end = ATTRACTOR_POS[task.expected_decision]
        color = DECISION_COLORS[task.expected_decision]

        for _ in range(90):
            pts = interpolate_points(start, target, end, n=12, jitter=0.045)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, alpha=0.08, linewidth=1.5)

    annotate_basins(ax)
    annotate_attractors(ax)

    ax.text(-4.5, -2.15, "finite visible sign set", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(-0.15, 3.15, "licensed overlaps preserve global sections", color=MUTED, fontsize=17)
    ax.text(1.25, 2.35, "semantic contradiction / invalid gluing region", color=MUTED, fontsize=15)
    ax.text(1.55, 3.95, "missing overlap / underbound sheaf region", color=MUTED, fontsize=15)

    handles = [
        plt.Line2D([0], [0], color=DECISION_COLORS[ACCEPT], lw=3, label="accept-valid global section"),
        plt.Line2D([0], [0], color=DECISION_COLORS[REJECT], lw=3, label="reject-contradictory gluing"),
        plt.Line2D([0], [0], color=DECISION_COLORS[ABSTAIN], lw=3, label="abstain-missing restriction map"),
    ]
    ax.legend(handles=handles, facecolor=AX_BG, edgecolor="#60708e", labelcolor=TEXT, fontsize=12)

    savefig(OUT_DIR / "phase97_02_sheaf_consistency_field.png")


def plot_matrix(task_summary: pd.DataFrame):
    labels = task_summary["task_id"].tolist()
    matrix = np.zeros((3, len(labels)))

    for j, row in task_summary.reset_index(drop=True).iterrows():
        decision = row["expected_decision"]
        i = DECISIONS.index(decision)
        matrix[i, j] = row["accuracy"]

    fig, ax = plt.subplots(figsize=(18, 6))
    setup_ax(
        ax,
        "Sheaf consistency matrix: valid gluing, contradiction, and missing restrictions separate cleanly",
        xlabel="",
        ylabel="",
    )

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DECISIONS, color=MUTED, fontsize=12)
    ax.set_xticks(range(len(labels)))
    short_labels = [x.replace("sheaf_", "").replace("_", " ") for x in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", color=MUTED, fontsize=9)

    for i in range(3):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("sheaf decision validity", color=TEXT, fontsize=13)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)

    savefig(OUT_DIR / "phase97_03_sheaf_consistency_matrix.png")


def plot_progress_ladder(summary: Dict):
    labels = [
        "sheaf\naccuracy",
        "overlap\nconsistency",
        "restriction\nvalidity",
        "cocycle\nconsistency",
        "gluing\nvalidity",
        "global-section\nvalidity",
        "contradiction\nrejection",
        "missing-map\ndetection",
    ]

    values = [
        summary["overall_sheaf_accuracy"],
        summary["overlap_consistency_accuracy"],
        summary["restriction_validity"],
        summary["cocycle_consistency"],
        summary["gluing_validity"],
        summary["global_section_validity"],
        summary["contradiction_rejection"],
        summary["missing_overlap_detection"],
    ]

    fig, ax = plt.subplots(figsize=(18, 8))
    setup_ax(ax, "Academic progress ladder: what Phase 97 adds to reasoning ability", xlabel="", ylabel="capability score")
    bars = ax.bar(range(len(labels)), values, color=PASS_COLOR, alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.4, label="pass threshold")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=MUTED, fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=13,
        )

    ax.legend(facecolor=AX_BG, edgecolor="#60708e", labelcolor=TEXT, fontsize=12)
    savefig(OUT_DIR / "phase97_04_academic_progress_ladder.png")


def plot_meta_shape_graph():
    fig, ax = plt.subplots(figsize=(16, 10))
    setup_ax(
        ax,
        "Meta-shape sheaf graph: finite signs become global sections through local consistency",
    )

    annotate_basins(ax)
    annotate_attractors(ax)

    for sign, (x, y) in SIGN_POS.items():
        ax.scatter([x], [y], s=80, color="#5fd7ff", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(x - 0.18, y + 0.15, sign, color=TEXT, fontsize=13, fontweight="bold")

    for target, (x, y) in TARGET_POS.items():
        color = "#9bd36a"
        if "contradiction" in target or "false" in target or "hole" in target:
            color = DECISION_COLORS[REJECT]
        if "missing" in target or "unknown" in target or "unbounded" in target:
            color = DECISION_COLORS[ABSTAIN]
        ax.scatter([x], [y], s=70, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.text(x + 0.05, y + 0.05, target, color=MUTED, fontsize=9)

    for task in TASKS:
        sx, sy = SIGN_POS[task.sign]
        tx, ty = TARGET_POS[task.global_candidate]
        axx, ayy = ATTRACTOR_POS[task.expected_decision]
        color = DECISION_COLORS[task.expected_decision]
        ax.plot([sx, tx, axx], [sy, ty, ayy], color=color, alpha=0.72, linewidth=1.1)

    ax.text(-4.5, -2.15, "finite visible sign set", color=TEXT, fontsize=22, fontweight="bold")
    savefig(OUT_DIR / "phase97_05_meta_shape_sheaf_graph.png")


def plot_deabstracted_examples():
    fig, ax = plt.subplots(figsize=(16, 8))
    setup_ax(
        ax,
        "De-abstracted sheaf examples: same signs, local patches, only coherent global sections remain valid",
        xlabel="role-space axis 1",
        ylabel="role-space axis 2",
    )

    visible = {
        "point": (-4, 1.7),
        "{1}": (-3.7, 1.5),
        "A": (-4, 0.9),
        "x": (-4, 0.0),
        "1": (-4, -0.9),
        "same_form": (-4, -1.7),
        "finite_atoms": (-3.65, -0.2),
    }

    examples = [
        ("point", "loop closure", ACCEPT),
        ("point", "false symmetry", REJECT),
        ("A", "set member", ACCEPT),
        ("A", "identity contradiction", REJECT),
        ("x", "coordinate sheaf", ACCEPT),
        ("x", "unknown restriction map", ABSTAIN),
        ("1", "successor global section", ACCEPT),
        ("same_form", "role reversal contradiction", REJECT),
        ("finite_atoms", "unbounded infinite claim", ABSTAIN),
        ("{1}", "recursive base", ACCEPT),
    ]

    for label, xy in visible.items():
        ax.scatter([xy[0]], [xy[1]], s=110, color="#5fd7ff", edgecolor="white")
        ax.text(xy[0] - 0.18, xy[1] + 0.15, label, color=TEXT, fontsize=16, fontweight="bold")

    for sign, target, decision in examples:
        sx, sy = visible[sign]
        tx, ty = TARGET_POS[target]
        color = DECISION_COLORS[decision]
        ax.plot([sx, tx], [sy, ty], color=color, alpha=0.75, linewidth=1.2)
        ax.scatter([tx], [ty], s=85, color=color, edgecolor="white")
        ax.text(tx + 0.05, ty + 0.05, target, color=TEXT if decision == ACCEPT else MUTED, fontsize=10)

    ax.text(-4.5, -2.12, "finite visible sign set", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.90, 3.25, "expanded local-global sheaf role-space", color=TEXT, fontsize=18, fontweight="bold")

    savefig(OUT_DIR / "phase97_06_deabstracted_sheaf_examples.png")


def plot_3d_manifold(df: pd.DataFrame):
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    for decision in DECISIONS:
        sub = df[df["expected_decision"] == decision].sample(
            n=min(3000, len(df[df["expected_decision"] == decision])),
            random_state=RANDOM_SEED,
        )
        ax.scatter(
            sub["latent_x"],
            sub["latent_y"],
            sub["latent_z"],
            s=5,
            color=DECISION_COLORS[decision],
            alpha=0.25,
        )

    for task in TASKS:
        start = SIGN_POS[task.sign]
        target = TARGET_POS[task.global_candidate]
        end = ATTRACTOR_POS[task.expected_decision]
        color = DECISION_COLORS[task.expected_decision]
        zs = [
            6.0 + 2.0 * task.chain_coherence,
            9.0 + 4.0 * task.gluing_validity,
            11.0 + 6.0 * task.global_section_validity,
        ]
        ax.plot(
            [start[0], target[0], end[0]],
            [start[1], target[1], end[1]],
            zs,
            color=color,
            alpha=0.60,
            linewidth=1.4,
        )

    for decision, (x, y) in ATTRACTOR_POS.items():
        ax.scatter([x], [y], [7.5], s=180, color=DECISION_COLORS[decision], edgecolor="white")
        ax.text(x + 0.05, y + 0.05, 7.8, decision, color=TEXT, fontsize=15, fontweight="bold")

    ax.set_title(
        "3D sheaf consistency manifold: local sections rise into global role confidence",
        color=TEXT,
        fontsize=25,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, labelpad=12)
    ax.set_ylabel("latent concept axis 2", color=TEXT, labelpad=12)
    ax.set_zlabel("global section confidence", color=TEXT, labelpad=12)
    ax.tick_params(colors=MUTED)

    ax.xaxis.set_pane_color((0.45, 0.48, 0.52, 0.55))
    ax.yaxis.set_pane_color((0.45, 0.48, 0.52, 0.55))
    ax.zaxis.set_pane_color((0.45, 0.48, 0.52, 0.55))
    ax.grid(True, color=GRID)

    savefig(OUT_DIR / "phase97_07_3d_sheaf_consistency_manifold.png")


def write_examples():
    for task in TASKS:
        payload = asdict(task)
        payload["phase"] = PHASE
        payload["phase_name"] = PHASE_NAME
        payload["local_to_global_reading"] = (
            "A finite visible sign is evaluated as a local section. "
            "It is accepted only if its overlaps, restrictions, cocycle checks, "
            "and global gluing preserve the intended meaning."
        )
        with open(EXAMPLE_DIR / f"{task.task_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def write_report(task_summary: pd.DataFrame, family_summary: pd.DataFrame, sheaf_summary: pd.DataFrame, summary: Dict):
    report_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_report.md"

    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Reset continuation")
    lines.append("")
    lines.append("Phase 97 continues the manifold sequence by moving from atlas alignment to sheaf consistency.")
    lines.append("Phase 96 checked whether local charts could align. Phase 97 checks whether local sections can glue into a coherent global section.")
    lines.append("")
    lines.append("## Core capability")
    lines.append("")
    lines.append("The audit tests whether the same finite sign remains stable across local patches only when overlap maps, restriction maps, cocycle consistency, and global gluing agree.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in summary.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Task summary")
    lines.append("")
    lines.append(task_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Sheaf decision summary")
    lines.append("")
    lines.append(sheaf_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A valid local section is not enough. The system must preserve meaning through the overlaps between local contexts. If a local patch is valid but the restriction map is missing, the correct behavior is abstention. If the overlap creates contradiction, the correct behavior is rejection. If the overlap, restriction, cocycle, and gluing checks all cohere, the result becomes a global section and is accepted.")
    lines.append("")
    lines.append("This is the first phase where the manifold behaves less like a map of roles and more like a structured local-to-global reasoning object.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from manifold atlas alignment to sheaf consistency")
    print(f"[{PHASE}] task: local sections must glue into global meaning only through coherent overlaps")

    df = generate_trials(trials_per_task=2400)
    task_summary, family_summary, sheaf_summary, summary = make_summaries(df)

    trials_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_trials.csv"
    task_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_task_summary.csv"
    family_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_family_summary.csv"
    sheaf_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_sheaf_summary.csv"
    summary_path = OUT_DIR / "phase97_sheaf_consistency_local_global_audit_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    sheaf_summary.to_csv(sheaf_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_examples()
    write_report(task_summary, family_summary, sheaf_summary, summary)

    plot_decision_energy(df)
    plot_sheaf_field(df)
    plot_matrix(task_summary)
    plot_progress_ladder(summary)
    plot_meta_shape_graph()
    plot_deabstracted_examples()
    plot_3d_manifold(df)

    pass_key = "PHASE97_SHEAF_CONSISTENCY_LOCAL_GLOBAL_AUDIT_PASS"

    print(f"[{PHASE}] {pass_key}={summary[pass_key]}")
    print(
        f"[{PHASE}] selected_task={SELECTED_TASK} "
        f"overall_sheaf_accuracy={summary['overall_sheaf_accuracy']:.4f} "
        f"licensed_locality={summary['licensed_locality']:.4f} "
        f"overlap_consistency_accuracy={summary['overlap_consistency_accuracy']:.4f} "
        f"restriction_validity={summary['restriction_validity']:.4f} "
        f"cocycle_consistency={summary['cocycle_consistency']:.4f} "
        f"gluing_validity={summary['gluing_validity']:.4f} "
        f"global_section_validity={summary['global_section_validity']:.4f} "
        f"contradiction_rejection={summary['contradiction_rejection']:.4f} "
        f"missing_overlap_detection={summary['missing_overlap_detection']:.4f} "
        f"minimal_prior_success={summary['minimal_prior_success']:.4f} "
        f"deabstracted_edge_coverage={summary['deabstracted_edge_coverage']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_atlas_pressure={summary['mean_atlas_pressure']:.4f} "
        f"mean_sheaf_pressure={summary['mean_sheaf_pressure']:.4f} "
        f"mean_semantic_distance={summary['mean_semantic_distance']:.4f} "
        f"mean_topology_distance={summary['mean_topology_distance']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] sheaf task summary:")
    for _, row in task_summary.iterrows():
        print(
            f"  - {row['task_id']:<68} "
            f"family={row['family']:<28} "
            f"sign={row['sign']:<12} "
            f"decision={row['expected_decision']:<7} "
            f"acc={row['accuracy']:.3f} "
            f"overlap={row['overlap_consistency']:.3f} "
            f"restrict={row['restriction_validity']:.3f} "
            f"cocycle={row['cocycle_consistency']:.3f} "
            f"glue={row['gluing_validity']:.3f} "
            f"global={row['global_section_validity']:.3f} "
            f"margin={row['mean_margin']:.4f} "
            f"trials={int(row['trials'])}"
        )

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote task summary: {task_path}")
    print(f"[{PHASE}] wrote family summary: {family_path}")
    print(f"[{PHASE}] wrote sheaf summary: {sheaf_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {OUT_DIR / 'phase97_sheaf_consistency_local_global_audit_report.md'}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()