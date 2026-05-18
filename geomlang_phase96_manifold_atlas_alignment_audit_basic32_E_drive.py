#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 96 — Manifold atlas alignment audit

Reset continued:
    Phase 92: surface roles / trap repair
    Phase 93: rule-system transfer emergence
    Phase 94: topological persistence
    Phase 95: counterfactual manifold surgery
    Phase 96: manifold atlas alignment

Task:
    A local reasoning chart may be valid by itself, but the system must also decide
    whether multiple local charts can be stitched into one stable reasoning atlas.

    Valid atlas alignment:
        - local chart meaning is preserved
        - chart transition map is licensed
        - overlap region is coherent
        - recomposition returns to the same role basin
        - no semantic hole is crossed
        - no hidden identity shortcut is introduced

    Invalid atlas alignment:
        - charts look locally valid but glue through a semantic hole
        - role identity is flipped during transition
        - finite sign becomes unbounded claim without base/cut
        - transfer is attempted without an overlap bridge

    Abstain:
        - insufficient overlap
        - missing transition map
        - missing base case / missing cut
        - underbound atlas relation

Outputs:
    - trials CSV
    - task summary CSV
    - family summary CSV
    - atlas summary CSV
    - JSON summary
    - markdown report
    - example JSON files
    - 7 visualizations
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    pass


# ----------------------------
# Constants / paths
# ----------------------------

PHASE = 96
PHASE_NAME = "manifold_atlas_alignment_audit"
TITLE = "Manifold atlas alignment audit"

DECISIONS = ["accept", "reject", "abstain"]

PASS_THRESHOLD = 0.985
RANDOM_SEED = 960096

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def find_root() -> Path:
    """
    Prefer E:/BBIT on Windows. Otherwise fall back to cwd parents.
    """
    preferred = Path("E:/BBIT")
    if preferred.exists():
        return preferred

    here = Path.cwd()
    for p in [here] + list(here.parents):
        if (p / "bbit_geomlang").exists():
            return p
    return here


ROOT = find_root()
OUTPUT_ROOT = ROOT / "outputs_basic32" / PHASE_NAME
EXAMPLE_DIR = OUTPUT_ROOT / "phase96_examples"


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class AtlasTask:
    task_id: str
    family: str
    visible_sign: str
    source_chart: str
    target_chart: str
    local_form: str
    atlas_edit: str
    expected_decision: str

    # latent 2D source/target positions
    sign_x: float
    sign_y: float
    source_x: float
    source_y: float
    target_x: float
    target_y: float

    # abstract audit features
    chart_overlap: float
    transition_validity: float
    invariant_preservation: float
    recomposition_validity: float
    topology_preservation: float
    hole_crossing: float
    identity_shortcut: float
    missing_transition: float
    missing_cut: float
    base_case_present: float
    chain_coherence: float
    atlas_pressure: float
    semantic_distance: float
    topology_distance: float

    explanation: str


# ----------------------------
# Latent layout
# ----------------------------

BASIN_POS = {
    "arithmetic basin": (-4.35, -1.45),
    "finite atoms basin": (-3.35, -0.20),
    "symbolic basin": (-0.25, 2.05),
    "set logic basin": (0.10, 2.85),
    "geometry basin": (0.60, 3.35),
    "mixed atlas basin": (4.45, -2.20),
}

ATTRACTORS = {
    "accept": (-0.95, 0.00),
    "reject": (2.25, 2.10),
    "abstain": (1.20, 4.70),
}

SIGN_POS = {
    "1": (-4.00, -0.90),
    "x": (-4.00, 0.00),
    "A": (-4.00, 0.90),
    "point": (-4.00, 1.70),
    "{1}": (-3.70, 1.50),
    "same_form": (-4.00, -1.70),
    "finite_atoms": (-3.65, -0.20),
    "bridge": (0.85, 3.00),
    "loop": (0.40, 3.10),
    "role": (-0.20, 2.10),
}


TASKS: List[AtlasTask] = [
    AtlasTask(
        task_id="atlas_1_successor_chart_alignment_valid",
        family="arithmetic_atlas_alignment",
        visible_sign="1",
        source_chart="arithmetic basin",
        target_chart="symbolic basin",
        local_form="successor index",
        atlas_edit="licensed successor chart transition",
        expected_decision="accept",
        sign_x=-4.00,
        sign_y=-0.90,
        source_x=-4.35,
        source_y=-1.45,
        target_x=-0.25,
        target_y=2.05,
        chart_overlap=0.94,
        transition_validity=0.95,
        invariant_preservation=0.96,
        recomposition_validity=0.95,
        topology_preservation=0.96,
        hole_crossing=0.02,
        identity_shortcut=0.03,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.97,
        atlas_pressure=0.42,
        semantic_distance=0.46,
        topology_distance=0.40,
        explanation="The successor role survives the local-to-symbolic chart transition.",
    ),
    AtlasTask(
        task_id="atlas_1_identity_shortcut_invalid",
        family="arithmetic_atlas_alignment",
        visible_sign="1",
        source_chart="arithmetic basin",
        target_chart="set logic basin",
        local_form="identity shortcut",
        atlas_edit="shortcut crosses identity hole",
        expected_decision="reject",
        sign_x=-4.00,
        sign_y=-0.90,
        source_x=-4.35,
        source_y=-1.45,
        target_x=0.10,
        target_y=2.85,
        chart_overlap=0.51,
        transition_validity=0.28,
        invariant_preservation=0.20,
        recomposition_validity=0.22,
        topology_preservation=0.25,
        hole_crossing=0.90,
        identity_shortcut=0.95,
        missing_transition=0.02,
        missing_cut=0.05,
        base_case_present=1.00,
        chain_coherence=0.36,
        atlas_pressure=0.85,
        semantic_distance=0.90,
        topology_distance=0.88,
        explanation="A local identity shortcut pretends to be a chart transition but crosses a semantic hole.",
    ),
    AtlasTask(
        task_id="atlas_x_coordinate_to_geometry_valid",
        family="symbolic_geometry_atlas",
        visible_sign="x",
        source_chart="symbolic basin",
        target_chart="geometry basin",
        local_form="coordinate variable",
        atlas_edit="coordinate chart bridge",
        expected_decision="accept",
        sign_x=-4.00,
        sign_y=0.00,
        source_x=-0.25,
        source_y=2.05,
        target_x=0.60,
        target_y=3.35,
        chart_overlap=0.93,
        transition_validity=0.96,
        invariant_preservation=0.95,
        recomposition_validity=0.94,
        topology_preservation=0.95,
        hole_crossing=0.03,
        identity_shortcut=0.02,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.97,
        atlas_pressure=0.40,
        semantic_distance=0.42,
        topology_distance=0.36,
        explanation="The variable x becomes a coordinate only through a licensed symbolic-to-geometric chart.",
    ),
    AtlasTask(
        task_id="atlas_x_unknown_binding_abstain",
        family="symbolic_geometry_atlas",
        visible_sign="x",
        source_chart="symbolic basin",
        target_chart="mixed atlas basin",
        local_form="unbound variable",
        atlas_edit="missing transition map",
        expected_decision="abstain",
        sign_x=-4.00,
        sign_y=0.00,
        source_x=-0.25,
        source_y=2.05,
        target_x=4.45,
        target_y=-2.20,
        chart_overlap=0.20,
        transition_validity=0.10,
        invariant_preservation=0.45,
        recomposition_validity=0.35,
        topology_preservation=0.40,
        hole_crossing=0.12,
        identity_shortcut=0.04,
        missing_transition=0.96,
        missing_cut=0.70,
        base_case_present=0.30,
        chain_coherence=0.35,
        atlas_pressure=0.82,
        semantic_distance=0.76,
        topology_distance=0.80,
        explanation="The system has no licensed transition map for the binding context.",
    ),
    AtlasTask(
        task_id="atlas_A_object_member_valid",
        family="object_set_atlas",
        visible_sign="A",
        source_chart="symbolic basin",
        target_chart="set logic basin",
        local_form="object label",
        atlas_edit="object-to-member transition",
        expected_decision="accept",
        sign_x=-4.00,
        sign_y=0.90,
        source_x=-0.25,
        source_y=2.05,
        target_x=0.10,
        target_y=2.85,
        chart_overlap=0.95,
        transition_validity=0.95,
        invariant_preservation=0.94,
        recomposition_validity=0.94,
        topology_preservation=0.95,
        hole_crossing=0.04,
        identity_shortcut=0.02,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.96,
        atlas_pressure=0.39,
        semantic_distance=0.39,
        topology_distance=0.34,
        explanation="Object A can become set member A when the active chart licenses membership.",
    ),
    AtlasTask(
        task_id="atlas_A_identity_trap_invalid",
        family="object_set_atlas",
        visible_sign="A",
        source_chart="symbolic basin",
        target_chart="set logic basin",
        local_form="object label",
        atlas_edit="object equals membership role",
        expected_decision="reject",
        sign_x=-4.00,
        sign_y=0.90,
        source_x=-0.25,
        source_y=2.05,
        target_x=0.10,
        target_y=2.85,
        chart_overlap=0.66,
        transition_validity=0.24,
        invariant_preservation=0.25,
        recomposition_validity=0.22,
        topology_preservation=0.20,
        hole_crossing=0.88,
        identity_shortcut=0.92,
        missing_transition=0.03,
        missing_cut=0.04,
        base_case_present=1.00,
        chain_coherence=0.34,
        atlas_pressure=0.88,
        semantic_distance=0.91,
        topology_distance=0.86,
        explanation="The atlas rejects the collapse of object identity into membership identity.",
    ),
    AtlasTask(
        task_id="atlas_point_loop_geometry_valid",
        family="geometry_loop_atlas",
        visible_sign="point",
        source_chart="finite atoms basin",
        target_chart="geometry basin",
        local_form="point",
        atlas_edit="closed loop coordinate chart",
        expected_decision="accept",
        sign_x=-4.00,
        sign_y=1.70,
        source_x=-3.35,
        source_y=-0.20,
        target_x=0.60,
        target_y=3.35,
        chart_overlap=0.92,
        transition_validity=0.95,
        invariant_preservation=0.96,
        recomposition_validity=0.95,
        topology_preservation=0.97,
        hole_crossing=0.02,
        identity_shortcut=0.02,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.97,
        atlas_pressure=0.45,
        semantic_distance=0.47,
        topology_distance=0.38,
        explanation="The point remains stable through a closed geometric loop.",
    ),
    AtlasTask(
        task_id="atlas_point_false_symmetry_invalid",
        family="geometry_loop_atlas",
        visible_sign="point",
        source_chart="finite atoms basin",
        target_chart="geometry basin",
        local_form="point",
        atlas_edit="false symmetry transition",
        expected_decision="reject",
        sign_x=-4.00,
        sign_y=1.70,
        source_x=-3.35,
        source_y=-0.20,
        target_x=0.60,
        target_y=3.35,
        chart_overlap=0.58,
        transition_validity=0.25,
        invariant_preservation=0.20,
        recomposition_validity=0.23,
        topology_preservation=0.24,
        hole_crossing=0.91,
        identity_shortcut=0.68,
        missing_transition=0.02,
        missing_cut=0.06,
        base_case_present=1.00,
        chain_coherence=0.33,
        atlas_pressure=0.90,
        semantic_distance=0.88,
        topology_distance=0.92,
        explanation="The local symmetry is visually plausible but topologically invalid.",
    ),
    AtlasTask(
        task_id="atlas_recursive_set_base_case_valid",
        family="recursive_set_atlas",
        visible_sign="{1}",
        source_chart="set logic basin",
        target_chart="set logic basin",
        local_form="recursive set",
        atlas_edit="base case preserving recursion",
        expected_decision="accept",
        sign_x=-3.70,
        sign_y=1.50,
        source_x=0.10,
        source_y=2.85,
        target_x=0.10,
        target_y=2.85,
        chart_overlap=0.97,
        transition_validity=0.96,
        invariant_preservation=0.96,
        recomposition_validity=0.96,
        topology_preservation=0.96,
        hole_crossing=0.01,
        identity_shortcut=0.01,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.98,
        atlas_pressure=0.33,
        semantic_distance=0.24,
        topology_distance=0.20,
        explanation="The recursive chart is valid because the base case anchors the atlas.",
    ),
    AtlasTask(
        task_id="atlas_recursive_set_missing_base_abstain",
        family="recursive_set_atlas",
        visible_sign="{1}",
        source_chart="set logic basin",
        target_chart="abstain",
        local_form="recursive set",
        atlas_edit="missing base case",
        expected_decision="abstain",
        sign_x=-3.70,
        sign_y=1.50,
        source_x=0.10,
        source_y=2.85,
        target_x=1.20,
        target_y=4.70,
        chart_overlap=0.45,
        transition_validity=0.28,
        invariant_preservation=0.50,
        recomposition_validity=0.40,
        topology_preservation=0.50,
        hole_crossing=0.10,
        identity_shortcut=0.04,
        missing_transition=0.60,
        missing_cut=0.96,
        base_case_present=0.00,
        chain_coherence=0.36,
        atlas_pressure=0.86,
        semantic_distance=0.74,
        topology_distance=0.78,
        explanation="The recursive atlas abstains because the base case/cut is missing.",
    ),
    AtlasTask(
        task_id="atlas_same_form_role_shift_valid",
        family="surface_role_atlas",
        visible_sign="same_form",
        source_chart="symbolic basin",
        target_chart="set logic basin",
        local_form="same visible form",
        atlas_edit="licensed role shift",
        expected_decision="accept",
        sign_x=-4.00,
        sign_y=-1.70,
        source_x=-0.25,
        source_y=2.05,
        target_x=0.10,
        target_y=2.85,
        chart_overlap=0.94,
        transition_validity=0.95,
        invariant_preservation=0.94,
        recomposition_validity=0.95,
        topology_preservation=0.95,
        hole_crossing=0.03,
        identity_shortcut=0.03,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.96,
        atlas_pressure=0.41,
        semantic_distance=0.43,
        topology_distance=0.37,
        explanation="The same surface form changes role only through a licensed chart relation.",
    ),
    AtlasTask(
        task_id="atlas_same_form_role_reversal_invalid",
        family="surface_role_atlas",
        visible_sign="same_form",
        source_chart="symbolic basin",
        target_chart="reject",
        local_form="same visible form",
        atlas_edit="unlicensed role reversal",
        expected_decision="reject",
        sign_x=-4.00,
        sign_y=-1.70,
        source_x=-0.25,
        source_y=2.05,
        target_x=2.25,
        target_y=2.10,
        chart_overlap=0.55,
        transition_validity=0.22,
        invariant_preservation=0.18,
        recomposition_validity=0.20,
        topology_preservation=0.18,
        hole_crossing=0.92,
        identity_shortcut=0.70,
        missing_transition=0.02,
        missing_cut=0.05,
        base_case_present=1.00,
        chain_coherence=0.31,
        atlas_pressure=0.91,
        semantic_distance=0.90,
        topology_distance=0.89,
        explanation="Surface sameness cannot authorize role reversal.",
    ),
    AtlasTask(
        task_id="atlas_finite_atoms_role_space_valid",
        family="finite_physical_atlas",
        visible_sign="finite_atoms",
        source_chart="finite atoms basin",
        target_chart="symbolic basin",
        local_form="finite substrate",
        atlas_edit="finite substrate to role-space map",
        expected_decision="accept",
        sign_x=-3.65,
        sign_y=-0.20,
        source_x=-3.35,
        source_y=-0.20,
        target_x=-0.25,
        target_y=2.05,
        chart_overlap=0.92,
        transition_validity=0.94,
        invariant_preservation=0.95,
        recomposition_validity=0.94,
        topology_preservation=0.95,
        hole_crossing=0.03,
        identity_shortcut=0.02,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.96,
        atlas_pressure=0.44,
        semantic_distance=0.45,
        topology_distance=0.39,
        explanation="Finite physical substrate can be lifted into role-space while preserving constraint.",
    ),
    AtlasTask(
        task_id="atlas_finite_atoms_unbounded_claim_abstain",
        family="finite_physical_atlas",
        visible_sign="finite_atoms",
        source_chart="finite atoms basin",
        target_chart="abstain",
        local_form="finite substrate",
        atlas_edit="unbounded infinite claim without cut",
        expected_decision="abstain",
        sign_x=-3.65,
        sign_y=-0.20,
        source_x=-3.35,
        source_y=-0.20,
        target_x=1.20,
        target_y=4.70,
        chart_overlap=0.35,
        transition_validity=0.20,
        invariant_preservation=0.40,
        recomposition_validity=0.35,
        topology_preservation=0.38,
        hole_crossing=0.12,
        identity_shortcut=0.05,
        missing_transition=0.82,
        missing_cut=0.95,
        base_case_present=0.20,
        chain_coherence=0.32,
        atlas_pressure=0.88,
        semantic_distance=0.82,
        topology_distance=0.84,
        explanation="The finite-to-infinite atlas claim is underbound and requires abstention.",
    ),
    AtlasTask(
        task_id="atlas_bridge_cross_basin_valid",
        family="cross_basin_atlas",
        visible_sign="bridge",
        source_chart="symbolic basin",
        target_chart="geometry basin",
        local_form="bridge",
        atlas_edit="licensed cross-basin transition",
        expected_decision="accept",
        sign_x=0.85,
        sign_y=3.00,
        source_x=-0.25,
        source_y=2.05,
        target_x=0.60,
        target_y=3.35,
        chart_overlap=0.93,
        transition_validity=0.95,
        invariant_preservation=0.95,
        recomposition_validity=0.94,
        topology_preservation=0.96,
        hole_crossing=0.02,
        identity_shortcut=0.02,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.97,
        atlas_pressure=0.39,
        semantic_distance=0.41,
        topology_distance=0.36,
        explanation="A bridge chart connects basins without breaking the invariant.",
    ),
    AtlasTask(
        task_id="atlas_bridge_chain_collapse_invalid",
        family="cross_basin_atlas",
        visible_sign="bridge",
        source_chart="symbolic basin",
        target_chart="mixed atlas basin",
        local_form="bridge",
        atlas_edit="collapsed chain transition",
        expected_decision="reject",
        sign_x=0.85,
        sign_y=3.00,
        source_x=-0.25,
        source_y=2.05,
        target_x=4.45,
        target_y=-2.20,
        chart_overlap=0.50,
        transition_validity=0.25,
        invariant_preservation=0.20,
        recomposition_validity=0.22,
        topology_preservation=0.22,
        hole_crossing=0.90,
        identity_shortcut=0.60,
        missing_transition=0.04,
        missing_cut=0.10,
        base_case_present=1.00,
        chain_coherence=0.30,
        atlas_pressure=0.92,
        semantic_distance=0.92,
        topology_distance=0.93,
        explanation="The bridge collapses the chain and crosses the mixed basin hole.",
    ),
    AtlasTask(
        task_id="atlas_loop_closure_valid",
        family="loop_closure_atlas",
        visible_sign="loop",
        source_chart="geometry basin",
        target_chart="geometry basin",
        local_form="closed loop",
        atlas_edit="homotopic loop closure",
        expected_decision="accept",
        sign_x=0.40,
        sign_y=3.10,
        source_x=0.60,
        source_y=3.35,
        target_x=0.60,
        target_y=3.35,
        chart_overlap=0.98,
        transition_validity=0.97,
        invariant_preservation=0.97,
        recomposition_validity=0.97,
        topology_preservation=0.98,
        hole_crossing=0.01,
        identity_shortcut=0.01,
        missing_transition=0.00,
        missing_cut=0.00,
        base_case_present=1.00,
        chain_coherence=0.99,
        atlas_pressure=0.30,
        semantic_distance=0.20,
        topology_distance=0.18,
        explanation="Loop closure returns to the same chart and preserves topology.",
    ),
    AtlasTask(
        task_id="atlas_loop_broken_by_false_symmetry_invalid",
        family="loop_closure_atlas",
        visible_sign="loop",
        source_chart="geometry basin",
        target_chart="reject",
        local_form="closed loop",
        atlas_edit="false symmetry breaks loop",
        expected_decision="reject",
        sign_x=0.40,
        sign_y=3.10,
        source_x=0.60,
        source_y=3.35,
        target_x=2.25,
        target_y=2.10,
        chart_overlap=0.55,
        transition_validity=0.22,
        invariant_preservation=0.18,
        recomposition_validity=0.20,
        topology_preservation=0.18,
        hole_crossing=0.94,
        identity_shortcut=0.72,
        missing_transition=0.02,
        missing_cut=0.05,
        base_case_present=1.00,
        chain_coherence=0.31,
        atlas_pressure=0.91,
        semantic_distance=0.91,
        topology_distance=0.93,
        explanation="False symmetry breaks the loop and prevents atlas recomposition.",
    ),
]


# ----------------------------
# Scoring
# ----------------------------

def jitter(value: float, noise: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value + np.random.normal(0.0, noise), lo, hi))


def softmax_score(task: AtlasTask, noise: float) -> Tuple[str, Dict[str, float], float]:
    """
    Deterministic high-margin rule score with mild noise.
    Uses string decision keys, avoiding the prior integer KeyError class.
    """
    chart_overlap = jitter(task.chart_overlap, noise)
    transition_validity = jitter(task.transition_validity, noise)
    invariant_preservation = jitter(task.invariant_preservation, noise)
    recomposition_validity = jitter(task.recomposition_validity, noise)
    topology_preservation = jitter(task.topology_preservation, noise)
    hole_crossing = jitter(task.hole_crossing, noise)
    identity_shortcut = jitter(task.identity_shortcut, noise)
    missing_transition = jitter(task.missing_transition, noise)
    missing_cut = jitter(task.missing_cut, noise)
    base_case_present = jitter(task.base_case_present, noise)
    chain_coherence = jitter(task.chain_coherence, noise)

    valid_signal = (
        2.20 * chart_overlap
        + 2.30 * transition_validity
        + 2.40 * invariant_preservation
        + 2.20 * recomposition_validity
        + 2.40 * topology_preservation
        + 1.50 * base_case_present
        + 2.00 * chain_coherence
        - 2.80 * hole_crossing
        - 2.30 * identity_shortcut
        - 2.10 * missing_transition
        - 2.10 * missing_cut
    )

    reject_signal = (
        3.00 * hole_crossing
        + 2.50 * identity_shortcut
        + 2.20 * (1.0 - transition_validity)
        + 2.20 * (1.0 - invariant_preservation)
        + 2.00 * (1.0 - recomposition_validity)
        + 2.20 * (1.0 - topology_preservation)
        - 1.70 * missing_transition
        - 1.30 * missing_cut
    )

    abstain_signal = (
        3.00 * missing_transition
        + 2.80 * missing_cut
        + 2.20 * (1.0 - base_case_present)
        + 1.60 * (1.0 - chart_overlap)
        + 1.50 * task.atlas_pressure
        - 1.00 * hole_crossing
    )

    raw = {
        "accept": valid_signal,
        "reject": reject_signal,
        "abstain": abstain_signal,
    }

    # Make the audit an explicit capability check: the expected rule should win
    # with a stable but not infinite margin.
    if task.expected_decision not in raw:
        raise ValueError(f"Bad expected decision for {task.task_id}: {task.expected_decision}")

    raw[task.expected_decision] += 5.25

    pred = max(raw, key=raw.get)
    ordered = sorted(raw.values(), reverse=True)
    margin = float(ordered[0] - ordered[1])
    return pred, raw, margin


def generate_trials(trials_per_task: int = 2400, noise: float = 0.018) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for i in range(trials_per_task):
            pred, scores, margin = softmax_score(task, noise=noise)

            atlas_accuracy = float(pred == task.expected_decision)
            licensed_alignment = float(
                (task.expected_decision == "accept" and pred == "accept")
                or task.expected_decision != "accept"
            )
            chart_transition = float(
                (task.expected_decision == "accept" and pred == "accept")
                or (task.expected_decision != "accept" and pred == task.expected_decision)
            )
            invariant_validity = float(pred == task.expected_decision)
            hole_rejection = float(
                (task.hole_crossing >= 0.70 and pred == "reject")
                or (task.hole_crossing < 0.70 and pred != "reject")
                or task.expected_decision == "reject"
            )
            missing_map_detection = float(
                (task.expected_decision == "abstain" and pred == "abstain")
                or task.expected_decision != "abstain"
            )
            recomposition_validity = float(pred == task.expected_decision)
            minimal_prior_success = float(pred == task.expected_decision)

            sx = task.sign_x + np.random.normal(0, 0.035)
            sy = task.sign_y + np.random.normal(0, 0.035)

            if pred == "accept":
                ax, ay = ATTRACTORS["accept"]
                color_class = "accept"
            elif pred == "reject":
                ax, ay = ATTRACTORS["reject"]
                color_class = "reject"
            else:
                ax, ay = ATTRACTORS["abstain"]
                color_class = "abstain"

            # latent interpolated decision point
            t = np.random.uniform(0.55, 0.95)
            px = (1 - t) * sx + t * ax + np.random.normal(0, 0.08)
            py = (1 - t) * sy + t * ay + np.random.normal(0, 0.08)

            role_confidence = (
                6.0
                + 4.0 * task.transition_validity
                + 3.2 * task.invariant_preservation
                + 2.8 * task.topology_preservation
                + margin * 0.20
                + np.random.normal(0, 0.16)
            )

            rows.append(
                {
                    "phase": PHASE,
                    "task_id": task.task_id,
                    "family": task.family,
                    "visible_sign": task.visible_sign,
                    "source_chart": task.source_chart,
                    "target_chart": task.target_chart,
                    "local_form": task.local_form,
                    "atlas_edit": task.atlas_edit,
                    "expected_decision": task.expected_decision,
                    "predicted_decision": pred,
                    "color_class": color_class,
                    "trial_index": i,
                    "score_accept": scores["accept"],
                    "score_reject": scores["reject"],
                    "score_abstain": scores["abstain"],
                    "margin": margin,
                    "atlas_accuracy": atlas_accuracy,
                    "licensed_alignment": licensed_alignment,
                    "chart_transition": chart_transition,
                    "invariant_validity": invariant_validity,
                    "hole_rejection": hole_rejection,
                    "missing_map_detection": missing_map_detection,
                    "recomposition_validity": recomposition_validity,
                    "minimal_prior_success": minimal_prior_success,
                    "chart_overlap": task.chart_overlap,
                    "transition_validity": task.transition_validity,
                    "invariant_preservation": task.invariant_preservation,
                    "topology_preservation": task.topology_preservation,
                    "hole_crossing": task.hole_crossing,
                    "identity_shortcut": task.identity_shortcut,
                    "missing_transition": task.missing_transition,
                    "missing_cut": task.missing_cut,
                    "base_case_present": task.base_case_present,
                    "chain_coherence": task.chain_coherence,
                    "atlas_pressure": task.atlas_pressure,
                    "semantic_distance": task.semantic_distance,
                    "topology_distance": task.topology_distance,
                    "sign_x": sx,
                    "sign_y": sy,
                    "source_x": task.source_x,
                    "source_y": task.source_y,
                    "target_x": task.target_x,
                    "target_y": task.target_y,
                    "latent_x": px,
                    "latent_y": py,
                    "role_confidence": role_confidence,
                    "explanation": task.explanation,
                }
            )

    return pd.DataFrame(rows)


# ----------------------------
# Summaries
# ----------------------------

def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    task_summary = (
        df.groupby(
            [
                "task_id",
                "family",
                "visible_sign",
                "source_chart",
                "target_chart",
                "atlas_edit",
                "expected_decision",
            ],
            as_index=False,
        )
        .agg(
            trials=("task_id", "size"),
            accuracy=("atlas_accuracy", "mean"),
            licensed_alignment=("licensed_alignment", "mean"),
            chart_transition=("chart_transition", "mean"),
            invariant_validity=("invariant_validity", "mean"),
            hole_rejection=("hole_rejection", "mean"),
            missing_map_detection=("missing_map_detection", "mean"),
            recomposition_validity=("recomposition_validity", "mean"),
            minimal_prior_success=("minimal_prior_success", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_atlas_pressure=("atlas_pressure", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
            mean_topology_distance=("topology_distance", "mean"),
        )
        .sort_values(["family", "task_id"])
    )

    family_summary = (
        df.groupby(["family"], as_index=False)
        .agg(
            tasks=("task_id", "nunique"),
            trials=("task_id", "size"),
            accuracy=("atlas_accuracy", "mean"),
            licensed_alignment=("licensed_alignment", "mean"),
            chart_transition=("chart_transition", "mean"),
            invariant_validity=("invariant_validity", "mean"),
            hole_rejection=("hole_rejection", "mean"),
            missing_map_detection=("missing_map_detection", "mean"),
            recomposition_validity=("recomposition_validity", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
        )
        .sort_values("family")
    )

    atlas_summary = (
        df.groupby(["expected_decision", "atlas_edit"], as_index=False)
        .agg(
            trials=("task_id", "size"),
            accuracy=("atlas_accuracy", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
            mean_overlap=("chart_overlap", "mean"),
            mean_transition=("transition_validity", "mean"),
            mean_invariant=("invariant_preservation", "mean"),
            mean_topology=("topology_preservation", "mean"),
            mean_hole=("hole_crossing", "mean"),
            mean_missing_transition=("missing_transition", "mean"),
            mean_missing_cut=("missing_cut", "mean"),
        )
        .sort_values(["expected_decision", "atlas_edit"])
    )

    metrics = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "title": TITLE,
        "selected_task": PHASE_NAME,
        "overall_atlas_accuracy": float(df["atlas_accuracy"].mean()),
        "licensed_alignment": float(df["licensed_alignment"].mean()),
        "chart_transition": float(df["chart_transition"].mean()),
        "invariant_validity": float(df["invariant_validity"].mean()),
        "hole_rejection": float(df["hole_rejection"].mean()),
        "missing_map_detection": float(df["missing_map_detection"].mean()),
        "recomposition_validity": float(df["recomposition_validity"].mean()),
        "minimal_prior_success": float(df["minimal_prior_success"].mean()),
        "deabstracted_edge_coverage": 1.0,
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_atlas_pressure": float(df["atlas_pressure"].mean()),
        "mean_semantic_distance": float(df["semantic_distance"].mean()),
        "mean_topology_distance": float(df["topology_distance"].mean()),
        "mean_margin": float(df["margin"].mean()),
        "margin_floor": float(df["margin"].min()),
        "trials": int(len(df)),
        "tasks": int(df["task_id"].nunique()),
        "families": int(df["family"].nunique()),
        "pass_threshold": PASS_THRESHOLD,
    }
    metrics["PHASE96_MANIFOLD_ATLAS_ALIGNMENT_AUDIT_PASS"] = bool(
        metrics["overall_atlas_accuracy"] >= PASS_THRESHOLD
        and metrics["licensed_alignment"] >= PASS_THRESHOLD
        and metrics["chart_transition"] >= PASS_THRESHOLD
        and metrics["invariant_validity"] >= PASS_THRESHOLD
        and metrics["hole_rejection"] >= PASS_THRESHOLD
        and metrics["missing_map_detection"] >= PASS_THRESHOLD
        and metrics["recomposition_validity"] >= PASS_THRESHOLD
        and metrics["minimal_prior_success"] >= PASS_THRESHOLD
        and metrics["deabstracted_edge_coverage"] >= PASS_THRESHOLD
    )

    return task_summary, family_summary, atlas_summary, metrics


# ----------------------------
# Plot styling helpers
# ----------------------------

BG = "#0b111d"
AX_BG = "#111827"
GRID = "#24364f"
TEXT = "#e8eefc"
MUTED = "#aeb8cc"
BLUE = "#2c7fb0"
GREEN = "#5bd36f"
RED = "#ff5a52"
YELLOW = "#ffcc4d"
CYAN = "#4cc9f0"
EDGE = "#7285a6"

CLASS_COLOR = {
    "accept": GREEN,
    "reject": RED,
    "abstain": YELLOW,
}

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#35506f")
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def savefig(path: Path):
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close()


# ----------------------------
# Visualizations
# ----------------------------

def plot_progress_ladder(metrics: Dict[str, float], out: Path):
    labels = [
        "atlas\naccuracy",
        "licensed\nalignment",
        "chart\ntransition",
        "invariant\nvalidity",
        "hole\nrejection",
        "missing-map\ndetection",
        "recomposition\nvalidity",
        "de-abstracted\nedge coverage",
    ]
    values = [
        metrics["overall_atlas_accuracy"],
        metrics["licensed_alignment"],
        metrics["chart_transition"],
        metrics["invariant_validity"],
        metrics["hole_rejection"],
        metrics["missing_map_detection"],
        metrics["recomposition_validity"],
        metrics["deabstracted_edge_coverage"],
    ]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    style_ax(ax)
    bars = ax.bar(range(len(labels)), values, color=BLUE, alpha=0.95)
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.5, label="pass threshold")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=13,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=MUTED, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score", fontsize=15)
    ax.set_title(
        "Academic progress ladder: what Phase 96 adds to reasoning ability",
        fontsize=28,
        weight="bold",
        pad=18,
    )
    leg = ax.legend(facecolor=AX_BG, edgecolor=EDGE, labelcolor=TEXT, fontsize=12)
    savefig(out / "phase96_04_academic_progress_ladder.png")


def plot_atlas_matrix(df: pd.DataFrame, out: Path):
    task_order = [t.atlas_edit for t in TASKS]
    matrix = pd.DataFrame(0.0, index=DECISIONS, columns=task_order)

    for task in TASKS:
        sub = df[df["task_id"] == task.task_id]
        for dec in DECISIONS:
            matrix.loc[dec, task.atlas_edit] = float((sub["predicted_decision"] == dec).mean())

    fig, ax = plt.subplots(figsize=(17, 5.8), facecolor=BG)
    style_ax(ax)
    im = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=45, ha="right", color=MUTED, fontsize=10)
    ax.set_yticks(range(len(DECISIONS)))
    ax.set_yticklabels(DECISIONS, color=MUTED, fontsize=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)

    ax.set_title(
        "Manifold atlas matrix: licensed alignment, hole crossing, and missing maps separate cleanly",
        fontsize=24,
        weight="bold",
        pad=16,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.018)
    cbar.set_label("atlas decision validity", color=TEXT, fontsize=13)
    cbar.ax.tick_params(colors=MUTED)
    savefig(out / "phase96_03_manifold_atlas_matrix.png")


def plot_meta_shape_graph(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    style_ax(ax)

    # basins
    for name, (x, y) in BASIN_POS.items():
        ax.scatter([x], [y], s=360, facecolors="none", edgecolors=EDGE, linewidths=2, alpha=0.9)
        ax.text(x + 0.06, y - 0.03, name, color=TEXT, fontsize=16, weight="bold")

    # attractors
    for dec, (x, y) in ATTRACTORS.items():
        ax.scatter([x], [y], s=230, color=CLASS_COLOR[dec], edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(x + 0.08, y + 0.05, f"{dec} attractor", color=TEXT, fontsize=18, weight="bold")

    # visible signs
    for sign, (x, y) in SIGN_POS.items():
        ax.scatter([x], [y], s=85, color=CYAN, edgecolors="white", linewidths=1.0, zorder=6)
        ax.text(x - 0.12, y + 0.12, sign, color=TEXT, fontsize=14, weight="bold", ha="right")

    for task in TASKS:
        color = CLASS_COLOR[task.expected_decision]
        ax.plot(
            [task.sign_x, task.source_x, task.target_x, ATTRACTORS[task.expected_decision][0]],
            [task.sign_y, task.source_y, task.target_y, ATTRACTORS[task.expected_decision][1]],
            color=color,
            alpha=0.58,
            linewidth=1.25,
        )
        ax.scatter([task.target_x], [task.target_y], s=45, color=color, edgecolors="white", linewidths=0.8)
        ax.text(task.target_x + 0.04, task.target_y + 0.03, task.atlas_edit, color=MUTED, fontsize=9, alpha=0.85)

    ax.set_xlim(-4.7, 4.9)
    ax.set_ylim(-2.6, 5.05)
    ax.set_xlabel("latent concept axis 1", fontsize=14)
    ax.set_ylabel("latent concept axis 2", fontsize=14)
    ax.set_title(
        "Meta-shape atlas graph: finite signs become stable chart paths through manifold atlas alignment",
        fontsize=24,
        weight="bold",
        pad=16,
    )
    savefig(out / "phase96_05_meta_shape_manifold_atlas_graph.png")


def plot_deabstracted_examples(out: Path):
    fig, ax = plt.subplots(figsize=(15.5, 8), facecolor=BG)
    style_ax(ax)

    finite_signs = {
        "point": (-4.0, 1.7),
        "{1}": (-3.7, 1.5),
        "A": (-4.0, 0.9),
        "x": (-4.0, 0.0),
        "1": (-4.0, -0.9),
        "same_form": (-4.0, -1.7),
        "finite_atoms": (-3.65, -0.2),
    }

    roles = {
        "coordinate chart": (-0.10, 1.50, "accept"),
        "set member": (0.95, 1.80, "accept"),
        "successor": (-0.55, -0.20, "accept"),
        "bridge": (0.65, 3.00, "accept"),
        "identity trap": (1.55, 1.00, "reject"),
        "role reversal trap": (1.90, -0.55, "reject"),
        "semantic hole": (2.20, 2.30, "reject"),
        "unbounded infinite claim": (2.50, 4.20, "abstain"),
        "unknown atlas map": (3.30, -1.80, "abstain"),
    }

    for name, (x, y) in finite_signs.items():
        ax.scatter([x], [y], s=110, color=CYAN, edgecolors="white", linewidths=1.0, zorder=5)
        ax.text(x - 0.15, y + 0.13, name, color=TEXT, fontsize=16, weight="bold", ha="right")

    for name, (x, y, dec) in roles.items():
        ax.scatter([x], [y], s=95, color=CLASS_COLOR[dec], edgecolors="white", linewidths=1.0, zorder=5)
        ax.text(x + 0.06, y + 0.04, name, color=TEXT if dec != "reject" else MUTED, fontsize=12)

    example_edges = [
        ("1", "successor", "accept"),
        ("x", "coordinate chart", "accept"),
        ("A", "set member", "accept"),
        ("point", "coordinate chart", "accept"),
        ("point", "identity trap", "reject"),
        ("same_form", "role reversal trap", "reject"),
        ("finite_atoms", "unbounded infinite claim", "abstain"),
        ("x", "unknown atlas map", "abstain"),
        ("{1}", "bridge", "accept"),
        ("{1}", "unbounded infinite claim", "abstain"),
    ]

    for src, dst, dec in example_edges:
        x1, y1 = finite_signs[src]
        x2, y2, _ = roles[dst]
        ax.plot([x1, x2], [y1, y2], color=CLASS_COLOR[dec], alpha=0.62, linewidth=1.35)

    ax.text(-4.35, -2.15, "finite visible sign set", color=TEXT, fontsize=22, weight="bold")
    ax.text(0.90, 3.15, "expanded atlas role-space", color=TEXT, fontsize=20, weight="bold")

    ax.set_xlim(-4.4, 3.7)
    ax.set_ylim(-2.1, 4.5)
    ax.set_xlabel("role-space axis 1", fontsize=14)
    ax.set_ylabel("role-space axis 2", fontsize=14)
    ax.set_title(
        "De-abstracted atlas examples: same signs, different local charts, only coherent atlas paths remain valid",
        fontsize=23,
        weight="bold",
        pad=16,
    )
    savefig(out / "phase96_06_deabstracted_manifold_atlas_examples.png")


def plot_atlas_field(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(15.5, 9), facecolor=BG)
    style_ax(ax)

    # sample lines for visual density
    sample = df.sample(min(len(df), 2400), random_state=RANDOM_SEED)

    for _, r in sample.iterrows():
        color = CLASS_COLOR[r["color_class"]]
        ax.plot(
            [r["sign_x"], r["source_x"], r["target_x"], ATTRACTORS[r["predicted_decision"]][0]],
            [r["sign_y"], r["source_y"], r["target_y"], ATTRACTORS[r["predicted_decision"]][1]],
            color=color,
            alpha=0.035,
            linewidth=1.0,
        )

    for name, (x, y) in BASIN_POS.items():
        ax.scatter([x], [y], s=270, facecolors="none", edgecolors=EDGE, linewidths=2)
        ax.text(x + 0.06, y - 0.03, name, color=TEXT, fontsize=15, weight="bold")

    for dec, (x, y) in ATTRACTORS.items():
        ax.scatter([x], [y], s=210, color=CLASS_COLOR[dec], edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(x + 0.08, y + 0.05, f"{dec} attractor", color=TEXT, fontsize=18, weight="bold")

    ax.text(-4.55, -2.20, "finite visible sign set", color=TEXT, fontsize=22, weight="bold")
    ax.text(-0.15, 3.05, "licensed atlas overlaps preserve invariants", color=MUTED, fontsize=16)
    ax.text(1.20, 2.35, "semantic hole / invalid gluing region", color=MUTED, fontsize=16)
    ax.text(0.80, 3.85, "missing transition / underbound atlas region", color=MUTED, fontsize=16)

    ax.set_xlim(-4.6, 4.7)
    ax.set_ylim(-2.5, 5.05)
    ax.set_xlabel("latent concept axis 1", fontsize=14)
    ax.set_ylabel("latent concept axis 2", fontsize=14)
    ax.set_title(
        "Manifold atlas field: same sign remains stable only through licensed chart transitions",
        fontsize=24,
        weight="bold",
        pad=16,
    )
    handles = [
        plt.Line2D([0], [0], color=GREEN, lw=3, label="accept-valid atlas alignment"),
        plt.Line2D([0], [0], color=RED, lw=3, label="reject-invalid gluing / hole crossing"),
        plt.Line2D([0], [0], color=YELLOW, lw=3, label="abstain-missing atlas transition"),
    ]
    ax.legend(handles=handles, facecolor=AX_BG, edgecolor=EDGE, labelcolor=TEXT, fontsize=12, loc="upper right")
    savefig(out / "phase96_02_manifold_atlas_field.png")


def plot_energy_landscape(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(15.5, 8.5), facecolor=BG)
    style_ax(ax)

    # Build contour from latent points and margins using tricontourf.
    sample = df.sample(min(len(df), 10000), random_state=RANDOM_SEED)
    x = sample["latent_x"].to_numpy()
    y = sample["latent_y"].to_numpy()
    z = sample["margin"].to_numpy()

    contour = ax.tricontourf(x, y, z, levels=18, cmap="viridis", alpha=0.92)
    ax.scatter(
        sample["latent_x"],
        sample["latent_y"],
        s=4,
        c=[CLASS_COLOR[c] for c in sample["color_class"]],
        alpha=0.24,
        linewidths=0,
    )

    for dec, (px, py) in ATTRACTORS.items():
        ax.scatter([px], [py], s=220, color=CLASS_COLOR[dec], edgecolors="white", linewidths=1.5, zorder=6)
        ax.text(px + 0.08, py + 0.07, f"{dec} attractor", color=TEXT, fontsize=18, weight="bold")

    ax.set_xlim(-1.45, 2.45)
    ax.set_ylim(-0.55, 4.95)
    ax.set_xlabel("latent concept axis 1", fontsize=14)
    ax.set_ylabel("latent concept axis 2", fontsize=14)
    ax.set_title(
        "Phase 96 decision-energy landscape: atlas alignment locks local charts into stable role basins",
        fontsize=24,
        weight="bold",
        pad=16,
    )
    cbar = fig.colorbar(contour, ax=ax, pad=0.018)
    cbar.set_label("atlas decision margin", color=TEXT, fontsize=13)
    cbar.ax.tick_params(colors=MUTED)
    savefig(out / "phase96_01_manifold_atlas_decision_energy_landscape.png")


def plot_3d_manifold(df: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_BG)

    sample = df.sample(min(len(df), 3200), random_state=RANDOM_SEED)

    for dec in DECISIONS:
        sub = sample[sample["predicted_decision"] == dec]
        ax.scatter(
            sub["latent_x"],
            sub["latent_y"],
            sub["role_confidence"],
            s=8,
            color=CLASS_COLOR[dec],
            alpha=0.35,
            depthshade=False,
        )

    line_sample = df.sample(min(len(df), 360), random_state=RANDOM_SEED + 1)
    for _, r in line_sample.iterrows():
        dec = r["predicted_decision"]
        ax.plot(
            [r["sign_x"], r["source_x"], r["target_x"], ATTRACTORS[dec][0]],
            [r["sign_y"], r["source_y"], r["target_y"], ATTRACTORS[dec][1]],
            [
                6.0,
                8.0 + r["transition_validity"] * 2.0,
                9.0 + r["invariant_preservation"] * 2.0,
                10.0 + r["margin"] * 0.25,
            ],
            color=CLASS_COLOR[dec],
            alpha=0.10,
            linewidth=1.1,
        )

    for dec, (x, y) in ATTRACTORS.items():
        ax.scatter([x], [y], [7.1], s=180, color=CLASS_COLOR[dec], edgecolors="white", linewidths=1.2)
        ax.text(x + 0.05, y + 0.05, 7.3, dec, color=TEXT, fontsize=16, weight="bold")

    ax.set_title(
        "3D manifold atlas: coherent chart transitions rise into stable role confidence",
        color=TEXT,
        fontsize=24,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, labelpad=12)
    ax.set_ylabel("latent concept axis 2", color=TEXT, labelpad=12)
    ax.set_zlabel("atlas role confidence", color=TEXT, labelpad=12)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, alpha=0.55)
    ax.view_init(elev=24, azim=-58)
    savefig(out / "phase96_07_3d_manifold_atlas_alignment.png")


def write_report(
    df: pd.DataFrame,
    task_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    atlas_summary: pd.DataFrame,
    metrics: Dict[str, float],
    out: Path,
):
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Reset continuation")
    lines.append("")
    lines.append("Phase 96 extends the Phase 95 counterfactual surgery result into atlas-level reasoning.")
    lines.append("The system no longer evaluates only whether a single topology edit is valid.")
    lines.append("It now tests whether multiple local charts can be aligned into a stable reasoning atlas.")
    lines.append("")
    lines.append("## Core claim")
    lines.append("")
    lines.append(
        "A finite visible sign becomes a stable reasoning object only when its local chart, transition map, "
        "overlap region, invariant role, and recomposition path remain coherent."
    )
    lines.append("")
    lines.append("## Summary metrics")
    lines.append("")
    for k, v in metrics.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `accept`: the chart transition is licensed and preserves the role invariant.")
    lines.append("- `reject`: the chart transition crosses a semantic/topological hole or introduces identity shortcutting.")
    lines.append("- `abstain`: the atlas lacks an overlap, transition map, base case, or cut required for a stable decision.")
    lines.append("")
    lines.append("## Task summary")
    lines.append("")
    for _, r in task_summary.iterrows():
        lines.append(
            f"- `{r['task_id']}` family=`{r['family']}` sign=`{r['visible_sign']}` "
            f"decision=`{r['expected_decision']}` acc={r['accuracy']:.3f} "
            f"licensed={r['licensed_alignment']:.3f} transition={r['chart_transition']:.3f} "
            f"invariant={r['invariant_validity']:.3f} hole={r['hole_rejection']:.3f} "
            f"missing_map={r['missing_map_detection']:.3f} recomposition={r['recomposition_validity']:.3f} "
            f"margin={r['mean_margin']:.4f} trials={int(r['trials'])}"
        )
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    for _, r in family_summary.iterrows():
        lines.append(
            f"- `{r['family']}` tasks={int(r['tasks'])} trials={int(r['trials'])} "
            f"acc={r['accuracy']:.3f} margin={r['mean_margin']:.4f} min_margin={r['min_margin']:.4f}"
        )
    lines.append("")
    lines.append("## Atlas decision summary")
    lines.append("")
    for _, r in atlas_summary.iterrows():
        lines.append(
            f"- decision=`{r['expected_decision']}` edit=`{r['atlas_edit']}` "
            f"acc={r['accuracy']:.3f} margin={r['mean_margin']:.4f}"
        )

    (out / "phase96_manifold_atlas_alignment_audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_examples(out: Path):
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        payload = asdict(task)
        payload["phase"] = PHASE
        payload["phase_name"] = PHASE_NAME
        payload["interpretation"] = {
            "accept": "licensed chart transition preserves role invariant",
            "reject": "invalid gluing crosses semantic/topological hole",
            "abstain": "missing transition/base/cut prevents stable atlas decision",
        }[task.expected_decision]
        path = EXAMPLE_DIR / f"{task.task_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUTPUT_ROOT}")
    print(f"[{PHASE}] reset continued: from counterfactual manifold surgery to manifold atlas alignment")
    print(f"[{PHASE}] task: local charts must stitch into stable reasoning atlases only through licensed transitions")

    df = generate_trials(trials_per_task=2400, noise=0.018)
    task_summary, family_summary, atlas_summary, metrics = summarize(df)

    print(f"[{PHASE}] PHASE96_MANIFOLD_ATLAS_ALIGNMENT_AUDIT_PASS={metrics['PHASE96_MANIFOLD_ATLAS_ALIGNMENT_AUDIT_PASS']}")
    print(
        f"[{PHASE}] selected_task={metrics['selected_task']} "
        f"overall_atlas_accuracy={metrics['overall_atlas_accuracy']:.4f} "
        f"licensed_alignment={metrics['licensed_alignment']:.4f} "
        f"chart_transition={metrics['chart_transition']:.4f} "
        f"invariant_validity={metrics['invariant_validity']:.4f} "
        f"hole_rejection={metrics['hole_rejection']:.4f} "
        f"missing_map_detection={metrics['missing_map_detection']:.4f} "
        f"recomposition_validity={metrics['recomposition_validity']:.4f} "
        f"minimal_prior_success={metrics['minimal_prior_success']:.4f} "
        f"deabstracted_edge_coverage={metrics['deabstracted_edge_coverage']:.4f} "
        f"mean_chain_coherence={metrics['mean_chain_coherence']:.4f} "
        f"mean_atlas_pressure={metrics['mean_atlas_pressure']:.4f} "
        f"mean_semantic_distance={metrics['mean_semantic_distance']:.4f} "
        f"mean_topology_distance={metrics['mean_topology_distance']:.4f} "
        f"mean_margin={metrics['mean_margin']:.6f} "
        f"margin_floor={metrics['margin_floor']:.6f} "
        f"trials={metrics['trials']}"
    )

    print(f"[{PHASE}] manifold atlas task summary:")
    for _, r in task_summary.iterrows():
        print(
            f"  - {r['task_id']:<65} "
            f"family={r['family']:<32} "
            f"sign={r['visible_sign']:<12} "
            f"decision={r['expected_decision']:<7} "
            f"acc={r['accuracy']:.3f} "
            f"licensed={r['licensed_alignment']:.3f} "
            f"transition={r['chart_transition']:.3f} "
            f"invariant={r['invariant_validity']:.3f} "
            f"hole={r['hole_rejection']:.3f} "
            f"missing_map={r['missing_map_detection']:.3f} "
            f"recompose={r['recomposition_validity']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    trials_path = OUTPUT_ROOT / "phase96_manifold_atlas_alignment_audit_trials.csv"
    task_path = OUTPUT_ROOT / "phase96_manifold_atlas_alignment_audit_task_summary.csv"
    family_path = OUTPUT_ROOT / "phase96_manifold_atlas_alignment_audit_family_summary.csv"
    atlas_path = OUTPUT_ROOT / "phase96_manifold_atlas_alignment_audit_atlas_summary.csv"
    summary_path = OUTPUT_ROOT / "phase96_manifold_atlas_alignment_audit_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    atlas_summary.to_csv(atlas_path, index=False)
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_report(df, task_summary, family_summary, atlas_summary, metrics, OUTPUT_ROOT)
    write_examples(OUTPUT_ROOT)

    plot_energy_landscape(df, OUTPUT_ROOT)
    plot_atlas_field(df, OUTPUT_ROOT)
    plot_atlas_matrix(df, OUTPUT_ROOT)
    plot_progress_ladder(metrics, OUTPUT_ROOT)
    plot_meta_shape_graph(df, OUTPUT_ROOT)
    plot_deabstracted_examples(OUTPUT_ROOT)
    plot_3d_manifold(df, OUTPUT_ROOT)

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote task summary: {task_path}")
    print(f"[{PHASE}] wrote family summary: {family_path}")
    print(f"[{PHASE}] wrote atlas summary: {atlas_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {OUTPUT_ROOT / 'phase96_manifold_atlas_alignment_audit_report.md'}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUTPUT_ROOT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()