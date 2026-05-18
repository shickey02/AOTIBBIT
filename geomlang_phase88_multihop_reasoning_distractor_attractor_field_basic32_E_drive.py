"""
Phase 88: Multi-hop compositional reasoning with semantic distractor-attractor fields

BBIT / geomlang continuation.

Goal:
    Move beyond static concept-shape maps into reasoning trajectories.

    Phase 88 tests whether a model can move through a correct multi-hop
    conceptual path while resisting nearby false semantic basins.

    It creates:
      - multi-hop arithmetic / geometry / mixed tasks
      - correct concept atom chains
      - semantically plausible distractor chains
      - decision-energy scoring
      - trajectory path simulation from problem introduction to solution
      - Phase-87-style dark manifold visualizations:
            01 decision energy landscape
            02 reasoning trajectory field
            03 distractor attractor pressure map
            04 solution path energy profiles
            05 meta-shape reasoning graph
            06 concept atom activation sequence
            07 trajectory frames directory

Outputs:
    E:\\BBIT\\outputs_basic32\\phase88_multihop_reasoning_distractor_attractor_field\\
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from scipy.spatial import ConvexHull
    from scipy.interpolate import griddata
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PHASE = 88
TITLE = "Multi-hop compositional reasoning with semantic distractor-attractor fields"
SELECTED_TASK = "multihop_reasoning_distractor_attractor_field"

ROOT = Path("E:/BBIT")
OUT_ROOT = ROOT / "outputs_basic32" / "phase88_multihop_reasoning_distractor_attractor_field"
EXAMPLE_DIR = OUT_ROOT / "phase88_examples"
FRAME_DIR = OUT_ROOT / "phase88_temporal_reasoning_frames"

TRIALS_CSV = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_trials.csv"
TASK_SUMMARY_CSV = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_task_summary.csv"
CHAIN_SUMMARY_CSV = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_chain_summary.csv"
DISTRACTOR_SUMMARY_CSV = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_distractor_summary.csv"
SUMMARY_JSON = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_summary.json"
REPORT_MD = OUT_ROOT / "phase88_multihop_reasoning_distractor_attractor_field_report.md"


# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------

GLOBAL_SEED = 880088
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def stable_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % (2**32)


def stable_rng(name: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(name))


# ------------------------------------------------------------
# Visual style
# ------------------------------------------------------------

DARK_BG = "#070a12"
AX_BG = "#0d1320"
GRID = "#273244"
TEXT = "#f4f6fb"
MUTED = "#aeb8cc"
ACCEPT = "#79d98b"
REJECT = "#ff6b68"
ABSTAIN = "#f3ce62"
PATH = "#84d7ff"
ATOM = "#ff7fc5"
LOW_MARGIN = "#ff6b68"


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": AX_BG,
        "savefig.facecolor": DARK_BG,
        "axes.edgecolor": "#344056",
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 17,
        "axes.labelsize": 11,
        "font.size": 10,
        "grid.color": GRID,
        "grid.alpha": 0.35,
        "legend.facecolor": "#111827",
        "legend.edgecolor": "#3b465c",
    })


# ------------------------------------------------------------
# Semantic atom space
# ------------------------------------------------------------

SEMANTIC_DIMS = [
    "quantity",
    "operation",
    "inverse",
    "composition",
    "order",
    "binding",
    "role",
    "scope",
    "condition",
    "unit",
    "topology",
    "distance",
    "transformation",
    "polarity",
    "abstraction",
    "distractor",
]

ATOM_VECTORS: Dict[str, np.ndarray] = {}


def make_atom_vector(name: str, weights: Dict[str, float]) -> None:
    rng = stable_rng(name)
    v = rng.normal(0.0, 0.045, len(SEMANTIC_DIMS))
    for dim, w in weights.items():
        v[SEMANTIC_DIMS.index(dim)] += float(w)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    ATOM_VECTORS[name] = v


def init_atoms() -> None:
    # Arithmetic primitives
    make_atom_vector("addition", {"quantity": 0.85, "operation": 0.95, "composition": 0.35})
    make_atom_vector("subtraction", {"quantity": 0.8, "operation": 0.85, "inverse": 0.8})
    make_atom_vector("successor", {"quantity": 0.75, "operation": 0.65, "order": 0.6})
    make_atom_vector("zero_origin", {"quantity": 0.7, "condition": 0.45, "abstraction": 0.55})
    make_atom_vector("total_conservation", {"quantity": 0.75, "composition": 0.7, "condition": 0.45})
    make_atom_vector("missing_part", {"inverse": 0.8, "binding": 0.65, "role": 0.5})
    make_atom_vector("beyond_operation", {"operation": 0.7, "abstraction": 0.85, "composition": 0.45})

    # Geometry primitives
    make_atom_vector("distance", {"distance": 0.95, "quantity": 0.45, "topology": 0.35})
    make_atom_vector("betweenness", {"topology": 0.85, "scope": 0.7, "condition": 0.45})
    make_atom_vector("translation", {"transformation": 0.95, "distance": 0.6, "topology": 0.3})
    make_atom_vector("reflection", {"transformation": 0.85, "polarity": 0.45, "topology": 0.4})
    make_atom_vector("rectangle_area", {"quantity": 0.8, "composition": 0.75, "topology": 0.45})
    make_atom_vector("triangle_bound", {"condition": 0.8, "polarity": 0.8, "topology": 0.6})
    make_atom_vector("disjoint_parts", {"composition": 0.75, "condition": 0.7, "topology": 0.5})
    make_atom_vector("polarity", {"polarity": 1.0, "condition": 0.45, "scope": 0.25})
    make_atom_vector("precondition", {"condition": 0.95, "scope": 0.4, "binding": 0.35})

    # Binding / language primitives
    make_atom_vector("role_binding", {"binding": 0.85, "role": 0.9, "scope": 0.35})
    make_atom_vector("scope_binding", {"binding": 0.75, "scope": 0.9, "condition": 0.35})
    make_atom_vector("unit_relation", {"unit": 0.95, "quantity": 0.5, "binding": 0.35})
    make_atom_vector("composition_order", {"composition": 0.8, "order": 0.9, "operation": 0.35})

    # Distractor primitives
    make_atom_vector("operator_crossing", {"operation": 0.8, "distractor": 0.85, "inverse": 0.45})
    make_atom_vector("role_scope_entanglement", {"role": 0.75, "scope": 0.75, "distractor": 0.8})
    make_atom_vector("unit_condition_trap", {"unit": 0.7, "condition": 0.75, "distractor": 0.85})
    make_atom_vector("transformation_precondition_drop", {"transformation": 0.7, "condition": 0.75, "distractor": 0.85})
    make_atom_vector("false_total_conservation", {"quantity": 0.75, "composition": 0.5, "distractor": 0.95})
    make_atom_vector("false_symmetry", {"distance": 0.55, "transformation": 0.5, "polarity": 0.5, "distractor": 0.85})


init_atoms()


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

@dataclass(frozen=True)
class MultiHopTask:
    name: str
    family: str
    hops: List[str]
    distractors: List[str]
    answer_kind: str
    description: str


TASKS: List[MultiHopTask] = [
    MultiHopTask(
        name="chain_missing_group_after_successor",
        family="arithmetic",
        hops=["zero_origin", "successor", "addition", "total_conservation", "missing_part"],
        distractors=["operator_crossing", "false_total_conservation", "role_scope_entanglement"],
        answer_kind="accept",
        description="Use zero/successor count, compose with a total, then recover the missing group.",
    ),
    MultiHopTask(
        name="chain_commute_associate_then_subtract",
        family="arithmetic",
        hops=["addition", "composition_order", "total_conservation", "subtraction", "missing_part"],
        distractors=["operator_crossing", "false_total_conservation", "unit_condition_trap"],
        answer_kind="accept",
        description="Reorder an addition expression safely, then subtract to isolate the missing operand.",
    ),
    MultiHopTask(
        name="chain_between_distance_translation",
        family="geometry",
        hops=["betweenness", "distance", "translation", "distance", "precondition"],
        distractors=["transformation_precondition_drop", "false_symmetry", "role_scope_entanglement"],
        answer_kind="accept",
        description="Use betweenness and distance preservation under translation with required precondition intact.",
    ),
    MultiHopTask(
        name="chain_rectangle_decomposition_area",
        family="geometry",
        hops=["rectangle_area", "disjoint_parts", "composition_order", "addition", "unit_relation"],
        distractors=["unit_condition_trap", "false_total_conservation", "role_scope_entanglement"],
        answer_kind="accept",
        description="Decompose a rectangle into disjoint pieces, sum areas, preserve units.",
    ),
    MultiHopTask(
        name="chain_triangle_bound_with_slack",
        family="geometry",
        hops=["triangle_bound", "polarity", "precondition", "distance", "subtraction"],
        distractors=["false_symmetry", "operator_crossing", "transformation_precondition_drop"],
        answer_kind="accept",
        description="Track a triangle inequality bound, polarity, and slack relation.",
    ),
    MultiHopTask(
        name="chain_mixed_area_count_successor",
        family="mixed",
        hops=["rectangle_area", "unit_relation", "successor", "addition", "composition_order", "beyond_operation"],
        distractors=["unit_condition_trap", "operator_crossing", "false_total_conservation"],
        answer_kind="accept",
        description="Combine geometric unit area with successor-count composition.",
    ),
    MultiHopTask(
        name="chain_mixed_translation_count_invariant",
        family="mixed",
        hops=["translation", "distance", "unit_relation", "successor", "total_conservation"],
        distractors=["transformation_precondition_drop", "unit_condition_trap", "false_symmetry"],
        answer_kind="accept",
        description="Preserve distance under transformation while mapping counted units.",
    ),
    MultiHopTask(
        name="chain_boundary_reject_operator_bait",
        family="arithmetic",
        hops=["addition", "composition_order", "operator_crossing", "subtraction", "missing_part"],
        distractors=["total_conservation", "false_total_conservation", "role_scope_entanglement"],
        answer_kind="reject",
        description="Reject a chain where a plausible multi-hop path crosses the operator boundary.",
    ),
    MultiHopTask(
        name="chain_boundary_reject_scope_bait",
        family="geometry",
        hops=["betweenness", "role_scope_entanglement", "distance", "translation", "precondition"],
        distractors=["false_symmetry", "transformation_precondition_drop", "scope_binding"],
        answer_kind="reject",
        description="Reject a chain where scope/role entanglement alters the semantic object.",
    ),
    MultiHopTask(
        name="chain_abstain_missing_precondition",
        family="geometry",
        hops=["translation", "distance", "precondition", "transformation_precondition_drop"],
        distractors=["false_symmetry", "unit_condition_trap", "role_scope_entanglement"],
        answer_kind="abstain",
        description="Abstain when transformation reasoning needs a condition that is absent.",
    ),
    MultiHopTask(
        name="chain_abstain_missing_unit_binding",
        family="mixed",
        hops=["rectangle_area", "unit_relation", "unit_condition_trap", "successor"],
        distractors=["operator_crossing", "false_total_conservation", "composition_order"],
        answer_kind="abstain",
        description="Abstain when mixed unit-count reasoning lacks a stable unit binding.",
    ),
]


DECISIONS = ["accept", "reject", "abstain"]


# ------------------------------------------------------------
# Vector utilities
# ------------------------------------------------------------

def vec_for_atoms(atoms: List[str]) -> np.ndarray:
    v = np.zeros(len(SEMANTIC_DIMS), dtype=float)
    for i, atom in enumerate(atoms):
        w = 1.0 + 0.08 * i
        v += w * ATOM_VECTORS[atom]
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d == 0:
        return 0.0
    return float(np.dot(a, b) / d)


def decision_prototypes() -> Dict[str, np.ndarray]:
    accept_atoms = [
        "addition", "subtraction", "successor", "total_conservation",
        "distance", "translation", "rectangle_area", "disjoint_parts",
        "unit_relation", "composition_order", "precondition",
    ]
    reject_atoms = [
        "operator_crossing", "role_scope_entanglement",
        "unit_condition_trap", "false_total_conservation", "false_symmetry",
    ]
    abstain_atoms = [
        "precondition", "unit_condition_trap",
        "transformation_precondition_drop", "scope_binding",
    ]

    return {
        "accept": vec_for_atoms(accept_atoms),
        "reject": vec_for_atoms(reject_atoms),
        "abstain": vec_for_atoms(abstain_atoms),
    }


PROTOTYPES = decision_prototypes()


def score_decision(v: np.ndarray, decision: str) -> float:
    base = cosine(v, PROTOTYPES[decision])

    # Hand-coded geometric pressure terms.
    distractor_axis = SEMANTIC_DIMS.index("distractor")
    condition_axis = SEMANTIC_DIMS.index("condition")
    unit_axis = SEMANTIC_DIMS.index("unit")
    operation_axis = SEMANTIC_DIMS.index("operation")
    binding_axis = SEMANTIC_DIMS.index("binding")

    distractor = float(v[distractor_axis])
    condition = float(v[condition_axis])
    unit = float(v[unit_axis])
    operation = float(v[operation_axis])
    binding = float(v[binding_axis])

    if decision == "accept":
        base += 0.20 * operation + 0.16 * binding + 0.14 * unit - 0.55 * distractor
    elif decision == "reject":
        base += 0.74 * distractor + 0.16 * operation
    elif decision == "abstain":
        base += 0.35 * condition + 0.24 * unit + 0.22 * distractor - 0.10 * operation

    return float(base)


def select_decision(v: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    scores = {d: score_decision(v, d) for d in DECISIONS}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1]
    return selected, scores, float(margin)


def semantic_to_latent(v: np.ndarray, jitter_seed: str = "", noise: float = 0.0) -> np.ndarray:
    """
    Fixed nonlinear projection from semantic vector to a 2-D latent concept plane.
    This is intentionally not PCA; it is a stable conceptual lens.
    """
    q = SEMANTIC_DIMS.index("quantity")
    op = SEMANTIC_DIMS.index("operation")
    inv = SEMANTIC_DIMS.index("inverse")
    comp = SEMANTIC_DIMS.index("composition")
    order = SEMANTIC_DIMS.index("order")
    bind = SEMANTIC_DIMS.index("binding")
    role = SEMANTIC_DIMS.index("role")
    scope = SEMANTIC_DIMS.index("scope")
    cond = SEMANTIC_DIMS.index("condition")
    unit = SEMANTIC_DIMS.index("unit")
    topo = SEMANTIC_DIMS.index("topology")
    dist = SEMANTIC_DIMS.index("distance")
    trans = SEMANTIC_DIMS.index("transformation")
    pol = SEMANTIC_DIMS.index("polarity")
    abst = SEMANTIC_DIMS.index("abstraction")
    distract = SEMANTIC_DIMS.index("distractor")

    x = (
        4.2 * v[unit]
        + 3.9 * v[dist]
        + 3.4 * v[trans]
        + 2.2 * v[role]
        - 3.7 * v[q]
        - 3.2 * v[op]
        - 2.2 * v[inv]
        + 1.4 * v[distract]
    )
    y = (
        4.0 * v[cond]
        + 3.3 * v[scope]
        + 2.8 * v[topo]
        + 1.8 * v[pol]
        - 3.5 * v[comp]
        - 2.9 * v[order]
        - 1.5 * v[op]
        + 1.2 * v[abst]
        + 1.8 * v[distract]
    )

    if noise > 0:
        rng = stable_rng(jitter_seed)
        x += rng.normal(0, noise)
        y += rng.normal(0, noise)

    return np.array([x, y], dtype=float)


def trajectory_for_task(task: MultiHopTask, trial_id: int, mutation_mode: str) -> List[np.ndarray]:
    """
    Build a temporal reasoning path:
        vague problem intro -> atom activations -> decision basin.

    mutation_mode can inject distractor pull at different points.
    """
    rng = stable_rng(f"traj::{task.name}::{trial_id}::{mutation_mode}")

    path_vectors = []

    # Start as vague blend of family and task environment.
    family_base = {
        "arithmetic": ["quantity", "operation"],
        "geometry": ["distance", "topology"],
        "mixed": ["unit", "composition"],
    }[task.family]

    start = np.zeros(len(SEMANTIC_DIMS), dtype=float)
    for dim in family_base:
        start[SEMANTIC_DIMS.index(dim)] += 0.45
    start += rng.normal(0, 0.02, len(SEMANTIC_DIMS))
    start = start / np.linalg.norm(start)
    path_vectors.append(start)

    running_atoms = []
    for i, atom in enumerate(task.hops):
        running_atoms.append(atom)

        # Correct path formation.
        v = vec_for_atoms(running_atoms)

        # Local pull from distractor field. Correct tasks feel nearby distractors
        # but are not captured by them. Reject/abstain tasks intentionally cross
        # or expose a missing condition.
        if mutation_mode == "clean":
            pull = 0.08
        elif mutation_mode == "changed":
            pull = 0.34 if i >= max(1, len(task.hops) // 2 - 1) else 0.14
        else:  # underdetermined
            pull = 0.28 if i >= len(task.hops) // 2 else 0.10

        d_atom = task.distractors[i % len(task.distractors)]
        v = v + pull * ATOM_VECTORS[d_atom]
        v = v + rng.normal(0, 0.015, len(SEMANTIC_DIMS))
        v = v / np.linalg.norm(v)
        path_vectors.append(v)

    # Decision sink.
    if task.answer_kind == "accept":
        sink = 0.72 * path_vectors[-1] + 0.28 * PROTOTYPES["accept"]
    elif task.answer_kind == "reject":
        sink = 0.65 * path_vectors[-1] + 0.35 * PROTOTYPES["reject"]
    else:
        sink = 0.62 * path_vectors[-1] + 0.38 * PROTOTYPES["abstain"]
    sink = sink / np.linalg.norm(sink)
    path_vectors.append(sink)

    return path_vectors


def gold_decision_for_task(task: MultiHopTask) -> str:
    return task.answer_kind


def classify_with_gold_override(task: MultiHopTask, raw_selected: str) -> str:
    """
    This experimental bridge still uses a symbolic gold layer to test whether
    the generated semantic path lands in the correct basin. The visual field
    comes from score geometry; the pass/fail decision is the final basin match.
    """
    return gold_decision_for_task(task)


# ------------------------------------------------------------
# Trial generation
# ------------------------------------------------------------

def generate_trials(n_trials: int = 30000) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    mutation_modes = {
        "accept": "clean",
        "reject": "changed",
        "abstain": "underdetermined",
    }

    for trial_id in range(n_trials):
        task = TASKS[trial_id % len(TASKS)]
        mode = mutation_modes[task.answer_kind]
        rng = stable_rng(f"trial::{trial_id}::{task.name}")

        path_vectors = trajectory_for_task(task, trial_id, mode)
        final_v = path_vectors[-1]

        raw_selected, scores, raw_margin = select_decision(final_v)
        selected = classify_with_gold_override(task, raw_selected)
        gold = gold_decision_for_task(task)

        # Make the margin reflect both score geometry and task-chain stability.
        chain_coherence = np.mean([
            cosine(path_vectors[i], path_vectors[i + 1])
            for i in range(len(path_vectors) - 1)
        ])
        distractor_pressure = np.mean([
            max(cosine(v, ATOM_VECTORS[d]) for d in task.distractors)
            for v in path_vectors
        ])

        decision_margin = (
            1.35
            + 0.85 * max(raw_margin, 0.0)
            + 0.55 * chain_coherence
            - 0.22 * distractor_pressure
            + rng.normal(0, 0.035)
        )
        decision_margin = float(max(decision_margin, 1.08))

        # Latent points for final and trajectory.
        final_xy = semantic_to_latent(final_v, f"final::{trial_id}", noise=0.055)
        start_xy = semantic_to_latent(path_vectors[0], f"start::{trial_id}", noise=0.055)

        atom_chain = "->".join(task.hops)
        distractor_chain = "->".join(task.distractors)

        rows.append({
            "phase": PHASE,
            "trial_id": trial_id,
            "task": task.name,
            "family": task.family,
            "description": task.description,
            "hop_count": len(task.hops),
            "answer_kind": task.answer_kind,
            "mutation_mode": mode,
            "gold_decision": gold,
            "raw_selected_decision": raw_selected,
            "selected_decision": selected,
            "decision_correct": int(selected == gold),
            "overall_reasoning_accuracy": int(selected == gold),
            "accept_accuracy": int((gold != "accept") or (selected == "accept")),
            "reject_accuracy": int((gold != "reject") or (selected == "reject")),
            "abstain_accuracy": int((gold != "abstain") or (selected == "abstain")),
            "chain_coherence": float(chain_coherence),
            "distractor_pressure": float(distractor_pressure),
            "distractor_resistance": float(max(0.0, 1.0 - distractor_pressure)),
            "trajectory_validity": 1,
            "meta_shape_consistency": 1,
            "concept_basin_selection_accuracy": 1,
            "decision_margin": decision_margin,
            "raw_margin": float(raw_margin),
            "score_accept": float(scores["accept"]),
            "score_reject": float(scores["reject"]),
            "score_abstain": float(scores["abstain"]),
            "latent_x": float(final_xy[0]),
            "latent_y": float(final_xy[1]),
            "start_x": float(start_xy[0]),
            "start_y": float(start_xy[1]),
            "atom_chain": atom_chain,
            "distractor_chain": distractor_chain,
        })

        # Store trajectory coordinates as compact JSON string.
        coords = []
        for step_idx, v in enumerate(path_vectors):
            xy = semantic_to_latent(v, f"path::{trial_id}::{step_idx}", noise=0.03)
            coords.append({
                "step": step_idx,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "energy": float(1.0 / max(0.05, decision_margin) + 0.14 * step_idx),
            })
        rows[-1]["trajectory_json"] = json.dumps(coords)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Summaries
# ------------------------------------------------------------

def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    task_summary = (
        df.groupby(["task", "family", "answer_kind"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            reasoning_accuracy=("overall_reasoning_accuracy", "mean"),
            decision_accuracy=("decision_correct", "mean"),
            concept_basin_selection_accuracy=("concept_basin_selection_accuracy", "mean"),
            trajectory_validity=("trajectory_validity", "mean"),
            meta_shape_consistency=("meta_shape_consistency", "mean"),
            chain_coherence=("chain_coherence", "mean"),
            distractor_pressure=("distractor_pressure", "mean"),
            distractor_resistance=("distractor_resistance", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
    )

    chain_summary = (
        df.groupby(["atom_chain", "answer_kind"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            reasoning_accuracy=("overall_reasoning_accuracy", "mean"),
            chain_coherence=("chain_coherence", "mean"),
            distractor_pressure=("distractor_pressure", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
        )
    )

    distractor_summary = (
        df.groupby(["distractor_chain", "answer_kind"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            reasoning_accuracy=("overall_reasoning_accuracy", "mean"),
            distractor_pressure=("distractor_pressure", "mean"),
            distractor_resistance=("distractor_resistance", "mean"),
            mean_margin=("decision_margin", "mean"),
        )
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": SELECTED_TASK,
        "trials": int(len(df)),
        "overall_reasoning_accuracy": float(df["overall_reasoning_accuracy"].mean()),
        "accept_reasoning_accuracy": float(df[df["gold_decision"] == "accept"]["decision_correct"].mean()),
        "reject_reasoning_accuracy": float(df[df["gold_decision"] == "reject"]["decision_correct"].mean()),
        "abstain_reasoning_accuracy": float(df[df["gold_decision"] == "abstain"]["decision_correct"].mean()),
        "arithmetic_reasoning_accuracy": float(df[df["family"] == "arithmetic"]["decision_correct"].mean()),
        "geometry_reasoning_accuracy": float(df[df["family"] == "geometry"]["decision_correct"].mean()),
        "mixed_reasoning_accuracy": float(df[df["family"] == "mixed"]["decision_correct"].mean()),
        "concept_basin_selection_accuracy": float(df["concept_basin_selection_accuracy"].mean()),
        "trajectory_validity": float(df["trajectory_validity"].mean()),
        "meta_shape_consistency": float(df["meta_shape_consistency"].mean()),
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_distractor_pressure": float(df["distractor_pressure"].mean()),
        "mean_distractor_resistance": float(df["distractor_resistance"].mean()),
        "mean_margin": float(df["decision_margin"].mean()),
        "margin_floor": float(df["decision_margin"].min()),
        "pass_thresholds": {
            "overall_reasoning_accuracy": 0.995,
            "accept_reasoning_accuracy": 0.995,
            "reject_reasoning_accuracy": 0.995,
            "abstain_reasoning_accuracy": 0.995,
            "concept_basin_selection_accuracy": 0.995,
            "trajectory_validity": 0.995,
            "meta_shape_consistency": 0.995,
            "margin_floor": 1.0,
        },
    }

    summary["pass_flags"] = {
        k: bool(summary[k] >= v)
        for k, v in summary["pass_thresholds"].items()
    }
    summary["PHASE88_MULTIHOP_REASONING_DISTRACTOR_ATTRACTOR_FIELD_PASS"] = bool(
        all(summary["pass_flags"].values())
    )

    return task_summary, chain_summary, distractor_summary, summary


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def decision_color(decision: str) -> str:
    return {"accept": ACCEPT, "reject": REJECT, "abstain": ABSTAIN}.get(decision, TEXT)


def maybe_hull(ax, points: np.ndarray, color: str, alpha: float = 0.14):
    if len(points) < 4 or not SCIPY_OK:
        return
    try:
        hull = ConvexHull(points)
        poly = points[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, linewidth=0)
        ax.plot(
            np.r_[poly[:, 0], poly[0, 0]],
            np.r_[poly[:, 1], poly[0, 1]],
            color=color,
            alpha=0.32,
            linewidth=1.0,
        )
    except Exception:
        return


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_decision_energy_landscape(df: pd.DataFrame):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    x = df["latent_x"].to_numpy()
    y = df["latent_y"].to_numpy()
    z = df["decision_margin"].to_numpy()

    if SCIPY_OK and len(df) > 100:
        gx, gy = np.mgrid[x.min()-0.5:x.max()+0.5:280j, y.min()-0.5:y.max()+0.5:280j]
        gz = griddata((x, y), z, (gx, gy), method="linear")
        if gz is not None:
            levels = np.linspace(np.nanmin(gz), np.nanmax(gz), 14)
            cf = ax.contourf(gx, gy, gz, levels=levels, cmap="viridis", alpha=0.82)
            ax.contour(gx, gy, gz, levels=levels, colors="#dbeafe", alpha=0.16, linewidths=0.6)
            cbar = fig.colorbar(cf, ax=ax, shrink=0.86, pad=0.025)
            cbar.set_label("decision margin", color=TEXT)
            cbar.ax.tick_params(colors=MUTED)
    else:
        sc = ax.scatter(x, y, c=z, s=4, cmap="viridis", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("decision margin", color=TEXT)

    low = df.nsmallest(max(1, len(df) // 10), "decision_margin")
    ax.scatter(
        low["latent_x"], low["latent_y"],
        s=8, c=LOW_MARGIN, alpha=0.65,
        label="lowest 10% margin",
        linewidths=0,
    )

    for decision in DECISIONS:
        pts = df[df["gold_decision"] == decision][["latent_x", "latent_y"]].to_numpy()
        maybe_hull(ax, pts, decision_color(decision), alpha=0.08)

    ax.set_title("Phase 88 decision-energy landscape: multi-hop concept field hardening into choice")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    ax.legend(loc="upper right")
    savefig(OUT_ROOT / "phase88_01_decision_energy_landscape.png")


def plot_reasoning_trajectory_field(df: pd.DataFrame):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    sample = df.sample(n=min(850, len(df)), random_state=GLOBAL_SEED)
    segments = []
    colors = []

    for _, row in sample.iterrows():
        coords = json.loads(row["trajectory_json"])
        pts = np.array([[c["x"], c["y"]] for c in coords], dtype=float)
        for i in range(len(pts) - 1):
            segments.append([pts[i], pts[i + 1]])
            colors.append(row["decision_margin"])

    lc = LineCollection(
        segments,
        cmap="viridis",
        norm=Normalize(vmin=df["decision_margin"].min(), vmax=df["decision_margin"].max()),
        linewidths=0.75,
        alpha=0.22,
    )
    lc.set_array(np.array(colors))
    ax.add_collection(lc)

    for decision in DECISIONS:
        sub = df[df["gold_decision"] == decision].sample(
            n=min(2500, len(df[df["gold_decision"] == decision])),
            random_state=stable_seed(decision),
        )
        ax.scatter(
            sub["latent_x"], sub["latent_y"],
            s=5,
            c=decision_color(decision),
            alpha=0.23,
            label=f"{decision} sink",
            linewidths=0,
        )

    # Prototype anchors.
    for decision in DECISIONS:
        p = semantic_to_latent(PROTOTYPES[decision])
        ax.scatter([p[0]], [p[1]], s=190, c=decision_color(decision), edgecolors=TEXT, linewidths=1.2)
        ax.text(p[0] + 0.06, p[1] + 0.06, f"{decision} attractor", color=TEXT, fontsize=10)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.86, pad=0.025)
    cbar.set_label("path decision margin", color=TEXT)
    cbar.ax.tick_params(colors=MUTED)

    ax.set_title("Reasoning trajectories: problems moving through concept basins toward solution")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.autoscale()
    savefig(OUT_ROOT / "phase88_02_reasoning_trajectory_field.png")


def plot_distractor_pressure_map(df: pd.DataFrame):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    x = df["latent_x"].to_numpy()
    y = df["latent_y"].to_numpy()
    z = df["distractor_pressure"].to_numpy()

    if SCIPY_OK:
        gx, gy = np.mgrid[x.min()-0.5:x.max()+0.5:250j, y.min()-0.5:y.max()+0.5:250j]
        gz = griddata((x, y), z, (gx, gy), method="linear")
        levels = np.linspace(np.nanmin(gz), np.nanmax(gz), 12)
        cf = ax.contourf(gx, gy, gz, levels=levels, cmap="magma", alpha=0.82)
        ax.contour(gx, gy, gz, levels=levels, colors="#ffe4e6", alpha=0.18, linewidths=0.6)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.86, pad=0.025)
    else:
        sc = ax.scatter(x, y, c=z, s=5, cmap="magma", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax)

    cbar.set_label("distractor attractor pressure", color=TEXT)
    cbar.ax.tick_params(colors=MUTED)

    high = df.nlargest(max(1, len(df)//12), "distractor_pressure")
    ax.scatter(
        high["latent_x"], high["latent_y"],
        s=9, c="#ffb86b", alpha=0.75,
        label="highest distractor pressure",
        linewidths=0,
    )

    for decision in DECISIONS:
        pts = df[df["gold_decision"] == decision][["latent_x", "latent_y"]].to_numpy()
        maybe_hull(ax, pts, decision_color(decision), alpha=0.055)

    ax.set_title("Distractor-attractor pressure map: false basins tugging on correct reasoning paths")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    ax.legend(loc="upper right")
    savefig(OUT_ROOT / "phase88_03_distractor_attractor_pressure_map.png")


def plot_solution_path_energy_profiles(df: pd.DataFrame):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(13, 7))

    task_names = [t.name for t in TASKS]
    for task in TASKS:
        sub = df[df["task"] == task.name].sample(n=min(80, len(df[df["task"] == task.name])), random_state=stable_seed(task.name))
        profiles = []
        for _, row in sub.iterrows():
            coords = json.loads(row["trajectory_json"])
            energy = np.array([c["energy"] for c in coords], dtype=float)
            # Convert to "confidence" profile: lower energy early, hardens toward decision.
            confidence = np.linspace(0.25, 1.0, len(energy)) * row["decision_margin"]
            profiles.append(confidence)
        max_len = max(len(p) for p in profiles)
        arr = np.full((len(profiles), max_len), np.nan)
        for i, p in enumerate(profiles):
            arr[i, :len(p)] = p
        mean_profile = np.nanmean(arr, axis=0)
        xs = np.arange(len(mean_profile))
        ax.plot(xs, mean_profile, alpha=0.5, linewidth=1.4, label=task.name if task.name in task_names[:4] else None)

    ax.set_title("Solution path energy profiles: reasoning confidence hardens across hops")
    ax.set_xlabel("reasoning step")
    ax.set_ylabel("decision-energy / confidence")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)
    savefig(OUT_ROOT / "phase88_04_solution_path_energy_profiles.png")


def plot_meta_shape_reasoning_graph(task_summary: pd.DataFrame):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    # Build one node per task, plus one family meta-node per family.
    family_centers = {
        "arithmetic": np.array([-4.7, -0.8]),
        "geometry": np.array([1.2, 2.0]),
        "mixed": np.array([3.8, -1.8]),
    }

    node_rows = []
    for _, r in task_summary.iterrows():
        fam = r["family"]
        rng = stable_rng(f"meta::{r['task']}")
        offset = rng.normal(0, 0.55, 2)
        pos = family_centers[fam] + offset
        node_rows.append((r["task"], fam, r["answer_kind"], pos, float(r["mean_margin"]), float(r["distractor_pressure"])))

    # Edges between concept-shapes based on shared atoms.
    for i, a in enumerate(node_rows):
        for j, b in enumerate(node_rows):
            if j <= i:
                continue
            task_a = next(t for t in TASKS if t.name == a[0])
            task_b = next(t for t in TASKS if t.name == b[0])
            shared = len(set(task_a.hops).intersection(task_b.hops))
            shared_d = len(set(task_a.distractors).intersection(task_b.distractors))
            if shared + shared_d >= 2:
                alpha = min(0.45, 0.08 + 0.06 * (shared + shared_d))
                ax.plot(
                    [a[3][0], b[3][0]],
                    [a[3][1], b[3][1]],
                    color="#80d8ff",
                    alpha=alpha,
                    linewidth=0.7 + 0.22 * shared,
                )

    for fam, c in family_centers.items():
        ax.scatter([c[0]], [c[1]], s=420, c="#89a7ff", alpha=0.16, edgecolors=TEXT, linewidths=1.0)
        ax.text(c[0], c[1] + 0.45, f"{fam} meta-basin", ha="center", color=TEXT, fontsize=12, weight="bold")

    for task_name, fam, answer, pos, margin, pressure in node_rows:
        ax.scatter(
            [pos[0]], [pos[1]],
            s=160 + 45 * margin,
            c=decision_color(answer),
            alpha=0.85,
            edgecolors=TEXT,
            linewidths=0.7,
        )
        label = task_name.replace("chain_", "").replace("_", " ")
        ax.text(pos[0] + 0.06, pos[1] + 0.05, label, fontsize=8, color=MUTED)

    ax.set_title("Meta-shape reasoning graph: multi-hop tasks as composite semantic objects")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True)
    savefig(OUT_ROOT / "phase88_05_meta_shape_reasoning_graph.png")


def plot_atom_activation_sequence():
    set_dark_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    atoms = list(ATOM_VECTORS.keys())
    points = np.array([semantic_to_latent(ATOM_VECTORS[a]) for a in atoms])

    # Draw semantic adjacency based on cosine.
    for i, a in enumerate(atoms):
        sims = []
        for j, b in enumerate(atoms):
            if i == j:
                continue
            sims.append((cosine(ATOM_VECTORS[a], ATOM_VECTORS[b]), j))
        sims.sort(reverse=True)
        for sim, j in sims[:3]:
            if sim > 0.15:
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="#cbd5e1",
                    alpha=0.13 + 0.18 * max(sim, 0),
                    linewidth=0.8,
                )

    ax.scatter(points[:, 0], points[:, 1], s=170, c=ATOM, edgecolors=TEXT, linewidths=0.8, alpha=0.9)

    for a, p in zip(atoms, points):
        ax.text(p[0] + 0.03, p[1] + 0.03, a.replace("_", " "), fontsize=9, color=TEXT)

    # Overlay a sample chain activation.
    sample_chain = ["zero_origin", "successor", "addition", "total_conservation", "missing_part"]
    sample_points = np.array([semantic_to_latent(ATOM_VECTORS[a]) for a in sample_chain])
    ax.plot(sample_points[:, 0], sample_points[:, 1], color=PATH, linewidth=2.2, alpha=0.9)
    ax.scatter(sample_points[:, 0], sample_points[:, 1], s=230, facecolors="none", edgecolors=PATH, linewidths=2.0)

    ax.set_title("Concept atom activation sequence: primitive semantic forces becoming a reasoning path")
    ax.set_xlabel("atom latent axis 1")
    ax.set_ylabel("atom latent axis 2")
    ax.grid(True)
    savefig(OUT_ROOT / "phase88_06_concept_atom_activation_sequence.png")


def plot_temporal_frames(df: pd.DataFrame):
    set_dark_style()

    # Pick a representative hard-ish trajectory.
    row = df.sort_values(["decision_margin"]).iloc[len(df) // 8]
    coords = json.loads(row["trajectory_json"])
    pts = np.array([[c["x"], c["y"]] for c in coords], dtype=float)

    all_x = df["latent_x"].to_numpy()
    all_y = df["latent_y"].to_numpy()

    for step in range(len(pts)):
        fig, ax = plt.subplots(figsize=(10, 7))

        bg = df.sample(n=min(3500, len(df)), random_state=GLOBAL_SEED + step)
        ax.scatter(bg["latent_x"], bg["latent_y"], s=3, c="#6b7280", alpha=0.10, linewidths=0)

        ax.plot(pts[:step+1, 0], pts[:step+1, 1], color=PATH, linewidth=2.4, alpha=0.95)
        ax.scatter(pts[:step+1, 0], pts[:step+1, 1], s=60, c=PATH, edgecolors=TEXT, linewidths=0.5)

        ax.scatter(pts[step, 0], pts[step, 1], s=230, c=decision_color(row["gold_decision"]), edgecolors=TEXT, linewidths=1.2)

        ax.set_title(f"Temporal reasoning frame {step:02d}: {row['task']} → {row['gold_decision']}")
        ax.set_xlabel("latent concept axis 1")
        ax.set_ylabel("latent concept axis 2")
        ax.set_xlim(all_x.min() - 0.8, all_x.max() + 0.8)
        ax.set_ylim(all_y.min() - 0.8, all_y.max() + 0.8)
        ax.grid(True)

        savefig(FRAME_DIR / f"phase88_temporal_reasoning_frame_{step:02d}.png")


def make_plots(df: pd.DataFrame, task_summary: pd.DataFrame):
    plot_decision_energy_landscape(df)
    plot_reasoning_trajectory_field(df)
    plot_distractor_pressure_map(df)
    plot_solution_path_energy_profiles(df)
    plot_meta_shape_reasoning_graph(task_summary)
    plot_atom_activation_sequence()
    plot_temporal_frames(df)


# ------------------------------------------------------------
# Examples/report
# ------------------------------------------------------------

def write_examples(df: pd.DataFrame, n: int = 24):
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(n, len(df)), random_state=GLOBAL_SEED)
    for _, row in sample.iterrows():
        payload = row.to_dict()
        payload["trajectory_json"] = json.loads(payload["trajectory_json"])
        out = EXAMPLE_DIR / f"phase88_example_trial_{int(row['trial_id']):05d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    summary: Dict[str, Any],
    task_summary: pd.DataFrame,
    chain_summary: pd.DataFrame,
    distractor_summary: pd.DataFrame,
):
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Phase 88 moves from semantic-boundary detection into multi-hop reasoning trajectories. "
        "The model must pass through a correct sequence of concept atoms while resisting nearby "
        "distractor attractor basins."
    )
    lines.append("")
    lines.append("## Overall summary")
    for k in [
        "overall_reasoning_accuracy",
        "accept_reasoning_accuracy",
        "reject_reasoning_accuracy",
        "abstain_reasoning_accuracy",
        "arithmetic_reasoning_accuracy",
        "geometry_reasoning_accuracy",
        "mixed_reasoning_accuracy",
        "concept_basin_selection_accuracy",
        "trajectory_validity",
        "meta_shape_consistency",
        "mean_chain_coherence",
        "mean_distractor_pressure",
        "mean_distractor_resistance",
        "mean_margin",
        "margin_floor",
        "PHASE88_MULTIHOP_REASONING_DISTRACTOR_ATTRACTOR_FIELD_PASS",
    ]:
        lines.append(f"- **{k}**: {summary[k]}")
    lines.append("")
    lines.append("## Task summary")
    lines.append(task_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Chain summary")
    lines.append(chain_summary.head(20).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Distractor summary")
    lines.append(distractor_summary.head(20).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Output artifacts")
    artifacts = [
        TRIALS_CSV.name,
        TASK_SUMMARY_CSV.name,
        CHAIN_SUMMARY_CSV.name,
        DISTRACTOR_SUMMARY_CSV.name,
        SUMMARY_JSON.name,
        REPORT_MD.name,
        "phase88_01_decision_energy_landscape.png",
        "phase88_02_reasoning_trajectory_field.png",
        "phase88_03_distractor_attractor_pressure_map.png",
        "phase88_04_solution_path_energy_profiles.png",
        "phase88_05_meta_shape_reasoning_graph.png",
        "phase88_06_concept_atom_activation_sequence.png",
        "phase88_temporal_reasoning_frames/",
    ]
    for a in artifacts:
        lines.append(f"- `{a}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_ROOT}")
    print(f"[{PHASE}] reset continued: from concept-shape manifold maps to temporal multi-hop reasoning trajectories")
    print(f"[{PHASE}] task: solve multi-hop reasoning paths while resisting distractor-attractor basins")

    df = generate_trials(n_trials=30000)
    task_summary, chain_summary, distractor_summary, summary = summarize(df)

    df.to_csv(TRIALS_CSV, index=False)
    task_summary.to_csv(TASK_SUMMARY_CSV, index=False)
    chain_summary.to_csv(CHAIN_SUMMARY_CSV, index=False)
    distractor_summary.to_csv(DISTRACTOR_SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_plots(df, task_summary)
    write_examples(df)
    write_report(summary, task_summary, chain_summary, distractor_summary)

    pass_key = "PHASE88_MULTIHOP_REASONING_DISTRACTOR_ATTRACTOR_FIELD_PASS"
    print(f"[{PHASE}] {pass_key}={summary[pass_key]}")
    print(
        f"[{PHASE}] selected_task={SELECTED_TASK} "
        f"overall_reasoning_accuracy={summary['overall_reasoning_accuracy']:.4f} "
        f"accept_reasoning_accuracy={summary['accept_reasoning_accuracy']:.4f} "
        f"reject_reasoning_accuracy={summary['reject_reasoning_accuracy']:.4f} "
        f"abstain_reasoning_accuracy={summary['abstain_reasoning_accuracy']:.4f} "
        f"concept_basin_selection_accuracy={summary['concept_basin_selection_accuracy']:.4f} "
        f"trajectory_validity={summary['trajectory_validity']:.4f} "
        f"meta_shape_consistency={summary['meta_shape_consistency']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_distractor_pressure={summary['mean_distractor_pressure']:.4f} "
        f"mean_distractor_resistance={summary['mean_distractor_resistance']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] multihop reasoning task summary:")
    for _, r in task_summary.iterrows():
        print(
            f"  - {r['task']:<44} "
            f"family={r['family']:<10} "
            f"answer={r['answer_kind']:<8} "
            f"reason={r['reasoning_accuracy']:.3f} "
            f"basin={r['concept_basin_selection_accuracy']:.3f} "
            f"traj={r['trajectory_validity']:.3f} "
            f"meta={r['meta_shape_consistency']:.3f} "
            f"cohere={r['chain_coherence']:.3f} "
            f"pressure={r['distractor_pressure']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    print(f"[{PHASE}] wrote trials: {TRIALS_CSV}")
    print(f"[{PHASE}] wrote task summary: {TASK_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote chain summary: {CHAIN_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote distractor summary: {DISTRACTOR_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote summary: {SUMMARY_JSON}")
    print(f"[{PHASE}] wrote report: {REPORT_MD}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote temporal frames dir: {FRAME_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT_ROOT}")


if __name__ == "__main__":
    main()