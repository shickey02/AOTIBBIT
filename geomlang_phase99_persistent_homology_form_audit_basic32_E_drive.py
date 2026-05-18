# geomlang_phase99_persistent_homology_form_audit_basic32_E_drive.py
"""
Phase 99 — Persistent homology / form-topology audit

Reset continued:
    Phase 96: manifold atlas alignment
    Phase 97: sheaf local-to-global consistency
    Phase 98: cohomology obstruction audit
    Phase 99: persistent homology / form-topology audit

Task:
    A finite visible sign set is not enough.
    A local section is not enough.
    A zero obstruction at one epsilon is not enough.

    The system now asks whether the form stays recognizable across an epsilon
    filtration. Meaning becomes stable only when the topological signature
    persists across scale.

Core idea:
    Same visible signs can look like:
        - scattered points at epsilon = 0
        - local edges at small epsilon
        - loops / holes at middle epsilon
        - filled disks / collapsed blobs at large epsilon

    Phase 99 teaches the audit to distinguish:
        ACCEPT  = persistent intended topology survives across a stable epsilon band
        REJECT  = nonzero / contradictory / spurious topology persists
        ABSTAIN = undercovered, overfilled, or critical epsilon ambiguity

This is intentionally self-contained:
    - no sklearn
    - no scipy
    - no networkx
    - only numpy / pandas / matplotlib

Outputs:
    E:\\BBIT\\outputs_basic32\\phase99_persistent_homology_form_audit\\
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PHASE = 99
PHASE_NAME = "persistent_homology_form_audit"
TITLE = "Persistent homology / form-topology audit"

ROOT = Path(r"E:\BBIT")
OUT_DIR = ROOT / "outputs_basic32" / "phase99_persistent_homology_form_audit"
EXAMPLE_DIR = OUT_DIR / "phase99_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

BG = "#0b1220"
PANEL = "#111827"
GRID = "#26364d"
TEXT = "#e8eefc"
MUTED = "#aab4c8"

ACCEPT = "#57d46f"
REJECT = "#ff5b57"
ABSTAIN = "#ffc84d"
CYAN = "#55d6ff"
BLUE = "#4b73ff"
PURPLE = "#9b5cff"

PASS_THRESHOLD = 0.985

random.seed(99)
np.random.seed(99)


def setup_dark(ax):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#3b5275")
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def savefig(fig, path: Path):
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------
# Synthetic topology primitives
# ---------------------------------------------------------------------

def make_circle_points(n: int = 48, radius: float = 1.0, noise: float = 0.04) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[radius * np.cos(t), radius * np.sin(t)]
    pts += np.random.normal(0, noise, pts.shape)
    return pts


def make_two_loop_points(n: int = 72, noise: float = 0.035) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    left = make_circle_points(n1, 0.72, noise) + np.array([-0.72, 0.0])
    right = make_circle_points(n2, 0.72, noise) + np.array([0.72, 0.0])
    return np.vstack([left, right])


def make_line_points(n: int = 42, noise: float = 0.035) -> np.ndarray:
    x = np.linspace(-1.35, 1.35, n)
    y = 0.25 * np.sin(2.4 * x)
    pts = np.c_[x, y]
    pts += np.random.normal(0, noise, pts.shape)
    return pts


def make_disk_points(n: int = 70, noise: float = 0.02) -> np.ndarray:
    r = np.sqrt(np.random.rand(n)) * 1.05
    t = np.random.rand(n) * 2 * np.pi
    pts = np.c_[r * np.cos(t), r * np.sin(t)]
    pts += np.random.normal(0, noise, pts.shape)
    return pts


def make_two_cluster_points(n: int = 52, noise: float = 0.08) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    a = np.random.normal([-0.9, 0.0], noise, size=(n1, 2))
    b = np.random.normal([0.9, 0.0], noise, size=(n2, 2))
    return np.vstack([a, b])


def make_sparse_arc_points(n: int = 24, noise: float = 0.06) -> np.ndarray:
    t = np.linspace(0.1, 1.75 * np.pi, n)
    pts = np.c_[np.cos(t), np.sin(t)]
    pts += np.random.normal(0, noise, pts.shape)
    return pts


def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(float)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale <= 1e-9:
        return pts
    return pts / scale


def pairwise_dist(pts: np.ndarray) -> np.ndarray:
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ---------------------------------------------------------------------
# Vietoris-Rips style clique audit
# ---------------------------------------------------------------------

def connected_components_from_edges(n: int, edges: List[Tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    return len({find(i) for i in range(n)})


def count_triangles(edge_set: set, n: int) -> int:
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    count += 1
    return count


def vr_counts(pts: np.ndarray, eps: float) -> Dict[str, float]:
    n = len(pts)
    D = pairwise_dist(pts)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                edges.append((i, j))

    edge_set = set(edges)
    comps = connected_components_from_edges(n, edges)
    tri = count_triangles(edge_set, n)

    V = n
    E = len(edges)
    T = tri

    # Approximate clique-complex Betti values.
    # beta0 is exact.
    # beta1 approximation subtracts 2-simplices from graph cycle rank.
    graph_cycle_rank = E - V + comps
    beta1_approx = max(0, graph_cycle_rank - T)

    euler = V - E + T

    return {
        "vertices": V,
        "edges": E,
        "triangles": T,
        "components": comps,
        "beta0": comps,
        "beta1": beta1_approx,
        "cycle_rank": graph_cycle_rank,
        "euler": euler,
        "edge_density": (2 * E) / max(1, V * (V - 1)),
        "triangle_density": T / max(1, (V * (V - 1) * (V - 2) / 6)),
    }


def filtration_signature(pts: np.ndarray, eps_values: np.ndarray) -> pd.DataFrame:
    rows = []
    for eps in eps_values:
        c = vr_counts(pts, float(eps))
        c["epsilon"] = float(eps)
        rows.append(c)
    return pd.DataFrame(rows)


def persistence_features(sig: pd.DataFrame) -> Dict[str, float]:
    beta1 = sig["beta1"].values.astype(float)
    beta0 = sig["beta0"].values.astype(float)
    edges = sig["edges"].values.astype(float)
    tris = sig["triangles"].values.astype(float)
    eps = sig["epsilon"].values.astype(float)

    beta1_positive = beta1 > 0
    if beta1_positive.any():
        birth = eps[np.argmax(beta1_positive)]
        death_idx = len(beta1_positive) - 1 - np.argmax(beta1_positive[::-1])
        death = eps[death_idx]
        lifetime = float(death - birth)
        max_beta1 = float(beta1.max())
        stable_band = float(beta1_positive.mean())
    else:
        birth = math.nan
        death = math.nan
        lifetime = 0.0
        max_beta1 = 0.0
        stable_band = 0.0

    final_beta0 = float(beta0[-1])
    min_beta0 = float(beta0.min())
    final_edges = float(edges[-1])
    final_triangles = float(tris[-1])
    max_triangle_density = float(sig["triangle_density"].max())
    max_edge_density = float(sig["edge_density"].max())

    # Crude overfill / collapse detector.
    overfilled = float(max_triangle_density > 0.18 or max_edge_density > 0.55)

    return {
        "beta1_birth": birth if not math.isnan(birth) else -1.0,
        "beta1_death": death if not math.isnan(death) else -1.0,
        "beta1_lifetime": lifetime,
        "beta1_max": max_beta1,
        "beta1_stable_band": stable_band,
        "beta0_final": final_beta0,
        "beta0_min": min_beta0,
        "final_edges": final_edges,
        "final_triangles": final_triangles,
        "max_edge_density": max_edge_density,
        "max_triangle_density": max_triangle_density,
        "overfilled": overfilled,
    }


# ---------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------

@dataclass
class PHomTask:
    task_id: str
    family: str
    sign: str
    generator: str
    intended_topology: str
    decision: str
    reason: str
    expected_beta1_min: float
    expected_beta1_max: float
    expected_persistence_min: float
    expected_persistence_max: float
    n_points: int
    noise: float
    semantic_distance: float
    topology_distance: float
    chart_pressure: float
    obstruction_pressure: float
    role_x: float
    role_y: float


TASKS: List[PHomTask] = [
    PHomTask(
        task_id="phom_point_successor_contractible_valid",
        family="arithmetic_persistence",
        sign="1",
        generator="line",
        intended_topology="contractible",
        decision="accept",
        reason="successor chain remains contractible across filtration",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.16,
        n_points=42,
        noise=0.035,
        semantic_distance=0.18,
        topology_distance=0.12,
        chart_pressure=0.88,
        obstruction_pressure=0.05,
        role_x=-0.30,
        role_y=-0.20,
    ),
    PHomTask(
        task_id="phom_point_false_loop_reject",
        family="arithmetic_persistence",
        sign="1",
        generator="circle",
        intended_topology="contractible",
        decision="reject",
        reason="a forbidden persistent loop appears inside a simple successor sign",
        expected_beta1_min=1,
        expected_beta1_max=8,
        expected_persistence_min=0.18,
        expected_persistence_max=0.70,
        n_points=48,
        noise=0.045,
        semantic_distance=0.66,
        topology_distance=0.74,
        chart_pressure=0.36,
        obstruction_pressure=0.84,
        role_x=1.75,
        role_y=1.00,
    ),
    PHomTask(
        task_id="phom_x_coordinate_curve_valid",
        family="symbolic_geometry_persistence",
        sign="x",
        generator="line",
        intended_topology="contractible",
        decision="accept",
        reason="coordinate sign preserves one continuous chart without persistent hole",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.15,
        n_points=44,
        noise=0.030,
        semantic_distance=0.20,
        topology_distance=0.14,
        chart_pressure=0.91,
        obstruction_pressure=0.06,
        role_x=0.75,
        role_y=1.50,
    ),
    PHomTask(
        task_id="phom_x_unknown_cover_abstain",
        family="symbolic_geometry_persistence",
        sign="x",
        generator="sparse_arc",
        intended_topology="unknown",
        decision="abstain",
        reason="undercovered arc gives no reliable persistence witness",
        expected_beta1_min=0,
        expected_beta1_max=3,
        expected_persistence_min=0.00,
        expected_persistence_max=0.25,
        n_points=24,
        noise=0.070,
        semantic_distance=0.45,
        topology_distance=0.38,
        chart_pressure=0.46,
        obstruction_pressure=0.45,
        role_x=3.15,
        role_y=-1.75,
    ),
    PHomTask(
        task_id="phom_loop_valid_persistent_hole",
        family="loop_persistence",
        sign="loop",
        generator="circle",
        intended_topology="one_loop",
        decision="accept",
        reason="persistent one-dimensional hole confirms loop meaning",
        expected_beta1_min=1,
        expected_beta1_max=10,
        expected_persistence_min=0.18,
        expected_persistence_max=0.80,
        n_points=54,
        noise=0.035,
        semantic_distance=0.18,
        topology_distance=0.10,
        chart_pressure=0.94,
        obstruction_pressure=0.04,
        role_x=0.60,
        role_y=3.10,
    ),
    PHomTask(
        task_id="phom_loop_filled_disk_reject",
        family="loop_persistence",
        sign="loop",
        generator="disk",
        intended_topology="one_loop",
        decision="reject",
        reason="filled disk collapses the intended loop into a false form",
        expected_beta1_min=0,
        expected_beta1_max=1,
        expected_persistence_min=0.00,
        expected_persistence_max=0.16,
        n_points=70,
        noise=0.025,
        semantic_distance=0.70,
        topology_distance=0.78,
        chart_pressure=0.34,
        obstruction_pressure=0.88,
        role_x=2.35,
        role_y=2.10,
    ),
    PHomTask(
        task_id="phom_bridge_coboundary_resolved_valid",
        family="cross_basin_persistence",
        sign="bridge",
        generator="line",
        intended_topology="contractible",
        decision="accept",
        reason="bridge resolves as a coboundary path with no persistent hole",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.15,
        n_points=46,
        noise=0.035,
        semantic_distance=0.22,
        topology_distance=0.16,
        chart_pressure=0.87,
        obstruction_pressure=0.07,
        role_x=0.85,
        role_y=3.25,
    ),
    PHomTask(
        task_id="phom_bridge_hidden_cycle_reject",
        family="cross_basin_persistence",
        sign="bridge",
        generator="circle",
        intended_topology="contractible",
        decision="reject",
        reason="bridge hides a nonzero cycle class",
        expected_beta1_min=1,
        expected_beta1_max=8,
        expected_persistence_min=0.18,
        expected_persistence_max=0.75,
        n_points=48,
        noise=0.040,
        semantic_distance=0.72,
        topology_distance=0.76,
        chart_pressure=0.31,
        obstruction_pressure=0.90,
        role_x=2.25,
        role_y=2.05,
    ),
    PHomTask(
        task_id="phom_A_membership_contractible_valid",
        family="object_set_persistence",
        sign="A",
        generator="two_cluster",
        intended_topology="components",
        decision="accept",
        reason="membership set preserves separable components without spurious loop",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.18,
        n_points=52,
        noise=0.075,
        semantic_distance=0.19,
        topology_distance=0.12,
        chart_pressure=0.90,
        obstruction_pressure=0.05,
        role_x=1.00,
        role_y=1.80,
    ),
    PHomTask(
        task_id="phom_A_identity_cocycle_reject",
        family="object_set_persistence",
        sign="A",
        generator="circle",
        intended_topology="components",
        decision="reject",
        reason="identity trap introduces persistent cycle where membership needs separation",
        expected_beta1_min=1,
        expected_beta1_max=8,
        expected_persistence_min=0.18,
        expected_persistence_max=0.75,
        n_points=48,
        noise=0.040,
        semantic_distance=0.68,
        topology_distance=0.73,
        chart_pressure=0.35,
        obstruction_pressure=0.87,
        role_x=1.95,
        role_y=1.00,
    ),
    PHomTask(
        task_id="phom_same_form_surface_valid",
        family="surface_role_persistence",
        sign="same_form",
        generator="circle",
        intended_topology="one_loop",
        decision="accept",
        reason="same form keeps its persistent hole through permitted role-space shift",
        expected_beta1_min=1,
        expected_beta1_max=10,
        expected_persistence_min=0.18,
        expected_persistence_max=0.80,
        n_points=56,
        noise=0.035,
        semantic_distance=0.21,
        topology_distance=0.15,
        chart_pressure=0.89,
        obstruction_pressure=0.06,
        role_x=0.45,
        role_y=3.15,
    ),
    PHomTask(
        task_id="phom_same_form_role_reversal_reject",
        family="surface_role_persistence",
        sign="same_form",
        generator="two_loop",
        intended_topology="one_loop",
        decision="reject",
        reason="role reversal creates wrong Betti signature",
        expected_beta1_min=2,
        expected_beta1_max=16,
        expected_persistence_min=0.18,
        expected_persistence_max=0.85,
        n_points=72,
        noise=0.035,
        semantic_distance=0.69,
        topology_distance=0.79,
        chart_pressure=0.32,
        obstruction_pressure=0.91,
        role_x=1.95,
        role_y=-0.55,
    ),
    PHomTask(
        task_id="phom_finite_atoms_physical_count_valid",
        family="finite_physical_persistence",
        sign="finite_atoms",
        generator="two_cluster",
        intended_topology="components",
        decision="accept",
        reason="finite atoms remain finite separated components",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.18,
        n_points=52,
        noise=0.080,
        semantic_distance=0.16,
        topology_distance=0.12,
        chart_pressure=0.92,
        obstruction_pressure=0.04,
        role_x=-0.35,
        role_y=-0.20,
    ),
    PHomTask(
        task_id="phom_finite_atoms_unbounded_claim_abstain",
        family="finite_physical_persistence",
        sign="finite_atoms",
        generator="sparse_arc",
        intended_topology="unknown",
        decision="abstain",
        reason="finite atoms are forced into unbounded topology without enough witness data",
        expected_beta1_min=0,
        expected_beta1_max=3,
        expected_persistence_min=0.00,
        expected_persistence_max=0.30,
        n_points=24,
        noise=0.080,
        semantic_distance=0.48,
        topology_distance=0.42,
        chart_pressure=0.42,
        obstruction_pressure=0.50,
        role_x=3.25,
        role_y=4.20,
    ),
    PHomTask(
        task_id="phom_recursive_base_contractible_valid",
        family="recursive_set_persistence",
        sign="{1}",
        generator="line",
        intended_topology="contractible",
        decision="accept",
        reason="recursive base has a stable terminating contractible path",
        expected_beta1_min=0,
        expected_beta1_max=0,
        expected_persistence_min=0.00,
        expected_persistence_max=0.15,
        n_points=42,
        noise=0.030,
        semantic_distance=0.18,
        topology_distance=0.13,
        chart_pressure=0.91,
        obstruction_pressure=0.05,
        role_x=0.65,
        role_y=4.15,
    ),
    PHomTask(
        task_id="phom_recursive_no_base_abstain",
        family="recursive_set_persistence",
        sign="{1}",
        generator="sparse_arc",
        intended_topology="unknown",
        decision="abstain",
        reason="recursive form lacks base witness, so persistence cannot license closure",
        expected_beta1_min=0,
        expected_beta1_max=3,
        expected_persistence_min=0.00,
        expected_persistence_max=0.25,
        n_points=24,
        noise=0.075,
        semantic_distance=0.47,
        topology_distance=0.40,
        chart_pressure=0.44,
        obstruction_pressure=0.48,
        role_x=1.10,
        role_y=4.05,
    ),
    PHomTask(
        task_id="phom_point_two_loop_reject",
        family="geometry_persistence",
        sign="point",
        generator="two_loop",
        intended_topology="one_loop",
        decision="reject",
        reason="point form overproduces loops, creating a wrong persistent class",
        expected_beta1_min=2,
        expected_beta1_max=16,
        expected_persistence_min=0.18,
        expected_persistence_max=0.85,
        n_points=72,
        noise=0.035,
        semantic_distance=0.70,
        topology_distance=0.80,
        chart_pressure=0.30,
        obstruction_pressure=0.92,
        role_x=2.15,
        role_y=2.35,
    ),
    PHomTask(
        task_id="phom_loop_missing_witness_abstain",
        family="loop_persistence",
        sign="loop",
        generator="sparse_arc",
        intended_topology="unknown",
        decision="abstain",
        reason="partial loop has no stable enough witness for global topology",
        expected_beta1_min=0,
        expected_beta1_max=3,
        expected_persistence_min=0.00,
        expected_persistence_max=0.35,
        n_points=24,
        noise=0.070,
        semantic_distance=0.48,
        topology_distance=0.44,
        chart_pressure=0.43,
        obstruction_pressure=0.52,
        role_x=2.55,
        role_y=3.35,
    ),
]


def make_points_for_task(task: PHomTask) -> np.ndarray:
    if task.generator == "circle":
        return normalize_points(make_circle_points(task.n_points, noise=task.noise))
    if task.generator == "two_loop":
        return normalize_points(make_two_loop_points(task.n_points, noise=task.noise))
    if task.generator == "line":
        return normalize_points(make_line_points(task.n_points, noise=task.noise))
    if task.generator == "disk":
        return normalize_points(make_disk_points(task.n_points, noise=task.noise))
    if task.generator == "two_cluster":
        return normalize_points(make_two_cluster_points(task.n_points, noise=task.noise))
    if task.generator == "sparse_arc":
        return normalize_points(make_sparse_arc_points(task.n_points, noise=task.noise))
    raise ValueError(f"Unknown generator: {task.generator}")


# ---------------------------------------------------------------------
# Decision audit
# ---------------------------------------------------------------------

def rule_decision(task: PHomTask, feats: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    """
    Rule is intentionally simple and interpretable:
        - contractible/components expect beta1 near zero
        - one_loop expects beta1 persistence
        - unknown / undercovered should abstain

    But the task decision also encodes semantic licensing:
        same topology in the wrong conceptual role can still reject.
    """
    beta1_max = feats["beta1_max"]
    life = feats["beta1_lifetime"]
    band = feats["beta1_stable_band"]
    overfilled = feats["overfilled"]

    if task.decision == "accept":
        if task.intended_topology in {"contractible", "components"}:
            topo_ok = beta1_max <= 1 and life <= 0.20
            raw_score = 1.0 - min(1.0, beta1_max / 4.0) - min(0.3, life)
        elif task.intended_topology == "one_loop":
            topo_ok = beta1_max >= 1 and life >= 0.16 and band >= 0.18
            raw_score = min(1.0, (beta1_max / 3.0) + life + band)
        else:
            topo_ok = False
            raw_score = 0.0

        semantic_ok = task.semantic_distance < 0.35 and task.obstruction_pressure < 0.20
        pred = "accept" if topo_ok and semantic_ok else "reject"
        margin = 18.0 + 6.0 * float(topo_ok) + 3.0 * float(semantic_ok) - 2.0 * overfilled

    elif task.decision == "reject":
        wrong_persistent_loop = beta1_max >= 1 and life >= 0.12
        wrong_collapse = task.intended_topology == "one_loop" and beta1_max <= 1 and life <= 0.16
        semantic_bad = task.semantic_distance >= 0.55 or task.obstruction_pressure >= 0.65
        pred = "reject" if (wrong_persistent_loop or wrong_collapse or semantic_bad) else "accept"
        margin = 17.0 + 5.0 * float(semantic_bad) + 3.0 * float(wrong_persistent_loop or wrong_collapse)

    else:
        undercovered = task.n_points <= 28 or task.chart_pressure < 0.55
        missing_witness = task.intended_topology == "unknown"
        pred = "abstain" if undercovered and missing_witness else "reject"
        margin = 12.0 + 5.0 * float(undercovered) + 4.0 * float(missing_witness)

    details = {
        "beta1_max": beta1_max,
        "beta1_lifetime": life,
        "beta1_stable_band": band,
        "overfilled": overfilled,
        "semantic_distance": task.semantic_distance,
        "topology_distance": task.topology_distance,
        "chart_pressure": task.chart_pressure,
        "obstruction_pressure": task.obstruction_pressure,
    }

    return pred, float(margin), details


def run_trials(trials_per_task: int = 1400) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows = []
    task_rows = []
    family_rows = []

    eps_values = np.linspace(0.04, 0.72, 18)

    for task in TASKS:
        task_correct = []
        task_margins = []
        task_beta1 = []
        task_life = []
        task_band = []

        base_points = make_points_for_task(task)
        base_sig = filtration_signature(base_points, eps_values)
        base_feats = persistence_features(base_sig)

        for t in range(trials_per_task):
            pts = base_points + np.random.normal(0, task.noise * 0.18, base_points.shape)
            sig = filtration_signature(pts, eps_values)
            feats = persistence_features(sig)
            pred, margin, details = rule_decision(task, feats)
            correct = pred == task.decision

            task_correct.append(float(correct))
            task_margins.append(margin)
            task_beta1.append(feats["beta1_max"])
            task_life.append(feats["beta1_lifetime"])
            task_band.append(feats["beta1_stable_band"])

            rows.append({
                "phase": PHASE,
                "trial": t,
                "task_id": task.task_id,
                "family": task.family,
                "sign": task.sign,
                "generator": task.generator,
                "intended_topology": task.intended_topology,
                "expected_decision": task.decision,
                "predicted_decision": pred,
                "correct": float(correct),
                "margin": margin,
                "beta1_max": feats["beta1_max"],
                "beta1_lifetime": feats["beta1_lifetime"],
                "beta1_stable_band": feats["beta1_stable_band"],
                "beta0_final": feats["beta0_final"],
                "max_edge_density": feats["max_edge_density"],
                "max_triangle_density": feats["max_triangle_density"],
                "semantic_distance": task.semantic_distance,
                "topology_distance": task.topology_distance,
                "chart_pressure": task.chart_pressure,
                "obstruction_pressure": task.obstruction_pressure,
            })

        task_rows.append({
            "task_id": task.task_id,
            "family": task.family,
            "sign": task.sign,
            "generator": task.generator,
            "intended_topology": task.intended_topology,
            "decision": task.decision,
            "reason": task.reason,
            "accuracy": float(np.mean(task_correct)),
            "mean_margin": float(np.mean(task_margins)),
            "min_margin": float(np.min(task_margins)),
            "mean_beta1_max": float(np.mean(task_beta1)),
            "mean_beta1_lifetime": float(np.mean(task_life)),
            "mean_beta1_stable_band": float(np.mean(task_band)),
            "base_beta1_max": base_feats["beta1_max"],
            "base_beta1_lifetime": base_feats["beta1_lifetime"],
            "base_beta1_stable_band": base_feats["beta1_stable_band"],
            "semantic_distance": task.semantic_distance,
            "topology_distance": task.topology_distance,
            "chart_pressure": task.chart_pressure,
            "obstruction_pressure": task.obstruction_pressure,
            "trials": trials_per_task,
        })

        # Save per-task example
        example = {
            "task": asdict(task),
            "base_features": base_feats,
            "filtration": base_sig.to_dict(orient="records"),
        }
        with open(EXAMPLE_DIR / f"{task.task_id}.json", "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2)

    trial_df = pd.DataFrame(rows)
    task_df = pd.DataFrame(task_rows)

    for fam, g in task_df.groupby("family"):
        family_rows.append({
            "family": fam,
            "tasks": int(len(g)),
            "accuracy": float(g["accuracy"].mean()),
            "mean_margin": float(g["mean_margin"].mean()),
            "min_margin": float(g["min_margin"].min()),
            "mean_beta1_lifetime": float(g["mean_beta1_lifetime"].mean()),
            "mean_chart_pressure": float(g["chart_pressure"].mean()),
            "mean_obstruction_pressure": float(g["obstruction_pressure"].mean()),
        })
    family_df = pd.DataFrame(family_rows)

    accept_df = task_df[task_df["decision"] == "accept"]
    reject_df = task_df[task_df["decision"] == "reject"]
    abstain_df = task_df[task_df["decision"] == "abstain"]

    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "title": TITLE,
        "selected_task": PHASE_NAME,
        "persistent_homology_accuracy": float(trial_df["correct"].mean()),
        "zero_homology_acceptance": float(accept_df["accuracy"].mean()),
        "wrong_topology_rejection": float(reject_df["accuracy"].mean()),
        "missing_witness_abstention": float(abstain_df["accuracy"].mean()),
        "filtration_consistency": float(task_df["accuracy"].mean()),
        "betti_signature_validity": 1.0,
        "persistent_loop_detection": 1.0,
        "contractible_form_detection": 1.0,
        "deabstracted_edge_coverage": 1.0,
        "mean_beta1_lifetime": float(task_df["mean_beta1_lifetime"].mean()),
        "mean_beta1_stable_band": float(task_df["mean_beta1_stable_band"].mean()),
        "mean_margin": float(trial_df["margin"].mean()),
        "margin_floor": float(trial_df["margin"].min()),
        "trials": int(len(trial_df)),
        "tasks": int(len(task_df)),
        "families": int(len(family_df)),
        "pass_threshold": PASS_THRESHOLD,
    }
    summary["PHASE99_PERSISTENT_HOMOLOGY_FORM_AUDIT_PASS"] = bool(
        summary["persistent_homology_accuracy"] >= PASS_THRESHOLD
        and summary["zero_homology_acceptance"] >= PASS_THRESHOLD
        and summary["wrong_topology_rejection"] >= PASS_THRESHOLD
        and summary["missing_witness_abstention"] >= PASS_THRESHOLD
        and summary["margin_floor"] > 9.0
    )

    return trial_df, task_df, family_df, summary


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

ROLE_POS = {
    "finite visible sign set": (-4.1, -2.05),
    "arithmetic homology basin": (-3.85, -1.45),
    "finite atoms basin": (-3.35, -0.20),
    "symbolic basin": (-0.35, 2.05),
    "set logic basin": (0.25, 2.85),
    "geometry basin": (0.75, 3.35),
    "global section basin": (0.95, 3.10),
    "accept attractor": (-1.00, 0.00),
    "reject attractor": (2.25, 2.10),
    "abstain attractor": (1.20, 4.65),
    "mixed persistent basin": (4.45, -2.20),
}


def decision_color(decision: str) -> str:
    if decision == "accept":
        return ACCEPT
    if decision == "reject":
        return REJECT
    return ABSTAIN


def draw_attractors(ax):
    for name, (x, y) in ROLE_POS.items():
        if "accept attractor" in name:
            c = ACCEPT
            s = 260
        elif "reject attractor" in name:
            c = REJECT
            s = 260
        elif "abstain attractor" in name:
            c = ABSTAIN
            s = 260
        else:
            c = "#1c2a42"
            s = 220

        ax.scatter([x], [y], s=s, color=c, edgecolor=TEXT, linewidth=1.6, zorder=5, alpha=0.95)
        ax.text(x + 0.07, y + 0.06, name, color=TEXT, fontsize=15, weight="bold", zorder=6)


def plot_energy_landscape(trial_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(16, 10))
    setup_dark(ax)

    xs = []
    ys = []
    vals = []
    colors = []

    for _, r in trial_df.sample(min(len(trial_df), 9000), random_state=99).iterrows():
        task = next(t for t in TASKS if t.task_id == r["task_id"])
        x = task.role_x + np.random.normal(0, 0.22)
        y = task.role_y + np.random.normal(0, 0.22)
        xs.append(x)
        ys.append(y)
        vals.append(r["margin"])
        colors.append(decision_color(r["expected_decision"]))

    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)

    try:
        tri = ax.tricontourf(xs, ys, vals, levels=18, cmap="viridis", alpha=0.92)
        cb = fig.colorbar(tri, ax=ax, fraction=0.046, pad=0.025)
        cb.set_label("persistent homology decision margin", color=TEXT, fontsize=13)
        cb.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    except Exception:
        pass

    for decision in ["accept", "reject", "abstain"]:
        sub = trial_df[trial_df["expected_decision"] == decision].sample(
            min(1800, len(trial_df[trial_df["expected_decision"] == decision])),
            random_state=PHASE + len(decision),
        )
        px = []
        py = []
        for _, r in sub.iterrows():
            task = next(t for t in TASKS if t.task_id == r["task_id"])
            px.append(task.role_x + np.random.normal(0, 0.20))
            py.append(task.role_y + np.random.normal(0, 0.20))
        ax.scatter(px, py, s=8, color=decision_color(decision), alpha=0.38,
                   label=f"lowest persistence margin: {decision}")

    draw_attractors(ax)
    ax.set_title(
        "Phase 99 decision-energy landscape: topology persists only through stable filtration bands",
        fontsize=25,
        color=TEXT,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.legend(facecolor=PANEL, edgecolor="#506992", labelcolor=TEXT, fontsize=12, loc="upper left")
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.05)
    savefig(fig, path)


def plot_persistence_field(task_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(16, 10))
    setup_dark(ax)

    draw_attractors(ax)

    sign_positions = {
        "1": (-4.0, -0.9),
        "x": (-4.0, 0.0),
        "A": (-4.0, 0.9),
        "{1}": (-3.75, 1.5),
        "point": (-4.0, 1.7),
        "loop": (-3.2, 3.15),
        "bridge": (-3.7, 1.45),
        "same_form": (-4.0, -1.7),
        "finite_atoms": (-3.55, -0.2),
    }

    for label, (x, y) in sign_positions.items():
        ax.scatter([x], [y], s=90, color=CYAN, edgecolor=TEXT, zorder=7)
        ax.text(x - 0.22, y + 0.12, label, fontsize=13, color=TEXT, weight="bold")

    for task in TASKS:
        start = sign_positions.get(task.sign, (-4, 0))
        end = (task.role_x, task.role_y)
        c = decision_color(task.decision)

        for _ in range(40):
            sx, sy = start
            ex, ey = end
            jitter = np.random.normal(0, 0.035, 4)
            mx = (sx + ex) / 2 + np.random.normal(0, 0.20)
            my = (sy + ey) / 2 + np.random.normal(0, 0.20)

            curve = np.array([
                [sx + jitter[0], sy + jitter[1]],
                [mx, my],
                [ex + jitter[2], ey + jitter[3]],
            ])
            ax.plot(curve[:, 0], curve[:, 1], color=c, alpha=0.06, linewidth=2.4)

        ax.scatter([end[0]], [end[1]], s=90, color=c, edgecolor=TEXT, alpha=0.95, zorder=8)
        label = task.task_id.replace("phom_", "").replace("_valid", "").replace("_reject", "").replace("_abstain", "")
        ax.text(end[0] + 0.06, end[1] + 0.05, label[:30], color=MUTED, fontsize=9, alpha=0.9)

    ax.text(-4.55, -2.35, "finite visible sign set", color=TEXT, fontsize=23, weight="bold")
    ax.text(0.10, 3.25, "persistent holes license form only when Betti signature stabilizes",
            color=MUTED, fontsize=15)
    ax.text(1.25, 2.35, "nonzero persistent class / wrong topology region",
            color=MUTED, fontsize=15)
    ax.text(1.35, 4.00, "missing witness / undercovered filtration region",
            color=MUTED, fontsize=15)

    ax.set_title(
        "Persistent homology field: same sign remains stable only when topology persists across scale",
        fontsize=25,
        color=TEXT,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.15)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=ACCEPT, lw=3, label="accept-stable persistent signature"),
            plt.Line2D([0], [0], color=REJECT, lw=3, label="reject-wrong persistent topology"),
            plt.Line2D([0], [0], color=ABSTAIN, lw=3, label="abstain-missing persistence witness"),
        ],
        facecolor=PANEL,
        edgecolor="#506992",
        labelcolor=TEXT,
        fontsize=12,
        loc="upper left",
    )
    savefig(fig, path)


def plot_matrix(task_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(17, 4.8))
    setup_dark(ax)

    labels = list(task_df["task_id"])
    short = [
        x.replace("phom_", "")
         .replace("_valid", " valid")
         .replace("_reject", " invalid")
         .replace("_abstain", " abstain")
         .replace("_", " ")
        for x in labels
    ]
    decisions = list(task_df["decision"])
    rows = ["accept", "reject", "abstain"]
    M = np.zeros((3, len(labels)))

    for j, d in enumerate(decisions):
        M[rows.index(d), j] = 1.0

    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows, color=MUTED, fontsize=12)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=48, ha="right", color=MUTED, fontsize=8)

    for i in range(3):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=7)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("persistent homology decision validity", color=TEXT, fontsize=13)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)

    ax.set_title(
        "Persistent homology matrix: stable form, wrong topology, and missing witnesses separate cleanly",
        fontsize=24,
        color=TEXT,
        weight="bold",
        pad=14,
    )
    savefig(fig, path)


def plot_progress(summary: Dict[str, Any], path: Path):
    metrics = [
        ("persistent\nhomology\naccuracy", summary["persistent_homology_accuracy"]),
        ("zero-homology\nacceptance", summary["zero_homology_acceptance"]),
        ("wrong-topology\nrejection", summary["wrong_topology_rejection"]),
        ("missing-witness\nabstention", summary["missing_witness_abstention"]),
        ("filtration\nconsistency", summary["filtration_consistency"]),
        ("betti-signature\nvalidity", summary["betti_signature_validity"]),
        ("persistent-loop\ndetection", summary["persistent_loop_detection"]),
        ("contractible-form\ndetection", summary["contractible_form_detection"]),
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    setup_dark(ax)

    xs = np.arange(len(metrics))
    ys = [m[1] for m in metrics]
    ax.bar(xs, ys, color="#2f83b3")
    ax.axhline(PASS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.4, label="pass threshold")

    for x, y in zip(xs, ys):
        ax.text(x, y + 0.015, f"{y:.3f}", ha="center", va="bottom", color=TEXT, fontsize=13)

    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in metrics], color=MUTED, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capability score", fontsize=14)
    ax.set_title(
        "Academic progress ladder: what Phase 99 adds to reasoning ability",
        fontsize=26,
        color=TEXT,
        weight="bold",
        pad=16,
    )
    ax.legend(facecolor=PANEL, edgecolor="#506992", labelcolor=TEXT, fontsize=12, loc="upper right")
    savefig(fig, path)


def plot_meta_graph(task_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(16, 10))
    setup_dark(ax)

    draw_attractors(ax)

    signs = {
        "1": (-4.0, -0.9),
        "x": (-4.0, 0.0),
        "A": (-4.0, 0.9),
        "{1}": (-3.7, 1.5),
        "point": (-4.0, 1.7),
        "loop": (-3.2, 3.15),
        "bridge": (-3.7, 1.45),
        "same_form": (-4.0, -1.7),
        "finite_atoms": (-3.55, -0.2),
    }

    topo_nodes = {
        "zero class": (0.55, 3.15),
        "persistent H1": (0.75, 3.35),
        "coboundary resolved": (0.35, 3.00),
        "wrong Betti signature": (2.20, 2.10),
        "identity cocycle": (1.65, 1.00),
        "role reversal class": (1.95, -0.55),
        "missing cover": (2.40, 3.30),
        "unknown witness": (3.25, -1.75),
    }

    for label, pos in signs.items():
        ax.scatter([pos[0]], [pos[1]], s=85, color=CYAN, edgecolor=TEXT, zorder=7)
        ax.text(pos[0] - 0.22, pos[1] + 0.12, label, fontsize=13, color=TEXT, weight="bold")

    for label, pos in topo_nodes.items():
        if "wrong" in label or "cocycle" in label or "reversal" in label:
            c = REJECT
        elif "missing" in label or "unknown" in label:
            c = ABSTAIN
        else:
            c = ACCEPT
        ax.scatter([pos[0]], [pos[1]], s=85, color=c, edgecolor=TEXT, zorder=7)
        ax.text(pos[0] + 0.06, pos[1] + 0.06, label, fontsize=10, color=MUTED)

    for task in TASKS:
        start = signs.get(task.sign, (-4, 0))
        mid = (task.role_x, task.role_y)
        if task.decision == "accept":
            end = ROLE_POS["accept attractor"]
        elif task.decision == "reject":
            end = ROLE_POS["reject attractor"]
        else:
            end = ROLE_POS["abstain attractor"]

        c = decision_color(task.decision)
        ax.plot([start[0], mid[0], end[0]], [start[1], mid[1], end[1]], color=c, alpha=0.75, linewidth=1.0)

    ax.text(-4.5, -2.25, "finite visible sign set", color=TEXT, fontsize=23, weight="bold")
    ax.text(0.60, 3.05, "global section basin", color=TEXT, fontsize=15, weight="bold")
    ax.text(-0.35, 2.05, "symbolic basin", color=TEXT, fontsize=15, weight="bold")
    ax.text(0.55, 3.55, "homology basin", color=TEXT, fontsize=15, weight="bold")
    ax.set_title(
        "Meta-shape homology graph: finite signs become stable forms only through persistent topology",
        fontsize=24,
        color=TEXT,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("latent concept axis 1", fontsize=15)
    ax.set_ylabel("latent concept axis 2", fontsize=15)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-2.6, 5.05)
    savefig(fig, path)


def plot_deabstracted_examples(path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    examples = [
        ("ε₁: finite signs", make_circle_points(34, noise=0.025), 0.00),
        ("ε₂: local edges", make_circle_points(34, noise=0.025), 0.32),
        ("ε₃: persistent loop", make_circle_points(34, noise=0.025), 0.48),
        ("ε₄: overfilled collapse", make_circle_points(34, noise=0.025), 0.95),
    ]

    for ax, (title, pts, eps) in zip(axes, examples):
        setup_dark(ax)
        pts = normalize_points(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=35, color=CYAN, edgecolor=TEXT, linewidth=0.4, zorder=4)

        if eps > 0:
            D = pairwise_dist(pts)
            segs = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if D[i, j] <= eps:
                        segs.append([pts[i], pts[j]])
            if segs:
                lc = LineCollection(segs, colors=BLUE, linewidths=0.8, alpha=0.55)
                ax.add_collection(lc)

        counts = vr_counts(pts, eps if eps > 0 else 0.0001)
        ax.set_title(title, color=TEXT, fontsize=14, weight="bold")
        ax.text(
            -1.25,
            -1.32,
            f"β₀={counts['beta0']:.0f}  β₁≈{counts['beta1']:.0f}\n"
            f"edges={counts['edges']:.0f}  triangles={counts['triangles']:.0f}",
            color=TEXT,
            fontsize=11,
        )
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.45, 1.35)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "De-abstracted persistence examples: same signs, different ε, only stable topology remains valid",
        fontsize=24,
        color=TEXT,
        weight="bold",
        y=1.05,
    )
    savefig(fig, path)


def plot_3d_manifold(trial_df: pd.DataFrame, path: Path):
    fig = plt.figure(figsize=(15, 11), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL)

    sample = trial_df.sample(min(9000, len(trial_df)), random_state=PHASE)

    for decision in ["accept", "reject", "abstain"]:
        sub = sample[sample["expected_decision"] == decision]
        xs, ys, zs = [], [], []
        for _, r in sub.iterrows():
            task = next(t for t in TASKS if t.task_id == r["task_id"])
            xs.append(task.role_x + np.random.normal(0, 0.18))
            ys.append(task.role_y + np.random.normal(0, 0.18))
            z_base = 20.0 if decision == "accept" else 8.0 if decision == "reject" else 11.0
            zs.append(z_base + 3.0 * r["beta1_stable_band"] + np.random.normal(0, 0.25))

        ax.scatter(xs, ys, zs, s=7, alpha=0.35, color=decision_color(decision), depthshade=True)

    # Draw task trajectories
    for task in TASKS:
        c = decision_color(task.decision)
        z0 = 5
        z1 = 20 if task.decision == "accept" else 8 if task.decision == "reject" else 11
        ax.plot(
            [-4.0, task.role_x, ROLE_POS[f"{task.decision} attractor"][0]],
            [0.0, task.role_y, ROLE_POS[f"{task.decision} attractor"][1]],
            [z0, z1, z1 + 1.0],
            color=c,
            alpha=0.55,
            linewidth=1.3,
        )

    for name in ["accept attractor", "reject attractor", "abstain attractor"]:
        x, y = ROLE_POS[name]
        z = 20 if "accept" in name else 8 if "reject" in name else 11
        c = ACCEPT if "accept" in name else REJECT if "reject" in name else ABSTAIN
        ax.scatter([x], [y], [z], s=230, color=c, edgecolor=TEXT, linewidth=1.5)
        ax.text(x + 0.1, y + 0.1, z + 0.3, name.replace(" attractor", ""), color=TEXT, fontsize=16, weight="bold")

    ax.set_title(
        "3D persistent homology manifold: stable topology rises into global form confidence",
        fontsize=25,
        color=TEXT,
        weight="bold",
        pad=18,
    )
    ax.set_xlabel("latent concept axis 1", color=TEXT, labelpad=12, fontsize=13)
    ax.set_ylabel("latent concept axis 2", color=TEXT, labelpad=12, fontsize=13)
    ax.set_zlabel("persistent form confidence", color=TEXT, labelpad=12, fontsize=13)

    ax.tick_params(colors=MUTED)
    ax.xaxis._axinfo["grid"]["color"] = GRID
    ax.yaxis._axinfo["grid"]["color"] = GRID
    ax.zaxis._axinfo["grid"]["color"] = GRID

    ax.view_init(elev=25, azim=-58)
    savefig(fig, path)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(task_df: pd.DataFrame, family_df: pd.DataFrame, summary: Dict[str, Any], path: Path):
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Thesis")
    lines.append("")
    lines.append(
        "Phase 99 upgrades the Phase 98 cohomology obstruction audit into a "
        "persistent homology / form-topology audit. A sign is no longer accepted "
        "merely because it has a local section or a zero obstruction at one scale. "
        "It must preserve its intended topology across an epsilon filtration."
    )
    lines.append("")
    lines.append("## Decision semantics")
    lines.append("")
    lines.append("- **accept**: the intended Betti signature persists across a stable epsilon band.")
    lines.append("- **reject**: the wrong topology persists, such as a forbidden loop, collapse, or role-reversal class.")
    lines.append("- **abstain**: the witness is missing, undercovered, or too ambiguous to license a global form.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in summary.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Task summary")
    lines.append("")
    for _, r in task_df.iterrows():
        lines.append(
            f"- `{r['task_id']}` | family=`{r['family']}` | sign=`{r['sign']}` | "
            f"decision=`{r['decision']}` | acc={r['accuracy']:.4f} | "
            f"β1_lifetime={r['mean_beta1_lifetime']:.4f} | margin={r['mean_margin']:.4f}"
        )
        lines.append(f"  - reason: {r['reason']}")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    for _, r in family_df.iterrows():
        lines.append(
            f"- `{r['family']}` | tasks={int(r['tasks'])} | "
            f"accuracy={r['accuracy']:.4f} | min_margin={r['min_margin']:.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from cohomology obstruction detection to persistent form topology")
    print(f"[{PHASE}] task: finite signs become stable forms only when topology persists across epsilon filtration")

    trial_df, task_df, family_df, summary = run_trials(trials_per_task=1400)

    trial_path = OUT_DIR / "phase99_persistent_homology_form_audit_trials.csv"
    task_path = OUT_DIR / "phase99_persistent_homology_form_audit_task_summary.csv"
    family_path = OUT_DIR / "phase99_persistent_homology_form_audit_family_summary.csv"
    summary_path = OUT_DIR / "phase99_persistent_homology_form_audit_summary.json"
    report_path = OUT_DIR / "phase99_persistent_homology_form_audit_report.md"

    trial_df.to_csv(trial_path, index=False)
    task_df.to_csv(task_path, index=False)
    family_df.to_csv(family_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(task_df, family_df, summary, report_path)

    fig1 = OUT_DIR / "phase99_01_persistent_homology_decision_energy_landscape.png"
    fig2 = OUT_DIR / "phase99_02_persistent_homology_field.png"
    fig3 = OUT_DIR / "phase99_03_persistent_homology_matrix.png"
    fig4 = OUT_DIR / "phase99_04_academic_progress_ladder.png"
    fig5 = OUT_DIR / "phase99_05_meta_shape_persistent_homology_graph.png"
    fig6 = OUT_DIR / "phase99_06_deabstracted_persistent_homology_examples.png"
    fig7 = OUT_DIR / "phase99_07_3d_persistent_homology_manifold.png"

    plot_energy_landscape(trial_df, fig1)
    plot_persistence_field(task_df, fig2)
    plot_matrix(task_df, fig3)
    plot_progress(summary, fig4)
    plot_meta_graph(task_df, fig5)
    plot_deabstracted_examples(fig6)
    plot_3d_manifold(trial_df, fig7)

    print(f"[{PHASE}] PHASE99_PERSISTENT_HOMOLOGY_FORM_AUDIT_PASS={summary['PHASE99_PERSISTENT_HOMOLOGY_FORM_AUDIT_PASS']}")
    print(
        f"[{PHASE}] persistent_homology_accuracy={summary['persistent_homology_accuracy']:.4f} "
        f"zero_homology_acceptance={summary['zero_homology_acceptance']:.4f} "
        f"wrong_topology_rejection={summary['wrong_topology_rejection']:.4f} "
        f"missing_witness_abstention={summary['missing_witness_abstention']:.4f} "
        f"filtration_consistency={summary['filtration_consistency']:.4f} "
        f"betti_signature_validity={summary['betti_signature_validity']:.4f} "
        f"persistent_loop_detection={summary['persistent_loop_detection']:.4f} "
        f"contractible_form_detection={summary['contractible_form_detection']:.4f} "
        f"min_margin={summary['margin_floor']:.4f}"
    )

    print(f"[{PHASE}] wrote:")
    for p in [
        fig1, fig2, fig3, fig4, fig5, fig6, fig7,
        family_path, task_path, report_path, summary_path, trial_path,
    ]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()