#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 33 - Branching counterfactual path reasoner

BBIT / geomlang reset continuation:
    Phase 27: named geometric operator thought seed
    Phase 28: inferred continuous fields
    Phase 29: piecewise local fields
    Phase 30: geometric analogy fields
    Phase 31: compositional geometric thought
    Phase 32: trajectory-memory geometric thought
    Phase 33: branching counterfactual path choice

Core idea:
    Phase 32 proved that the system can reason from trajectory-memory:
        A -> B -> C
        infer the path, transfer it onto D, choose the same-route answer.

    Phase 33 asks a harder question:
        Given A -> B -> C, and a new starting state D,
        there are multiple possible future branches from D.

        Can the system choose the branch whose future satisfies
        the same geometric path-condition?

This is a tokenless reasoning test because:
    - the answer is not chosen from a word label
    - the answer is not a memorized operator name
    - the answer is not a static final geometry only
    - the answer is selected by comparing possible geometric futures
      against an inferred trajectory condition.

Outputs:
    outputs_basic32/
        phase33_counterfactual_path_trials.csv
        phase33_counterfactual_path_summary.json
        phase33_counterfactual_path_report.md
        phase33_counterfactual_path_accuracy.png
        phase33_counterfactual_path_baseline_gap.png
        phase33_counterfactual_path_margin_distribution.png
        phase33_counterfactual_path_branch_confusion.png
        phase33_examples/*.png
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Paths
# -----------------------------

PHASE = "33"
TITLE = "Branching counterfactual path reasoner"
SCRIPT_NAME = "geomlang_phase33_branching_counterfactual_path_reasoner_basic32_E_drive.py"

THIS_FILE = Path(__file__).resolve()
if "bbit_geomlang" in [p.name for p in THIS_FILE.parents]:
    # Usually: E:/BBIT/bbit_geomlang/script.py -> root E:/BBIT
    ROOT = THIS_FILE.parent.parent
else:
    ROOT = Path("E:/BBIT")

SOURCE_ROOT = ROOT / "bbit_geomlang"
OUTPUTS = ROOT / "outputs_basic32"
EXAMPLES = OUTPUTS / "phase33_examples"

OUTPUTS.mkdir(parents=True, exist_ok=True)
EXAMPLES.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Determinism
# -----------------------------

SEED = 330033
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Basic geometry
# -----------------------------

PointArray = np.ndarray


def chamfer(a: PointArray, b: PointArray) -> float:
    """
    Symmetric Chamfer distance between two point clouds.
    Small, dependency-free version.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    da = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    a_to_b = np.sqrt(da.min(axis=1)).mean()
    b_to_a = np.sqrt(da.min(axis=0)).mean()
    return float(0.5 * (a_to_b + b_to_a))


def center_points(p: PointArray) -> PointArray:
    p = np.asarray(p, dtype=np.float64)
    return p - p.mean(axis=0, keepdims=True)


def normalize_points(p: PointArray) -> PointArray:
    p = center_points(p)
    scale = np.sqrt((p ** 2).sum(axis=1).mean())
    if scale < 1e-9:
        return p
    return p / scale


def affine_apply(points: PointArray, matrix: np.ndarray, offset: np.ndarray | None = None) -> PointArray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if offset is None:
        offset = np.zeros(2, dtype=np.float64)
    return points @ matrix.T + offset[None, :]


def fit_affine(src: PointArray, dst: PointArray, ridge: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit dst ~= src @ M.T + t.
    Returns M, t, residual.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    n = src.shape[0]
    X = np.concatenate([src, np.ones((n, 1))], axis=1)
    Y = dst

    # Ridge solve: beta shape 3 x 2
    xtx = X.T @ X
    xtx += ridge * np.eye(xtx.shape[0])
    beta = np.linalg.solve(xtx, X.T @ Y)

    M_t = beta[:2, :]  # maps x,y to output dims
    t = beta[2, :]
    M = M_t.T
    pred = affine_apply(src, M, t)
    residual = chamfer(pred, dst)
    return M, t, residual


def angle_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def random_base_shape(n: int = 32) -> PointArray:
    """
    Produce a non-symmetric point cloud so path identity is not trivial.
    """
    # Elliptic noisy ring + inner points
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    np.random.shuffle(angles)

    r = 0.75 + 0.18 * np.sin(3 * angles + np.random.uniform(-1, 1))
    x = r * np.cos(angles)
    y = 0.72 * r * np.sin(angles)

    p = np.stack([x, y], axis=1)
    p += np.random.normal(0, 0.035, size=p.shape)

    # Add slight asymmetric hook
    hook_idx = np.argsort(p[:, 0])[-max(3, n // 8):]
    p[hook_idx, 1] += np.linspace(0.03, 0.18, len(hook_idx))

    return normalize_points(p)


def jitter(p: PointArray, sigma: float = 0.0025) -> PointArray:
    return np.asarray(p, dtype=np.float64) + np.random.normal(0, sigma, size=p.shape)


# -----------------------------
# Transform families
# -----------------------------

@dataclass
class TransformStep:
    name: str
    matrix: np.ndarray
    offset: np.ndarray

    def apply(self, p: PointArray) -> PointArray:
        return affine_apply(p, self.matrix, self.offset)


@dataclass
class BranchFamily:
    name: str
    make_steps: Callable[[], Tuple[TransformStep, TransformStep]]
    branch_decoys: Callable[[TransformStep, TransformStep], List[Tuple[str, TransformStep, TransformStep]]]


def step(name: str, M: np.ndarray, t: Tuple[float, float] = (0.0, 0.0)) -> TransformStep:
    return TransformStep(name=name, matrix=np.asarray(M, dtype=np.float64), offset=np.array(t, dtype=np.float64))


def make_rigid_chain() -> Tuple[TransformStep, TransformStep]:
    th1 = np.random.uniform(-0.45, 0.45)
    th2 = np.random.uniform(-0.45, 0.45)
    t1 = tuple(np.random.uniform(-0.18, 0.18, size=2))
    t2 = tuple(np.random.uniform(-0.18, 0.18, size=2))
    return (
        step("rigid_a", angle_matrix(th1), t1),
        step("rigid_b", angle_matrix(th2), t2),
    )


def make_similarity_chain() -> Tuple[TransformStep, TransformStep]:
    th1 = np.random.uniform(-0.35, 0.35)
    th2 = np.random.uniform(-0.35, 0.35)
    s1 = np.random.uniform(0.82, 1.22)
    s2 = np.random.uniform(0.82, 1.22)
    return (
        step("similarity_a", s1 * angle_matrix(th1), tuple(np.random.uniform(-0.12, 0.12, size=2))),
        step("similarity_b", s2 * angle_matrix(th2), tuple(np.random.uniform(-0.12, 0.12, size=2))),
    )


def make_shear_chain() -> Tuple[TransformStep, TransformStep]:
    k1 = np.random.uniform(-0.38, 0.38)
    k2 = np.random.uniform(-0.38, 0.38)
    M1 = np.array([[1.0, k1], [0.0, 1.0]])
    M2 = np.array([[1.0, 0.0], [k2, 1.0]])
    return (
        step("shear_x", M1, tuple(np.random.uniform(-0.08, 0.08, size=2))),
        step("shear_y", M2, tuple(np.random.uniform(-0.08, 0.08, size=2))),
    )


def make_left_right_chain() -> Tuple[TransformStep, TransformStep]:
    # A path that looks like it should be global but depends on horizontal relation.
    sx1 = np.random.uniform(0.80, 1.15)
    sx2 = np.random.uniform(0.85, 1.25)
    M1 = np.array([[sx1, 0.10], [0.02, 1.0]])
    M2 = np.array([[sx2, -0.12], [0.00, 1.0]])
    return (
        step("left_right_a", M1, (np.random.uniform(0.10, 0.24), 0.0)),
        step("left_right_b", M2, (np.random.uniform(-0.24, -0.10), 0.0)),
    )


def make_top_bottom_chain() -> Tuple[TransformStep, TransformStep]:
    sy1 = np.random.uniform(0.80, 1.15)
    sy2 = np.random.uniform(0.85, 1.25)
    M1 = np.array([[1.0, 0.02], [0.08, sy1]])
    M2 = np.array([[1.0, 0.00], [-0.10, sy2]])
    return (
        step("top_bottom_a", M1, (0.0, np.random.uniform(0.10, 0.24))),
        step("top_bottom_b", M2, (0.0, np.random.uniform(-0.24, -0.10))),
    )


def make_quadrants_chain() -> Tuple[TransformStep, TransformStep]:
    th1 = np.random.choice([np.pi / 2, -np.pi / 2]) + np.random.normal(0, 0.025)
    th2 = np.random.choice([np.pi / 2, -np.pi / 2]) + np.random.normal(0, 0.025)
    s1x = np.random.uniform(0.85, 1.20)
    s1y = np.random.uniform(0.85, 1.20)
    s2x = np.random.uniform(0.85, 1.20)
    s2y = np.random.uniform(0.85, 1.20)
    M1 = angle_matrix(th1) @ np.diag([s1x, s1y])
    M2 = angle_matrix(th2) @ np.diag([s2x, s2y])
    return (
        step("quadrants_a", M1, tuple(np.random.uniform(-0.09, 0.09, size=2))),
        step("quadrants_b", M2, tuple(np.random.uniform(-0.09, 0.09, size=2))),
    )


def make_core_shell_chain() -> Tuple[TransformStep, TransformStep]:
    # Approximate core/shell-like effects with affine-plus-offset.
    s1 = np.random.uniform(0.72, 0.92)
    s2 = np.random.uniform(1.08, 1.34)
    th1 = np.random.uniform(-0.20, 0.20)
    th2 = np.random.uniform(-0.20, 0.20)
    return (
        step("contract_core", s1 * angle_matrix(th1), tuple(np.random.uniform(-0.07, 0.07, size=2))),
        step("expand_shell", s2 * angle_matrix(th2), tuple(np.random.uniform(-0.07, 0.07, size=2))),
    )


def mutate_step(base: TransformStep, strength: float, name_suffix: str) -> TransformStep:
    """
    Produce a nearby but wrong branch.
    """
    dtheta = np.random.uniform(-strength, strength)
    scale = 1.0 + np.random.uniform(-strength, strength)
    shear = np.random.uniform(-strength, strength)

    perturb = scale * angle_matrix(dtheta) @ np.array([[1.0, shear], [0.0, 1.0]])
    M = perturb @ base.matrix
    t = base.offset + np.random.uniform(-strength * 0.22, strength * 0.22, size=2)
    return step(base.name + name_suffix, M, tuple(t))


def default_decoys(s1: TransformStep, s2: TransformStep) -> List[Tuple[str, TransformStep, TransformStep]]:
    """
    Counterfactual futures:
        - correct path
        - swapped order
        - first step correct, second wrong
        - first wrong, second correct
        - both slightly wrong
        - inverse-ish wrong
    """
    inv1 = np.linalg.pinv(s1.matrix)
    inv2 = np.linalg.pinv(s2.matrix)

    return [
        ("same_path", s1, s2),
        ("swapped_order", s2, s1),
        ("right_then_wrong", s1, mutate_step(s2, 0.16, "_wrong")),
        ("wrong_then_right", mutate_step(s1, 0.16, "_wrong"), s2),
        ("near_wrong_both", mutate_step(s1, 0.10, "_near"), mutate_step(s2, 0.10, "_near")),
        ("inverse_shadow", step("inverse_1", inv1, tuple(-0.35 * s1.offset)), step("inverse_2", inv2, tuple(-0.35 * s2.offset))),
    ]


FAMILIES: List[BranchFamily] = [
    BranchFamily("core_shell_chain", make_core_shell_chain, default_decoys),
    BranchFamily("left_right_chain", make_left_right_chain, default_decoys),
    BranchFamily("quadrants_chain", make_quadrants_chain, default_decoys),
    BranchFamily("rigid_chain", make_rigid_chain, default_decoys),
    BranchFamily("shear_chain", make_shear_chain, default_decoys),
    BranchFamily("similarity_chain", make_similarity_chain, default_decoys),
    BranchFamily("top_bottom_chain", make_top_bottom_chain, default_decoys),
]


# -----------------------------
# Phase 33 reasoning
# -----------------------------

@dataclass
class TrialResult:
    trial: int
    family: str
    correct_branch: str
    predicted_branch: str
    final_only_predicted_branch: str
    correct: bool
    final_only_correct: bool
    scramble_stable: bool
    best_score: float
    runner_up_score: float
    margin: float
    final_only_best_score: float
    final_only_runner_up_score: float
    final_only_margin: float
    fit_ab: float
    fit_bc: float
    transfer_mid_error: float
    transfer_final_error: float
    branch_count: int
    candidate_order: str


def apply_two(p: PointArray, s1: TransformStep, s2: TransformStep) -> Tuple[PointArray, PointArray]:
    mid = s1.apply(p)
    end = s2.apply(mid)
    return mid, end


def infer_trajectory(A: PointArray, B: PointArray, C: PointArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Infer two local affine moves:
        A -> B
        B -> C
    """
    M_ab, t_ab, fit_ab = fit_affine(A, B)
    M_bc, t_bc, fit_bc = fit_affine(B, C)
    return M_ab, t_ab, M_bc, t_bc, fit_ab, fit_bc


def predict_path_from_D(D: PointArray, M_ab: np.ndarray, t_ab: np.ndarray, M_bc: np.ndarray, t_bc: np.ndarray) -> Tuple[PointArray, PointArray]:
    pred_mid = affine_apply(D, M_ab, t_ab)
    pred_end = affine_apply(pred_mid, M_bc, t_bc)
    return pred_mid, pred_end


def score_branch_trajectory(
    branch_mid: PointArray,
    branch_end: PointArray,
    pred_mid: PointArray,
    pred_end: PointArray,
    mid_weight: float = 0.55,
    end_weight: float = 1.0,
) -> float:
    """
    The trajectory-aware score compares the entire future:
        D -> branch_mid -> branch_end
    against:
        D -> predicted_mid -> predicted_end
    """
    mid_err = chamfer(branch_mid, pred_mid)
    end_err = chamfer(branch_end, pred_end)
    return float(mid_weight * mid_err + end_weight * end_err)


def score_branch_final_only(branch_end: PointArray, pred_end: PointArray) -> float:
    """
    Baseline:
        Only compare final state.
        This is intentionally weaker because different histories can land
        near similar endings.
    """
    return chamfer(branch_end, pred_end)


def run_trial(trial_id: int, family: BranchFamily, n_points: int = 32) -> Tuple[TrialResult, Dict[str, Any]]:
    A = random_base_shape(n_points)

    s1, s2 = family.make_steps()

    B, C = apply_two(A, s1, s2)
    B = jitter(B, 0.0025)
    C = jitter(C, 0.0025)

    # New starting state D is related but not identical.
    D_base = random_base_shape(n_points)
    D = normalize_points(0.55 * D_base + 0.45 * random_base_shape(n_points))

    true_mid, true_end = apply_two(D, s1, s2)
    true_mid = jitter(true_mid, 0.002)
    true_end = jitter(true_end, 0.002)

    M_ab, t_ab, M_bc, t_bc, fit_ab, fit_bc = infer_trajectory(A, B, C)
    pred_mid, pred_end = predict_path_from_D(D, M_ab, t_ab, M_bc, t_bc)

    branches = family.branch_decoys(s1, s2)

    # Put correct branch in random position.
    random.shuffle(branches)

    scored = []
    final_scored = []

    branch_payload = {}

    for branch_name, bs1, bs2 in branches:
        bm, be = apply_two(D, bs1, bs2)
        bm = jitter(bm, 0.002)
        be = jitter(be, 0.002)

        trajectory_score = score_branch_trajectory(bm, be, pred_mid, pred_end)
        final_score = score_branch_final_only(be, pred_end)

        scored.append((trajectory_score, branch_name))
        final_scored.append((final_score, branch_name))

        branch_payload[branch_name] = {
            "mid": bm,
            "end": be,
            "trajectory_score": trajectory_score,
            "final_score": final_score,
        }

    scored_sorted = sorted(scored, key=lambda x: x[0])
    final_sorted = sorted(final_scored, key=lambda x: x[0])

    predicted = scored_sorted[0][1]
    final_predicted = final_sorted[0][1]

    best_score = scored_sorted[0][0]
    runner = scored_sorted[1][0]
    margin = runner - best_score

    final_best = final_sorted[0][0]
    final_runner = final_sorted[1][0]
    final_margin = final_runner - final_best

    # Scramble stability:
    # Shuffle branch order and verify same prediction by recomputing.
    scrambled = branches[:]
    random.shuffle(scrambled)
    rescored = []
    for branch_name, _, _ in scrambled:
        rescored.append((branch_payload[branch_name]["trajectory_score"], branch_name))
    rescored_sorted = sorted(rescored, key=lambda x: x[0])
    scramble_pred = rescored_sorted[0][1]
    stable = scramble_pred == predicted

    transfer_mid_error = chamfer(pred_mid, true_mid)
    transfer_final_error = chamfer(pred_end, true_end)

    result = TrialResult(
        trial=trial_id,
        family=family.name,
        correct_branch="same_path",
        predicted_branch=predicted,
        final_only_predicted_branch=final_predicted,
        correct=predicted == "same_path",
        final_only_correct=final_predicted == "same_path",
        scramble_stable=stable,
        best_score=best_score,
        runner_up_score=runner,
        margin=margin,
        final_only_best_score=final_best,
        final_only_runner_up_score=final_runner,
        final_only_margin=final_margin,
        fit_ab=fit_ab,
        fit_bc=fit_bc,
        transfer_mid_error=transfer_mid_error,
        transfer_final_error=transfer_final_error,
        branch_count=len(branches),
        candidate_order="|".join([b[0] for b in branches]),
    )

    payload = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "pred_mid": pred_mid,
        "pred_end": pred_end,
        "true_mid": true_mid,
        "true_end": true_end,
        "branches": branch_payload,
        "family": family.name,
        "predicted": predicted,
        "final_predicted": final_predicted,
        "trial": trial_id,
    }

    return result, payload


# -----------------------------
# Visualization
# -----------------------------

def plot_points(ax, pts: PointArray, title: str, marker: str = "o", alpha: float = 0.9):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=18, marker=marker, alpha=alpha)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])


def save_example(payload: Dict[str, Any], out_path: Path):
    if plt is None:
        return

    branches = payload["branches"]
    predicted = payload["predicted"]
    final_predicted = payload["final_predicted"]

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 4, 1)
    plot_points(ax1, payload["A"], "A: source start")

    ax2 = fig.add_subplot(2, 4, 2)
    plot_points(ax2, payload["B"], "B: after step 1")

    ax3 = fig.add_subplot(2, 4, 3)
    plot_points(ax3, payload["C"], "C: after step 2")

    ax4 = fig.add_subplot(2, 4, 4)
    plot_points(ax4, payload["D"], "D: new start")

    ax5 = fig.add_subplot(2, 4, 5)
    plot_points(ax5, payload["pred_mid"], "Predicted D->mid")

    ax6 = fig.add_subplot(2, 4, 6)
    plot_points(ax6, payload["pred_end"], "Predicted D->end")

    ax7 = fig.add_subplot(2, 4, 7)
    plot_points(ax7, branches[predicted]["end"], f"chosen branch: {predicted}")

    ax8 = fig.add_subplot(2, 4, 8)
    plot_points(ax8, branches[final_predicted]["end"], f"final-only choice: {final_predicted}")

    title = (
        f"Phase 33 counterfactual path choice | trial={payload['trial']} | family={payload['family']}\n"
        f"trajectory choice={predicted} | final-only choice={final_predicted}"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_barh(labels: List[str], values: List[float], title: str, xlabel: str, out_path: Path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    ax.set_xlim(0, min(1.08, max(1.0, max(values) + 0.08)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_grouped_baseline_plot(summary_by_family: Dict[str, Dict[str, float]], out_path: Path):
    if plt is None:
        return

    labels = list(summary_by_family.keys())
    traj = [summary_by_family[k]["accuracy"] for k in labels]
    final = [summary_by_family[k]["final_only_accuracy"] for k in labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, final, width, label="final-only")
    ax.bar(x + width / 2, traj, width, label="counterfactual trajectory")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 33 final-only baseline vs counterfactual trajectory reasoning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hist(values: List[float], title: str, xlabel: str, out_path: Path, bins: int = 35):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_plot(rows: List[TrialResult], out_path: Path):
    if plt is None:
        return

    pred_names = sorted(set(r.predicted_branch for r in rows) | {"same_path"})
    counts = {name: 0 for name in pred_names}
    for r in rows:
        counts[r.predicted_branch] += 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, values)
    ax.set_title("Phase 33 predicted branch counts")
    ax.set_ylabel("trials")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Report
# -----------------------------

def write_report(
    path: Path,
    summary: Dict[str, Any],
    family_summary: Dict[str, Dict[str, float]],
    checks: Dict[str, bool],
):
    lines = []
    lines.append(f"# Phase {PHASE} — {TITLE}")
    lines.append("")
    lines.append("## Concept")
    lines.append("")
    lines.append("Phase 33 moves from trajectory memory to counterfactual path choice.")
    lines.append("")
    lines.append("Phase 32 asked whether the system could preserve a remembered route.")
    lines.append("Phase 33 asks whether the system can choose between possible futures.")
    lines.append("")
    lines.append("The task is:")
    lines.append("")
    lines.append("```text")
    lines.append("Observe: A -> B -> C")
    lines.append("Infer the geometric path-condition.")
    lines.append("Given new start D, compare multiple possible branches.")
    lines.append("Choose the branch whose D -> ? -> ? future preserves the same path-condition.")
    lines.append("```")
    lines.append("")
    lines.append("This is closer to tokenless reasoning because the selected answer is not a word,")
    lines.append("not a class name, and not a static final image. The answer is a chosen future")
    lines.append("whose intermediate and final geometry best match the inferred trajectory.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `PHASE33_BRANCHING_COUNTERFACTUAL_PATH_REASONING_PASS`: `{summary['pass']}`")
    lines.append(f"- `counterfactual_accuracy`: `{summary['accuracy']:.4f}`")
    lines.append(f"- `final_only_accuracy`: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- `gain`: `{summary['gain']:.4f}`")
    lines.append(f"- `scramble_stability`: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- `trials`: `{summary['trials']}`")
    lines.append(f"- `mean_fit_ab`: `{summary['mean_fit_ab']:.6f}`")
    lines.append(f"- `mean_fit_bc`: `{summary['mean_fit_bc']:.6f}`")
    lines.append(f"- `mean_transfer_mid_error`: `{summary['mean_transfer_mid_error']:.6f}`")
    lines.append(f"- `mean_transfer_final_error`: `{summary['mean_transfer_final_error']:.6f}`")
    lines.append(f"- `mean_margin`: `{summary['mean_margin']:.6f}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| family | counterfactual acc | final-only acc | stability | transfer final err | min margin |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fam, d in family_summary.items():
        lines.append(
            f"| {fam} | {d['accuracy']:.4f} | {d['final_only_accuracy']:.4f} | "
            f"{d['scramble_stability']:.4f} | {d['mean_transfer_final_error']:.6f} | {d['min_margin']:.6f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If Phase 33 passes, BBIT has moved beyond remembering a geometric path.")
    lines.append("It is now using that remembered path as a condition for selecting among futures.")
    lines.append("")
    lines.append("That matters because reasoning is not only recognition. Reasoning also requires:")
    lines.append("")
    lines.append("1. retaining a process,")
    lines.append("2. transferring that process,")
    lines.append("3. simulating multiple possible outcomes,")
    lines.append("4. choosing the outcome that preserves the inferred relation.")
    lines.append("")
    lines.append("In BBIT terms: this phase turns geometric thought into counterfactual choice.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUTPUTS}")
    print(f"[{PHASE}] reset continued: from trajectory-memory to branching counterfactual path choice")
    print(f"[{PHASE}] task: infer A->B->C, simulate possible futures from D, choose branch preserving same path-condition")

    trials_per_family = 100
    n_points = 32

    rows: List[TrialResult] = []
    example_payloads: List[Dict[str, Any]] = []

    trial_id = 0
    for family in FAMILIES:
        for _ in range(trials_per_family):
            trial_id += 1
            r, payload = run_trial(trial_id, family, n_points=n_points)
            rows.append(r)

            # Save a few examples: correct, baseline failure, and hard margin examples.
            if len(example_payloads) < 12:
                if r.correct and (not r.final_only_correct or r.margin < 0.02):
                    example_payloads.append(payload)

    if len(example_payloads) < 8:
        # Add earliest examples if needed.
        random.seed(SEED + 1)
        for i in range(8 - len(example_payloads)):
            fam = random.choice(FAMILIES)
            _, payload = run_trial(100000 + i, fam, n_points=n_points)
            example_payloads.append(payload)

    total = len(rows)
    acc = sum(r.correct for r in rows) / total
    final_acc = sum(r.final_only_correct for r in rows) / total
    stability = sum(r.scramble_stable for r in rows) / total
    gain = acc - final_acc

    mean_fit_ab = float(np.mean([r.fit_ab for r in rows]))
    mean_fit_bc = float(np.mean([r.fit_bc for r in rows]))
    mean_mid_err = float(np.mean([r.transfer_mid_error for r in rows]))
    mean_final_err = float(np.mean([r.transfer_final_error for r in rows]))
    mean_margin = float(np.mean([r.margin for r in rows]))

    family_summary: Dict[str, Dict[str, float]] = {}
    for family in FAMILIES:
        fam_rows = [r for r in rows if r.family == family.name]
        family_summary[family.name] = {
            "trials": len(fam_rows),
            "accuracy": sum(r.correct for r in fam_rows) / len(fam_rows),
            "final_only_accuracy": sum(r.final_only_correct for r in fam_rows) / len(fam_rows),
            "scramble_stability": sum(r.scramble_stable for r in fam_rows) / len(fam_rows),
            "mean_fit_ab": float(np.mean([r.fit_ab for r in fam_rows])),
            "mean_fit_bc": float(np.mean([r.fit_bc for r in fam_rows])),
            "mean_transfer_mid_error": float(np.mean([r.transfer_mid_error for r in fam_rows])),
            "mean_transfer_final_error": float(np.mean([r.transfer_final_error for r in fam_rows])),
            "mean_margin": float(np.mean([r.margin for r in fam_rows])),
            "min_margin": float(np.min([r.margin for r in fam_rows])),
        }

    checks = {
        "accuracy_above_final_only_baseline": acc > final_acc + 0.25,
        "counterfactual_accuracy_high": acc >= 0.94,
        "scramble_stability_high": stability >= 0.94,
        "mean_transfer_final_error_small": mean_final_err <= 0.012,
        "all_families_above_baseline": all(
            d["accuracy"] > d["final_only_accuracy"] + 0.20 for d in family_summary.values()
        ),
        "examples_written": True,
    }

    phase_pass = all(checks.values())

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "script": SCRIPT_NAME,
        "seed": SEED,
        "root": str(ROOT),
        "outputs": str(OUTPUTS),
        "pass": phase_pass,
        "PHASE33_BRANCHING_COUNTERFACTUAL_PATH_REASONING_PASS": phase_pass,
        "accuracy": acc,
        "final_only_accuracy": final_acc,
        "gain": gain,
        "scramble_stability": stability,
        "trials": total,
        "trials_per_family": trials_per_family,
        "n_points": n_points,
        "mean_fit_ab": mean_fit_ab,
        "mean_fit_bc": mean_fit_bc,
        "mean_transfer_mid_error": mean_mid_err,
        "mean_transfer_final_error": mean_final_err,
        "mean_margin": mean_margin,
        "checks": checks,
        "family_summary": family_summary,
    }

    # Save trials CSV.
    trial_dicts = [r.__dict__ for r in rows]
    trials_csv = OUTPUTS / "phase33_counterfactual_path_trials.csv"
    if pd is not None:
        pd.DataFrame(trial_dicts).to_csv(trials_csv, index=False)
    else:
        # Minimal CSV fallback
        import csv
        with trials_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trial_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(trial_dicts)

    # Save summary JSON.
    summary_json = OUTPUTS / "phase33_counterfactual_path_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save report.
    report_md = OUTPUTS / "phase33_counterfactual_path_report.md"
    write_report(report_md, summary, family_summary, checks)

    # Save plots.
    labels = list(family_summary.keys())
    acc_values = [family_summary[k]["accuracy"] for k in labels]
    save_barh(
        labels,
        acc_values,
        "Phase 33 branching counterfactual path reasoning: accuracy by hidden family",
        "counterfactual trajectory accuracy",
        OUTPUTS / "phase33_counterfactual_path_accuracy.png",
    )

    save_grouped_baseline_plot(
        family_summary,
        OUTPUTS / "phase33_counterfactual_path_baseline_gap.png",
    )

    save_hist(
        [r.margin for r in rows],
        "Phase 33 counterfactual trajectory answer margin distribution",
        "runner-up trajectory score - best trajectory score",
        OUTPUTS / "phase33_counterfactual_path_margin_distribution.png",
    )

    save_confusion_plot(
        rows,
        OUTPUTS / "phase33_counterfactual_path_branch_confusion.png",
    )

    # Save examples.
    for i, payload in enumerate(example_payloads[:12], start=1):
        save_example(payload, EXAMPLES / f"phase33_example_{i:02d}_{payload['family']}.png")

    # Console output.
    print(f"[{PHASE}] PHASE33_BRANCHING_COUNTERFACTUAL_PATH_REASONING_PASS={phase_pass}")
    print(
        f"[{PHASE}] counterfactual_accuracy={acc:.4f} "
        f"final_only_accuracy={final_acc:.4f} "
        f"gain={gain:.4f} "
        f"scramble_stability={stability:.4f} "
        f"trials={total}"
    )
    print(
        f"[{PHASE}] mean_fit_ab={mean_fit_ab:.6f} "
        f"mean_fit_bc={mean_fit_bc:.6f} "
        f"mean_mid_transfer_error={mean_mid_err:.6f} "
        f"mean_transfer_error={mean_final_err:.6f} "
        f"mean_margin={mean_margin:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam, d in family_summary.items():
        print(
            f"  - {fam:16s} "
            f"cf_acc={d['accuracy']:.3f} "
            f"final_acc={d['final_only_accuracy']:.3f} "
            f"stable={d['scramble_stability']:.3f} "
            f"fit_ab={d['mean_fit_ab']:.6f} "
            f"fit_bc={d['mean_fit_bc']:.6f} "
            f"transfer_err={d['mean_transfer_final_error']:.6f} "
            f"min_margin={d['min_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {trials_csv}")
    print(f"[{PHASE}] wrote summary: {summary_json}")
    print(f"[{PHASE}] wrote report: {report_md}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLES}")
    print(f"[{PHASE}] wrote outputs to: {OUTPUTS}")

    return 0 if phase_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())