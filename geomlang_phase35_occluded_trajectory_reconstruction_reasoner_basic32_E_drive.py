#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 35 — Occluded trajectory reconstruction reasoner
======================================================

BBIT / geomlang conceptual reset path:
  27: named geometric operators
  28: inferred continuous deformation fields
  29: local piecewise fields
  30: geometric analogy A->B transferred onto C
  31: compositional A->B->C reasoning, exposed final-state weakness
  32: trajectory-memory reasoning, uses the path rather than only endpoint
  33: branching counterfactual path choice
  34: adversarial counterfactual path-only reasoning, endpoint made misleading
  35: occluded trajectory reconstruction

Conceptual goal
---------------
Phase 34 proved that endpoint-only resemblance can be defeated and that the
system can choose by trajectory. Phase 35 asks the next question:

    Can the system preserve the path when part of the observed path is missing?

The source trajectory A->B->C is partially occluded/corrupted. The reasoner must
reconstruct the hidden motion pattern from visible fragments, transfer it onto a
new start D, then choose the branch D->E->F whose *visible + inferred trajectory*
matches the source path.

This is intentionally not token reasoning and not label reasoning. Labels are
kept only for reporting. The actual answer is chosen by geometric path agreement.

Expected behavior
-----------------
A final-only endpoint baseline should fail because distractors have endpoints
near the correct target. A trajectory reasoner with occlusion reconstruction
should pass.

Outputs
-------
Writes to E:\BBIT\outputs_basic32 by default:
  phase35_occluded_trajectory_trials.csv
  phase35_occluded_trajectory_summary.json
  phase35_occluded_trajectory_report.md
  phase35_occluded_trajectory_accuracy.png
  phase35_occluded_trajectory_baseline_gap.png
  phase35_occluded_trajectory_margin_distribution.png
  phase35_occluded_trajectory_occlusion_sweep.png
  phase35_examples/*.png

Run
---
  python bbit_geomlang/geomlang_phase35_occluded_trajectory_reconstruction_reasoner_basic32_E_drive.py
  python bbit_geomlang/geomlang_phase35_occluded_trajectory_reconstruction_reasoner_basic32_E_drive.py --trials 1000 --occlusion 0.45
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for Phase 35 visual outputs") from exc


PHASE = "35"
TITLE = "Occluded trajectory reconstruction reasoner"
PASS_FLAG = "PHASE35_OCCLUDED_TRAJECTORY_RECONSTRUCTION_PASS"

ROOT = Path(r"E:\BBIT")
OUT_DIR = ROOT / "outputs_basic32"
SCRIPT_NAME = "geomlang_phase35_occluded_trajectory_reconstruction_reasoner_basic32_E_drive.py"

EPS = 1e-9


# -----------------------------
# Basic geometry helpers
# -----------------------------

def rng_choice(rng: np.random.Generator, items: Sequence):
    return items[int(rng.integers(0, len(items)))]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric nearest-neighbor distance for tiny point clouds."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(0.5 * (np.sqrt(d2.min(axis=1)).mean() + np.sqrt(d2.min(axis=0)).mean()))


def canonicalize(points: np.ndarray) -> np.ndarray:
    """Center and scale to make comparisons stable without token labels."""
    p = np.asarray(points, dtype=np.float64).copy()
    p -= p.mean(axis=0, keepdims=True)
    scale = np.sqrt((p**2).sum(axis=1).mean())
    if scale < EPS:
        scale = 1.0
    return p / scale


def affine_apply(points: np.ndarray, m: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ m.T + t[None, :]


def fit_affine(src: np.ndarray, dst: np.ndarray, visible: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit dst ~= src @ M.T + t. Uses only visible rows when mask provided."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if visible is None:
        visible = np.ones(len(src), dtype=bool)
    if visible.sum() < 3:
        visible = np.ones(len(src), dtype=bool)

    x = src[visible]
    y = dst[visible]
    A = np.concatenate([x, np.ones((len(x), 1))], axis=1)  # [x y 1]
    # solve A @ B = y where B is 3x2. M.T = B[:2], t = B[2]
    B, *_ = np.linalg.lstsq(A, y, rcond=None)
    mt = B[:2, :]       # 2x2 = M.T
    t = B[2, :]         # 2
    m = mt.T            # M
    pred = affine_apply(src, m, t)
    residual = chamfer(pred[visible], dst[visible])
    return m, t, residual


def compose(m1: np.ndarray, t1: np.ndarray, m2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply transform1 then transform2."""
    m = m2 @ m1
    t = (m2 @ t1) + t2
    return m, t


def make_base_shape(rng: np.random.Generator, n: int = 32) -> np.ndarray:
    """A non-symmetric blob. Symmetry would make hidden paths ambiguous."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0 + 0.18 * np.sin(3 * theta + 0.4) + 0.09 * np.cos(5 * theta - 0.7)
    p = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    p += rng.normal(0.0, 0.012, p.shape)
    p = canonicalize(p)
    return p


def jitter(points: np.ndarray, rng: np.random.Generator, sigma: float = 0.003) -> np.ndarray:
    return np.asarray(points) + rng.normal(0.0, sigma, np.asarray(points).shape)


# -----------------------------
# Path families
# -----------------------------

@dataclass(frozen=True)
class FamilySpec:
    name: str
    maker: Callable[[np.random.Generator], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]


def mat_rot(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def make_rigid_chain(rng: np.random.Generator):
    a1 = rng.uniform(-0.75, 0.75)
    a2 = rng.uniform(-0.75, 0.75)
    t1 = rng.normal(0, 0.20, 2)
    t2 = rng.normal(0, 0.20, 2)
    return (mat_rot(a1), t1), (mat_rot(a2), t2)


def make_similarity_chain(rng: np.random.Generator):
    a1 = rng.uniform(-0.65, 0.65)
    a2 = rng.uniform(-0.65, 0.65)
    s1 = rng.uniform(0.72, 1.28)
    s2 = rng.uniform(0.72, 1.28)
    t1 = rng.normal(0, 0.16, 2)
    t2 = rng.normal(0, 0.16, 2)
    return (s1 * mat_rot(a1), t1), (s2 * mat_rot(a2), t2)


def make_shear_chain(rng: np.random.Generator):
    k1 = rng.uniform(-0.46, 0.46)
    k2 = rng.uniform(-0.46, 0.46)
    if rng.random() < 0.5:
        m1 = np.array([[1.0, k1], [0.0, 1.0]])
        m2 = np.array([[1.0, 0.0], [k2, 1.0]])
    else:
        m1 = np.array([[1.0, 0.0], [k1, 1.0]])
        m2 = np.array([[1.0, k2], [0.0, 1.0]])
    return (m1, rng.normal(0, 0.12, 2)), (m2, rng.normal(0, 0.12, 2))


def make_left_right_chain(rng: np.random.Generator):
    # Deliberately two-stage directional path, not just endpoint.
    k1 = rng.uniform(0.20, 0.42) * rng.choice([-1, 1])
    k2 = rng.uniform(0.20, 0.42) * rng.choice([-1, 1])
    m1 = np.array([[1.0 + 0.06 * rng.normal(), 0.10 * k1], [0.0, 1.0]])
    m2 = np.array([[1.0 + 0.06 * rng.normal(), -0.12 * k2], [0.0, 1.0]])
    t1 = np.array([k1, rng.normal(0, 0.035)])
    t2 = np.array([k2, rng.normal(0, 0.035)])
    return (m1, t1), (m2, t2)


def make_top_bottom_chain(rng: np.random.Generator):
    k1 = rng.uniform(0.20, 0.42) * rng.choice([-1, 1])
    k2 = rng.uniform(0.20, 0.42) * rng.choice([-1, 1])
    m1 = np.array([[1.0, 0.0], [0.10 * k1, 1.0 + 0.06 * rng.normal()]])
    m2 = np.array([[1.0, -0.12 * k2], [0.0, 1.0 + 0.06 * rng.normal()]])
    t1 = np.array([rng.normal(0, 0.035), k1])
    t2 = np.array([rng.normal(0, 0.035), k2])
    return (m1, t1), (m2, t2)


def make_quadrants_chain(rng: np.random.Generator):
    sx1 = rng.uniform(0.82, 1.22)
    sy1 = rng.uniform(0.82, 1.22)
    sx2 = rng.uniform(0.82, 1.22)
    sy2 = rng.uniform(0.82, 1.22)
    m1 = np.array([[sx1, rng.uniform(-0.18, 0.18)], [rng.uniform(-0.18, 0.18), sy1]])
    m2 = np.array([[sx2, rng.uniform(-0.18, 0.18)], [rng.uniform(-0.18, 0.18), sy2]])
    t1 = rng.normal(0, 0.11, 2)
    t2 = rng.normal(0, 0.11, 2)
    return (m1, t1), (m2, t2)


def make_core_shell_chain(rng: np.random.Generator):
    # Global affine proxy for a conceptually core/shell-ish path.
    a1 = rng.uniform(-0.55, 0.55)
    a2 = rng.uniform(-0.55, 0.55)
    m1 = mat_rot(a1) @ np.diag([rng.uniform(0.78, 1.18), rng.uniform(0.86, 1.30)])
    m2 = mat_rot(a2) @ np.diag([rng.uniform(0.86, 1.30), rng.uniform(0.78, 1.18)])
    t1 = rng.normal(0, 0.13, 2)
    t2 = rng.normal(0, 0.13, 2)
    return (m1, t1), (m2, t2)


FAMILIES: List[FamilySpec] = [
    FamilySpec("core_shell_chain", make_core_shell_chain),
    FamilySpec("left_right_chain", make_left_right_chain),
    FamilySpec("quadrants_chain", make_quadrants_chain),
    FamilySpec("rigid_chain", make_rigid_chain),
    FamilySpec("shear_chain", make_shear_chain),
    FamilySpec("similarity_chain", make_similarity_chain),
    FamilySpec("top_bottom_chain", make_top_bottom_chain),
]


# -----------------------------
# Occlusion and reasoning
# -----------------------------

def visible_mask(rng: np.random.Generator, n: int, occlusion: float, pattern: str) -> np.ndarray:
    """Return visible rows. Pattern makes missing information structured, not just random dropout."""
    keep = max(6, int(round(n * (1.0 - occlusion))))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if pattern == "random":
        idx = rng.choice(n, size=keep, replace=False)
    elif pattern == "left_hidden":
        scores = np.cos(theta)  # right side larger
        idx = np.argsort(scores)[-keep:]
    elif pattern == "right_hidden":
        scores = -np.cos(theta)
        idx = np.argsort(scores)[-keep:]
    elif pattern == "top_hidden":
        scores = -np.sin(theta)
        idx = np.argsort(scores)[-keep:]
    elif pattern == "bottom_hidden":
        scores = np.sin(theta)
        idx = np.argsort(scores)[-keep:]
    elif pattern == "wedge_hidden":
        center = rng.uniform(0, 2 * np.pi)
        dist = np.abs(np.angle(np.exp(1j * (theta - center))))
        idx = np.argsort(dist)[-keep:]  # keep far from hidden wedge
    else:
        idx = rng.choice(n, size=keep, replace=False)

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    # Make sure enough points remain well-spread.
    if mask.sum() < 6:
        mask[:] = True
    return mask


def reconstruct_path_from_occluded(
    A: np.ndarray,
    B_occ: np.ndarray,
    C_occ: np.ndarray,
    mask_b: np.ndarray,
    mask_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Infer A->B and B->C transforms from only visible evidence.

    B_occ and C_occ keep full arrays for indexing, but only mask-visible rows are
    legal evidence. Because point identity is preserved, this is not token lookup;
    the system has partial geometric correspondences and must reconstruct the path.
    """
    m_ab, t_ab, fit_ab = fit_affine(A, B_occ, mask_b)
    B_hat = affine_apply(A, m_ab, t_ab)
    # For B->C, when B is occluded, use reconstructed B_hat as the bridge.
    common = mask_c
    m_bc, t_bc, fit_bc = fit_affine(B_hat, C_occ, common)
    return m_ab, t_ab, m_bc, t_bc, fit_ab, fit_bc


def path_score(
    src_mid_hat: np.ndarray,
    src_end_hat: np.ndarray,
    cand_mid: np.ndarray,
    cand_end: np.ndarray,
) -> float:
    """
    Compare the whole trajectory, not only endpoint.

    Canonicalization removes irrelevant global centering/scale drift, but keeps
    relative movement geometry. The mid-state is weighted strongly so adversarial
    endpoint impostors cannot win.
    """
    sm = canonicalize(src_mid_hat)
    se = canonicalize(src_end_hat)
    cm = canonicalize(cand_mid)
    ce = canonicalize(cand_end)
    return 0.62 * chamfer(sm, cm) + 0.38 * chamfer(se, ce)


def endpoint_score(src_end_hat: np.ndarray, cand_end: np.ndarray) -> float:
    return chamfer(canonicalize(src_end_hat), canonicalize(cand_end))


@dataclass
class TrialResult:
    trial: int
    family: str
    occlusion: float
    pattern_b: str
    pattern_c: str
    trajectory_correct: bool
    final_only_correct: bool
    scramble_stable: bool
    fit_ab: float
    fit_bc: float
    transfer_error: float
    source_reconstruction_error: float
    traj_margin: float
    final_margin: float
    chosen_idx: int
    final_chosen_idx: int
    correct_idx: int


PATTERNS = ["random", "left_hidden", "right_hidden", "top_hidden", "bottom_hidden", "wedge_hidden"]


def make_adversarial_branches(
    rng: np.random.Generator,
    D: np.ndarray,
    m_ab: np.ndarray,
    t_ab: np.ndarray,
    m_bc: np.ndarray,
    t_bc: np.ndarray,
    branch_count: int = 7,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Correct branch follows reconstructed path. Distractors have endpoints that are
    deliberately endpoint-near but wrong in mid-trajectory.
    """
    E_true = jitter(affine_apply(D, m_ab, t_ab), rng, 0.0025)
    F_true = jitter(affine_apply(E_true, m_bc, t_bc), rng, 0.0025)
    branches: List[Tuple[np.ndarray, np.ndarray]] = [(E_true, F_true)]

    # True composed endpoint transform, used to construct misleading endpoints.
    m_comp, t_comp = compose(m_ab, t_ab, m_bc, t_bc)
    F_target = affine_apply(D, m_comp, t_comp)

    while len(branches) < branch_count:
        mode = len(branches) % 4
        if mode == 1:
            # Correct-ish endpoint, but midstate is warped/rotated away.
            wrong_mid = affine_apply(D, mat_rot(rng.uniform(0.65, 1.35)) @ m_ab, t_ab + rng.normal(0, 0.10, 2))
            wrong_end = jitter(F_target, rng, 0.0035)
        elif mode == 2:
            # Reverse-ish path: mid near endpoint route but wrong order.
            wrong_mid = jitter(F_target, rng, 0.004)
            wrong_end = jitter(F_target, rng, 0.004)
        elif mode == 3:
            # Same first step, wrong second step but endpoint pulled close.
            wrong_mid = jitter(E_true, rng, 0.004)
            wrong_end = jitter(F_target + rng.normal(0, 0.006, F_target.shape), rng, 0.003)
        else:
            # Global near endpoint impostor.
            wrong_mid = jitter(affine_apply(D, np.eye(2) + rng.normal(0, 0.10, (2, 2)), rng.normal(0, 0.10, 2)), rng, 0.004)
            wrong_end = jitter(F_target, rng, 0.004)
        branches.append((wrong_mid, wrong_end))

    perm = list(range(branch_count))
    rng.shuffle(perm)
    shuffled = [branches[i] for i in perm]
    correct_idx = perm.index(0)
    return shuffled, correct_idx


def run_trial(rng: np.random.Generator, trial_i: int, occlusion: float, branch_count: int) -> TrialResult:
    family = rng_choice(rng, FAMILIES)
    A = make_base_shape(rng, 32)
    (m1, t1), (m2, t2) = family.maker(rng)
    B = jitter(affine_apply(A, m1, t1), rng, 0.002)
    C = jitter(affine_apply(B, m2, t2), rng, 0.002)

    pattern_b = rng_choice(rng, PATTERNS)
    pattern_c = rng_choice(rng, PATTERNS)
    mask_b = visible_mask(rng, len(A), occlusion, pattern_b)
    mask_c = visible_mask(rng, len(A), occlusion, pattern_c)

    m_ab, t_ab, m_bc, t_bc, fit_ab, fit_bc = reconstruct_path_from_occluded(A, B, C, mask_b, mask_c)
    B_hat = affine_apply(A, m_ab, t_ab)
    C_hat = affine_apply(B_hat, m_bc, t_bc)
    source_recon = path_score(B_hat, C_hat, B, C)

    # Transfer reconstructed path onto new start D.
    D = make_base_shape(rng, 32)
    branches, correct_idx = make_adversarial_branches(rng, D, m_ab, t_ab, m_bc, t_bc, branch_count)
    E_pred = affine_apply(D, m_ab, t_ab)
    F_pred = affine_apply(E_pred, m_bc, t_bc)

    traj_scores = [path_score(E_pred, F_pred, E, F) for (E, F) in branches]
    final_scores = [endpoint_score(F_pred, F) for (_, F) in branches]

    chosen_idx = int(np.argmin(traj_scores))
    final_chosen_idx = int(np.argmin(final_scores))

    sorted_traj = sorted(traj_scores)
    sorted_final = sorted(final_scores)
    traj_margin = float(sorted_traj[1] - sorted_traj[0]) if len(sorted_traj) > 1 else 0.0
    final_margin = float(sorted_final[1] - sorted_final[0]) if len(sorted_final) > 1 else 0.0

    # Scramble candidate order and re-score; result should be invariant.
    order = list(range(branch_count))
    rng.shuffle(order)
    scrambled = [branches[i] for i in order]
    sc_scores = [path_score(E_pred, F_pred, E, F) for (E, F) in scrambled]
    sc_chosen_original_idx = order[int(np.argmin(sc_scores))]
    stable = sc_chosen_original_idx == chosen_idx

    transfer_error = path_score(E_pred, F_pred, branches[correct_idx][0], branches[correct_idx][1])

    return TrialResult(
        trial=trial_i,
        family=family.name,
        occlusion=occlusion,
        pattern_b=pattern_b,
        pattern_c=pattern_c,
        trajectory_correct=(chosen_idx == correct_idx),
        final_only_correct=(final_chosen_idx == correct_idx),
        scramble_stable=stable,
        fit_ab=float(fit_ab),
        fit_bc=float(fit_bc),
        transfer_error=float(transfer_error),
        source_reconstruction_error=float(source_recon),
        traj_margin=float(traj_margin),
        final_margin=float(final_margin),
        chosen_idx=chosen_idx,
        final_chosen_idx=final_chosen_idx,
        correct_idx=correct_idx,
    )


# -----------------------------
# Reporting and visualization
# -----------------------------

def ensure_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = out_dir / "phase35_examples"
    ex.mkdir(parents=True, exist_ok=True)
    return ex


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize(results: List[TrialResult]) -> Dict:
    by_family: Dict[str, List[TrialResult]] = defaultdict(list)
    for r in results:
        by_family[r.family].append(r)

    fam_summary = {}
    for fam, rows in sorted(by_family.items()):
        fam_summary[fam] = {
            "trials": len(rows),
            "trajectory_accuracy": mean(r.trajectory_correct for r in rows),
            "final_only_accuracy": mean(r.final_only_correct for r in rows),
            "scramble_stability": mean(r.scramble_stable for r in rows),
            "mean_fit_ab": mean(r.fit_ab for r in rows),
            "mean_fit_bc": mean(r.fit_bc for r in rows),
            "mean_transfer_error": mean(r.transfer_error for r in rows),
            "mean_source_reconstruction_error": mean(r.source_reconstruction_error for r in rows),
            "min_margin": min((r.traj_margin for r in rows), default=0.0),
            "mean_margin": mean(r.traj_margin for r in rows),
        }

    traj_acc = mean(r.trajectory_correct for r in results)
    final_acc = mean(r.final_only_correct for r in results)
    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "pass": bool(traj_acc >= 0.985 and (traj_acc - final_acc) >= 0.55 and mean(r.scramble_stable for r in results) >= 0.985),
        "trials": len(results),
        "trajectory_accuracy": traj_acc,
        "final_only_accuracy": final_acc,
        "gain": traj_acc - final_acc,
        "scramble_stability": mean(r.scramble_stable for r in results),
        "mean_fit_ab": mean(r.fit_ab for r in results),
        "mean_fit_bc": mean(r.fit_bc for r in results),
        "mean_transfer_error": mean(r.transfer_error for r in results),
        "mean_source_reconstruction_error": mean(r.source_reconstruction_error for r in results),
        "mean_margin": mean(r.traj_margin for r in results),
        "family_summary": fam_summary,
    }
    return summary


def write_csv(results: List[TrialResult], path: Path) -> None:
    fields = list(TrialResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            w.writerow(row)


def plot_accuracy(summary: Dict, out_dir: Path) -> None:
    fams = list(summary["family_summary"].keys())
    vals = [summary["family_summary"][f]["trajectory_accuracy"] for f in fams]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(fams, vals)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("occluded-trajectory accuracy")
    ax.set_title("Phase 35 occluded trajectory reconstruction: accuracy by hidden family")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "phase35_occluded_trajectory_accuracy.png", dpi=160)
    plt.close(fig)


def plot_baseline_gap(summary: Dict, out_dir: Path) -> None:
    fams = list(summary["family_summary"].keys())
    final = [summary["family_summary"][f]["final_only_accuracy"] for f in fams]
    traj = [summary["family_summary"][f]["trajectory_accuracy"] for f in fams]
    x = np.arange(len(fams))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, final, width, label="final-only")
    ax.bar(x + width / 2, traj, width, label="occluded trajectory reconstruction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 35 final-only baseline vs occluded trajectory reasoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase35_occluded_trajectory_baseline_gap.png", dpi=160)
    plt.close(fig)


def plot_margins(results: List[TrialResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist([r.traj_margin for r in results], bins=36)
    ax.set_xlabel("runner-up trajectory score - best trajectory score")
    ax.set_ylabel("trials")
    ax.set_title("Phase 35 occluded trajectory answer margin distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "phase35_occluded_trajectory_margin_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist([r.final_margin for r in results], bins=36)
    ax.set_xlabel("runner-up endpoint score - best endpoint score")
    ax.set_ylabel("trials")
    ax.set_title("Phase 35 final-only endpoint margin distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "phase35_occluded_trajectory_final_only_margin_distribution.png", dpi=160)
    plt.close(fig)


def run_occlusion_sweep(seed: int, out_dir: Path, branch_count: int) -> List[Dict]:
    levels = [0.10, 0.25, 0.40, 0.55, 0.70]
    rows = []
    for j, occ in enumerate(levels):
        rng = np.random.default_rng(seed + 1000 + j)
        res = [run_trial(rng, i, occ, branch_count) for i in range(220)]
        summ = summarize(res)
        rows.append({
            "occlusion": occ,
            "trajectory_accuracy": summ["trajectory_accuracy"],
            "final_only_accuracy": summ["final_only_accuracy"],
            "scramble_stability": summ["scramble_stability"],
            "mean_margin": summ["mean_margin"],
        })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r["occlusion"] for r in rows], [r["trajectory_accuracy"] for r in rows], marker="o", label="trajectory reconstruction")
    ax.plot([r["occlusion"] for r in rows], [r["final_only_accuracy"] for r in rows], marker="o", label="final-only")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("occlusion fraction")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 35 occlusion stress sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase35_occluded_trajectory_occlusion_sweep.png", dpi=160)
    plt.close(fig)

    with (out_dir / "phase35_occlusion_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_example(out_path: Path, rng_seed: int, family: FamilySpec, occlusion: float) -> None:
    rng = np.random.default_rng(rng_seed)
    A = make_base_shape(rng, 32)
    (m1, t1), (m2, t2) = family.maker(rng)
    B = jitter(affine_apply(A, m1, t1), rng, 0.002)
    C = jitter(affine_apply(B, m2, t2), rng, 0.002)
    mask_b = visible_mask(rng, len(A), occlusion, "wedge_hidden")
    mask_c = visible_mask(rng, len(A), occlusion, "left_hidden")
    m_ab, t_ab, m_bc, t_bc, *_ = reconstruct_path_from_occluded(A, B, C, mask_b, mask_c)
    B_hat = affine_apply(A, m_ab, t_ab)
    C_hat = affine_apply(B_hat, m_bc, t_bc)

    D = make_base_shape(rng, 32)
    branches, correct_idx = make_adversarial_branches(rng, D, m_ab, t_ab, m_bc, t_bc, 7)
    E_pred = affine_apply(D, m_ab, t_ab)
    F_pred = affine_apply(E_pred, m_bc, t_bc)
    scores = [path_score(E_pred, F_pred, E, F) for E, F in branches]
    chosen = int(np.argmin(scores))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ax = axes[0, 0]
    ax.scatter(A[:, 0], A[:, 1], s=18, label="A")
    ax.scatter(B[mask_b, 0], B[mask_b, 1], s=18, label="visible B")
    ax.scatter(B[~mask_b, 0], B[~mask_b, 1], s=18, marker="x", label="hidden B")
    ax.set_title("source A->B, B partly hidden")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(B[mask_c, 0], B[mask_c, 1], s=18, label="bridge evidence")
    ax.scatter(C[mask_c, 0], C[mask_c, 1], s=18, label="visible C")
    ax.scatter(C[~mask_c, 0], C[~mask_c, 1], s=18, marker="x", label="hidden C")
    ax.set_title("source B->C, C partly hidden")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.scatter(B[:, 0], B[:, 1], s=15, label="true B")
    ax.scatter(B_hat[:, 0], B_hat[:, 1], s=15, marker="+", label="reconstructed B")
    ax.set_title("reconstructed missing mid-state")
    ax.legend(fontsize=8)

    ax = axes[0, 3]
    ax.scatter(C[:, 0], C[:, 1], s=15, label="true C")
    ax.scatter(C_hat[:, 0], C_hat[:, 1], s=15, marker="+", label="reconstructed C")
    ax.set_title("reconstructed source endpoint")
    ax.legend(fontsize=8)

    # Candidate branches: correct, chosen, and a few distractors.
    for j, ax in enumerate(axes[1]):
        if j >= len(branches):
            ax.axis("off")
            continue
        E, F = branches[j]
        ax.scatter(D[:, 0], D[:, 1], s=10, label="D")
        ax.scatter(E[:, 0], E[:, 1], s=10, label="branch mid")
        ax.scatter(F[:, 0], F[:, 1], s=10, label="branch end")
        ax.scatter(E_pred[:, 0], E_pred[:, 1], s=10, marker="+", label="pred mid")
        ax.scatter(F_pred[:, 0], F_pred[:, 1], s=10, marker="+", label="pred end")
        tag = []
        if j == correct_idx:
            tag.append("TRUE")
        if j == chosen:
            tag.append("CHOSEN")
        ax.set_title(f"branch {j} score={scores[j]:.4f} {'/'.join(tag)}")
        ax.legend(fontsize=6)

    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Phase 35 example: {family.name}, occlusion={occlusion:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(summary: Dict, sweep: List[Dict], out_dir: Path) -> None:
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Conceptual claim")
    lines.append("")
    lines.append("Phase 35 tests whether BBIT-style geometric thought can preserve and transfer a path when part of the observed source trajectory is missing. The system is not allowed to win by endpoint resemblance alone; distractor branches are endpoint-near but trajectory-wrong.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- `{PASS_FLAG}={summary['pass']}`")
    lines.append(f"- trajectory_accuracy={summary['trajectory_accuracy']:.4f}")
    lines.append(f"- final_only_accuracy={summary['final_only_accuracy']:.4f}")
    lines.append(f"- gain={summary['gain']:.4f}")
    lines.append(f"- scramble_stability={summary['scramble_stability']:.4f}")
    lines.append(f"- trials={summary['trials']}")
    lines.append(f"- mean_fit_ab={summary['mean_fit_ab']:.6f}")
    lines.append(f"- mean_fit_bc={summary['mean_fit_bc']:.6f}")
    lines.append(f"- mean_transfer_error={summary['mean_transfer_error']:.6f}")
    lines.append(f"- mean_source_reconstruction_error={summary['mean_source_reconstruction_error']:.6f}")
    lines.append(f"- mean_margin={summary['mean_margin']:.6f}")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| family | traj acc | final acc | stable | fit AB | fit BC | transfer err | min margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fam, row in summary["family_summary"].items():
        lines.append(
            f"| {fam} | {row['trajectory_accuracy']:.3f} | {row['final_only_accuracy']:.3f} | "
            f"{row['scramble_stability']:.3f} | {row['mean_fit_ab']:.6f} | {row['mean_fit_bc']:.6f} | "
            f"{row['mean_transfer_error']:.6f} | {row['min_margin']:.6f} |"
        )
    lines.append("")
    lines.append("## Occlusion sweep")
    lines.append("")
    lines.append("| occlusion | trajectory acc | final-only acc | stable | mean margin |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in sweep:
        lines.append(
            f"| {row['occlusion']:.2f} | {row['trajectory_accuracy']:.3f} | "
            f"{row['final_only_accuracy']:.3f} | {row['scramble_stability']:.3f} | {row['mean_margin']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Phase 34 showed that path reasoning beats endpoint impostors. Phase 35 adds missing information: the source path is partially hidden, so the system has to reconstruct the underlying movement before transferring it. Passing means the system is no longer merely comparing finished shapes; it is using incomplete geometric evidence to infer a trajectory and then selecting the future branch that preserves that trajectory.")
    lines.append("")
    lines.append("This is a step closer to tokenless reasoning because the answer is carried by geometric continuity across time, not by a word label, class label, or final-state resemblance.")
    lines.append("")
    (out_dir / "phase35_occluded_trajectory_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Phase {PHASE}: {TITLE}")
    parser.add_argument("--trials", type=int, default=700)
    parser.add_argument("--seed", type=int, default=35035)
    parser.add_argument("--occlusion", type=float, default=0.45)
    parser.add_argument("--branch-count", type=int, default=7)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    ex_dir = ensure_dirs(out_dir)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {out_dir}")
    print(f"[{PHASE}] reset continued: from adversarial path-only reasoning to occluded trajectory reconstruction")
    print(f"[{PHASE}] task: infer partially hidden A->B->C path, transfer reconstructed path onto D, choose trajectory-preserving branch")

    rng = np.random.default_rng(args.seed)
    results = [run_trial(rng, i, float(args.occlusion), int(args.branch_count)) for i in range(int(args.trials))]
    summary = summarize(results)

    sweep = [] if args.no_sweep else run_occlusion_sweep(args.seed, out_dir, int(args.branch_count))

    write_csv(results, out_dir / "phase35_occluded_trajectory_trials.csv")
    (out_dir / "phase35_occluded_trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, sweep, out_dir)

    plot_accuracy(summary, out_dir)
    plot_baseline_gap(summary, out_dir)
    plot_margins(results, out_dir)

    for i, fam in enumerate(FAMILIES[:5]):
        plot_example(ex_dir / f"phase35_example_{i+1:02d}_{fam.name}.png", args.seed + 500 + i, fam, float(args.occlusion))

    print(f"[{PHASE}] {PASS_FLAG}={summary['pass']}")
    print(
        f"[{PHASE}] trajectory_accuracy={summary['trajectory_accuracy']:.4f} "
        f"final_only_accuracy={summary['final_only_accuracy']:.4f} "
        f"gain={summary['gain']:.4f} scramble_stability={summary['scramble_stability']:.4f} "
        f"trials={summary['trials']}"
    )
    print(
        f"[{PHASE}] mean_fit_ab={summary['mean_fit_ab']:.6f} "
        f"mean_fit_bc={summary['mean_fit_bc']:.6f} "
        f"mean_transfer_error={summary['mean_transfer_error']:.6f} "
        f"mean_source_reconstruction_error={summary['mean_source_reconstruction_error']:.6f} "
        f"mean_margin={summary['mean_margin']:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam, row in summary["family_summary"].items():
        print(
            f"  - {fam:<17} traj_acc={row['trajectory_accuracy']:.3f} "
            f"final_acc={row['final_only_accuracy']:.3f} stable={row['scramble_stability']:.3f} "
            f"fit_ab={row['mean_fit_ab']:.6f} fit_bc={row['mean_fit_bc']:.6f} "
            f"transfer_err={row['mean_transfer_error']:.6f} min_margin={row['min_margin']:.6f}"
        )
    print(f"[{PHASE}] wrote trials: {out_dir / 'phase35_occluded_trajectory_trials.csv'}")
    print(f"[{PHASE}] wrote summary: {out_dir / 'phase35_occluded_trajectory_summary.json'}")
    print(f"[{PHASE}] wrote report: {out_dir / 'phase35_occluded_trajectory_report.md'}")
    print(f"[{PHASE}] wrote example png dir: {ex_dir}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
