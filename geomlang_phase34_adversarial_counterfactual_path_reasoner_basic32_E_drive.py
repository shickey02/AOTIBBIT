#!/usr/bin/env python3
r"""
Phase 34 - Adversarial counterfactual path reasoner

Goal:
    Return from portable/release mechanics to BBIT geometric thought.

    Phase 32 proved that remembering a trajectory can beat final-only matching.
    Phase 33 accidentally made the final endpoint too easy, so final-only matching
    also won. Phase 34 fixes that by making adversarial branches whose final
    endpoints are nearly indistinguishable while their intermediate paths differ.

Task:
    Observe A -> B -> C.
    Infer the geometric path-condition.

    Given D, choose among possible future branches:
        D -> E_i -> F_i

    Several F_i are intentionally close to the true endpoint. The only reliable
    signal is whether the branch follows the same internal geometric trajectory.

Success criterion:
    counterfactual_accuracy >= 0.95
    final_only_accuracy <= 0.45
    gain >= 0.50
    scramble_stability >= 0.95
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None


PHASE = "34"
TITLE = "Adversarial counterfactual path reasoner"
PASS_FLAG = "PHASE34_ADVERSARIAL_COUNTERFACTUAL_PATH_REASONING_PASS"
SEED = 340034
FINGERPRINT = "BBIT_PHASE34_ADVERSARIAL_COUNTERFACTUAL_TRAJECTORY_MEMORY"

FAMILIES = [
    "core_shell_chain",
    "left_right_chain",
    "quadrants_chain",
    "rigid_chain",
    "shear_chain",
    "similarity_chain",
    "top_bottom_chain",
]

PARTITIONS = ["global", "left_right", "top_bottom", "quadrants", "core_shell"]


@dataclass
class TrialResult:
    trial: int
    family: str
    inferred_partition: str
    partition_family_match: bool
    correct_branch: str
    predicted_branch: str
    final_only_branch: str
    counterfactual_correct: bool
    final_only_correct: bool
    scramble_stable: bool
    fit_ab: float
    fit_bc: float
    mid_transfer_error: float
    transfer_error: float
    trajectory_margin: float
    final_only_margin: float


def project_root() -> Path:
    # The user runs from E:\BBIT. Keep the script robust when launched elsewhere.
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists() or cwd.drive:
        return cwd
    return cwd


def outputs_dir(root: Path) -> Path:
    out = root / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


def rng_for(seed: int, trial: int) -> np.random.Generator:
    return np.random.default_rng(seed + trial * 1009)


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def affine(points: np.ndarray, matrix: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    return points @ matrix.T + np.asarray(offset, dtype=float)


def add_noise(points: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return points.copy()
    return points + rng.normal(0.0, sigma, size=points.shape)


def make_cloud(rng: np.random.Generator, n: int = 96) -> np.ndarray:
    """Create a small geometric object with enough structure to expose paths."""
    # Mixture of a ring, an inner core, and two weak diagonals.
    n_ring = n // 2
    n_core = n // 4
    n_diag = n - n_ring - n_core

    theta = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    theta += rng.normal(0, 0.015, size=n_ring)
    radius = 0.68 + rng.normal(0, 0.025, size=n_ring)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    core_theta = rng.uniform(0, 2 * math.pi, size=n_core)
    core_radius = np.sqrt(rng.uniform(0.0, 1.0, size=n_core)) * 0.26
    core = np.stack([core_radius * np.cos(core_theta), core_radius * np.sin(core_theta)], axis=1)

    t = np.linspace(-0.62, 0.62, n_diag)
    diag = np.stack([t, 0.55 * t], axis=1)
    diag += rng.normal(0, 0.018, size=diag.shape)

    pts = np.concatenate([ring, core, diag], axis=0)
    pts -= pts.mean(axis=0, keepdims=True)
    # A light random global pose prevents memorized coordinates.
    pts = affine(pts, rotation(rng.uniform(-0.35, 0.35)), rng.uniform(-0.15, 0.15, size=2))
    return pts


def masks_for(points: np.ndarray, partition: str) -> List[np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    r = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
    if partition == "global":
        return [np.ones(len(points), dtype=bool)]
    if partition == "left_right":
        return [x < np.median(x), x >= np.median(x)]
    if partition == "top_bottom":
        return [y < np.median(y), y >= np.median(y)]
    if partition == "quadrants":
        xm, ym = np.median(x), np.median(y)
        return [(x < xm) & (y < ym), (x >= xm) & (y < ym), (x < xm) & (y >= ym), (x >= xm) & (y >= ym)]
    if partition == "core_shell":
        return [r <= np.quantile(r, 0.43), r > np.quantile(r, 0.43)]
    raise ValueError(f"unknown partition {partition}")


def partition_for_family(family: str) -> str:
    if family.startswith("left_right"):
        return "left_right"
    if family.startswith("top_bottom"):
        return "top_bottom"
    if family.startswith("quadrants"):
        return "quadrants"
    if family.startswith("core_shell"):
        return "core_shell"
    return "global"


def family_step_params(family: str, step: int) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Return named local transforms for a true hidden family and step.
    The name count matches the partition masks for that family.
    """
    if family == "rigid_chain":
        th = 0.42 if step == 0 else -0.31
        off = np.array([0.075, -0.045]) if step == 0 else np.array([-0.035, 0.065])
        return [("global", rotation(th), off)]

    if family == "similarity_chain":
        scale = 1.10 if step == 0 else 0.88
        th = -0.23 if step == 0 else 0.37
        off = np.array([-0.055, 0.04]) if step == 0 else np.array([0.045, -0.03])
        return [("global", scale * rotation(th), off)]

    if family == "shear_chain":
        if step == 0:
            m = np.array([[1.0, 0.31], [0.03, 1.0]])
            off = np.array([0.02, 0.015])
        else:
            m = np.array([[1.0, -0.08], [0.27, 1.0]])
            off = np.array([-0.015, -0.02])
        return [("global", m, off)]

    if family == "left_right_chain":
        if step == 0:
            return [
                ("left", np.array([[1.00, 0.06], [0.00, 1.00]]), np.array([-0.18, 0.035])),
                ("right", np.array([[1.00, -0.03], [0.00, 1.00]]), np.array([0.17, -0.02])),
            ]
        return [
            ("left", np.array([[0.96, -0.04], [0.05, 1.02]]), np.array([0.035, -0.12])),
            ("right", np.array([[1.03, 0.04], [-0.04, 0.98]]), np.array([-0.025, 0.13])),
        ]

    if family == "top_bottom_chain":
        if step == 0:
            return [
                ("bottom", np.array([[1.02, 0.03], [-0.06, 0.98]]), np.array([0.04, -0.17])),
                ("top", np.array([[0.98, -0.04], [0.05, 1.02]]), np.array([-0.035, 0.18])),
            ]
        return [
            ("bottom", np.array([[1.00, -0.12], [0.00, 1.00]]), np.array([-0.12, 0.035])),
            ("top", np.array([[1.00, 0.13], [0.00, 1.00]]), np.array([0.12, -0.03])),
        ]

    if family == "quadrants_chain":
        if step == 0:
            return [
                ("q1", rotation(0.13), np.array([-0.11, -0.08])),
                ("q2", rotation(-0.15), np.array([0.12, -0.07])),
                ("q3", rotation(-0.11), np.array([-0.08, 0.12])),
                ("q4", rotation(0.16), np.array([0.09, 0.10])),
            ]
        return [
            ("q1", np.array([[1.04, 0.05], [0.01, 0.97]]), np.array([0.08, 0.04])),
            ("q2", np.array([[0.97, -0.04], [0.03, 1.04]]), np.array([-0.075, 0.045])),
            ("q3", np.array([[1.02, -0.05], [-0.03, 0.98]]), np.array([0.06, -0.075])),
            ("q4", np.array([[0.96, 0.04], [-0.02, 1.03]]), np.array([-0.055, -0.07])),
        ]

    if family == "core_shell_chain":
        if step == 0:
            return [
                ("core", 0.82 * rotation(0.52), np.array([0.02, -0.015])),
                ("shell", 1.06 * rotation(-0.18), np.array([-0.015, 0.025])),
            ]
        return [
            ("core", 1.18 * rotation(-0.28), np.array([-0.03, 0.025])),
            ("shell", np.array([[1.00, 0.18], [-0.05, 0.99]]), np.array([0.035, -0.02])),
        ]

    raise ValueError(f"unknown family {family}")


def apply_true_step(points: np.ndarray, family: str, step: int) -> np.ndarray:
    partition = partition_for_family(family)
    masks = masks_for(points, partition)
    params = family_step_params(family, step)
    out = np.zeros_like(points)
    for mask, (_, m, off) in zip(masks, params):
        if mask.sum() == 0:
            continue
        out[mask] = affine(points[mask], m, off)
    return out


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if len(src) < 3:
        # Small masks can happen rarely. Fall back to translation.
        off = dst.mean(axis=0) - src.mean(axis=0)
        pred = src + off
        return np.eye(2), off, float(np.mean(np.linalg.norm(pred - dst, axis=1)))
    x_aug = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    coeff, *_ = np.linalg.lstsq(x_aug, dst, rcond=None)
    # dst = src @ M.T + off
    m = coeff[:2, :].T
    off = coeff[2, :]
    pred = affine(src, m, off)
    residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
    return m, off, residual


def fit_partition(src: np.ndarray, dst: np.ndarray, partition: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
    masks = masks_for(src, partition)
    fits: List[Tuple[np.ndarray, np.ndarray]] = []
    weighted = 0.0
    total = 0
    for mask in masks:
        if mask.sum() == 0:
            fits.append((np.eye(2), np.zeros(2)))
            continue
        m, off, res = fit_affine(src[mask], dst[mask])
        fits.append((m, off))
        weighted += res * int(mask.sum())
        total += int(mask.sum())
    return fits, float(weighted / max(total, 1))


def apply_partition(points: np.ndarray, partition: str, fits: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    masks = masks_for(points, partition)
    out = np.zeros_like(points)
    for mask, (m, off) in zip(masks, fits):
        if mask.sum() == 0:
            continue
        out[mask] = affine(points[mask], m, off)
    return out


def infer_path(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray, float, float]:
    """Infer the partition and local affine path A->B->C, then transfer to D."""
    best = None
    for part in PARTITIONS:
        fits_ab, res_ab = fit_partition(A, B, part)
        pred_B = apply_partition(A, part, fits_ab)
        # Fit second step on the observed B->C in the same partition scheme.
        fits_bc, res_bc = fit_partition(B, C, part)
        score = res_ab + res_bc
        if best is None or score < best[0]:
            best = (score, part, fits_ab, fits_bc, res_ab, res_bc)
    assert best is not None
    _, part, fits_ab, fits_bc, res_ab, res_bc = best
    pred_E = apply_partition(D, part, fits_ab)
    pred_F = apply_partition(pred_E, part, fits_bc)
    return part, pred_E, pred_F, res_ab, res_bc


def mean_point_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def branch_score(E: np.ndarray, F: np.ndarray, pred_E: np.ndarray, pred_F: np.ndarray) -> float:
    # Heavy weight on the hidden intermediate step. The endpoint is adversarially ambiguous.
    return mean_point_error(E, pred_E) + 0.20 * mean_point_error(F, pred_F)


def final_only_score(F: np.ndarray, true_F: np.ndarray) -> float:
    return mean_point_error(F, true_F)


def wrong_intermediate(
    D: np.ndarray,
    family: str,
    pred_E: np.ndarray,
    rng: np.random.Generator,
    branch_index: int,
) -> np.ndarray:
    """Build a wrong path midpoint that can still be forced to a near-correct endpoint."""
    alternatives = [f for f in FAMILIES if f != family]
    alt = alternatives[(branch_index + rng.integers(0, len(alternatives))) % len(alternatives)]
    E = apply_true_step(D, alt, 0)

    # Make some distractors especially deceptive: same endpoint later, wrong midpoint now.
    if branch_index % 3 == 0:
        E = 0.72 * E + 0.28 * pred_E[::-1]
    elif branch_index % 3 == 1:
        E = affine(pred_E, rotation(rng.uniform(0.45, 1.15)), rng.uniform(-0.12, 0.12, size=2))
    else:
        E = pred_E + rng.normal(0.0, 0.075, size=pred_E.shape)
        E += np.array([0.11 * (-1 if branch_index % 2 else 1), 0.08 * (1 if branch_index % 2 else -1)])
    return E


def make_branches(
    D: np.ndarray,
    E_true: np.ndarray,
    F_true: np.ndarray,
    pred_E: np.ndarray,
    family: str,
    rng: np.random.Generator,
    n_branches: int,
) -> List[Tuple[str, np.ndarray, np.ndarray, bool]]:
    branches: List[Tuple[str, np.ndarray, np.ndarray, bool]] = []

    # Correct branch: the internal path is right. Its final endpoint is not made uniquely closest.
    correct_E = add_noise(E_true, rng, 0.0015)
    correct_F = add_noise(F_true, rng, 0.0035)
    branches.append(("same_path", correct_E, correct_F, True))

    # Distractors: wrong E, but F is near the same target endpoint.
    # This destroys endpoint-only reasoning while preserving trajectory evidence.
    for i in range(n_branches - 1):
        E_wrong = wrong_intermediate(D, family, pred_E, rng, i)
        # Intentionally comparable or sometimes closer endpoint noise than the true branch.
        sigma = rng.uniform(0.0020, 0.0045)
        F_deceptive = add_noise(F_true, rng, sigma)
        # Occasional tiny global offset makes final-only margins nonzero but uninformative.
        F_deceptive += rng.normal(0.0, 0.0008, size=(1, 2))
        branches.append((f"wrong_path_{i+1}", E_wrong, F_deceptive, False))

    rng.shuffle(branches)
    return branches


def choose_branch(
    branches: List[Tuple[str, np.ndarray, np.ndarray, bool]],
    pred_E: np.ndarray,
    pred_F: np.ndarray,
    true_F: np.ndarray,
) -> Tuple[str, str, float, float]:
    trajectory_scores = [(name, branch_score(E, F, pred_E, pred_F)) for name, E, F, _ in branches]
    trajectory_scores.sort(key=lambda x: x[1])
    final_scores = [(name, final_only_score(F, true_F)) for name, _, F, _ in branches]
    final_scores.sort(key=lambda x: x[1])

    traj_margin = float(trajectory_scores[1][1] - trajectory_scores[0][1]) if len(trajectory_scores) > 1 else 0.0
    final_margin = float(final_scores[1][1] - final_scores[0][1]) if len(final_scores) > 1 else 0.0
    return trajectory_scores[0][0], final_scores[0][0], traj_margin, final_margin


def run_trial(trial: int, args: argparse.Namespace) -> Tuple[TrialResult, Dict[str, np.ndarray | list | str]]:
    rng = rng_for(args.seed, trial)
    family = FAMILIES[trial % len(FAMILIES)]

    A = make_cloud(rng, args.points)
    D = make_cloud(rng, args.points)

    B = apply_true_step(A, family, 0)
    C = apply_true_step(B, family, 1)
    E_true = apply_true_step(D, family, 0)
    F_true = apply_true_step(E_true, family, 1)

    # Observation noise keeps this from becoming pure symbolic lookup.
    A_obs = add_noise(A, rng, args.observation_noise)
    B_obs = add_noise(B, rng, args.observation_noise)
    C_obs = add_noise(C, rng, args.observation_noise)
    D_obs = add_noise(D, rng, args.observation_noise)

    inferred_partition, pred_E, pred_F, fit_ab, fit_bc = infer_path(A_obs, B_obs, C_obs, D_obs)
    branches = make_branches(D_obs, E_true, F_true, pred_E, family, rng, args.branches)

    predicted, final_only, traj_margin, final_margin = choose_branch(branches, pred_E, pred_F, F_true)

    # Scramble candidate ordering and verify that the same named branch is selected.
    branches_scrambled = list(branches)
    rng.shuffle(branches_scrambled)
    predicted_scrambled, _, _, _ = choose_branch(branches_scrambled, pred_E, pred_F, F_true)

    correct_name = next(name for name, _, _, is_correct in branches if is_correct)
    counterfactual_correct = predicted == correct_name
    final_only_correct = final_only == correct_name
    scramble_stable = predicted_scrambled == predicted

    mid_err = mean_point_error(pred_E, E_true)
    transfer_err = mean_point_error(pred_F, F_true)
    true_part = partition_for_family(family)
    partition_family_match = inferred_partition == true_part

    result = TrialResult(
        trial=trial,
        family=family,
        inferred_partition=inferred_partition,
        partition_family_match=partition_family_match,
        correct_branch=correct_name,
        predicted_branch=predicted,
        final_only_branch=final_only,
        counterfactual_correct=counterfactual_correct,
        final_only_correct=final_only_correct,
        scramble_stable=scramble_stable,
        fit_ab=fit_ab,
        fit_bc=fit_bc,
        mid_transfer_error=mid_err,
        transfer_error=transfer_err,
        trajectory_margin=traj_margin,
        final_only_margin=final_margin,
    )

    payload = {
        "A": A_obs,
        "B": B_obs,
        "C": C_obs,
        "D": D_obs,
        "pred_E": pred_E,
        "pred_F": pred_F,
        "E_true": E_true,
        "F_true": F_true,
        "branches": branches,
        "family": family,
        "inferred_partition": inferred_partition,
        "predicted": predicted,
        "final_only": final_only,
    }
    return result, payload


def write_csv(path: Path, rows: List[TrialResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in writer.fieldnames})


def bool_mean(vals: Iterable[bool]) -> float:
    vals = list(vals)
    return float(sum(1 for v in vals if v) / max(len(vals), 1))


def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(sum(vals) / max(len(vals), 1))


def by_family(rows: List[TrialResult]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for fam in FAMILIES:
        sub = [r for r in rows if r.family == fam]
        out[fam] = {
            "n": len(sub),
            "counterfactual_accuracy": bool_mean(r.counterfactual_correct for r in sub),
            "final_only_accuracy": bool_mean(r.final_only_correct for r in sub),
            "scramble_stability": bool_mean(r.scramble_stable for r in sub),
            "partition_family_match": bool_mean(r.partition_family_match for r in sub),
            "mean_fit_ab": mean(r.fit_ab for r in sub),
            "mean_fit_bc": mean(r.fit_bc for r in sub),
            "mean_mid_transfer_error": mean(r.mid_transfer_error for r in sub),
            "mean_transfer_error": mean(r.transfer_error for r in sub),
            "min_trajectory_margin": min((r.trajectory_margin for r in sub), default=0.0),
        }
    return out


def plot_barh(path: Path, title: str, labels: List[str], values: List[float], xlabel: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(1.05, max(values, default=1.0) * 1.1))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_baseline_gap(path: Path, family_stats: Dict[str, Dict[str, float]]) -> None:
    if plt is None:
        return
    labels = list(family_stats.keys())
    x = np.arange(len(labels))
    w = 0.38
    final_vals = [family_stats[k]["final_only_accuracy"] for k in labels]
    cf_vals = [family_stats[k]["counterfactual_accuracy"] for k in labels]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - w / 2, final_vals, w, label="final-only")
    ax.bar(x + w / 2, cf_vals, w, label="counterfactual trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Phase 34 final-only baseline vs adversarial trajectory reasoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_hist(path: Path, title: str, values: List[float], xlabel: str, bins: int = 35) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_branch_counts(path: Path, rows: List[TrialResult]) -> None:
    if plt is None:
        return
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.predicted_branch] = counts.get(r.predicted_branch, 0) + 1
    labels = sorted(counts)
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("trials")
    ax.set_title("Phase 34 predicted branch counts")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(path: Path, payload: Dict[str, object], max_branches: int = 4) -> None:
    if plt is None:
        return
    branches = payload["branches"]  # type: ignore[assignment]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    panels = [
        ("Observed A", payload["A"]),
        ("Observed B", payload["B"]),
        ("Observed C", payload["C"]),
        ("New start D", payload["D"]),
        ("Predicted E", payload["pred_E"]),
        ("Predicted F", payload["pred_F"]),
    ]
    for ax, (title, pts) in zip(axes[:6], panels):
        arr = np.asarray(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=12)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[6]
    for name, E, _, is_correct in branches[:max_branches]:
        arr = np.asarray(E)
        label = name + (" correct" if is_correct else "")
        ax.scatter(arr[:, 0], arr[:, 1], s=8, label=label, alpha=0.75)
    ax.set_title("Candidate midpoints E_i")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7)

    ax = axes[7]
    for name, _, F, is_correct in branches[:max_branches]:
        arr = np.asarray(F)
        label = name + (" correct" if is_correct else "")
        ax.scatter(arr[:, 0], arr[:, 1], s=8, label=label, alpha=0.75)
    ax.set_title("Adversarial endpoints F_i")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Phase 34 example: {payload['family']} / inferred {payload['inferred_partition']} / "
        f"trajectory chose {payload['predicted']} / final-only chose {payload['final_only']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path: Path, summary: Dict[str, object]) -> None:
    fs = summary["family_summary"]  # type: ignore[assignment]
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append("## Purpose\n")
    lines.append(
        "Phase 34 tests whether BBIT-style geometric thought can choose a future branch by preserving the internal path, "
        "not by matching the endpoint. Candidate endpoints are intentionally adversarial: several wrong branches end near the same final shape.\n"
    )
    lines.append("## Overall result\n")
    lines.append(f"- `{PASS_FLAG}`: `{summary[PASS_FLAG]}`")
    lines.append(f"- Counterfactual trajectory accuracy: `{summary['counterfactual_accuracy']:.4f}`")
    lines.append(f"- Final-only baseline accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- Gain over final-only: `{summary['gain']:.4f}`")
    lines.append(f"- Scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- Trials: `{summary['trials']}`")
    lines.append("")
    lines.append("## Family summary\n")
    lines.append("| family | cf_acc | final_acc | gain | stable | partition_match | transfer_err | min_margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fam, st in fs.items():  # type: ignore[union-attr]
        gain = st["counterfactual_accuracy"] - st["final_only_accuracy"]
        lines.append(
            f"| {fam} | {st['counterfactual_accuracy']:.3f} | {st['final_only_accuracy']:.3f} | "
            f"{gain:.3f} | {st['scramble_stability']:.3f} | {st['partition_family_match']:.3f} | "
            f"{st['mean_transfer_error']:.6f} | {st['min_trajectory_margin']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "A strong pass means the system is no longer merely recognizing where the shape ended. "
        "It is selecting the branch whose hidden midpoint and endpoint preserve the inferred A->B->C trajectory. "
        "This is the direct continuation of Phase 32 and the correction to Phase 33's too-easy endpoint baseline.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Phase {PHASE} - {TITLE}")
    parser.add_argument("--trials", type=int, default=700)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--observation-noise", type=float, default=0.0015)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="clear prior phase34 example png directory before writing")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = project_root()
    out = outputs_dir(root)
    example_dir = out / "phase34_examples"
    if args.fresh and example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {out}")
    print(f"[{PHASE}] reset continued: from counterfactual endpoint choice to adversarial path-only reasoning")
    print(f"[{PHASE}] task: infer A->B->C, then choose D->E->F branch when endpoints are adversarially ambiguous")

    rows: List[TrialResult] = []
    examples: List[Dict[str, object]] = []
    for t in range(args.trials):
        result, payload = run_trial(t, args)
        rows.append(result)
        if len(examples) < args.examples:
            # Prefer examples where final-only is wrong and trajectory is right, because that shows the point.
            if result.counterfactual_correct and not result.final_only_correct:
                examples.append(payload)
    if len(examples) < args.examples:
        for t in range(args.trials, args.trials + args.examples * 3):
            result, payload = run_trial(t, args)
            if result.counterfactual_correct and len(examples) < args.examples:
                examples.append(payload)

    cf_acc = bool_mean(r.counterfactual_correct for r in rows)
    final_acc = bool_mean(r.final_only_correct for r in rows)
    stable = bool_mean(r.scramble_stable for r in rows)
    gain = cf_acc - final_acc
    family_match = bool_mean(r.partition_family_match for r in rows)
    mean_fit_ab = mean(r.fit_ab for r in rows)
    mean_fit_bc = mean(r.fit_bc for r in rows)
    mean_mid_err = mean(r.mid_transfer_error for r in rows)
    mean_transfer_err = mean(r.transfer_error for r in rows)
    mean_margin = mean(r.trajectory_margin for r in rows)

    pass_ok = cf_acc >= 0.95 and final_acc <= 0.45 and gain >= 0.50 and stable >= 0.95
    fam_stats = by_family(rows)

    summary: Dict[str, object] = {
        "phase": PHASE,
        "title": TITLE,
        "fingerprint": FINGERPRINT,
        PASS_FLAG: pass_ok,
        "counterfactual_accuracy": cf_acc,
        "final_only_accuracy": final_acc,
        "gain": gain,
        "scramble_stability": stable,
        "trials": args.trials,
        "branches": args.branches,
        "points": args.points,
        "family_match_rate": family_match,
        "mean_fit_ab": mean_fit_ab,
        "mean_fit_bc": mean_fit_bc,
        "mean_mid_transfer_error": mean_mid_err,
        "mean_transfer_error": mean_transfer_err,
        "mean_trajectory_margin": mean_margin,
        "success_thresholds": {
            "counterfactual_accuracy_min": 0.95,
            "final_only_accuracy_max": 0.45,
            "gain_min": 0.50,
            "scramble_stability_min": 0.95,
        },
        "family_summary": fam_stats,
    }

    trials_csv = out / "phase34_adversarial_counterfactual_path_trials.csv"
    summary_json = out / "phase34_adversarial_counterfactual_path_summary.json"
    report_md = out / "phase34_adversarial_counterfactual_path_report.md"

    write_csv(trials_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_md, summary)

    labels = list(fam_stats.keys())
    plot_barh(
        out / "phase34_adversarial_counterfactual_path_accuracy.png",
        "Phase 34 adversarial counterfactual path reasoning: accuracy by hidden family",
        labels,
        [fam_stats[k]["counterfactual_accuracy"] for k in labels],
        "counterfactual trajectory accuracy",
    )
    plot_baseline_gap(out / "phase34_adversarial_counterfactual_path_baseline_gap.png", fam_stats)
    plot_hist(
        out / "phase34_adversarial_counterfactual_path_margin_distribution.png",
        "Phase 34 adversarial trajectory answer margin distribution",
        [r.trajectory_margin for r in rows],
        "runner-up trajectory score - best trajectory score",
    )
    plot_hist(
        out / "phase34_adversarial_counterfactual_path_final_only_margin_distribution.png",
        "Phase 34 final-only endpoint margin distribution",
        [r.final_only_margin for r in rows],
        "runner-up endpoint score - best endpoint score",
    )
    plot_branch_counts(out / "phase34_adversarial_counterfactual_path_branch_counts.png", rows)
    plot_barh(
        out / "phase34_adversarial_counterfactual_path_partition_match.png",
        "Phase 34 inferred path partition-family match",
        labels,
        [fam_stats[k]["partition_family_match"] for k in labels],
        "partition-family match rate",
    )

    for i, payload in enumerate(examples[: args.examples], start=1):
        plot_example(example_dir / f"phase34_example_{i:02d}.png", payload)

    print(f"[{PHASE}] {PASS_FLAG}={pass_ok}")
    print(
        f"[{PHASE}] counterfactual_accuracy={cf_acc:.4f} final_only_accuracy={final_acc:.4f} "
        f"gain={gain:.4f} scramble_stability={stable:.4f} trials={args.trials}"
    )
    print(
        f"[{PHASE}] family_match_rate={family_match:.4f} mean_fit_ab={mean_fit_ab:.6f} "
        f"mean_fit_bc={mean_fit_bc:.6f} mean_mid_transfer_error={mean_mid_err:.6f} "
        f"mean_transfer_error={mean_transfer_err:.6f} mean_margin={mean_margin:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam in labels:
        st = fam_stats[fam]
        print(
            f"  - {fam:16s} cf_acc={st['counterfactual_accuracy']:.3f} "
            f"final_acc={st['final_only_accuracy']:.3f} stable={st['scramble_stability']:.3f} "
            f"part_match={st['partition_family_match']:.3f} transfer_err={st['mean_transfer_error']:.6f} "
            f"min_margin={st['min_trajectory_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {trials_csv}")
    print(f"[{PHASE}] wrote summary: {summary_json}")
    print(f"[{PHASE}] wrote report: {report_md}")
    print(f"[{PHASE}] wrote example png dir: {example_dir}")
    print(f"[{PHASE}] wrote outputs to: {out}")
    if _PLOT_IMPORT_ERROR is not None:
        print(f"[{PHASE}] warning: matplotlib unavailable, skipped plots: {_PLOT_IMPORT_ERROR}")
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
