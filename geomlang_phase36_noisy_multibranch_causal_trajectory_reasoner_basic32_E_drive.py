#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 36 - Noisy multi-branch causal trajectory disambiguation reasoner

Run from PowerShell:

    python bbit_geomlang/geomlang_phase36_noisy_multibranch_causal_trajectory_reasoner_basic32_E_drive.py

What this phase tests
---------------------
Phase 35 proved that the system can reconstruct an occluded trajectory and choose
the branch preserving that trajectory.

Phase 36 makes the answer set more hostile:

    A -> B -> C is only partially visible and is corrupted by noise.
    D has multiple possible E -> F futures.
    Every wrong future preserves a partial truth:
        - near final endpoint
        - near midpoint
        - near first-step direction
        - near second-step direction
        - near path length
        - near partition identity
    Only one future preserves the whole reconstructed causal path.

The key contrast is:

    final-only endpoint scoring should fail
    trajectory-memory scoring should pass

This is the next step after Phase 35:
not just "can I fill in hidden movement?"
but "can I reject plausible partial truths under noise?"
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# Headless-safe plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "36"
TITLE = "Noisy multi-branch causal trajectory disambiguation reasoner"
SEED = 36036


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_root() -> Path:
    """
    Prefer E:\BBIT when present, otherwise infer from this script location.
    This keeps the file usable both inside your E-drive project and elsewhere.
    """
    e_root = Path(r"E:\BBIT")
    if e_root.exists():
        return e_root
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if p.name.lower() == "bbit" or (p / "bbit_geomlang").exists():
            return p
    return here.parent.parent


ROOT = find_root()
OUT_DIR = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT_DIR / "phase36_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def rng_uniform(rng: np.random.Generator, lo: float, hi: float, size=None):
    return rng.uniform(lo, hi, size=size)


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric squared-distance Chamfer for small 2D point clouds.
    Implemented without scipy so the phase stays dependency-light.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d2 = ((aa[:, None, :] - bb[None, :, :]) ** 2).sum(axis=2)
    return float(d2.min(axis=1).mean() + d2.min(axis=0).mean())


def center_cloud(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def normalize_cloud(x: np.ndarray) -> np.ndarray:
    x = center_cloud(x)
    scale = np.sqrt((x * x).sum(axis=1).mean()) + 1e-9
    return x / scale


def make_base_cloud(rng: np.random.Generator, n: int = 36) -> np.ndarray:
    """
    A mixed discrete/continuous point field:
    - jittered ring
    - internal cross
    - a few asymmetry markers

    The asymmetry is intentional; it prevents rotations/reflections from becoming
    too interchangeable.
    """
    k_ring = n // 2
    angles = np.linspace(0, 2 * np.pi, k_ring, endpoint=False)
    radii = 0.72 + rng.normal(0, 0.035, size=k_ring)
    ring = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)

    k_inner = n - k_ring
    inner = rng.normal(0, 0.22, size=(k_inner, 2))
    # Add a faint directional spine.
    for i in range(min(8, k_inner)):
        inner[i, 0] += (i - 3.5) * 0.055
        inner[i, 1] += math.sin(i) * 0.035

    pts = np.vstack([ring, inner])
    pts += rng.normal(0, 0.012, size=pts.shape)
    return normalize_cloud(pts)


def affine_apply(x: np.ndarray, m: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x @ m.T + t[None, :]


@dataclass(frozen=True)
class PathFamily:
    name: str
    description: str
    step1: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    step2: Callable[[np.ndarray, np.random.Generator], np.ndarray]


def global_affine_step(theta: float, sx: float, sy: float, shear: float, tx: float, ty: float):
    m = rot(theta) @ np.array([[sx, shear], [0.0, sy]], dtype=np.float64)
    t = np.array([tx, ty], dtype=np.float64)

    def f(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return affine_apply(x, m, t)

    return f


def partition_lr_step(left_shift: Tuple[float, float], right_shift: Tuple[float, float], bend: float = 0.0):
    left_shift = np.array(left_shift, dtype=np.float64)
    right_shift = np.array(right_shift, dtype=np.float64)

    def f(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        mask = y[:, 0] < np.median(y[:, 0])
        y[mask] += left_shift
        y[~mask] += right_shift
        y[:, 1] += bend * np.tanh(2.5 * y[:, 0])
        return y

    return f


def partition_tb_step(bot_shift: Tuple[float, float], top_shift: Tuple[float, float], bend: float = 0.0):
    bot_shift = np.array(bot_shift, dtype=np.float64)
    top_shift = np.array(top_shift, dtype=np.float64)

    def f(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        mask = y[:, 1] < np.median(y[:, 1])
        y[mask] += bot_shift
        y[~mask] += top_shift
        y[:, 0] += bend * np.tanh(2.5 * y[:, 1])
        return y

    return f


def partition_quad_step(strength: float, swirl: float):
    def f(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        sx = np.where(y[:, 0] >= np.median(y[:, 0]), 1.0, -1.0)
        sy = np.where(y[:, 1] >= np.median(y[:, 1]), 1.0, -1.0)
        y[:, 0] += strength * sx
        y[:, 1] += strength * sy
        # small rotational local term
        r = np.stack([-y[:, 1], y[:, 0]], axis=1)
        y += swirl * r
        return y
    return f


def core_shell_step(core_shift: Tuple[float, float], shell_scale: float, shell_twist: float):
    core_shift = np.array(core_shift, dtype=np.float64)

    def f(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        r = np.sqrt((center_cloud(y) ** 2).sum(axis=1))
        mask = r < np.median(r)
        y[mask] += core_shift
        c = y[~mask].mean(axis=0, keepdims=True) if np.any(~mask) else np.zeros((1, 2))
        shell = y[~mask] - c
        y[~mask] = c + shell @ (shell_scale * rot(shell_twist)).T
        return y

    return f


FAMILIES: List[PathFamily] = [
    PathFamily(
        "rigid_chain",
        "global rotation/translation followed by another rigid move",
        global_affine_step(0.34, 1.00, 1.00, 0.00, 0.16, -0.07),
        global_affine_step(-0.22, 1.00, 1.00, 0.00, -0.08, 0.12),
    ),
    PathFamily(
        "similarity_chain",
        "scale/rotate/translate followed by inverse-like similarity",
        global_affine_step(0.25, 1.10, 1.10, 0.00, -0.10, 0.08),
        global_affine_step(-0.18, 0.92, 0.92, 0.00, 0.13, -0.05),
    ),
    PathFamily(
        "shear_chain",
        "opposing shear fields with small translations",
        global_affine_step(0.04, 1.00, 1.00, 0.28, 0.08, 0.02),
        global_affine_step(-0.03, 1.00, 1.00, -0.23, -0.05, 0.10),
    ),
    PathFamily(
        "left_right_chain",
        "left and right halves move differently over two steps",
        partition_lr_step((-0.15, 0.05), (0.11, -0.03), bend=0.020),
        partition_lr_step((0.06, -0.12), (-0.09, 0.14), bend=-0.018),
    ),
    PathFamily(
        "top_bottom_chain",
        "top and bottom halves move differently over two steps",
        partition_tb_step((0.09, -0.12), (-0.08, 0.10), bend=-0.015),
        partition_tb_step((-0.12, 0.06), (0.13, -0.05), bend=0.017),
    ),
    PathFamily(
        "quadrants_chain",
        "four quadrant expansion then softened local swirl",
        partition_quad_step(0.075, 0.018),
        partition_quad_step(-0.055, -0.021),
    ),
    PathFamily(
        "core_shell_chain",
        "inner core and outer shell follow different local laws",
        core_shell_step((0.10, -0.07), shell_scale=1.045, shell_twist=0.12),
        core_shell_step((-0.08, 0.09), shell_scale=0.965, shell_twist=-0.10),
    ),
]


# ---------------------------------------------------------------------------
# Observation/noise/reconstruction
# ---------------------------------------------------------------------------

def add_noise(x: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)


def visible_mask(rng: np.random.Generator, n: int, keep_frac: float) -> np.ndarray:
    keep = max(8, int(round(n * keep_frac)))
    idx = rng.choice(n, size=keep, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Least-squares affine map dst ~= src @ M.T + t.
    """
    x = np.asarray(src, dtype=np.float64)
    y = np.asarray(dst, dtype=np.float64)
    a = np.hstack([x, np.ones((len(x), 1))])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    m = beta[:2, :].T
    t = beta[2, :]
    pred = affine_apply(x, m, t)
    residual = float(np.sqrt(((pred - y) ** 2).sum(axis=1).mean()))
    return m, t, residual


def infer_step_vector(src_obs: np.ndarray, dst_obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    A hybrid reconstruction primitive:
    - fit a global affine on visible matched points
    - store residual as source-fit error

    This is intentionally simple and "geometric", not learned.
    The phase difficulty comes from candidate branch disambiguation.
    """
    m, t, residual = fit_affine(src_obs[mask], dst_obs[mask])
    pred = affine_apply(src_obs, m, t)
    # Convert the fitted motion into a per-point displacement field.
    displacement = pred - src_obs
    return displacement, residual


def reconstruct_path(
    a_obs: np.ndarray,
    b_obs: np.ndarray,
    c_obs: np.ndarray,
    mask_ab: np.ndarray,
    mask_bc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    disp_ab, fit_ab = infer_step_vector(a_obs, b_obs, mask_ab)
    disp_bc, fit_bc = infer_step_vector(b_obs, c_obs, mask_bc)
    return disp_ab, disp_bc, fit_ab, fit_bc


def transfer_path(d: np.ndarray, disp_ab: np.ndarray, disp_bc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer a reconstructed source trajectory onto a new start cloud D.
    Since D is shape-compatible with A, apply the recovered per-point fields by index.
    """
    e_pred = d + disp_ab
    f_pred = e_pred + disp_bc
    return e_pred, f_pred


# ---------------------------------------------------------------------------
# Adversarial candidates
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    label: str
    e: np.ndarray
    f: np.ndarray


def permute_like(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x[rng.permutation(len(x))]


def make_adversarial_branches(
    d: np.ndarray,
    e_true: np.ndarray,
    f_true: np.ndarray,
    rng: np.random.Generator,
    candidate_noise: float,
) -> List[Branch]:
    """
    Build wrong candidates that each preserve a partial truth but not the whole path.

    The important trick:
        final endpoint clouds are deliberately close for all branches,
        so endpoint-only choice should hover near chance.
    """
    branches: List[Branch] = []

    # 1. Correct branch.
    branches.append(Branch(
        "full_path",
        add_noise(e_true, rng, candidate_noise),
        add_noise(f_true, rng, candidate_noise),
    ))

    # Shared near-final endpoint for endpoint-adversarial candidates.
    # These are close enough that final-only scoring cannot reliably choose.
    near_f = add_noise(f_true, rng, candidate_noise * 1.3)

    # 2. Endpoint-only: final is near correct, midpoint is wrong.
    e_endpoint_only = d + (near_f - d) * 0.45 + rng.normal(0, 0.020, size=d.shape)
    branches.append(Branch("endpoint_only", e_endpoint_only, near_f))

    # 3. Midpoint-only: midpoint is near correct, final veers off then returns near endpoint.
    e_mid_only = add_noise(e_true, rng, candidate_noise * 1.1)
    f_mid_only = add_noise(f_true, rng, candidate_noise * 1.5)
    # corrupt local point identity while preserving final cloud shape
    f_mid_only = 0.55 * f_mid_only + 0.45 * permute_like(f_mid_only, rng)
    branches.append(Branch("midpoint_only", e_mid_only, f_mid_only))

    # 4. First-vector only: D->E direction is plausible; E->F wrong.
    v1 = e_true - d
    e_v1 = d + v1 + rng.normal(0, candidate_noise * 1.2, size=d.shape)
    v_wrong = permute_like(f_true - e_true, rng)
    f_v1 = e_v1 + v_wrong + rng.normal(0, 0.018, size=d.shape)
    f_v1 = 0.65 * f_v1 + 0.35 * f_true
    branches.append(Branch("first_step_only", e_v1, f_v1))

    # 5. Second-vector only: E->F direction is plausible; D->E wrong.
    v2 = f_true - e_true
    e_wrong = d + permute_like(e_true - d, rng) + rng.normal(0, 0.018, size=d.shape)
    f_v2 = e_wrong + v2 + rng.normal(0, candidate_noise * 1.2, size=d.shape)
    f_v2 = 0.60 * f_v2 + 0.40 * f_true
    branches.append(Branch("second_step_only", e_wrong, f_v2))

    # 6. Path-length only: total displacement magnitude is similar, direction/identity wrong.
    total = f_true - d
    mag = np.linalg.norm(total, axis=1, keepdims=True) + 1e-9
    random_dir = rng.normal(0, 1, size=total.shape)
    random_dir /= np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-9
    f_len = d + random_dir * mag
    f_len = 0.70 * f_len + 0.30 * f_true
    e_len = d + 0.50 * (f_len - d) + rng.normal(0, 0.020, size=d.shape)
    branches.append(Branch("path_length_only", e_len, f_len))

    # 7. Partition-ish: rough spatial regions preserved, but internal identity scrambled.
    e_part = e_true.copy()
    f_part = f_true.copy()
    for arr in (e_part, f_part):
        xmed, ymed = np.median(arr[:, 0]), np.median(arr[:, 1])
        groups = [
            np.where((arr[:, 0] < xmed) & (arr[:, 1] < ymed))[0],
            np.where((arr[:, 0] >= xmed) & (arr[:, 1] < ymed))[0],
            np.where((arr[:, 0] < xmed) & (arr[:, 1] >= ymed))[0],
            np.where((arr[:, 0] >= xmed) & (arr[:, 1] >= ymed))[0],
        ]
        for g in groups:
            if len(g) > 1:
                arr[g] = arr[rng.permutation(g)]
    e_part = add_noise(e_part, rng, 0.020)
    f_part = add_noise(0.75 * f_part + 0.25 * f_true, rng, 0.012)
    branches.append(Branch("partition_only", e_part, f_part))

    rng.shuffle(branches)
    return branches


def trajectory_score(e_pred: np.ndarray, f_pred: np.ndarray, branch: Branch) -> float:
    """
    Full trajectory score:
    - midpoint cloud
    - final cloud
    - first-step field
    - second-step field

    The step-field terms make this resistant to endpoint mimicry.
    """
    e, f = branch.e, branch.f
    s_mid = chamfer(e_pred, e)
    s_fin = chamfer(f_pred, f)
    s_v1 = float(np.sqrt(((e_pred - e) ** 2).sum(axis=1).mean()))
    s_v2 = float(np.sqrt((((f_pred - e_pred) - (f - e)) ** 2).sum(axis=1).mean()))
    return s_mid + s_fin + 0.65 * s_v1 + 0.65 * s_v2


def final_only_score(f_pred: np.ndarray, branch: Branch) -> float:
    return chamfer(f_pred, branch.f)


def choose(scores: List[float]) -> Tuple[int, float]:
    order = np.argsort(scores)
    best = int(order[0])
    if len(order) > 1:
        margin = float(scores[order[1]] - scores[order[0]])
    else:
        margin = 0.0
    return best, margin


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cloud(ax, pts: np.ndarray, title: str, marker: str = "o"):
    ax.scatter(pts[:, 0], pts[:, 1], s=24, marker=marker)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_example(
    path: Path,
    family: str,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e_pred: np.ndarray,
    f_pred: np.ndarray,
    branches: List[Branch],
    chosen_label: str,
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    plot_cloud(axes[0], a, "A source start")
    plot_cloud(axes[1], b, "B noisy/partially observed")
    plot_cloud(axes[2], c, "C noisy/partially observed")
    plot_cloud(axes[3], d, "D new start")

    axes[4].scatter(d[:, 0], d[:, 1], s=16, label="D")
    axes[4].scatter(e_pred[:, 0], e_pred[:, 1], s=16, label="pred E")
    axes[4].scatter(f_pred[:, 0], f_pred[:, 1], s=16, label="pred F")
    for i in range(min(18, len(d))):
        axes[4].plot([d[i, 0], e_pred[i, 0], f_pred[i, 0]], [d[i, 1], e_pred[i, 1], f_pred[i, 1]], alpha=0.28)
    axes[4].set_title("reconstructed transferred path")
    axes[4].set_aspect("equal", adjustable="box")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(fontsize=8)

    # Show three representative branches: true, chosen, and a random wrong.
    full = next((br for br in branches if br.label == "full_path"), branches[0])
    chosen = next((br for br in branches if br.label == chosen_label), branches[0])
    wrongs = [br for br in branches if br.label not in ("full_path", chosen_label)]
    wrong = wrongs[0] if wrongs else full

    for ax, br, title in [
        (axes[5], full, "true full-path branch"),
        (axes[6], chosen, f"chosen: {chosen_label}"),
        (axes[7], wrong, f"plausible distractor: {wrong.label}"),
    ]:
        ax.scatter(d[:, 0], d[:, 1], s=12, label="D")
        ax.scatter(br.e[:, 0], br.e[:, 1], s=12, label="E")
        ax.scatter(br.f[:, 0], br.f[:, 1], s=12, label="F")
        for i in range(min(14, len(d))):
            ax.plot([d[i, 0], br.e[i, 0], br.f[i, 0]], [d[i, 1], br.e[i, 1], br.f[i, 1]], alpha=0.22)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"Phase 36 example - {family}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def barh_metric(path: Path, title: str, labels: List[str], values: List[float], xlabel: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(1.05, max(values) * 1.05 if values else 1.0))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_baseline_plot(path: Path, labels: List[str], traj_vals: List[float], final_vals: List[float]):
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - w / 2, final_vals, width=w, label="final-only")
    ax.bar(x + w / 2, traj_vals, width=w, label="noisy trajectory reasoning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 36 final-only baseline vs noisy trajectory reasoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram(path: Path, title: str, values: List[float], xlabel: str, bins: int = 36):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def branch_counts_plot(path: Path, counts: Dict[str, int]):
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, vals)
    ax.set_title("Phase 36 predicted branch counts")
    ax.set_ylabel("trials")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def noise_sweep_plot(path: Path, rows: List[Dict[str, float]]):
    sigmas = [r["noise_sigma"] for r in rows]
    traj = [r["trajectory_accuracy"] for r in rows]
    final = [r["final_only_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sigmas, traj, marker="o", label="trajectory reasoning")
    ax.plot(sigmas, final, marker="o", label="final-only")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("source observation noise sigma")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 36 noise stress sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trial engine
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial: int
    family: str
    correct: bool
    final_correct: bool
    stable: bool
    chosen_branch: str
    final_chosen_branch: str
    margin: float
    final_margin: float
    fit_ab: float
    fit_bc: float
    transfer_error: float
    source_reconstruction_error: float
    keep_frac: float
    noise_sigma: float


def run_one_trial(
    trial_id: int,
    family: PathFamily,
    rng: np.random.Generator,
    noise_sigma: float = 0.018,
    keep_frac: float = 0.62,
    candidate_noise: float = 0.006,
    write_example: bool = False,
) -> TrialResult:
    a = make_base_cloud(rng, n=40)

    # Source true path.
    b_true = family.step1(a, rng)
    c_true = family.step2(b_true, rng)

    # Observed, noisy, partially visible path.
    a_obs = add_noise(a, rng, noise_sigma * 0.45)
    b_obs = add_noise(b_true, rng, noise_sigma)
    c_obs = add_noise(c_true, rng, noise_sigma)

    mask_ab = visible_mask(rng, len(a), keep_frac)
    mask_bc = visible_mask(rng, len(a), keep_frac)

    disp_ab, disp_bc, fit_ab, fit_bc = reconstruct_path(a_obs, b_obs, c_obs, mask_ab, mask_bc)

    # New start D is a transformed version of A so index-correspondence remains meaningful
    # while endpoints are not trivially identical to source.
    d = affine_apply(a, rot(rng_uniform(rng, -0.45, 0.45)), np.array([rng_uniform(rng, -0.10, 0.10), rng_uniform(rng, -0.10, 0.10)]))
    d += rng.normal(0, 0.006, size=d.shape)

    # True target path is generated by transferring the actual source displacement fields.
    true_disp_ab = b_true - a
    true_disp_bc = c_true - b_true
    e_true = d + true_disp_ab
    f_true = e_true + true_disp_bc

    e_pred, f_pred = transfer_path(d, disp_ab, disp_bc)

    branches = make_adversarial_branches(d, e_true, f_true, rng, candidate_noise=candidate_noise)

    traj_scores = [trajectory_score(e_pred, f_pred, br) for br in branches]
    final_scores = [final_only_score(f_pred, br) for br in branches]

    best_idx, margin = choose(traj_scores)
    final_idx, final_margin = choose(final_scores)

    chosen = branches[best_idx].label
    final_chosen = branches[final_idx].label
    correct = chosen == "full_path"
    final_correct = final_chosen == "full_path"

    # Scramble stability: candidate ordering should not matter.
    perm = rng.permutation(len(branches))
    branches_scrambled = [branches[i] for i in perm]
    scrambled_scores = [trajectory_score(e_pred, f_pred, br) for br in branches_scrambled]
    scrambled_best, _ = choose(scrambled_scores)
    stable = branches_scrambled[scrambled_best].label == chosen

    transfer_error = chamfer(f_pred, f_true)
    source_recon = 0.5 * (chamfer(a_obs + disp_ab, b_true) + chamfer(b_obs + disp_bc, c_true))

    if write_example:
        ex_path = EXAMPLE_DIR / f"phase36_example_{trial_id:03d}_{family.name}.png"
        plot_example(ex_path, family.name, a_obs, b_obs, c_obs, d, e_pred, f_pred, branches, chosen)

    return TrialResult(
        trial=trial_id,
        family=family.name,
        correct=bool(correct),
        final_correct=bool(final_correct),
        stable=bool(stable),
        chosen_branch=chosen,
        final_chosen_branch=final_chosen,
        margin=float(margin),
        final_margin=float(final_margin),
        fit_ab=float(fit_ab),
        fit_bc=float(fit_bc),
        transfer_error=float(transfer_error),
        source_reconstruction_error=float(source_recon),
        keep_frac=float(keep_frac),
        noise_sigma=float(noise_sigma),
    )


def summarize(results: List[TrialResult]) -> Dict:
    fam_names = [f.name for f in FAMILIES]
    by_family = {}
    for fam in fam_names:
        rows = [r for r in results if r.family == fam]
        if not rows:
            continue
        by_family[fam] = {
            "trials": len(rows),
            "trajectory_accuracy": float(np.mean([r.correct for r in rows])),
            "final_only_accuracy": float(np.mean([r.final_correct for r in rows])),
            "scramble_stability": float(np.mean([r.stable for r in rows])),
            "mean_fit_ab": float(np.mean([r.fit_ab for r in rows])),
            "mean_fit_bc": float(np.mean([r.fit_bc for r in rows])),
            "mean_transfer_error": float(np.mean([r.transfer_error for r in rows])),
            "mean_source_reconstruction_error": float(np.mean([r.source_reconstruction_error for r in rows])),
            "mean_margin": float(np.mean([r.margin for r in rows])),
            "min_margin": float(np.min([r.margin for r in rows])),
            "mean_final_only_margin": float(np.mean([r.final_margin for r in rows])),
        }

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "trials": len(results),
        "trajectory_accuracy": float(np.mean([r.correct for r in results])),
        "final_only_accuracy": float(np.mean([r.final_correct for r in results])),
        "gain": float(np.mean([r.correct for r in results]) - np.mean([r.final_correct for r in results])),
        "scramble_stability": float(np.mean([r.stable for r in results])),
        "mean_fit_ab": float(np.mean([r.fit_ab for r in results])),
        "mean_fit_bc": float(np.mean([r.fit_bc for r in results])),
        "mean_transfer_error": float(np.mean([r.transfer_error for r in results])),
        "mean_source_reconstruction_error": float(np.mean([r.source_reconstruction_error for r in results])),
        "mean_margin": float(np.mean([r.margin for r in results])),
        "min_margin": float(np.min([r.margin for r in results])),
        "mean_final_only_margin": float(np.mean([r.final_margin for r in results])),
        "by_family": by_family,
    }

    # Phase pass criteria intentionally emphasize rejecting partial truths.
    summary["PHASE36_NOISY_MULTIBRANCH_CAUSAL_TRAJECTORY_PASS"] = bool(
        summary["trajectory_accuracy"] >= 0.95
        and summary["final_only_accuracy"] <= 0.35
        and summary["scramble_stability"] >= 0.95
        and summary["gain"] >= 0.55
    )
    return summary


def run_trials(
    n_trials: int = 700,
    noise_sigma: float = 0.018,
    keep_frac: float = 0.62,
    candidate_noise: float = 0.006,
    write_examples: bool = True,
) -> Tuple[List[TrialResult], Dict]:
    rng = np.random.default_rng(SEED)
    results: List[TrialResult] = []

    examples_written = set()
    for i in range(n_trials):
        family = FAMILIES[i % len(FAMILIES)]
        # Randomize within deterministic seed.
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        write_example = False
        if write_examples and family.name not in examples_written and len(examples_written) < len(FAMILIES):
            write_example = True
            examples_written.add(family.name)

        result = run_one_trial(
            i,
            family,
            local_rng,
            noise_sigma=noise_sigma,
            keep_frac=keep_frac,
            candidate_noise=candidate_noise,
            write_example=write_example,
        )
        results.append(result)

    return results, summarize(results)


def run_noise_sweep() -> List[Dict[str, float]]:
    rows = []
    for sigma in [0.006, 0.012, 0.018, 0.026, 0.034, 0.045]:
        results, summary = run_trials(
            n_trials=280,
            noise_sigma=sigma,
            keep_frac=0.62,
            candidate_noise=0.006,
            write_examples=False,
        )
        rows.append({
            "noise_sigma": float(sigma),
            "trajectory_accuracy": float(summary["trajectory_accuracy"]),
            "final_only_accuracy": float(summary["final_only_accuracy"]),
            "mean_margin": float(summary["mean_margin"]),
            "mean_transfer_error": float(summary["mean_transfer_error"]),
        })
    return rows


def write_csv(results: List[TrialResult], path: Path):
    fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            w.writerow(row)


def write_sweep_csv(rows: List[Dict[str, float]], path: Path):
    fieldnames = ["noise_sigma", "trajectory_accuracy", "final_only_accuracy", "mean_margin", "mean_transfer_error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_report(summary: Dict, path: Path):
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Intent")
    lines.append("")
    lines.append("Phase 36 continues the reset from endpoint choice into causal path reasoning.")
    lines.append("The source trajectory is noisy and partially visible, and every wrong answer preserves one partial truth.")
    lines.append("The correct answer is the only branch preserving the full reconstructed path.")
    lines.append("")
    lines.append("## Overall result")
    lines.append("")
    lines.append(f"- pass: `{summary['PHASE36_NOISY_MULTIBRANCH_CAUSAL_TRAJECTORY_PASS']}`")
    lines.append(f"- trajectory accuracy: `{summary['trajectory_accuracy']:.4f}`")
    lines.append(f"- final-only accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- gain: `{summary['gain']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- mean source reconstruction error: `{summary['mean_source_reconstruction_error']:.6f}`")
    lines.append(f"- mean transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append(f"- mean trajectory margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- minimum trajectory margin: `{summary['min_margin']:.6f}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| family | traj acc | final acc | stable | transfer err | recon err | min margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, row in summary["by_family"].items():
        lines.append(
            f"| {fam} | {row['trajectory_accuracy']:.3f} | {row['final_only_accuracy']:.3f} | "
            f"{row['scramble_stability']:.3f} | {row['mean_transfer_error']:.6f} | "
            f"{row['mean_source_reconstruction_error']:.6f} | {row['min_margin']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If this phase passes, the model is no longer merely selecting a visually close endpoint.")
    lines.append("It is using a reconstructed noisy path as memory, then choosing the only future branch whose midpoint, endpoint, and step-fields cohere.")
    lines.append("That is a stricter test than Phase 35 because the distractors are not random negatives; they are plausible partial truths.")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(results: List[TrialResult], summary: Dict, sweep_rows: List[Dict[str, float]]):
    fams = list(summary["by_family"].keys())
    traj_vals = [summary["by_family"][f]["trajectory_accuracy"] for f in fams]
    final_vals = [summary["by_family"][f]["final_only_accuracy"] for f in fams]

    barh_metric(
        OUT_DIR / "phase36_noisy_multibranch_accuracy.png",
        "Phase 36 noisy multi-branch causal trajectory reasoning: accuracy by hidden family",
        fams,
        traj_vals,
        "trajectory accuracy",
    )

    grouped_baseline_plot(
        OUT_DIR / "phase36_noisy_multibranch_baseline_gap.png",
        fams,
        traj_vals,
        final_vals,
    )

    histogram(
        OUT_DIR / "phase36_noisy_multibranch_margin_distribution.png",
        "Phase 36 noisy trajectory answer margin distribution",
        [r.margin for r in results],
        "runner-up trajectory score - best trajectory score",
        bins=38,
    )

    histogram(
        OUT_DIR / "phase36_noisy_multibranch_final_only_margin_distribution.png",
        "Phase 36 final-only endpoint margin distribution",
        [r.final_margin for r in results],
        "runner-up endpoint score - best endpoint score",
        bins=38,
    )

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.chosen_branch] = counts.get(r.chosen_branch, 0) + 1
    branch_counts_plot(OUT_DIR / "phase36_noisy_multibranch_branch_counts.png", counts)

    noise_sweep_plot(OUT_DIR / "phase36_noisy_multibranch_noise_sweep.png", sweep_rows)


def main():
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT_DIR}")
    print(f"[{PHASE}] reset continued: from occluded trajectory reconstruction to noisy partial-truth rejection")
    print(f"[{PHASE}] task: infer noisy/occluded A->B->C, transfer to D, reject multi-branch partial truths")

    results, summary = run_trials(
        n_trials=700,
        noise_sigma=0.018,
        keep_frac=0.62,
        candidate_noise=0.006,
        write_examples=True,
    )
    sweep_rows = run_noise_sweep()

    trials_path = OUT_DIR / "phase36_noisy_multibranch_trials.csv"
    summary_path = OUT_DIR / "phase36_noisy_multibranch_summary.json"
    report_path = OUT_DIR / "phase36_noisy_multibranch_report.md"
    sweep_path = OUT_DIR / "phase36_noise_sweep.csv"

    write_csv(results, trials_path)
    write_sweep_csv(sweep_rows, sweep_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, report_path)
    make_plots(results, summary, sweep_rows)

    print(f"[{PHASE}] PHASE36_NOISY_MULTIBRANCH_CAUSAL_TRAJECTORY_PASS={summary['PHASE36_NOISY_MULTIBRANCH_CAUSAL_TRAJECTORY_PASS']}")
    print(
        f"[{PHASE}] trajectory_accuracy={summary['trajectory_accuracy']:.4f} "
        f"final_only_accuracy={summary['final_only_accuracy']:.4f} "
        f"gain={summary['gain']:.4f} "
        f"scramble_stability={summary['scramble_stability']:.4f} "
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
    for fam, row in summary["by_family"].items():
        print(
            f"  - {fam:<18} traj_acc={row['trajectory_accuracy']:.3f} "
            f"final_acc={row['final_only_accuracy']:.3f} "
            f"stable={row['scramble_stability']:.3f} "
            f"transfer_err={row['mean_transfer_error']:.6f} "
            f"recon_err={row['mean_source_reconstruction_error']:.6f} "
            f"min_margin={row['min_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote noise sweep: {sweep_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
