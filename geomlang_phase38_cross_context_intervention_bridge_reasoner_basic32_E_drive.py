#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 38: Cross-context intervention bridge reasoner

Reset continuation:
  Phase 34 proved path-only reasoning when endpoints are adversarially ambiguous.
  Phase 35 proved reconstruction under occlusion.
  Phase 36 proved noisy multi-branch causal trajectory disambiguation.

Phase 38 adds cross-context intervention pressure: the observed source path contains an
intervened middle state, and the target branch set contains a second context-shifted
intervention lure. The reasoner must bridge across the source intervention and the target
context shift, preserving the invariant causal path rather than endpoint similarity, copied
intervention artifacts, or context-specific distortions.

Task:
  Observe A -> B_observed -> C, where B_observed = B_latent + partial intervention.
  Reconstruct the invariant latent two-step causal path.
  Given D and several D->E->F candidate branches under a mild context shift, choose the
  branch preserving the latent path, not the nearest endpoint, the copied source
  intervention, or the branch that matches only the target-side context distortion.

Outputs:
  E:\BBIT\outputs_basic32\phase38_cross_context_intervention_bridge_trials.csv
  E:\BBIT\outputs_basic32\phase38_cross_context_intervention_bridge_summary.json
  E:\BBIT\outputs_basic32\phase38_cross_context_intervention_bridge_report.md
  E:\BBIT\outputs_basic32\phase38_context_intervention_sweep.csv
  E:\BBIT\outputs_basic32\phase38_examples\*.png
  plus summary png charts.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "38"
TITLE = "Cross-context intervention bridge reasoner"
SCRIPT_NAME = "geomlang_phase38_cross_context_intervention_bridge_causal_path_reasoner_basic32_E_drive.py"
ROOT = Path(r"E:\BBIT") if Path(r"E:\BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase38_examples"

SEED = 37037
TRIALS = 700
N_POINTS = 64
SOURCE_NOISE_SIGMA = 0.010
CANDIDATE_NOISE_SIGMA = 0.0012
INTERVENTION_FRACTION = 0.36
INTERVENTION_MAG = 0.145
TARGET_CONTEXT_MAG = 0.13

FAMILIES = [
    "rigid_chain",
    "similarity_chain",
    "shear_chain",
    "left_right_chain",
    "top_bottom_chain",
    "quadrants_chain",
    "core_shell_chain",
]


@dataclass
class Affine:
    A: np.ndarray  # 2x2
    b: np.ndarray  # 2,

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.A.T + self.b


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def normalize_points(p: np.ndarray) -> np.ndarray:
    p = p - p.mean(axis=0, keepdims=True)
    scale = np.sqrt((p * p).sum(axis=1)).max()
    if scale < 1e-9:
        scale = 1.0
    return p / scale


def make_cloud(rng: np.random.Generator, family: str, n: int = N_POINTS) -> np.ndarray:
    """Create a stable indexed 2D cloud with enough structure for the family."""
    if family == "quadrants_chain":
        per = n // 4
        centers = np.array([[-0.55, -0.55], [0.55, -0.55], [-0.55, 0.55], [0.55, 0.55]])
        pts = []
        for c in centers:
            pts.append(c + rng.normal(0, 0.105, size=(per, 2)))
        p = np.vstack(pts)
    elif family == "core_shell_chain":
        n_core = n // 3
        n_shell = n - n_core
        core = rng.normal(0, 0.12, size=(n_core, 2))
        ang = rng.uniform(0, 2 * np.pi, size=n_shell)
        rad = rng.normal(0.74, 0.045, size=n_shell)
        shell = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
        p = np.vstack([core, shell])
    elif family in ("left_right_chain", "top_bottom_chain"):
        x = rng.uniform(-0.75, 0.75, size=n)
        y = 0.25 * np.sin(3.1 * x) + rng.normal(0, 0.13, size=n)
        p = np.c_[x, y]
        if family == "top_bottom_chain":
            p = p[:, ::-1]
    else:
        ang = rng.uniform(0, 2 * np.pi, size=n)
        rad = rng.beta(1.7, 2.8, size=n) * 0.95
        p = np.c_[rad * np.cos(ang), 0.72 * rad * np.sin(ang)]
        p += rng.normal(0, 0.035, size=p.shape)
    return normalize_points(p)


def family_affines(rng: np.random.Generator, family: str) -> Tuple[Affine, Affine]:
    """Return the latent causal operators T1 and T2 for A->B and B->C."""
    if family == "rigid_chain":
        t1 = Affine(rot(rng.uniform(0.23, 0.42)), np.array([0.16, -0.05]))
        t2 = Affine(rot(rng.uniform(-0.44, -0.24)), np.array([-0.08, 0.14]))
    elif family == "similarity_chain":
        t1 = Affine(1.10 * rot(rng.uniform(0.15, 0.35)), np.array([0.09, 0.11]))
        t2 = Affine(0.88 * rot(rng.uniform(-0.31, -0.12)), np.array([0.13, -0.07]))
    elif family == "shear_chain":
        t1 = Affine(np.array([[1.00, 0.26], [0.02, 1.00]]), np.array([0.07, -0.11]))
        t2 = Affine(np.array([[1.00, -0.19], [0.05, 1.00]]), np.array([-0.10, 0.10]))
    elif family == "left_right_chain":
        t1 = Affine(np.array([[-1.0, 0.0], [0.0, 1.0]]) @ rot(rng.uniform(-0.10, 0.10)), np.array([0.18, 0.04]))
        t2 = Affine(rot(rng.uniform(0.08, 0.22)) @ np.array([[-1.0, 0.0], [0.0, 1.0]]), np.array([-0.14, 0.13]))
    elif family == "top_bottom_chain":
        t1 = Affine(np.array([[1.0, 0.0], [0.0, -1.0]]) @ rot(rng.uniform(-0.10, 0.10)), np.array([0.03, -0.18]))
        t2 = Affine(rot(rng.uniform(-0.22, -0.08)) @ np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([0.15, 0.11]))
    elif family == "quadrants_chain":
        t1 = Affine(np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([0.09, 0.07]))
        t2 = Affine(np.array([[0.0, 1.0], [-1.0, 0.0]]), np.array([-0.11, 0.08]))
    elif family == "core_shell_chain":
        # Mild radial-ish affine surrogate: expansion then contraction/turn.
        t1 = Affine(np.array([[1.13, 0.08], [-0.03, 1.08]]), np.array([0.04, 0.10]))
        t2 = Affine(np.array([[0.91, -0.10], [0.06, 0.88]]), np.array([0.11, -0.08]))
    else:
        raise ValueError(f"Unknown family: {family}")
    return t1, t2


def add_observation_noise(rng: np.random.Generator, p: np.ndarray, sigma: float) -> np.ndarray:
    return p + rng.normal(0.0, sigma, size=p.shape)


def intervention_vector_field(p: np.ndarray, family: str) -> np.ndarray:
    """External artifact that should be rejected, not copied as the causal path."""
    x, y = p[:, 0], p[:, 1]
    if family == "left_right_chain":
        v = np.c_[0.9 + 0.25 * np.sin(4 * y), 0.15 * np.cos(3 * x)]
    elif family == "top_bottom_chain":
        v = np.c_[0.18 * np.sin(3 * y), -(0.9 + 0.20 * np.cos(4 * x))]
    elif family == "quadrants_chain":
        v = np.c_[np.sign(x + 1e-6), -np.sign(y + 1e-6)]
    elif family == "core_shell_chain":
        r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-6
        v = p / r
    elif family == "shear_chain":
        v = np.c_[y, 0.35 * x]
    elif family == "similarity_chain":
        v = np.c_[-y, x]
    else:
        v = np.c_[0.6 * x + 0.2, -0.5 * y + 0.1]
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n


def apply_intervention(
    rng: np.random.Generator,
    b_latent: np.ndarray,
    family: str,
    frac: float = INTERVENTION_FRACTION,
    mag: float = INTERVENTION_MAG,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(b_latent)
    k = max(3, int(round(frac * n)))
    # Intervene on one spatially coherent side, not random salt-and-pepper noise.
    projection = b_latent @ np.array([0.73, -0.41])
    cutoff = np.quantile(projection, 1.0 - frac)
    mask = projection >= cutoff
    # make exact count stable
    if mask.sum() != k:
        idx = np.argsort(projection)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
    v = intervention_vector_field(b_latent, family)
    b_obs = b_latent.copy()
    b_obs[mask] = b_obs[mask] + mag * v[mask]
    b_obs = b_obs + rng.normal(0, SOURCE_NOISE_SIGMA, size=b_obs.shape)
    return b_obs, mask


def fit_affine_lstsq(p: np.ndarray, q: np.ndarray, weights: np.ndarray | None = None) -> Affine:
    x = np.c_[p, np.ones(len(p))]
    if weights is None:
        m, *_ = np.linalg.lstsq(x, q, rcond=None)
    else:
        w = np.sqrt(np.maximum(weights, 1e-9))[:, None]
        m, *_ = np.linalg.lstsq(x * w, q * w, rcond=None)
    A = m[:2, :].T
    b = m[2, :]
    return Affine(A, b)


def robust_fit_affine(p: np.ndarray, q: np.ndarray, keep_frac: float = 0.62, iters: int = 5) -> Tuple[Affine, float]:
    """Trimmed affine fit: treats the intervention-corrupted points as outliers."""
    n = len(p)
    keep_n = max(8, int(round(keep_frac * n)))
    weights = np.ones(n, dtype=np.float64)
    fit = fit_affine_lstsq(p, q, weights)
    for _ in range(iters):
        pred = fit(p)
        err = np.linalg.norm(pred - q, axis=1)
        keep = np.argsort(err)[:keep_n]
        weights = np.full(n, 1e-4, dtype=np.float64)
        weights[keep] = 1.0
        fit = fit_affine_lstsq(p, q, weights)
    residual = float(np.mean(np.sort(np.linalg.norm(fit(p) - q, axis=1))[:keep_n]))
    return fit, residual


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    # The synthetic point clouds preserve point identity across causal transforms,
    # so the intended metric is fast indexed mean displacement rather than an
    # unordered nearest-neighbor Chamfer.
    return float(np.linalg.norm(a - b, axis=1).mean())


def affine_compose(t2: Affine, t1: Affine) -> Affine:
    return Affine(t2.A @ t1.A, t2.A @ t1.b + t2.b)


def make_candidates(
    rng: np.random.Generator,
    d: np.ndarray,
    e_true: np.ndarray,
    f_true: np.ndarray,
    t1: Affine,
    t2: Affine,
    family: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    noise = lambda p, s=CANDIDATE_NOISE_SIGMA: p + rng.normal(0, s, size=p.shape)
    combo = affine_compose(t2, t1)
    predicted_endpoint = combo(d)

    # Correct: full latent path. Endpoint is slightly observational, not the exact final-only ideal.
    correct = (noise(e_true, 0.0010), noise(f_true, 0.0016))

    # Wrong 1: endpoint lure. Final endpoint is nearly perfect but midpoint is causally wrong.
    wrong_mid = normalize_points(make_cloud(rng, family, len(d))) * 0.82 + e_true.mean(axis=0)
    endpoint_lure = (wrong_mid, predicted_endpoint + rng.normal(0, 0.00018, size=f_true.shape))

    # Wrong 2: copies the observed intervention artifact as if it were causal.
    artifact = INTERVENTION_MAG * 0.92 * intervention_vector_field(e_true, family)
    artifact_branch = (noise(e_true + artifact, 0.0012), noise(t2(e_true + artifact), 0.0015))

    # Wrong 2b: target-side context-shift lure. It is coherent as a branch, but the
    # whole path carries a context artifact that should not be mistaken for the invariant path.
    context_angle = rng.uniform(0.0, 2.0 * math.pi)
    context_vec = TARGET_CONTEXT_MAG * np.array([math.cos(context_angle), math.sin(context_angle)])
    context_shift_lure = (noise(e_true + context_vec, 0.0010), noise(f_true + context_vec, 0.0014))

    # Wrong 3: preserves first step only; second step is rotated/sheared away.
    wrong_t2 = Affine(rot(0.34) @ t2.A, t2.b + np.array([0.028, -0.019]))
    first_step_only = (noise(e_true, 0.0011), noise(wrong_t2(e_true), 0.0013))

    # Wrong 4: preserves second step from a false midpoint.
    delta = np.array([0.055, -0.045])
    false_e = e_true + delta + 0.025 * intervention_vector_field(e_true, family)
    second_step_only = (noise(false_e, 0.0012), noise(t2(false_e), 0.0014))

    # Wrong 5: inverse/anti-causal branch.
    anti = Affine(-0.72 * t1.A, t1.b + np.array([0.02, 0.03]))
    anti_branch = (noise(anti(d), 0.0013), noise(combo(d) + 0.030 * intervention_vector_field(f_true, family), 0.0013))

    return {
        "full_path": correct,
        "endpoint_lure": endpoint_lure,
        "intervention_copy": artifact_branch,
        "context_shift_lure": context_shift_lure,
        "first_step_only": first_step_only,
        "second_step_only": second_step_only,
        "anti_causal": anti_branch,
    }


def trajectory_score(d: np.ndarray, e: np.ndarray, f: np.ndarray, fit1: Affine, fit2: Affine) -> float:
    pred_e = fit1(d)
    pred_f_from_e = fit2(e)
    pred_f_direct = affine_compose(fit2, fit1)(d)
    # Causal path score: midpoint, endpoint-through-midpoint, direct endpoint, and local step coherence.
    return (
        1.20 * chamfer(e, pred_e)
        + 1.00 * chamfer(f, pred_f_from_e)
        + 0.55 * chamfer(f, pred_f_direct)
        + 0.35 * abs(chamfer(e, d) - chamfer(pred_e, d))
    )


def final_only_score(f: np.ndarray, fit1: Affine, fit2: Affine, d: np.ndarray) -> float:
    return chamfer(f, affine_compose(fit2, fit1)(d))


def run_trial(rng: np.random.Generator, trial_id: int, intervention_fraction: float = INTERVENTION_FRACTION) -> Dict[str, object]:
    family = FAMILIES[trial_id % len(FAMILIES)]
    a = make_cloud(rng, family)
    d = make_cloud(rng, family)
    t1, t2 = family_affines(rng, family)
    b_latent = t1(a)
    c = t2(b_latent)
    b_obs, mask = apply_intervention(rng, b_latent, family, frac=intervention_fraction)

    a_obs = add_observation_noise(rng, a, SOURCE_NOISE_SIGMA)
    c_obs = add_observation_noise(rng, c, SOURCE_NOISE_SIGMA)

    # Robustly infer latent causal operators despite the intervened middle state.
    fit1, fit_ab = robust_fit_affine(a_obs, b_obs, keep_frac=max(0.42, 1.0 - intervention_fraction - 0.03))
    fit2, fit_bc = robust_fit_affine(b_obs, c_obs, keep_frac=max(0.42, 1.0 - intervention_fraction - 0.03))

    e_true = t1(d)
    f_true = t2(e_true)
    candidates = make_candidates(rng, d, e_true, f_true, t1, t2, family)

    traj_scores = {name: trajectory_score(d, e, f, fit1, fit2) for name, (e, f) in candidates.items()}
    final_scores = {name: final_only_score(f, fit1, fit2, d) for name, (e, f) in candidates.items()}

    traj_rank = sorted(traj_scores.items(), key=lambda kv: kv[1])
    final_rank = sorted(final_scores.items(), key=lambda kv: kv[1])
    pred = traj_rank[0][0]
    final_pred = final_rank[0][0]

    # Scramble candidate order and verify answer is order-invariant.
    names = list(candidates.keys())
    rng.shuffle(names)
    scrambled_rank = sorted([(n, traj_scores[n]) for n in names], key=lambda kv: kv[1])
    stable = int(scrambled_rank[0][0] == pred)

    return {
        "trial": trial_id,
        "family": family,
        "pred_branch": pred,
        "final_only_pred_branch": final_pred,
        "correct": int(pred == "full_path"),
        "final_only_correct": int(final_pred == "full_path"),
        "stable": stable,
        "fit_ab": fit_ab,
        "fit_bc": fit_bc,
        "intervention_fraction": intervention_fraction,
        "intervened_points": int(mask.sum()),
        "trajectory_margin": float(traj_rank[1][1] - traj_rank[0][1]),
        "final_only_margin": float(final_rank[1][1] - final_rank[0][1]),
        "score_full_path": traj_scores["full_path"],
        "score_endpoint_lure": traj_scores["endpoint_lure"],
        "score_intervention_copy": traj_scores["intervention_copy"],
        "score_context_shift_lure": traj_scores["context_shift_lure"],
        "score_first_step_only": traj_scores["first_step_only"],
        "score_second_step_only": traj_scores["second_step_only"],
        "score_anti_causal": traj_scores["anti_causal"],
        "source_reconstruction_error": float(chamfer(fit1(a), b_latent) + chamfer(fit2(b_latent), c)),
        "transfer_error": float(chamfer(fit1(d), e_true) + chamfer(fit2(e_true), f_true)),
        "_viz": (a, b_latent, b_obs, c, d, candidates, mask, fit1, fit2),
    }


def strip_viz(row: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in row.items() if k != "_viz"}


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    clean = [strip_viz(r) for r in rows]
    overall = {
        "phase": PHASE,
        "title": TITLE,
        "pass": False,
        "trials": len(rows),
        "trajectory_accuracy": float(np.mean([r["correct"] for r in clean])),
        "final_only_accuracy": float(np.mean([r["final_only_correct"] for r in clean])),
        "gain": 0.0,
        "scramble_stability": float(np.mean([r["stable"] for r in clean])),
        "mean_fit_ab": float(np.mean([r["fit_ab"] for r in clean])),
        "mean_fit_bc": float(np.mean([r["fit_bc"] for r in clean])),
        "mean_source_reconstruction_error": float(np.mean([r["source_reconstruction_error"] for r in clean])),
        "mean_transfer_error": float(np.mean([r["transfer_error"] for r in clean])),
        "mean_margin": float(np.mean([r["trajectory_margin"] for r in clean])),
        "min_margin": float(np.min([r["trajectory_margin"] for r in clean])),
    }
    overall["gain"] = overall["trajectory_accuracy"] - overall["final_only_accuracy"]

    fams = {}
    for fam in FAMILIES:
        fr = [r for r in clean if r["family"] == fam]
        fams[fam] = {
            "trajectory_accuracy": float(np.mean([r["correct"] for r in fr])),
            "final_only_accuracy": float(np.mean([r["final_only_correct"] for r in fr])),
            "scramble_stability": float(np.mean([r["stable"] for r in fr])),
            "mean_transfer_error": float(np.mean([r["transfer_error"] for r in fr])),
            "mean_source_reconstruction_error": float(np.mean([r["source_reconstruction_error"] for r in fr])),
            "min_margin": float(np.min([r["trajectory_margin"] for r in fr])),
        }
    overall["family_summary"] = fams
    overall["branch_counts"] = {name: int(sum(r["pred_branch"] == name for r in clean)) for name in sorted(set(r["pred_branch"] for r in clean))}

    overall["pass"] = bool(
        overall["trajectory_accuracy"] >= 0.985
        and overall["scramble_stability"] >= 0.985
        and overall["gain"] >= 0.20
        and overall["min_margin"] > 0.001
    )
    return overall


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    clean = [strip_viz(r) for r in rows]
    keys = list(clean[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(clean)


def write_report(path: Path, summary: Dict[str, object]) -> None:
    fam = summary["family_summary"]
    lines = []
    lines.append("# Phase 38: Cross-context intervention bridge reasoner\n")
    lines.append("## Intent\n")
    lines.append("Phase 38 tests whether the system can separate a causal path from an external intervention artifact. The observed middle state is partially corrupted, so a final-point or artifact-copy strategy should fail.\n")
    lines.append("## Overall result\n")
    lines.append(f"- pass: `{summary['pass']}`")
    lines.append(f"- trajectory accuracy: `{summary['trajectory_accuracy']:.4f}`")
    lines.append(f"- final-only accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- gain: `{summary['gain']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- mean source reconstruction error: `{summary['mean_source_reconstruction_error']:.6f}`")
    lines.append(f"- mean transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append(f"- mean trajectory margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- minimum trajectory margin: `{summary['min_margin']:.6f}`\n")
    lines.append("## Family summary\n")
    lines.append("| family | traj acc | final acc | stable | transfer err | recon err | min margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in fam.items():
        lines.append(
            f"| {k} | {v['trajectory_accuracy']:.3f} | {v['final_only_accuracy']:.3f} | {v['scramble_stability']:.3f} | "
            f"{v['mean_transfer_error']:.6f} | {v['mean_source_reconstruction_error']:.6f} | {v['min_margin']:.6f} |"
        )
    lines.append("\n## Interpretation\n")
    lines.append("If this phase passes, the reasoner is no longer merely ignoring a single intervention. It can bridge across an intervened source observation and a target-side context artifact. The correct answer is the branch that preserves the hidden causal path, while distractors preserve endpoint closeness, copied intervention structure, context-shift coherence, partial steps, or anti-causal inversion.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy(summary: Dict[str, object]) -> None:
    fam = summary["family_summary"]
    labels = list(fam.keys())
    vals = [fam[k]["trajectory_accuracy"] for k in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("trajectory accuracy")
    ax.set_title("Phase 38 cross-context intervention bridge causal path reasoning: accuracy by hidden family")
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_accuracy.png", dpi=150)
    plt.close(fig)


def plot_baseline_gap(summary: Dict[str, object]) -> None:
    fam = summary["family_summary"]
    labels = list(fam.keys())
    traj = [fam[k]["trajectory_accuracy"] for k in labels]
    final = [fam[k]["final_only_accuracy"] for k in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, final, width, label="final-only")
    ax.bar(x + width / 2, traj, width, label="cross-context trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 38 final-only baseline vs cross-context trajectory reasoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_baseline_gap.png", dpi=150)
    plt.close(fig)


def plot_margins(rows: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist([r["trajectory_margin"] for r in rows], bins=36)
    ax.set_title("Phase 38 cross-context trajectory answer margin distribution")
    ax.set_xlabel("runner-up trajectory score - best trajectory score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_margin_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist([r["final_only_margin"] for r in rows], bins=36)
    ax.set_title("Phase 38 final-only endpoint margin distribution")
    ax.set_xlabel("runner-up endpoint score - best endpoint score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_final_only_margin_distribution.png", dpi=150)
    plt.close(fig)


def plot_branch_counts(summary: Dict[str, object]) -> None:
    counts = summary["branch_counts"]
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, vals)
    ax.set_title("Phase 38 predicted branch counts")
    ax.set_ylabel("trials")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_branch_counts.png", dpi=150)
    plt.close(fig)


def intervention_sweep() -> List[Dict[str, float]]:
    rows = []
    for frac in [0.10, 0.22, 0.34, 0.46, 0.58, 0.70]:
        rng = np.random.default_rng(SEED + int(frac * 1000))
        trial_rows = [run_trial(rng, i, intervention_fraction=frac) for i in range(80)]
        rows.append({
            "intervention_fraction": frac,
            "trajectory_accuracy": float(np.mean([r["correct"] for r in trial_rows])),
            "final_only_accuracy": float(np.mean([r["final_only_correct"] for r in trial_rows])),
            "mean_margin": float(np.mean([r["trajectory_margin"] for r in trial_rows])),
        })
    with (OUT / "phase38_context_intervention_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot([r["intervention_fraction"] for r in rows], [r["trajectory_accuracy"] for r in rows], marker="o", label="trajectory reasoning")
    ax.plot([r["intervention_fraction"] for r in rows], [r["final_only_accuracy"] for r in rows], marker="o", label="final-only")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("intervention fraction")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 38 intervention stress sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase38_cross_context_intervention_bridge_intervention_sweep.png", dpi=150)
    plt.close(fig)
    return rows


def plot_example(row: Dict[str, object], idx: int) -> None:
    a, b_latent, b_obs, c, d, candidates, mask, fit1, fit2 = row["_viz"]
    pred_e = fit1(d)
    pred_f = fit2(pred_e)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax = axes[0, 0]
    ax.scatter(a[:, 0], a[:, 1], s=10, label="A")
    ax.scatter(b_latent[:, 0], b_latent[:, 1], s=10, label="B latent")
    ax.scatter(b_obs[:, 0], b_obs[:, 1], s=10, label="B observed")
    ax.scatter(b_obs[mask, 0], b_obs[mask, 1], s=20, marker="x", label="intervened")
    ax.set_title("source: latent path vs observed intervention")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.scatter(a[:, 0], a[:, 1], s=10, label="A")
    ax.scatter(fit1(a)[:, 0], fit1(a)[:, 1], s=10, label="robust A->B")
    ax.scatter(c[:, 0], c[:, 1], s=10, label="C")
    ax.set_title("robust fitted causal operators")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.scatter(d[:, 0], d[:, 1], s=10, label="D")
    ax.scatter(pred_e[:, 0], pred_e[:, 1], s=10, label="pred E")
    ax.scatter(pred_f[:, 0], pred_f[:, 1], s=10, label="pred F")
    ax.set_title("transferred latent trajectory")
    ax.legend(fontsize=7)

    for ax, name in zip(axes[1], ["full_path", "endpoint_lure", "intervention_copy"]):
        e, f = candidates[name]
        ax.scatter(d[:, 0], d[:, 1], s=8, label="D")
        ax.scatter(e[:, 0], e[:, 1], s=8, label="E")
        ax.scatter(f[:, 0], f[:, 1], s=8, label="F")
        ax.set_title(name)
        ax.legend(fontsize=7)
    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Phase 38 example {idx}: {row['family']} | pred={row['pred_branch']} | final-only={row['final_only_pred_branch']}")
    fig.tight_layout()
    fig.savefig(EXAMPLE_DIR / f"phase37_example_{idx:02d}_{row['family']}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from cross-context intervention bridge reconstruction to cross-context causal bridge reasoning")
    print(f"[{PHASE}] task: infer latent A->B->C across source intervention and target context lures; reject artifact/final-only/context-copy branches")

    rows = [run_trial(rng, i) for i in range(TRIALS)]
    summary = aggregate(rows)

    write_csv(OUT / "phase38_cross_context_intervention_bridge_trials.csv", rows)
    (OUT / "phase38_cross_context_intervention_bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT / "phase38_cross_context_intervention_bridge_report.md", summary)

    plot_accuracy(summary)
    plot_baseline_gap(summary)
    plot_margins(rows)
    plot_branch_counts(summary)
    intervention_sweep()

    # A few concrete visual examples.
    for idx, row in enumerate(rows[:6], start=1):
        plot_example(row, idx)

    print(f"[{PHASE}] PHASE38_CROSS_CONTEXT_INTERVENTION_BRIDGE_REASONING_PASS={summary['pass']}")
    print(
        f"[{PHASE}] trajectory_accuracy={summary['trajectory_accuracy']:.4f} "
        f"final_only_accuracy={summary['final_only_accuracy']:.4f} "
        f"gain={summary['gain']:.4f} scramble_stability={summary['scramble_stability']:.4f} trials={summary['trials']}"
    )
    print(
        f"[{PHASE}] mean_fit_ab={summary['mean_fit_ab']:.6f} mean_fit_bc={summary['mean_fit_bc']:.6f} "
        f"mean_source_reconstruction_error={summary['mean_source_reconstruction_error']:.6f} "
        f"mean_transfer_error={summary['mean_transfer_error']:.6f} mean_margin={summary['mean_margin']:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam, v in summary["family_summary"].items():
        print(
            f"  - {fam:<18} traj_acc={v['trajectory_accuracy']:.3f} "
            f"final_acc={v['final_only_accuracy']:.3f} stable={v['scramble_stability']:.3f} "
            f"transfer_err={v['mean_transfer_error']:.6f} recon_err={v['mean_source_reconstruction_error']:.6f} "
            f"min_margin={v['min_margin']:.6f}"
        )
    print(f"[{PHASE}] wrote trials: {OUT / 'phase38_cross_context_intervention_bridge_trials.csv'}")
    print(f"[{PHASE}] wrote summary: {OUT / 'phase38_cross_context_intervention_bridge_summary.json'}")
    print(f"[{PHASE}] wrote report: {OUT / 'phase38_cross_context_intervention_bridge_report.md'}")
    print(f"[{PHASE}] wrote intervention sweep: {OUT / 'phase38_context_intervention_sweep.csv'}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
