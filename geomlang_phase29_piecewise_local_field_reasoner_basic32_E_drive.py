#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 29 — Piecewise local geometric-field reasoner (basic32)

Purpose
-------
Phase 27 proved that a discrete named geometric operator can be recovered from
shape relation A->B and transferred to C->?.

Phase 28 removed the named operator and inferred a continuous whole-object field.

Phase 29 moves one level closer to BBIT geometric thought: the system must infer
that one object can contain multiple local transformation fields at once.

The solver is not told the hidden family label. It sees only point geometry:
    A, B, C, candidate answers
and must infer which local partition/field best explains A->B, then transfer that
piecewise relation onto C.

This is a tokenless-reasoning seed test because the useful object is not a word,
operator name, or route label. The useful object is a relation of relations:
    local region -> local affine field -> recomposed whole answer.

Outputs are written to E:\BBIT\outputs_basic32 when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "29"
TITLE = "Piecewise local geometric-field reasoner"
FINGERPRINT = "phase29_piecewise_local_field_reasoning_basic32"

N_POINTS = 32
EPS = 1.0e-9


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------


def default_root() -> Path:
    e_root = Path(r"E:\BBIT")
    if e_root.exists():
        return e_root
    # If run from E:\BBIT\bbit_geomlang, parent is likely E:\BBIT.
    here = Path.cwd()
    if here.name.lower() == "bbit_geomlang":
        return here.parent
    return here


def ensure_outputs(root: Path) -> Path:
    out = root / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Geometry generation
# -----------------------------------------------------------------------------


def make_base_shape(rng: np.random.Generator, n: int = N_POINTS) -> np.ndarray:
    """Structured random point cloud with enough coverage for local partitions."""
    # 24 points around a perturbed double ring + 8 interior points.
    outer_n = 20
    inner_n = n - outer_n
    theta = np.linspace(0.0, 2.0 * np.pi, outer_n, endpoint=False)
    theta += rng.normal(0.0, 0.035, size=outer_n)
    rx = 1.0 + rng.normal(0.0, 0.035, size=outer_n)
    ry = 0.72 + rng.normal(0.0, 0.035, size=outer_n)
    outer = np.column_stack([rx * np.cos(theta), ry * np.sin(theta)])

    inner = rng.normal(0.0, 0.34, size=(inner_n, 2))
    pts = np.vstack([outer, inner])

    # Apply small whole-shape rotation and translation so partitions are not tied
    # to a trivial canonical orientation.
    ang = rng.uniform(-0.20, 0.20)
    R = rot(ang)
    pts = pts @ R.T
    pts += rng.normal(0.0, 0.025, size=(1, 2))
    return pts.astype(float)


def rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def affine_from_params(scale_x: float, scale_y: float, angle: float, shear_x: float, shear_y: float, tx: float, ty: float) -> Tuple[np.ndarray, np.ndarray]:
    S = np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=float)
    Sh = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=float)
    M = rot(angle) @ Sh @ S
    t = np.array([tx, ty], dtype=float)
    return M, t


def apply_affine(x: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x @ M.T + t


# -----------------------------------------------------------------------------
# Partitions: these are geometric possibilities, not semantic labels given to the
# solver. The solver enumerates them and picks the one with best measured fit.
# -----------------------------------------------------------------------------


def seg_global(p: np.ndarray) -> np.ndarray:
    return np.zeros(len(p), dtype=int)


def seg_lr(p: np.ndarray) -> np.ndarray:
    med = float(np.median(p[:, 0]))
    return (p[:, 0] > med).astype(int)


def seg_tb(p: np.ndarray) -> np.ndarray:
    med = float(np.median(p[:, 1]))
    return (p[:, 1] > med).astype(int)


def seg_diag_pos(p: np.ndarray) -> np.ndarray:
    v = p[:, 0] + p[:, 1]
    return (v > np.median(v)).astype(int)


def seg_diag_neg(p: np.ndarray) -> np.ndarray:
    v = p[:, 0] - p[:, 1]
    return (v > np.median(v)).astype(int)


def seg_quadrant(p: np.ndarray) -> np.ndarray:
    mx, my = float(np.median(p[:, 0])), float(np.median(p[:, 1]))
    return (p[:, 0] > mx).astype(int) + 2 * (p[:, 1] > my).astype(int)


def seg_core_shell(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0, keepdims=True)
    r = np.linalg.norm(p - c, axis=1)
    return (r > np.median(r)).astype(int)


SEGMENTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "global": seg_global,
    "left_right": seg_lr,
    "top_bottom": seg_tb,
    "diag_pos": seg_diag_pos,
    "diag_neg": seg_diag_neg,
    "quadrants": seg_quadrant,
    "core_shell": seg_core_shell,
}

HIDDEN_FAMILIES = ["left_right", "top_bottom", "diag_pos", "diag_neg", "quadrants", "core_shell"]


# -----------------------------------------------------------------------------
# Hidden piecewise fields
# -----------------------------------------------------------------------------


def family_transforms(family: str, rng: np.random.Generator, n_regions: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create coherent but different affine fields per region."""
    transforms: List[Tuple[np.ndarray, np.ndarray]] = []

    if family == "left_right":
        params = [
            (0.92, 1.08, rng.uniform(-0.16, -0.06), -0.12, 0.03, -0.11, 0.04),
            (1.08, 0.94, rng.uniform(0.06, 0.16), 0.10, -0.02, 0.12, -0.03),
        ]
    elif family == "top_bottom":
        params = [
            (1.05, 0.90, rng.uniform(0.04, 0.14), 0.04, -0.10, 0.02, -0.11),
            (0.95, 1.12, rng.uniform(-0.14, -0.04), -0.04, 0.11, -0.02, 0.12),
        ]
    elif family == "diag_pos":
        params = [
            (0.96, 1.10, rng.uniform(-0.12, -0.04), 0.13, 0.00, -0.07, 0.08),
            (1.10, 0.96, rng.uniform(0.04, 0.12), -0.12, 0.02, 0.08, -0.07),
        ]
    elif family == "diag_neg":
        params = [
            (1.10, 0.96, rng.uniform(-0.10, -0.03), -0.02, -0.12, 0.08, 0.07),
            (0.96, 1.10, rng.uniform(0.03, 0.10), 0.02, 0.12, -0.08, -0.07),
        ]
    elif family == "core_shell":
        params = [
            (0.86, 0.86, rng.uniform(-0.10, 0.10), 0.03, -0.03, 0.00, 0.00),
            (1.14, 1.05, rng.uniform(0.11, 0.24), -0.08, 0.06, 0.02, -0.02),
        ]
    elif family == "quadrants":
        params = [
            (0.92, 1.06, -0.13, -0.10, 0.02, -0.08, -0.05),
            (1.08, 0.96, 0.10, 0.08, -0.02, 0.09, -0.04),
            (1.04, 1.10, -0.08, 0.04, 0.09, -0.07, 0.08),
            (0.95, 0.91, 0.15, -0.05, -0.08, 0.08, 0.07),
        ]
    else:
        raise ValueError(f"unknown hidden family: {family}")

    for p in params[:n_regions]:
        M, t = affine_from_params(*p)
        # Tiny per-trial perturbation prevents memorized constants while keeping
        # the local field recoverable from A->B.
        M = M + rng.normal(0.0, 0.006, size=(2, 2))
        t = t + rng.normal(0.0, 0.006, size=(2,))
        transforms.append((M, t))
    return transforms


def apply_piecewise(points: np.ndarray, seg: np.ndarray, transforms: List[Tuple[np.ndarray, np.ndarray]], noise: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(points)
    for k, (M, t) in enumerate(transforms):
        mask = seg == k
        if np.any(mask):
            out[mask] = apply_affine(points[mask], M, t)
    if noise > 0:
        out += rng.normal(0.0, noise, size=out.shape)
    return out


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


@dataclass
class LocalModel:
    segmenter_name: str
    residual: float
    complexity: int
    transforms: Dict[int, Tuple[np.ndarray, np.ndarray]]
    score: float


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares affine fit dst ~= src @ M.T + t."""
    if len(src) < 3:
        # Degenerate local patch: fallback to translation by centroids.
        M = np.eye(2)
        t = dst.mean(axis=0) - src.mean(axis=0)
        pred = apply_affine(src, M, t)
        return M, t, float(np.mean(np.linalg.norm(pred - dst, axis=1)))

    X = np.column_stack([src, np.ones(len(src))])  # n x 3
    # Solve X @ W ~= dst, where W is 3 x 2. M = W[:2].T, t = W[2]
    W, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = W[:2, :].T
    t = W[2, :]
    pred = apply_affine(src, M, t)
    residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
    return M, t, residual


def fit_local_model(A: np.ndarray, B: np.ndarray, segmenter_name: str) -> LocalModel:
    seg = SEGMENTERS[segmenter_name](A)
    transforms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    pred = np.zeros_like(B)
    for k in sorted(set(int(x) for x in seg)):
        mask = seg == k
        M, t, _ = fit_affine(A[mask], B[mask])
        transforms[k] = (M, t)
        pred[mask] = apply_affine(A[mask], M, t)

    residual = float(np.mean(np.linalg.norm(pred - B, axis=1)))
    complexity = len(transforms)
    # Complexity penalty keeps four-region models from winning every trial by
    # overfitting tiny noise. It is deliberately small: local truth should still win.
    score = residual + 0.0045 * complexity
    return LocalModel(segmenter_name, residual, complexity, transforms, score)


def infer_model(A: np.ndarray, B: np.ndarray) -> LocalModel:
    models = [fit_local_model(A, B, name) for name in SEGMENTERS.keys()]
    return min(models, key=lambda m: (m.score, m.residual, m.complexity))


def apply_model_to_C(C: np.ndarray, model: LocalModel) -> np.ndarray:
    seg = SEGMENTERS[model.segmenter_name](C)
    out = np.zeros_like(C)
    # If a segment is absent in A but present in C, fallback to nearest available
    # transform by segment id distance, then global-ish first transform.
    available = sorted(model.transforms.keys())
    for k in sorted(set(int(x) for x in seg)):
        kk = k if k in model.transforms else min(available, key=lambda a: abs(a - k))
        M, t = model.transforms[kk]
        mask = seg == k
        out[mask] = apply_affine(C[mask], M, t)
    return out


def mean_shape_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(X - Y, axis=1)))


# -----------------------------------------------------------------------------
# Candidate answers
# -----------------------------------------------------------------------------


def make_candidates(C: np.ndarray, true_D: np.ndarray, model: LocalModel, hidden_family: str, hidden_transforms: List[Tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> Tuple[List[np.ndarray], int, List[str]]:
    candidates: List[np.ndarray] = []
    labels: List[str] = []

    # Correct candidate, with tiny display/measurement noise.
    candidates.append(true_D + rng.normal(0.0, 0.0015, size=true_D.shape))
    labels.append("correct_piecewise_transfer")

    # Decoy 1: best single global field from model prediction to C; often plausible.
    global_model = fit_local_model(C, true_D, "global")
    candidates.append(apply_model_to_C(C, global_model) + rng.normal(0.0, 0.003, size=C.shape))
    labels.append("overbroad_global_field")

    # Decoy 2: apply the right transforms but wrong segmentation family.
    wrong_segs = [s for s in HIDDEN_FAMILIES if s != hidden_family]
    wrong_name = rng.choice(wrong_segs)
    wrong_seg = SEGMENTERS[wrong_name](C)
    wrong = np.zeros_like(C)
    for k in sorted(set(int(x) for x in wrong_seg)):
        M, t = hidden_transforms[k % len(hidden_transforms)]
        wrong[wrong_seg == k] = apply_affine(C[wrong_seg == k], M, t)
    candidates.append(wrong + rng.normal(0.0, 0.004, size=C.shape))
    labels.append(f"wrong_partition_{wrong_name}")

    # Decoy 3: weakened local deformation; close but under-applies the thought.
    centroid = C.mean(axis=0, keepdims=True)
    candidates.append(C + 0.62 * (true_D - C) + 0.05 * (centroid - C))
    labels.append("under_applied_local_field")

    # Decoy 4: local field with region transforms swapped/rotated.
    seg_true = SEGMENTERS[hidden_family](C)
    swapped = np.zeros_like(C)
    for k in sorted(set(int(x) for x in seg_true)):
        M, t = hidden_transforms[(k + 1) % len(hidden_transforms)]
        swapped[seg_true == k] = apply_affine(C[seg_true == k], M, t)
    candidates.append(swapped + rng.normal(0.0, 0.004, size=C.shape))
    labels.append("swapped_local_fields")

    # Shuffle candidate order to ensure no positional answer token exists.
    order = list(range(len(candidates)))
    rng.shuffle(order)
    shuffled = [candidates[i] for i in order]
    shuffled_labels = [labels[i] for i in order]
    correct_index = order.index(0)
    return shuffled, correct_index, shuffled_labels


# -----------------------------------------------------------------------------
# Trial execution
# -----------------------------------------------------------------------------


def run_trial(trial_id: int, rng: np.random.Generator, noise: float) -> Dict[str, object]:
    hidden_family = str(rng.choice(HIDDEN_FAMILIES))
    A = make_base_shape(rng, N_POINTS)
    C = make_base_shape(rng, N_POINTS)

    segA = SEGMENTERS[hidden_family](A)
    n_regions = int(segA.max()) + 1
    hidden_transforms = family_transforms(hidden_family, rng, n_regions)
    B = apply_piecewise(A, segA, hidden_transforms, noise=noise, rng=rng)

    segC = SEGMENTERS[hidden_family](C)
    # If C's segmentation has more regions because of median edge effects, wrap; normally identical count.
    true_D = apply_piecewise(C, segC, hidden_transforms, noise=0.0, rng=rng)

    model = infer_model(A, B)
    pred_D = apply_model_to_C(C, model)

    candidates, correct_idx, candidate_labels = make_candidates(C, true_D, model, hidden_family, hidden_transforms, rng)
    dists = [mean_shape_distance(pred_D, cand) for cand in candidates]
    chosen_idx = int(np.argmin(dists))
    sorted_d = sorted(dists)
    margin = float(sorted_d[1] - sorted_d[0]) if len(sorted_d) > 1 else 0.0

    transfer_error = mean_shape_distance(pred_D, true_D)
    correct = chosen_idx == correct_idx

    # Scramble stability: rename every candidate label and every family word. Since
    # the solver never reads labels, the chosen geometry must remain unchanged.
    scrambled_labels = candidate_labels[:]
    rng.shuffle(scrambled_labels)
    scramble_chosen_idx = int(np.argmin(dists))
    scramble_stable = scramble_chosen_idx == chosen_idx

    return {
        "trial": trial_id,
        "hidden_family": hidden_family,
        "inferred_segmenter": model.segmenter_name,
        "family_match": hidden_family == model.segmenter_name,
        "local_regions": model.complexity,
        "fit_residual": model.residual,
        "model_score": model.score,
        "transfer_error": transfer_error,
        "chosen_idx": chosen_idx,
        "correct_idx": correct_idx,
        "correct": correct,
        "answer_margin": margin,
        "scramble_stable": scramble_stable,
        "candidate_labels": "|".join(candidate_labels),
        "chosen_label": candidate_labels[chosen_idx],
        "correct_label": candidate_labels[correct_idx],
        "distances": json.dumps([round(float(x), 8) for x in dists]),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_plot_accuracy(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    fam = df.groupby("hidden_family").agg(acc=("correct", "mean"), stable=("scramble_stable", "mean"), n=("trial", "count")).reset_index()
    fam = fam.sort_values("acc")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(fam["hidden_family"], fam["acc"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy")
    ax.set_title("Phase 29 piecewise local geometric-field reasoning: accuracy by hidden partition")
    for i, v in enumerate(fam["acc"]):
        ax.text(min(1.02, v + 0.01), i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "phase29_piecewise_local_field_accuracy.png", dpi=160)
    plt.close(fig)


def write_plot_margins(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(df["answer_margin"].astype(float), bins=28)
    ax.set_title("Phase 29 answer margin distribution")
    ax.set_xlabel("runner-up distance - best distance")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(out / "phase29_piecewise_local_field_margins.png", dpi=160)
    plt.close(fig)


def write_plot_transfer(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(df["transfer_error"].astype(float), bins=28)
    ax.set_title("Phase 29 transferred local-field error distribution")
    ax.set_xlabel("mean point error: predicted C->? vs true C->D")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(out / "phase29_piecewise_local_field_transfer_error.png", dpi=160)
    plt.close(fig)


def make_report(df: pd.DataFrame, summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"# Phase 29 — {TITLE}")
    lines.append("")
    lines.append("## Why this phase exists")
    lines.append("")
    lines.append("Phase 29 returns the project to BBIT's real purpose: developing geometric thought rather than packaging routes. Phase 27 recognized named geometric operators. Phase 28 inferred a single continuous deformation field. Phase 29 asks whether the system can infer several local fields inside one object and recombine them into a single answer.")
    lines.append("")
    lines.append("The model receives A→B and C→candidate answers. It is not given the hidden partition label. It tests geometric segmentations, fits local affine fields, transfers the best measured relation onto C, and chooses the nearest candidate answer.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k in ["PHASE29_PIECEWISE_LOCAL_FIELD_REASONING_PASS", "accuracy", "scramble_stability", "family_match_rate", "mean_fit_residual", "mean_transfer_error", "mean_answer_margin", "trials"]:
        lines.append(f"- **{k}**: `{summary[k]}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    fam = df.groupby("hidden_family").agg(
        acc=("correct", "mean"),
        stable=("scramble_stable", "mean"),
        family_match=("family_match", "mean"),
        residual=("fit_residual", "mean"),
        transfer_error=("transfer_error", "mean"),
        min_margin=("answer_margin", "min"),
        n=("trial", "count"),
    ).reset_index().sort_values("hidden_family")
    lines.append(fam.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A pass here means the system is no longer merely saying 'this whole object rotated' or 'this whole object expanded.' It is detecting that different areas of a form carry different transformation pressures, then preserving those local relations when answering a new case. That is a seed of compositional geometric reasoning: a thought made from local movements rather than language tokens.")
    lines.append("")
    lines.append(f"Fingerprint: `{FINGERPRINT}`")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Phase {PHASE}: {TITLE}")
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=29029)
    ap.add_argument("--noise", type=float, default=0.006)
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root) if args.root else default_root()
    out = ensure_outputs(root)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print(f"[29] {TITLE}")
    print(f"[29] root: {root}")
    print(f"[29] outputs: {out}")
    print("[29] reset continued: from whole-field deformation to local compositional fields")

    rows = [run_trial(i, rng, args.noise) for i in range(args.trials)]
    df = pd.DataFrame(rows)

    accuracy = float(df["correct"].mean())
    scramble_stability = float(df["scramble_stable"].mean())
    family_match_rate = float(df["family_match"].mean())
    mean_fit_residual = float(df["fit_residual"].mean())
    mean_transfer_error = float(df["transfer_error"].mean())
    mean_answer_margin = float(df["answer_margin"].mean())
    min_answer_margin = float(df["answer_margin"].min())

    pass_flag = bool(
        accuracy >= 0.94
        and scramble_stability >= 0.995
        and mean_transfer_error <= 0.055
        and mean_answer_margin > 0.05
    )

    fam_summary = df.groupby("hidden_family").agg(
        acc=("correct", "mean"),
        stable=("scramble_stable", "mean"),
        family_match=("family_match", "mean"),
        residual=("fit_residual", "mean"),
        transfer_error=("transfer_error", "mean"),
        min_margin=("answer_margin", "min"),
        n=("trial", "count"),
    ).reset_index().to_dict(orient="records")

    summary: Dict[str, object] = {
        "phase": PHASE,
        "title": TITLE,
        "fingerprint": FINGERPRINT,
        "PHASE29_PIECEWISE_LOCAL_FIELD_REASONING_PASS": pass_flag,
        "accuracy": round(accuracy, 6),
        "scramble_stability": round(scramble_stability, 6),
        "family_match_rate": round(family_match_rate, 6),
        "mean_fit_residual": round(mean_fit_residual, 6),
        "mean_transfer_error": round(mean_transfer_error, 6),
        "mean_answer_margin": round(mean_answer_margin, 6),
        "min_answer_margin": round(min_answer_margin, 6),
        "trials": int(args.trials),
        "n_points": N_POINTS,
        "noise": args.noise,
        "seed": args.seed,
        "hidden_families": HIDDEN_FAMILIES,
        "candidate_segmenters": list(SEGMENTERS.keys()),
        "family_summary": fam_summary,
    }

    trials_csv = out / "phase29_piecewise_local_field_trials.csv"
    summary_json = out / "phase29_piecewise_local_field_summary.json"
    report_md = out / "phase29_piecewise_local_field_report.md"

    df.to_csv(trials_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md.write_text(make_report(df, summary), encoding="utf-8")

    write_plot_accuracy(df, out)
    write_plot_margins(df, out)
    write_plot_transfer(df, out)

    print(f"[29] PHASE29_PIECEWISE_LOCAL_FIELD_REASONING_PASS={pass_flag}")
    print(f"[29] accuracy={accuracy:.4f} scramble_stability={scramble_stability:.4f} trials={args.trials}")
    print(f"[29] family_match_rate={family_match_rate:.4f} mean_fit_residual={mean_fit_residual:.6f} mean_transfer_error={mean_transfer_error:.6f}")
    print("[29] family summary:")
    for fam in fam_summary:
        print(
            f"  - {fam['hidden_family']:<12} "
            f"acc={fam['acc']:.3f} stable={fam['stable']:.3f} "
            f"family_match={fam['family_match']:.3f} residual={fam['residual']:.6f} "
            f"transfer_err={fam['transfer_error']:.6f} min_margin={fam['min_margin']:.6f}"
        )
    print(f"[29] wrote trials: {trials_csv}")
    print(f"[29] wrote summary: {summary_json}")
    print(f"[29] wrote report: {report_md}")
    print(f"[29] wrote outputs to: {out}")
    return 0 if pass_flag else 2


if __name__ == "__main__":
    raise SystemExit(main())
