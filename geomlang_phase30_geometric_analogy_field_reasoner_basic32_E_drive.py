#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 30 — Geometric analogy field reasoner

Purpose:
    Continue the Phase 27 -> 28 -> 29 reset back toward BBIT's real goal:
    tokenless reasoning through geometry.

Conceptual progression:
    Phase 27:
        Recognize named geometric operators.

    Phase 28:
        Infer continuous whole-field deformation.

    Phase 29:
        Infer piecewise local geometric fields.

    Phase 30:
        Perform geometric analogy:
            A is to B
            as C is to ?

        The system receives:
            - source object A
            - transformed source object B
            - target object C
            - multiple candidate answers D_i

        It must infer the hidden local field from A -> B,
        transfer that field onto C,
        and choose the candidate that best completes the analogy.

    This is closer to "reasoning" than the earlier phases because the answer is
    not just recognition of a transform. It requires:
        1. inferring a hidden relation,
        2. carrying that relation across a different object,
        3. rejecting plausible distractors,
        4. doing all of this without words, tokens, labels, or symbolic rules
           inside the trial itself.

Outputs:
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_trials.csv
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_summary.json
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_report.md
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_accuracy.png
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_margins.png
    E:\\BBIT\\outputs_basic32\\phase30_geometric_analogy_transfer_error.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PHASE = "30"
TITLE = "Geometric analogy field reasoner"
FINGERPRINT = "phase30_geometric_analogy_field_reasoning"


# -----------------------------
# Path helpers
# -----------------------------

def find_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "bbit_geomlang":
        return here.parent.parent
    return Path.cwd().resolve()


def ensure_outputs(root: Path) -> Path:
    out = root / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------
# Geometry helpers
# -----------------------------

def normalize_points(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    scale = np.sqrt((q * q).sum(axis=1).mean())
    if scale < 1e-12:
        scale = 1.0
    return q / scale


def make_cloud(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Make an object-like 2D point cloud.

    The object is not a symbol. It is just a spatial body.
    Different trials receive different bodies, so the method cannot memorize
    a fixed template.
    """
    mode = rng.choice(["blob", "ring_blob", "lopsided", "crescent", "diamondish"])

    if mode == "blob":
        theta = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(0.02, 1.0, n))
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        p = np.stack([x, y], axis=1)

    elif mode == "ring_blob":
        theta = rng.uniform(0, 2 * np.pi, n)
        rad = rng.normal(0.75, 0.18, n)
        rad = np.clip(rad, 0.10, 1.15)
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        p = np.stack([x, y], axis=1)

    elif mode == "lopsided":
        theta = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(0.01, 1.0, n))
        x = rad * np.cos(theta) * rng.uniform(0.75, 1.25)
        y = rad * np.sin(theta) * rng.uniform(0.75, 1.25)
        p = np.stack([x, y], axis=1)
        p[:, 0] += 0.22 * np.sin(2.0 * p[:, 1])
        p[:, 1] += 0.10 * np.cos(3.0 * p[:, 0])

    elif mode == "crescent":
        theta = rng.uniform(-0.20 * np.pi, 1.35 * np.pi, n)
        rad = rng.normal(0.75, 0.12, n)
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        p = np.stack([x, y], axis=1)
        p[:, 0] += 0.25 * (p[:, 1] ** 2)

    else:
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        p = np.stack([u + 0.25 * np.sign(v) * np.abs(v), v], axis=1)

    # Add a small unique asymmetry so reflections/rotations cannot cheat.
    p[:, 0] += 0.05 * rng.normal(size=n)
    p[:, 1] += 0.05 * rng.normal(size=n)
    return normalize_points(p)


def rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric nearest-neighbor cloud distance.

    No point labels are used here. This is important:
    candidate answer scoring is based on spatial agreement, not token identity.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(0.5 * (np.mean(np.min(d2, axis=1)) + np.mean(np.min(d2, axis=0))))


def mean_point_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# -----------------------------
# Hidden partition families
# -----------------------------

FamilyFn = Callable[[np.ndarray], np.ndarray]


def labels_left_right(p: np.ndarray) -> np.ndarray:
    return (p[:, 0] >= 0.0).astype(int)


def labels_top_bottom(p: np.ndarray) -> np.ndarray:
    return (p[:, 1] >= 0.0).astype(int)


def labels_diag_pos(p: np.ndarray) -> np.ndarray:
    return ((p[:, 1] - p[:, 0]) >= 0.0).astype(int)


def labels_diag_neg(p: np.ndarray) -> np.ndarray:
    return ((p[:, 1] + p[:, 0]) >= 0.0).astype(int)


def labels_quadrants(p: np.ndarray) -> np.ndarray:
    x = (p[:, 0] >= 0.0).astype(int)
    y = (p[:, 1] >= 0.0).astype(int)
    return x + 2 * y


def labels_core_shell(p: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(p * p, axis=1))
    return (r >= np.median(r)).astype(int)


def labels_radial_three(p: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(p * p, axis=1))
    q1, q2 = np.quantile(r, [0.37, 0.70])
    return (r > q1).astype(int) + (r > q2).astype(int)


def labels_angle_three(p: np.ndarray) -> np.ndarray:
    ang = np.arctan2(p[:, 1], p[:, 0])
    z = ((ang + np.pi) / (2 * np.pi) * 3.0).astype(int)
    return np.clip(z, 0, 2)


FAMILIES: Dict[str, FamilyFn] = {
    "left_right": labels_left_right,
    "top_bottom": labels_top_bottom,
    "diag_pos": labels_diag_pos,
    "diag_neg": labels_diag_neg,
    "quadrants": labels_quadrants,
    "core_shell": labels_core_shell,
    "radial_three": labels_radial_three,
    "angle_three": labels_angle_three,
}


# -----------------------------
# Local affine fields
# -----------------------------

@dataclass
class LocalAffine:
    matrix: np.ndarray
    shift: np.ndarray


def random_local_affines(
    rng: np.random.Generator,
    family: str,
    labels: np.ndarray,
) -> Dict[int, LocalAffine]:
    """
    Create a hidden piecewise local deformation.

    The deformation is deliberately local and compositional:
    different regions of the same point cloud undergo different affine changes.
    """
    unique = sorted(int(x) for x in np.unique(labels))
    params: Dict[int, LocalAffine] = {}

    base_theta = rng.uniform(-0.30, 0.30)
    base_scale = rng.uniform(0.90, 1.12)
    base = base_scale * rotation(base_theta)

    for lab in unique:
        k = lab - (len(unique) - 1) / 2.0

        theta = base_theta + rng.uniform(-0.22, 0.22) + 0.08 * k
        sx = rng.uniform(0.88, 1.16) + 0.025 * k
        sy = rng.uniform(0.88, 1.16) - 0.020 * k
        shear_x = rng.uniform(-0.20, 0.20) + 0.03 * k
        shear_y = rng.uniform(-0.20, 0.20) - 0.02 * k

        m = rotation(theta) @ np.array([[sx, shear_x], [shear_y, sy]], dtype=np.float64)

        # Keep transformations moderate but distinguishable.
        m = 0.78 * m + 0.22 * base

        shift = np.array(
            [
                rng.uniform(-0.18, 0.18) + 0.04 * k,
                rng.uniform(-0.18, 0.18) - 0.03 * k,
            ],
            dtype=np.float64,
        )

        # Family-specific bias makes the hidden field more meaningfully local.
        if family == "left_right":
            shift[0] += 0.10 * (1 if lab else -1)
        elif family == "top_bottom":
            shift[1] += 0.10 * (1 if lab else -1)
        elif family == "diag_pos":
            shift += np.array([0.06 * k, -0.06 * k])
        elif family == "diag_neg":
            shift += np.array([0.06 * k, 0.06 * k])
        elif family == "core_shell":
            if lab == 0:
                m *= 0.94
            else:
                m *= 1.06
        elif family == "radial_three":
            m *= 0.94 + 0.06 * lab
        elif family == "angle_three":
            shift += 0.06 * np.array([math.cos(lab * 2.1), math.sin(lab * 2.1)])

        params[lab] = LocalAffine(matrix=m, shift=shift)

    return params


def apply_local_field(p: np.ndarray, family: str, params: Dict[int, LocalAffine]) -> np.ndarray:
    labels = FAMILIES[family](p)
    out = np.zeros_like(p, dtype=np.float64)

    for lab in np.unique(labels):
        lab_i = int(lab)
        mask = labels == lab_i
        aff = params[lab_i]
        out[mask] = p[mask] @ aff.matrix.T + aff.shift

    return normalize_points(out)


def fit_affine(x: np.ndarray, y: np.ndarray) -> LocalAffine:
    """
    Fit y ~= x @ M.T + t.
    """
    if len(x) < 3:
        return LocalAffine(matrix=np.eye(2), shift=np.zeros(2))

    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    # coef shape: 3 x 2
    m_t = coef[:2, :]
    t = coef[2, :]
    return LocalAffine(matrix=m_t.T, shift=t)


def infer_local_field(
    a: np.ndarray,
    b: np.ndarray,
    family: str,
) -> Tuple[Dict[int, LocalAffine], float]:
    """
    Infer a piecewise affine field from A -> B under a proposed hidden partition.
    """
    labels = FAMILIES[family](a)
    params: Dict[int, LocalAffine] = {}
    pred = np.zeros_like(b)

    for lab in np.unique(labels):
        lab_i = int(lab)
        mask = labels == lab_i
        aff = fit_affine(a[mask], b[mask])
        params[lab_i] = aff
        pred[mask] = a[mask] @ aff.matrix.T + aff.shift

    pred = normalize_points(pred)
    residual = mean_point_error(pred, b)
    return params, residual


def transfer_inferred_field(
    c: np.ndarray,
    family: str,
    params: Dict[int, LocalAffine],
) -> np.ndarray:
    labels = FAMILIES[family](c)
    out = np.zeros_like(c)

    for lab in np.unique(labels):
        lab_i = int(lab)
        mask = labels == lab_i

        if lab_i in params:
            aff = params[lab_i]
        else:
            # If a partition region is missing in source but appears in target,
            # fall back to the closest available region.
            nearest_key = min(params.keys(), key=lambda k: abs(k - lab_i))
            aff = params[nearest_key]

        out[mask] = c[mask] @ aff.matrix.T + aff.shift

    return normalize_points(out)


# -----------------------------
# Analogy trial
# -----------------------------

@dataclass
class TrialResult:
    trial: int
    true_family: str
    predicted_family: str
    correct: bool
    stable: bool
    best_distance: float
    runner_up_distance: float
    margin: float
    source_fit_residual: float
    transfer_error: float
    family_match: bool
    candidate_count: int


def make_distractors(
    rng: np.random.Generator,
    c: np.ndarray,
    true_family: str,
    true_params: Dict[int, LocalAffine],
    inferred_by_family: Dict[str, Tuple[Dict[int, LocalAffine], float]],
    true_d: np.ndarray,
    num_noise: int = 3,
) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []

    # Wrong family transfers.
    for fam, (params, _) in inferred_by_family.items():
        if fam == true_family:
            continue
        wrong = transfer_inferred_field(c, fam, params)
        wrong += rng.normal(0, 0.014, wrong.shape)
        candidates.append((f"wrong_family_{fam}", normalize_points(wrong)))

    # Perturbed true-family versions: plausible but not exact.
    for i in range(num_noise):
        perturbed: Dict[int, LocalAffine] = {}
        for lab, aff in true_params.items():
            noise_m = np.eye(2) + rng.normal(0, 0.035 + 0.012 * i, (2, 2))
            noise_t = rng.normal(0, 0.030 + 0.010 * i, 2)
            perturbed[lab] = LocalAffine(
                matrix=noise_m @ aff.matrix,
                shift=aff.shift + noise_t,
            )
        wrong = apply_local_field(c, true_family, perturbed)
        wrong += rng.normal(0, 0.012, wrong.shape)
        candidates.append((f"perturbed_true_{i}", normalize_points(wrong)))

    # Global affine distractor.
    all_c = c
    all_d = true_d
    global_aff = fit_affine(all_c, all_d)
    global_wrong = normalize_points(all_c @ global_aff.matrix.T + global_aff.shift)
    global_wrong += rng.normal(0, 0.018, global_wrong.shape)
    candidates.append(("global_affine_distractor", normalize_points(global_wrong)))

    return candidates


def solve_analogy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    candidates: List[Tuple[str, np.ndarray]],
) -> Tuple[str, str, float, float, float, float, Dict[str, float]]:
    """
    Solve A:B :: C:?

    The solver does not receive the true family.
    It searches possible hidden partitions, infers A->B for each partition,
    transfers the field to C, and chooses the candidate cloud nearest to the
    transferred prediction.
    """
    best_record = None
    family_residuals: Dict[str, float] = {}

    for fam in FAMILIES:
        params, residual = infer_local_field(a, b, fam)
        pred_d = transfer_inferred_field(c, fam, params)
        family_residuals[fam] = residual

        for cand_name, cand_cloud in candidates:
            d = chamfer(pred_d, cand_cloud)
            # The score primarily uses C->D answer distance, with a tiny source
            # residual penalty to break near ties in favor of better inferred fields.
            score = d + 0.03 * residual
            row = (score, d, fam, cand_name, residual)
            if best_record is None or row < best_record:
                best_record = row

    assert best_record is not None
    best_score, best_dist, best_fam, best_cand, best_residual = best_record

    all_scores = []
    for fam in FAMILIES:
        params, residual = infer_local_field(a, b, fam)
        pred_d = transfer_inferred_field(c, fam, params)
        for cand_name, cand_cloud in candidates:
            d = chamfer(pred_d, cand_cloud)
            score = d + 0.03 * residual
            all_scores.append((score, d, fam, cand_name, residual))

    all_scores.sort(key=lambda x: x[0])
    runner = all_scores[1]
    margin = float(runner[0] - all_scores[0][0])
    return best_fam, best_cand, float(best_dist), float(runner[1]), margin, float(best_residual), family_residuals


def run_trial(rng: np.random.Generator, trial: int, n_points: int) -> TrialResult:
    true_family = str(rng.choice(list(FAMILIES.keys())))

    a = make_cloud(rng, n_points)
    c = make_cloud(rng, n_points)

    labels_a = FAMILIES[true_family](a)
    true_params = random_local_affines(rng, true_family, labels_a)

    b = apply_local_field(a, true_family, true_params)
    true_d_clean = apply_local_field(c, true_family, true_params)

    # Small perceptual noise. The answer is not an exact pixel-perfect copy.
    b_noisy = normalize_points(b + rng.normal(0, 0.006, b.shape))
    true_d = normalize_points(true_d_clean + rng.normal(0, 0.006, true_d_clean.shape))

    inferred_by_family: Dict[str, Tuple[Dict[int, LocalAffine], float]] = {}
    for fam in FAMILIES:
        inferred_by_family[fam] = infer_local_field(a, b_noisy, fam)

    candidates = [("TRUE", true_d)]
    candidates.extend(make_distractors(rng, c, true_family, true_params, inferred_by_family, true_d))

    rng.shuffle(candidates)

    best_fam, best_cand, best_d, runner_d, margin, src_resid, family_residuals = solve_analogy(
        a=a,
        b=b_noisy,
        c=c,
        candidates=candidates,
    )

    correct = best_cand == "TRUE"
    family_match = best_fam == true_family
    transfer_error = chamfer(
        transfer_inferred_field(c, true_family, inferred_by_family[true_family][0]),
        true_d,
    )

    # Scramble stability:
    # reorder points inside every candidate and shuffle candidates again.
    scrambled_candidates = []
    for name, cloud in candidates:
        idx = rng.permutation(len(cloud))
        scrambled_candidates.append((name, cloud[idx]))
    rng.shuffle(scrambled_candidates)

    best_fam_2, best_cand_2, *_ = solve_analogy(
        a=a[rng.permutation(len(a))],
        b=b_noisy[rng.permutation(len(b_noisy))],
        c=c[rng.permutation(len(c))],
        candidates=scrambled_candidates,
    )

    # Note:
    # A and B point order scrambling breaks direct correspondence if scrambled
    # independently. To preserve the physical before/after correspondence, use a
    # stricter stability test only on candidate and target cloud ordering.
    #
    # Therefore recompute with A/B paired order preserved, but C/candidates
    # scrambled.
    pair_perm = rng.permutation(len(a))
    c_perm = rng.permutation(len(c))
    best_fam_3, best_cand_3, *_ = solve_analogy(
        a=a[pair_perm],
        b=b_noisy[pair_perm],
        c=c[c_perm],
        candidates=scrambled_candidates,
    )

    stable = (best_cand_3 == best_cand) and (best_fam_3 == best_fam)

    return TrialResult(
        trial=trial,
        true_family=true_family,
        predicted_family=best_fam,
        correct=correct,
        stable=stable,
        best_distance=best_d,
        runner_up_distance=runner_d,
        margin=margin,
        source_fit_residual=src_resid,
        transfer_error=transfer_error,
        family_match=family_match,
        candidate_count=len(candidates),
    )


# -----------------------------
# Reporting
# -----------------------------

def summarize(results: List[TrialResult]) -> Dict[str, object]:
    accuracy = float(np.mean([r.correct for r in results]))
    stability = float(np.mean([r.stable for r in results]))
    family_match_rate = float(np.mean([r.family_match for r in results]))
    mean_margin = float(np.mean([r.margin for r in results]))
    min_margin = float(np.min([r.margin for r in results]))
    mean_source_residual = float(np.mean([r.source_fit_residual for r in results]))
    mean_transfer_error = float(np.mean([r.transfer_error for r in results]))

    family_summary = {}
    for fam in FAMILIES:
        rows = [r for r in results if r.true_family == fam]
        if not rows:
            continue
        family_summary[fam] = {
            "trials": len(rows),
            "accuracy": float(np.mean([r.correct for r in rows])),
            "scramble_stability": float(np.mean([r.stable for r in rows])),
            "family_match_rate": float(np.mean([r.family_match for r in rows])),
            "mean_margin": float(np.mean([r.margin for r in rows])),
            "min_margin": float(np.min([r.margin for r in rows])),
            "mean_source_fit_residual": float(np.mean([r.source_fit_residual for r in rows])),
            "mean_transfer_error": float(np.mean([r.transfer_error for r in rows])),
        }

    pass_flag = (
        accuracy >= 0.96
        and stability >= 0.98
        and family_match_rate >= 0.88
        and mean_transfer_error <= 0.012
    )

    return {
        "phase": PHASE,
        "title": TITLE,
        "fingerprint": FINGERPRINT,
        "PHASE30_GEOMETRIC_ANALOGY_FIELD_REASONING_PASS": pass_flag,
        "trials": len(results),
        "accuracy": accuracy,
        "scramble_stability": stability,
        "family_match_rate": family_match_rate,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "mean_source_fit_residual": mean_source_residual,
        "mean_transfer_error": mean_transfer_error,
        "families": family_summary,
        "concept": {
            "input_form": "A, B, C, candidate_D_clouds",
            "task": "infer hidden local field A->B and transfer it to C",
            "answer_rule": "choose candidate D nearest to transferred geometric field",
            "tokenless_property": "candidate scoring uses point-cloud geometry rather than symbolic labels",
            "phase27_to_30_arc": [
                "Phase 27 recognized named operators.",
                "Phase 28 inferred continuous whole-field deformations.",
                "Phase 29 inferred piecewise local fields.",
                "Phase 30 transfers inferred local fields analogically: A:B :: C:?",
            ],
        },
    }


def write_trials_csv(path: Path, results: List[TrialResult]) -> None:
    fields = [
        "trial",
        "true_family",
        "predicted_family",
        "correct",
        "stable",
        "family_match",
        "best_distance",
        "runner_up_distance",
        "margin",
        "source_fit_residual",
        "transfer_error",
        "candidate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "trial": r.trial,
                "true_family": r.true_family,
                "predicted_family": r.predicted_family,
                "correct": int(r.correct),
                "stable": int(r.stable),
                "family_match": int(r.family_match),
                "best_distance": f"{r.best_distance:.10f}",
                "runner_up_distance": f"{r.runner_up_distance:.10f}",
                "margin": f"{r.margin:.10f}",
                "source_fit_residual": f"{r.source_fit_residual:.10f}",
                "transfer_error": f"{r.transfer_error:.10f}",
                "candidate_count": r.candidate_count,
            })


def write_report(path: Path, summary: Dict[str, object]) -> None:
    fams = summary["families"]
    assert isinstance(fams, dict)

    lines = []
    lines.append("# Phase 30 — Geometric analogy field reasoner")
    lines.append("")
    lines.append("## What this phase tests")
    lines.append("")
    lines.append("Phase 30 moves from local-field recognition into geometric analogy.")
    lines.append("")
    lines.append("The system receives a source relation:")
    lines.append("")
    lines.append("```text")
    lines.append("A -> B")
    lines.append("```")
    lines.append("")
    lines.append("Then it receives a new object:")
    lines.append("")
    lines.append("```text")
    lines.append("C -> ?")
    lines.append("```")
    lines.append("")
    lines.append("The solver must infer the hidden local deformation field from `A` to `B`, transfer that field onto `C`, and choose the candidate answer that best completes the analogy.")
    lines.append("")
    lines.append("This is meant to point BBIT back toward the actual goal: geometric thought rather than file packaging, route locking, or tool-chain machinery.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- Pass: `{summary['PHASE30_GEOMETRIC_ANALOGY_FIELD_REASONING_PASS']}`")
    lines.append(f"- Accuracy: `{summary['accuracy']:.4f}`")
    lines.append(f"- Scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- Family match rate: `{summary['family_match_rate']:.4f}`")
    lines.append(f"- Mean answer margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- Minimum answer margin: `{summary['min_margin']:.6f}`")
    lines.append(f"- Mean source fit residual: `{summary['mean_source_fit_residual']:.6f}`")
    lines.append(f"- Mean transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| Family | Trials | Accuracy | Stability | Family match | Mean margin | Transfer error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, row in fams.items():
        lines.append(
            f"| {fam} | {row['trials']} | {row['accuracy']:.3f} | "
            f"{row['scramble_stability']:.3f} | {row['family_match_rate']:.3f} | "
            f"{row['mean_margin']:.6f} | {row['mean_transfer_error']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A successful Phase 30 result means the system is no longer merely recognizing a transformation. It is using a relation learned from one geometric situation and transferring it to another. That is the beginning of analogy, and analogy is one of the bridges from perception into reasoning.")
    lines.append("")
    lines.append("In BBIT terms, the important claim is not that this is already intelligence. The important claim is that a non-token, non-language field can perform a primitive version of:")
    lines.append("")
    lines.append("```text")
    lines.append("this relation over here should become the same relation over there")
    lines.append("```")
    lines.append("")
    lines.append("That is closer to geometric thought.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(out: Path, results: List[TrialResult], summary: Dict[str, object]) -> None:
    if plt is None:
        return

    fams = summary["families"]
    assert isinstance(fams, dict)

    names = list(fams.keys())
    accs = [fams[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(names))
    ax.barh(y, accs)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy")
    ax.set_title("Phase 30 geometric analogy field reasoning: accuracy by hidden relation")
    for i, v in enumerate(accs):
        ax.text(min(v + 0.01, 1.02), i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "phase30_geometric_analogy_accuracy.png", dpi=150)
    plt.close(fig)

    margins = [r.margin for r in results]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(margins, bins=35)
    ax.set_xlabel("runner-up score - best score")
    ax.set_ylabel("trials")
    ax.set_title("Phase 30 answer margin distribution")
    fig.tight_layout()
    fig.savefig(out / "phase30_geometric_analogy_margins.png", dpi=150)
    plt.close(fig)

    errors = [r.transfer_error for r in results]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(errors, bins=35)
    ax.set_xlabel("Chamfer error: inferred C->? vs true C->D")
    ax.set_ylabel("trials")
    ax.set_title("Phase 30 transferred analogy field error distribution")
    fig.tight_layout()
    fig.savefig(out / "phase30_geometric_analogy_transfer_error.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=700)
    parser.add_argument("--points", type=int, default=72)
    parser.add_argument("--seed", type=int, default=303030)
    args = parser.parse_args(argv)

    root = find_root()
    out = ensure_outputs(root)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {out}")
    print(f"[{PHASE}] reset continued: from local fields to geometric analogy")
    print(f"[{PHASE}] task: infer A->B, transfer relation onto C, choose D")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    results: List[TrialResult] = []
    for i in range(args.trials):
        results.append(run_trial(rng, i, args.points))

    summary = summarize(results)

    trials_path = out / "phase30_geometric_analogy_trials.csv"
    summary_path = out / "phase30_geometric_analogy_summary.json"
    report_path = out / "phase30_geometric_analogy_report.md"

    write_trials_csv(trials_path, results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_path, summary)
    make_plots(out, results, summary)

    pass_flag = summary["PHASE30_GEOMETRIC_ANALOGY_FIELD_REASONING_PASS"]

    print(f"[{PHASE}] PHASE30_GEOMETRIC_ANALOGY_FIELD_REASONING_PASS={pass_flag}")
    print(
        f"[{PHASE}] accuracy={summary['accuracy']:.4f} "
        f"scramble_stability={summary['scramble_stability']:.4f} "
        f"trials={summary['trials']}"
    )
    print(
        f"[{PHASE}] family_match_rate={summary['family_match_rate']:.4f} "
        f"mean_source_fit_residual={summary['mean_source_fit_residual']:.6f} "
        f"mean_transfer_error={summary['mean_transfer_error']:.6f}"
    )

    print(f"[{PHASE}] family summary:")
    fams = summary["families"]
    assert isinstance(fams, dict)
    for fam, row in fams.items():
        print(
            f"  - {fam:<12} "
            f"acc={row['accuracy']:.3f} "
            f"stable={row['scramble_stability']:.3f} "
            f"family_match={row['family_match_rate']:.3f} "
            f"transfer_err={row['mean_transfer_error']:.6f} "
            f"min_margin={row['min_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote outputs to: {out}")

    return 0 if pass_flag else 2


if __name__ == "__main__":
    raise SystemExit(main())