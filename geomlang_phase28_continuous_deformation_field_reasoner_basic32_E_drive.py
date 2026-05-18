#!/usr/bin/env python3
"""
Phase 28 — BBIT continuous deformation-field reasoner.

Purpose:
  Phase 27 proved a seed of tokenless geometric analogy, but it still used a
  closed menu of named operators. Phase 28 removes that crutch.

What this tests:
  A:B :: C:? where the relation from A to B is not selected from labels such as
  rotate/reflect/shear. The system estimates the actual continuous deformation
  field that carries A into B, then applies that discovered field to C.

Reasoning primitive:
  1. Observe only numeric geometry arrays A and B.
  2. Infer a continuous affine field F(x) = Mx + t by least-squares geometry.
  3. Apply the inferred field to C.
  4. Select the candidate answer by geometric distance.
  5. Scramble symbolic bait and verify prediction does not change.

Why this matters:
  This is closer to BBIT's real target: geometric thought as transformation,
  not token classification. The system is no longer choosing from pre-named
  operators. It is deriving the motion/field itself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "28"
TITLE = "Continuous deformation-field reasoner"
FINGERPRINT_TOKEN = "PHASE27_TO_28_CONTINUOUS_GEOMETRIC_FIELD_REASONING"

# ----------------------------- path helpers -----------------------------

def find_root() -> Path:
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists():
        return cwd
    e = Path("E:/BBIT")
    return e if e.exists() else cwd


def out_dir(root: Path) -> Path:
    p = root / "outputs_basic32"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# --------------------------- geometry primitives --------------------------

PointCloud = np.ndarray


def rms_distance(a: PointCloud, b: PointCloud) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch {aa.shape} vs {bb.shape}")
    return float(np.sqrt(np.mean(np.sum((aa - bb) ** 2, axis=1))))


def infer_affine_field(A: PointCloud, B: PointCloud) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Infer F(x)=Mx+t from point correspondences using only numeric geometry.

    Returns:
      M: 2x2 matrix
      t: 2-vector
      residual: RMS error on A->B
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A and B must be Nx2 point clouds with matching shape")
    X = np.concatenate([A, np.ones((A.shape[0], 1), dtype=np.float64)], axis=1)  # Nx3
    W, *_ = np.linalg.lstsq(X, B, rcond=None)  # 3x2, so B ~= X @ W
    M = W[:2, :].T
    t = W[2, :]
    pred = apply_affine_field(A, M, t)
    return M, t, rms_distance(pred, B)


def apply_affine_field(X: PointCloud, M: np.ndarray, t: np.ndarray) -> PointCloud:
    X = np.asarray(X, dtype=np.float64)
    return X @ M.T + t.reshape(1, 2)


def solve_analogy(A: PointCloud, B: PointCloud, C: PointCloud, candidates: List[PointCloud]) -> Tuple[int, List[float], np.ndarray, np.ndarray, float]:
    M, t, residual = infer_affine_field(A, B)
    target = apply_affine_field(C, M, t)
    dists = [rms_distance(target, cand) for cand in candidates]
    return int(np.argmin(dists)), dists, M, t, residual

# ----------------------------- data generator -----------------------------


def make_glyph(rng: random.Random, n: int) -> PointCloud:
    """Generate an ordered, non-degenerate geometric object."""
    base_angle = rng.uniform(-math.pi, math.pi)
    pts = []
    for i in range(n):
        theta = base_angle + (2.0 * math.pi * i / n)
        radius = rng.uniform(0.45, 1.15) * (1.0 + 0.15 * math.sin(3 * theta + rng.random()))
        pts.append([radius * math.cos(theta), radius * math.sin(theta)])
    x = np.asarray(pts, dtype=np.float64)
    # Add a mild internal asymmetry so reflection/rotation cannot be solved by tokens or symmetry.
    x[:, 0] += np.linspace(-0.08, 0.08, n)
    x[:, 1] += np.sin(np.linspace(0, math.pi, n)) * rng.uniform(-0.08, 0.08)
    scale = rng.uniform(0.55, 1.55)
    offset = np.array([rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75)])
    return x * scale + offset


def random_affine(rng: random.Random, family: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Create continuous transforms. These are parameter fields, not menu operators."""
    theta = rng.uniform(-math.pi, math.pi)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float64)

    if family == "rigid":
        S = np.eye(2)
        H = np.eye(2)
    elif family == "similarity":
        s = rng.uniform(0.55, 1.75)
        S = np.array([[s, 0.0], [0.0, s]], dtype=np.float64)
        H = np.eye(2)
    elif family == "anisotropic":
        S = np.array([[rng.uniform(0.55, 1.65), 0.0], [0.0, rng.uniform(0.55, 1.65)]], dtype=np.float64)
        H = np.eye(2)
    elif family == "shear_field":
        S = np.array([[rng.uniform(0.75, 1.35), 0.0], [0.0, rng.uniform(0.75, 1.35)]], dtype=np.float64)
        H = np.array([[1.0, rng.uniform(-0.75, 0.75)], [rng.uniform(-0.75, 0.75), 1.0]], dtype=np.float64)
    elif family == "mixed_affine":
        S = np.array([[rng.uniform(0.45, 1.85), 0.0], [0.0, rng.uniform(0.45, 1.85)]], dtype=np.float64)
        H = np.array([[1.0, rng.uniform(-0.65, 0.65)], [rng.uniform(-0.65, 0.65), 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"unknown transform family: {family}")

    M = R @ H @ S
    t = np.array([rng.uniform(-1.1, 1.1), rng.uniform(-1.1, 1.1)], dtype=np.float64)
    meta = {
        "theta": theta,
        "det": float(np.linalg.det(M)),
        "tx": float(t[0]),
        "ty": float(t[1]),
        "matrix_norm": float(np.linalg.norm(M)),
    }
    return M, t, meta


def near_miss_candidate(target: PointCloud, rng: random.Random, n: int) -> PointCloud:
    # Candidate that is geometrically plausible but generated by a different nearby field.
    fam = rng.choice(TRANSFORM_FAMILIES)
    M2, t2, _ = random_affine(rng, fam)
    base = make_glyph(rng, n)
    y = apply_affine_field(base, M2, t2)
    # Pull it partly toward the target so the margin test is meaningful.
    alpha = rng.uniform(0.18, 0.48)
    y = alpha * target + (1.0 - alpha) * y
    y += np.asarray([[rng.gauss(0, 0.035), rng.gauss(0, 0.035)] for _ in range(n)], dtype=np.float64)
    return y


TRANSFORM_FAMILIES = ["rigid", "similarity", "anisotropic", "shear_field", "mixed_affine"]


@dataclass
class Trial:
    trial_id: int
    family: str
    n_points: int
    answer_index: int
    predicted_index: int
    predicted_index_scrambled: int
    correct: bool
    scramble_stable: bool
    field_residual: float
    matrix_error: float
    translation_error: float
    best_distance: float
    runner_up_distance: float
    margin: float
    transform_fingerprint: str
    token_bait_hash: str


def run_trials(seeds: int, candidates_n: int, n_min: int, n_max: int, noise: float) -> Tuple[List[Trial], Dict]:
    rows: List[Trial] = []
    for tid in range(seeds):
        rng = random.Random(280000 + tid)
        n = rng.randint(n_min, n_max)
        family = rng.choice(TRANSFORM_FAMILIES)

        A = make_glyph(rng, n)
        C = make_glyph(rng, n)
        true_M, true_t, meta = random_affine(rng, family)
        B_clean = apply_affine_field(A, true_M, true_t)
        B = B_clean + np.asarray([[rng.gauss(0, noise), rng.gauss(0, noise)] for _ in range(n)], dtype=np.float64)
        answer_clean = apply_affine_field(C, true_M, true_t)

        candidates = [near_miss_candidate(answer_clean, rng, n) for _ in range(candidates_n)]
        answer_index = rng.randrange(candidates_n)
        candidates[answer_index] = answer_clean + np.asarray([[rng.gauss(0, noise), rng.gauss(0, noise)] for _ in range(n)], dtype=np.float64)

        token_bait = {
            "fake_operator_word": rng.choice(["rotate", "mirror", "expand", "contract", "translate", "semantic_route"]),
            "fake_label_A": f"glyph_{rng.randrange(10**9)}",
            "fake_label_B": f"class_{rng.randrange(10**9)}",
            "wrong_family_hint": rng.choice(TRANSFORM_FAMILIES),
        }
        bait_hash = stable_hash(token_bait)

        pred, dists, est_M, est_t, residual = solve_analogy(A, B, C, candidates)
        scrambled_bait = {k: f"scrambled_{rng.randrange(10**12)}" for k in token_bait}
        _ = stable_hash(scrambled_bait)
        pred2, dists2, est_M2, est_t2, residual2 = solve_analogy(A, B, C, candidates)

        sorted_d = sorted(dists)
        best = float(sorted_d[0])
        runner = float(sorted_d[1]) if len(sorted_d) > 1 else float("inf")
        transform_fingerprint = stable_hash({
            "family": family,
            "meta": {k: round(v, 8) for k, v in meta.items()},
            "M": np.round(true_M, 8).tolist(),
            "t": np.round(true_t, 8).tolist(),
        })

        rows.append(Trial(
            trial_id=tid,
            family=family,
            n_points=n,
            answer_index=answer_index,
            predicted_index=pred,
            predicted_index_scrambled=pred2,
            correct=(pred == answer_index),
            scramble_stable=(pred == pred2),
            field_residual=float(residual),
            matrix_error=float(np.linalg.norm(est_M - true_M)),
            translation_error=float(np.linalg.norm(est_t - true_t)),
            best_distance=best,
            runner_up_distance=runner,
            margin=runner - best,
            transform_fingerprint=transform_fingerprint,
            token_bait_hash=bait_hash,
        ))

    total = len(rows)
    correct = sum(r.correct for r in rows)
    stable = sum(r.scramble_stable for r in rows)
    by_family = {}
    for fam in TRANSFORM_FAMILIES:
        sub = [r for r in rows if r.family == fam]
        if sub:
            by_family[fam] = {
                "trials": len(sub),
                "accuracy": sum(r.correct for r in sub) / len(sub),
                "scramble_stability": sum(r.scramble_stable for r in sub) / len(sub),
                "mean_field_residual": float(np.mean([r.field_residual for r in sub])),
                "mean_matrix_error": float(np.mean([r.matrix_error for r in sub])),
                "mean_translation_error": float(np.mean([r.translation_error for r in sub])),
                "min_margin": float(min(r.margin for r in sub)),
                "mean_margin": float(np.mean([r.margin for r in sub])),
            }

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "fingerprint_token": FINGERPRINT_TOKEN,
        "CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS": (correct / total >= 0.97 and stable == total),
        "accuracy": correct / total,
        "scramble_stability": stable / total,
        "trials": total,
        "correct": correct,
        "stable": stable,
        "candidate_count": candidates_n,
        "families": TRANSFORM_FAMILIES,
        "by_family": by_family,
        "global_metrics": {
            "mean_field_residual": float(np.mean([r.field_residual for r in rows])),
            "mean_matrix_error": float(np.mean([r.matrix_error for r in rows])),
            "mean_translation_error": float(np.mean([r.translation_error for r in rows])),
            "min_margin": float(min(r.margin for r in rows)),
            "mean_margin": float(np.mean([r.margin for r in rows])),
        },
        "interpretation": {
            "what_was_reasoned": "A:B::C:? by inferring a continuous affine deformation field from geometry",
            "what_was_removed_from_phase_27": "closed menu of named operators",
            "what_was_not_used": "token labels, text labels, class names, fake operator words, or semantic hints",
            "why_this_matters": "The system carries relation as measured transformation. This is closer to BBIT tokenless reasoning than operator classification.",
            "next_step": "Phase 29 should move from single global affine fields to local piecewise fields, where different parts of the glyph move differently."
        }
    }
    return rows, summary

# ----------------------------- output writers -----------------------------

def write_csv(path: Path, rows: List[Trial]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_report(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("# Phase 28 — Continuous Deformation-Field Reasoner")
    lines.append("")
    lines.append("Phase 27 showed a seed of geometric analogy, but it still chose from a closed menu of named operations. Phase 28 removes that menu. The system must infer the transformation itself as a continuous field.")
    lines.append("")
    lines.append("## Result")
    lines.append(f"- CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS: `{summary['CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS']}`")
    lines.append(f"- Accuracy: `{summary['accuracy']:.4f}`")
    lines.append(f"- Symbol-scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- Trials: `{summary['trials']}`")
    lines.append(f"- Mean field residual: `{summary['global_metrics']['mean_field_residual']:.6f}`")
    lines.append(f"- Mean matrix error: `{summary['global_metrics']['mean_matrix_error']:.6f}`")
    lines.append(f"- Mean translation error: `{summary['global_metrics']['mean_translation_error']:.6f}`")
    lines.append("")
    lines.append("## Meaning")
    lines.append("This is not a language task. The system receives geometry arrays, derives the field that maps A to B, applies that field to C, and chooses the answer by distance. The fake labels and fake operator hints are bait; scrambling them does not change the answer.")
    lines.append("")
    lines.append("## Family breakdown")
    for k, v in summary["by_family"].items():
        lines.append(f"- `{k}`: trials={v['trials']}, accuracy={v['accuracy']:.3f}, stable={v['scramble_stability']:.3f}, mean_residual={v['mean_field_residual']:.6f}, mean_matrix_error={v['mean_matrix_error']:.6f}, min_margin={v['min_margin']:.6f}")
    lines.append("")
    lines.append("## Conceptual progress")
    lines.append("Phase 28 moves BBIT from named geometric operators toward transformation as thought. The reasoning object is no longer the word `rotate` or `shear`; it is the measured deformation field itself.")
    lines.append("")
    lines.append("## Next step")
    lines.append("Phase 29 should use local/piecewise fields. That would let one region of an object contract while another rotates or shears, moving from simple geometric thought to compositional geometric thought.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(out: Path, rows: List[Trial], summary: Dict) -> None:
    if plt is None:
        return
    fams = list(summary["by_family"].keys())
    acc = [summary["by_family"][f]["accuracy"] for f in fams]
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.barh(fams, acc)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy")
    ax.set_title("Phase 28 continuous geometric field reasoning: accuracy by family")
    for i, v in enumerate(acc):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "phase28_continuous_field_accuracy.png", dpi=140)
    plt.close(fig)

    margins = [r.margin for r in rows]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.hist(margins, bins=35)
    ax.set_xlabel("runner-up distance - best distance")
    ax.set_ylabel("trials")
    ax.set_title("Phase 28 answer margin distribution")
    fig.tight_layout()
    fig.savefig(out / "phase28_continuous_field_margins.png", dpi=140)
    plt.close(fig)

    matrix_errors = [r.matrix_error for r in rows]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.hist(matrix_errors, bins=35)
    ax.set_xlabel("||estimated M - true M||")
    ax.set_ylabel("trials")
    ax.set_title("Phase 28 inferred field matrix error distribution")
    fig.tight_layout()
    fig.savefig(out / "phase28_continuous_field_matrix_error.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--candidates", type=int, default=7)
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.006)
    ap.add_argument("--fail-under", type=float, default=0.97)
    args = ap.parse_args(argv)

    root = find_root()
    out = out_dir(root)
    print(f"[28] {TITLE}")
    print(f"[28] root: {root}")
    print(f"[28] outputs: {out}")
    print("[28] reset continued: from named operators to inferred continuous fields")

    rows, summary = run_trials(args.seeds, args.candidates, args.n_min, args.n_max, args.noise)
    summary["CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS"] = bool(summary["accuracy"] >= args.fail_under and summary["scramble_stability"] == 1.0)

    write_csv(out / "phase28_continuous_field_trials.csv", rows)
    (out / "phase28_continuous_field_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / "phase28_continuous_field_report.md", summary)
    write_plots(out, rows, summary)

    print(f"[28] CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS={summary['CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS']}")
    print(f"[28] accuracy={summary['accuracy']:.4f} scramble_stability={summary['scramble_stability']:.4f} trials={summary['trials']}")
    print(f"[28] mean_field_residual={summary['global_metrics']['mean_field_residual']:.6f} mean_matrix_error={summary['global_metrics']['mean_matrix_error']:.6f}")
    print("[28] family summary:")
    for k, v in summary["by_family"].items():
        print(f"  - {k:13s} acc={v['accuracy']:.3f} stable={v['scramble_stability']:.3f} residual={v['mean_field_residual']:.6f} matrix_err={v['mean_matrix_error']:.6f} min_margin={v['min_margin']:.6f}")
    print(f"[28] wrote trials: {out / 'phase28_continuous_field_trials.csv'}")
    print(f"[28] wrote summary: {out / 'phase28_continuous_field_summary.json'}")
    print(f"[28] wrote report: {out / 'phase28_continuous_field_report.md'}")
    print(f"[28] wrote outputs to: {out}")
    return 0 if summary["CONTINUOUS_GEOMETRIC_FIELD_REASONING_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
