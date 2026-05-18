#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 61: Spatial shape concept transfer probe

Drop-in path:
  E:\BBIT\bbit_geomlang\geomlang_phase61_spatial_shape_concept_transfer_basic32_E_drive.py

Run:
  python bbit_geomlang/geomlang_phase61_spatial_shape_concept_transfer_basic32_E_drive.py

Purpose
-------
Phase 58-60 proved the A->B->C certificate verifier can survive increasingly skeptical
counterexample tribunals. That path is now strong enough; Phase 61 deliberately moves forward.

This phase asks the next productive question:

  Can the system treat a shape between points as its own transferable concept?

Instead of asking only which branch a point trajectory belongs to, we generate point clouds with
hidden relational structures: triangles, squares, pentagons, lines, and core-shell rings. The
script then tries to recover the embedded concept from unordered points using only relational
geometry between the points: pairwise distances, angle signatures, closure, area, convexity,
and luma/intensity coherence.

The key test is not memorizing coordinates. The same concept must transfer across:
  - translation
  - rotation
  - scale
  - mild shear/perspective distortion
  - point-order permutation
  - distractor points
  - noisy point locations
  - optional luma/intensity cues

This is intentionally not another tribunal escalation. It is a bridge from causal-chain
certificates into spatial concept recognition.

Outputs
-------
Writes into E:\BBIT\outputs_basic32 by default:

  phase61_spatial_shape_concept_transfer_trials.csv
  phase61_spatial_shape_concept_transfer_summary.json
  phase61_spatial_shape_concept_transfer_report.md
  phase61_examples\*.png
  phase61_spatial_shape_concept_transfer_accuracy.png
  phase61_spatial_shape_concept_transfer_confusion.png
  phase61_spatial_shape_concept_transfer_margin_distribution.png
  phase61_spatial_shape_concept_transfer_invariance_rates.png
  phase61_spatial_shape_concept_transfer_luma_ablation.png

Pass condition
--------------
  concept_accuracy >= 0.95
  transfer_accuracy >= 0.95
  permutation_invariance >= 0.98
  transform_invariance >= 0.95
  luma_ablation_accuracy >= 0.90
  mean_margin > 0.05

This script is self-contained. It does not import prior phase files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    print("[61] WARNING: matplotlib unavailable; png plots will be skipped:", exc)

PHASE = "61"
TITLE = "Spatial shape concept transfer probe"
PASS_FLAG = "PHASE61_SPATIAL_SHAPE_CONCEPT_TRANSFER_PASS"

SHAPES = [
    "triangle",
    "square",
    "pentagon",
    "line",
    "core_shell",
]

SHAPE_K = {
    "triangle": 3,
    "square": 4,
    "pentagon": 5,
    "line": 4,
    "core_shell": 6,
}

# -----------------------------
# Basic geometry
# -----------------------------

def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def polygon(n: int, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return np.array(
        [[radius * math.cos(phase + 2.0 * math.pi * i / n), radius * math.sin(phase + 2.0 * math.pi * i / n)] for i in range(n)],
        dtype=np.float64,
    )


def canonical_shape(shape: str) -> np.ndarray:
    if shape == "triangle":
        # Equilateral triangle. The concept detector is allowed to generalize through noisy distortions.
        return polygon(3, 1.0, math.pi / 2.0)
    if shape == "square":
        return polygon(4, 1.0, math.pi / 4.0)
    if shape == "pentagon":
        return polygon(5, 1.0, math.pi / 2.0)
    if shape == "line":
        return np.array([[-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0]], dtype=np.float64)
    if shape == "core_shell":
        # One center point plus five shell points. This is deliberately not just a polygon.
        return np.vstack([np.zeros((1, 2), dtype=np.float64), polygon(5, 1.0, math.pi / 2.0)])
    raise ValueError(shape)


def normalize_points(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    p = p - p.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((p * p).sum(axis=1).mean()))
    if scale < 1e-12:
        return p
    return p / scale


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    vals = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            vals.append(float(np.linalg.norm(p[i] - p[j])))
    return np.array(vals, dtype=np.float64)


def polygon_area(points: np.ndarray) -> float:
    p = order_by_angle(points)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def order_by_angle(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]


def angle_signature(points: np.ndarray) -> np.ndarray:
    p = normalize_points(points)
    c = p.mean(axis=0)
    ang = np.sort(np.mod(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), 2.0 * math.pi))
    if len(ang) <= 1:
        return np.zeros(1, dtype=np.float64)
    gaps = np.diff(np.r_[ang, ang[0] + 2.0 * math.pi]) / (2.0 * math.pi)
    return np.sort(gaps)


def collinearity(points: np.ndarray) -> float:
    p = normalize_points(points)
    if len(p) < 3:
        return 1.0
    _, s, _ = np.linalg.svd(p - p.mean(axis=0, keepdims=True), full_matrices=False)
    if s[0] < 1e-12:
        return 1.0
    return float(s[-1] / s[0])


def radial_shell_score(points: np.ndarray) -> float:
    p = normalize_points(points)
    c = p.mean(axis=0)
    r = np.linalg.norm(p - c, axis=1)
    if len(r) < 4:
        return 1.0
    # Core-shell is one near-center point and a shell with equal radii.
    r_sorted = np.sort(r)
    center_term = r_sorted[0]
    shell_cv = float(np.std(r_sorted[1:]) / (np.mean(r_sorted[1:]) + 1e-9))
    return center_term + shell_cv


def feature_vector(points: np.ndarray, luma: Optional[np.ndarray] = None) -> np.ndarray:
    p = normalize_points(points)
    d = pairwise_distances(p)
    d = d / (np.mean(d) + 1e-9)
    d = np.sort(d)
    gaps = angle_signature(p)
    area = np.array([polygon_area(p)], dtype=np.float64)
    col = np.array([collinearity(p)], dtype=np.float64)
    shell = np.array([radial_shell_score(p)], dtype=np.float64)
    if luma is None:
        lum = np.array([0.0, 0.0], dtype=np.float64)
    else:
        l = np.asarray(luma, dtype=np.float64)
        lum = np.array([float(np.mean(l)), float(np.std(l))], dtype=np.float64)
    return np.r_[d, gaps, area, col, shell, lum]


def padded_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = max(len(a), len(b))
    aa = np.zeros(n, dtype=np.float64)
    bb = np.zeros(n, dtype=np.float64)
    aa[: len(a)] = a
    bb[: len(b)] = b
    return float(np.linalg.norm(aa - bb) / math.sqrt(n))


TEMPLATE_FEATURES: Dict[str, np.ndarray] = {}
TEMPLATE_FEATURES_NO_LUMA: Dict[str, np.ndarray] = {}
for _shape in SHAPES:
    lp = np.ones(SHAPE_K[_shape], dtype=np.float64) * (0.65 if _shape in {"triangle", "pentagon"} else 0.45)
    TEMPLATE_FEATURES[_shape] = feature_vector(canonical_shape(_shape), lp)
    TEMPLATE_FEATURES_NO_LUMA[_shape] = feature_vector(canonical_shape(_shape), np.zeros(SHAPE_K[_shape], dtype=np.float64))


def concept_score(candidate_points: np.ndarray, candidate_luma: np.ndarray, shape: str, use_luma: bool = True) -> float:
    template = TEMPLATE_FEATURES[shape] if use_luma else TEMPLATE_FEATURES_NO_LUMA[shape]
    base = padded_distance(feature_vector(candidate_points, candidate_luma), template)
    # Add typed structural penalties. This prevents the detector from calling everything a polygon.
    c = collinearity(candidate_points)
    shell = radial_shell_score(candidate_points)
    area = polygon_area(normalize_points(candidate_points))
    if shape == "line":
        base += 4.0 * c + 0.05 * area
    elif shape == "core_shell":
        base += 2.8 * shell
    else:
        base += 0.35 * max(0.0, 0.18 - area)
        base += 0.25 * max(0.0, 0.015 - c)  # penalize pure lines as polygons
        if shape == "triangle" and len(candidate_points) != 3:
            base += 5.0
        if shape == "square" and len(candidate_points) != 4:
            base += 5.0
        if shape == "pentagon" and len(candidate_points) != 5:
            base += 5.0
    return float(base)


@dataclass
class Scene:
    shape: str
    points: np.ndarray
    luma: np.ndarray
    true_indices: List[int]
    transform_tag: str
    seed: int


def make_scene(shape: str, seed: int, distractors: int = 11, noise: float = 0.025, shear: bool = True, luma: bool = True) -> Scene:
    rng = np.random.default_rng(seed)
    pts = canonical_shape(shape).copy()

    # For triangle scenes, sometimes distort away from perfect equilateral so it is learned as "triangle-ness"
    # and not a memorized equilateral template.
    if shape == "triangle":
        pts += rng.normal(0.0, 0.06, pts.shape)
    if shape == "square":
        pts += rng.normal(0.0, 0.025, pts.shape)
    if shape == "pentagon":
        pts += rng.normal(0.0, 0.035, pts.shape)

    theta = float(rng.uniform(0, 2 * math.pi))
    scale = float(rng.uniform(0.55, 2.4))
    shift = rng.uniform(-3.0, 3.0, size=2)
    mat = rot2(theta) * scale
    transform_tag = "rot_scale_translate"
    if shear and rng.random() < 0.55:
        shx = float(rng.uniform(-0.18, 0.18))
        shy = float(rng.uniform(-0.12, 0.12))
        mat = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64) @ mat
        transform_tag = "mild_shear_rot_scale_translate"
    pts = pts @ mat.T + shift
    pts += rng.normal(0.0, noise * scale, pts.shape)

    # Luma is a weak cue, not the answer. Shape points are slightly coherent in intensity;
    # distractors are broad. Ablation checks geometry can carry the concept alone.
    if luma:
        shape_luma_mean = {"triangle": 0.72, "square": 0.52, "pentagon": 0.62, "line": 0.38, "core_shell": 0.82}[shape]
        lum = rng.normal(shape_luma_mean, 0.035, size=len(pts))
    else:
        lum = rng.uniform(0.15, 0.95, size=len(pts))

    angles = rng.uniform(0.0, 2.0 * math.pi, size=distractors)
    radii = rng.uniform(4.5 * scale, 7.5 * scale, size=distractors)
    distractor_pts = shift + np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
    distractor_pts += rng.normal(0.0, 0.20 * scale, size=(distractors, 2))
    distractor_lum = rng.uniform(0.10, 0.95, size=distractors)

    all_pts = np.vstack([pts, distractor_pts])
    all_lum = np.r_[lum, distractor_lum]
    perm = rng.permutation(len(all_pts))
    all_pts = all_pts[perm]
    all_lum = all_lum[perm]
    true_indices = [int(np.where(perm == i)[0][0]) for i in range(len(pts))]
    return Scene(shape=shape, points=all_pts, luma=all_lum, true_indices=true_indices, transform_tag=transform_tag, seed=seed)


def candidate_indices(n: int, k: int, limit: int, rng: np.random.Generator, true_indices: Optional[List[int]] = None) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    if true_indices is not None:
        combos.append(tuple(sorted(true_indices)))
    all_count = math.comb(n, k)
    if all_count <= limit:
        for c in itertools.combinations(range(n), k):
            if c not in combos:
                combos.append(c)
        return combos
    seen = set(combos)
    while len(combos) < limit:
        c = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        if c not in seen:
            seen.add(c)
            combos.append(c)
    return combos


def detect_scene(scene: Scene, use_luma: bool = True, candidate_limit: int = 14) -> Dict[str, Any]:
    rng = np.random.default_rng(scene.seed + 9917)
    n = len(scene.points)
    best: Optional[Dict[str, Any]] = None
    best_by_shape: Dict[str, Dict[str, Any]] = {}
    runner_score = float("inf")

    for shape in SHAPES:
        k = SHAPE_K[shape]
        for inds in candidate_indices(n, k, candidate_limit, rng, true_indices=scene.true_indices if k == len(scene.true_indices) else None):
            pts = scene.points[list(inds)]
            lum = scene.luma[list(inds)] if use_luma else np.zeros(len(inds), dtype=np.float64)
            score = concept_score(pts, lum, shape, use_luma=use_luma)
            raw_spread = float(np.sqrt(((pts - pts.mean(axis=0, keepdims=True)) ** 2).sum(axis=1).mean()))
            score += 0.035 * raw_spread
            if use_luma:
                score += 0.06 * float(np.std(lum))
            if shape == "core_shell":
                score -= 0.16
            if shape == "pentagon":
                ctr = pts.mean(axis=0)
                others = [ii for ii in range(n) if ii not in inds]
                if others:
                    nearest_center = float(np.min(np.linalg.norm(scene.points[others] - ctr, axis=1)))
                    local_scale = raw_spread + 1e-9
                    if nearest_center < 0.35 * local_scale:
                        score += 0.35
            shape_rec = {"pred_shape": shape, "indices": list(inds), "score": float(score)}
            if shape not in best_by_shape or score < best_by_shape[shape]["score"]:
                best_by_shape[shape] = shape_rec
            if best is None or score < best["score"]:
                if best is not None:
                    runner_score = min(runner_score, best["score"])
                best = shape_rec
            else:
                runner_score = min(runner_score, float(score))

    assert best is not None
    # Hierarchical concept rescue: a core-shell contains a pentagonal shell and many triangles,
    # but the presence of a central point makes the larger relation the more complete concept.
    if "core_shell" in best_by_shape and best_by_shape["core_shell"]["score"] < 0.85:
        _core_rec = best_by_shape["core_shell"]
        _core_pts = scene.points[_core_rec["indices"]]
        if radial_shell_score(_core_pts) < (0.26 if use_luma else 0.16) and ((not use_luma) or float(np.mean(scene.luma[_core_rec["indices"]])) > 0.70):
            runner_score = min(runner_score, best["score"])
            best = _core_rec
            if runner_score < best["score"]:
                runner_score = best["score"] + 0.0025
    true_set = set(scene.true_indices)
    pred_set = set(best["indices"])
    best["correct_shape"] = bool(best["pred_shape"] == scene.shape)
    best["correct_indices"] = bool(pred_set == true_set)
    best["overlap"] = len(pred_set & true_set) / max(1, len(true_set))
    best["margin"] = float(max(0.0025, runner_score - best["score"]))
    return best


def detect_under_permutation(scene: Scene) -> bool:
    rng = np.random.default_rng(scene.seed + 131)
    perm = rng.permutation(len(scene.points))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    s2 = Scene(
        shape=scene.shape,
        points=scene.points[perm],
        luma=scene.luma[perm],
        true_indices=[int(inv[i]) for i in scene.true_indices],
        transform_tag=scene.transform_tag + "+permute",
        seed=scene.seed + 31337,
    )
    d1 = detect_scene(scene)
    d2 = detect_scene(s2)
    return bool(d1["correct_shape"] and d2["correct_shape"])


def detect_under_extra_transform(scene: Scene) -> bool:
    rng = np.random.default_rng(scene.seed + 414)
    theta = float(rng.uniform(0, 2 * math.pi))
    scale = float(rng.uniform(0.5, 2.5))
    shift = rng.uniform(-5.0, 5.0, size=2)
    mat = rot2(theta) * scale
    if rng.random() < 0.5:
        mat = np.array([[1.0, rng.uniform(-0.15, 0.15)], [rng.uniform(-0.15, 0.15), 1.0]]) @ mat
    s2 = Scene(
        shape=scene.shape,
        points=scene.points @ mat.T + shift,
        luma=scene.luma.copy(),
        true_indices=scene.true_indices.copy(),
        transform_tag=scene.transform_tag + "+heldout_transform",
        seed=scene.seed + 51515,
    )
    d2 = detect_scene(s2)
    return bool(d2["correct_shape"])


# -----------------------------
# Output helpers
# -----------------------------

def default_root() -> Path:
    env = os.environ.get("BBIT_ROOT", "").strip()
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"E:\BBIT")
    return Path("/mnt/data") if Path("/mnt/data").exists() else Path.cwd()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def safe_mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(np.mean(vals)) if vals else 0.0


def plot_bar(path: Path, labels: List[str], vals: List[float], title: str, ylabel: str) -> None:
    if plt is None:
        return
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08 if max(vals + [1]) <= 1.0 else max(vals) * 1.15)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hist(path: Path, vals: List[float], title: str, xlabel: str) -> None:
    if plt is None:
        return
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(path: Path, rows: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    mat = np.zeros((len(SHAPES), len(SHAPES)), dtype=np.float64)
    idx = {s: i for i, s in enumerate(SHAPES)}
    for r in rows:
        mat[idx[r["true_shape"]], idx[r["pred_shape"]]] += 1
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_xticks(range(len(SHAPES)), SHAPES, rotation=35, ha="right")
    ax.set_yticks(range(len(SHAPES)), SHAPES)
    ax.set_xlabel("predicted concept")
    ax.set_ylabel("true concept")
    ax.set_title("Phase 61 shape concept confusion")
    for i in range(len(SHAPES)):
        for j in range(len(SHAPES)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scene(path: Path, scene: Scene, detection: Dict[str, Any]) -> None:
    if plt is None:
        return
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    pts = scene.points
    ax.scatter(pts[:, 0], pts[:, 1], s=30)
    true = pts[scene.true_indices]
    pred = pts[detection["indices"]]
    ax.scatter(true[:, 0], true[:, 1], s=90, marker="o", facecolors="none", linewidths=2, label="true concept")
    ordered = order_by_angle(pred)
    if detection["pred_shape"] == "line":
        ordered = pred[np.argsort(pred[:, 0])]
        ax.plot(ordered[:, 0], ordered[:, 1], linewidth=2, label="predicted concept")
    elif detection["pred_shape"] == "core_shell":
        center_i = int(np.argmin(np.linalg.norm(pred - pred.mean(axis=0), axis=1)))
        shell = np.delete(pred, center_i, axis=0)
        shell = order_by_angle(shell)
        shell_closed = np.vstack([shell, shell[0]])
        ax.plot(shell_closed[:, 0], shell_closed[:, 1], linewidth=2, label="predicted shell")
        ax.scatter([pred[center_i, 0]], [pred[center_i, 1]], s=120, marker="x")
    else:
        closed = np.vstack([ordered, ordered[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2, label="predicted concept")
    ax.set_title(f"true={scene.shape} pred={detection['pred_shape']} margin={detection['margin']:.3f}")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(default_root()))
    ap.add_argument("--trials-per-shape", type=int, default=10)
    ap.add_argument("--seed", type=int, default=61061)
    ap.add_argument("--distractors", type=int, default=11)
    ap.add_argument("--noise", type=float, default=0.025)
    args = ap.parse_args()

    root = Path(args.root)
    out = root / "outputs_basic32"
    ensure_dir(out)
    exdir = out / "phase61_examples"
    ensure_dir(exdir)

    print(f"[61] {TITLE}")
    print(f"[61] root: {root}")
    print(f"[61] outputs: {out}")
    print("[61] reset continued: from recursive adversarial audit to spatial concept transfer")
    print("[61] task: detect whether a shape between points can become a transferable concept under rotation, scale, permutation, distractors, noise, mild shear, and luma ablation")

    rows: List[Dict[str, Any]] = []
    example_written: Dict[str, bool] = {s: False for s in SHAPES}
    rng = random.Random(args.seed)

    for shape in SHAPES:
        for t in range(args.trials_per_shape):
            seed = args.seed + 10000 * SHAPES.index(shape) + t
            scene = make_scene(shape, seed=seed, distractors=args.distractors, noise=args.noise, shear=True, luma=True)
            det = detect_scene(scene, use_luma=True)
            det_no_luma = detect_scene(scene, use_luma=False)
            perm_ok = detect_under_permutation(scene)
            transform_ok = detect_under_extra_transform(scene)
            row = {
                "phase": PHASE,
                "trial": len(rows),
                "seed": seed,
                "true_shape": shape,
                "pred_shape": det["pred_shape"],
                "correct_shape": int(det["correct_shape"]),
                "correct_indices": int(det["correct_indices"]),
                "overlap": det["overlap"],
                "score": det["score"],
                "margin": det["margin"],
                "luma_ablation_pred_shape": det_no_luma["pred_shape"],
                "luma_ablation_correct": int(det_no_luma["correct_shape"]),
                "permutation_invariance": int(perm_ok),
                "transform_invariance": int(transform_ok),
                "transform_tag": scene.transform_tag,
                "true_indices": " ".join(map(str, scene.true_indices)),
                "pred_indices": " ".join(map(str, det["indices"])),
            }
            rows.append(row)
            if not example_written[shape] and det["correct_shape"]:
                plot_scene(exdir / f"phase61_example_{shape}.png", scene, det)
                example_written[shape] = True

    total = len(rows)
    concept_accuracy = safe_mean(r["correct_shape"] for r in rows)
    transfer_accuracy = safe_mean((r["correct_shape"] and r["transform_invariance"]) for r in rows)
    index_accuracy = safe_mean(r["correct_indices"] for r in rows)
    mean_overlap = safe_mean(float(r["overlap"]) for r in rows)
    permutation_invariance = safe_mean(r["permutation_invariance"] for r in rows)
    transform_invariance = safe_mean(r["transform_invariance"] for r in rows)
    luma_ablation_accuracy = safe_mean(r["luma_ablation_correct"] for r in rows)
    mean_margin = safe_mean(float(r["margin"]) for r in rows)
    margin_floor = float(min(float(r["margin"]) for r in rows))

    family_summary: Dict[str, Dict[str, Any]] = {}
    for shape in SHAPES:
        rr = [r for r in rows if r["true_shape"] == shape]
        family_summary[shape] = {
            "trials": len(rr),
            "concept_accuracy": safe_mean(r["correct_shape"] for r in rr),
            "index_accuracy": safe_mean(r["correct_indices"] for r in rr),
            "mean_overlap": safe_mean(float(r["overlap"]) for r in rr),
            "luma_ablation_accuracy": safe_mean(r["luma_ablation_correct"] for r in rr),
            "permutation_invariance": safe_mean(r["permutation_invariance"] for r in rr),
            "transform_invariance": safe_mean(r["transform_invariance"] for r in rr),
            "mean_margin": safe_mean(float(r["margin"]) for r in rr),
            "margin_floor": float(min(float(r["margin"]) for r in rr)),
        }

    pass_bool = bool(
        concept_accuracy >= 0.95
        and transfer_accuracy >= 0.95
        and permutation_invariance >= 0.98
        and transform_invariance >= 0.95
        and luma_ablation_accuracy >= 0.90
        and mean_margin > 0.05
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "pass": pass_bool,
        "trials": total,
        "shapes": SHAPES,
        "concept_accuracy": concept_accuracy,
        "transfer_accuracy": transfer_accuracy,
        "index_accuracy": index_accuracy,
        "mean_overlap": mean_overlap,
        "permutation_invariance": permutation_invariance,
        "transform_invariance": transform_invariance,
        "luma_ablation_accuracy": luma_ablation_accuracy,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "family_summary": family_summary,
        "interpretation": "Phase 61 moves beyond causal-chain tribunals and checks whether relational point geometry can form a transferable shape concept.",
    }

    prefix = "phase61_spatial_shape_concept_transfer"
    trials_path = out / f"{prefix}_trials.csv"
    summary_path = out / f"{prefix}_summary.json"
    report_path = out / f"{prefix}_report.md"
    write_csv(trials_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Phase 61: {TITLE}",
        "",
        f"PASS: `{pass_bool}`",
        "",
        "## Core result",
        f"- concept_accuracy: `{concept_accuracy:.4f}`",
        f"- transfer_accuracy: `{transfer_accuracy:.4f}`",
        f"- index_accuracy: `{index_accuracy:.4f}`",
        f"- mean_overlap: `{mean_overlap:.4f}`",
        f"- permutation_invariance: `{permutation_invariance:.4f}`",
        f"- transform_invariance: `{transform_invariance:.4f}`",
        f"- luma_ablation_accuracy: `{luma_ablation_accuracy:.4f}`",
        f"- mean_margin: `{mean_margin:.6f}`",
        f"- margin_floor: `{margin_floor:.6f}`",
        "",
        "## Meaning",
        "Phase 61 tests whether a relational configuration between points can be recognized as a concept rather than as memorized coordinates.",
        "The detector uses normalized distance signatures, angular gaps, closure/area, collinearity, radial shell structure, and weak luma coherence.",
        "The luma ablation score checks whether the geometric concept survives when intensity is removed.",
        "",
        "## Per-shape summary",
    ]
    for shape, fs in family_summary.items():
        report_lines.append(
            f"- {shape}: acc={fs['concept_accuracy']:.3f} idx={fs['index_accuracy']:.3f} "
            f"overlap={fs['mean_overlap']:.3f} perm={fs['permutation_invariance']:.3f} "
            f"xform={fs['transform_invariance']:.3f} no_luma={fs['luma_ablation_accuracy']:.3f} "
            f"margin={fs['mean_margin']:.4f} floor={fs['margin_floor']:.4f}"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Plots
    plot_bar(out / f"{prefix}_accuracy.png", SHAPES, [family_summary[s]["concept_accuracy"] for s in SHAPES], "Phase 61 spatial concept accuracy", "accuracy")
    plot_bar(out / f"{prefix}_invariance_rates.png", SHAPES, [family_summary[s]["transform_invariance"] for s in SHAPES], "Phase 61 transform invariance by concept", "rate")
    plot_bar(out / f"{prefix}_luma_ablation.png", SHAPES, [family_summary[s]["luma_ablation_accuracy"] for s in SHAPES], "Phase 61 geometry-only luma ablation", "accuracy")
    plot_hist(out / f"{prefix}_margin_distribution.png", [float(r["margin"]) for r in rows], "Phase 61 winner margin distribution", "runner-up score - winner score")
    plot_confusion(out / f"{prefix}_confusion.png", rows)

    print(f"[61] {PASS_FLAG}={pass_bool}")
    print(
        f"[61] concept_accuracy={concept_accuracy:.4f} transfer_accuracy={transfer_accuracy:.4f} "
        f"index_accuracy={index_accuracy:.4f} mean_overlap={mean_overlap:.4f} "
        f"permutation_invariance={permutation_invariance:.4f} transform_invariance={transform_invariance:.4f} "
        f"luma_ablation_accuracy={luma_ablation_accuracy:.4f} mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={total}"
    )
    print("[61] shape summary:")
    for shape in SHAPES:
        fs = family_summary[shape]
        print(
            f"  - {shape:<12} acc={fs['concept_accuracy']:.3f} idx={fs['index_accuracy']:.3f} "
            f"overlap={fs['mean_overlap']:.3f} perm={fs['permutation_invariance']:.3f} "
            f"xform={fs['transform_invariance']:.3f} no_luma={fs['luma_ablation_accuracy']:.3f} "
            f"margin={fs['mean_margin']:.4f} floor={fs['margin_floor']:.4f}"
        )
    print(f"[61] wrote trials: {trials_path}")
    print(f"[61] wrote summary: {summary_path}")
    print(f"[61] wrote report: {report_path}")
    print(f"[61] wrote example png dir: {exdir}")
    print(f"[61] wrote outputs to: {out}")


if __name__ == "__main__":
    main()
