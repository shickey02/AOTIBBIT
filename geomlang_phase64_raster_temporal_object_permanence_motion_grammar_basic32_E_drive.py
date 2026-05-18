#!/usr/bin/env python3
"""
Phase 64 — Raster temporal object-permanence + motion grammar probe, BASIC32, E-drive.

Goal:
  Move beyond static raster recognition into a real next step:
  1) read 32x32 grayscale raster images,
  2) recover two shape concepts and their spatial relation,
  3) preserve object identity across a second temporal frame,
  4) classify a motion primitive from the two-frame change,
  5) verify that static geometry still works when luma role-binding is ablated.

This is deliberately not another adversarial tribunal phase. It is a forward bridge from
"shape/relation in one raster" to "object permanence + temporal grammar in raster space." 
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = 64
TITLE = "Raster temporal object-permanence motion grammar"
PASS_FLAG = "PHASE64_RASTER_TEMPORAL_OBJECT_PERMANENCE_MOTION_GRAMMAR_PASS"
SEED = 6464
IMG_N = 32
TRIALS_PER_CASE = 7

ROOT = Path(os.environ.get("BBIT_ROOT", r"E:\BBIT"))
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR = OUT / "phase64_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SHAPES = ["triangle", "square", "pentagon", "line", "core_shell"]
POINT_COUNTS = {"triangle": 3, "square": 4, "pentagon": 5, "line": 3, "core_shell": 6}
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
MOTIONS = ["together", "a_toward_b", "a_away_from_b", "a_orbits_b", "swap_sides"]

CASES = [
    ("triangle_inside_square", "triangle", "square", "inside"),
    ("triangle_inside_pentagon", "triangle", "pentagon", "inside"),
    ("line_left_of_triangle", "line", "triangle", "left_of"),
    ("square_right_of_core", "square", "core_shell", "right_of"),
    ("pentagon_above_line", "pentagon", "line", "above"),
    ("line_below_square", "line", "square", "below"),
    ("triangle_near_core", "triangle", "core_shell", "near"),
    ("square_separate_pentagon", "square", "pentagon", "separate"),
]


@dataclass
class Trial:
    case: str
    a_shape: str
    b_shape: str
    relation0: str
    motion: str
    image0: np.ndarray
    image1: np.ndarray
    image0_noluma: np.ndarray


def regular_polygon(n: int, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return np.array([[math.cos(phase + 2 * math.pi * k / n), math.sin(phase + 2 * math.pi * k / n)] for k in range(n)], dtype=float) * radius


def base_shape(shape: str) -> np.ndarray:
    if shape == "triangle":
        return regular_polygon(3, 0.48, math.pi / 2)
    if shape == "square":
        return regular_polygon(4, 0.48, math.pi / 4)
    if shape == "pentagon":
        return regular_polygon(5, 0.48, math.pi / 2)
    if shape == "line":
        return np.array([[-0.50, 0.0], [0.0, 0.0], [0.50, 0.0]], dtype=float)
    if shape == "core_shell":
        return np.vstack([np.array([[0.0, 0.0]], dtype=float), regular_polygon(5, 0.54, math.pi / 2)])
    raise ValueError(shape)


def rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return points @ R.T


def relation_centers(relation: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if relation == "inside":
        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.43, 0.98
    if relation == "left_of":
        return np.array([-0.58, 0.0]), np.array([0.58, 0.0]), 0.72, 0.72
    if relation == "right_of":
        return np.array([0.58, 0.0]), np.array([-0.58, 0.0]), 0.72, 0.72
    if relation == "above":
        return np.array([0.0, 0.58]), np.array([0.0, -0.58]), 0.72, 0.72
    if relation == "below":
        return np.array([0.0, -0.58]), np.array([0.0, 0.58]), 0.72, 0.72
    if relation == "near":
        return np.array([-0.28, 0.28]), np.array([0.28, -0.28]), 0.58, 0.58
    if relation == "separate":
        return np.array([-0.80, -0.34]), np.array([0.80, 0.34]), 0.64, 0.64
    raise ValueError(relation)


def make_objects(a_shape: str, b_shape: str, relation: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    ca, cb, sa, sb = relation_centers(relation)
    theta_a = 0.17 * idx + 0.11 * len(a_shape)
    theta_b = -0.13 * idx + 0.07 * len(b_shape)
    A = rotate(base_shape(a_shape), theta_a) * sa + ca
    B = rotate(base_shape(b_shape), theta_b) * sb + cb
    rng = np.random.default_rng(SEED + idx * 97 + len(a_shape) * 11 + len(b_shape) * 13)
    # Keep jitter small enough that the semantic relation is preserved but real enough to be raster-like.
    A += rng.normal(0.0, 0.0045, A.shape)
    B += rng.normal(0.0, 0.0045, B.shape)
    return A, B


def apply_motion(A: np.ndarray, B: np.ndarray, motion: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    ca, cb = centroid(A), centroid(B)
    v = np.array([0.10 + 0.015 * math.sin(idx), 0.065 + 0.010 * math.cos(idx * 0.7)])
    if motion == "together":
        return A + v, B + v
    direction = cb - ca
    norm = float(np.linalg.norm(direction)) + 1e-9
    u = direction / norm
    if motion == "a_toward_b":
        return A + 0.18 * u, B
    if motion == "a_away_from_b":
        return A - 0.18 * u, B
    if motion == "a_orbits_b":
        # Rotate A around B's centroid; B remains the anchor.
        angle = 0.23
        return rotate(A - cb, angle) + cb, B
    if motion == "swap_sides":
        # Partial side-swap: enough to flip the A-vs-B horizontal sign in non-inside scenes.
        mid = (ca + cb) * 0.5
        return A + (cb - ca), B + (ca - cb)
    raise ValueError(motion)


def to_pixel(points: np.ndarray) -> np.ndarray:
    return (points + 1.25) / 2.50 * 23.0 + 4.0


def render(A: np.ndarray, B: np.ndarray, idx: int, luma: bool = True) -> np.ndarray:
    img = np.zeros((IMG_N, IMG_N), dtype=float)
    Ap, Bp = to_pixel(A), to_pixel(B)
    la, lb = (0.92, 0.58) if luma else (0.72, 0.72)

    def stamp(p: np.ndarray, amp: float) -> None:
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if 1 <= x < IMG_N - 1 and 1 <= y < IMG_N - 1:
            img[y, x] = max(img[y, x], amp)
            for dx, dy, f in [(-1, 0, 0.28), (1, 0, 0.28), (0, -1, 0.28), (0, 1, 0.28)]:
                img[y + dy, x + dx] = max(img[y + dy, x + dx], amp * f)

    for p in Ap:
        stamp(p, la)
    for p in Bp:
        stamp(p, lb)

    rng = np.random.default_rng(SEED + idx * 131)
    for _ in range(10):
        x = int(rng.integers(1, IMG_N - 1))
        y = int(rng.integers(1, IMG_N - 1))
        img[y, x] = max(img[y, x], float(rng.uniform(0.035, 0.13)))
    img += rng.normal(0.0, 0.0055, img.shape)
    return np.clip(img, 0.0, 1.0)


def make_trial(case_tuple: Tuple[str, str, str, str], motion: str, idx: int) -> Trial:
    case, a_shape, b_shape, rel = case_tuple
    A0, B0 = make_objects(a_shape, b_shape, rel, idx)
    A1, B1 = apply_motion(A0, B0, motion, idx)
    return Trial(
        case=case,
        a_shape=a_shape,
        b_shape=b_shape,
        relation0=rel,
        motion=motion,
        image0=render(A0, B0, idx, luma=True),
        image1=render(A1, B1, idx + 1000, luma=True),
        image0_noluma=render(A0, B0, idx + 2000, luma=False),
    )


# ------------------------- raster evidence extraction -------------------------

def extract_peaks(img: np.ndarray, threshold: float = 0.22, min_dist: float = 1.05) -> Tuple[np.ndarray, np.ndarray]:
    candidates: List[Tuple[float, int, int]] = []
    for y in range(1, IMG_N - 1):
        for x in range(1, IMG_N - 1):
            v = float(img[y, x])
            if v < threshold:
                continue
            patch = img[y - 1 : y + 2, x - 1 : x + 2]
            if v >= float(np.max(patch)):
                candidates.append((v, x, y))
    candidates.sort(reverse=True)
    chosen: List[Tuple[float, float, float]] = []
    for v, x, y in candidates:
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for _, px, py in chosen):
            ys = slice(max(0, y - 1), min(IMG_N, y + 2))
            xs = slice(max(0, x - 1), min(IMG_N, x + 2))
            patch = img[ys, xs]
            yy, xx = np.mgrid[ys, xs]
            w = np.maximum(patch - 0.045, 0.0)
            if float(np.sum(w)) > 1e-9:
                cx = float(np.sum(xx * w) / np.sum(w))
                cy = float(np.sum(yy * w) / np.sum(w))
            else:
                cx, cy = float(x), float(y)
            chosen.append((float(v), cx, cy))
        if len(chosen) >= 13:
            break
    vals = np.array([c[0] for c in chosen], dtype=float)
    pix = np.array([[c[1], c[2]] for c in chosen], dtype=float)
    pts = (pix - 4.0) / 23.0 * 2.50 - 1.25
    return pts, vals


def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def radius(points: np.ndarray) -> float:
    c = centroid(points)
    return float(np.max(np.linalg.norm(points - c, axis=1)))


def polygon_score(points: np.ndarray, n: int) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) != n:
        return 1e9
    c = centroid(pts)
    r = np.linalg.norm(pts - c, axis=1)
    radial = float(np.std(r) / (np.mean(r) + 1e-9))
    ang = np.sort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    angular = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
    return radial + 0.62 * angular


def line_score(points: np.ndarray) -> float:
    if len(points) != 3:
        return 1e9
    X = points - centroid(points)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return float(s[1] / (s[0] + 1e-9)) + 0.08 / (float(np.max(np.linalg.norm(X, axis=1))) + 1e-6)


def core_shell_score(points: np.ndarray) -> float:
    if len(points) != 6:
        return 1e9
    best = 1e9
    for k in range(6):
        core = points[k]
        shell = np.delete(points, k, axis=0)
        shell_s = polygon_score(shell, 5)
        offset = float(np.linalg.norm(centroid(shell) - core) / (np.mean(np.linalg.norm(shell - core, axis=1)) + 1e-9))
        best = min(best, shell_s + 0.42 * offset)
    return best


def shape_score(points: np.ndarray, shape: str) -> float:
    if shape == "triangle":
        return polygon_score(points, 3)
    if shape == "square":
        return polygon_score(points, 4)
    if shape == "pentagon":
        return polygon_score(points, 5)
    if shape == "line":
        return line_score(points)
    if shape == "core_shell":
        return core_shell_score(points)
    return 1e9


def relation_score(A: np.ndarray, B: np.ndarray, relation: str) -> float:
    ca, cb = centroid(A), centroid(B)
    dx, dy = float(ca[0] - cb[0]), float(ca[1] - cb[1])
    d = math.hypot(dx, dy)
    ra, rb = radius(A), radius(B)
    if relation == "inside":
        return max(0.0, d + ra - rb * 0.98) * 3.0
    if relation == "left_of":
        return max(0.0, dx + 0.42) + 0.30 * max(0.0, abs(dy) - abs(dx) * 0.85)
    if relation == "right_of":
        return max(0.0, -dx + 0.42) + 0.30 * max(0.0, abs(dy) - abs(dx) * 0.85)
    if relation == "above":
        return max(0.0, -dy + 0.42) + 0.30 * max(0.0, abs(dx) - abs(dy) * 0.85)
    if relation == "below":
        return max(0.0, dy + 0.42) + 0.30 * max(0.0, abs(dx) - abs(dy) * 0.85)
    if relation == "near":
        return abs(d - 0.78) * 0.42 + max(0.0, d - 1.10) + max(0.0, 0.38 - d)
    if relation == "separate":
        return max(0.0, 1.32 - d) * 0.80
    return 1e9


def infer_relation(A: np.ndarray, B: np.ndarray) -> str:
    scores = {rel: relation_score(A, B, rel) for rel in RELATIONS}
    return min(scores, key=scores.get)


def explain_static(img: np.ndarray, allow_luma: bool = True) -> Dict[str, object]:
    pts, vals = extract_peaks(img)
    npts = len(pts)
    best: Dict[str, object] | None = None
    runner = 1e9

    # Luma path: bright peaks are object A, dimmer peaks are object B.
    if allow_luma:
        hi = [i for i, v in enumerate(vals) if v > 0.75]
        lo = [i for i, v in enumerate(vals) if 0.34 < v <= 0.75]
        if hi and lo:
            A, B = pts[hi], pts[lo]
            for case, a_shape, b_shape, rel in CASES:
                if len(A) != POINT_COUNTS[a_shape] or len(B) != POINT_COUNTS[b_shape]:
                    continue
                score = shape_score(A, a_shape) + shape_score(B, b_shape) + relation_score(A, B, rel)
                item = {"case": case, "a_shape": a_shape, "b_shape": b_shape, "relation": rel, "A": A, "B": B, "score": float(score), "detected_points": int(npts)}
                if best is None or score < float(best["score"]):
                    if best is not None:
                        runner = float(best["score"])
                    best = item
                elif score < runner:
                    runner = float(score)
            if best is not None:
                best["margin"] = float(max(0.0, runner - float(best["score"]))) if runner < 1e8 else 0.25
                best["luma_role_ok"] = True
                return best

    # Geometry-only path: exhaustive partition among the brightest evidence points.
    for case, a_shape, b_shape, rel in CASES:
        na, nb = POINT_COUNTS[a_shape], POINT_COUNTS[b_shape]
        if npts < na + nb:
            continue
        order = np.argsort(vals)[::-1][: na + nb]
        cand = pts[order]
        all_idx = range(len(cand))
        for ia in combinations(all_idx, na):
            ia_set = set(ia)
            ib = [j for j in all_idx if j not in ia_set]
            if len(ib) != nb:
                continue
            A, B = cand[list(ia)], cand[ib]
            score = shape_score(A, a_shape) + shape_score(B, b_shape) + relation_score(A, B, rel)
            item = {"case": case, "a_shape": a_shape, "b_shape": b_shape, "relation": rel, "A": A, "B": B, "score": float(score), "detected_points": int(npts)}
            if best is None or score < float(best["score"]):
                if best is not None:
                    runner = float(best["score"])
                best = item
            elif score < runner:
                runner = float(score)

    if best is None:
        return {"case": "none", "a_shape": "none", "b_shape": "none", "relation": "none", "A": np.empty((0, 2)), "B": np.empty((0, 2)), "score": 1e9, "margin": 0.0, "luma_role_ok": False, "detected_points": int(npts)}
    best["margin"] = float(max(0.0, runner - float(best["score"]))) if runner < 1e8 else 0.25
    best["luma_role_ok"] = False
    return best



def explain_luma_objects(img: np.ndarray) -> Dict[str, object]:
    """Recover A/B object identities from luma, independent of which static relation is present."""
    pts, vals = extract_peaks(img)
    hi = [i for i, v in enumerate(vals) if v > 0.75]
    lo = [i for i, v in enumerate(vals) if 0.34 < v <= 0.75]
    A, B = pts[hi], pts[lo]
    best_a = min(SHAPES, key=lambda sh: shape_score(A, sh) if len(A) == POINT_COUNTS[sh] else 1e9) if len(A) else "none"
    best_b = min(SHAPES, key=lambda sh: shape_score(B, sh) if len(B) == POINT_COUNTS[sh] else 1e9) if len(B) else "none"
    return {"a_shape": best_a, "b_shape": best_b, "A": A, "B": B, "detected_points": int(len(pts))}

def classify_motion(A0: np.ndarray, B0: np.ndarray, A1: np.ndarray, B1: np.ndarray) -> str:
    ca0, cb0, ca1, cb1 = centroid(A0), centroid(B0), centroid(A1), centroid(B1)
    va, vb = ca1 - ca0, cb1 - cb0
    d0 = float(np.linalg.norm(ca0 - cb0))
    d1 = float(np.linalg.norm(ca1 - cb1))
    dx0, dx1 = float(ca0[0] - cb0[0]), float(ca1[0] - cb1[0])
    same_motion = float(np.linalg.norm(va - vb))
    mean_motion = float((np.linalg.norm(va) + np.linalg.norm(vb)) * 0.5)

    if mean_motion > 0.06 and same_motion < 0.055:
        return "together"
    if float(np.linalg.norm(ca1 - cb0)) < 0.23 and float(np.linalg.norm(cb1 - ca0)) < 0.23:
        return "swap_sides"
    if abs(dx0) > 0.22 and dx0 * dx1 < 0 and abs(dx1) > 0.18:
        return "swap_sides"
    if d1 < d0 - 0.10:
        return "a_toward_b"
    if d1 > d0 + 0.10:
        return "a_away_from_b"
    angle0 = math.atan2(float(ca0[1] - cb0[1]), float(ca0[0] - cb0[0]))
    angle1 = math.atan2(float(ca1[1] - cb1[1]), float(ca1[0] - cb1[0]))
    dang = abs(math.atan2(math.sin(angle1 - angle0), math.cos(angle1 - angle0)))
    if dang > 0.12 and float(np.linalg.norm(vb)) < 0.05:
        return "a_orbits_b"
    return "together"


def score_trial(t: Trial) -> Dict[str, object]:
    p0 = explain_static(t.image0, allow_luma=True)
    p1 = explain_luma_objects(t.image1)
    pno = explain_static(t.image0_noluma, allow_luma=False)
    pred_motion = classify_motion(p0["A"], p0["B"], p1["A"], p1["B"])

    scene_ok = p0["case"] == t.case
    relation_ok = p0["relation"] == t.relation0
    concept_ok = p0["a_shape"] == t.a_shape and p0["b_shape"] == t.b_shape
    temporal_identity_ok = p1["a_shape"] == t.a_shape and p1["b_shape"] == t.b_shape
    motion_ok = pred_motion == t.motion
    noluma_ok = pno["case"] == t.case and pno["relation"] == t.relation0

    return {
        "case": t.case,
        "motion": t.motion,
        "true_relation": t.relation0,
        "pred_case": p0["case"],
        "pred_relation": p0["relation"],
        "pred_motion": pred_motion,
        "scene_ok": bool(scene_ok),
        "relation_ok": bool(relation_ok),
        "concept_pair_ok": bool(concept_ok),
        "temporal_identity_ok": bool(temporal_identity_ok),
        "motion_ok": bool(motion_ok),
        "luma_binding_ok": bool(p0["luma_role_ok"]),
        "noluma_static_ok": bool(noluma_ok),
        "margin": float(p0["margin"]),
        "motion_margin_proxy": float(abs(np.linalg.norm(centroid(p1["A"]) - centroid(p1["B"])) - np.linalg.norm(centroid(p0["A"]) - centroid(p0["B"])))),
        "detected_points0": int(p0["detected_points"]),
        "detected_points1": int(p1["detected_points"]),
    }


def mean_bool(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else 0.0


def plot_bar(labels: List[str], values: List[float], title: str, ylabel: str, path: Path, ylim: Tuple[float, float] = (0, 1.08)) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(rows: List[Dict[str, object]], true_key: str, pred_key: str, labels: List[str], title: str, path: Path) -> None:
    if plt is None:
        return
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    counts = np.zeros(len(labels), dtype=float)
    li = {x: i for i, x in enumerate(labels)}
    for r in rows:
        if r[true_key] in li and r[pred_key] in li:
            i, j = li[r[true_key]], li[r[pred_key]]
            mat[i, j] += 1
            counts[i] += 1
    for i in range(len(labels)):
        if counts[i] > 0:
            mat[i] /= counts[i]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_margin(rows: List[Dict[str, object]], path: Path) -> None:
    if plt is None:
        return
    vals = [float(r["margin"]) for r in rows]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.hist(vals, bins=36)
    ax.set_title("Phase 64 raster temporal grammar winner margin distribution")
    ax.set_xlabel("runner-up explanation score - winner score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(t: Trial, path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(t.image0, vmin=0, vmax=1, origin="lower")
    axes[0].set_title("frame 0")
    axes[1].imshow(t.image1, vmin=0, vmax=1, origin="lower")
    axes[1].set_title(f"frame 1: {t.motion}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Phase 64 example: {t.case}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from raster luma scene recognition to temporal object permanence")
    print(f"[{PHASE}] task: recover static shape/relation concepts from 32x32 rasters, preserve object identity across frame change, and classify motion grammar")

    trials: List[Trial] = []
    k = 0
    for case_tuple in CASES:
        rel = case_tuple[3]
        if rel in ("inside", "near"):
            # Overlap/near-core scenes are kept as static raster concepts; temporal motion is tested on separable scenes where object identity is not physically occluded.
            valid_motions = ["together"]
        else:
            valid_motions = ["together", "a_toward_b", "a_away_from_b", "a_orbits_b", "swap_sides"]
        for motion in valid_motions:
            for i in range(TRIALS_PER_CASE):
                trials.append(make_trial(case_tuple, motion, k + i))
            k += TRIALS_PER_CASE

    rows = [score_trial(t) for t in trials]
    scene_accuracy = mean_bool(rows, "scene_ok")
    relation_accuracy = mean_bool(rows, "relation_ok")
    concept_pair_accuracy = mean_bool(rows, "concept_pair_ok")
    temporal_identity_accuracy = mean_bool(rows, "temporal_identity_ok")
    motion_accuracy = mean_bool(rows, "motion_ok")
    luma_binding_accuracy = mean_bool(rows, "luma_binding_ok")
    noluma_static_accuracy = mean_bool(rows, "noluma_static_ok")
    mean_margin = float(np.mean([r["margin"] for r in rows]))
    margin_floor = float(np.min([r["margin"] for r in rows]))

    by_case: Dict[str, Dict[str, float]] = {}
    for case, _, _, _ in CASES:
        sub = [r for r in rows if r["case"] == case]
        by_case[case] = {
            "trials": len(sub),
            "scene_accuracy": mean_bool(sub, "scene_ok"),
            "relation_accuracy": mean_bool(sub, "relation_ok"),
            "temporal_identity_accuracy": mean_bool(sub, "temporal_identity_ok"),
            "motion_accuracy": mean_bool(sub, "motion_ok"),
            "noluma_static_accuracy": mean_bool(sub, "noluma_static_ok"),
            "mean_margin": float(np.mean([r["margin"] for r in sub])),
            "margin_floor": float(np.min([r["margin"] for r in sub])),
        }

    by_motion: Dict[str, Dict[str, float]] = {}
    for motion in MOTIONS:
        sub = [r for r in rows if r["motion"] == motion]
        by_motion[motion] = {
            "trials": len(sub),
            "motion_accuracy": mean_bool(sub, "motion_ok"),
            "scene_accuracy": mean_bool(sub, "scene_ok"),
            "temporal_identity_accuracy": mean_bool(sub, "temporal_identity_ok"),
        }

    pass_flag = bool(
        scene_accuracy >= 0.94
        and relation_accuracy >= 0.94
        and temporal_identity_accuracy >= 0.94
        and motion_accuracy >= 0.90
        and noluma_static_accuracy >= 0.90
        and margin_floor >= 0.02
    )

    prefix = "phase64_raster_temporal_object_permanence_motion_grammar"
    trials_path = OUT / f"{prefix}_trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "pass": pass_flag,
        "trials": len(rows),
        "scene_accuracy": scene_accuracy,
        "relation_accuracy": relation_accuracy,
        "concept_pair_accuracy": concept_pair_accuracy,
        "temporal_identity_accuracy": temporal_identity_accuracy,
        "motion_accuracy": motion_accuracy,
        "luma_binding_accuracy": luma_binding_accuracy,
        "noluma_static_accuracy": noluma_static_accuracy,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "case_summary": by_case,
        "motion_summary": by_motion,
        "interpretation": "Phase 64 is the first raster temporal grammar bridge: static scene concepts become persistent objects whose motion can be classified across frames.",
    }
    summary_path = OUT / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = OUT / f"{prefix}_report.md"
    lines = [
        f"# Phase {PHASE}: {TITLE}",
        "",
        f"PASS: `{pass_flag}`",
        "",
        "## Core result",
        f"- scene_accuracy: `{scene_accuracy:.4f}`",
        f"- relation_accuracy: `{relation_accuracy:.4f}`",
        f"- concept_pair_accuracy: `{concept_pair_accuracy:.4f}`",
        f"- temporal_identity_accuracy: `{temporal_identity_accuracy:.4f}`",
        f"- motion_accuracy: `{motion_accuracy:.4f}`",
        f"- luma_binding_accuracy: `{luma_binding_accuracy:.4f}`",
        f"- noluma_static_accuracy: `{noluma_static_accuracy:.4f}`",
        f"- mean_margin: `{mean_margin:.6f}`",
        f"- margin_floor: `{margin_floor:.6f}`",
        "",
        "## Meaning",
        "Phase 64 stops treating raster scenes as frozen diagrams and asks whether the same concepts can persist across time.",
        "The recognizer receives two grayscale frames, recovers A/B object identity, reads the static scene relation, and classifies a motion primitive.",
        "The geometry-only no-luma ablation still has to solve the static frame, so luma is a binding witness rather than the whole answer.",
        "",
        "## Case summary",
    ]
    for case, stats in by_case.items():
        lines.append(
            f"- {case}: scene={stats['scene_accuracy']:.3f} relation={stats['relation_accuracy']:.3f} "
            f"temporal={stats['temporal_identity_accuracy']:.3f} motion={stats['motion_accuracy']:.3f} "
            f"no_luma={stats['noluma_static_accuracy']:.3f} margin={stats['mean_margin']:.4f} floor={stats['margin_floor']:.4f}"
        )
    lines += ["", "## Motion summary"]
    for motion, stats in by_motion.items():
        lines.append(
            f"- {motion}: motion_acc={stats['motion_accuracy']:.3f} scene_acc={stats['scene_accuracy']:.3f} temporal={stats['temporal_identity_accuracy']:.3f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    case_labels = [c[0] for c in CASES]
    plot_bar(case_labels, [by_case[c]["scene_accuracy"] for c in case_labels], "Phase 64 raster temporal scene accuracy", "accuracy", OUT / f"{prefix}_scene_accuracy.png")
    plot_bar(case_labels, [by_case[c]["relation_accuracy"] for c in case_labels], "Phase 64 raster temporal relation accuracy", "accuracy", OUT / f"{prefix}_relation_accuracy.png")
    plot_bar(case_labels, [by_case[c]["temporal_identity_accuracy"] for c in case_labels], "Phase 64 object permanence accuracy", "accuracy", OUT / f"{prefix}_object_permanence.png")
    plot_bar(MOTIONS, [by_motion[m]["motion_accuracy"] for m in MOTIONS], "Phase 64 motion primitive accuracy", "accuracy", OUT / f"{prefix}_motion_accuracy.png")
    plot_bar(case_labels, [by_case[c]["noluma_static_accuracy"] for c in case_labels], "Phase 64 no-luma static geometry ablation", "accuracy", OUT / f"{prefix}_noluma_ablation.png")
    plot_confusion(rows, "motion", "pred_motion", MOTIONS, "Phase 64 motion confusion", OUT / f"{prefix}_motion_confusion.png")
    plot_confusion(rows, "true_relation", "pred_relation", RELATIONS, "Phase 64 relation confusion", OUT / f"{prefix}_relation_confusion.png")
    plot_margin(rows, OUT / f"{prefix}_margin_distribution.png")
    for t in trials[:8]:
        plot_example(t, EXAMPLE_DIR / f"{t.case}_{t.motion}.png")

    print(f"[{PHASE}] {PASS_FLAG}={pass_flag}")
    print(
        f"[{PHASE}] scene_accuracy={scene_accuracy:.4f} relation_accuracy={relation_accuracy:.4f} "
        f"concept_pair_accuracy={concept_pair_accuracy:.4f} temporal_identity_accuracy={temporal_identity_accuracy:.4f} "
        f"motion_accuracy={motion_accuracy:.4f} luma_binding_accuracy={luma_binding_accuracy:.4f} "
        f"noluma_static_accuracy={noluma_static_accuracy:.4f} mean_margin={mean_margin:.6f} "
        f"margin_floor={margin_floor:.6f} trials={len(rows)}"
    )
    print(f"[{PHASE}] case summary:")
    for case, stats in by_case.items():
        print(
            f"  - {case:27s} scene={stats['scene_accuracy']:.3f} relation={stats['relation_accuracy']:.3f} "
            f"temporal={stats['temporal_identity_accuracy']:.3f} motion={stats['motion_accuracy']:.3f} "
            f"no_luma={stats['noluma_static_accuracy']:.3f} margin={stats['mean_margin']:.4f} floor={stats['margin_floor']:.4f}"
        )
    print(f"[{PHASE}] motion summary:")
    for motion, stats in by_motion.items():
        print(f"  - {motion:14s} motion_acc={stats['motion_accuracy']:.3f} scene_acc={stats['scene_accuracy']:.3f} temporal={stats['temporal_identity_accuracy']:.3f}")
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
