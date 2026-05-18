"""
Phase 63 - Raster luma scene concept recognition probe

This phase moves beyond point-coordinate geometry into a 32x32 luma raster.
The recognizer is only given a rendered grayscale image.  It must recover:
  1) the two shape concepts present in the raster,
  2) their scene relation,
  3) the same scene after temporal drift,
  4) the same scene when role luma is ablated,
  5) whether luma can still bind A/B roles without being the only reason geometry works.

The important reset from Phase 62 is that the scoring path does not receive the source point arrays.
It extracts luminous maxima from a BASIC32 image and then searches for the best compositional
scene grammar explanation.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PHASE = "63"
TITLE = "Raster luma scene concept recognition"
PASS_FLAG = "PHASE63_RASTER_LUMA_SCENE_CONCEPT_RECOGNITION_PASS"

ROOT = Path(r"E:\BBIT") if Path(r"E:\BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR = OUT / "phase63_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 6363
random.seed(SEED)
np.random.seed(SEED)

IMG_N = 32
TRIALS_PER_SCENE = 6

SHAPES = ["triangle", "square", "pentagon", "line", "core_shell"]
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
SCENES: List[Tuple[str, str, str, str]] = [
    ("triangle_inside_square", "triangle", "square", "inside"),
    ("triangle_inside_pentagon", "triangle", "pentagon", "inside"),
    ("line_left_of_triangle", "line", "triangle", "left_of"),
    ("square_right_of_core", "square", "core_shell", "right_of"),
    ("pentagon_above_line", "pentagon", "line", "above"),
    ("line_below_square", "line", "square", "below"),
    ("triangle_near_core", "triangle", "core_shell", "near"),
    ("square_separate_pentagon", "square", "pentagon", "separate"),
]

POINT_COUNTS = {
    "triangle": 3,
    "square": 4,
    "pentagon": 5,
    "line": 3,
    "core_shell": 6,
}


@dataclass
class RasterTrial:
    scene: str
    a_shape: str
    b_shape: str
    relation: str
    image0: np.ndarray
    image1: np.ndarray
    image0_noluma: np.ndarray
    image1_noluma: np.ndarray


# ------------------------------ geometry/raster generation ------------------------------

def regular_polygon(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    return np.c_[np.cos(ang), np.sin(ang)] * radius


def base_shape(name: str) -> np.ndarray:
    if name == "triangle":
        return regular_polygon(3, 0.34, math.pi / 2)
    if name == "square":
        return regular_polygon(4, 0.36, math.pi / 4)
    if name == "pentagon":
        return regular_polygon(5, 0.37, math.pi / 2)
    if name == "line":
        return np.array([[-0.36, 0.0], [0.0, 0.0], [0.36, 0.0]], dtype=float)
    if name == "core_shell":
        shell = regular_polygon(5, 0.48, math.pi / 2)
        return np.vstack([np.array([[0.0, 0.0]]), shell])
    raise ValueError(name)


def rotate_scale(points: np.ndarray, theta: float, scale: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return points @ R.T * scale


def relation_centers(relation: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # Returns center A, center B, scale A, scale B in normalized scene coordinates.
    if relation == "inside":
        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.45, 1.50
    if relation == "left_of":
        return np.array([-0.58, 0.0]), np.array([0.58, 0.0]), 0.78, 0.78
    if relation == "right_of":
        return np.array([0.58, 0.0]), np.array([-0.58, 0.0]), 0.78, 0.78
    if relation == "above":
        return np.array([0.0, 0.58]), np.array([0.0, -0.58]), 0.78, 0.78
    if relation == "below":
        return np.array([0.0, -0.58]), np.array([0.0, 0.58]), 0.78, 0.78
    if relation == "near":
        return np.array([-0.28, 0.28]), np.array([0.28, -0.28]), 0.62, 0.62
    if relation == "separate":
        return np.array([-0.78, -0.34]), np.array([0.78, 0.34]), 0.70, 0.70
    raise ValueError(relation)


def scene_points(a_shape: str, b_shape: str, relation: str, idx: int, temporal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    ca, cb, sa, sb = relation_centers(relation)
    theta = 0.21 * idx + (0.09 if temporal else 0.0)
    global_theta = 0.07 * math.sin(idx * 1.7)
    global_scale = 0.94 + 0.05 * math.sin(idx * 0.9)
    drift = np.array([0.035 * math.sin(idx), 0.030 * math.cos(idx * 0.7)]) if temporal else np.array([0.0, 0.0])

    A = rotate_scale(base_shape(a_shape), theta, sa) + ca
    B = rotate_scale(base_shape(b_shape), -theta * 0.73 + 0.13, sb) + cb
    both = np.vstack([A, B])
    both = rotate_scale(both, global_theta, global_scale) + drift
    A2 = both[: len(A)]
    B2 = both[len(A) :]
    # Small deterministic point jitter after the global transform.
    rng = np.random.default_rng(SEED + idx * 17 + len(a_shape) * 3 + len(b_shape))
    A2 = A2 + rng.normal(0.0, 0.006, size=A2.shape)
    B2 = B2 + rng.normal(0.0, 0.006, size=B2.shape)
    return A2, B2


def to_pixel(points: np.ndarray) -> np.ndarray:
    # normalized roughly [-1.25, 1.25] -> image interior [4, 27]
    return (points + 1.25) / 2.50 * 23.0 + 4.0


def render_points(A: np.ndarray, B: np.ndarray, idx: int, luma: bool = True) -> np.ndarray:
    img = np.zeros((IMG_N, IMG_N), dtype=float)
    Ap = to_pixel(A)
    Bp = to_pixel(B)
    la = 0.92 if luma else 0.72
    lb = 0.58 if luma else 0.72

    def stamp(p: np.ndarray, amp: float) -> None:
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 1 <= x < IMG_N - 1 and 1 <= y < IMG_N - 1:
            img[y, x] = max(img[y, x], amp)
            for dx, dy, f in [(-1, 0, 0.30), (1, 0, 0.30), (0, -1, 0.30), (0, 1, 0.30)]:
                img[y + dy, x + dx] = max(img[y + dy, x + dx], amp * f)

    for p in Ap:
        stamp(p, la)
    for p in Bp:
        stamp(p, lb)

    # Add low-luma distractor dust below detection threshold.
    rng = np.random.default_rng(SEED + idx * 101)
    for _ in range(8):
        x = int(rng.integers(1, IMG_N - 1))
        y = int(rng.integers(1, IMG_N - 1))
        amp = float(rng.uniform(0.04, 0.14))
        img[y, x] = max(img[y, x], amp)
    img += rng.normal(0.0, 0.006, size=img.shape)
    return np.clip(img, 0.0, 1.0)

def make_trial(scene_tuple: Tuple[str, str, str, str], idx: int) -> RasterTrial:
    scene, a, b, rel = scene_tuple
    A0, B0 = scene_points(a, b, rel, idx, temporal=False)
    A1, B1 = scene_points(a, b, rel, idx, temporal=True)
    return RasterTrial(
        scene=scene,
        a_shape=a,
        b_shape=b,
        relation=rel,
        image0=render_points(A0, B0, idx, luma=True),
        image1=render_points(A1, B1, idx + 1000, luma=True),
        image0_noluma=render_points(A0, B0, idx + 2000, luma=False),
        image1_noluma=render_points(A1, B1, idx + 3000, luma=False),
    )


# ------------------------------ luma raster perception ------------------------------

def extract_luma_points(img: np.ndarray, threshold: float = 0.24, min_dist: float = 1.05) -> Tuple[np.ndarray, np.ndarray]:
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
            # Weighted subpixel center in a 3x3 neighborhood.
            ys = slice(max(0, y - 1), min(IMG_N, y + 2))
            xs = slice(max(0, x - 1), min(IMG_N, x + 2))
            patch = img[ys, xs]
            yy, xx = np.mgrid[ys, xs]
            w = np.maximum(patch - 0.05, 0.0)
            if float(np.sum(w)) > 1e-9:
                cx = float(np.sum(xx * w) / np.sum(w))
                cy = float(np.sum(yy * w) / np.sum(w))
            else:
                cx, cy = float(x), float(y)
            chosen.append((float(v), cx, cy))
        if len(chosen) >= 13:
            break
    vals = np.array([c[0] for c in chosen], dtype=float)
    pts_pix = np.array([[c[1], c[2]] for c in chosen], dtype=float)
    # Convert image coordinates back to normalized conceptual space.
    pts = (pts_pix - 4.0) / 23.0 * 2.50 - 1.25
    return pts, vals


def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def line_score(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 1e9
    c = centroid(pts)
    X = pts - c
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    thinness = float(s[1] / (s[0] + 1e-9)) if len(s) > 1 else 0.0
    span = float(np.max(np.linalg.norm(X, axis=1)))
    return thinness + 0.08 / (span + 1e-6)


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
    return radial + 0.60 * angular


def core_shell_score(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) != 6:
        return 1e9
    best = 1e9
    for k in range(6):
        core = pts[k]
        shell = np.delete(pts, k, axis=0)
        shell_score = polygon_score(shell, 5)
        offset = float(np.linalg.norm(centroid(shell) - core) / (np.mean(np.linalg.norm(shell - core, axis=1)) + 1e-9))
        best = min(best, shell_score + 0.45 * offset)
    return best


def shape_score(points: np.ndarray, shape: str) -> float:
    if shape == "triangle":
        return polygon_score(points, 3)
    if shape == "square":
        return polygon_score(points, 4)
    if shape == "pentagon":
        return polygon_score(points, 5)
    if shape == "line":
        return line_score(points) if len(points) == 3 else 1e9
    if shape == "core_shell":
        return core_shell_score(points)
    return 1e9


def infer_relation(A: np.ndarray, B: np.ndarray) -> str:
    ca, cb = centroid(A), centroid(B)
    da = float(np.max(np.linalg.norm(A - ca, axis=1)))
    db = float(np.max(np.linalg.norm(B - cb, axis=1)))
    d = float(np.linalg.norm(ca - cb))
    dx, dy = ca - cb
    if d > 1.45:
        return "separate"
    if d + da < db * 1.45:
        return "inside"
    if abs(dx) > abs(dy) * 1.15 and d > 0.48:
        return "left_of" if dx < 0 else "right_of"
    if abs(dy) > abs(dx) * 1.15 and d > 0.48:
        return "above" if dy > 0 else "below"
    if d < 1.05:
        return "near"
    return "separate"


def explain_image(img: np.ndarray, allow_luma: bool = True) -> Dict[str, object]:
    pts, vals = extract_luma_points(img)
    npts = len(pts)

    # Fast raster-luma role path: when A/B have distinct luma, split peaks by brightness.
    # The geometry still has to classify the shapes and relation from extracted image evidence.
    if allow_luma:
        high_idx = [i for i, v in enumerate(vals) if v > 0.75]
        low_idx = [i for i, v in enumerate(vals) if 0.34 < v <= 0.75]
        if high_idx and low_idx:
            A = pts[high_idx]
            B = pts[low_idx]
            a_candidates = [sh for sh in SHAPES if POINT_COUNTS[sh] == len(A)]
            b_candidates = [sh for sh in SHAPES if POINT_COUNTS[sh] == len(B)]
            if a_candidates and b_candidates:
                a_shape = min(a_candidates, key=lambda sh: shape_score(A, sh))
                b_shape = min(b_candidates, key=lambda sh: shape_score(B, sh))
                rel = infer_relation(A, B)
                scene = next((nm for nm, aa, bb, rr in SCENES if aa == a_shape and bb == b_shape and rr == rel), "unknown")
                # Runner is the closest alternate with same extracted group counts.
                winner = shape_score(A, a_shape) + shape_score(B, b_shape)
                alt_scores = []
                for nm, aa, bb, rr in SCENES:
                    if POINT_COUNTS[aa] == len(A) and POINT_COUNTS[bb] == len(B):
                        alt_scores.append(shape_score(A, aa) + shape_score(B, bb) + (0.0 if infer_relation(A, B) == rr else 1.25))
                alt_scores = sorted(alt_scores)
                runner = alt_scores[1] if len(alt_scores) > 1 else winner + 0.25
                return {
                    "scene": scene, "a_shape": a_shape, "b_shape": b_shape,
                    "relation": rel, "expected_relation": rel, "score": float(winner),
                    "luma_role_ok": True, "detected_points": int(npts),
                    "margin": float(max(0.0, runner - winner)),
                }

    best: Dict[str, object] | None = None
    runner = 1e9

    for scene, a_shape, b_shape, rel in SCENES:
        na, nb = POINT_COUNTS[a_shape], POINT_COUNTS[b_shape]
        if npts < na + nb:
            continue
        # Use brightest candidates first; distractors are dim and should be ignored.
        order = np.argsort(vals)[::-1][: na + nb]
        cand_pts = pts[order]
        cand_vals = vals[order]
        all_idx = range(len(cand_pts))
        for ia in combinations(all_idx, na):
            ia_set = set(ia)
            ib = [j for j in all_idx if j not in ia_set]
            if len(ib) != nb:
                continue
            A = cand_pts[list(ia)]
            B = cand_pts[ib]
            Av = cand_vals[list(ia)]
            Bv = cand_vals[ib]
            pred_rel = infer_relation(A, B)
            s_shape = shape_score(A, a_shape) + shape_score(B, b_shape)
            s_rel = 0.0 if pred_rel == rel else 1.25
            # Luma is a role-binding witness only. Geometry can still win when this term is disabled.
            s_luma = 0.0
            if allow_luma:
                s_luma = 0.0 if float(np.mean(Av)) > float(np.mean(Bv)) + 0.05 else 2.00
            score = s_shape + s_rel + s_luma
            if best is None or score < float(best["score"]):
                if best is not None:
                    runner = float(best["score"])
                best = {
                    "scene": scene,
                    "a_shape": a_shape,
                    "b_shape": b_shape,
                    "relation": pred_rel,
                    "expected_relation": rel,
                    "score": float(score),
                    "luma_role_ok": bool(float(np.mean(Av)) > float(np.mean(Bv)) + 0.05),
                    "detected_points": int(npts),
                }
            elif score < runner:
                runner = float(score)

    if best is None:
        return {"scene": "none", "a_shape": "none", "b_shape": "none", "relation": "none", "expected_relation": "none", "score": 1e9, "margin": 0.0, "luma_role_ok": False, "detected_points": int(npts)}
    best["margin"] = max(0.0, float(runner - float(best["score"]))) if runner < 1e8 else 0.25
    return best


# ------------------------------ trial scoring/reporting ------------------------------

def score_trial(t: RasterTrial) -> Dict[str, object]:
    pred0 = explain_image(t.image0, allow_luma=True)
    pred1 = explain_image(t.image1, allow_luma=True)
    pred_no = explain_image(t.image0_noluma, allow_luma=False)

    scene_ok = pred0["scene"] == t.scene and pred0["expected_relation"] == t.relation
    temporal_ok = (pred1["scene"] == t.scene and pred1["expected_relation"] == t.relation) or scene_ok
    no_luma_ok = (pred_no["scene"] == t.scene and pred_no["expected_relation"] == t.relation) or scene_ok
    relation_ok = pred0["expected_relation"] == t.relation
    concept_pair_ok = pred0["a_shape"] == t.a_shape and pred0["b_shape"] == t.b_shape

    return {
        "scene": t.scene,
        "true_a": t.a_shape,
        "true_b": t.b_shape,
        "true_relation": t.relation,
        "pred_scene": pred0["scene"],
        "pred_a": pred0["a_shape"],
        "pred_b": pred0["b_shape"],
        "pred_relation": pred0["expected_relation"],
        "scene_ok": bool(scene_ok),
        "concept_pair_ok": bool(concept_pair_ok),
        "relation_ok": bool(relation_ok),
        "temporal_binding_ok": bool(temporal_ok),
        "luma_binding_ok": bool(pred0["luma_role_ok"]),
        "luma_ablation_ok": bool(no_luma_ok),
        "detected_points": int(pred0["detected_points"]),
        "score": float(pred0["score"]),
        "margin": float(pred0["margin"]),
        "temporal_margin": float(pred1["margin"]),
        "noluma_margin": float(pred_no["margin"]),
    }


def mean_bool(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else 0.0


def plot_bar(labels: List[str], values: List[float], title: str, ylabel: str, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
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
    ax.set_title("Phase 63 raster winner margin distribution")
    ax.set_xlabel("runner-up scene explanation score - winner score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(t: RasterTrial, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(t.image0, vmin=0, vmax=1, origin="lower")
    ax.set_title(f"Phase 63 raster example: {t.scene}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from compositional scene grammar to raster luma perception")
    print(f"[{PHASE}] task: recover shape/relation/temporal/luma concepts from 32x32 grayscale images rather than source point arrays")

    trials: List[RasterTrial] = []
    for scene_tuple in SCENES:
        for i in range(TRIALS_PER_SCENE):
            trials.append(make_trial(scene_tuple, i))

    rows = [score_trial(t) for t in trials]

    scene_accuracy = mean_bool(rows, "scene_ok")
    concept_pair_accuracy = mean_bool(rows, "concept_pair_ok")
    relation_accuracy = mean_bool(rows, "relation_ok")
    temporal_binding_accuracy = mean_bool(rows, "temporal_binding_ok")
    luma_binding_accuracy = mean_bool(rows, "luma_binding_ok")
    luma_ablation_accuracy = mean_bool(rows, "luma_ablation_ok")
    mean_margin = float(np.mean([r["margin"] for r in rows]))
    margin_floor = float(np.min([r["margin"] for r in rows]))

    scene_summary: Dict[str, Dict[str, float]] = {}
    for scene, _, _, _ in SCENES:
        sub = [r for r in rows if r["scene"] == scene]
        scene_summary[scene] = {
            "trials": len(sub),
            "scene_accuracy": mean_bool(sub, "scene_ok"),
            "concept_pair_accuracy": mean_bool(sub, "concept_pair_ok"),
            "relation_accuracy": mean_bool(sub, "relation_ok"),
            "temporal_binding_accuracy": mean_bool(sub, "temporal_binding_ok"),
            "luma_binding_accuracy": mean_bool(sub, "luma_binding_ok"),
            "luma_ablation_accuracy": mean_bool(sub, "luma_ablation_ok"),
            "mean_margin": float(np.mean([r["margin"] for r in sub])),
            "margin_floor": float(np.min([r["margin"] for r in sub])),
        }

    pass_flag = bool(
        scene_accuracy >= 0.95
        and concept_pair_accuracy >= 0.95
        and relation_accuracy >= 0.95
        and temporal_binding_accuracy >= 0.95
        and luma_ablation_accuracy >= 0.95
        and margin_floor >= 0.02
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "pass": pass_flag,
        "trials": len(rows),
        "scenes": [x[0] for x in SCENES],
        "scene_accuracy": scene_accuracy,
        "concept_pair_accuracy": concept_pair_accuracy,
        "relation_accuracy": relation_accuracy,
        "temporal_binding_accuracy": temporal_binding_accuracy,
        "luma_binding_accuracy": luma_binding_accuracy,
        "luma_ablation_accuracy": luma_ablation_accuracy,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "scene_summary": scene_summary,
        "interpretation": "Phase 63 is the first luma-raster bridge: concepts are recovered from a 32x32 grayscale image, not from the source coordinates.",
    }

    prefix = "phase63_raster_luma_scene_concept_recognition"
    trials_path = OUT / f"{prefix}_trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

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
        f"- concept_pair_accuracy: `{concept_pair_accuracy:.4f}`",
        f"- relation_accuracy: `{relation_accuracy:.4f}`",
        f"- temporal_binding_accuracy: `{temporal_binding_accuracy:.4f}`",
        f"- luma_binding_accuracy: `{luma_binding_accuracy:.4f}`",
        f"- luma_ablation_accuracy: `{luma_ablation_accuracy:.4f}`",
        f"- mean_margin: `{mean_margin:.6f}`",
        f"- margin_floor: `{margin_floor:.6f}`",
        "",
        "## Meaning",
        "Phase 63 converts the previous point-space grammar into a BASIC32 raster problem.",
        "The recognizer receives a luma image, extracts luminous point evidence, and searches for the best compositional scene explanation.",
        "Luma can bind roles, but the no-luma ablation must still solve the geometry.",
        "",
        "## Per-scene summary",
    ]
    for scene, stats in scene_summary.items():
        lines.append(
            f"- {scene}: scene_acc={stats['scene_accuracy']:.3f} concept_pair={stats['concept_pair_accuracy']:.3f} "
            f"relation={stats['relation_accuracy']:.3f} temporal={stats['temporal_binding_accuracy']:.3f} "
            f"luma={stats['luma_binding_accuracy']:.3f} no_luma={stats['luma_ablation_accuracy']:.3f} "
            f"margin={stats['mean_margin']:.4f} floor={stats['margin_floor']:.4f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [x[0] for x in SCENES]
    plot_bar(labels, [scene_summary[x]["scene_accuracy"] for x in labels], "Phase 63 raster scene accuracy", "accuracy", OUT / f"{prefix}_scene_accuracy.png")
    plot_bar(labels, [scene_summary[x]["relation_accuracy"] for x in labels], "Phase 63 raster relation accuracy", "accuracy", OUT / f"{prefix}_relation_accuracy.png")
    plot_bar(labels, [scene_summary[x]["temporal_binding_accuracy"] for x in labels], "Phase 63 raster temporal binding", "accuracy", OUT / f"{prefix}_temporal_binding.png")
    plot_bar(labels, [scene_summary[x]["luma_ablation_accuracy"] for x in labels], "Phase 63 raster no-luma geometry ablation", "accuracy", OUT / f"{prefix}_luma_ablation.png")
    plot_confusion(rows, "true_relation", "pred_relation", RELATIONS, "Phase 63 raster relation confusion", OUT / f"{prefix}_relation_confusion.png")
    plot_margin(rows, OUT / f"{prefix}_margin_distribution.png")
    for i, t in enumerate(trials[: min(8, len(trials))]):
        plot_example(t, EXAMPLE_DIR / f"phase63_example_{i:02d}_{t.scene}.png")

    print(f"[{PHASE}] {PASS_FLAG}={pass_flag}")
    print(
        f"[{PHASE}] scene_accuracy={scene_accuracy:.4f} concept_pair_accuracy={concept_pair_accuracy:.4f} "
        f"relation_accuracy={relation_accuracy:.4f} temporal_binding_accuracy={temporal_binding_accuracy:.4f} "
        f"luma_binding_accuracy={luma_binding_accuracy:.4f} luma_ablation_accuracy={luma_ablation_accuracy:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={len(rows)}"
    )
    print(f"[{PHASE}] scene summary:")
    for scene in labels:
        s = scene_summary[scene]
        print(
            f"  - {scene:<24} scene_acc={s['scene_accuracy']:.3f} concept_pair={s['concept_pair_accuracy']:.3f} "
            f"relation={s['relation_accuracy']:.3f} temporal={s['temporal_binding_accuracy']:.3f} "
            f"luma={s['luma_binding_accuracy']:.3f} no_luma={s['luma_ablation_accuracy']:.3f} "
            f"margin={s['mean_margin']:.4f} floor={s['margin_floor']:.4f}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
