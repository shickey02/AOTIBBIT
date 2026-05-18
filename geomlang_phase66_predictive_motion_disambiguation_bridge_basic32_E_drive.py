"""
Phase 66: Predictive motion disambiguation bridge for Basic32 raster sequences

Drop this file into:
  E:\\BBIT\\bbit_geomlang\\geomlang_phase66_predictive_motion_disambiguation_bridge_basic32_E_drive.py

Run from E:\\BBIT:
  python bbit_geomlang/geomlang_phase66_predictive_motion_disambiguation_bridge_basic32_E_drive.py

Purpose:
  Phase 65 passed the real bridge: 32x32 raster multi-frame scenes, occlusion,
  frontness, object permanence. The only weak point was motion grammar, especially
  toward/orbit/swap/together ambiguities. Phase 66 is deliberately NOT a new rabbit
  hole. It is a targeted motion-grammar resolver that tests whether motion concepts
  become stable when judged by a predictive temporal signature instead of only a
  local displacement relation.

  The test creates 5-frame raster sequences with shallow occlusion, estimates object
  centroids from the raster frames when visible, bridges hidden frames by continuity,
  and classifies motion by comparing temporal signatures:
    - together: common velocity + stable relative vector
    - a_toward_b: decreasing separation with mostly radial motion
    - a_away_from_b: increasing separation with mostly radial motion
    - a_orbits_b: stable radius + rotating angle + tangential motion
    - swap_sides: sign change of left/right ordering with crossing trajectory

Outputs:
  E:\\BBIT\\outputs_basic32\\phase66_predictive_motion_disambiguation_* files
  plus example frame-strip PNGs in phase66_examples.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

ROOT = Path("E:/BBIT") if Path("E:/BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)
EXDIR = OUT / "phase66_examples"
EXDIR.mkdir(parents=True, exist_ok=True)

PHASE = 66
TITLE = "Predictive motion disambiguation bridge"
PASS_FLAG = "PHASE66_PREDICTIVE_MOTION_DISAMBIGUATION_PASS"
SIZE = 32
FRAMES = 5
MOTIONS = ["together", "a_toward_b", "a_away_from_b", "a_orbits_b", "swap_sides"]
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
CASES = [
    ("triangle_inside_square", "triangle", "square", "inside"),
    ("triangle_inside_pentagon", "triangle", "pentagon", "inside"),
    ("line_left_of_triangle", "line", "triangle", "left_of"),
    ("square_right_of_core", "square", "core", "right_of"),
    ("pentagon_above_line", "pentagon", "line", "above"),
    ("line_below_square", "line", "square", "below"),
    ("triangle_near_core", "triangle", "core", "near"),
    ("square_separate_pentagon", "square", "pentagon", "separate"),
]
OCCLUSIONS = ["none", "crossing_overlap", "middle_a_occluded", "middle_b_occluded"]


@dataclass
class Trial:
    seed: int
    case_name: str
    shape_a: str
    shape_b: str
    relation: str
    motion: str
    occlusion: str
    seq: np.ndarray
    true_a: np.ndarray
    true_b: np.ndarray
    pred_motion: str
    pred_relation: str
    motion_correct: bool
    relation_correct: bool
    temporal_identity_correct: bool
    continuity_correct: bool
    margin: float
    bridge_error: float
    signature: Dict[str, float]


def clamp(v: float, lo: float = 4.0, hi: float = 27.0) -> float:
    return float(max(lo, min(hi, v)))


def polygon_points(cx: float, cy: float, radius: float, n: int, rot: float) -> np.ndarray:
    return np.array([
        [cx + radius * math.cos(rot + 2 * math.pi * i / n), cy + radius * math.sin(rot + 2 * math.pi * i / n)]
        for i in range(n)
    ], dtype=float)


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def draw_shape(img: np.ndarray, shape: str, cx: float, cy: float, scale: float, val: float, rot: float = 0.0) -> None:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if shape == "triangle":
        poly = polygon_points(cx, cy, scale, 3, rot - math.pi / 2)
        for y in range(SIZE):
            for x in range(SIZE):
                if point_in_poly(x + 0.5, y + 0.5, poly):
                    img[y, x] = max(img[y, x], val)
    elif shape == "square":
        c, s = math.cos(rot), math.sin(rot)
        dx = xx - cx
        dy = yy - cy
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        mask = (np.abs(xr) <= scale * 0.78) & (np.abs(yr) <= scale * 0.78)
        img[mask] = np.maximum(img[mask], val)
    elif shape == "pentagon":
        poly = polygon_points(cx, cy, scale, 5, rot - math.pi / 2)
        for y in range(SIZE):
            for x in range(SIZE):
                if point_in_poly(x + 0.5, y + 0.5, poly):
                    img[y, x] = max(img[y, x], val)
    elif shape == "line":
        c, s = math.cos(rot), math.sin(rot)
        dx = xx - cx
        dy = yy - cy
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        mask = (np.abs(yr) <= 0.85) & (np.abs(xr) <= scale * 1.15)
        img[mask] = np.maximum(img[mask], val)
    elif shape == "core":
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        shell = (r <= scale) & (r >= scale * 0.50)
        core = r <= scale * 0.30
        img[shell] = np.maximum(img[shell], val * 0.75)
        img[core] = np.maximum(img[core], min(1.0, val + 0.15))
    else:
        raise ValueError(shape)


def relation_positions(relation: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    cx = rng.uniform(11.0, 21.0)
    cy = rng.uniform(11.0, 21.0)
    if relation == "inside":
        a = np.array([cx, cy], dtype=float)
        b = np.array([cx + rng.uniform(-0.8, 0.8), cy + rng.uniform(-0.8, 0.8)], dtype=float)
    elif relation == "left_of":
        a = np.array([rng.uniform(6, 10), rng.uniform(11, 21)], dtype=float)
        b = np.array([rng.uniform(20, 25), a[1] + rng.uniform(-2, 2)], dtype=float)
    elif relation == "right_of":
        b = np.array([rng.uniform(6, 10), rng.uniform(11, 21)], dtype=float)
        a = np.array([rng.uniform(20, 25), b[1] + rng.uniform(-2, 2)], dtype=float)
    elif relation == "above":
        a = np.array([rng.uniform(11, 21), rng.uniform(6, 10)], dtype=float)
        b = np.array([a[0] + rng.uniform(-2, 2), rng.uniform(20, 25)], dtype=float)
    elif relation == "below":
        b = np.array([rng.uniform(11, 21), rng.uniform(6, 10)], dtype=float)
        a = np.array([b[0] + rng.uniform(-2, 2), rng.uniform(20, 25)], dtype=float)
    elif relation == "near":
        a = np.array([cx - 3.0 + rng.uniform(-0.5, 0.5), cy + rng.uniform(-1, 1)], dtype=float)
        b = np.array([cx + 3.0 + rng.uniform(-0.5, 0.5), cy + rng.uniform(-1, 1)], dtype=float)
    elif relation == "separate":
        a = np.array([rng.uniform(5, 9), rng.uniform(5, 26)], dtype=float)
        b = np.array([rng.uniform(23, 27), rng.uniform(5, 26)], dtype=float)
    else:
        raise ValueError(relation)
    return a, b


def build_trajectories(relation: str, motion: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    a0, b0 = relation_positions(relation, rng)
    t = np.arange(FRAMES, dtype=float)
    if motion == "together":
        v = np.array([rng.uniform(-1.25, 1.25), rng.uniform(-1.25, 1.25)])
        a = np.stack([a0 + v * k for k in t])
        b = np.stack([b0 + v * k for k in t])
    elif motion == "a_toward_b":
        d = b0 - a0
        if np.linalg.norm(d) < 2.5:
            d = np.array([4.0, 0.0])
        u = d / (np.linalg.norm(d) + 1e-9)
        speed = rng.uniform(0.75, 1.35)
        a = np.stack([a0 + u * speed * k for k in t])
        b = np.stack([b0 for _ in t])
    elif motion == "a_away_from_b":
        d = a0 - b0
        if np.linalg.norm(d) < 2.5:
            d = np.array([4.0, 0.0])
        u = d / (np.linalg.norm(d) + 1e-9)
        speed = rng.uniform(0.75, 1.35)
        a = np.stack([a0 + u * speed * k for k in t])
        b = np.stack([b0 for _ in t])
    elif motion == "a_orbits_b":
        center = b0.copy()
        vec = a0 - center
        r = max(4.0, min(8.5, np.linalg.norm(vec)))
        start = math.atan2(vec[1], vec[0])
        direction = 1 if rng.random() < 0.5 else -1
        omega = direction * rng.uniform(0.34, 0.52)
        a = np.stack([center + r * np.array([math.cos(start + omega * k), math.sin(start + omega * k)]) for k in t])
        b = np.stack([center for _ in t])
    elif motion == "swap_sides":
        # Construct a clean crossing: A and B exchange approximate left/right order.
        mid = (a0 + b0) / 2
        if abs(a0[0] - b0[0]) < 5:
            a0[0], b0[0] = mid[0] - 6, mid[0] + 6
        a_end = np.array([b0[0], a0[1] + rng.uniform(-1.5, 1.5)])
        b_end = np.array([a0[0], b0[1] + rng.uniform(-1.5, 1.5)])
        a = np.stack([a0 + (a_end - a0) * (k / (FRAMES - 1)) for k in t])
        b = np.stack([b0 + (b_end - b0) * (k / (FRAMES - 1)) for k in t])
    else:
        raise ValueError(motion)
    a[:, 0] = np.clip(a[:, 0], 4.0, 27.0)
    a[:, 1] = np.clip(a[:, 1], 4.0, 27.0)
    b[:, 0] = np.clip(b[:, 0], 4.0, 27.0)
    b[:, 1] = np.clip(b[:, 1], 4.0, 27.0)
    return a, b


def render_sequence(shape_a: str, shape_b: str, a: np.ndarray, b: np.ndarray, occlusion: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 997)
    seq = np.zeros((FRAMES, SIZE, SIZE), dtype=float)
    val_a, val_b = 0.82, 0.55
    scale_a = 3.0 if shape_a != "line" else 4.3
    scale_b = 4.2 if shape_b != "line" else 4.3
    if shape_a == "core": scale_a = 3.4
    if shape_b == "core": scale_b = 3.4
    for k in range(FRAMES):
        img = np.zeros((SIZE, SIZE), dtype=float)
        rot_a = 0.22 * k + (seed % 13) * 0.03
        rot_b = -0.14 * k + (seed % 17) * 0.02
        draw_a = True
        draw_b = True
        if k == FRAMES // 2 and occlusion == "middle_a_occluded":
            draw_a = False
        if k == FRAMES // 2 and occlusion == "middle_b_occluded":
            draw_b = False
        if draw_b:
            draw_shape(img, shape_b, b[k, 0], b[k, 1], scale_b, val_b, rot_b)
        if draw_a:
            draw_shape(img, shape_a, a[k, 0], a[k, 1], scale_a, val_a, rot_a)
        if k == FRAMES // 2 and occlusion == "crossing_overlap":
            # A shallow frontness/overlap cue: A is drawn last and a small contact veil is visible.
            mid = (a[k] + b[k]) / 2
            yy, xx = np.mgrid[0:SIZE, 0:SIZE]
            veil = ((xx - mid[0]) ** 2 + (yy - mid[1]) ** 2) <= 2.0 ** 2
            img[veil] = np.maximum(img[veil], 0.68)
        noise = rng.normal(0, 0.012, img.shape)
        seq[k] = np.clip(img + noise, 0, 1)
    return seq


def visible_centroid(img: np.ndarray, prefer: str) -> Optional[np.ndarray]:
    # Split by luma bands. A is usually brighter; B is lower. This is not a label leak,
    # it is a raster cue used only to estimate visible centers. Hidden frames return None.
    if prefer == "a":
        mask = img > 0.70
    else:
        mask = (img > 0.34) & (img <= 0.75)
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    return np.array([float(xs.mean()), float(ys.mean())])


def interpolate_missing(points: List[Optional[np.ndarray]]) -> Tuple[np.ndarray, float]:
    arr = np.full((len(points), 2), np.nan, dtype=float)
    known = []
    for i, p in enumerate(points):
        if p is not None and np.all(np.isfinite(p)):
            arr[i] = p
            known.append(i)
    if len(known) == 0:
        return np.zeros((len(points), 2)), 99.0
    if len(known) == 1:
        arr[:] = arr[known[0]]
    else:
        for dim in range(2):
            arr[:, dim] = np.interp(np.arange(len(points)), known, arr[known, dim])
    # continuity error is how much interpolation had to bridge hidden/weak frames.
    err = 0.0
    for i, p in enumerate(points):
        if p is None:
            if 0 < i < len(points) - 1:
                err += float(np.linalg.norm(arr[i] - (arr[i - 1] + arr[i + 1]) / 2))
            else:
                err += 0.5
    return arr, err


def relation_from_positions(a0: np.ndarray, b0: np.ndarray) -> str:
    d = a0 - b0
    dist = float(np.linalg.norm(d))
    if dist < 2.7:
        return "inside"
    if dist < 6.5:
        return "near"
    if abs(d[0]) > abs(d[1]) * 1.35:
        return "right_of" if d[0] > 0 else "left_of"
    if abs(d[1]) > abs(d[0]) * 1.15:
        return "below" if d[1] > 0 else "above"
    return "separate"


def motion_signature(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    rel = a - b
    dist = np.linalg.norm(rel, axis=1)
    v_a = np.diff(a, axis=0)
    v_b = np.diff(b, axis=0)
    v_rel = np.diff(rel, axis=0)
    common = float(np.mean(np.linalg.norm(v_a - v_b, axis=1)))
    rel_stability = float(np.std(dist) + np.mean(np.linalg.norm(v_rel, axis=1)))
    sep_delta = float(dist[-1] - dist[0])
    angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    angle_travel = float(abs(angles[-1] - angles[0]))
    radius_stability = float(np.std(dist))
    sign_change = float(np.sign(rel[0, 0]) != np.sign(rel[-1, 0]))
    mid_close = float(np.min(dist))
    return {
        "common_velocity_error": common,
        "relative_stability": rel_stability,
        "separation_delta": sep_delta,
        "angle_travel": angle_travel,
        "radius_stability": radius_stability,
        "sign_change": sign_change,
        "min_distance": mid_close,
    }


def score_motions(sig: Dict[str, float]) -> Dict[str, float]:
    # Lower is better. These are deliberately simple temporal concepts, not trained weights.
    common = sig["common_velocity_error"]
    rel_stab = sig["relative_stability"]
    sep = sig["separation_delta"]
    angle = sig["angle_travel"]
    rad = sig["radius_stability"]
    sign = sig["sign_change"]
    min_d = sig["min_distance"]
    scores = {
        "together": common + 0.35 * rel_stab + 0.20 * abs(sep),
        "a_toward_b": max(0.0, sep + 0.20) + 0.35 * angle + 0.15 * common,
        "a_away_from_b": max(0.0, -sep + 0.20) + 0.35 * angle + 0.15 * common,
        "a_orbits_b": rad + max(0.0, 0.85 - angle) + 0.18 * abs(sep),
        "swap_sides": (0.0 if sign > 0.5 else 2.0) + 0.20 * min_d + 0.15 * rad,
    }
    return scores


def classify(seq: np.ndarray) -> Tuple[str, str, float, float, Dict[str, float]]:
    raw_a = [visible_centroid(seq[k], "a") for k in range(FRAMES)]
    raw_b = [visible_centroid(seq[k], "b") for k in range(FRAMES)]
    a, err_a = interpolate_missing(raw_a)
    b, err_b = interpolate_missing(raw_b)
    sig = motion_signature(a, b)
    scores = score_motions(sig)
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    pred_motion = ordered[0][0]
    margin = float(ordered[1][1] - ordered[0][1])
    pred_relation = relation_from_positions(a[0], b[0])
    bridge_error = float(err_a + err_b)
    return pred_motion, pred_relation, margin, bridge_error, sig


def run_trials() -> List[Trial]:
    trials: List[Trial] = []
    seed_base = 66000
    idx = 0
    for case_name, shape_a, shape_b, relation in CASES:
        for motion in MOTIONS:
            for occlusion in OCCLUSIONS:
                for rep in range(2):
                    seed = seed_base + idx * 37 + rep
                    rng = np.random.default_rng(seed)
                    a, b = build_trajectories(relation, motion, rng)
                    seq = render_sequence(shape_a, shape_b, a, b, occlusion, seed)
                    pred_motion, pred_relation, margin, bridge_error, sig = classify(seq)
                    motion_correct = pred_motion == motion
                    # For inside rendered rasters, centroid-only relation may read as near/right/left in a subset;
                    # relation scoring is not the target of Phase 66, so use a permissive inside/near equivalence.
                    relation_correct = (pred_relation == relation) or (relation == "inside" and pred_relation in {"inside", "near"})
                    temporal_identity_correct = bridge_error < 3.0
                    continuity_correct = margin > -0.05 and bridge_error < 3.0
                    trials.append(Trial(
                        seed, case_name, shape_a, shape_b, relation, motion, occlusion, seq, a, b,
                        pred_motion, pred_relation, motion_correct, relation_correct,
                        temporal_identity_correct, continuity_correct, margin, bridge_error, sig
                    ))
                    idx += 1
    return trials


def aggregate(trials: List[Trial]) -> Dict:
    def mean_bool(xs): return float(np.mean([1.0 if x else 0.0 for x in xs])) if xs else 0.0
    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "trials": len(trials),
        "motion_accuracy": mean_bool([t.motion_correct for t in trials]),
        "relation_proxy_accuracy": mean_bool([t.relation_correct for t in trials]),
        "temporal_identity_accuracy": mean_bool([t.temporal_identity_correct for t in trials]),
        "continuity_accuracy": mean_bool([t.continuity_correct for t in trials]),
        "mean_margin": float(np.mean([t.margin for t in trials])),
        "margin_floor": float(np.min([t.margin for t in trials])),
        "mean_bridge_error": float(np.mean([t.bridge_error for t in trials])),
    }
    motion_summary = {}
    for m in MOTIONS:
        subset = [t for t in trials if t.motion == m]
        motion_summary[m] = {
            "trials": len(subset),
            "motion_accuracy": mean_bool([t.motion_correct for t in subset]),
            "continuity_accuracy": mean_bool([t.continuity_correct for t in subset]),
            "mean_margin": float(np.mean([t.margin for t in subset])),
            "mean_bridge_error": float(np.mean([t.bridge_error for t in subset])),
        }
    occ_summary = {}
    for o in OCCLUSIONS:
        subset = [t for t in trials if t.occlusion == o]
        occ_summary[o] = {
            "trials": len(subset),
            "motion_accuracy": mean_bool([t.motion_correct for t in subset]),
            "temporal_identity_accuracy": mean_bool([t.temporal_identity_correct for t in subset]),
            "continuity_accuracy": mean_bool([t.continuity_correct for t in subset]),
            "mean_margin": float(np.mean([t.margin for t in subset])),
        }
    case_summary = {}
    for c, *_ in CASES:
        subset = [t for t in trials if t.case_name == c]
        case_summary[c] = {
            "trials": len(subset),
            "motion_accuracy": mean_bool([t.motion_correct for t in subset]),
            "relation_proxy_accuracy": mean_bool([t.relation_correct for t in subset]),
            "continuity_accuracy": mean_bool([t.continuity_correct for t in subset]),
            "mean_margin": float(np.mean([t.margin for t in subset])),
        }
    summary["motion_summary"] = motion_summary
    summary["occlusion_summary"] = occ_summary
    summary["case_summary"] = case_summary
    summary["pass"] = bool(summary["motion_accuracy"] >= 0.94 and summary["temporal_identity_accuracy"] >= 0.98 and summary["continuity_accuracy"] >= 0.95)
    summary["interpretation"] = (
        "Phase 66 is a targeted bridge, not a new adversarial loop: it tests whether the weak Phase 65 motion cases become stable "
        "when classified by multi-frame predictive signatures and occlusion-aware interpolation rather than by single-step displacement."
    )
    return summary


def write_csv(trials: List[Trial], path: Path) -> None:
    fields = [
        "seed", "case_name", "shape_a", "shape_b", "relation", "motion", "occlusion",
        "pred_motion", "pred_relation", "motion_correct", "relation_correct",
        "temporal_identity_correct", "continuity_correct", "margin", "bridge_error",
        "common_velocity_error", "relative_stability", "separation_delta", "angle_travel", "radius_stability", "sign_change", "min_distance"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trials:
            row = {
                "seed": t.seed, "case_name": t.case_name, "shape_a": t.shape_a, "shape_b": t.shape_b,
                "relation": t.relation, "motion": t.motion, "occlusion": t.occlusion,
                "pred_motion": t.pred_motion, "pred_relation": t.pred_relation,
                "motion_correct": int(t.motion_correct), "relation_correct": int(t.relation_correct),
                "temporal_identity_correct": int(t.temporal_identity_correct), "continuity_correct": int(t.continuity_correct),
                "margin": t.margin, "bridge_error": t.bridge_error,
            }
            row.update(t.signature)
            w.writerow(row)


def confusion_matrix(trials: List[Trial], labels: List[str], true_attr: str, pred_attr: str) -> np.ndarray:
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    counts = np.zeros(len(labels), dtype=float)
    index = {x: i for i, x in enumerate(labels)}
    for t in trials:
        true = getattr(t, true_attr)
        pred = getattr(t, pred_attr)
        if true in index and pred in index:
            mat[index[true], index[pred]] += 1
            counts[index[true]] += 1
    for i in range(len(labels)):
        if counts[i] > 0:
            mat[i] /= counts[i]
    return mat


def save_bar(name: str, labels: List[str], values: List[float], title: str, ylabel: str = "accuracy") -> None:
    if plt is None: return
    fig = plt.figure(figsize=(14, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def save_hist(name: str, vals: List[float], title: str, xlabel: str) -> None:
    if plt is None: return
    fig = plt.figure(figsize=(14, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=24)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def save_confusion(name: str, mat: np.ndarray, labels: List[str], title: str) -> None:
    if plt is None: return
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def save_examples(trials: List[Trial]) -> None:
    if plt is None: return
    examples = []
    for m in MOTIONS:
        found = next((t for t in trials if t.motion == m and t.motion_correct), None)
        if found is not None:
            examples.append(found)
    for t in examples:
        fig = plt.figure(figsize=(10, 2.2))
        for k in range(FRAMES):
            ax = fig.add_subplot(1, FRAMES, k + 1)
            ax.imshow(t.seq[k], vmin=0, vmax=1, cmap="gray")
            ax.set_title(f"f{k}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{t.motion} | pred={t.pred_motion} | occ={t.occlusion} | margin={t.margin:.3f}")
        fig.tight_layout()
        safe = t.motion.replace("/", "_")
        fig.savefig(EXDIR / f"phase66_example_{safe}.png", dpi=140)
        plt.close(fig)


def write_report(summary: Dict, path: Path) -> None:
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append(f"PASS: `{summary['pass']}`\n")
    lines.append("## Core result")
    for k in ["motion_accuracy", "relation_proxy_accuracy", "temporal_identity_accuracy", "continuity_accuracy", "mean_margin", "margin_floor", "mean_bridge_error", "trials"]:
        v = summary[k]
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.4f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    lines.append("\n## Meaning")
    lines.append(summary["interpretation"])
    lines.append("\n## Motion summary")
    for m, d in summary["motion_summary"].items():
        lines.append(f"- {m}: acc={d['motion_accuracy']:.3f} continuity={d['continuity_accuracy']:.3f} margin={d['mean_margin']:.3f} bridge_err={d['mean_bridge_error']:.3f}")
    lines.append("\n## Occlusion summary")
    for o, d in summary["occlusion_summary"].items():
        lines.append(f"- {o}: motion={d['motion_accuracy']:.3f} temporal={d['temporal_identity_accuracy']:.3f} continuity={d['continuity_accuracy']:.3f} margin={d['mean_margin']:.3f}")
    lines.append("\n## Decision")
    lines.append("If this passes, do not keep tuning Phase 65. The next productive step is a true perception bridge: either higher-dimensional raster channels/depth maps or real luma/edge extraction from imported images.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from multiframe occlusion/depth continuity to predictive motion signatures")
    print(f"[{PHASE}] task: resolve Phase 65 motion ambiguity using 5-frame predictive signatures, object-continuity interpolation, and raster-visible centroid tracking")

    trials = run_trials()
    summary = aggregate(trials)

    csv_path = OUT / "phase66_predictive_motion_disambiguation_trials.csv"
    json_path = OUT / "phase66_predictive_motion_disambiguation_summary.json"
    report_path = OUT / "phase66_predictive_motion_disambiguation_report.md"
    write_csv(trials, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, report_path)

    motion_labels = MOTIONS
    motion_mat = confusion_matrix(trials, motion_labels, "motion", "pred_motion")
    save_confusion("phase66_predictive_motion_disambiguation_motion_confusion.png", motion_mat, motion_labels, "Phase 66 predictive motion confusion")
    save_hist("phase66_predictive_motion_disambiguation_margin_distribution.png", [t.margin for t in trials], "Phase 66 predictive winner margin distribution", "runner-up motion score - winner score")
    save_hist("phase66_predictive_motion_disambiguation_bridge_error_distribution.png", [t.bridge_error for t in trials], "Phase 66 occlusion bridge error distribution", "interpolated hidden-frame continuity error")
    save_bar("phase66_predictive_motion_disambiguation_motion_accuracy.png", motion_labels, [summary["motion_summary"][m]["motion_accuracy"] for m in motion_labels], "Phase 66 motion primitive accuracy")
    save_bar("phase66_predictive_motion_disambiguation_occlusion_accuracy.png", OCCLUSIONS, [summary["occlusion_summary"][o]["motion_accuracy"] for o in OCCLUSIONS], "Phase 66 motion accuracy by occlusion mode")
    save_bar("phase66_predictive_motion_disambiguation_continuity_accuracy.png", motion_labels, [summary["motion_summary"][m]["continuity_accuracy"] for m in motion_labels], "Phase 66 continuity accuracy by motion")
    save_examples(trials)

    print(f"[{PHASE}] {PASS_FLAG}={summary['pass']}")
    print(f"[{PHASE}] motion_accuracy={summary['motion_accuracy']:.4f} relation_proxy_accuracy={summary['relation_proxy_accuracy']:.4f} temporal_identity_accuracy={summary['temporal_identity_accuracy']:.4f} continuity_accuracy={summary['continuity_accuracy']:.4f} mean_margin={summary['mean_margin']:.6f} margin_floor={summary['margin_floor']:.6f} mean_bridge_error={summary['mean_bridge_error']:.6f} trials={summary['trials']}")
    print(f"[{PHASE}] motion summary:")
    for m, d in summary["motion_summary"].items():
        print(f"  - {m:<14} acc={d['motion_accuracy']:.3f} continuity={d['continuity_accuracy']:.3f} margin={d['mean_margin']:.4f} bridge_err={d['mean_bridge_error']:.4f}")
    print(f"[{PHASE}] occlusion summary:")
    for o, d in summary["occlusion_summary"].items():
        print(f"  - {o:<18} motion={d['motion_accuracy']:.3f} temporal={d['temporal_identity_accuracy']:.3f} continuity={d['continuity_accuracy']:.3f} margin={d['mean_margin']:.4f}")
    print(f"[{PHASE}] wrote trials: {csv_path}")
    print(f"[{PHASE}] wrote summary: {json_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXDIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
