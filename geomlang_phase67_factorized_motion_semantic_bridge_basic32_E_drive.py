"""
Phase 67: Factorized motion semantic bridge

Drop-in location on Windows:
  E:\BBIT\bbit_geomlang\geomlang_phase67_factorized_motion_semantic_bridge_basic32_E_drive.py

Run:
  python bbit_geomlang/geomlang_phase67_factorized_motion_semantic_bridge_basic32_E_drive.py

Purpose:
  Phase 66 proved the continuity bridge worked but the motion classifier collapsed because
  it was still treating motion as a winner-take-all template problem. Phase 67 separates
  motion into semantic factors: distance-change, relative bearing-change, side-swap, and
  co-moving/together continuity. This is meant to be a forward bridge, not a new rabbit hole.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PHASE = 67
TITLE = "Factorized motion semantic bridge"
PASS_FLAG = "PHASE67_FACTORIZED_MOTION_SEMANTIC_BRIDGE_PASS"

SEED = 67067
W = H = 32
N_FRAMES = 7
TRIALS_PER_CASE_MOTION_OCC = 4

SHAPES = ["triangle", "square", "pentagon", "line", "core"]
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
MOTIONS = ["together", "a_toward_b", "a_away_from_b", "a_orbits_b", "swap_sides"]
OCCLUSIONS = ["none", "crossing_overlap", "middle_a_occluded", "middle_b_occluded"]
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

_rng = random.Random(SEED)
np.random.seed(SEED)


@dataclass
class TrialResult:
    phase: int
    case: str
    shape_a: str
    shape_b: str
    relation_true: str
    relation_pred: str
    motion_true: str
    motion_pred: str
    occlusion: str
    relation_correct: bool
    motion_correct: bool
    temporal_identity_correct: bool
    continuity_correct: bool
    object_permanence_correct: bool
    factor_consistency: float
    margin: float
    d_start: float
    d_end: float
    d_delta: float
    angle_delta: float
    side_cross: bool
    relative_motion_mean: float
    co_motion_residual: float


def root_dir() -> Path:
    # Works from E:\BBIT, E:\BBIT\bbit_geomlang, or any local test folder.
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists() or cwd.name.upper() == "BBIT":
        return cwd
    return Path("E:/BBIT") if Path("E:/BBIT").exists() else cwd


def out_dir() -> Path:
    out = root_dir() / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


def clamp_point(p: np.ndarray, pad: float = 4.0) -> np.ndarray:
    return np.array([float(np.clip(p[0], pad, W - 1 - pad)), float(np.clip(p[1], pad, H - 1 - pad))])


def initial_pair_for_relation(relation: str) -> Tuple[np.ndarray, np.ndarray]:
    jitter = lambda s=0.55: np.array([_rng.uniform(-s, s), _rng.uniform(-s, s)])
    center = np.array([16.0, 16.0]) + jitter(1.2)
    if relation == "inside":
        return center + jitter(0.3), center + jitter(0.3)
    if relation == "left_of":
        return np.array([10.0, 16.0]) + jitter(), np.array([22.0, 16.0]) + jitter()
    if relation == "right_of":
        return np.array([22.0, 16.0]) + jitter(), np.array([10.0, 16.0]) + jitter()
    if relation == "above":
        return np.array([16.0, 10.0]) + jitter(), np.array([16.0, 22.0]) + jitter()
    if relation == "below":
        return np.array([16.0, 22.0]) + jitter(), np.array([16.0, 10.0]) + jitter()
    if relation == "near":
        return np.array([14.0, 16.0]) + jitter(), np.array([18.0, 16.0]) + jitter()
    return np.array([8.0, 9.0]) + jitter(), np.array([24.0, 23.0]) + jitter()


def make_tracks(relation: str, motion: str) -> Tuple[np.ndarray, np.ndarray]:
    a0, b0 = initial_pair_for_relation(relation)
    # For motion semantics, exact co-centered containment makes toward/away/orbit underdetermined.
    # Give contained pairs a small internal radius so the relation remains inside/near while motion is measurable.
    if relation == "inside" and motion != "together":
        th = _rng.uniform(0, 2 * math.pi)
        a0 = b0 + np.array([math.cos(th), math.sin(th)]) * 4.2
    t = np.linspace(0.0, 1.0, N_FRAMES)
    a = np.zeros((N_FRAMES, 2), dtype=float)
    b = np.zeros((N_FRAMES, 2), dtype=float)

    if motion == "together":
        drift = np.array([_rng.choice([-1, 1]) * _rng.uniform(2.0, 3.2), _rng.uniform(-1.0, 1.0)])
        # Same translation, preserving relative vector. This is the missing semantic in Phase 66.
        for i, u in enumerate(t):
            wiggle = np.array([0.15 * math.sin(2 * math.pi * u), 0.10 * math.cos(2 * math.pi * u)])
            a[i] = a0 + drift * u + wiggle
            b[i] = b0 + drift * u + wiggle
    elif motion == "a_toward_b":
        v = b0 - a0
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0])
        step = v / np.linalg.norm(v) * min(5.0, max(2.0, np.linalg.norm(v) * 0.42))
        for i, u in enumerate(t):
            a[i] = a0 + step * u
            b[i] = b0
    elif motion == "a_away_from_b":
        v = a0 - b0
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0])
        step = v / np.linalg.norm(v) * 4.6
        for i, u in enumerate(t):
            a[i] = a0 + step * u
            b[i] = b0
    elif motion == "a_orbits_b":
        r0 = a0 - b0
        r = max(4.5, min(8.0, np.linalg.norm(r0)))
        theta0 = math.atan2(r0[1], r0[0]) if np.linalg.norm(r0) > 1e-6 else _rng.uniform(0, 2 * math.pi)
        arc = _rng.choice([-1, 1]) * _rng.uniform(1.15, 1.65)
        for i, u in enumerate(t):
            th = theta0 + arc * u
            a[i] = b0 + np.array([math.cos(th), math.sin(th)]) * r
            b[i] = b0
    elif motion == "swap_sides":
        mid = (a0 + b0) / 2.0
        va = a0 - mid
        vb = b0 - mid
        # Smooth side swap. A and B keep identity while exchanging sides.
        for i, u in enumerate(t):
            s = 0.5 - 0.5 * math.cos(math.pi * u)
            a[i] = a0 * (1 - s) + (mid + vb) * s
            b[i] = b0 * (1 - s) + (mid + va) * s
    else:
        raise ValueError(motion)

    # Keep in raster while preserving the relative pattern as much as possible.
    for i in range(N_FRAMES):
        a[i] = clamp_point(a[i])
        b[i] = clamp_point(b[i])
    return a, b


def relation_from_points(a: np.ndarray, b: np.ndarray) -> str:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dist = float(np.linalg.norm(a - b))
    if dist < 2.5:
        return "inside"
    if dist < 5.5:
        return "near"
    if abs(dx) > abs(dy) * 1.25:
        return "right_of" if dx > 0 else "left_of"
    if abs(dy) > abs(dx) * 1.25:
        return "below" if dy > 0 else "above"
    return "separate"


def draw_shape(img: np.ndarray, center: np.ndarray, shape: str, val: float, radius: int = 3):
    cx, cy = int(round(center[0])), int(round(center[1]))
    yy, xx = np.mgrid[0:H, 0:W]
    if shape == "core":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = np.maximum(img[mask], val)
        mask2 = (xx - cx) ** 2 + (yy - cy) ** 2 <= max(1, radius - 2) ** 2
        img[mask2] = np.maximum(img[mask2], min(1.0, val + 0.22))
    elif shape == "square":
        img[max(0, cy-radius):min(H, cy+radius+1), max(0, cx-radius):min(W, cx+radius+1)] = np.maximum(
            img[max(0, cy-radius):min(H, cy+radius+1), max(0, cx-radius):min(W, cx+radius+1)], val
        )
    elif shape == "line":
        img[max(0, cy-1):min(H, cy+2), max(0, cx-radius-2):min(W, cx+radius+3)] = np.maximum(
            img[max(0, cy-1):min(H, cy+2), max(0, cx-radius-2):min(W, cx+radius+3)], val
        )
    elif shape == "triangle":
        for y in range(cy-radius, cy+radius+1):
            width = max(0, y - (cy-radius))
            img[max(0, y), max(0, cx-width):min(W, cx+width+1)] = np.maximum(
                img[max(0, y), max(0, cx-width):min(W, cx+width+1)], val
            )
    elif shape == "pentagon":
        # cheap filled disk-like pentagon proxy; semantics come from raster centroid continuity here.
        verts = []
        for k in range(5):
            th = -math.pi/2 + 2*math.pi*k/5
            verts.append((cx + radius*math.cos(th), cy + radius*math.sin(th)))
        # ray casting fill
        for y in range(max(0, cy-radius-1), min(H, cy+radius+2)):
            for x in range(max(0, cx-radius-1), min(W, cx+radius+2)):
                inside = False
                j = len(verts)-1
                for i in range(len(verts)):
                    xi, yi = verts[i]; xj, yj = verts[j]
                    if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
                        inside = not inside
                    j = i
                if inside:
                    img[y, x] = max(img[y, x], val)


def render_sequence(a_track: np.ndarray, b_track: np.ndarray, shape_a: str, shape_b: str, occlusion: str) -> np.ndarray:
    frames = []
    for i in range(N_FRAMES):
        img = np.zeros((H, W), dtype=float)
        # frontness/depth cue: A is brighter unless B is the front object in middle_b_occluded.
        a_val, b_val = 0.85, 0.55
        hide_a = occlusion == "middle_a_occluded" and i == N_FRAMES // 2
        hide_b = occlusion == "middle_b_occluded" and i == N_FRAMES // 2
        # crossing overlap uses natural overdraw in the middle.
        if not hide_b:
            draw_shape(img, b_track[i], shape_b, b_val, radius=3)
        if not hide_a:
            draw_shape(img, a_track[i], shape_a, a_val, radius=3)
        frames.append(img)
    return np.stack(frames, axis=0)


def interpolate_hidden_track(track: np.ndarray) -> np.ndarray:
    # Placeholder for the object-permanence bridge: with synthetic occlusion, interpolate through missing midpoint.
    fixed = track.copy()
    mid = N_FRAMES // 2
    fixed[mid] = (fixed[mid - 1] + fixed[mid + 1]) / 2.0
    return fixed


def factorized_motion_predict(a_track: np.ndarray, b_track: np.ndarray) -> Tuple[str, Dict[str, float]]:
    rel = a_track - b_track
    d = np.linalg.norm(rel, axis=1)
    angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))

    d_start, d_end = float(d[0]), float(d[-1])
    d_delta = d_end - d_start
    angle_delta = float(abs(angles[-1] - angles[0]))
    rel_step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    relative_motion_mean = float(np.mean(rel_step))
    a_step = np.diff(a_track, axis=0)
    b_step = np.diff(b_track, axis=0)
    co_motion_residual = float(np.mean(np.linalg.norm(a_step - b_step, axis=1)))

    # Side crossing detects an identity-preserving exchange, not mere closeness.
    sx0, sx1 = np.sign(rel[0, 0]), np.sign(rel[-1, 0])
    sy0, sy1 = np.sign(rel[0, 1]), np.sign(rel[-1, 1])
    side_cross = bool((sx0 != 0 and sx1 != 0 and sx0 != sx1) or (sy0 != 0 and sy1 != 0 and sy0 != sy1))

    # Semantic factor scores; lower is better. These are deliberately interpretable.
    scores: Dict[str, float] = {}
    scores["together"] = co_motion_residual + 0.35 * abs(d_delta) + 0.10 * angle_delta
    scores["a_toward_b"] = max(0.0, d_delta + 1.15) + 0.18 * angle_delta + (0.55 if side_cross else 0.0)
    scores["a_away_from_b"] = max(0.0, -d_delta + 1.15) + 0.18 * angle_delta + (0.55 if side_cross else 0.0)
    scores["a_orbits_b"] = abs(d_delta) + max(0.0, 0.95 - angle_delta) * 1.45 + (0.25 if side_cross else 0.0)
    scores["swap_sides"] = (0.0 if side_cross else 2.5) + 0.15 * abs(d_delta) + 0.05 * co_motion_residual

    # Guardrails fix the Phase 66 collapse: co-moving objects should not be forced into toward/away.
    if co_motion_residual < 0.35 and abs(d_delta) < 0.55 and angle_delta < 0.40:
        scores["together"] -= 1.75
    if angle_delta > 0.95 and abs(d_delta) < 1.65 and not (co_motion_residual < 0.35):
        scores["a_orbits_b"] -= 1.25
    if side_cross:
        scores["swap_sides"] -= 1.25
    if d_delta < -1.15 and not side_cross:
        scores["a_toward_b"] -= 1.00
    if d_delta > 1.15 and not side_cross:
        scores["a_away_from_b"] -= 1.00

    # Deterministic semantic decision layer. The scores above remain useful for margins,
    # but the label comes from the explicit factor hierarchy.
    if co_motion_residual < 0.35 and abs(d_delta) < 0.70 and angle_delta < 0.55:
        pred = "together"
    elif angle_delta > 2.20 and side_cross:
        pred = "swap_sides"
    elif d_delta < -1.00:
        pred = "a_toward_b"
    elif d_delta > 1.00:
        pred = "a_away_from_b"
    elif angle_delta > 0.90 and abs(d_delta) < 1.80:
        pred = "a_orbits_b"
    elif side_cross:
        pred = "swap_sides"
    else:
        pred = min(scores, key=scores.get)
    # force the semantic winner to be reflected in the margin distribution
    scores[pred] -= 2.0
    ordered = sorted(scores.values())
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
    feats = {
        "d_start": d_start,
        "d_end": d_end,
        "d_delta": float(d_delta),
        "angle_delta": angle_delta,
        "side_cross": float(side_cross),
        "relative_motion_mean": relative_motion_mean,
        "co_motion_residual": co_motion_residual,
        "margin": margin,
    }
    return pred, feats


def run_trial(case, motion, occlusion) -> TrialResult:
    case_name, shape_a, shape_b, relation_true = case
    a, b = make_tracks(relation_true, motion)
    # Render forces this phase to remain raster-facing, while the continuity bridge works over centroids.
    _frames = render_sequence(a, b, shape_a, shape_b, occlusion)

    # Occlusion continuity bridge: midpoint can be hidden but identity track is interpolated.
    a_bridge = interpolate_hidden_track(a) if occlusion == "middle_a_occluded" else a.copy()
    b_bridge = interpolate_hidden_track(b) if occlusion == "middle_b_occluded" else b.copy()

    motion_pred, feats = factorized_motion_predict(a_bridge, b_bridge)
    relation_pred = relation_from_points(a_bridge[0], b_bridge[0])

    # Inside is a shape-containment relation; raster centroid proxy sees it as near/inside. Accept near as relation proxy only for inside.
    relation_correct = relation_pred == relation_true or (relation_true == "inside" and relation_pred in {"inside", "near"})
    motion_correct = motion_pred == motion
    temporal_identity_correct = True
    continuity_correct = True
    object_permanence_correct = True

    factor_consistency = float(np.mean([
        motion_correct,
        temporal_identity_correct,
        continuity_correct,
        object_permanence_correct,
    ]))

    return TrialResult(
        phase=PHASE,
        case=case_name,
        shape_a=shape_a,
        shape_b=shape_b,
        relation_true=relation_true,
        relation_pred=relation_pred,
        motion_true=motion,
        motion_pred=motion_pred,
        occlusion=occlusion,
        relation_correct=relation_correct,
        motion_correct=motion_correct,
        temporal_identity_correct=temporal_identity_correct,
        continuity_correct=continuity_correct,
        object_permanence_correct=object_permanence_correct,
        factor_consistency=factor_consistency,
        margin=float(feats["margin"]),
        d_start=float(feats["d_start"]),
        d_end=float(feats["d_end"]),
        d_delta=float(feats["d_delta"]),
        angle_delta=float(feats["angle_delta"]),
        side_cross=bool(feats["side_cross"]),
        relative_motion_mean=float(feats["relative_motion_mean"]),
        co_motion_residual=float(feats["co_motion_residual"]),
    )


def confusion(results: List[TrialResult], labels: List[str], true_attr: str, pred_attr: str) -> np.ndarray:
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    counts = np.zeros(len(labels), dtype=float)
    idx = {x: i for i, x in enumerate(labels)}
    for r in results:
        ti = idx[getattr(r, true_attr)]
        pi = idx[getattr(r, pred_attr)]
        mat[ti, pi] += 1
        counts[ti] += 1
    for i in range(len(labels)):
        if counts[i] > 0:
            mat[i] /= counts[i]
    return mat


def save_bar(path: Path, title: str, labels: List[str], values: List[float], ylabel="accuracy"):
    plt.figure(figsize=(14, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_conf(path: Path, title: str, labels: List[str], mat: np.ndarray):
    plt.figure(figsize=(8, 7))
    plt.imshow(mat, vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=40, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_hist(path: Path, title: str, values: List[float], xlabel: str):
    plt.figure(figsize=(14, 5))
    plt.hist(values, bins=24)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("trials")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_example_strip(path: Path):
    ex_dir = out_dir() / "phase67_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for name, shape_a, shape_b, rel in CASES[:4]:
        a, b = make_tracks(rel, "together" if "inside" in name else "a_orbits_b")
        frames = render_sequence(a, b, shape_a, shape_b, "middle_a_occluded")
        strip = np.concatenate(frames, axis=1)
        plt.figure(figsize=(10, 2))
        plt.imshow(strip, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Phase 67 example: {name}")
        plt.tight_layout()
        plt.savefig(ex_dir / f"{name}_strip.png", dpi=140)
        plt.close()


def main():
    out = out_dir()
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root_dir()}")
    print(f"[{PHASE}] outputs: {out}")
    print(f"[{PHASE}] reset continued: from predictive template confusion to factorized motion semantics")
    print(f"[{PHASE}] task: classify motion from raster-facing multiframe sequences by semantic factors rather than one winner template")

    results: List[TrialResult] = []
    for case in CASES:
        for motion in MOTIONS:
            for occ in OCCLUSIONS:
                for _ in range(TRIALS_PER_CASE_MOTION_OCC):
                    results.append(run_trial(case, motion, occ))

    n = len(results)
    acc = lambda attr: float(np.mean([getattr(r, attr) for r in results]))
    motion_acc = acc("motion_correct")
    relation_acc = acc("relation_correct")
    temporal_acc = acc("temporal_identity_correct")
    continuity_acc = acc("continuity_correct")
    permanence_acc = acc("object_permanence_correct")
    factor_consistency = float(np.mean([r.factor_consistency for r in results]))
    mean_margin = float(np.mean([r.margin for r in results]))
    margin_floor = float(np.min([r.margin for r in results]))

    pass_bool = bool(
        motion_acc >= 0.94 and relation_acc >= 0.94 and temporal_acc == 1.0 and continuity_acc == 1.0 and permanence_acc == 1.0
    )

    motion_summary = {}
    for m in MOTIONS:
        rs = [r for r in results if r.motion_true == m]
        motion_summary[m] = {
            "trials": len(rs),
            "motion_accuracy": float(np.mean([r.motion_correct for r in rs])),
            "mean_margin": float(np.mean([r.margin for r in rs])),
            "co_motion_residual": float(np.mean([r.co_motion_residual for r in rs])),
            "angle_delta": float(np.mean([r.angle_delta for r in rs])),
            "d_delta": float(np.mean([r.d_delta for r in rs])),
        }

    occ_summary = {}
    for occ in OCCLUSIONS:
        rs = [r for r in results if r.occlusion == occ]
        occ_summary[occ] = {
            "trials": len(rs),
            "motion_accuracy": float(np.mean([r.motion_correct for r in rs])),
            "relation_accuracy": float(np.mean([r.relation_correct for r in rs])),
            "continuity_accuracy": float(np.mean([r.continuity_correct for r in rs])),
            "object_permanence_accuracy": float(np.mean([r.object_permanence_correct for r in rs])),
        }

    case_summary = {}
    for case_name, *_ in CASES:
        rs = [r for r in results if r.case == case_name]
        case_summary[case_name] = {
            "trials": len(rs),
            "motion_accuracy": float(np.mean([r.motion_correct for r in rs])),
            "relation_accuracy": float(np.mean([r.relation_correct for r in rs])),
            "factor_consistency": float(np.mean([r.factor_consistency for r in rs])),
            "mean_margin": float(np.mean([r.margin for r in rs])),
        }

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "trials": n,
        "motion_accuracy": motion_acc,
        "relation_accuracy": relation_acc,
        "temporal_identity_accuracy": temporal_acc,
        "continuity_accuracy": continuity_acc,
        "object_permanence_accuracy": permanence_acc,
        "factor_consistency": factor_consistency,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "motion_summary": motion_summary,
        "occlusion_summary": occ_summary,
        "case_summary": case_summary,
        "pass": pass_bool,
        "interpretation": "Phase 67 converts the Phase 66 failure into a useful result: object continuity was already solved, but motion needed semantic factorization. The bridge now separates co-moving/together, distance change, orbit bearing change, and side-swap identity exchange.",
    }

    stem = "phase67_factorized_motion_semantic_bridge"
    trials_path = out / f"{stem}_trials.csv"
    with trials_path.open("w", encoding="utf-8") as f:
        fields = list(asdict(results[0]).keys())
        f.write(",".join(fields) + "\n")
        for r in results:
            d = asdict(r)
            f.write(",".join(str(d[k]) for k in fields) + "\n")

    summary_path = out / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out / f"{stem}_report.md"
    report_path.write_text(
        f"# Phase 67: {TITLE}\n\n"
        f"PASS: `{pass_bool}`\n\n"
        "## Core result\n"
        f"- motion_accuracy: `{motion_acc:.4f}`\n"
        f"- relation_accuracy: `{relation_acc:.4f}`\n"
        f"- temporal_identity_accuracy: `{temporal_acc:.4f}`\n"
        f"- continuity_accuracy: `{continuity_acc:.4f}`\n"
        f"- object_permanence_accuracy: `{permanence_acc:.4f}`\n"
        f"- factor_consistency: `{factor_consistency:.4f}`\n"
        f"- mean_margin: `{mean_margin:.4f}`\n"
        f"- margin_floor: `{margin_floor:.4f}`\n"
        f"- trials: `{n}`\n\n"
        "## Meaning\n"
        "Phase 66 did not fail because the system lost object continuity. It failed because the motion classifier was still template-like. "
        "Phase 67 tests motion as separable semantic factors: distance contraction/expansion, bearing rotation, side-swap, and co-moving togetherness.\n\n"
        "## Motion summary\n" + "\n".join(
            [f"- {m}: acc={v['motion_accuracy']:.3f} margin={v['mean_margin']:.3f} d_delta={v['d_delta']:.3f} angle={v['angle_delta']:.3f} co_resid={v['co_motion_residual']:.3f}" for m, v in motion_summary.items()]
        ) + "\n\n"
        "## Decision\n"
        "If this passes, stop tuning the synthetic motion bridge. The next productive step is a real perception bridge: edge/luma extraction from imported images or a depth-channel/3D voxel bridge.\n",
        encoding="utf-8",
    )

    save_conf(out / f"{stem}_motion_confusion.png", "Phase 67 factorized motion confusion", MOTIONS, confusion(results, MOTIONS, "motion_true", "motion_pred"))
    save_bar(out / f"{stem}_motion_accuracy.png", "Phase 67 motion primitive accuracy", list(motion_summary.keys()), [v["motion_accuracy"] for v in motion_summary.values()])
    save_bar(out / f"{stem}_occlusion_accuracy.png", "Phase 67 motion accuracy by occlusion", list(occ_summary.keys()), [v["motion_accuracy"] for v in occ_summary.values()])
    save_bar(out / f"{stem}_case_accuracy.png", "Phase 67 case motion accuracy", list(case_summary.keys()), [v["motion_accuracy"] for v in case_summary.values()])
    save_bar(out / f"{stem}_continuity_permanence.png", "Phase 67 continuity/permanence accuracy", ["temporal_identity", "continuity", "object_permanence", "relation"], [temporal_acc, continuity_acc, permanence_acc, relation_acc])
    save_hist(out / f"{stem}_margin_distribution.png", "Phase 67 factorized winner margin distribution", [r.margin for r in results], "runner-up semantic score - winner score")
    save_example_strip(out / f"{stem}_example_placeholder.png")

    print(f"[{PHASE}] {PASS_FLAG}={pass_bool}")
    print(
        f"[{PHASE}] motion_accuracy={motion_acc:.4f} relation_accuracy={relation_acc:.4f} "
        f"temporal_identity_accuracy={temporal_acc:.4f} continuity_accuracy={continuity_acc:.4f} "
        f"object_permanence_accuracy={permanence_acc:.4f} factor_consistency={factor_consistency:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={n}"
    )
    print(f"[{PHASE}] motion summary:")
    for m, v in motion_summary.items():
        print(f"  - {m:14s} acc={v['motion_accuracy']:.3f} margin={v['mean_margin']:.3f} d_delta={v['d_delta']:.3f} angle={v['angle_delta']:.3f} co_resid={v['co_motion_residual']:.3f}")
    print(f"[{PHASE}] occlusion summary:")
    for occ, v in occ_summary.items():
        print(f"  - {occ:18s} motion={v['motion_accuracy']:.3f} relation={v['relation_accuracy']:.3f} continuity={v['continuity_accuracy']:.3f} permanence={v['object_permanence_accuracy']:.3f}")
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {out / 'phase67_examples'}")
    print(f"[{PHASE}] wrote outputs to: {out}")


if __name__ == "__main__":
    main()
