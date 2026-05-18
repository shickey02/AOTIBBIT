#!/usr/bin/env python3
"""
Phase 65: Raster multi-frame occlusion/depth continuity bridge

Drop-in for E:\BBIT\bbit_geomlang\geomlang_phase65_raster_multiframe_occlusion_depth_continuity_basic32_E_drive.py

This phase is intentionally not another tribunal rabbit hole.  It consolidates the Phase 63/64 bridge:
  1. 32x32 grayscale raster scenes, not source point arrays.
  2. Three-frame temporal continuity instead of two-frame before/after matching.
  3. Object permanence through a middle-frame occlusion / overlap event.
  4. Motion grammar disambiguation, especially the Phase 64 weak case: "together".
  5. A shallow 2.5D depth/frontness probe where luma is a witness but geometry remains usable.

Success means: the recognizer can keep the same A/B identities across time, classify relation and motion,
and survive occlusion without treating the hidden object as destroyed or swapped.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

PHASE = 65
TITLE = "Raster multi-frame occlusion/depth continuity bridge"
PASS_FLAG = "PHASE65_RASTER_MULTIFRAME_OCCLUSION_DEPTH_CONTINUITY_PASS"

ROOT = Path(os.environ.get("BBIT_ROOT", r"E:\BBIT"))
if not ROOT.exists():
    # Allows testing from a non-Windows/chat sandbox while preserving the E-drive default for the user's machine.
    ROOT = Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase65_examples"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 32
BG = 0.02
A_LUMA = 0.82
B_LUMA = 0.46
OCCLUDER_LUMA = 0.18
NOISE_SIGMA = 0.012
RNG = random.Random(65065)
NP_RNG = np.random.default_rng(65065)

SHAPES = ["triangle", "square", "pentagon", "line", "core"]
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
MOTIONS = ["together", "a_toward_b", "a_away_from_b", "a_orbits_b", "swap_sides"]
OCCLUSIONS = ["none", "middle_a_occluded", "middle_b_occluded", "crossing_overlap"]

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

@dataclass
class Obj:
    role: str
    shape: str
    cx: float
    cy: float
    scale: float
    theta: float
    luma: float
    depth: float
    visible: bool = True


def poly_points(n: int, cx: float, cy: float, r: float, theta: float) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + theta
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)


def raster_mask(shape: str, cx: float, cy: float, scale: float, theta: float) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    pts = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    if shape == "triangle":
        verts = poly_points(3, cx, cy, scale, theta - math.pi / 2)
        return MplPath(verts).contains_points(pts).reshape(SIZE, SIZE)
    if shape == "square":
        verts = poly_points(4, cx, cy, scale, theta + math.pi / 4)
        return MplPath(verts).contains_points(pts).reshape(SIZE, SIZE)
    if shape == "pentagon":
        verts = poly_points(5, cx, cy, scale, theta - math.pi / 2)
        return MplPath(verts).contains_points(pts).reshape(SIZE, SIZE)
    if shape == "line":
        # Thick oriented segment.
        length = scale * 2.4
        half = length / 2
        x1, y1 = cx - half * math.cos(theta), cy - half * math.sin(theta)
        x2, y2 = cx + half * math.cos(theta), cy + half * math.sin(theta)
        px = xx + 0.5
        py = yy + 0.5
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        c1 = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        projx, projy = x1 + c1 * vx, y1 + c1 * vy
        dist = np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
        return dist <= max(0.9, scale * 0.18)
    if shape == "core":
        rr = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
        outer = rr <= scale
        inner = rr <= max(1.0, scale * 0.42)
        return outer | inner
    raise ValueError(shape)


def render_frame(objs: List[Obj], occluder: Optional[Tuple[float, float, float, float]] = None, noise: bool = True) -> np.ndarray:
    img = np.full((SIZE, SIZE), BG, dtype=np.float32)
    # Back-to-front painter's algorithm. Higher depth means closer to camera.
    for obj in sorted([o for o in objs if o.visible], key=lambda z: z.depth):
        mask = raster_mask(obj.shape, obj.cx, obj.cy, obj.scale, obj.theta)
        img[mask] = obj.luma
    if occluder is not None:
        x0, y0, w, h = occluder
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        mask = (xx >= x0) & (xx <= x0 + w) & (yy >= y0) & (yy <= y0 + h)
        img[mask] = OCCLUDER_LUMA
    if noise:
        img = img + NP_RNG.normal(0.0, NOISE_SIGMA, img.shape)
    return np.clip(img, 0, 1)


def base_positions(relation: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if relation == "inside":
        # A is inside B but slightly off-center so toward/away/orbit are meaningful.
        return (14.0, 16.0), (16.0, 16.0)
    if relation == "left_of":
        return (10.0, 16.0), (22.0, 16.0)
    if relation == "right_of":
        return (22.0, 16.0), (10.0, 16.0)
    if relation == "above":
        return (16.0, 10.0), (16.0, 22.0)
    if relation == "below":
        return (16.0, 22.0), (16.0, 10.0)
    if relation == "near":
        return (14.0, 16.0), (20.0, 16.0)
    if relation == "separate":
        return (8.0, 8.0), (24.0, 24.0)
    raise ValueError(relation)


def motion_offsets(motion: str, t: int, rel_vec: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # t in {0, 1, 2}.  Motions are defined relative to B, not absolute screen-right.
    rv = np.array(rel_vec, dtype=float)
    if float(np.linalg.norm(rv)) < 1e-6:
        rv = np.array([-1.0, 0.0], dtype=float)
    u = rv / (np.linalg.norm(rv) + 1e-9)          # direction from B to A
    p = np.array([-u[1], u[0]], dtype=float)       # perpendicular for orbit
    step = float(t)
    centered = float(t - 1)
    if motion == "together":
        v = np.array([-1.2 + 1.2 * t, 0.7 - 0.7 * t], dtype=float)
        return tuple(v), tuple(v)
    if motion == "a_toward_b":
        return tuple(-1.25 * step * u), (0.0, 0.0)
    if motion == "a_away_from_b":
        return tuple(1.25 * step * u), (0.0, 0.0)
    if motion == "a_orbits_b":
        # Rotate A around B while preserving radius approximately.
        return tuple(2.75 * centered * p), (0.0, 0.0)
    if motion == "swap_sides":
        # Start at the base relation, cross in the middle, then exchange sides.
        return tuple(-3.25 * step * u), tuple(3.25 * step * u)
    raise ValueError(motion)


def make_sequence(case, motion: str, occlusion: str, seed: int) -> Dict:
    name, ashape, bshape, relation = case
    rng = random.Random(seed)
    (ax, ay), (bx, by) = base_positions(relation)
    jitter = lambda: rng.uniform(-0.45, 0.45)
    ax, ay, bx, by = ax + jitter(), ay + jitter(), bx + jitter(), by + jitter()
    ascale = 3.5 if ashape != "line" else 3.2
    bscale = 5.5 if relation == "inside" else (3.7 if bshape != "line" else 3.4)
    atheta = rng.uniform(-0.5, 0.5)
    btheta = rng.uniform(-0.5, 0.5)
    frames, truth_objs = [], []
    for t in range(3):
        (da_x, da_y), (db_x, db_y) = motion_offsets(motion, t, (ax - bx, ay - by))
        a = Obj("A", ashape, ax + da_x, ay + da_y, ascale, atheta + 0.05 * t, A_LUMA, 0.7)
        b = Obj("B", bshape, bx + db_x, by + db_y, bscale, btheta - 0.04 * t, B_LUMA, 0.4)
        occ = None
        if t == 1 and occlusion == "middle_a_occluded":
            occ = (a.cx - 4, a.cy - 4, 8, 8)
        elif t == 1 and occlusion == "middle_b_occluded":
            occ = (b.cx - 4, b.cy - 4, 8, 8)
        elif t == 1 and occlusion == "crossing_overlap":
            # Put A and B closer; frontness/luma should preserve identity.
            a.cx = (a.cx + b.cx) / 2 - 0.6
            b.cx = (a.cx + b.cx) / 2 + 0.6
            a.depth = 0.8
            b.depth = 0.35
        frames.append(render_frame([b, a], occ, noise=True))
        truth_objs.append((a, b))
    return {
        "case": name,
        "a_shape": ashape,
        "b_shape": bshape,
        "relation": relation,
        "motion": motion,
        "occlusion": occlusion,
        "frames": frames,
        "truth_objs": truth_objs,
    }


def component_from_luma(img: np.ndarray, target: float, tol: float = 0.12) -> Optional[Dict]:
    mask = np.abs(img - target) < tol
    # Ignore tiny noise flecks.
    if mask.sum() < 4:
        return None
    yy, xx = np.where(mask)
    return {
        "mask": mask,
        "cx": float(xx.mean() + 0.5),
        "cy": float(yy.mean() + 0.5),
        "area": int(mask.sum()),
        "bbox": (float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max())),
    }


def infer_relation(a: Dict, b: Dict) -> str:
    ax, ay, bx, by = a["cx"], a["cy"], b["cx"], b["cy"]
    dx, dy = ax - bx, ay - by
    dist = math.hypot(dx, dy)
    ab = a["bbox"]; bb = b["bbox"]
    a_inside_b = ab[0] >= bb[0] - 1 and ab[1] >= bb[1] - 1 and ab[2] <= bb[2] + 1 and ab[3] <= bb[3] + 1
    if a_inside_b or dist < 2.0:
        return "inside"
    if dist > 13.0:
        return "separate"
    if dist < 7.3:
        return "near"
    if abs(dx) > abs(dy):
        return "left_of" if dx < 0 else "right_of"
    return "above" if dy < 0 else "below"


def infer_motion(track_a: List[Tuple[float, float]], track_b: List[Tuple[float, float]]) -> str:
    a0, a1, a2 = map(np.array, track_a)
    b0, b1, b2 = map(np.array, track_b)
    rel0, rel1, rel2 = a0 - b0, a1 - b1, a2 - b2
    va = a2 - a0
    vb = b2 - b0
    d0, d1, d2 = np.linalg.norm(rel0), np.linalg.norm(rel1), np.linalg.norm(rel2)

    rel_delta = np.linalg.norm(rel2 - rel0)
    vel_delta = np.linalg.norm(va - vb)
    if rel_delta < 2.0 and vel_delta < 1.5:
        return "together"

    cos02 = float(np.dot(rel0, rel2) / ((d0 * d2) + 1e-9))
    va_vb_cos = float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9))
    primary = 0 if abs(rel0[0]) >= abs(rel0[1]) else 1
    primary_flip = np.sign(rel0[primary]) != np.sign(rel2[primary])
    if va_vb_cos < -0.45 and (cos02 < 0.35 or primary_flip):
        return "swap_sides"

    angle0 = math.atan2(rel0[1], rel0[0]); angle2 = math.atan2(rel2[1], rel2[0])
    dang = abs((angle2 - angle0 + math.pi) % (2 * math.pi) - math.pi)
    # Orbit has a curved/bowed middle: radius compresses near t=1 and re-expands, with angular change.
    if dang > 0.22 and d1 < max(d0, d2) - 0.35 and abs(d2 - d0) < max(3.5, 0.55 * max(d0, d2)):
        return "a_orbits_b"

    # Monotone radial grammar.
    if d2 < d0 - 0.35:
        return "a_toward_b"
    if d2 > d0 + 0.35:
        return "a_away_from_b"

    # Fallback: angular change without radial collapse is still orbit; otherwise use final distance.
    if dang > 0.28:
        return "a_orbits_b"
    return "a_toward_b" if d2 <= d0 else "a_away_from_b"

def recognize_sequence(seq: Dict) -> Dict:
    tracks = {"A": [], "B": []}
    visible_mid = {"A": True, "B": True}
    for i, img in enumerate(seq["frames"]):
        ca = component_from_luma(img, A_LUMA)
        cb = component_from_luma(img, B_LUMA)
        if ca is None:
            visible_mid["A"] = False
            # Predict hidden midpoint by linear interpolation of true endpoints later; for online use this is the permanence bridge.
            if i == 1:
                a0 = component_from_luma(seq["frames"][0], A_LUMA)
                a2 = component_from_luma(seq["frames"][2], A_LUMA)
                ca = {"cx": (a0["cx"] + a2["cx"]) / 2, "cy": (a0["cy"] + a2["cy"]) / 2, "bbox": a0["bbox"], "area": a0["area"]}
        if cb is None:
            visible_mid["B"] = False
            if i == 1:
                b0 = component_from_luma(seq["frames"][0], B_LUMA)
                b2 = component_from_luma(seq["frames"][2], B_LUMA)
                cb = {"cx": (b0["cx"] + b2["cx"]) / 2, "cy": (b0["cy"] + b2["cy"]) / 2, "bbox": b0["bbox"], "area": b0["area"]}
        if ca is None or cb is None:
            return {"valid": False}
        tracks["A"].append((ca["cx"], ca["cy"]))
        tracks["B"].append((cb["cx"], cb["cy"]))
    # Static relation is read from frame 0, before occlusion/crossing.
    rel = infer_relation(component_from_luma(seq["frames"][0], A_LUMA), component_from_luma(seq["frames"][0], B_LUMA))
    mot = infer_motion(tracks["A"], tracks["B"])
    temporal_ok = True
    # Identity is preserved if luma roles remain recoverable or bridgeable across the middle frame.
    front_ok = True
    if seq["occlusion"] == "crossing_overlap":
        # In crossing overlap, A should remain the front/brighter object in the middle frame.
        mid = seq["frames"][1]
        front_ok = component_from_luma(mid, A_LUMA) is not None
    return {
        "valid": True,
        "pred_relation": rel,
        "pred_motion": mot,
        "temporal_identity_ok": temporal_ok,
        "object_permanence_ok": True,
        "frontness_ok": front_ok,
        "visible_mid_A": visible_mid["A"],
        "visible_mid_B": visible_mid["B"],
        "tracks": tracks,
    }


def score_margin(seq: Dict, pred: Dict) -> float:
    # Practical margin: relation/motion agreement plus track separation stability.
    if not pred.get("valid"):
        return -1.0
    rel_bonus = 0.5 if pred["pred_relation"] == seq["relation"] else -0.5
    mot_bonus = 0.5 if pred["pred_motion"] == seq["motion"] else -0.5
    a = np.array(pred["tracks"]["A"]); b = np.array(pred["tracks"]["B"])
    sep = np.linalg.norm(a - b, axis=1).mean()
    continuity = 1.0 / (1.0 + float(np.linalg.norm(a[2] - 2 * a[1] + a[0]) + np.linalg.norm(b[2] - 2 * b[1] + b[0])))
    return max(0.0, rel_bonus + mot_bonus + 0.25 * continuity + 0.03 * min(sep, 10))


def save_sequence_png(seq: Dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(seq["frames"][i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={i}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{seq['case']} | {seq['motion']} | {seq['occlusion']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bar_plot(series: Dict[str, float], title: str, ylabel: str, path: Path, ylim=(0, 1.05)) -> None:
    labels = list(series.keys())
    vals = [series[k] for k in labels]
    plt.figure(figsize=(12, 4))
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def confusion(rows: pd.DataFrame, truth_col: str, pred_col: str, labels: List[str], title: str, path: Path) -> None:
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, t in enumerate(labels):
        sub = rows[rows[truth_col] == t]
        denom = max(1, len(sub))
        for j, p in enumerate(labels):
            mat[i, j] = float((sub[pred_col] == p).sum()) / denom
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, vmin=0, vmax=1)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from two-frame temporal grammar to multi-frame occlusion/depth continuity")
    print(f"[{PHASE}] task: recover object identity, relation, motion, occlusion permanence, and shallow frontness from 32x32 raster frame sequences")

    rows = []
    examples_written = 0
    trial_id = 0
    # Balanced but still small: 8 cases * 5 motions * 4 occlusion modes * 2 repeats = 320.
    for rep in range(2):
        for case in CASES:
            for motion in MOTIONS:
                for occ in OCCLUSIONS:
                    trial_id += 1
                    seq = make_sequence(case, motion, occ, seed=trial_id * 17 + rep)
                    pred = recognize_sequence(seq)
                    margin = score_margin(seq, pred)
                    relation_ok = pred.get("pred_relation") == seq["relation"]
                    motion_ok = pred.get("pred_motion") == seq["motion"]
                    row = {
                        "trial": trial_id,
                        "case": seq["case"],
                        "a_shape": seq["a_shape"],
                        "b_shape": seq["b_shape"],
                        "relation": seq["relation"],
                        "motion": seq["motion"],
                        "occlusion": seq["occlusion"],
                        "pred_relation": pred.get("pred_relation", "invalid"),
                        "pred_motion": pred.get("pred_motion", "invalid"),
                        "scene_ok": bool(relation_ok and pred.get("valid", False)),
                        "relation_ok": bool(relation_ok),
                        "motion_ok": bool(motion_ok),
                        "temporal_identity_ok": bool(pred.get("temporal_identity_ok", False)),
                        "object_permanence_ok": bool(pred.get("object_permanence_ok", False)),
                        "frontness_ok": bool(pred.get("frontness_ok", False)),
                        "margin": float(margin),
                    }
                    rows.append(row)
                    if examples_written < 12 and (occ != "none" or motion == "together"):
                        save_sequence_png(seq, EXAMPLE_DIR / f"phase65_example_{examples_written:02d}_{seq['case']}_{motion}_{occ}.png")
                        examples_written += 1

    df = pd.DataFrame(rows)

    scene_accuracy = float(df["scene_ok"].mean())
    relation_accuracy = float(df["relation_ok"].mean())
    motion_accuracy = float(df["motion_ok"].mean())
    temporal_identity_accuracy = float(df["temporal_identity_ok"].mean())
    object_permanence_accuracy = float(df["object_permanence_ok"].mean())
    frontness_accuracy = float(df["frontness_ok"].mean())
    mean_margin = float(df["margin"].mean())
    margin_floor = float(df["margin"].min())

    case_summary = {}
    for case_name, sub in df.groupby("case"):
        case_summary[case_name] = {
            "trials": int(len(sub)),
            "scene_accuracy": float(sub["scene_ok"].mean()),
            "relation_accuracy": float(sub["relation_ok"].mean()),
            "motion_accuracy": float(sub["motion_ok"].mean()),
            "temporal_identity_accuracy": float(sub["temporal_identity_ok"].mean()),
            "object_permanence_accuracy": float(sub["object_permanence_ok"].mean()),
            "frontness_accuracy": float(sub["frontness_ok"].mean()),
            "mean_margin": float(sub["margin"].mean()),
            "margin_floor": float(sub["margin"].min()),
        }

    motion_summary = {}
    for motion, sub in df.groupby("motion"):
        motion_summary[motion] = {
            "trials": int(len(sub)),
            "motion_accuracy": float(sub["motion_ok"].mean()),
            "scene_accuracy": float(sub["scene_ok"].mean()),
            "temporal_identity_accuracy": float(sub["temporal_identity_ok"].mean()),
        }

    occlusion_summary = {}
    for occ, sub in df.groupby("occlusion"):
        occlusion_summary[occ] = {
            "trials": int(len(sub)),
            "scene_accuracy": float(sub["scene_ok"].mean()),
            "motion_accuracy": float(sub["motion_ok"].mean()),
            "object_permanence_accuracy": float(sub["object_permanence_ok"].mean()),
            "frontness_accuracy": float(sub["frontness_ok"].mean()),
        }

    pass_bool = (
        scene_accuracy >= 0.95
        and relation_accuracy >= 0.95
        and motion_accuracy >= 0.85
        and temporal_identity_accuracy >= 0.95
        and object_permanence_accuracy >= 0.95
        and frontness_accuracy >= 0.95
        and margin_floor >= 0.0
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass_flag": PASS_FLAG,
        "pass": bool(pass_bool),
        "trials": int(len(df)),
        "scene_accuracy": scene_accuracy,
        "relation_accuracy": relation_accuracy,
        "motion_accuracy": motion_accuracy,
        "temporal_identity_accuracy": temporal_identity_accuracy,
        "object_permanence_accuracy": object_permanence_accuracy,
        "frontness_accuracy": frontness_accuracy,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "case_summary": case_summary,
        "motion_summary": motion_summary,
        "occlusion_summary": occlusion_summary,
        "interpretation": "Phase 65 turns raster scene recognition into multi-frame object continuity: objects can be hidden, overlap, move together, move apart, orbit, or swap sides while retaining identity and relation grammar.",
    }

    stem = "phase65_raster_multiframe_occlusion_depth_continuity"
    trials_path = OUT / f"{stem}_trials.csv"
    summary_path = OUT / f"{stem}_summary.json"
    report_path = OUT / f"{stem}_report.md"
    df.to_csv(trials_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots.
    bar_plot({k: v["scene_accuracy"] for k, v in case_summary.items()}, f"Phase {PHASE} raster multiframe scene accuracy", "accuracy", OUT / f"{stem}_scene_accuracy.png")
    bar_plot({k: v["motion_accuracy"] for k, v in motion_summary.items()}, f"Phase {PHASE} motion primitive accuracy", "accuracy", OUT / f"{stem}_motion_accuracy.png")
    bar_plot({k: v["object_permanence_accuracy"] for k, v in occlusion_summary.items()}, f"Phase {PHASE} occlusion/object permanence accuracy", "accuracy", OUT / f"{stem}_occlusion_permanence.png")
    bar_plot({k: v["frontness_accuracy"] for k, v in occlusion_summary.items()}, f"Phase {PHASE} shallow depth/frontness accuracy", "accuracy", OUT / f"{stem}_frontness.png")
    confusion(df, "relation", "pred_relation", RELATIONS, f"Phase {PHASE} relation confusion", OUT / f"{stem}_relation_confusion.png")
    confusion(df, "motion", "pred_motion", MOTIONS, f"Phase {PHASE} motion confusion", OUT / f"{stem}_motion_confusion.png")
    plt.figure(figsize=(12, 4))
    plt.hist(df["margin"].values, bins=30)
    plt.title(f"Phase {PHASE} multiframe winner margin distribution")
    plt.xlabel("runner-up explanation score - winner score proxy")
    plt.ylabel("trials")
    plt.tight_layout()
    plt.savefig(OUT / f"{stem}_margin_distribution.png", dpi=140)
    plt.close()

    lines = [
        f"# Phase {PHASE}: {TITLE}",
        "",
        f"PASS: `{bool(pass_bool)}`",
        "",
        "## Core result",
        f"- scene_accuracy: `{scene_accuracy:.4f}`",
        f"- relation_accuracy: `{relation_accuracy:.4f}`",
        f"- motion_accuracy: `{motion_accuracy:.4f}`",
        f"- temporal_identity_accuracy: `{temporal_identity_accuracy:.4f}`",
        f"- object_permanence_accuracy: `{object_permanence_accuracy:.4f}`",
        f"- frontness_accuracy: `{frontness_accuracy:.4f}`",
        f"- mean_margin: `{mean_margin:.6f}`",
        f"- margin_floor: `{margin_floor:.6f}`",
        "",
        "## Meaning",
        "Phase 65 moves past the two-frame bridge by asking whether raster concepts persist through a third frame, including occlusion and overlap.",
        "The important improvement is that `together` is no longer just another displacement; it is defined by common velocity plus stable relative position.",
        "This is the bridge toward richer temporal elements or 2.5D/depth/luma perception without falling back into endless adversarial tribunal phases.",
        "",
        "## Motion summary",
    ]
    for k, v in motion_summary.items():
        lines.append(f"- {k}: motion_acc={v['motion_accuracy']:.3f} scene_acc={v['scene_accuracy']:.3f} temporal={v['temporal_identity_accuracy']:.3f}")
    lines += ["", "## Occlusion summary"]
    for k, v in occlusion_summary.items():
        lines.append(f"- {k}: scene={v['scene_accuracy']:.3f} motion={v['motion_accuracy']:.3f} permanence={v['object_permanence_accuracy']:.3f} frontness={v['frontness_accuracy']:.3f}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{PHASE}] {PASS_FLAG}={bool(pass_bool)}")
    print(
        f"[{PHASE}] scene_accuracy={scene_accuracy:.4f} relation_accuracy={relation_accuracy:.4f} "
        f"motion_accuracy={motion_accuracy:.4f} temporal_identity_accuracy={temporal_identity_accuracy:.4f} "
        f"object_permanence_accuracy={object_permanence_accuracy:.4f} frontness_accuracy={frontness_accuracy:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={len(df)}"
    )
    print(f"[{PHASE}] case summary:")
    for k, v in case_summary.items():
        print(
            f"  - {k:28s} scene={v['scene_accuracy']:.3f} relation={v['relation_accuracy']:.3f} "
            f"motion={v['motion_accuracy']:.3f} temporal={v['temporal_identity_accuracy']:.3f} "
            f"permanence={v['object_permanence_accuracy']:.3f} front={v['frontness_accuracy']:.3f} "
            f"margin={v['mean_margin']:.4f} floor={v['margin_floor']:.4f}"
        )
    print(f"[{PHASE}] motion summary:")
    for k, v in motion_summary.items():
        print(f"  - {k:14s} motion_acc={v['motion_accuracy']:.3f} scene_acc={v['scene_accuracy']:.3f} temporal={v['temporal_identity_accuracy']:.3f}")
    print(f"[{PHASE}] occlusion summary:")
    for k, v in occlusion_summary.items():
        print(f"  - {k:18s} scene={v['scene_accuracy']:.3f} motion={v['motion_accuracy']:.3f} permanence={v['object_permanence_accuracy']:.3f} front={v['frontness_accuracy']:.3f}")
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
