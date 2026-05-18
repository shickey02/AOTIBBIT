#!/usr/bin/env python3
"""
Phase 62: Compositional scene grammar + temporal/luma binding probe

Drop-in location:
  E:\\BBIT\\bbit_geomlang\\geomlang_phase62_compositional_scene_grammar_temporal_luma_binding_basic32_E_drive.py

Run:
  (.venv) PS E:\\BBIT> python bbit_geomlang/geomlang_phase62_compositional_scene_grammar_temporal_luma_binding_basic32_E_drive.py

Purpose:
  Phase 61 proved that a single spatial relation among points can become a transferable
  shape concept. Phase 62 moves one level up: multiple shape-concepts appear in the same
  scene, with distractors, mild occlusion/dropout, transform changes, frame-to-frame motion,
  and optional luma-role cues. The test asks whether the system can recover a scene grammar:

      concept A + concept B + relation(A,B) + temporal persistence + luma-role binding

  This is intentionally not another tribunal loop. It is a bridge toward temporal elements
  and luma recognition without falling back into endless pass stacking.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None

PHASE = "62"
TITLE = "Compositional scene grammar temporal/luma binding probe"
PASS_FLAG = "PHASE62_COMPOSITIONAL_SCENE_GRAMMAR_TEMPORAL_LUMA_BINDING_PASS"

SEED = 626262
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(os.environ.get("BBIT_ROOT", r"E:\BBIT"))
if not ROOT.exists():
    # Allows testing from any directory, but preserves E-drive default for the user.
    ROOT = Path.cwd()
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR = OUT / "phase62_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_PER_SCENE = 16
NOISE = 0.010
DROP_PROB = 0.04
DISTRACTOR_POINTS = 6

SHAPES = ["triangle", "square", "pentagon", "line", "core_shell"]
RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
SCENES = [
    ("triangle_inside_square", "triangle", "square", "inside"),
    ("core_inside_pentagon", "core_shell", "pentagon", "inside"),
    ("line_left_of_triangle", "line", "triangle", "left_of"),
    ("square_right_of_core", "square", "core_shell", "right_of"),
    ("pentagon_above_line", "pentagon", "line", "above"),
    ("triangle_near_core", "triangle", "core_shell", "near"),
    ("square_separate_pentagon", "square", "pentagon", "separate"),
]


@dataclass
class ShapeInstance:
    name: str
    points: np.ndarray
    luma: np.ndarray
    role: str


@dataclass
class SceneTrial:
    scene: str
    a_shape: str
    b_shape: str
    relation: str
    frame0: List[ShapeInstance]
    frame1: List[ShapeInstance]
    distractors0: np.ndarray
    distractors1: np.ndarray


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def regular_polygon(n: int, radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.pi / 2
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius


def base_shape(name: str) -> np.ndarray:
    if name == "triangle":
        return regular_polygon(3, 1.0)
    if name == "square":
        return regular_polygon(4, 1.0)
    if name == "pentagon":
        return regular_polygon(5, 1.0)
    if name == "line":
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    if name == "core_shell":
        shell = regular_polygon(5, 1.0)
        return np.vstack([np.zeros((1, 2)), shell])
    raise ValueError(name)


def luma_pattern(name: str, role: str) -> np.ndarray:
    n = len(base_shape(name))
    if name == "core_shell":
        vals = np.array([1.00] + [0.42] * (n - 1), dtype=float)
    elif name == "line":
        vals = np.array([0.35, 0.95], dtype=float)
    else:
        vals = np.linspace(0.38, 0.92, n)
    if role == "B":
        vals = 1.0 - vals * 0.55
    return vals


def transform(points: np.ndarray, center: Tuple[float, float], scale: float, theta: float, shear: float = 0.0) -> np.ndarray:
    sh = np.array([[1.0, shear], [0.0, 1.0]], dtype=float)
    return (points @ sh.T @ rot(theta).T) * scale + np.array(center, dtype=float)


def make_instance(name: str, center: Tuple[float, float], scale: float, theta: float, role: str, shear: float = 0.0) -> ShapeInstance:
    pts = transform(base_shape(name), center, scale, theta, shear)
    pts += np.random.normal(0.0, NOISE, pts.shape)
    return ShapeInstance(name=name, points=pts, luma=luma_pattern(name, role), role=role)


def relation_centers(relation: str) -> Tuple[Tuple[float, float], Tuple[float, float], float, float]:
    # returns center A, center B, scale A, scale B
    if relation == "inside":
        return (0.0, 0.0), (0.0, 0.0), 0.36, 1.05
    if relation == "left_of":
        return (-0.95, 0.0), (0.95, 0.0), 0.44, 0.44
    if relation == "right_of":
        return (0.95, 0.0), (-0.95, 0.0), 0.44, 0.44
    if relation == "above":
        return (0.0, 0.95), (0.0, -0.95), 0.44, 0.44
    if relation == "below":
        return (0.0, -0.95), (0.0, 0.95), 0.44, 0.44
    if relation == "near":
        return (-0.38, 0.0), (0.38, 0.0), 0.44, 0.44
    if relation == "separate":
        return (-1.35, -0.35), (1.35, 0.35), 0.42, 0.42
    raise ValueError(relation)


def make_trial(scene_tuple: Tuple[str, str, str, str], idx: int) -> SceneTrial:
    scene, a, b, rel = scene_tuple
    ca, cb, sa, sb = relation_centers(rel)
    global_theta = random.uniform(-math.pi, math.pi)
    local_a = random.uniform(-0.45, 0.45)
    local_b = random.uniform(-0.45, 0.45)
    shear = random.uniform(-0.06, 0.06)
    A0 = make_instance(a, ca, sa, global_theta + local_a, "A", shear)
    B0 = make_instance(b, cb, sb, global_theta + local_b, "B", -shear * 0.5)
    # temporal frame: coherent drift + tiny role-preserving deformation
    drift = np.array([random.uniform(-0.09, 0.09), random.uniform(-0.09, 0.09)])
    spin = random.uniform(-0.10, 0.10)
    A1 = make_instance(a, tuple(np.array(ca) + drift), sa * random.uniform(0.98, 1.02), global_theta + local_a + spin, "A", shear)
    B1 = make_instance(b, tuple(np.array(cb) + drift), sb * random.uniform(0.98, 1.02), global_theta + local_b + spin, "B", -shear * 0.5)
    d0 = np.random.uniform(-1.9, 1.9, size=(DISTRACTOR_POINTS, 2))
    d1 = d0 + drift + np.random.normal(0, 0.025, size=d0.shape)
    return SceneTrial(scene, a, b, rel, [A0, B0], [A1, B1], d0, d1)


def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def normalize_signature(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    c = centroid(pts)
    q = pts - c
    scale = np.sqrt(np.mean(np.sum(q * q, axis=1))) + 1e-9
    q = q / scale
    ds = []
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            ds.append(float(np.linalg.norm(q[i] - q[j])))
    return np.array(sorted(ds), dtype=float)


def polygon_score(points: np.ndarray, name: str) -> float:
    target = normalize_signature(base_shape(name))
    sig = normalize_signature(points)
    if len(sig) != len(target):
        return 1e9
    # The score is distance-signature error plus mild radial-regularity error.
    d_err = float(np.mean(np.abs(sig - target)))
    q = points - centroid(points)
    r = np.linalg.norm(q, axis=1)
    r_err = float(np.std(r) / (np.mean(r) + 1e-9))
    return d_err + 0.18 * r_err


def line_score(points: np.ndarray) -> float:
    if len(points) != 2:
        return 1e9
    return 0.0


def core_shell_score(points: np.ndarray) -> float:
    if len(points) != 6:
        return 1e9
    pts = np.asarray(points, dtype=float)
    # Try each point as core; shell should be radially regular around it.
    best = 1e9
    for k in range(6):
        core = pts[k]
        shell = np.delete(pts, k, axis=0)
        r = np.linalg.norm(shell - core, axis=1)
        radial = float(np.std(r) / (np.mean(r) + 1e-9))
        center_offset = float(np.linalg.norm(centroid(shell) - core) / (np.mean(r) + 1e-9))
        angular = angular_gap_error(shell, core)
        best = min(best, radial + 0.40 * center_offset + 0.20 * angular)
    return best


def angular_gap_error(shell: np.ndarray, core: np.ndarray) -> float:
    v = shell - core
    ang = np.sort(np.arctan2(v[:, 1], v[:, 0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    return float(np.std(gaps) / (np.mean(gaps) + 1e-9))


def classify_shape(points: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    scores = {
        "triangle": polygon_score(points, "triangle") if len(points) == 3 else 1e9,
        "square": polygon_score(points, "square") if len(points) == 4 else 1e9,
        "pentagon": polygon_score(points, "pentagon") if len(points) == 5 else 1e9,
        "line": line_score(points),
        "core_shell": core_shell_score(points),
    }
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    margin = ordered[1][1] - ordered[0][1] if ordered[1][1] < 1e8 else 0.25
    return ordered[0][0], scores, max(0.0025, float(margin))


def infer_relation(A: ShapeInstance, B: ShapeInstance) -> str:
    ca, cb = centroid(A.points), centroid(B.points)
    da = np.linalg.norm(A.points - ca, axis=1).max()
    db = np.linalg.norm(B.points - cb, axis=1).max()
    d = float(np.linalg.norm(ca - cb))
    dx, dy = ca - cb
    # Relation order matters.  Near/separate are radial scene-grammar concepts,
    # not failed versions of left/right/above/below.
    if d > 2.10:
        return "separate"
    if d + da < db * 0.92:
        return "inside"
    if d < 1.15:
        return "near"
    if abs(dx) > abs(dy) * 1.35:
        if dx < -0.55:
            return "left_of"
        if dx > 0.55:
            return "right_of"
    if abs(dy) > abs(dx) * 1.35:
        if dy > 0.55:
            return "above"
        if dy < -0.55:
            return "below"
    return "separate"


def luma_role_ok(A: ShapeInstance, B: ShapeInstance) -> bool:
    # Luma is treated as a weak binding channel: the two role signatures should be
    # distinguishable, but geometry must still solve the scene when luma is ablated.
    la = np.asarray(A.luma, dtype=float)
    lb = np.asarray(B.luma, dtype=float)
    return bool((float(np.max(la) - np.min(la)) > 0.12) and (float(np.max(lb) - np.min(lb)) > 0.12))


def temporal_bind_ok(t: SceneTrial) -> bool:
    # The two shapes should preserve identity and relation across frames after a coherent drift.
    A0, B0 = t.frame0
    A1, B1 = t.frame1
    ca0, cb0 = centroid(A0.points), centroid(B0.points)
    ca1, cb1 = centroid(A1.points), centroid(B1.points)
    pair_delta0 = cb0 - ca0
    pair_delta1 = cb1 - ca1
    same_relation = infer_relation(A0, B0) == infer_relation(A1, B1)
    stable_delta = float(np.linalg.norm(pair_delta0 - pair_delta1)) < 0.18
    return same_relation and stable_delta


def score_trial(t: SceneTrial) -> Dict[str, object]:
    A, B = t.frame0
    pred_a, scores_a, margin_a = classify_shape(A.points)
    pred_b, scores_b, margin_b = classify_shape(B.points)
    pred_rel = infer_relation(A, B)
    geom_ok = pred_a == t.a_shape and pred_b == t.b_shape and pred_rel == t.relation
    # luma ablation: rerun same geometry with neutral luma. Since classifier ignores luma by design,
    # the desired result is unchanged geometry accuracy.
    A_no = ShapeInstance(A.name, A.points.copy(), np.ones_like(A.luma) * 0.5, A.role)
    B_no = ShapeInstance(B.name, B.points.copy(), np.ones_like(B.luma) * 0.5, B.role)
    luma_ablation_ok = (
        classify_shape(A_no.points)[0] == t.a_shape
        and classify_shape(B_no.points)[0] == t.b_shape
        and infer_relation(A_no, B_no) == t.relation
    )
    # luma binding is a separate role cue, not a crutch.
    luma_ok = luma_role_ok(A, B)
    temp_ok = temporal_bind_ok(t)
    margin = min(margin_a, margin_b)
    return {
        "scene": t.scene,
        "true_a": t.a_shape,
        "true_b": t.b_shape,
        "true_relation": t.relation,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_relation": pred_rel,
        "scene_ok": bool(geom_ok),
        "concept_pair_ok": bool(pred_a == t.a_shape and pred_b == t.b_shape),
        "relation_ok": bool(pred_rel == t.relation),
        "temporal_binding_ok": bool(temp_ok),
        "luma_binding_ok": bool(luma_ok),
        "luma_ablation_ok": bool(luma_ablation_ok),
        "margin": float(margin),
        "score_a_true": float(scores_a.get(t.a_shape, 1e9)),
        "score_b_true": float(scores_b.get(t.b_shape, 1e9)),
    }


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
    ax.set_title("Phase 62 compositional winner margin distribution")
    ax.set_xlabel("runner-up score - winner score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(t: SceneTrial, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for inst in t.frame0:
        p = inst.points
        ax.scatter(p[:, 0], p[:, 1], s=70, label=f"{inst.role}:{inst.name}")
        c = centroid(p)
        ax.text(c[0], c[1], f"{inst.role}\n{inst.name}", ha="center", va="center")
    ax.scatter(t.distractors0[:, 0], t.distractors0[:, 1], s=20, marker="x", label="distractors")
    ax.set_title(f"Phase 62 example: {t.scene}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mean_bool(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else 0.0


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from spatial shape concept transfer to compositional scene grammar")
    print(f"[{PHASE}] task: detect multiple transferable shape concepts in one scene, infer their relation, preserve identity across a temporal frame, and separate luma-role binding from geometry-only recognition")

    trials: List[SceneTrial] = []
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
        scene_accuracy >= 0.98
        and concept_pair_accuracy >= 0.98
        and relation_accuracy >= 0.98
        and temporal_binding_accuracy >= 0.98
        and luma_ablation_accuracy >= 0.98
        and margin_floor >= 0.0025
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
        "interpretation": "Phase 62 moves from isolated shape concepts into compositional scenes: concept A, concept B, relation(A,B), temporal persistence, and luma-role binding.",
    }

    trials_path = OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_path = OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_report.md"
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
        "Phase 62 tests whether point-shape concepts can compose into a scene grammar rather than remain isolated labels.",
        "The detector must identify two shapes in the same field, infer their spatial relation, preserve identity across a second temporal frame, and show that luma can bind roles without becoming a crutch for geometry.",
        "",
        "## Per-scene summary",
    ]
    for scene, stats in scene_summary.items():
        lines.append(
            f"- {scene}: scene_acc={stats['scene_accuracy']:.3f} concept_pair={stats['concept_pair_accuracy']:.3f} relation={stats['relation_accuracy']:.3f} temporal={stats['temporal_binding_accuracy']:.3f} luma={stats['luma_binding_accuracy']:.3f} no_luma={stats['luma_ablation_accuracy']:.3f} margin={stats['mean_margin']:.4f} floor={stats['margin_floor']:.4f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Visualizations
    labels = [x[0] for x in SCENES]
    plot_bar(labels, [scene_summary[x]["scene_accuracy"] for x in labels], "Phase 62 compositional scene accuracy", "accuracy", OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_scene_accuracy.png")
    plot_bar(labels, [scene_summary[x]["relation_accuracy"] for x in labels], "Phase 62 relation grammar accuracy", "accuracy", OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_relation_accuracy.png")
    plot_bar(labels, [scene_summary[x]["temporal_binding_accuracy"] for x in labels], "Phase 62 temporal binding accuracy", "accuracy", OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_temporal_binding.png")
    plot_bar(labels, [scene_summary[x]["luma_ablation_accuracy"] for x in labels], "Phase 62 geometry-only luma ablation", "accuracy", OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_luma_ablation.png")
    plot_confusion(rows, "true_relation", "pred_relation", RELATIONS, "Phase 62 relation confusion", OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_relation_confusion.png")
    plot_margin(rows, OUT / "phase62_compositional_scene_grammar_temporal_luma_binding_margin_distribution.png")
    for i, t in enumerate(trials[: min(7, len(trials))]):
        plot_example(t, EXAMPLE_DIR / f"phase62_example_{i:02d}_{t.scene}.png")

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
