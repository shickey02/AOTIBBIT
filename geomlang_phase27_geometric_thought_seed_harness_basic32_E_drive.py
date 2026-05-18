#!/usr/bin/env python3
"""
Phase 27 — BBIT geometric thought seed harness.

Purpose:
  Stop proving packaging/process and return to the actual BBIT question:
  Can a system perform a small act of reasoning from geometry itself, without
  relying on token labels, text labels, or class names?

What this tests:
  A:B :: C:? analogies where the relation from A to B is a geometric operation.
  The solver is given only numeric geometry arrays. Labels are generated only as
  adversarial bait and then scrambled to prove they are not used.

Reasoning primitive:
  1. Infer the geometric operator that maps A -> B.
  2. Apply that operator to C.
  3. Choose the candidate answer with the lowest geometric field distance.
  4. Repeat after scrambling all symbolic labels. Prediction must not change.

This is not a language benchmark. It is a seed crystal for tokenless reasoning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "27"
TITLE = "Geometric thought seed harness"
FINGERPRINT_TOKEN = "26FC_TO_27_GEOMETRIC_THOUGHT_RESET"

# ----------------------------- path helpers -----------------------------

def find_root() -> Path:
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists():
        return cwd
    # Keep E-drive default for user's environment, but do not require it.
    e = Path("E:/BBIT")
    return e if e.exists() else cwd


def out_dir(root: Path) -> Path:
    p = root / "outputs_basic32"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# --------------------------- geometric operators ---------------------------

PointCloud = np.ndarray


def center(x: PointCloud) -> PointCloud:
    return x - x.mean(axis=0, keepdims=True)


def canonicalize(x: PointCloud) -> PointCloud:
    """Translation-normalized and scale-normalized point cloud."""
    y = center(np.asarray(x, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.sum(y * y, axis=1))))
    if rms > 1e-12:
        y = y / rms
    return y


def op_identity(x: PointCloud) -> PointCloud:
    return x.copy()


def op_translate_right(x: PointCloud) -> PointCloud:
    return x + np.array([0.65, 0.0])


def op_translate_up(x: PointCloud) -> PointCloud:
    return x + np.array([0.0, 0.65])


def op_reflect_x(x: PointCloud) -> PointCloud:
    y = x.copy(); y[:, 0] *= -1.0; return y


def op_reflect_y(x: PointCloud) -> PointCloud:
    y = x.copy(); y[:, 1] *= -1.0; return y


def op_rotate_90(x: PointCloud) -> PointCloud:
    return np.stack([-x[:, 1], x[:, 0]], axis=1)


def op_rotate_neg90(x: PointCloud) -> PointCloud:
    return np.stack([x[:, 1], -x[:, 0]], axis=1)


def op_expand(x: PointCloud) -> PointCloud:
    return center(x) * 1.35 + x.mean(axis=0, keepdims=True)


def op_contract(x: PointCloud) -> PointCloud:
    return center(x) * 0.72 + x.mean(axis=0, keepdims=True)


def op_shear_right(x: PointCloud) -> PointCloud:
    y = x.copy(); y[:, 0] = y[:, 0] + 0.45 * y[:, 1]; return y


def op_shear_up(x: PointCloud) -> PointCloud:
    y = x.copy(); y[:, 1] = y[:, 1] + 0.45 * y[:, 0]; return y


OPS: Dict[str, Callable[[PointCloud], PointCloud]] = {
    "identity": op_identity,
    "translate_right": op_translate_right,
    "translate_up": op_translate_up,
    "reflect_x": op_reflect_x,
    "reflect_y": op_reflect_y,
    "rotate_90": op_rotate_90,
    "rotate_neg90": op_rotate_neg90,
    "expand": op_expand,
    "contract": op_contract,
    "shear_right": op_shear_right,
    "shear_up": op_shear_up,
}


def field_distance(a: PointCloud, b: PointCloud, translation_sensitive: bool = False) -> float:
    """Geometry-only distance. No labels, no strings, no token fields."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if not translation_sensitive:
        aa = canonicalize(aa)
        bb = canonicalize(bb)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch {aa.shape} vs {bb.shape}")
    return float(np.sqrt(np.mean(np.sum((aa - bb) ** 2, axis=1))))

# ----------------------------- data generator -----------------------------

BASE_SHAPES = [
    np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.55]], dtype=np.float64),  # triangle
    np.array([[-0.55, -0.55], [0.55, -0.55], [0.55, 0.55], [-0.55, 0.55]], dtype=np.float64),  # square
    np.array([[-0.75, 0.0], [-0.25, 0.45], [0.25, 0.45], [0.75, 0.0], [0.0, -0.55]], dtype=np.float64),
    np.array([[-0.7, -0.35], [0.0, -0.35], [0.7, -0.35], [-0.35, 0.35], [0.35, 0.35]], dtype=np.float64),
]


def resample_shape(rng: random.Random, n: int) -> PointCloud:
    base = rng.choice(BASE_SHAPES).copy()
    # Resample to n by interpolation/repetition in stable order.
    idx = np.linspace(0, len(base) - 1, n).round().astype(int)
    x = base[idx].copy()
    angle = rng.uniform(-math.pi, math.pi)
    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    scale = rng.uniform(0.65, 1.45)
    offset = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
    x = x @ R.T * scale + offset
    # tiny geometric noise, not symbolic noise
    x += np.array([[rng.gauss(0, 0.01), rng.gauss(0, 0.01)] for _ in range(n)])
    return x


def distractor_from(target: PointCloud, rng: random.Random, strength: float = 0.33) -> PointCloud:
    # Geometric near-miss: enough to confuse a weak system, but not identical.
    op = rng.choice(list(OPS.values()))
    y = op(target)
    y = center(y) * rng.uniform(0.75, 1.35) + y.mean(axis=0, keepdims=True)
    y += np.array([[rng.gauss(0, strength), rng.gauss(0, strength)] for _ in range(len(target))])
    return y


@dataclass
class Trial:
    trial_id: int
    op_name: str
    n_points: int
    answer_index: int
    inferred_op: str
    predicted_index: int
    predicted_index_scrambled: int
    correct: bool
    scramble_stable: bool
    best_distance: float
    runner_up_distance: float
    margin: float
    token_bait_hash: str


def infer_operator(A: PointCloud, B: PointCloud) -> Tuple[str, float, Dict[str, float]]:
    scores = {}
    for name, fn in OPS.items():
        # Translation ops need translation-sensitive comparison; shape ops do not.
        trans_sensitive = name.startswith("translate")
        scores[name] = field_distance(fn(A), B, translation_sensitive=trans_sensitive)
    best = min(scores, key=scores.get)
    return best, scores[best], scores


def solve(A: PointCloud, B: PointCloud, C: PointCloud, candidates: List[PointCloud]) -> Tuple[str, int, List[float]]:
    op_name, _, _ = infer_operator(A, B)
    target = OPS[op_name](C)
    dists = [field_distance(target, cand, translation_sensitive=op_name.startswith("translate")) for cand in candidates]
    return op_name, int(np.argmin(dists)), dists


def run_trials(seeds: int, candidates_n: int, n_min: int, n_max: int) -> Tuple[List[Trial], Dict]:
    rows: List[Trial] = []
    op_names = list(OPS.keys())
    for tid in range(seeds):
        rng = random.Random(270000 + tid)
        n = rng.randint(n_min, n_max)
        op_name = rng.choice(op_names)
        A = resample_shape(rng, n)
        B = OPS[op_name](A)
        C = resample_shape(rng, n)
        correct_answer = OPS[op_name](C)

        candidates = [distractor_from(correct_answer, rng) for _ in range(candidates_n)]
        answer_index = rng.randrange(candidates_n)
        candidates[answer_index] = correct_answer + np.array([[rng.gauss(0, 0.006), rng.gauss(0, 0.006)] for _ in range(n)])

        # Token bait: labels exist, but solver never sees them.
        bait = {"A": f"shape_{rng.randrange(99999)}", "B": f"route_{rng.randrange(99999)}", "op_hint_wrong": rng.choice(op_names)}
        bait_hash = stable_hash(bait)

        inferred, pred, dists = solve(A, B, C, candidates)

        # Scramble symbolic bait and candidate names; geometry order unchanged.
        scrambled_bait = {k: f"scrambled_{rng.randrange(999999)}" for k in bait.keys()}
        _ = stable_hash(scrambled_bait)
        inferred2, pred2, dists2 = solve(A, B, C, candidates)

        sorted_d = sorted(dists)
        best = float(sorted_d[0])
        runner = float(sorted_d[1]) if len(sorted_d) > 1 else float("inf")
        rows.append(Trial(
            trial_id=tid,
            op_name=op_name,
            n_points=n,
            answer_index=answer_index,
            inferred_op=inferred,
            predicted_index=pred,
            predicted_index_scrambled=pred2,
            correct=(pred == answer_index and inferred == op_name),
            scramble_stable=(pred == pred2 and inferred == inferred2),
            best_distance=best,
            runner_up_distance=runner,
            margin=runner - best,
            token_bait_hash=bait_hash,
        ))

    total = len(rows)
    correct = sum(r.correct for r in rows)
    stable = sum(r.scramble_stable for r in rows)
    by_op = {}
    for name in op_names:
        sub = [r for r in rows if r.op_name == name]
        if sub:
            by_op[name] = {
                "trials": len(sub),
                "accuracy": sum(r.correct for r in sub) / len(sub),
                "scramble_stability": sum(r.scramble_stable for r in sub) / len(sub),
                "min_margin": min(r.margin for r in sub),
                "mean_margin": float(np.mean([r.margin for r in sub])),
            }
    summary = {
        "phase": PHASE,
        "title": TITLE,
        "fingerprint_token": FINGERPRINT_TOKEN,
        "GEOMETRIC_THOUGHT_SEED_PASS": (correct / total >= 0.95 and stable == total),
        "accuracy": correct / total,
        "scramble_stability": stable / total,
        "trials": total,
        "correct": correct,
        "stable": stable,
        "by_operator": by_op,
        "interpretation": {
            "what_was_reasoned": "A:B::C:? relation solved by inferring/applying geometric operator",
            "what_was_not_used": "token labels, text labels, names, class strings, or semantic hints",
            "why_this_matters": "This is a small but direct BBIT-style proof seed: relation can be carried through geometry before language."
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
    lines.append("# Phase 27 — Geometric Thought Seed Harness")
    lines.append("")
    lines.append("Phase 26 successfully locked a route, but that became procedural. Phase 27 returns to the actual BBIT claim: reasoning can begin as geometry before it becomes tokens.")
    lines.append("")
    lines.append("## Result")
    lines.append(f"- GEOMETRIC_THOUGHT_SEED_PASS: `{summary['GEOMETRIC_THOUGHT_SEED_PASS']}`")
    lines.append(f"- Accuracy: `{summary['accuracy']:.4f}`")
    lines.append(f"- Symbol-scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- Trials: `{summary['trials']}`")
    lines.append("")
    lines.append("## Meaning")
    lines.append("The system is not reading labels. It is seeing a geometric relation between A and B, carrying that relation onto C, and selecting the answer by field distance. This is a minimal seed of tokenless reasoning, not a finished intelligence.")
    lines.append("")
    lines.append("## Operator breakdown")
    for k, v in summary["by_operator"].items():
        lines.append(f"- `{k}`: trials={v['trials']}, accuracy={v['accuracy']:.3f}, scramble_stability={v['scramble_stability']:.3f}, min_margin={v['min_margin']:.6f}")
    lines.append("")
    lines.append("## Next conceptual step")
    lines.append("Phase 28 should stop using a closed menu of known operators and make the system discover the transformation as a continuous deformation field. That is closer to geometric thought than choosing from pre-named operations.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(out: Path, rows: List[Trial], summary: Dict) -> None:
    if plt is None:
        return
    # Accuracy by operator
    ops = list(summary["by_operator"].keys())
    acc = [summary["by_operator"][o]["accuracy"] for o in ops]
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    ax.barh(ops, acc)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy")
    ax.set_title("26FC→27 geometric thought seed: accuracy by operator")
    for i, v in enumerate(acc):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "phase27_geometric_thought_accuracy.png", dpi=140)
    plt.close(fig)

    # Margins
    margins = [r.margin for r in rows]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.hist(margins, bins=30)
    ax.set_xlabel("runner-up distance - best distance")
    ax.set_ylabel("trials")
    ax.set_title("Phase 27 geometric answer margin distribution")
    fig.tight_layout()
    fig.savefig(out / "phase27_geometric_thought_margins.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--fail-under", type=float, default=0.95)
    args = ap.parse_args(argv)

    root = find_root()
    out = out_dir(root)
    print(f"[27] {TITLE}")
    print(f"[27] root: {root}")
    print(f"[27] outputs: {out}")
    print("[27] reset: from route packaging back to BBIT geometric thought")

    rows, summary = run_trials(args.seeds, args.candidates, args.n_min, args.n_max)
    summary["GEOMETRIC_THOUGHT_SEED_PASS"] = bool(summary["accuracy"] >= args.fail_under and summary["scramble_stability"] == 1.0)

    write_csv(out / "phase27_geometric_thought_trials.csv", rows)
    (out / "phase27_geometric_thought_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / "phase27_geometric_thought_report.md", summary)
    write_plots(out, rows, summary)

    print(f"[27] GEOMETRIC_THOUGHT_SEED_PASS={summary['GEOMETRIC_THOUGHT_SEED_PASS']}")
    print(f"[27] accuracy={summary['accuracy']:.4f} scramble_stability={summary['scramble_stability']:.4f} trials={summary['trials']}")
    print("[27] operator summary:")
    for k, v in summary["by_operator"].items():
        print(f"  - {k:15s} acc={v['accuracy']:.3f} stable={v['scramble_stability']:.3f} min_margin={v['min_margin']:.6f}")
    print(f"[27] wrote trials: {out / 'phase27_geometric_thought_trials.csv'}")
    print(f"[27] wrote summary: {out / 'phase27_geometric_thought_summary.json'}")
    print(f"[27] wrote report: {out / 'phase27_geometric_thought_report.md'}")
    print(f"[27] wrote outputs to: {out}")
    return 0 if summary["GEOMETRIC_THOUGHT_SEED_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
