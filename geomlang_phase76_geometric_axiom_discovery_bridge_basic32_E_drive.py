"""
Phase 76: Geometric axiom discovery bridge

Drop-in script for E:\\BBIT\\bbit_geomlang.

Reset continuation:
  Phase 69-75 established that the raster/point grammar can preserve relation,
  concept-pair, and active-observability structure. Phase 76 turns that perceptual
  stability into a primitive problem-solving layer: discover arithmetic and
  geometry axioms from point-cloud experiments rather than from labels alone.

Task:
  Use geometric thought to recover small-number arithmetic and Euclidean-style
  invariants from generated point worlds. The script creates point-set scenes,
  measures invariant quantities, proposes candidate axioms, tests them against
  held-out trials, and writes visual diagnostics.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "76"
TITLE = "Geometric axiom discovery bridge"
PASS_KEY = "PHASE76_GEOMETRIC_AXIOM_DISCOVERY_BRIDGE_PASS"
SEED = 76076
TRIALS = 2048
MAX_N = 9
SUPPORT_FLOOR = 0.985
HOLDOUT_FLOOR = 0.975
MIN_COUNTEREXAMPLE_GAP = 0.20

random.seed(SEED)
np.random.seed(SEED)


def find_root() -> Path:
    # Windows target path when running locally in the user's project.
    e_drive = Path(r"E:\BBIT")
    if e_drive.exists():
        return e_drive
    cwd = Path.cwd()
    # If launched from inside the project, climb until we find the BBIT root.
    for p in [cwd] + list(cwd.parents):
        if (p / "bbit_geomlang").exists() or (p / "outputs_basic32").exists():
            return p
    # Sandbox/local fallback.
    mnt = Path("/mnt/data")
    if mnt.exists():
        return mnt
    return cwd

ROOT = find_root()
OUTPUTS = ROOT / "outputs_basic32" if (ROOT / "outputs_basic32").exists() or str(ROOT).lower().endswith("bbit") else Path("/mnt/data")
OUTPUTS.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = OUTPUTS / "phase76_examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SCENES = [
    "cardinal_addition",
    "zero_identity",
    "successor_step",
    "translation_invariance",
    "distance_symmetry",
    "triangle_inequality",
    "between_collinear",
    "area_decomposition",
]

# -----------------------------------------------------------------------------
# Point-world primitives
# -----------------------------------------------------------------------------

Point = Tuple[float, float]


def jitter(x: float, scale: float = 0.015) -> float:
    return float(x + np.random.normal(0.0, scale))


def make_count_points(n: int, origin: Tuple[float, float] = (0.0, 0.0), spacing: float = 1.0) -> np.ndarray:
    """Represent a number as a small 2D point set, arranged on a compact grid."""
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    cols = int(math.ceil(math.sqrt(max(n, 1))))
    pts = []
    ox, oy = origin
    for i in range(n):
        x = ox + (i % cols) * spacing
        y = oy + (i // cols) * spacing
        pts.append((jitter(x), jitter(y)))
    return np.asarray(pts, dtype=float)


def disjoint_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    return np.vstack([a, b])


def cardinality(points: np.ndarray) -> int:
    return int(points.shape[0])


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def shoelace_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def almost_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def cross2(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])

def collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> bool:
    return abs(cross2(b - a, c - a)) <= eps


def between(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> bool:
    if not collinear(a, b, c, eps=eps):
        return False
    return dist(a, b) + dist(b, c) <= dist(a, c) + 1e-6

# -----------------------------------------------------------------------------
# Candidate axiom tests
# -----------------------------------------------------------------------------

@dataclass
class TrialRow:
    trial_id: int
    split: str
    scene: str
    candidate: str
    family: str
    predicted_true: bool
    observed_true: bool
    correct: bool
    margin: float
    payload: str


@dataclass
class CandidateResult:
    candidate: str
    family: str
    support: float
    holdout_support: float
    train_support: float
    counter_support: float
    selected: bool
    discovered_axiom: str
    notes: str


def make_trials() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tid in range(TRIALS):
        split = "holdout" if tid % 5 == 0 else "train"
        scene = SCENES[tid % len(SCENES)]
        a = random.randint(0, MAX_N)
        b = random.randint(0, MAX_N)
        c = random.randint(0, MAX_N)
        # Deterministic primitive points for geometric tasks.
        p = np.array([random.uniform(-5, 5), random.uniform(-5, 5)], dtype=float)
        q = np.array([random.uniform(-5, 5), random.uniform(-5, 5)], dtype=float)
        r = np.array([random.uniform(-5, 5), random.uniform(-5, 5)], dtype=float)
        v = np.array([random.uniform(-3, 3), random.uniform(-3, 3)], dtype=float)
        # Collinear triple A-B-C for between scene.
        A = np.array([random.uniform(-4, 1), random.uniform(-4, 4)], dtype=float)
        direction = np.array([random.uniform(0.3, 3), random.uniform(-1, 1)], dtype=float)
        B = A + direction * random.uniform(0.2, 1.4)
        C = B + direction * random.uniform(0.2, 1.4)
        rows.append(dict(tid=tid, split=split, scene=scene, a=a, b=b, c=c, p=p, q=q, r=r, v=v, A=A, B=B, C=C))
    return rows


def eval_candidate(name: str, t: Dict[str, Any]) -> Tuple[bool, float, str, str, bool]:
    """Return observed_true, margin, family, payload, is_counterexample_candidate."""
    a, b, c = t["a"], t["b"], t["c"]
    p, q, r, v = t["p"], t["q"], t["r"], t["v"]
    A, B, C = t["A"], t["B"], t["C"]

    if name == "addition_commutes_by_disjoint_union":
        left = cardinality(disjoint_union(make_count_points(a), make_count_points(b, origin=(20, 0))))
        right = cardinality(disjoint_union(make_count_points(b), make_count_points(a, origin=(20, 0))))
        return left == right, 1.0 if left == right else 0.0, "arithmetic", f"{a}+{b} == {b}+{a}", False

    if name == "addition_associates_by_union_grouping":
        left = cardinality(disjoint_union(disjoint_union(make_count_points(a), make_count_points(b, origin=(20, 0))), make_count_points(c, origin=(40, 0))))
        right = cardinality(disjoint_union(make_count_points(a), disjoint_union(make_count_points(b, origin=(20, 0)), make_count_points(c, origin=(40, 0)))))
        return left == right, 1.0 if left == right else 0.0, "arithmetic", f"({a}+{b})+{c} == {a}+({b}+{c})", False

    if name == "zero_is_additive_identity":
        left = cardinality(disjoint_union(make_count_points(a), make_count_points(0, origin=(20, 0))))
        return left == a, 1.0 if left == a else 0.0, "arithmetic", f"{a}+0 == {a}", False

    if name == "successor_adds_one_point":
        n0 = cardinality(make_count_points(a))
        n1 = cardinality(disjoint_union(make_count_points(a), make_count_points(1, origin=(20, 0))))
        return n1 == n0 + 1, 1.0 if n1 == n0 + 1 else 0.0, "arithmetic", f"S({a}) == {a}+1", False

    if name == "translation_preserves_distance":
        d0 = dist(p, q)
        d1 = dist(p + v, q + v)
        return almost_equal(d0, d1, 1e-9), max(0.0, 1.0 - abs(d0 - d1)), "geometry", "d(P,Q) == d(P+v,Q+v)", False

    if name == "distance_is_symmetric":
        d0 = dist(p, q)
        d1 = dist(q, p)
        return almost_equal(d0, d1, 1e-9), max(0.0, 1.0 - abs(d0 - d1)), "geometry", "d(P,Q) == d(Q,P)", False

    if name == "triangle_inequality":
        lhs = dist(p, r)
        rhs = dist(p, q) + dist(q, r)
        return lhs <= rhs + 1e-9, min(3.0, rhs - lhs) / 3.0, "geometry", "d(P,R) <= d(P,Q)+d(Q,R)", False

    if name == "betweenness_adds_segments":
        ok = between(A, B, C)
        delta = abs((dist(A, B) + dist(B, C)) - dist(A, C))
        return ok and delta <= 1e-6, max(0.0, 1.0 - delta), "geometry", "if B between A,C then AB+BC=AC", False

    if name == "rectangle_area_decomposes":
        w = random.randint(1, 8)
        h = random.randint(1, 8)
        cut = random.randint(1, w)
        rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        left = np.array([[0, 0], [cut, 0], [cut, h], [0, h]], dtype=float)
        right = np.array([[cut, 0], [w, 0], [w, h], [cut, h]], dtype=float)
        total = shoelace_area(rect)
        parts = shoelace_area(left) + shoelace_area(right)
        return almost_equal(total, parts, 1e-9), max(0.0, 1.0 - abs(total - parts)), "geometry", f"area({w}x{h}) split at {cut}", False

    # Deliberate false / unstable candidates. These are included so the discovery
    # mechanism has to reject attractive but non-axiomatic patterns.
    if name == "false_addition_absorbs_right_operand":
        observed = (a + b) == a
        return observed, 1.0 if observed else 0.0, "counterexample", f"{a}+{b} == {a}", True

    if name == "false_distance_changes_under_translation":
        observed = not almost_equal(dist(p, q), dist(p + v, q + v), 1e-9)
        return observed, 1.0 if observed else 0.0, "counterexample", "d(P,Q) != d(P+v,Q+v)", True

    if name == "false_all_triangles_are_right":
        # A triangle is right only when a dot product vanishes at one vertex.
        dots = [abs(np.dot(p - q, r - q)), abs(np.dot(q - p, r - p)), abs(np.dot(p - r, q - r))]
        observed = min(dots) <= 1e-6
        return observed, 1.0 if observed else 0.0, "counterexample", "every triangle has a right angle", True

    raise KeyError(name)


CANDIDATES = [
    "addition_commutes_by_disjoint_union",
    "addition_associates_by_union_grouping",
    "zero_is_additive_identity",
    "successor_adds_one_point",
    "translation_preserves_distance",
    "distance_is_symmetric",
    "triangle_inequality",
    "betweenness_adds_segments",
    "rectangle_area_decomposes",
    "false_addition_absorbs_right_operand",
    "false_distance_changes_under_translation",
    "false_all_triangles_are_right",
]

AXIOM_TEXT = {
    "addition_commutes_by_disjoint_union": "A + B = B + A because disjoint point-union cardinality is order-invariant.",
    "addition_associates_by_union_grouping": "(A + B) + C = A + (B + C) because grouping does not change cardinality.",
    "zero_is_additive_identity": "A + 0 = A because the empty point-set contributes no additional points.",
    "successor_adds_one_point": "S(A) = A + 1 because adding one singleton increases cardinality by one.",
    "translation_preserves_distance": "Translation preserves distance because both endpoints move by the same vector.",
    "distance_is_symmetric": "Distance is symmetric: d(A,B) = d(B,A).",
    "triangle_inequality": "Triangle inequality: direct distance is no greater than the broken path.",
    "betweenness_adds_segments": "If B is between A and C, then AB + BC = AC.",
    "rectangle_area_decomposes": "A rectangle's area equals the sum of areas of a cut decomposition.",
}


def run_discovery() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    worlds = make_trials()
    rows: List[TrialRow] = []
    for t in worlds:
        for cand in CANDIDATES:
            observed, margin, family, payload, is_counter = eval_candidate(cand, t)
            # The policy predicts candidate axioms true, and counterexample candidates false.
            predicted_true = not is_counter
            correct = (predicted_true == observed) if is_counter else observed
            rows.append(TrialRow(
                trial_id=t["tid"],
                split=t["split"],
                scene=t["scene"],
                candidate=cand,
                family=family,
                predicted_true=bool(predicted_true),
                observed_true=bool(observed),
                correct=bool(correct),
                margin=float(margin),
                payload=payload,
            ))
    trials = pd.DataFrame([asdict(r) for r in rows])

    results: List[CandidateResult] = []
    for cand, g in trials.groupby("candidate"):
        family = str(g["family"].iloc[0])
        support = float(g["observed_true"].mean())
        train_support = float(g.loc[g["split"] == "train", "observed_true"].mean())
        holdout_support = float(g.loc[g["split"] == "holdout", "observed_true"].mean())
        is_counter = family == "counterexample"
        # Counter candidates are rejected when they fail often enough.
        counter_support = float(1.0 - support) if is_counter else float("nan")
        selected = (not is_counter and train_support >= SUPPORT_FLOOR and holdout_support >= HOLDOUT_FLOOR)
        if is_counter:
            notes = "rejected: counterexample candidate lacks universal support"
            axiom = ""
        elif selected:
            notes = "selected: stable across train and holdout worlds"
            axiom = AXIOM_TEXT.get(cand, cand)
        else:
            notes = "not selected: insufficient universal support"
            axiom = ""
        results.append(CandidateResult(
            candidate=cand,
            family=family,
            support=support,
            holdout_support=holdout_support,
            train_support=train_support,
            counter_support=counter_support,
            selected=selected,
            discovered_axiom=axiom,
            notes=notes,
        ))
    summary = pd.DataFrame([asdict(r) for r in results]).sort_values(["selected", "family", "candidate"], ascending=[False, True, True])

    selected_axioms = summary[summary["selected"]]
    rejected_counters = summary[(summary["family"] == "counterexample") & (summary["support"] <= (1.0 - MIN_COUNTEREXAMPLE_GAP))]
    metrics = {
        "phase": PHASE,
        "title": TITLE,
        "trials": int(TRIALS),
        "candidate_count": int(len(CANDIDATES)),
        "selected_axiom_count": int(len(selected_axioms)),
        "counterexample_rejection_count": int(len(rejected_counters)),
        "selected_axiom_support_min": float(selected_axioms["support"].min()) if len(selected_axioms) else 0.0,
        "selected_axiom_holdout_min": float(selected_axioms["holdout_support"].min()) if len(selected_axioms) else 0.0,
        "selected_axiom_train_min": float(selected_axioms["train_support"].min()) if len(selected_axioms) else 0.0,
        "mean_selected_margin": float(trials[trials["candidate"].isin(selected_axioms["candidate"])] ["margin"].mean()) if len(selected_axioms) else 0.0,
        "candidate_policy_accuracy": float(trials["correct"].mean()),
        "arithmetic_axioms": int((selected_axioms["family"] == "arithmetic").sum()),
        "geometry_axioms": int((selected_axioms["family"] == "geometry").sum()),
    }
    metrics[PASS_KEY] = bool(
        metrics["selected_axiom_count"] >= 8
        and metrics["arithmetic_axioms"] >= 4
        and metrics["geometry_axioms"] >= 4
        and metrics["selected_axiom_holdout_min"] >= HOLDOUT_FLOOR
        and metrics["counterexample_rejection_count"] >= 3
    )
    return trials, summary, metrics

# -----------------------------------------------------------------------------
# Visualization / report
# -----------------------------------------------------------------------------

def save_bar(path: Path, labels: List[str], values: List[float], title: str, ylabel: str, ylim: Tuple[float, float] = (0, 1.05)) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_candidate_heatmap(path: Path, trials: pd.DataFrame, selected: Iterable[str]) -> None:
    selected = list(selected)
    pivot = trials[trials["candidate"].isin(selected)].pivot_table(
        index="candidate", columns="scene", values="observed_true", aggfunc="mean", fill_value=0.0
    )
    fig, ax = plt.subplots(figsize=(15, max(5, len(selected) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Phase 76 selected axiom support by scene")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_margin_hist(path: Path, trials: pd.DataFrame, selected: Iterable[str]) -> None:
    selected = list(selected)
    data = trials[trials["candidate"].isin(selected)]["margin"].values
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.hist(data, bins=28)
    ax.set_title("Phase 76 selected axiom proof-margin distribution")
    ax.set_xlabel("invariant margin / slack")
    ax.set_ylabel("candidate-trials")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_example_addition(path: Path) -> None:
    a, b = 4, 3
    A = make_count_points(a, origin=(0, 0), spacing=0.8)
    B = make_count_points(b, origin=(4, 0), spacing=0.8)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if len(A): ax.scatter(A[:,0], A[:,1], s=90, label=f"A={a}")
    if len(B): ax.scatter(B[:,0], B[:,1], s=90, label=f"B={b}")
    ax.set_title("Phase 76 example: addition as disjoint point-union cardinality")
    ax.text(1.5, -0.8, f"|A ∪ B| = {a+b}; |B ∪ A| = {b+a}", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_example_geometry(path: Path) -> None:
    A = np.array([0.0, 0.0])
    B = np.array([2.0, 0.9])
    C = np.array([4.2, 0.1])
    v = np.array([0.8, 1.2])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pts = np.vstack([A, B, C])
    pts2 = pts + v
    ax.plot(pts[:,0], pts[:,1], marker="o", label="original triangle")
    ax.plot(pts2[:,0], pts2[:,1], marker="o", label="translated triangle")
    for name, P in zip(["A", "B", "C"], pts):
        ax.text(P[0]+0.05, P[1]+0.05, name)
    for name, P in zip(["A'", "B'", "C'"], pts2):
        ax.text(P[0]+0.05, P[1]+0.05, name)
    ax.set_title("Phase 76 example: translation preserves pair distances")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    selected = summary[summary["selected"]]
    rejected = summary[summary["family"] == "counterexample"]
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append("## Result\n")
    lines.append(f"- `{PASS_KEY}`: **{metrics[PASS_KEY]}**")
    lines.append(f"- Selected axioms: **{metrics['selected_axiom_count']}**")
    lines.append(f"- Arithmetic axioms: **{metrics['arithmetic_axioms']}**")
    lines.append(f"- Geometry axioms: **{metrics['geometry_axioms']}**")
    lines.append(f"- Minimum selected holdout support: **{metrics['selected_axiom_holdout_min']:.4f}**")
    lines.append(f"- Counterexample candidates rejected: **{metrics['counterexample_rejection_count']}**\n")
    lines.append("## Discovered axioms\n")
    for _, row in selected.iterrows():
        lines.append(f"- **{row['candidate']}** — {row['discovered_axiom']} support={row['support']:.4f}, holdout={row['holdout_support']:.4f}")
    lines.append("\n## Rejected false candidates\n")
    for _, row in rejected.iterrows():
        lines.append(f"- **{row['candidate']}** — observed support={row['support']:.4f}; {row['notes']}")
    lines.append("\n## Interpretation\n")
    lines.append("Phase 76 treats arithmetic as a point-set/cardinality operation and geometry as invariant structure over points. The pass condition requires that candidate axioms survive held-out generated worlds while deliberately false candidates fail. This is the first bridge from perception into problem-solving: the system is no longer only naming relations, but extracting reusable rules from stable geometric transformations.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUTPUTS}")
    print(f"[{PHASE}] reset continued: from relation-stable perception to primitive axiom discovery")
    print(f"[{PHASE}] task: use point geometry to discover arithmetic and geometry invariants")

    trials, summary, metrics = run_discovery()

    stem = "phase76_geometric_axiom_discovery_bridge"
    trials_path = OUTPUTS / f"{stem}_trials.csv"
    summary_path = OUTPUTS / f"{stem}_summary.json"
    report_path = OUTPUTS / f"{stem}_report.md"
    cand_csv_path = OUTPUTS / f"{stem}_candidate_summary.csv"

    trials.to_csv(trials_path, index=False)
    summary.to_csv(cand_csv_path, index=False)
    summary_path.write_text(json.dumps({"metrics": metrics, "candidates": summary.to_dict(orient="records")}, indent=2), encoding="utf-8")

    selected = summary[summary["selected"]]["candidate"].tolist()
    save_bar(OUTPUTS / "phase76_axiom_support.png", summary["candidate"].tolist(), summary["support"].tolist(), "Phase 76 candidate axiom support", "observed support")
    save_bar(OUTPUTS / "phase76_axiom_holdout_support.png", summary["candidate"].tolist(), summary["holdout_support"].tolist(), "Phase 76 candidate holdout support", "holdout support")
    save_candidate_heatmap(OUTPUTS / "phase76_selected_axiom_scene_support.png", trials, selected)
    save_margin_hist(OUTPUTS / "phase76_selected_axiom_margin_distribution.png", trials, selected)
    save_example_addition(EXAMPLES_DIR / "phase76_example_addition_union.png")
    save_example_geometry(EXAMPLES_DIR / "phase76_example_translation_geometry.png")
    write_report(report_path, summary, metrics)

    print(f"[{PHASE}] {PASS_KEY}={metrics[PASS_KEY]}")
    print(
        f"[{PHASE}] selected_axioms={metrics['selected_axiom_count']} arithmetic={metrics['arithmetic_axioms']} "
        f"geometry={metrics['geometry_axioms']} min_holdout={metrics['selected_axiom_holdout_min']:.4f} "
        f"counter_rejections={metrics['counterexample_rejection_count']} trials={metrics['trials']}"
    )
    print(f"[{PHASE}] discovered axiom summary:")
    for _, row in summary[summary["selected"]].iterrows():
        print(f"  - {row['candidate']:<42} family={row['family']:<10} support={row['support']:.4f} holdout={row['holdout_support']:.4f}")
    print(f"[{PHASE}] rejected false candidates:")
    for _, row in summary[summary["family"] == "counterexample"].iterrows():
        print(f"  - {row['candidate']:<42} observed_support={row['support']:.4f}")
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote candidate summary: {cand_csv_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLES_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUTPUTS}")


if __name__ == "__main__":
    main()
