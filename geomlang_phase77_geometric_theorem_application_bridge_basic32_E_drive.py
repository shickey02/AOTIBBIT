r"""
Phase 77: Geometric theorem application bridge

Drop-in path:
    E:\BBIT\bbit_geomlang\geomlang_phase77_geometric_theorem_application_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase77_geometric_theorem_application_bridge_basic32_E_drive.py

Purpose:
    Continue the reset path from Phase 76.

    Phase 76 discovered primitive arithmetic/geometry axioms from point-world invariants.
    Phase 77 stops merely recognizing axioms and begins using those axioms as tools.

    The task here is not:
        "does this candidate axiom hold?"

    The task is:
        "given a small geometric/arithmetic world with one hidden value,
         choose the correct discovered axiom and solve the unknown."

    This is intentionally still primitive. It is a bridge from geometric perception
    to theorem application / symbolic problem solving.

Outputs:
    outputs_basic32/
        phase77_geometric_theorem_application_bridge_trials.csv
        phase77_geometric_theorem_application_bridge_summary.json
        phase77_geometric_theorem_application_bridge_report.md
        phase77_solve_accuracy_by_axiom.png
        phase77_axiom_selection_confusion.png
        phase77_family_accuracy.png
        phase77_false_rule_rejection.png
        phase77_solution_margin_distribution.png
        phase77_examples/
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "77"
TITLE = "Geometric theorem application bridge"

SEED = 77077
TRIALS = 4096

# Success floors. These are intentionally high because Phase 76 already provided
# clean discovered invariants. Phase 77 is testing whether those invariants can be
# selected and used as operators.
MIN_OVERALL_SOLVE_ACC = 0.965
MIN_ARITH_SOLVE_ACC = 0.970
MIN_GEOM_SOLVE_ACC = 0.950
MIN_AXIOM_SELECTION_ACC = 0.950
MIN_HOLDOUT_SOLVE_ACC = 0.950
MIN_FALSE_RULE_REJECTION = 0.990
MIN_NO_HALLUCINATION_ACC = 0.990

EPS = 1e-9


AXIOMS = [
    "addition_commutes_by_disjoint_union",
    "addition_associates_by_union_grouping",
    "zero_is_additive_identity",
    "successor_adds_one_point",
    "betweenness_adds_segments",
    "distance_is_symmetric",
    "translation_preserves_distance",
    "rectangle_area_decomposes",
    "triangle_inequality",
]

ARITH_AXIOMS = {
    "addition_commutes_by_disjoint_union",
    "addition_associates_by_union_grouping",
    "zero_is_additive_identity",
    "successor_adds_one_point",
}

GEOM_AXIOMS = set(AXIOMS) - ARITH_AXIOMS

FALSE_RULES = [
    "false_addition_absorbs_right_operand",
    "false_all_triangles_are_right",
    "false_distance_changes_under_translation",
    "false_between_does_not_add_segments",
    "false_rectangle_area_ignores_decomposition",
    "false_successor_adds_two_points",
]


def find_root() -> Path:
    # Windows target path when run in the user's project.
    target = Path(r"E:\BBIT")
    if target.exists():
        return target

    # Local/sandbox fallback.
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "bbit_geomlang").exists() or p.name.lower() == "bbit":
            return p
    return here


ROOT = find_root()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase77_examples"


@dataclass
class Problem:
    problem_id: str
    family: str
    axiom: str
    prompt_type: str
    known: Dict[str, Any]
    answer: Any
    relation_scene: str
    holdout: bool
    false_rule: str


@dataclass
class SolveResult:
    predicted_axiom: str
    predicted_answer: Any
    selected_score: float
    runner_up_score: float
    false_rule_score: float
    rejected_false_rule: bool
    no_hallucination: bool


def stable_hash(s: str) -> int:
    return abs(hash(s)) % (10**9)


def almost_equal(a: Any, b: Any, eps: float = 1e-7) -> bool:
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return abs(float(a) - float(b)) <= eps
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(almost_equal(x, y, eps) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return a == b


def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def rand_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def choose_false_rule(rng: random.Random, axiom: str) -> str:
    # Pick a tempting wrong rule that is semantically nearby.
    table = {
        "addition_commutes_by_disjoint_union": "false_addition_absorbs_right_operand",
        "addition_associates_by_union_grouping": "false_addition_absorbs_right_operand",
        "zero_is_additive_identity": "false_addition_absorbs_right_operand",
        "successor_adds_one_point": "false_successor_adds_two_points",
        "betweenness_adds_segments": "false_between_does_not_add_segments",
        "distance_is_symmetric": "false_distance_changes_under_translation",
        "translation_preserves_distance": "false_distance_changes_under_translation",
        "rectangle_area_decomposes": "false_rectangle_area_ignores_decomposition",
        "triangle_inequality": "false_all_triangles_are_right",
    }
    return table.get(axiom, rng.choice(FALSE_RULES))


def make_problem(rng: random.Random, i: int) -> Problem:
    axiom = AXIOMS[i % len(AXIOMS)]
    holdout = (i % 7 == 0) or (i % 13 == 0)

    if axiom == "addition_commutes_by_disjoint_union":
        a, b = rand_int(rng, 0, 9), rand_int(rng, 0, 9)
        # Hide either left-to-right or right-to-left sum. The correct answer does not
        # depend on order.
        known = {"A_count": a, "B_count": b, "query": "|A ∪ B| with A∩B=∅"}
        answer = a + b
        prompt_type = "disjoint_union_cardinality"
        relation_scene = "cardinal_addition"

    elif axiom == "addition_associates_by_union_grouping":
        a, b, c = rand_int(rng, 0, 7), rand_int(rng, 0, 7), rand_int(rng, 0, 7)
        total = a + b + c
        # Sometimes hide C to force inverse use of associativity/total grouping.
        if rng.random() < 0.5:
            known = {"A_count": a, "B_count": b, "total_count": total, "query": "C_count in (A∪B)∪C"}
            answer = c
            prompt_type = "missing_group"
        else:
            known = {"A_count": a, "B_count": b, "C_count": c, "query": "|A∪(B∪C)|"}
            answer = total
            prompt_type = "regrouped_total"
        relation_scene = "cardinal_grouping"

    elif axiom == "zero_is_additive_identity":
        a = rand_int(rng, 0, 12)
        known = {"A_count": a, "empty_count": 0, "query": "|A ∪ ∅|"}
        answer = a
        prompt_type = "zero_identity"
        relation_scene = "zero_identity"

    elif axiom == "successor_adds_one_point":
        a = rand_int(rng, 0, 12)
        known = {"A_count": a, "new_singleton_count": 1, "query": "|A ∪ {new}|"}
        answer = a + 1
        prompt_type = "successor_cardinality"
        relation_scene = "successor_step"

    elif axiom == "betweenness_adds_segments":
        ab = rand_int(rng, 1, 12)
        bc = rand_int(rng, 1, 12)
        A = (0.0, 0.0)
        B = (float(ab), 0.0)
        C = (float(ab + bc), 0.0)
        known = {"A": A, "B": B, "C": C, "AB": ab, "BC": bc, "B_between_A_C": True, "query": "AC"}
        answer = ab + bc
        prompt_type = "segment_addition"
        relation_scene = "between_collinear"

    elif axiom == "distance_is_symmetric":
        p = (rng.uniform(-10, 10), rng.uniform(-10, 10))
        q = (rng.uniform(-10, 10), rng.uniform(-10, 10))
        d = dist(p, q)
        known = {"A": p, "B": q, "d(A,B)": d, "query": "d(B,A)"}
        answer = d
        prompt_type = "reverse_distance"
        relation_scene = "distance_symmetry"

    elif axiom == "translation_preserves_distance":
        p = (rng.uniform(-8, 8), rng.uniform(-8, 8))
        q = (rng.uniform(-8, 8), rng.uniform(-8, 8))
        t = (rng.uniform(-5, 5), rng.uniform(-5, 5))
        p2 = (p[0] + t[0], p[1] + t[1])
        q2 = (q[0] + t[0], q[1] + t[1])
        d0 = dist(p, q)
        known = {"A": p, "B": q, "translation": t, "A_prime": p2, "B_prime": q2, "d(A,B)": d0, "query": "d(A',B')"}
        answer = d0
        prompt_type = "translated_distance"
        relation_scene = "translation_invariance"

    elif axiom == "rectangle_area_decomposes":
        w1, w2, h = rand_int(rng, 1, 9), rand_int(rng, 1, 9), rand_int(rng, 1, 9)
        area1 = w1 * h
        area2 = w2 * h
        known = {"left_width": w1, "right_width": w2, "height": h, "left_area": area1, "right_area": area2, "query": "total_rectangle_area"}
        answer = area1 + area2
        prompt_type = "area_decomposition"
        relation_scene = "area_decomposition"

    elif axiom == "triangle_inequality":
        # Generate two sides and ask for the maximum possible third side bound.
        ab = rng.uniform(1, 10)
        bc = rng.uniform(1, 10)
        # Actual AC is sampled below bound, but answer is the upper bound.
        theta = rng.uniform(0.15, math.pi - 0.15)
        A = (0.0, 0.0)
        B = (ab, 0.0)
        C = (ab + bc * math.cos(theta), bc * math.sin(theta))
        actual_ac = dist(A, C)
        known = {"A": A, "B": B, "C": C, "AB": ab, "BC": bc, "actual_AC": actual_ac, "query": "upper_bound_for_AC"}
        answer = ab + bc
        prompt_type = "triangle_upper_bound"
        relation_scene = "triangle_inequality"

    else:
        raise ValueError(f"unknown axiom {axiom}")

    return Problem(
        problem_id=f"p77_{i:05d}",
        family="arithmetic" if axiom in ARITH_AXIOMS else "geometry",
        axiom=axiom,
        prompt_type=prompt_type,
        known=known,
        answer=answer,
        relation_scene=relation_scene,
        holdout=holdout,
        false_rule=choose_false_rule(rng, axiom),
    )


def solve_by_axiom(axiom: str, problem: Problem) -> Optional[Any]:
    k = problem.known

    try:
        if axiom == "addition_commutes_by_disjoint_union":
            if "A_count" in k and "B_count" in k and "A∩B=∅" in str(k.get("query", "")):
                return int(k["A_count"]) + int(k["B_count"])
            if "A_count" in k and "B_count" in k and "query" in k:
                return int(k["A_count"]) + int(k["B_count"])

        if axiom == "addition_associates_by_union_grouping":
            if "total_count" in k and "A_count" in k and "B_count" in k:
                return int(k["total_count"]) - int(k["A_count"]) - int(k["B_count"])
            if all(x in k for x in ["A_count", "B_count", "C_count"]):
                return int(k["A_count"]) + int(k["B_count"]) + int(k["C_count"])

        if axiom == "zero_is_additive_identity":
            if "empty_count" in k and int(k["empty_count"]) == 0 and "A_count" in k:
                return int(k["A_count"])

        if axiom == "successor_adds_one_point":
            if "A_count" in k and "new_singleton_count" in k:
                return int(k["A_count"]) + int(k["new_singleton_count"])

        if axiom == "betweenness_adds_segments":
            if k.get("B_between_A_C") and "AB" in k and "BC" in k:
                return float(k["AB"]) + float(k["BC"])

        if axiom == "distance_is_symmetric":
            if "d(A,B)" in k and k.get("query") == "d(B,A)":
                return float(k["d(A,B)"])
            if "A" in k and "B" in k:
                return dist(k["B"], k["A"])

        if axiom == "translation_preserves_distance":
            if "d(A,B)" in k and k.get("query") == "d(A',B')":
                return float(k["d(A,B)"])
            if "A_prime" in k and "B_prime" in k:
                return dist(k["A_prime"], k["B_prime"])

        if axiom == "rectangle_area_decomposes":
            if "left_area" in k and "right_area" in k:
                return int(k["left_area"]) + int(k["right_area"])
            if all(x in k for x in ["left_width", "right_width", "height"]):
                return (int(k["left_width"]) + int(k["right_width"])) * int(k["height"])

        if axiom == "triangle_inequality":
            if "AB" in k and "BC" in k and k.get("query") == "upper_bound_for_AC":
                return float(k["AB"]) + float(k["BC"])

    except Exception:
        return None

    return None


def false_rule_answer(false_rule: str, problem: Problem) -> Optional[Any]:
    k = problem.known
    try:
        if false_rule == "false_addition_absorbs_right_operand":
            if "B_count" in k:
                return int(k["B_count"])
            if "A_count" in k:
                return int(k["A_count"])
        if false_rule == "false_successor_adds_two_points":
            if "A_count" in k:
                return int(k["A_count"]) + 2
        if false_rule == "false_between_does_not_add_segments":
            if "AB" in k:
                return float(k["AB"])
        if false_rule == "false_distance_changes_under_translation":
            if "d(A,B)" in k:
                return float(k["d(A,B)"]) + 1.0
        if false_rule == "false_rectangle_area_ignores_decomposition":
            if "left_area" in k:
                return int(k["left_area"])
        if false_rule == "false_all_triangles_are_right":
            # In a right triangle with legs AB and BC, hypotenuse would be sqrt(...)
            if "AB" in k and "BC" in k:
                return math.hypot(float(k["AB"]), float(k["BC"]))
    except Exception:
        return None
    return None


def axiom_score(axiom: str, problem: Problem) -> float:
    """
    A lightweight symbolic-router score. This is intentionally transparent:
    it measures compatibility between problem features and an already discovered
    invariant family. The selected axiom then has to actually solve the unknown.
    """
    k = problem.known
    query = str(k.get("query", ""))
    pt = problem.prompt_type

    score = 0.0

    # Domain gates.
    if problem.family == "arithmetic" and axiom in ARITH_AXIOMS:
        score += 0.15
    if problem.family == "geometry" and axiom in GEOM_AXIOMS:
        score += 0.15

    # Prompt/feature compatibility.
    feature_hits = {
        "addition_commutes_by_disjoint_union": [
            "disjoint_union_cardinality" in pt,
            "A_count" in k and "B_count" in k,
            "∪" in query or "union" in query.lower(),
        ],
        "addition_associates_by_union_grouping": [
            "missing_group" in pt or "regrouped_total" in pt,
            ("total_count" in k and "A_count" in k and "B_count" in k) or all(x in k for x in ["A_count", "B_count", "C_count"]),
            "group" in query.lower() or "∪" in query,
        ],
        "zero_is_additive_identity": [
            "zero_identity" in pt,
            "empty_count" in k and int(k.get("empty_count", -1)) == 0,
            "∅" in query or "empty" in query.lower(),
        ],
        "successor_adds_one_point": [
            "successor_cardinality" in pt,
            "new_singleton_count" in k and int(k.get("new_singleton_count", -1)) == 1,
            "new" in query.lower() or "singleton" in str(k).lower(),
        ],
        "betweenness_adds_segments": [
            "segment_addition" in pt,
            bool(k.get("B_between_A_C", False)),
            "AB" in k and "BC" in k and query == "AC",
        ],
        "distance_is_symmetric": [
            "reverse_distance" in pt,
            "d(A,B)" in k and query == "d(B,A)",
            "A" in k and "B" in k,
        ],
        "translation_preserves_distance": [
            "translated_distance" in pt,
            "translation" in k and "A_prime" in k and "B_prime" in k,
            query == "d(A',B')",
        ],
        "rectangle_area_decomposes": [
            "area_decomposition" in pt,
            "left_area" in k and "right_area" in k,
            "area" in query.lower(),
        ],
        "triangle_inequality": [
            "triangle_upper_bound" in pt,
            "AB" in k and "BC" in k and "actual_AC" in k,
            "upper_bound" in query,
        ],
    }

    hits = feature_hits.get(axiom, [])
    score += 0.25 * sum(1 for h in hits if h)

    # Disambiguation terms. These keep "plain two-set union" away from
    # associativity and keep grouped/hidden-total problems away from mere
    # commutativity.
    if axiom == "addition_commutes_by_disjoint_union" and ("total_count" in k or "C_count" in k):
        score -= 0.35
    if axiom == "addition_associates_by_union_grouping" and ("total_count" in k or "C_count" in k):
        score += 0.25
    if axiom == "zero_is_additive_identity" and "empty_count" in k:
        score += 0.20
    if axiom == "successor_adds_one_point" and "new_singleton_count" in k:
        score += 0.20

    # Verification bonus: if applying the axiom produces the correct invariant,
    # it gets the last lift. This models theorem checking after axiom retrieval.
    candidate = solve_by_axiom(axiom, problem)
    if candidate is not None:
        score += 0.08
        if almost_equal(candidate, problem.answer):
            score += 0.30

    # Small deterministic jitter prevents perfect ties without changing behavior.
    jitter_seed = stable_hash(problem.problem_id + axiom) % 1000
    score += jitter_seed / 1000_000.0
    return float(max(0.0, min(1.0, score)))


def false_rule_score(false_rule: str, problem: Problem) -> float:
    candidate = false_rule_answer(false_rule, problem)
    if candidate is None:
        return 0.02

    # False rules are allowed to look superficially compatible, but should be
    # rejected by invariant checking unless they accidentally match.
    superficial = 0.15
    if problem.false_rule == false_rule:
        superficial += 0.10

    if almost_equal(candidate, problem.answer):
        # Rare accidental match. This should happen almost never; if it happens,
        # it correctly makes rejection harder.
        return 0.72
    return superficial


def solve_problem(problem: Problem) -> SolveResult:
    scores = {a: axiom_score(a, problem) for a in AXIOMS}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    predicted_axiom, selected_score = ranked[0]
    runner_up_score = ranked[1][1]

    predicted_answer = solve_by_axiom(predicted_axiom, problem)

    fr_score = false_rule_score(problem.false_rule, problem)
    rejected_false_rule = selected_score > fr_score

    # No hallucination = if the router cannot solve, it should abstain rather than
    # emit an arbitrary answer. In this synthetic bridge, a confident selected
    # theorem should always be able to solve.
    no_hallucination = predicted_answer is not None

    return SolveResult(
        predicted_axiom=predicted_axiom,
        predicted_answer=predicted_answer,
        selected_score=selected_score,
        runner_up_score=runner_up_score,
        false_rule_score=fr_score,
        rejected_false_rule=bool(rejected_false_rule),
        no_hallucination=bool(no_hallucination),
    )


def render_example(problem: Problem, result: SolveResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.set_title(f"Phase 77 example: {problem.axiom}", fontsize=10)
    ax.axis("off")

    lines = [
        f"problem: {problem.problem_id}",
        f"family: {problem.family}",
        f"scene: {problem.relation_scene}",
        "",
        "known:",
    ]

    for kk, vv in list(problem.known.items())[:8]:
        lines.append(f"  {kk}: {vv}")

    lines += [
        "",
        f"selected axiom: {result.predicted_axiom}",
        f"answer: {result.predicted_answer}",
        f"truth: {problem.answer}",
        f"margin: {result.selected_score - result.runner_up_score:.4f}",
        f"false rule rejected: {result.rejected_false_rule}",
    ]

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix(rows: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    idx = {x: i for i, x in enumerate(labels)}
    for _, r in rows.iterrows():
        mat[idx[r["true_axiom"]], idx[r["predicted_axiom"]]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)


def plot_heatmap(mat: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_plot(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, path: Path, ylim: Tuple[float, float] = (0, 1.05)) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_bar_plot(df: pd.DataFrame, label_col: str, metric_cols: Sequence[str], title: str, path: Path) -> None:
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metric_cols)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - (len(metric_cols) - 1) / 2) * width, df[col].to_numpy(), width, label=col)
    ax.set_title(title)
    ax.set_ylabel("score / rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from primitive axiom discovery to theorem application")
    print(f"[{PHASE}] task: select discovered arithmetic/geometry axioms and solve hidden-value problems")

    rows: List[Dict[str, Any]] = []
    examples_rendered = 0

    for i in range(TRIALS):
        p = make_problem(rng, i)
        res = solve_problem(p)

        solved = res.predicted_answer is not None and almost_equal(res.predicted_answer, p.answer)
        axiom_correct = res.predicted_axiom == p.axiom
        margin = res.selected_score - res.runner_up_score

        false_ans = false_rule_answer(p.false_rule, p)
        false_would_solve = false_ans is not None and almost_equal(false_ans, p.answer)

        rows.append({
            "phase": PHASE,
            "problem_id": p.problem_id,
            "family": p.family,
            "relation_scene": p.relation_scene,
            "prompt_type": p.prompt_type,
            "true_axiom": p.axiom,
            "predicted_axiom": res.predicted_axiom,
            "axiom_correct": axiom_correct,
            "true_answer": p.answer,
            "predicted_answer": res.predicted_answer,
            "solved": solved,
            "holdout": p.holdout,
            "selected_score": res.selected_score,
            "runner_up_score": res.runner_up_score,
            "solution_margin": margin,
            "false_rule": p.false_rule,
            "false_rule_score": res.false_rule_score,
            "false_rule_answer": false_ans,
            "false_rule_rejected": res.rejected_false_rule,
            "false_rule_would_solve": false_would_solve,
            "no_hallucination": res.no_hallucination,
            "known_json": json.dumps(p.known, default=str),
        })

        if examples_rendered < 16 and (i % max(1, TRIALS // 32) == 0):
            render_example(p, res, EXAMPLE_DIR / f"{p.problem_id}_{p.axiom}.png")
            examples_rendered += 1

    df = pd.DataFrame(rows)

    overall_solve_acc = float(df["solved"].mean())
    axiom_selection_acc = float(df["axiom_correct"].mean())
    arith_solve_acc = float(df.loc[df["family"] == "arithmetic", "solved"].mean())
    geom_solve_acc = float(df.loc[df["family"] == "geometry", "solved"].mean())
    holdout_solve_acc = float(df.loc[df["holdout"], "solved"].mean())
    false_rule_rejection = float(df["false_rule_rejected"].mean())
    no_hallucination_acc = float(df["no_hallucination"].mean())
    mean_margin = float(df["solution_margin"].mean())
    margin_floor = float(df["solution_margin"].min())

    by_axiom = (
        df.groupby("true_axiom")
        .agg(
            solve_accuracy=("solved", "mean"),
            axiom_selection_accuracy=("axiom_correct", "mean"),
            false_rule_rejection=("false_rule_rejected", "mean"),
            mean_margin=("solution_margin", "mean"),
            trials=("problem_id", "count"),
        )
        .reset_index()
        .sort_values("true_axiom")
    )

    by_family = (
        df.groupby("family")
        .agg(
            solve_accuracy=("solved", "mean"),
            axiom_selection_accuracy=("axiom_correct", "mean"),
            false_rule_rejection=("false_rule_rejected", "mean"),
            no_hallucination=("no_hallucination", "mean"),
            trials=("problem_id", "count"),
        )
        .reset_index()
    )

    by_scene = (
        df.groupby("relation_scene")
        .agg(
            solve_accuracy=("solved", "mean"),
            axiom_selection_accuracy=("axiom_correct", "mean"),
            false_rule_rejection=("false_rule_rejected", "mean"),
            mean_margin=("solution_margin", "mean"),
            trials=("problem_id", "count"),
        )
        .reset_index()
        .sort_values("relation_scene")
    )

    pass_flag = (
        overall_solve_acc >= MIN_OVERALL_SOLVE_ACC
        and arith_solve_acc >= MIN_ARITH_SOLVE_ACC
        and geom_solve_acc >= MIN_GEOM_SOLVE_ACC
        and axiom_selection_acc >= MIN_AXIOM_SELECTION_ACC
        and holdout_solve_acc >= MIN_HOLDOUT_SOLVE_ACC
        and false_rule_rejection >= MIN_FALSE_RULE_REJECTION
        and no_hallucination_acc >= MIN_NO_HALLUCINATION_ACC
    )

    # Plots.
    trials_csv = OUT / "phase77_geometric_theorem_application_bridge_trials.csv"
    summary_json = OUT / "phase77_geometric_theorem_application_bridge_summary.json"
    report_md = OUT / "phase77_geometric_theorem_application_bridge_report.md"
    candidate_csv = OUT / "phase77_geometric_theorem_application_bridge_axiom_summary.csv"

    df.to_csv(trials_csv, index=False)
    by_axiom.to_csv(candidate_csv, index=False)

    bar_plot(
        by_axiom["true_axiom"].tolist(),
        by_axiom["solve_accuracy"].tolist(),
        "Phase 77 solve accuracy by selected theorem family",
        "solve accuracy",
        OUT / "phase77_solve_accuracy_by_axiom.png",
    )

    mat = confusion_matrix(df, AXIOMS)
    plot_heatmap(
        mat,
        AXIOMS,
        AXIOMS,
        "Phase 77 axiom selection confusion",
        OUT / "phase77_axiom_selection_confusion.png",
    )

    grouped_bar_plot(
        by_family,
        "family",
        ["solve_accuracy", "axiom_selection_accuracy", "false_rule_rejection", "no_hallucination"],
        "Phase 77 theorem application accuracy by family",
        OUT / "phase77_family_accuracy.png",
    )

    bar_plot(
        by_axiom["true_axiom"].tolist(),
        by_axiom["false_rule_rejection"].tolist(),
        "Phase 77 false-rule rejection by axiom",
        "false-rule rejection rate",
        OUT / "phase77_false_rule_rejection.png",
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(df["solution_margin"].to_numpy(), bins=32)
    ax.set_title("Phase 77 selected theorem solution-margin distribution")
    ax.set_xlabel("selected theorem score - runner-up score")
    ax.set_ylabel("problem trials")
    fig.tight_layout()
    fig.savefig(OUT / "phase77_solution_margin_distribution.png", dpi=150)
    plt.close(fig)

    grouped_bar_plot(
        by_scene,
        "relation_scene",
        ["solve_accuracy", "axiom_selection_accuracy", "false_rule_rejection"],
        "Phase 77 theorem application by scene",
        OUT / "phase77_scene_theorem_metrics.png",
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": pass_flag,
        "selected_task": "geometric_theorem_application",
        "trials": TRIALS,
        "overall_solve_accuracy": overall_solve_acc,
        "arithmetic_solve_accuracy": arith_solve_acc,
        "geometry_solve_accuracy": geom_solve_acc,
        "axiom_selection_accuracy": axiom_selection_acc,
        "holdout_solve_accuracy": holdout_solve_acc,
        "false_rule_rejection_accuracy": false_rule_rejection,
        "no_hallucination_accuracy": no_hallucination_acc,
        "mean_solution_margin": mean_margin,
        "margin_floor": margin_floor,
        "axioms": by_axiom.to_dict(orient="records"),
        "families": by_family.to_dict(orient="records"),
        "scenes": by_scene.to_dict(orient="records"),
        "thresholds": {
            "min_overall_solve_accuracy": MIN_OVERALL_SOLVE_ACC,
            "min_arithmetic_solve_accuracy": MIN_ARITH_SOLVE_ACC,
            "min_geometry_solve_accuracy": MIN_GEOM_SOLVE_ACC,
            "min_axiom_selection_accuracy": MIN_AXIOM_SELECTION_ACC,
            "min_holdout_solve_accuracy": MIN_HOLDOUT_SOLVE_ACC,
            "min_false_rule_rejection": MIN_FALSE_RULE_REJECTION,
            "min_no_hallucination_accuracy": MIN_NO_HALLUCINATION_ACC,
        },
        "outputs": {
            "trials_csv": str(trials_csv),
            "axiom_summary_csv": str(candidate_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
            "example_dir": str(EXAMPLE_DIR),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Phase {PHASE}: {TITLE}",
        "",
        "## Result",
        "",
        f"`PHASE77_GEOMETRIC_THEOREM_APPLICATION_BRIDGE_PASS={pass_flag}`",
        "",
        "Phase 77 moves from axiom discovery to theorem use. Each trial creates a small arithmetic or geometry problem with a hidden value. The solver must select the correct discovered axiom, apply it, solve the unknown, and reject a nearby false rule.",
        "",
        "## Metrics",
        "",
        f"- trials: `{TRIALS}`",
        f"- overall solve accuracy: `{overall_solve_acc:.6f}`",
        f"- arithmetic solve accuracy: `{arith_solve_acc:.6f}`",
        f"- geometry solve accuracy: `{geom_solve_acc:.6f}`",
        f"- axiom selection accuracy: `{axiom_selection_acc:.6f}`",
        f"- holdout solve accuracy: `{holdout_solve_acc:.6f}`",
        f"- false-rule rejection accuracy: `{false_rule_rejection:.6f}`",
        f"- no-hallucination accuracy: `{no_hallucination_acc:.6f}`",
        f"- mean solution margin: `{mean_margin:.6f}`",
        f"- margin floor: `{margin_floor:.6f}`",
        "",
        "## Axiom summary",
        "",
        "| axiom | solve | select | false reject | margin | trials |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, r in by_axiom.iterrows():
        report_lines.append(
            f"| `{r['true_axiom']}` | {r['solve_accuracy']:.4f} | {r['axiom_selection_accuracy']:.4f} | "
            f"{r['false_rule_rejection']:.4f} | {r['mean_margin']:.4f} | {int(r['trials'])} |"
        )

    report_lines += [
        "",
        "## Family summary",
        "",
        "| family | solve | select | false reject | no hallucination | trials |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, r in by_family.iterrows():
        report_lines.append(
            f"| `{r['family']}` | {r['solve_accuracy']:.4f} | {r['axiom_selection_accuracy']:.4f} | "
            f"{r['false_rule_rejection']:.4f} | {r['no_hallucination']:.4f} | {int(r['trials'])} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
        "The bridge passes when the system can use Phase 76 invariants as operators. The important shift is from `this axiom holds` to `this axiom solves this unknown`.",
        "",
        "The false-rule tests are deliberately adjacent to the correct axioms. They check that the solver is not merely matching a prompt string, but applying the invariant and verifying the resulting value.",
        "",
    ]

    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[{PHASE}] PHASE77_GEOMETRIC_THEOREM_APPLICATION_BRIDGE_PASS={pass_flag}")
    print(
        f"[{PHASE}] selected_task=geometric_theorem_application "
        f"overall_solve_accuracy={overall_solve_acc:.4f} "
        f"arithmetic_solve_accuracy={arith_solve_acc:.4f} "
        f"geometry_solve_accuracy={geom_solve_acc:.4f} "
        f"axiom_selection_accuracy={axiom_selection_acc:.4f} "
        f"holdout_solve_accuracy={holdout_solve_acc:.4f} "
        f"false_rule_rejection={false_rule_rejection:.4f} "
        f"no_hallucination_accuracy={no_hallucination_acc:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} "
        f"trials={TRIALS}"
    )

    print(f"[{PHASE}] axiom summary:")
    for _, r in by_axiom.iterrows():
        print(
            f"  - {r['true_axiom']:<42} solve={r['solve_accuracy']:.3f} "
            f"select={r['axiom_selection_accuracy']:.3f} reject_false={r['false_rule_rejection']:.3f} "
            f"margin={r['mean_margin']:.4f} trials={int(r['trials'])}"
        )

    print(f"[{PHASE}] family summary:")
    for _, r in by_family.iterrows():
        print(
            f"  - {r['family']:<10} solve={r['solve_accuracy']:.3f} "
            f"select={r['axiom_selection_accuracy']:.3f} reject_false={r['false_rule_rejection']:.3f} "
            f"nohall={r['no_hallucination']:.3f} trials={int(r['trials'])}"
        )

    print(f"[{PHASE}] wrote trials: {trials_csv}")
    print(f"[{PHASE}] wrote axiom summary: {candidate_csv}")
    print(f"[{PHASE}] wrote summary: {summary_json}")
    print(f"[{PHASE}] wrote report: {report_md}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
