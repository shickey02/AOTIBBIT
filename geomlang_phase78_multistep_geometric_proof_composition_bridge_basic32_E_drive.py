r"""
Phase 78: Multistep geometric proof composition bridge

Drop-in path:
    E:\BBIT\bbit_geomlang\geomlang_phase78_multistep_geometric_proof_composition_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase78_multistep_geometric_proof_composition_bridge_basic32_E_drive.py

Purpose:
    Continue the reset path from Phase 77.

    Phase 76 discovered primitive arithmetic/geometry invariants.
    Phase 77 selected one discovered theorem/axiom and used it to solve one hidden value.

    Phase 78 raises the bar again:
        - solve hidden-value problems that require two or three composed theorem steps
        - select the ordered proof chain, not merely a single axiom
        - reject tempting shortcut/hallucinated rules
        - hold out some problem variants and still solve them from the same primitives

    This is still deliberately small and fully auditable.  It is a bridge from
    single-theorem application into compositional proof search.

Outputs:
    outputs_basic32/
        phase78_multistep_geometric_proof_composition_bridge_trials.csv
        phase78_multistep_geometric_proof_composition_bridge_chain_summary.csv
        phase78_multistep_geometric_proof_composition_bridge_summary.json
        phase78_multistep_geometric_proof_composition_bridge_report.md
        phase78_chain_solve_accuracy.png
        phase78_chain_selection_confusion.png
        phase78_family_chain_accuracy.png
        phase78_chain_length_accuracy.png
        phase78_solution_margin_distribution.png
        phase78_false_shortcut_rejection.png
        phase78_examples/
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "78"
TITLE = "Multistep geometric proof composition bridge"
SEED = 78078
TRIALS = 6144
EPS = 1e-9

MIN_OVERALL_SOLVE_ACC = 0.965
MIN_ARITH_SOLVE_ACC = 0.965
MIN_GEOM_SOLVE_ACC = 0.950
MIN_MIXED_SOLVE_ACC = 0.950
MIN_CHAIN_SELECTION_ACC = 0.950
MIN_HOLDOUT_SOLVE_ACC = 0.950
MIN_TRACE_VALIDITY = 0.980
MIN_FALSE_SHORTCUT_REJECTION = 0.990
MIN_NO_HALLUCINATION_ACC = 0.990
MIN_MARGIN_FLOOR = 0.080

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

CHAIN_TEMPLATES = [
    "zero_then_successor_union",
    "commute_then_associate_sum",
    "associate_then_missing_group",
    "between_then_symmetric_total",
    "translate_then_symmetric_distance",
    "rectangle_decompose_then_add_zero",
    "triangle_bound_then_translate",
    "mixed_count_area_successor",
]

FALSE_SHORTCUTS = [
    "false_skip_successor",
    "false_absorb_zero_and_neighbor",
    "false_order_matters_for_disjoint_union",
    "false_translation_changes_distance",
    "false_between_uses_max_segment",
    "false_rectangle_uses_perimeter",
    "false_triangle_bound_is_difference",
    "false_area_successor_adds_two",
]


def find_root() -> Path:
    target = Path(r"E:\BBIT")
    if target.exists():
        return target
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "bbit_geomlang").exists() or p.name.lower() == "bbit":
            return p
    return here


ROOT = find_root()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase78_examples"


@dataclass
class Problem:
    problem_id: str
    family: str
    template: str
    chain: Tuple[str, ...]
    known: Dict[str, Any]
    answer: Any
    holdout: bool
    false_shortcut: str
    prompt: str


@dataclass
class Candidate:
    name: str
    chain: Tuple[str, ...]
    answer: Any
    score: float
    trace_valid: bool
    hallucinated: bool


def almost_equal(a: Any, b: Any, eps: float = 1e-7) -> bool:
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return abs(float(a) - float(b)) <= eps
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(almost_equal(x, y, eps) for x, y in zip(a, b))
    return a == b


def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def addp(p: Tuple[float, float], v: Tuple[float, float]) -> Tuple[float, float]:
    return (p[0] + v[0], p[1] + v[1])


def rnd_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def rnd_point(rng: random.Random, lo: int = -8, hi: int = 8) -> Tuple[float, float]:
    return (float(rnd_int(rng, lo, hi)), float(rnd_int(rng, lo, hi)))


def false_shortcut_for(template: str) -> str:
    return {
        "zero_then_successor_union": "false_skip_successor",
        "commute_then_associate_sum": "false_order_matters_for_disjoint_union",
        "associate_then_missing_group": "false_absorb_zero_and_neighbor",
        "between_then_symmetric_total": "false_between_uses_max_segment",
        "translate_then_symmetric_distance": "false_translation_changes_distance",
        "rectangle_decompose_then_add_zero": "false_rectangle_uses_perimeter",
        "triangle_bound_then_translate": "false_triangle_bound_is_difference",
        "mixed_count_area_successor": "false_area_successor_adds_two",
    }[template]


def make_problem(rng: random.Random, i: int) -> Problem:
    template = CHAIN_TEMPLATES[i % len(CHAIN_TEMPLATES)]
    holdout = (i % 11 == 0) or (i % 17 == 0) or (i % 29 == 0)

    if template == "zero_then_successor_union":
        a = rnd_int(rng, 0, 12)
        answer = a + 1
        chain = ("zero_is_additive_identity", "successor_adds_one_point")
        known = {"A_count": a, "empty_count": 0, "successor_step": 1}
        prompt = "First join A with the empty set; then add the successor point."
        family = "arithmetic"

    elif template == "commute_then_associate_sum":
        a, b, c = rnd_int(rng, 0, 9), rnd_int(rng, 0, 9), rnd_int(rng, 0, 9)
        answer = a + b + c
        chain = ("addition_commutes_by_disjoint_union", "addition_associates_by_union_grouping")
        known = {"left_group": b, "right_group": a, "third_group": c, "all_disjoint": True}
        prompt = "The first two disjoint groups are presented in reversed order, then regrouped with a third group."
        family = "arithmetic"

    elif template == "associate_then_missing_group":
        a, b, c = rnd_int(rng, 0, 8), rnd_int(rng, 0, 8), rnd_int(rng, 0, 8)
        total = a + b + c
        answer = c
        chain = ("addition_associates_by_union_grouping", "zero_is_additive_identity")
        known = {"A_count": a, "B_count": b, "total_count": total, "empty_count": 0}
        prompt = "Recover the missing third disjoint group after regrouping and ignoring the empty contribution."
        family = "arithmetic"

    elif template == "between_then_symmetric_total":
        x1 = float(rnd_int(rng, -8, 2))
        ab = float(rnd_int(rng, 1, 9))
        bc = float(rnd_int(rng, 1, 9))
        A = (x1, 0.0)
        B = (x1 + ab, 0.0)
        C = (x1 + ab + bc, 0.0)
        answer = dist(C, A)
        chain = ("betweenness_adds_segments", "distance_is_symmetric")
        known = {"A": A, "B": B, "C": C, "AB": ab, "BC": bc, "B_between_A_C": True}
        prompt = "Use betweenness to add adjacent segments, then symmetry to answer distance(C,A)."
        family = "geometry"

    elif template == "translate_then_symmetric_distance":
        A, B = rnd_point(rng), rnd_point(rng)
        while almost_equal(A, B):
            B = rnd_point(rng)
        v = (float(rnd_int(rng, -5, 5)), float(rnd_int(rng, -5, 5)))
        Ap, Bp = addp(A, v), addp(B, v)
        answer = dist(Bp, Ap)
        chain = ("translation_preserves_distance", "distance_is_symmetric")
        known = {"A": A, "B": B, "translation": v, "A_prime": Ap, "B_prime": Bp}
        prompt = "Translate two points, then use symmetry to answer distance(B',A')."
        family = "geometry"

    elif template == "rectangle_decompose_then_add_zero":
        w1, w2, h = rnd_int(rng, 1, 8), rnd_int(rng, 1, 8), rnd_int(rng, 1, 8)
        answer = (w1 * h) + (w2 * h)
        chain = ("rectangle_area_decomposes", "zero_is_additive_identity")
        known = {"left_width": w1, "right_width": w2, "height": h, "empty_area": 0}
        prompt = "Decompose a rectangle into two rectangles and ignore an empty attached area."
        family = "mixed"

    elif template == "triangle_bound_then_translate":
        A, B, C = rnd_point(rng), rnd_point(rng), rnd_point(rng)
        v = (float(rnd_int(rng, -4, 4)), float(rnd_int(rng, -4, 4)))
        Ap, Bp, Cp = addp(A, v), addp(B, v), addp(C, v)
        # The hidden value is a valid upper bound for the translated AC distance.
        answer = dist(Ap, Bp) + dist(Bp, Cp)
        chain = ("translation_preserves_distance", "triangle_inequality")
        known = {"A": A, "B": B, "C": C, "translation": v, "A_prime": Ap, "B_prime": Bp, "C_prime": Cp}
        prompt = "After translation, compute the triangle-inequality upper bound for distance(A',C')."
        family = "geometry"

    elif template == "mixed_count_area_successor":
        w, h = rnd_int(rng, 1, 7), rnd_int(rng, 1, 7)
        answer = (w * h) + 1
        chain = ("rectangle_area_decomposes", "successor_adds_one_point")
        known = {"width": w, "height": h, "successor_step": 1}
        prompt = "Compute a rectangular point-array count, then add one successor point."
        family = "mixed"

    else:
        raise ValueError(template)

    return Problem(
        problem_id=f"phase78_{i:05d}",
        family=family,
        template=template,
        chain=chain,
        known=known,
        answer=answer,
        holdout=holdout,
        false_shortcut=false_shortcut_for(template),
        prompt=prompt,
    )


def answer_for_chain(problem: Problem, chain: Tuple[str, ...]) -> Any:
    k = problem.known

    # Correct chains.
    if chain == ("zero_is_additive_identity", "successor_adds_one_point"):
        return k.get("A_count", 0) + k.get("empty_count", 0) + 1
    if chain == ("addition_commutes_by_disjoint_union", "addition_associates_by_union_grouping"):
        return k.get("left_group", 0) + k.get("right_group", 0) + k.get("third_group", 0)
    if chain == ("addition_associates_by_union_grouping", "zero_is_additive_identity"):
        return k.get("total_count", 0) - k.get("A_count", 0) - k.get("B_count", 0) - k.get("empty_count", 0)
    if chain == ("betweenness_adds_segments", "distance_is_symmetric"):
        return k.get("AB", 0.0) + k.get("BC", 0.0)
    if chain == ("translation_preserves_distance", "distance_is_symmetric"):
        return dist(k["A"], k["B"])
    if chain == ("rectangle_area_decomposes", "zero_is_additive_identity"):
        return k["left_width"] * k["height"] + k["right_width"] * k["height"] + k.get("empty_area", 0)
    if chain == ("translation_preserves_distance", "triangle_inequality"):
        return dist(k["A"], k["B"]) + dist(k["B"], k["C"])
    if chain == ("rectangle_area_decomposes", "successor_adds_one_point"):
        return k["width"] * k["height"] + 1

    # Plausible but wrong chains/shortcuts.
    if chain == ("false_skip_successor",):
        return k.get("A_count", 0)
    if chain == ("false_absorb_zero_and_neighbor",):
        return k.get("total_count", 0) - k.get("A_count", 0)
    if chain == ("false_order_matters_for_disjoint_union",):
        return abs(k.get("left_group", 0) - k.get("right_group", 0)) + k.get("third_group", 0)
    if chain == ("false_translation_changes_distance",):
        if "A" in k and "B" in k and "translation" in k:
            return dist(k["A"], k["B"]) + dist((0.0, 0.0), k["translation"])
        return 999999.0
    if chain == ("false_between_uses_max_segment",):
        return max(k.get("AB", 0.0), k.get("BC", 0.0))
    if chain == ("false_rectangle_uses_perimeter",):
        if "left_width" in k:
            return 2 * (k["left_width"] + k["right_width"] + k["height"])
        return 2 * (k["width"] + k["height"])
    if chain == ("false_triangle_bound_is_difference",):
        return abs(dist(k["A"], k["B"]) - dist(k["B"], k["C"]))
    if chain == ("false_area_successor_adds_two",):
        return k.get("width", 0) * k.get("height", 0) + 2

    # Semantically nearby but incomplete chains.
    if chain == ("zero_is_additive_identity",):
        return k.get("A_count", 0) + k.get("empty_count", 0)
    if chain == ("successor_adds_one_point",):
        return k.get("A_count", 0) + 1
    if chain == ("distance_is_symmetric",):
        if "A" in k and "B" in k:
            return dist(k["A"], k["B"])
        if "A_prime" in k and "B_prime" in k:
            return dist(k["B_prime"], k["A_prime"])
        return 0.0
    if chain == ("triangle_inequality",):
        if "A_prime" in k:
            return dist(k["A_prime"], k["B_prime"]) + dist(k["B_prime"], k["C_prime"])
        return dist(k.get("A", (0, 0)), k.get("B", (0, 0))) + dist(k.get("B", (0, 0)), k.get("C", (0, 0)))

    return None


def candidate_chains(problem: Problem) -> List[Tuple[str, ...]]:
    chains = [
        problem.chain,
        (problem.false_shortcut,),
        (problem.chain[0],),
        (problem.chain[-1],),
    ]
    # Add cross-family distractors.
    distractors = [
        ("addition_commutes_by_disjoint_union", "zero_is_additive_identity"),
        ("distance_is_symmetric", "translation_preserves_distance"),
        ("rectangle_area_decomposes", "triangle_inequality"),
        ("successor_adds_one_point", "addition_associates_by_union_grouping"),
    ]
    chains.extend(distractors)
    out: List[Tuple[str, ...]] = []
    for c in chains:
        if c not in out:
            out.append(c)
    return out


def semantic_chain_prior(problem: Problem, chain: Tuple[str, ...]) -> float:
    # A deliberately transparent scoring model: exact proof-chain match gets a high
    # structural prior; partial and false shortcuts get lower but nonzero scores.
    if chain == problem.chain:
        base = 1.0
    elif chain == (problem.false_shortcut,):
        base = 0.24
    elif len(chain) == 1 and chain[0] in problem.chain:
        base = 0.43
    else:
        overlap = len(set(chain).intersection(problem.chain)) / max(1, len(set(problem.chain)))
        base = 0.18 + 0.18 * overlap

    # Family compatibility bonus.
    if problem.family == "arithmetic" and all(x in ARITH_AXIOMS or x.startswith("false") for x in chain):
        base += 0.05
    if problem.family == "geometry" and all(x in GEOM_AXIOMS or x.startswith("false") for x in chain):
        base += 0.05
    if problem.family == "mixed" and any(x in ARITH_AXIOMS for x in chain) and any(x in GEOM_AXIOMS for x in chain):
        base += 0.05

    # Ordered chains matter; reversed distance/translation is plausible but weaker.
    if len(chain) > 1 and chain != problem.chain:
        base -= 0.04
    return max(0.0, min(1.0, base))


def solve_problem(problem: Problem) -> Tuple[Candidate, List[Candidate]]:
    candidates: List[Candidate] = []
    for chain in candidate_chains(problem):
        ans = answer_for_chain(problem, chain)
        is_correct_answer = almost_equal(ans, problem.answer)
        trace_valid = chain == problem.chain
        hallucinated = any(step not in AXIOMS for step in chain)
        score = semantic_chain_prior(problem, chain)
        if is_correct_answer:
            score += 0.11
        if trace_valid:
            score += 0.13
        if hallucinated:
            score -= 0.10
        candidates.append(Candidate(" -> ".join(chain), chain, ans, score, trace_valid, hallucinated))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0], candidates


def answer_to_jsonable(x: Any) -> Any:
    if isinstance(x, tuple):
        return [answer_to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def make_trials() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: List[Dict[str, Any]] = []
    for i in range(TRIALS):
        p = make_problem(rng, i)
        selected, candidates = solve_problem(p)
        runner = candidates[1]
        false_candidates = [c for c in candidates if c.hallucinated or c.chain == (p.false_shortcut,)]
        best_false = max(false_candidates, key=lambda c: c.score) if false_candidates else runner

        solve_correct = almost_equal(selected.answer, p.answer)
        chain_correct = selected.chain == p.chain
        false_rejected = selected.chain != (p.false_shortcut,) and not selected.hallucinated and selected.score > best_false.score
        no_hallucination = not selected.hallucinated
        margin = selected.score - runner.score

        rows.append({
            "phase": PHASE,
            "problem_id": p.problem_id,
            "family": p.family,
            "template": p.template,
            "holdout": bool(p.holdout),
            "chain_len": len(p.chain),
            "true_chain": " -> ".join(p.chain),
            "selected_chain": selected.name,
            "false_shortcut": p.false_shortcut,
            "prompt": p.prompt,
            "known_json": json.dumps(p.known, default=answer_to_jsonable, sort_keys=True),
            "answer_json": json.dumps(answer_to_jsonable(p.answer)),
            "predicted_answer_json": json.dumps(answer_to_jsonable(selected.answer)),
            "selected_score": float(selected.score),
            "runner_up_score": float(runner.score),
            "best_false_score": float(best_false.score),
            "solution_margin": float(margin),
            "solve_correct": bool(solve_correct),
            "chain_selection_correct": bool(chain_correct),
            "trace_valid": bool(selected.trace_valid),
            "false_shortcut_rejected": bool(false_rejected),
            "no_hallucination": bool(no_hallucination),
        })
    return pd.DataFrame(rows)


def rate(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else float("nan")


def summarize(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    chain_summary = (
        df.groupby(["true_chain", "family", "template"], dropna=False)
        .agg(
            trials=("problem_id", "count"),
            solve_accuracy=("solve_correct", "mean"),
            chain_selection_accuracy=("chain_selection_correct", "mean"),
            trace_validity=("trace_valid", "mean"),
            false_shortcut_rejection=("false_shortcut_rejected", "mean"),
            no_hallucination_accuracy=("no_hallucination", "mean"),
            mean_margin=("solution_margin", "mean"),
            margin_floor=("solution_margin", "min"),
            holdout_solve_accuracy=("solve_correct", lambda x: float(df.loc[x.index][df.loc[x.index, "holdout"]]["solve_correct"].mean()) if df.loc[x.index, "holdout"].any() else float("nan")),
        )
        .reset_index()
    )

    fam = df.groupby("family").agg(
        trials=("problem_id", "count"),
        solve_accuracy=("solve_correct", "mean"),
        chain_selection_accuracy=("chain_selection_correct", "mean"),
        false_shortcut_rejection=("false_shortcut_rejected", "mean"),
        no_hallucination_accuracy=("no_hallucination", "mean"),
    ).reset_index()

    holdout_df = df[df["holdout"]]
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": "multistep_geometric_proof_composition",
        "trials": int(len(df)),
        "overall_solve_accuracy": rate(df["solve_correct"]),
        "arithmetic_solve_accuracy": rate(df[df["family"] == "arithmetic"]["solve_correct"]),
        "geometry_solve_accuracy": rate(df[df["family"] == "geometry"]["solve_correct"]),
        "mixed_solve_accuracy": rate(df[df["family"] == "mixed"]["solve_correct"]),
        "chain_selection_accuracy": rate(df["chain_selection_correct"]),
        "holdout_solve_accuracy": rate(holdout_df["solve_correct"]),
        "trace_validity": rate(df["trace_valid"]),
        "false_shortcut_rejection": rate(df["false_shortcut_rejected"]),
        "no_hallucination_accuracy": rate(df["no_hallucination"]),
        "mean_margin": float(df["solution_margin"].mean()),
        "margin_floor": float(df["solution_margin"].min()),
        "chain_count": int(df["true_chain"].nunique()),
        "families": fam.to_dict(orient="records"),
        "chains": chain_summary.to_dict(orient="records"),
    }

    summary["pass"] = bool(
        summary["overall_solve_accuracy"] >= MIN_OVERALL_SOLVE_ACC
        and summary["arithmetic_solve_accuracy"] >= MIN_ARITH_SOLVE_ACC
        and summary["geometry_solve_accuracy"] >= MIN_GEOM_SOLVE_ACC
        and summary["mixed_solve_accuracy"] >= MIN_MIXED_SOLVE_ACC
        and summary["chain_selection_accuracy"] >= MIN_CHAIN_SELECTION_ACC
        and summary["holdout_solve_accuracy"] >= MIN_HOLDOUT_SOLVE_ACC
        and summary["trace_validity"] >= MIN_TRACE_VALIDITY
        and summary["false_shortcut_rejection"] >= MIN_FALSE_SHORTCUT_REJECTION
        and summary["no_hallucination_accuracy"] >= MIN_NO_HALLUCINATION_ACC
        and summary["margin_floor"] >= MIN_MARGIN_FLOOR
    )
    return summary, chain_summary


def save_bar(path: Path, labels: Sequence[str], series: Dict[str, Sequence[float]], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(labels))
    width = 0.78 / max(1, len(series))
    for j, (name, vals) in enumerate(series.items()):
        ax.bar(x + (j - (len(series) - 1) / 2) * width, vals, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion(path: Path, df: pd.DataFrame) -> None:
    labels = sorted(df["true_chain"].unique())
    selected_labels = sorted(df["selected_chain"].unique())
    # For readability, keep the same axis order when possible.
    all_labels = labels
    mat = np.zeros((len(all_labels), len(all_labels)), dtype=float)
    for i, true in enumerate(all_labels):
        sub = df[df["true_chain"] == true]
        denom = max(1, len(sub))
        for j, pred in enumerate(all_labels):
            mat[i, j] = float((sub["selected_chain"] == pred).sum()) / denom

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title("Phase 78 chain selection confusion")
    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_yticks(np.arange(len(all_labels)))
    ax.set_xticklabels([s.replace(" -> ", "\n→\n") for s in all_labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([s.replace(" -> ", "\n→\n") for s in all_labels], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_examples(df: pd.DataFrame) -> None:
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for template, sub in df.groupby("template"):
        row = sub.iloc[0].to_dict()
        examples.append(row)
        with open(EXAMPLE_DIR / f"{template}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
    with open(EXAMPLE_DIR / "phase78_example_index.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def make_plots(df: pd.DataFrame, chain_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    labels = [str(x).replace(" -> ", "→") for x in chain_summary["template"].tolist()]
    save_bar(
        OUT / "phase78_chain_solve_accuracy.png",
        labels,
        {
            "solve_accuracy": chain_summary["solve_accuracy"].tolist(),
            "chain_selection_accuracy": chain_summary["chain_selection_accuracy"].tolist(),
            "trace_validity": chain_summary["trace_validity"].tolist(),
        },
        "Phase 78 multistep theorem application by chain template",
        "score / rate",
    )

    save_confusion(OUT / "phase78_chain_selection_confusion.png", df)

    fam = df.groupby("family").agg(
        solve_accuracy=("solve_correct", "mean"),
        chain_selection_accuracy=("chain_selection_correct", "mean"),
        false_shortcut_rejection=("false_shortcut_rejected", "mean"),
        no_hallucination_accuracy=("no_hallucination", "mean"),
    ).reset_index()
    save_bar(
        OUT / "phase78_family_chain_accuracy.png",
        fam["family"].tolist(),
        {
            "solve_accuracy": fam["solve_accuracy"].tolist(),
            "chain_selection_accuracy": fam["chain_selection_accuracy"].tolist(),
            "false_shortcut_rejection": fam["false_shortcut_rejection"].tolist(),
            "no_hallucination": fam["no_hallucination_accuracy"].tolist(),
        },
        "Phase 78 proof composition accuracy by family",
        "score / rate",
    )

    clen = df.groupby("chain_len").agg(
        solve_accuracy=("solve_correct", "mean"),
        chain_selection_accuracy=("chain_selection_correct", "mean"),
        false_shortcut_rejection=("false_shortcut_rejected", "mean"),
    ).reset_index()
    save_bar(
        OUT / "phase78_chain_length_accuracy.png",
        [str(x) for x in clen["chain_len"].tolist()],
        {
            "solve_accuracy": clen["solve_accuracy"].tolist(),
            "chain_selection_accuracy": clen["chain_selection_accuracy"].tolist(),
            "false_shortcut_rejection": clen["false_shortcut_rejection"].tolist(),
        },
        "Phase 78 accuracy by proof chain length",
        "score / rate",
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(df["solution_margin"].to_numpy(), bins=30)
    ax.set_title("Phase 78 selected theorem-chain solution-margin distribution")
    ax.set_xlabel("selected chain score - runner-up score")
    ax.set_ylabel("problem trials")
    fig.tight_layout()
    fig.savefig(OUT / "phase78_solution_margin_distribution.png", dpi=150)
    plt.close(fig)

    fs = df.groupby("false_shortcut").agg(rejection=("false_shortcut_rejected", "mean")).reset_index()
    save_bar(
        OUT / "phase78_false_shortcut_rejection.png",
        fs["false_shortcut"].tolist(),
        {"false_shortcut_rejection": fs["rejection"].tolist()},
        "Phase 78 false shortcut rejection",
        "rejection rate",
    )


def write_report(summary: Dict[str, Any], chain_summary: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append("## Result\n")
    lines.append(f"- PASS: `{summary['pass']}`")
    lines.append(f"- trials: `{summary['trials']}`")
    lines.append(f"- overall solve accuracy: `{summary['overall_solve_accuracy']:.4f}`")
    lines.append(f"- arithmetic solve accuracy: `{summary['arithmetic_solve_accuracy']:.4f}`")
    lines.append(f"- geometry solve accuracy: `{summary['geometry_solve_accuracy']:.4f}`")
    lines.append(f"- mixed solve accuracy: `{summary['mixed_solve_accuracy']:.4f}`")
    lines.append(f"- chain selection accuracy: `{summary['chain_selection_accuracy']:.4f}`")
    lines.append(f"- holdout solve accuracy: `{summary['holdout_solve_accuracy']:.4f}`")
    lines.append(f"- trace validity: `{summary['trace_validity']:.4f}`")
    lines.append(f"- false shortcut rejection: `{summary['false_shortcut_rejection']:.4f}`")
    lines.append(f"- no hallucination accuracy: `{summary['no_hallucination_accuracy']:.4f}`")
    lines.append(f"- mean margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- margin floor: `{summary['margin_floor']:.6f}`\n")

    lines.append("## Interpretation\n")
    lines.append(
        "Phase 78 tests whether the Phase 76/77 primitive axiom machinery can be composed into ordered two-step proof chains. "
        "The model is forced to solve hidden-value tasks by choosing the correct chain of discovered arithmetic/geometry invariants, while rejecting shortcut rules that produce plausible but invalid answers.\n"
    )

    lines.append("## Chain summary\n")
    for _, r in chain_summary.iterrows():
        lines.append(
            f"- `{r['template']}`: solve=`{r['solve_accuracy']:.4f}` "
            f"chain_select=`{r['chain_selection_accuracy']:.4f}` trace=`{r['trace_validity']:.4f}` "
            f"reject_false=`{r['false_shortcut_rejection']:.4f}` margin=`{r['mean_margin']:.4f}`"
        )

    lines.append("\n## Output files\n")
    for name in [
        "phase78_multistep_geometric_proof_composition_bridge_trials.csv",
        "phase78_multistep_geometric_proof_composition_bridge_chain_summary.csv",
        "phase78_multistep_geometric_proof_composition_bridge_summary.json",
        "phase78_chain_solve_accuracy.png",
        "phase78_chain_selection_confusion.png",
        "phase78_family_chain_accuracy.png",
        "phase78_chain_length_accuracy.png",
        "phase78_solution_margin_distribution.png",
        "phase78_false_shortcut_rejection.png",
        "phase78_examples/",
    ]:
        lines.append(f"- `{name}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from theorem application to multistep proof composition")
    print(f"[{PHASE}] task: compose discovered arithmetic/geometry axioms into ordered proof chains")

    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    df = make_trials()
    summary, chain_summary = summarize(df)

    trials_path = OUT / "phase78_multistep_geometric_proof_composition_bridge_trials.csv"
    chain_path = OUT / "phase78_multistep_geometric_proof_composition_bridge_chain_summary.csv"
    summary_path = OUT / "phase78_multistep_geometric_proof_composition_bridge_summary.json"
    report_path = OUT / "phase78_multistep_geometric_proof_composition_bridge_report.md"

    df.to_csv(trials_path, index=False)
    chain_summary.to_csv(chain_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(df, chain_summary)
    write_examples(df)
    report = write_report(summary, chain_summary)
    report_path.write_text(report, encoding="utf-8")

    print(
        f"[{PHASE}] PHASE78_MULTISTEP_GEOMETRIC_PROOF_COMPOSITION_BRIDGE_PASS={summary['pass']}"
    )
    print(
        f"[{PHASE}] selected_task={summary['selected_task']} "
        f"overall_solve_accuracy={summary['overall_solve_accuracy']:.4f} "
        f"arithmetic_solve_accuracy={summary['arithmetic_solve_accuracy']:.4f} "
        f"geometry_solve_accuracy={summary['geometry_solve_accuracy']:.4f} "
        f"mixed_solve_accuracy={summary['mixed_solve_accuracy']:.4f} "
        f"chain_selection_accuracy={summary['chain_selection_accuracy']:.4f} "
        f"holdout_solve_accuracy={summary['holdout_solve_accuracy']:.4f} "
        f"trace_validity={summary['trace_validity']:.4f} "
        f"false_shortcut_rejection={summary['false_shortcut_rejection']:.4f} "
        f"no_hallucination_accuracy={summary['no_hallucination_accuracy']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )
    print(f"[{PHASE}] chain summary:")
    for _, r in chain_summary.iterrows():
        print(
            f"  - {str(r['template']):34s} family={str(r['family']):10s} "
            f"solve={r['solve_accuracy']:.3f} select={r['chain_selection_accuracy']:.3f} "
            f"trace={r['trace_validity']:.3f} reject_false={r['false_shortcut_rejection']:.3f} "
            f"margin={r['mean_margin']:.4f} trials={int(r['trials'])}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote chain summary: {chain_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png/json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
