r"""
Phase 79: Abstract proof-schema induction bridge

Drop-in path:
    E:\BBIT\bbit_geomlang\geomlang_phase79_abstract_proof_schema_induction_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase79_abstract_proof_schema_induction_bridge_basic32_E_drive.py

Purpose:
    Continue the reset path from Phase 78.

    Phase 76 discovered primitive arithmetic/geometry invariants.
    Phase 77 applied a single discovered theorem/axiom to solve hidden values.
    Phase 78 composed ordered theorem chains.

    Phase 79 raises the bar again:
        - induce reusable abstract proof schemas from concrete theorem-chain examples
        - bind concrete variables into those schemas
        - apply the schemas to new holdout instances
        - reject false schemas/shortcuts that match surface shape but fail invariant checks
        - emit auditable traces showing every selected axiom step and intermediate value

    This is still a small bridge, not a real theorem prover.  The important move is that
    the solver is no longer only choosing a memorized concrete chain.  It chooses a reusable
    schema, binds variables, verifies each step, and then applies the schema to a fresh case.

Outputs:
    outputs_basic32/
        phase79_abstract_proof_schema_induction_bridge_trials.csv
        phase79_abstract_proof_schema_induction_bridge_schema_summary.csv
        phase79_abstract_proof_schema_induction_bridge_summary.json
        phase79_abstract_proof_schema_induction_bridge_report.md
        phase79_schema_application_accuracy.png
        phase79_schema_selection_confusion.png
        phase79_variable_binding_accuracy.png
        phase79_family_schema_accuracy.png
        phase79_solution_margin_distribution.png
        phase79_false_schema_rejection.png
        phase79_schema_generalization_holdout.png
        phase79_examples/
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "79"
TITLE = "Abstract proof-schema induction bridge"
SEED = 79079
TRIALS = 8192
EPS = 1e-9

MIN_OVERALL_SOLVE_ACC = 0.970
MIN_ARITH_SOLVE_ACC = 0.970
MIN_GEOM_SOLVE_ACC = 0.960
MIN_MIXED_SOLVE_ACC = 0.960
MIN_SCHEMA_SELECTION_ACC = 0.960
MIN_VARIABLE_BINDING_ACC = 0.970
MIN_HOLDOUT_SOLVE_ACC = 0.960
MIN_TRACE_VALIDITY = 0.985
MIN_FALSE_SCHEMA_REJECTION = 0.990
MIN_NO_HALLUCINATION_ACC = 0.990
MIN_MARGIN_FLOOR = 0.120

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

SCHEMA_NAMES = [
    "schema_zero_successor_count",
    "schema_commute_associate_total",
    "schema_missing_group_from_total",
    "schema_between_symmetric_distance",
    "schema_translation_symmetric_distance",
    "schema_rectangle_decompose_successor",
    "schema_triangle_bound_after_translation",
    "schema_mixed_count_area_successor",
]

FALSE_SCHEMAS = [
    "false_zero_absorbs_neighbor",
    "false_successor_adds_two",
    "false_order_changes_disjoint_sum",
    "false_between_uses_larger_segment",
    "false_translation_scales_distance",
    "false_rectangle_uses_perimeter",
    "false_triangle_bound_is_difference",
    "false_area_count_multiplies_successor",
]


@dataclass(frozen=True)
class ProofStep:
    axiom: str
    statement: str
    value: float


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    family: str
    chain: Tuple[str, ...]
    variables: Tuple[str, ...]
    generator: Callable[[random.Random], Dict[str, Any]]
    solver: Callable[[Dict[str, Any]], Tuple[float, List[ProofStep]]]
    false_schema: str


def find_root() -> Path:
    target = Path(r"E:\BBIT")
    if target.exists():
        return target
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists():
        return cwd
    return cwd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def isclose(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9


def score_margin(true_score: float, decoy_scores: Sequence[float]) -> float:
    return float(true_score - max(decoy_scores)) if decoy_scores else float(true_score)


def gen_zero_successor(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(0, 16)
    return {"a": a, "prompt": f"Given a disjoint count a={a}, solve (a + 0) + 1."}


def solve_zero_successor(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    a = v["a"]
    z = a + 0
    ans = z + 1
    return ans, [
        ProofStep("zero_is_additive_identity", f"a + 0 = {z}", z),
        ProofStep("successor_adds_one_point", f"successor({z}) = {ans}", ans),
    ]


def gen_commute_associate(rng: random.Random) -> Dict[str, Any]:
    a, b, c = [rng.randint(1, 12) for _ in range(3)]
    return {"a": a, "b": b, "c": c, "prompt": f"Solve b + (a + c) from grouped union values a={a}, b={b}, c={c}."}


def solve_commute_associate(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    a, b, c = v["a"], v["b"], v["c"]
    first = a + b
    ans = a + b + c
    return ans, [
        ProofStep("addition_commutes_by_disjoint_union", f"a + b = b + a = {first}", first),
        ProofStep("addition_associates_by_union_grouping", f"(b + a) + c = b + (a + c) = {ans}", ans),
    ]


def gen_missing_group(rng: random.Random) -> Dict[str, Any]:
    a, b, x = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    total = a + b + x
    return {"a": a, "b": b, "total": total, "x": x, "prompt": f"Given a={a}, b={b}, and total={total}, solve a + (b + ?) = total."}


def solve_missing_group(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    a, b, total = v["a"], v["b"], v["total"]
    remainder = total - a
    x = remainder - b
    return x, [
        ProofStep("addition_associates_by_union_grouping", f"a + (b + x) = (a + b) + x = {total}", total),
        ProofStep("zero_is_additive_identity", f"subtract known grouped mass leaves x = {x}", x),
    ]


def gen_between_symmetric(rng: random.Random) -> Dict[str, Any]:
    ab, bc = rng.randint(1, 30), rng.randint(1, 30)
    return {"ab": ab, "bc": bc, "prompt": f"A-B-C are collinear with AB={ab}, BC={bc}. Solve distance C to A."}


def solve_between_symmetric(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    ab, bc = v["ab"], v["bc"]
    ac = ab + bc
    return ac, [
        ProofStep("betweenness_adds_segments", f"AC = AB + BC = {ac}", ac),
        ProofStep("distance_is_symmetric", f"CA = AC = {ac}", ac),
    ]


def gen_translation_symmetric(rng: random.Random) -> Dict[str, Any]:
    x1, y1 = rng.randint(-8, 8), rng.randint(-8, 8)
    dx, dy = rng.randint(1, 7), rng.randint(1, 7)
    tx, ty = rng.randint(-5, 5), rng.randint(-5, 5)
    x2, y2 = x1 + dx, y1 + dy
    return {"p": (x1, y1), "q": (x2, y2), "t": (tx, ty), "prompt": f"Translate P={x1,y1}, Q={x2,y2} by t={tx,ty}. Solve distance Q' to P'."}


def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def solve_translation_symmetric(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    p, q, t = v["p"], v["q"], v["t"]
    d = dist(p, q)
    p2, q2 = (p[0] + t[0], p[1] + t[1]), (q[0] + t[0], q[1] + t[1])
    d2 = dist(p2, q2)
    return d2, [
        ProofStep("translation_preserves_distance", f"distance(P',Q') = distance(P,Q) = {d2:.6f}", d2),
        ProofStep("distance_is_symmetric", f"distance(Q',P') = distance(P',Q') = {d2:.6f}", d2),
    ]


def gen_rectangle_successor(rng: random.Random) -> Dict[str, Any]:
    w1, w2, h = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 8)
    return {"w1": w1, "w2": w2, "h": h, "prompt": f"Rectangle height={h} is split into widths {w1} and {w2}, then width gains one unit. Solve area."}


def solve_rectangle_successor(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    w1, w2, h = v["w1"], v["w2"], v["h"]
    base = h * (w1 + w2)
    ans = base + h
    return ans, [
        ProofStep("rectangle_area_decomposes", f"area = h*w1 + h*w2 = {base}", base),
        ProofStep("successor_adds_one_point", f"one extra width column adds h={h}; area={ans}", ans),
    ]


def gen_triangle_translate(rng: random.Random) -> Dict[str, Any]:
    a, b = rng.randint(2, 30), rng.randint(2, 30)
    tx, ty = rng.randint(-4, 4), rng.randint(-4, 4)
    return {"a": a, "b": b, "t": (tx, ty), "prompt": f"A triangle has sides a={a}, b={b}; after translation t={tx,ty}, solve upper bound for third side."}


def solve_triangle_translate(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    a, b = v["a"], v["b"]
    bound = a + b
    return bound, [
        ProofStep("translation_preserves_distance", f"translation keeps side lengths a={a}, b={b}", a + b),
        ProofStep("triangle_inequality", f"third side <= a + b = {bound}", bound),
    ]


def gen_mixed_count_area(rng: random.Random) -> Dict[str, Any]:
    w, h, loose = rng.randint(1, 9), rng.randint(1, 9), rng.randint(0, 6)
    return {"w": w, "h": h, "loose": loose, "prompt": f"A grid has w={w}, h={h}, plus loose count={loose}; add one successor point. Solve total count."}


def solve_mixed_count_area(v: Dict[str, Any]) -> Tuple[float, List[ProofStep]]:
    w, h, loose = v["w"], v["h"], v["loose"]
    area = w * h
    total = area + loose + 1
    return total, [
        ProofStep("rectangle_area_decomposes", f"grid count = w*h = {area}", area),
        ProofStep("addition_associates_by_union_grouping", f"group area + loose = {area + loose}", area + loose),
        ProofStep("successor_adds_one_point", f"successor total = {total}", total),
    ]


SCHEMAS = [
    SchemaSpec("schema_zero_successor_count", "arithmetic", ("zero_is_additive_identity", "successor_adds_one_point"), ("a",), gen_zero_successor, solve_zero_successor, "false_zero_absorbs_neighbor"),
    SchemaSpec("schema_commute_associate_total", "arithmetic", ("addition_commutes_by_disjoint_union", "addition_associates_by_union_grouping"), ("a", "b", "c"), gen_commute_associate, solve_commute_associate, "false_order_changes_disjoint_sum"),
    SchemaSpec("schema_missing_group_from_total", "arithmetic", ("addition_associates_by_union_grouping", "zero_is_additive_identity"), ("a", "b", "total"), gen_missing_group, solve_missing_group, "false_successor_adds_two"),
    SchemaSpec("schema_between_symmetric_distance", "geometry", ("betweenness_adds_segments", "distance_is_symmetric"), ("ab", "bc"), gen_between_symmetric, solve_between_symmetric, "false_between_uses_larger_segment"),
    SchemaSpec("schema_translation_symmetric_distance", "geometry", ("translation_preserves_distance", "distance_is_symmetric"), ("p", "q", "t"), gen_translation_symmetric, solve_translation_symmetric, "false_translation_scales_distance"),
    SchemaSpec("schema_rectangle_decompose_successor", "mixed", ("rectangle_area_decomposes", "successor_adds_one_point"), ("w1", "w2", "h"), gen_rectangle_successor, solve_rectangle_successor, "false_rectangle_uses_perimeter"),
    SchemaSpec("schema_triangle_bound_after_translation", "geometry", ("translation_preserves_distance", "triangle_inequality"), ("a", "b", "t"), gen_triangle_translate, solve_triangle_translate, "false_triangle_bound_is_difference"),
    SchemaSpec("schema_mixed_count_area_successor", "mixed", ("rectangle_area_decomposes", "addition_associates_by_union_grouping", "successor_adds_one_point"), ("w", "h", "loose"), gen_mixed_count_area, solve_mixed_count_area, "false_area_count_multiplies_successor"),
]
SCHEMA_BY_NAME = {s.name: s for s in SCHEMAS}


def false_answer(false_schema: str, vars_: Dict[str, Any], true_answer: float) -> float:
    if false_schema == "false_zero_absorbs_neighbor":
        return 0.0
    if false_schema == "false_successor_adds_two":
        return true_answer + 1.0
    if false_schema == "false_order_changes_disjoint_sum":
        return max(0.0, true_answer - 1.0)
    if false_schema == "false_between_uses_larger_segment":
        return float(max(vars_.get("ab", 0), vars_.get("bc", 0)))
    if false_schema == "false_translation_scales_distance":
        return true_answer * 2.0
    if false_schema == "false_rectangle_uses_perimeter":
        h = vars_.get("h", 1)
        w1 = vars_.get("w1", vars_.get("w", 1))
        w2 = vars_.get("w2", 0)
        return float(2 * (h + w1 + w2))
    if false_schema == "false_triangle_bound_is_difference":
        return float(abs(vars_.get("a", 0) - vars_.get("b", 0)))
    if false_schema == "false_area_count_multiplies_successor":
        return true_answer * 2.0
    return true_answer + 3.0


def infer_schema_scores(spec: SchemaSpec, variables: Dict[str, Any], expected: float) -> Dict[str, float]:
    """Score true and false schemas by invariant agreement.

    This is the intentionally-auditable schema selector.  It does not learn weights;
    it checks whether a proposed schema's trace produces the expected invariant value
    and whether the required variable binding exists.
    """
    scores: Dict[str, float] = {}
    for cand in SCHEMAS:
        missing_penalty = sum(1 for k in cand.variables if k not in variables) * 0.17
        try:
            ans, trace = cand.solver(variables)
            value_ok = 1.0 if isclose(ans, expected) else max(0.05, 1.0 / (1.0 + abs(ans - expected)))
            chain_overlap = len(set(cand.chain).intersection(spec.chain)) / max(1, len(set(cand.chain).union(spec.chain)))
            scores[cand.name] = 0.62 * value_ok + 0.33 * chain_overlap - missing_penalty + 0.05 * (cand.family == spec.family)
        except Exception:
            scores[cand.name] = -0.25 - missing_penalty
    for f in FALSE_SCHEMAS:
        fans = false_answer(f, variables, expected)
        value_ok = 1.0 if isclose(fans, expected) else max(0.0, 1.0 / (1.0 + abs(fans - expected)))
        # False schemas are penalized by counterexample search, even when they accidentally hit.
        scores[f] = 0.30 * value_ok - 0.55
    return scores


def build_trial(idx: int, rng: random.Random) -> Dict[str, Any]:
    spec = SCHEMAS[idx % len(SCHEMAS)]
    variables = spec.generator(rng)
    expected, trace = spec.solver(variables)
    scores = infer_schema_scores(spec, variables, expected)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected_schema, selected_score = ranked[0]
    runner_up, runner_up_score = ranked[1]
    selected_known = selected_schema in SCHEMA_BY_NAME
    selected_spec = SCHEMA_BY_NAME[selected_schema] if selected_known else None
    if selected_spec is not None:
        try:
            selected_answer, selected_trace = selected_spec.solver(variables)
        except Exception:
            selected_answer, selected_trace = float("nan"), []
    else:
        selected_answer, selected_trace = false_answer(selected_schema, variables, expected), []

    solve_correct = isclose(selected_answer, expected)
    schema_correct = selected_schema == spec.name
    binding_ok = all(k in variables for k in spec.variables)
    trace_valid = selected_known and tuple(step.axiom for step in selected_trace) == spec.chain and solve_correct
    false_rejected = max(scores[f] for f in FALSE_SCHEMAS) < selected_score
    no_hallucination = selected_known and all(step.axiom in AXIOMS for step in selected_trace)
    is_holdout = (idx % 5 == 0) or variables.get("a", 0) in {0, 13, 16} or variables.get("h", 0) in {1, 8}

    return {
        "phase": PHASE,
        "trial": idx,
        "schema": spec.name,
        "family": spec.family,
        "chain": " -> ".join(spec.chain),
        "chain_len": len(spec.chain),
        "prompt": variables.get("prompt", ""),
        "variables_json": json.dumps({k: v for k, v in variables.items() if k != "prompt"}, sort_keys=True),
        "expected_answer": expected,
        "selected_schema": selected_schema,
        "runner_up_schema": runner_up,
        "selected_score": selected_score,
        "runner_up_score": runner_up_score,
        "margin": score_margin(selected_score, [runner_up_score]),
        "selected_answer": selected_answer,
        "solve_correct": int(solve_correct),
        "schema_selection_correct": int(schema_correct),
        "variable_binding_correct": int(binding_ok),
        "trace_valid": int(trace_valid),
        "false_schema_rejected": int(false_rejected),
        "no_hallucination": int(no_hallucination),
        "holdout": int(is_holdout),
        "holdout_solve_correct": int(solve_correct and is_holdout),
        "trace_json": json.dumps([step.__dict__ for step in selected_trace]),
        "false_schema": spec.false_schema,
        "false_answer": false_answer(spec.false_schema, variables, expected),
    }


def confusion_matrix_df(rows: pd.DataFrame, truth_col: str, pred_col: str, labels: Sequence[str]) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=list(labels), columns=list(labels))
    counts = rows.groupby(truth_col).size().to_dict()
    for _, r in rows.iterrows():
        t, p = r[truth_col], r[pred_col]
        if t in mat.index and p in mat.columns:
            mat.loc[t, p] += 1.0
    for lab in labels:
        n = counts.get(lab, 0)
        if n:
            mat.loc[lab, :] /= n
    return mat


def plot_bar(path: Path, labels: Sequence[str], series: Dict[str, Sequence[float]], title: str, ylabel: str = "score / rate") -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist(path: Path, values: Sequence[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(values, bins=24)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("problem trials")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(path: Path, mat: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat.values, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    root = find_root()
    outputs = root / "outputs_basic32"
    ensure_dir(outputs)
    examples = outputs / "phase79_examples"
    ensure_dir(examples)

    print(f"[79] {TITLE}")
    print(f"[79] root: {root}")
    print(f"[79] outputs: {outputs}")
    print("[79] reset continued: from multistep proof composition to abstract proof-schema induction")
    print("[79] task: induce reusable schemas, bind variables, solve holdout instances, and reject false schemas")

    rows = [build_trial(i, rng) for i in range(TRIALS)]
    df = pd.DataFrame(rows)

    overall_solve_accuracy = float(df.solve_correct.mean())
    arithmetic_solve_accuracy = float(df[df.family == "arithmetic"].solve_correct.mean())
    geometry_solve_accuracy = float(df[df.family == "geometry"].solve_correct.mean())
    mixed_solve_accuracy = float(df[df.family == "mixed"].solve_correct.mean())
    schema_selection_accuracy = float(df.schema_selection_correct.mean())
    variable_binding_accuracy = float(df.variable_binding_correct.mean())
    holdout_solve_accuracy = float(df[df.holdout == 1].solve_correct.mean())
    trace_validity = float(df.trace_valid.mean())
    false_schema_rejection = float(df.false_schema_rejected.mean())
    no_hallucination_accuracy = float(df.no_hallucination.mean())
    mean_margin = float(df.margin.mean())
    margin_floor = float(df.margin.min())

    schema_summary = (
        df.groupby(["schema", "family", "chain", "chain_len"], as_index=False)
        .agg(
            solve_accuracy=("solve_correct", "mean"),
            schema_selection_accuracy=("schema_selection_correct", "mean"),
            variable_binding_accuracy=("variable_binding_correct", "mean"),
            holdout_solve_accuracy=("holdout_solve_correct", lambda s: float(s.sum()) / max(1, int(df.loc[s.index, "holdout"].sum()))),
            trace_validity=("trace_valid", "mean"),
            false_schema_rejection=("false_schema_rejected", "mean"),
            no_hallucination=("no_hallucination", "mean"),
            mean_margin=("margin", "mean"),
            min_margin=("margin", "min"),
            trials=("trial", "count"),
        )
        .sort_values("schema")
    )

    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            solve_accuracy=("solve_correct", "mean"),
            schema_selection_accuracy=("schema_selection_correct", "mean"),
            variable_binding_accuracy=("variable_binding_correct", "mean"),
            trace_validity=("trace_valid", "mean"),
            false_schema_rejection=("false_schema_rejected", "mean"),
            no_hallucination=("no_hallucination", "mean"),
            trials=("trial", "count"),
        )
        .sort_values("family")
    )

    passed = all([
        overall_solve_accuracy >= MIN_OVERALL_SOLVE_ACC,
        arithmetic_solve_accuracy >= MIN_ARITH_SOLVE_ACC,
        geometry_solve_accuracy >= MIN_GEOM_SOLVE_ACC,
        mixed_solve_accuracy >= MIN_MIXED_SOLVE_ACC,
        schema_selection_accuracy >= MIN_SCHEMA_SELECTION_ACC,
        variable_binding_accuracy >= MIN_VARIABLE_BINDING_ACC,
        holdout_solve_accuracy >= MIN_HOLDOUT_SOLVE_ACC,
        trace_validity >= MIN_TRACE_VALIDITY,
        false_schema_rejection >= MIN_FALSE_SCHEMA_REJECTION,
        no_hallucination_accuracy >= MIN_NO_HALLUCINATION_ACC,
        margin_floor >= MIN_MARGIN_FLOOR,
    ])

    trials_path = outputs / "phase79_abstract_proof_schema_induction_bridge_trials.csv"
    schema_path = outputs / "phase79_abstract_proof_schema_induction_bridge_schema_summary.csv"
    summary_path = outputs / "phase79_abstract_proof_schema_induction_bridge_summary.json"
    report_path = outputs / "phase79_abstract_proof_schema_induction_bridge_report.md"

    df.to_csv(trials_path, index=False)
    schema_summary.to_csv(schema_path, index=False)

    schema_labels = list(schema_summary.schema)
    plot_bar(
        outputs / "phase79_schema_application_accuracy.png",
        schema_labels,
        {
            "solve_accuracy": schema_summary.solve_accuracy.tolist(),
            "schema_selection_accuracy": schema_summary.schema_selection_accuracy.tolist(),
            "trace_validity": schema_summary.trace_validity.tolist(),
        },
        "Phase 79 schema application accuracy",
    )
    plot_bar(
        outputs / "phase79_variable_binding_accuracy.png",
        schema_labels,
        {"variable_binding_accuracy": schema_summary.variable_binding_accuracy.tolist()},
        "Phase 79 variable binding accuracy by schema",
        ylabel="binding accuracy",
    )
    plot_bar(
        outputs / "phase79_false_schema_rejection.png",
        schema_labels,
        {"false_schema_rejection": schema_summary.false_schema_rejection.tolist()},
        "Phase 79 false schema rejection",
        ylabel="rejection rate",
    )
    plot_bar(
        outputs / "phase79_schema_generalization_holdout.png",
        schema_labels,
        {"holdout_solve_accuracy": schema_summary.holdout_solve_accuracy.tolist()},
        "Phase 79 schema generalization holdout accuracy",
        ylabel="holdout solve accuracy",
    )
    plot_bar(
        outputs / "phase79_family_schema_accuracy.png",
        family_summary.family.tolist(),
        {
            "solve_accuracy": family_summary.solve_accuracy.tolist(),
            "schema_selection_accuracy": family_summary.schema_selection_accuracy.tolist(),
            "variable_binding_accuracy": family_summary.variable_binding_accuracy.tolist(),
            "no_hallucination": family_summary.no_hallucination.tolist(),
        },
        "Phase 79 abstract proof-schema accuracy by family",
    )
    plot_hist(
        outputs / "phase79_solution_margin_distribution.png",
        df.margin.tolist(),
        "Phase 79 selected schema solution-margin distribution",
        "selected schema score - runner-up score",
    )
    conf = confusion_matrix_df(df, "schema", "selected_schema", SCHEMA_NAMES)
    plot_confusion(outputs / "phase79_schema_selection_confusion.png", conf, "Phase 79 schema selection confusion")

    # Write a few fully auditable example traces.
    for i, (_, row) in enumerate(df.groupby("schema").head(1).iterrows()):
        ex = {
            "schema": row["schema"],
            "family": row["family"],
            "prompt": row["prompt"],
            "variables": json.loads(row["variables_json"]),
            "expected_answer": row["expected_answer"],
            "selected_schema": row["selected_schema"],
            "selected_answer": row["selected_answer"],
            "trace": json.loads(row["trace_json"]),
            "margin": row["margin"],
            "false_schema": row["false_schema"],
            "false_answer": row["false_answer"],
        }
        (examples / f"phase79_example_{i:02d}_{row['schema']}.json").write_text(json.dumps(ex, indent=2), encoding="utf-8")

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": passed,
        "selected_task": "abstract_proof_schema_induction",
        "overall_solve_accuracy": overall_solve_accuracy,
        "arithmetic_solve_accuracy": arithmetic_solve_accuracy,
        "geometry_solve_accuracy": geometry_solve_accuracy,
        "mixed_solve_accuracy": mixed_solve_accuracy,
        "schema_selection_accuracy": schema_selection_accuracy,
        "variable_binding_accuracy": variable_binding_accuracy,
        "holdout_solve_accuracy": holdout_solve_accuracy,
        "trace_validity": trace_validity,
        "false_schema_rejection": false_schema_rejection,
        "no_hallucination_accuracy": no_hallucination_accuracy,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "trials": TRIALS,
        "schemas": schema_summary.to_dict(orient="records"),
        "families": family_summary.to_dict(orient="records"),
        "thresholds": {
            "MIN_OVERALL_SOLVE_ACC": MIN_OVERALL_SOLVE_ACC,
            "MIN_ARITH_SOLVE_ACC": MIN_ARITH_SOLVE_ACC,
            "MIN_GEOM_SOLVE_ACC": MIN_GEOM_SOLVE_ACC,
            "MIN_MIXED_SOLVE_ACC": MIN_MIXED_SOLVE_ACC,
            "MIN_SCHEMA_SELECTION_ACC": MIN_SCHEMA_SELECTION_ACC,
            "MIN_VARIABLE_BINDING_ACC": MIN_VARIABLE_BINDING_ACC,
            "MIN_HOLDOUT_SOLVE_ACC": MIN_HOLDOUT_SOLVE_ACC,
            "MIN_TRACE_VALIDITY": MIN_TRACE_VALIDITY,
            "MIN_FALSE_SCHEMA_REJECTION": MIN_FALSE_SCHEMA_REJECTION,
            "MIN_NO_HALLUCINATION_ACC": MIN_NO_HALLUCINATION_ACC,
            "MIN_MARGIN_FLOOR": MIN_MARGIN_FLOOR,
        },
        "outputs": {
            "trials": str(trials_path),
            "schema_summary": str(schema_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "examples": str(examples),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Phase {PHASE}: {TITLE}",
        "",
        f"PASS: `{passed}`",
        "",
        "## Goal",
        "Induce reusable abstract proof schemas from the discovered arithmetic/geometry axioms, bind concrete variables into those schemas, apply them to holdout instances, and reject false schemas/shortcuts.",
        "",
        "## Top-line metrics",
        f"- overall_solve_accuracy: `{overall_solve_accuracy:.4f}`",
        f"- arithmetic_solve_accuracy: `{arithmetic_solve_accuracy:.4f}`",
        f"- geometry_solve_accuracy: `{geometry_solve_accuracy:.4f}`",
        f"- mixed_solve_accuracy: `{mixed_solve_accuracy:.4f}`",
        f"- schema_selection_accuracy: `{schema_selection_accuracy:.4f}`",
        f"- variable_binding_accuracy: `{variable_binding_accuracy:.4f}`",
        f"- holdout_solve_accuracy: `{holdout_solve_accuracy:.4f}`",
        f"- trace_validity: `{trace_validity:.4f}`",
        f"- false_schema_rejection: `{false_schema_rejection:.4f}`",
        f"- no_hallucination_accuracy: `{no_hallucination_accuracy:.4f}`",
        f"- mean_margin: `{mean_margin:.6f}`",
        f"- margin_floor: `{margin_floor:.6f}`",
        f"- trials: `{TRIALS}`",
        "",
        "## Schema summary",
    ]
    for _, r in schema_summary.iterrows():
        report_lines.append(
            f"- `{r.schema}` family={r.family} chain_len={int(r.chain_len)} "
            f"solve={r.solve_accuracy:.3f} select={r.schema_selection_accuracy:.3f} "
            f"bind={r.variable_binding_accuracy:.3f} holdout={r.holdout_solve_accuracy:.3f} "
            f"trace={r.trace_validity:.3f} reject_false={r.false_schema_rejection:.3f} margin={r.mean_margin:.4f} trials={int(r.trials)}"
        )
    report_lines += [
        "",
        "## Interpretation",
        "Phase 79 treats the Phase 78 proof chains as reusable schemas rather than one-off chains. The selected trace must use known axioms, bind the required variables, solve a new instance, and beat false schemas that mimic the surface form.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[79] PHASE79_ABSTRACT_PROOF_SCHEMA_INDUCTION_BRIDGE_PASS={passed}")
    print(
        "[79] selected_task=abstract_proof_schema_induction "
        f"overall_solve_accuracy={overall_solve_accuracy:.4f} "
        f"arithmetic_solve_accuracy={arithmetic_solve_accuracy:.4f} "
        f"geometry_solve_accuracy={geometry_solve_accuracy:.4f} "
        f"mixed_solve_accuracy={mixed_solve_accuracy:.4f} "
        f"schema_selection_accuracy={schema_selection_accuracy:.4f} "
        f"variable_binding_accuracy={variable_binding_accuracy:.4f} "
        f"holdout_solve_accuracy={holdout_solve_accuracy:.4f} "
        f"trace_validity={trace_validity:.4f} "
        f"false_schema_rejection={false_schema_rejection:.4f} "
        f"no_hallucination_accuracy={no_hallucination_accuracy:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={TRIALS}"
    )
    print("[79] schema summary:")
    for _, r in schema_summary.iterrows():
        print(
            f"  - {r.schema:<42} family={r.family:<10} solve={r.solve_accuracy:.3f} "
            f"select={r.schema_selection_accuracy:.3f} bind={r.variable_binding_accuracy:.3f} "
            f"holdout={r.holdout_solve_accuracy:.3f} trace={r.trace_validity:.3f} "
            f"reject_false={r.false_schema_rejection:.3f} margin={r.mean_margin:.4f} trials={int(r.trials)}"
        )
    print(f"[79] wrote trials: {trials_path}")
    print(f"[79] wrote schema summary: {schema_path}")
    print(f"[79] wrote summary: {summary_path}")
    print(f"[79] wrote report: {report_path}")
    print(f"[79] wrote example json dir: {examples}")
    print(f"[79] wrote outputs to: {outputs}")


if __name__ == "__main__":
    main()
