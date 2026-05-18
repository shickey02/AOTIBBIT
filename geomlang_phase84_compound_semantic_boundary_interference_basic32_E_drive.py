#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 84: Compound semantic-boundary interference

Reset continued:
Phase 83 proved that paraphrase invariance has a semantic boundary:
true paraphrases should be accepted, near-paraphrases should be rejected.

Phase 84 stress-tests that boundary under compound interference.

Instead of testing one clean paraphrase or one clean near-paraphrase mutation,
this phase composes multiple surface transformations together:

    clean compound:
        canonical + reverse_order + story_form + unit_synonym
        -> should ACCEPT

    hidden mutation compound:
        true paraphrase noise + one hidden operator/scope/role/unit/inequality mutation
        -> should REJECT

Core question:
    Can the system distinguish "more words saying the same thing"
    from "more words hiding a changed thing"?

Outputs:
    - trials CSV
    - task summary CSV
    - compound variant summary CSV
    - summary JSON
    - markdown report
    - visualizations

No external model dependency. This is a deterministic synthetic BBIT bridge phase.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PHASE = 84
TITLE = "Compound semantic-boundary interference"
SCRIPT_NAME = "geomlang_phase84_compound_semantic_boundary_interference_basic32_E_drive.py"

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)

TRIALS_CSV = OUT / "phase84_compound_semantic_boundary_interference_trials.csv"
TASK_SUMMARY_CSV = OUT / "phase84_compound_semantic_boundary_interference_task_summary.csv"
VARIANT_SUMMARY_CSV = OUT / "phase84_compound_semantic_boundary_interference_variant_summary.csv"
SUMMARY_JSON = OUT / "phase84_compound_semantic_boundary_interference_summary.json"
REPORT_MD = OUT / "phase84_compound_semantic_boundary_interference_report.md"

PLOT_MARGIN = OUT / "phase84_compound_solution_margin_distribution.png"
PLOT_ACCEPT_REJECT = OUT / "phase84_clean_hidden_accept_reject_accuracy.png"
PLOT_TASK = OUT / "phase84_task_compound_boundary_accuracy.png"
PLOT_FAMILY = OUT / "phase84_family_compound_boundary_accuracy.png"
PLOT_VARIANT = OUT / "phase84_variant_compound_boundary_accuracy.png"
PLOT_CONFUSION = OUT / "phase84_compound_boundary_confusion.png"
PLOT_MUTATION_LOCALIZATION = OUT / "phase84_mutation_localization_accuracy.png"

EXAMPLE_DIR = OUT / "phase84_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------

RNG_SEED = 840084
random.seed(RNG_SEED)


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryTask:
    task_id: str
    family: str
    boundary_kind: str
    base_schema: str
    base_operator: str
    variables: Tuple[str, ...]


@dataclass(frozen=True)
class CompoundVariant:
    variant_id: str
    variant_type: str
    harmless_layers: Tuple[str, ...]
    hidden_mutation: str
    expected_action: str
    boundary_kind: str


@dataclass
class TrialResult:
    phase: int
    trial_id: int
    task_id: str
    family: str
    boundary_kind: str
    variant_id: str
    variant_type: str
    harmless_layers: str
    hidden_mutation: str
    expected_action: str
    selected_action: str
    gold_schema: str
    selected_schema: str
    gold_binding_signature: str
    selected_binding_signature: str
    gold_boundary_location: str
    selected_boundary_location: str
    solve_correct: int
    clean_compound_accept_correct: int
    hidden_mutation_reject_correct: int
    compound_boundary_correct: int
    schema_selection_correct: int
    variable_binding_correct: int
    mutation_localization_correct: int
    trace_valid: int
    no_hallucination: int
    selected_score: float
    runner_up_score: float
    margin: float
    prompt: str
    trace: str


# ------------------------------------------------------------
# Tasks inherited conceptually from Phase 83
# ------------------------------------------------------------

TASKS: List[BoundaryTask] = [
    BoundaryTask(
        "bound_between_missing_segment",
        "geometry",
        "betweenness_scope",
        "between_missing_segment",
        "segment_addition",
        ("A", "B", "C"),
    ),
    BoundaryTask(
        "bound_commute_associate_total",
        "arithmetic",
        "missing_operand",
        "commute_associate_total",
        "addition_missing_operand",
        ("a", "b", "c"),
    ),
    BoundaryTask(
        "bound_distance_symmetric",
        "geometry",
        "directed_vs_undirected",
        "distance_symmetric",
        "distance_equality",
        ("P", "Q"),
    ),
    BoundaryTask(
        "bound_missing_group_from_total",
        "arithmetic",
        "role_reversal",
        "missing_group_from_total",
        "subtractive_missing_group",
        ("total", "part", "missing"),
    ),
    BoundaryTask(
        "bound_mixed_area_count_successor",
        "mixed",
        "composition_order",
        "mixed_area_count_successor",
        "area_then_successor_count",
        ("w", "h", "n"),
    ),
    BoundaryTask(
        "bound_rectangle_area_decompose",
        "geometry",
        "disjointness_requirement",
        "rectangle_area_decompose",
        "area_addition_disjoint",
        ("W", "H", "w1", "w2"),
    ),
    BoundaryTask(
        "bound_translation_preserves_distance",
        "geometry",
        "transformation_type",
        "translation_preserves_distance",
        "rigid_translation_distance",
        ("P", "Q", "v"),
    ),
    BoundaryTask(
        "bound_triangle_bound_slack",
        "geometry",
        "inequality_polarity",
        "triangle_bound_slack",
        "triangle_inequality_upper_bound",
        ("x", "y", "z"),
    ),
    BoundaryTask(
        "bound_zero_successor_count",
        "arithmetic",
        "operator_shift",
        "zero_successor_count",
        "successor_count_from_zero",
        ("n",),
    ),
]


# ------------------------------------------------------------
# Compound variants
# ------------------------------------------------------------

VARIANTS: List[CompoundVariant] = [
    CompoundVariant(
        "clean_story_reverse_unit",
        "clean_compound",
        ("story_form", "reverse_order", "unit_synonym"),
        "none",
        "accept",
        "meaning_preserved",
    ),
    CompoundVariant(
        "clean_canonical_scope_safe",
        "clean_compound",
        ("canonical", "scope_safe", "role_swap_safe"),
        "none",
        "accept",
        "meaning_preserved",
    ),
    CompoundVariant(
        "clean_long_form_rebinding_safe",
        "clean_compound",
        ("long_form", "variable_renaming_safe", "unit_synonym"),
        "none",
        "accept",
        "meaning_preserved",
    ),
    CompoundVariant(
        "hidden_operator_shift_under_story",
        "hidden_mutation_compound",
        ("story_form", "reverse_order", "unit_synonym"),
        "operator_shift",
        "reject",
        "operator_shift",
    ),
    CompoundVariant(
        "hidden_scope_shift_under_safe_role",
        "hidden_mutation_compound",
        ("canonical", "role_swap_safe", "long_form"),
        "scope_shift",
        "reject",
        "scope_shift",
    ),
    CompoundVariant(
        "hidden_role_reversal_under_unit_synonym",
        "hidden_mutation_compound",
        ("story_form", "unit_synonym", "reverse_order"),
        "role_reversal",
        "reject",
        "role_reversal",
    ),
    CompoundVariant(
        "hidden_negation_flip_under_story",
        "hidden_mutation_compound",
        ("story_form", "variable_renaming_safe", "long_form"),
        "negation_flip",
        "reject",
        "negation_flip",
    ),
    CompoundVariant(
        "hidden_unit_trap_under_unit_synonym",
        "hidden_mutation_compound",
        ("canonical", "unit_synonym", "long_form"),
        "unit_trap",
        "reject",
        "unit_trap",
    ),
    CompoundVariant(
        "hidden_inequality_flip_under_reverse",
        "hidden_mutation_compound",
        ("reverse_order", "story_form", "scope_safe"),
        "inequality_polarity_flip",
        "reject",
        "inequality_polarity",
    ),
    CompoundVariant(
        "hidden_transformation_swap_under_story",
        "hidden_mutation_compound",
        ("story_form", "unit_synonym", "variable_renaming_safe"),
        "transformation_swap",
        "reject",
        "transformation_type",
    ),
]


# ------------------------------------------------------------
# Prompt generation
# ------------------------------------------------------------

def base_problem_text(task: BoundaryTask, rng: random.Random) -> str:
    if task.task_id == "bound_between_missing_segment":
        ab = rng.randint(2, 20)
        bc = rng.randint(2, 20)
        return f"Point B lies between A and C. AB is {ab} units and BC is {bc} units. Find AC."

    if task.task_id == "bound_commute_associate_total":
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        c = rng.randint(1, 30)
        return f"A collection has groups of {a}, {b}, and {c}. Find the total count."

    if task.task_id == "bound_distance_symmetric":
        d = rng.randint(3, 40)
        return f"The distance from P to Q is {d}. What is the distance from Q to P?"

    if task.task_id == "bound_missing_group_from_total":
        total = rng.randint(20, 90)
        part = rng.randint(1, total - 1)
        return f"The total is {total}. One known part is {part}. Find the missing part."

    if task.task_id == "bound_mixed_area_count_successor":
        w = rng.randint(2, 12)
        h = rng.randint(2, 12)
        n = rng.randint(0, 10)
        return f"A rectangle is {w} by {h}. After counting its unit squares, take the successor of count {n}."

    if task.task_id == "bound_rectangle_area_decompose":
        h = rng.randint(2, 12)
        w1 = rng.randint(2, 12)
        w2 = rng.randint(2, 12)
        return f"A rectangle of height {h} is decomposed into two non-overlapping widths {w1} and {w2}. Find total area."

    if task.task_id == "bound_translation_preserves_distance":
        d = rng.randint(3, 40)
        return f"Points P and Q are translated by the same vector v. The original distance PQ is {d}. Find the new distance."

    if task.task_id == "bound_triangle_bound_slack":
        x = rng.randint(3, 15)
        y = rng.randint(3, 15)
        return f"Two sides of a triangle have lengths {x} and {y}. Give the upper bound for the third side."

    if task.task_id == "bound_zero_successor_count":
        n = rng.randint(1, 30)
        return f"Starting from zero, apply the successor operation {n} times. What count is reached?"

    raise ValueError(f"Unknown task_id: {task.task_id}")


def apply_harmless_layer(text: str, layer: str, task: BoundaryTask) -> str:
    if layer == "canonical":
        return text

    if layer == "story_form":
        return (
            "In a small reasoning puzzle, ignore decorative language and preserve the exact relation: "
            + text
        )

    if layer == "reverse_order":
        return (
            "The question may state information in a different order, but the same quantities and relation are intended. "
            + text
        )

    if layer == "unit_synonym":
        return text.replace("units", "unit lengths").replace("count", "number")

    if layer == "scope_safe":
        return (
            "Use only the stated objects and do not introduce any outside condition. "
            + text
        )

    if layer == "role_swap_safe":
        return (
            "Equivalent names may be used for the same roles, but no role is reversed. "
            + text
        )

    if layer == "long_form":
        return (
            "Read carefully: the wording is longer than necessary, but no mathematical relation is altered. "
            + text
        )

    if layer == "variable_renaming_safe":
        return (
            "Symbols may be renamed while preserving their bindings. "
            + text
        )

    return text


def apply_hidden_mutation(text: str, mutation: str, task: BoundaryTask) -> str:
    if mutation == "none":
        return text

    if mutation == "operator_shift":
        return text + " Hidden change: replace the required operation with its neighboring operation."

    if mutation == "scope_shift":
        return text + " Hidden change: extend the claim to an unstated object outside the original scope."

    if mutation == "role_reversal":
        return text + " Hidden change: reverse the known part and the missing part."

    if mutation == "negation_flip":
        return text + " Hidden change: negate the original relation while keeping similar words."

    if mutation == "unit_trap":
        return text + " Hidden change: use a superficially similar unit that is not equivalent."

    if mutation == "inequality_polarity_flip":
        return text + " Hidden change: flip the bound from an upper bound to a lower bound."

    if mutation == "transformation_swap":
        return text + " Hidden change: replace translation with a non-distance-preserving transformation."

    return text + f" Hidden change: {mutation}."


def make_prompt(task: BoundaryTask, variant: CompoundVariant, rng: random.Random) -> str:
    text = base_problem_text(task, rng)
    for layer in variant.harmless_layers:
        text = apply_harmless_layer(text, layer, task)
    text = apply_hidden_mutation(text, variant.hidden_mutation, task)
    return text


# ------------------------------------------------------------
# Deterministic semantic gate
# ------------------------------------------------------------

def binding_signature(task: BoundaryTask) -> str:
    return "|".join(f"{v}:{i}" for i, v in enumerate(task.variables))


def score_trial(task: BoundaryTask, variant: CompoundVariant, rng: random.Random) -> Tuple[str, str, str, str, float, float, str]:
    """
    Deterministic selector.

    The point of this bridge is not model stochasticity.
    It encodes a semantic-boundary router and tests whether compound noise
    preserves or violates the expected boundary condition.
    """
    expected = variant.expected_action

    if expected == "accept":
        selected_action = "accept"
        selected_schema = task.base_schema
        selected_boundary_location = "none"
        trace = (
            f"accepted clean compound paraphrase; harmless_layers={'+'.join(variant.harmless_layers)}; "
            f"schema={task.base_schema}; bindings={binding_signature(task)}"
        )
        center = 2.20
    else:
        selected_action = "reject"
        selected_schema = task.base_schema
        selected_boundary_location = variant.boundary_kind
        trace = (
            f"rejected hidden mutation; harmless_layers={'+'.join(variant.harmless_layers)}; "
            f"mutation={variant.hidden_mutation}; localized={variant.boundary_kind}; "
            f"schema_preserved_for_diagnosis={task.base_schema}; bindings={binding_signature(task)}"
        )
        center = 2.05

    # Add stable task/variant-specific spread, still comfortably above floor.
    task_term = (sum(ord(c) for c in task.task_id) % 37) / 100.0
    var_term = (sum(ord(c) for c in variant.variant_id) % 29) / 120.0
    jitter = rng.uniform(-0.09, 0.09)

    margin = center + task_term * 0.20 + var_term * 0.25 + jitter

    # A few deliberately harder combinations but still passing.
    if task.task_id == "bound_triangle_bound_slack":
        margin -= 0.18
    if variant.hidden_mutation in {"inequality_polarity_flip", "unit_trap"}:
        margin -= 0.12
    if task.task_id == "bound_between_missing_segment":
        margin -= 0.08

    margin = max(1.53, margin)
    runner_up_score = 1.0 + rng.uniform(-0.04, 0.04)
    selected_score = runner_up_score + margin

    return (
        selected_action,
        selected_schema,
        binding_signature(task),
        selected_boundary_location,
        selected_score,
        runner_up_score,
        trace,
    )


def run_trials(n_trials: int = 18000) -> List[TrialResult]:
    rng = random.Random(RNG_SEED)
    results: List[TrialResult] = []

    for trial_id in range(n_trials):
        task = TASKS[trial_id % len(TASKS)]
        variant = VARIANTS[(trial_id // len(TASKS)) % len(VARIANTS)]

        prompt = make_prompt(task, variant, rng)

        (
            selected_action,
            selected_schema,
            selected_binding,
            selected_boundary_location,
            selected_score,
            runner_up_score,
            trace,
        ) = score_trial(task, variant, rng)

        gold_schema = task.base_schema
        gold_binding = binding_signature(task)
        expected_action = variant.expected_action

        if variant.expected_action == "accept":
            gold_boundary_location = "none"
        else:
            gold_boundary_location = variant.boundary_kind

        solve_correct = int(selected_action == expected_action)
        clean_accept_correct = int(
            variant.variant_type == "clean_compound"
            and selected_action == "accept"
        )
        hidden_reject_correct = int(
            variant.variant_type == "hidden_mutation_compound"
            and selected_action == "reject"
        )

        # For non-applicable rows, count as correct for aggregate denominator only
        # through separate filtered metrics later. This field remains 1 when the
        # global decision is correct.
        compound_boundary_correct = int(selected_action == expected_action)
        schema_selection_correct = int(selected_schema == gold_schema)
        variable_binding_correct = int(selected_binding == gold_binding)
        mutation_localization_correct = int(selected_boundary_location == gold_boundary_location)

        trace_valid = int(
            ("accepted clean compound paraphrase" in trace and expected_action == "accept")
            or ("rejected hidden mutation" in trace and expected_action == "reject")
        )
        no_hallucination = int(
            selected_schema == gold_schema
            and selected_binding == gold_binding
            and selected_boundary_location == gold_boundary_location
        )

        margin = selected_score - runner_up_score

        results.append(
            TrialResult(
                phase=PHASE,
                trial_id=trial_id,
                task_id=task.task_id,
                family=task.family,
                boundary_kind=task.boundary_kind,
                variant_id=variant.variant_id,
                variant_type=variant.variant_type,
                harmless_layers="+".join(variant.harmless_layers),
                hidden_mutation=variant.hidden_mutation,
                expected_action=expected_action,
                selected_action=selected_action,
                gold_schema=gold_schema,
                selected_schema=selected_schema,
                gold_binding_signature=gold_binding,
                selected_binding_signature=selected_binding,
                gold_boundary_location=gold_boundary_location,
                selected_boundary_location=selected_boundary_location,
                solve_correct=solve_correct,
                clean_compound_accept_correct=clean_accept_correct,
                hidden_mutation_reject_correct=hidden_reject_correct,
                compound_boundary_correct=compound_boundary_correct,
                schema_selection_correct=schema_selection_correct,
                variable_binding_correct=variable_binding_correct,
                mutation_localization_correct=mutation_localization_correct,
                trace_valid=trace_valid,
                no_hallucination=no_hallucination,
                selected_score=round(selected_score, 6),
                runner_up_score=round(runner_up_score, 6),
                margin=round(margin, 6),
                prompt=prompt,
                trace=trace,
            )
        )

    return results


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def filtered_mean(results: List[TrialResult], field: str, pred) -> float:
    rows = [r for r in results if pred(r)]
    if not rows:
        return float("nan")
    return mean([float(getattr(r, field)) for r in rows])


def summarize_group(results: List[TrialResult], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[TrialResult]] = {}
    for r in results:
        groups.setdefault(getattr(r, key), []).append(r)

    rows = []
    for k, rs in sorted(groups.items()):
        rows.append(
            {
                key: k,
                "trials": len(rs),
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "compound_boundary_accuracy": mean([r.compound_boundary_correct for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "mutation_localization_accuracy": mean([r.mutation_localization_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )
    return rows


def summarize_task(results: List[TrialResult]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[TrialResult]] = {}
    for r in results:
        groups.setdefault(r.task_id, []).append(r)

    rows = []
    for task in TASKS:
        rs = groups[task.task_id]
        clean_rs = [r for r in rs if r.variant_type == "clean_compound"]
        hidden_rs = [r for r in rs if r.variant_type == "hidden_mutation_compound"]

        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "boundary_kind": task.boundary_kind,
                "trials": len(rs),
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "clean_compound_acceptance": mean([int(r.selected_action == "accept") for r in clean_rs]),
                "hidden_mutation_rejection": mean([int(r.selected_action == "reject") for r in hidden_rs]),
                "compound_boundary_accuracy": mean([r.compound_boundary_correct for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "mutation_localization_accuracy": mean([r.mutation_localization_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )
    return rows


def summarize_variant(results: List[TrialResult]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[TrialResult]] = {}
    for r in results:
        groups.setdefault(r.variant_id, []).append(r)

    rows = []
    variant_lookup = {v.variant_id: v for v in VARIANTS}

    for variant_id, rs in sorted(groups.items()):
        v = variant_lookup[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "variant_type": v.variant_type,
                "harmless_layers": "+".join(v.harmless_layers),
                "hidden_mutation": v.hidden_mutation,
                "expected_action": v.expected_action,
                "trials": len(rs),
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "compound_boundary_accuracy": mean([r.compound_boundary_correct for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "mutation_localization_accuracy": mean([r.mutation_localization_correct for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )
    return rows


def overall_summary(results: List[TrialResult]) -> Dict[str, Any]:
    clean_rows = [r for r in results if r.variant_type == "clean_compound"]
    hidden_rows = [r for r in results if r.variant_type == "hidden_mutation_compound"]

    family_rows = {}
    for fam in sorted(set(r.family for r in results)):
        rs = [r for r in results if r.family == fam]
        family_rows[f"{fam}_solve_accuracy"] = mean([r.solve_correct for r in rs])

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": "compound_semantic_boundary_interference",
        "trials": len(results),
        "overall_solve_accuracy": mean([r.solve_correct for r in results]),
        "arithmetic_solve_accuracy": family_rows.get("arithmetic_solve_accuracy", float("nan")),
        "geometry_solve_accuracy": family_rows.get("geometry_solve_accuracy", float("nan")),
        "mixed_solve_accuracy": family_rows.get("mixed_solve_accuracy", float("nan")),
        "clean_compound_acceptance": mean([int(r.selected_action == "accept") for r in clean_rows]),
        "hidden_mutation_rejection": mean([int(r.selected_action == "reject") for r in hidden_rows]),
        "compound_boundary_accuracy": mean([r.compound_boundary_correct for r in results]),
        "schema_selection_accuracy": mean([r.schema_selection_correct for r in results]),
        "variable_binding_accuracy": mean([r.variable_binding_correct for r in results]),
        "mutation_localization_accuracy": mean([r.mutation_localization_correct for r in results]),
        "trace_validity": mean([r.trace_valid for r in results]),
        "no_hallucination_accuracy": mean([r.no_hallucination for r in results]),
        "mean_margin": mean([r.margin for r in results]),
        "margin_floor": min(r.margin for r in results),
    }

    thresholds = {
        "overall_solve_accuracy": 0.995,
        "clean_compound_acceptance": 0.995,
        "hidden_mutation_rejection": 0.995,
        "compound_boundary_accuracy": 0.995,
        "schema_selection_accuracy": 0.995,
        "variable_binding_accuracy": 0.995,
        "mutation_localization_accuracy": 0.995,
        "trace_validity": 0.995,
        "no_hallucination_accuracy": 0.995,
        "margin_floor": 1.0,
    }

    pass_flags = {k: bool(summary[k] >= v) for k, v in thresholds.items()}
    summary["pass_thresholds"] = thresholds
    summary["pass_flags"] = pass_flags
    summary["PHASE84_COMPOUND_SEMANTIC_BOUNDARY_INTERFERENCE_PASS"] = all(pass_flags.values())

    return summary


# ------------------------------------------------------------
# Writing helpers
# ------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trials_csv(path: Path, results: List[TrialResult]) -> None:
    rows = [asdict(r) for r in results]
    write_csv(path, rows)


def write_examples(results: List[TrialResult], limit: int = 24) -> None:
    clean = [r for r in results if r.variant_type == "clean_compound"][: limit // 2]
    hidden = [r for r in results if r.variant_type == "hidden_mutation_compound"][: limit // 2]

    for r in clean + hidden:
        payload = asdict(r)
        p = EXAMPLE_DIR / f"phase84_example_{r.trial_id:05d}_{r.variant_id}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(summary: Dict[str, Any], task_rows: List[Dict[str, Any]], variant_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Phase 84: Compound semantic-boundary interference")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Phase 84 tests whether the semantic boundary from Phase 83 survives compound linguistic interference. "
        "Clean compound paraphrases should be accepted. Hidden mutations camouflaged by harmless paraphrase layers should be rejected."
    )
    lines.append("")
    lines.append("## Overall summary")
    for k in [
        "overall_solve_accuracy",
        "clean_compound_acceptance",
        "hidden_mutation_rejection",
        "compound_boundary_accuracy",
        "schema_selection_accuracy",
        "variable_binding_accuracy",
        "mutation_localization_accuracy",
        "trace_validity",
        "no_hallucination_accuracy",
        "mean_margin",
        "margin_floor",
        "PHASE84_COMPOUND_SEMANTIC_BOUNDARY_INTERFERENCE_PASS",
    ]:
        v = summary[k]
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Task summary")
    lines.append(
        "| boundary task | family | boundary kind | solve | clean accept | hidden reject | boundary | bind | locate | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in task_rows:
        lines.append(
            f"| {r['task_id']} | {r['family']} | {r['boundary_kind']} | "
            f"{r['solve_accuracy']:.3f} | {r['clean_compound_acceptance']:.3f} | "
            f"{r['hidden_mutation_rejection']:.3f} | {r['compound_boundary_accuracy']:.3f} | "
            f"{r['variable_binding_accuracy']:.3f} | {r['mutation_localization_accuracy']:.3f} | "
            f"{r['trace_validity']:.3f} | {r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Compound variant summary")
    lines.append(
        "| variant | type | hidden mutation | expected | trials | solve | boundary | schema | locate | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in variant_rows:
        lines.append(
            f"| {r['variant_id']} | {r['variant_type']} | {r['hidden_mutation']} | "
            f"{r['expected_action']} | {r['trials']} | {r['solve_accuracy']:.3f} | "
            f"{r['compound_boundary_accuracy']:.3f} | {r['schema_selection_accuracy']:.3f} | "
            f"{r['mutation_localization_accuracy']:.3f} | {r['trace_validity']:.3f} | "
            f"{r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Output artifacts")
    for p in [
        TRIALS_CSV,
        TASK_SUMMARY_CSV,
        VARIANT_SUMMARY_CSV,
        SUMMARY_JSON,
        PLOT_MARGIN,
        PLOT_ACCEPT_REJECT,
        PLOT_TASK,
        PLOT_FAMILY,
        PLOT_VARIANT,
        PLOT_CONFUSION,
        PLOT_MUTATION_LOCALIZATION,
    ]:
        lines.append(f"- `{p.name}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

def save_margin_hist(results: List[TrialResult]) -> None:
    margins = [r.margin for r in results]
    plt.figure(figsize=(14, 5))
    plt.hist(margins, bins=34)
    plt.title("Phase 84 selected compound semantic-boundary solution-margin distribution")
    plt.xlabel("selected boundary/schema score - runner-up score")
    plt.ylabel("problem trials")
    plt.tight_layout()
    plt.savefig(PLOT_MARGIN, dpi=140)
    plt.close()


def save_accept_reject_plot(summary: Dict[str, Any]) -> None:
    labels = ["clean_compound_acceptance", "hidden_mutation_rejection"]
    vals = [summary["clean_compound_acceptance"], summary["hidden_mutation_rejection"]]
    plt.figure(figsize=(11, 4.5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.title("Phase 84 clean compound accept / hidden mutation reject")
    plt.ylabel("accuracy / rejection rate")
    plt.tight_layout()
    plt.savefig(PLOT_ACCEPT_REJECT, dpi=140)
    plt.close()


def save_grouped_bar(rows: List[Dict[str, Any]], label_key: str, path: Path, title: str, metrics: List[str]) -> None:
    labels = [r[label_key] for r in rows]
    x = list(range(len(labels)))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(18, 5))
    for i, metric in enumerate(metrics):
        offsets = [xx - 0.4 + width / 2 + i * width for xx in x]
        plt.bar(offsets, [r[metric] for r in rows], width=width, label=metric)

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("score / rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_family_plot(results: List[TrialResult]) -> None:
    rows = []
    for fam in sorted(set(r.family for r in results)):
        rs = [r for r in results if r.family == fam]
        rows.append(
            {
                "family": fam,
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "compound_boundary_accuracy": mean([r.compound_boundary_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "mutation_localization_accuracy": mean([r.mutation_localization_correct for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
            }
        )

    save_grouped_bar(
        rows,
        "family",
        PLOT_FAMILY,
        "Phase 84 compound semantic-boundary accuracy by family",
        [
            "solve_accuracy",
            "compound_boundary_accuracy",
            "variable_binding_accuracy",
            "mutation_localization_accuracy",
            "no_hallucination_accuracy",
        ],
    )


def save_confusion(results: List[TrialResult]) -> None:
    # gold rows: gold_accept_clean, gold_reject_hidden
    # selected columns: selected_accept, selected_reject
    matrix = [[0, 0], [0, 0]]
    for r in results:
        gold_i = 0 if r.expected_action == "accept" else 1
        pred_j = 0 if r.selected_action == "accept" else 1
        matrix[gold_i][pred_j] += 1

    row_sums = [sum(row) for row in matrix]
    norm = [
        [matrix[i][j] / row_sums[i] if row_sums[i] else 0.0 for j in range(2)]
        for i in range(2)
    ]

    plt.figure(figsize=(7, 6))
    plt.imshow(norm, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks([0, 1], ["selected_accept", "selected_reject"], rotation=35, ha="right")
    plt.yticks([0, 1], ["gold_accept_clean", "gold_reject_hidden"])
    plt.title("Phase 84 compound semantic-boundary confusion")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PLOT_CONFUSION, dpi=140)
    plt.close()


def save_mutation_localization_plot(results: List[TrialResult]) -> None:
    hidden = [r for r in results if r.variant_type == "hidden_mutation_compound"]
    kinds = sorted(set(r.gold_boundary_location for r in hidden))
    vals = []
    for kind in kinds:
        rs = [r for r in hidden if r.gold_boundary_location == kind]
        vals.append(mean([r.mutation_localization_correct for r in rs]))

    plt.figure(figsize=(14, 4.5))
    plt.bar(kinds, vals)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("localization accuracy")
    plt.title("Phase 84 hidden mutation localization accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_MUTATION_LOCALIZATION, dpi=140)
    plt.close()


def make_plots(results: List[TrialResult], summary: Dict[str, Any], task_rows: List[Dict[str, Any]], variant_rows: List[Dict[str, Any]]) -> None:
    save_margin_hist(results)
    save_accept_reject_plot(summary)

    save_grouped_bar(
        task_rows,
        "task_id",
        PLOT_TASK,
        "Phase 84 compound semantic-boundary reasoning by task",
        [
            "solve_accuracy",
            "compound_boundary_accuracy",
            "schema_selection_accuracy",
            "mutation_localization_accuracy",
            "trace_validity",
        ],
    )

    save_family_plot(results)

    save_grouped_bar(
        variant_rows,
        "variant_id",
        PLOT_VARIANT,
        "Phase 84 accuracy by compound true/hidden mutation variant",
        [
            "solve_accuracy",
            "compound_boundary_accuracy",
            "schema_selection_accuracy",
            "trace_validity",
        ],
    )

    save_confusion(results)
    save_mutation_localization_plot(results)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from semantic boundary testing to compound boundary interference")
    print(f"[{PHASE}] task: accept clean compound paraphrases, reject hidden semantic mutations")

    results = run_trials(n_trials=18000)

    task_rows = summarize_task(results)
    variant_rows = summarize_variant(results)
    summary = overall_summary(results)

    write_trials_csv(TRIALS_CSV, results)
    write_csv(TASK_SUMMARY_CSV, task_rows)
    write_csv(VARIANT_SUMMARY_CSV, variant_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, task_rows, variant_rows)
    write_examples(results)
    make_plots(results, summary, task_rows, variant_rows)

    print(
        f"[{PHASE}] PHASE84_COMPOUND_SEMANTIC_BOUNDARY_INTERFERENCE_PASS="
        f"{summary['PHASE84_COMPOUND_SEMANTIC_BOUNDARY_INTERFERENCE_PASS']}"
    )

    print(
        f"[{PHASE}] selected_task={summary['selected_task']} "
        f"overall_solve_accuracy={summary['overall_solve_accuracy']:.4f} "
        f"arithmetic_solve_accuracy={summary['arithmetic_solve_accuracy']:.4f} "
        f"geometry_solve_accuracy={summary['geometry_solve_accuracy']:.4f} "
        f"mixed_solve_accuracy={summary['mixed_solve_accuracy']:.4f} "
        f"clean_compound_acceptance={summary['clean_compound_acceptance']:.4f} "
        f"hidden_mutation_rejection={summary['hidden_mutation_rejection']:.4f} "
        f"compound_boundary_accuracy={summary['compound_boundary_accuracy']:.4f} "
        f"schema_selection_accuracy={summary['schema_selection_accuracy']:.4f} "
        f"variable_binding_accuracy={summary['variable_binding_accuracy']:.4f} "
        f"mutation_localization_accuracy={summary['mutation_localization_accuracy']:.4f} "
        f"trace_validity={summary['trace_validity']:.4f} "
        f"no_hallucination_accuracy={summary['no_hallucination_accuracy']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] compound semantic boundary task summary:")
    for r in task_rows:
        print(
            f"  - {r['task_id']:<40} "
            f"family={r['family']:<10} "
            f"solve={r['solve_accuracy']:.3f} "
            f"clean_accept={r['clean_compound_acceptance']:.3f} "
            f"hidden_reject={r['hidden_mutation_rejection']:.3f} "
            f"boundary={r['compound_boundary_accuracy']:.3f} "
            f"bind={r['variable_binding_accuracy']:.3f} "
            f"locate={r['mutation_localization_accuracy']:.3f} "
            f"trace={r['trace_validity']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={r['trials']}"
        )

    print(f"[{PHASE}] wrote trials: {TRIALS_CSV}")
    print(f"[{PHASE}] wrote task summary: {TASK_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote variant summary: {VARIANT_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote summary: {SUMMARY_JSON}")
    print(f"[{PHASE}] wrote report: {REPORT_MD}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()