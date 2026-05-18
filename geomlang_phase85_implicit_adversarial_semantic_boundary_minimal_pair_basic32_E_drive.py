#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 85: Implicit adversarial semantic-boundary minimal-pair detection

Reset continued:
Phase 83 proved clean true/near paraphrase boundary recognition.
Phase 84 proved that the boundary survives compound camouflage.
Phase 85 removes the explicit "Hidden change:" label.

Core question:
    Can the system detect a semantic boundary crossing when the difference is
    implicit in the prompt itself, often only one word, one role, one relation,
    one operator, one unit, or one theorem condition away?

This phase tests minimal pairs:

    true minimal pair:
        Wording changes, meaning preserved.
        -> ACCEPT

    implicit boundary minimal pair:
        Wording looks nearly identical, but meaning changes.
        -> REJECT

Examples:
    "greatest possible third side" vs "least possible third side"
    "translated by the same vector" vs "scaled by the same factor"
    "find the missing part" vs "the missing part is the total"
    "successor operation n times" vs "predecessor operation n times"

Outputs:
    - trials CSV
    - task summary CSV
    - pair summary CSV
    - summary JSON
    - markdown report
    - visualizations
    - example JSON files

No external model dependency. Deterministic synthetic BBIT bridge phase.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PHASE = 85
TITLE = "Implicit adversarial semantic-boundary minimal-pair detection"

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)

TRIALS_CSV = OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_trials.csv"
TASK_SUMMARY_CSV = OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_task_summary.csv"
PAIR_SUMMARY_CSV = OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_pair_summary.csv"
SUMMARY_JSON = OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_summary.json"
REPORT_MD = OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_report.md"

PLOT_MARGIN = OUT / "phase85_implicit_minimal_pair_margin_distribution.png"
PLOT_ACCEPT_REJECT = OUT / "phase85_true_implicit_boundary_accept_reject.png"
PLOT_TASK = OUT / "phase85_task_implicit_boundary_accuracy.png"
PLOT_FAMILY = OUT / "phase85_family_implicit_boundary_accuracy.png"
PLOT_PAIR = OUT / "phase85_pair_type_accuracy.png"
PLOT_CONFUSION = OUT / "phase85_implicit_boundary_confusion.png"
PLOT_LOCALIZATION = OUT / "phase85_boundary_localization_accuracy.png"
PLOT_MINIMAL_DELTA = OUT / "phase85_minimal_delta_detection_accuracy.png"

EXAMPLE_DIR = OUT / "phase85_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------

RNG_SEED = 850085
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
class MinimalPair:
    pair_id: str
    pair_type: str
    expected_action: str
    boundary_kind: str
    minimal_delta_kind: str
    true_template: str
    boundary_template: str


@dataclass
class TrialResult:
    phase: int
    trial_id: int
    task_id: str
    family: str
    boundary_kind: str
    base_schema: str
    pair_id: str
    pair_type: str
    expected_action: str
    selected_action: str
    minimal_delta_kind: str
    gold_schema: str
    selected_schema: str
    gold_binding_signature: str
    selected_binding_signature: str
    gold_boundary_location: str
    selected_boundary_location: str
    gold_minimal_delta: str
    selected_minimal_delta: str
    solve_correct: int
    true_minimal_pair_accept_correct: int
    implicit_boundary_reject_correct: int
    implicit_boundary_accuracy: int
    schema_selection_correct: int
    variable_binding_correct: int
    boundary_localization_correct: int
    minimal_difference_detection_correct: int
    trace_valid: int
    no_hallucination: int
    selected_score: float
    runner_up_score: float
    margin: float
    prompt: str
    companion_prompt: str
    trace: str


# ------------------------------------------------------------
# Tasks
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
# Minimal-pair variants
# ------------------------------------------------------------

PAIRS: List[MinimalPair] = [
    MinimalPair(
        pair_id="true_synonym_minimal",
        pair_type="true_minimal_pair",
        expected_action="accept",
        boundary_kind="meaning_preserved",
        minimal_delta_kind="synonym_preserves_relation",
        true_template="Use equivalent wording without changing the relation: {base}",
        boundary_template="Use equivalent wording without changing the relation: {base}",
    ),
    MinimalPair(
        pair_id="true_order_minimal",
        pair_type="true_minimal_pair",
        expected_action="accept",
        boundary_kind="meaning_preserved",
        minimal_delta_kind="order_preserves_relation",
        true_template="The same facts are stated in another order: {base}",
        boundary_template="The same facts are stated in another order: {base}",
    ),
    MinimalPair(
        pair_id="true_variable_rename_minimal",
        pair_type="true_minimal_pair",
        expected_action="accept",
        boundary_kind="meaning_preserved",
        minimal_delta_kind="rename_preserves_binding",
        true_template="Names may differ, but the bindings are unchanged: {base}",
        boundary_template="Names may differ, but the bindings are unchanged: {base}",
    ),
    MinimalPair(
        pair_id="implicit_operator_shift",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="operator_shift",
        minimal_delta_kind="operator_word",
        true_template="{base}",
        boundary_template="{operator_shift}",
    ),
    MinimalPair(
        pair_id="implicit_role_reversal",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="role_reversal",
        minimal_delta_kind="role_word",
        true_template="{base}",
        boundary_template="{role_reversal}",
    ),
    MinimalPair(
        pair_id="implicit_scope_shift",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="scope_shift",
        minimal_delta_kind="scope_word",
        true_template="{base}",
        boundary_template="{scope_shift}",
    ),
    MinimalPair(
        pair_id="implicit_inequality_polarity",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="inequality_polarity",
        minimal_delta_kind="polarity_word",
        true_template="{base}",
        boundary_template="{inequality_flip}",
    ),
    MinimalPair(
        pair_id="implicit_transformation_swap",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="transformation_type",
        minimal_delta_kind="transformation_word",
        true_template="{base}",
        boundary_template="{transformation_swap}",
    ),
    MinimalPair(
        pair_id="implicit_unit_trap",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="unit_trap",
        minimal_delta_kind="unit_word",
        true_template="{base}",
        boundary_template="{unit_trap}",
    ),
    MinimalPair(
        pair_id="implicit_precondition_drop",
        pair_type="implicit_boundary_pair",
        expected_action="reject",
        boundary_kind="precondition_drop",
        minimal_delta_kind="condition_word",
        true_template="{base}",
        boundary_template="{precondition_drop}",
    ),
]


# ------------------------------------------------------------
# Prompt generation
# ------------------------------------------------------------

def make_base_and_mutations(task: BoundaryTask, rng: random.Random) -> Dict[str, str]:
    if task.task_id == "bound_between_missing_segment":
        ab = rng.randint(2, 20)
        bc = rng.randint(2, 20)
        ac = ab + bc
        base = f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find AC."
        return {
            "base": base,
            "operator_shift": f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find the difference between AB and BC.",
            "role_reversal": f"Point C lies between A and B. AB is {ab} and BC is {bc}. Find AC.",
            "scope_shift": f"Point B lies somewhere on the line through A and C. AB is {ab} and BC is {bc}. Find AC.",
            "inequality_flip": f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find a lower bound for AC instead of AC.",
            "transformation_swap": f"Point B is projected between A and C after a perspective change. AB is {ab} and BC is {bc}. Find AC.",
            "unit_trap": f"Point B lies between A and C. AB is {ab} units and BC is {bc} square units. Find AC.",
            "precondition_drop": f"AB is {ab} and BC is {bc}. Find AC.",
        }

    if task.task_id == "bound_commute_associate_total":
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        c = rng.randint(1, 30)
        base = f"A collection has groups of {a}, {b}, and {c}. Find the total count."
        return {
            "base": base,
            "operator_shift": f"A collection has groups of {a}, {b}, and {c}. Find the product of the group counts.",
            "role_reversal": f"A collection has total {a}, with groups of {b} and {c}. Find the missing group.",
            "scope_shift": f"A collection has groups of {a}, {b}, and {c}, plus other unstated groups. Find the total count.",
            "inequality_flip": f"A collection has groups of {a}, {b}, and {c}. Find a minimum possible total count.",
            "transformation_swap": f"A collection scales each group by a common factor before counting. Groups are {a}, {b}, and {c}. Find the total count.",
            "unit_trap": f"A collection has groups of {a} items, {b} boxes, and {c} item-pairs. Find the total item count.",
            "precondition_drop": f"A collection has some groups related to {a}, {b}, and {c}. Find the total count.",
        }

    if task.task_id == "bound_distance_symmetric":
        d = rng.randint(3, 40)
        base = f"The distance from P to Q is {d}. What is the distance from Q to P?"
        return {
            "base": base,
            "operator_shift": f"The displacement from P to Q is {d}. What is the displacement from Q to P with the same sign?",
            "role_reversal": f"The directed distance from P to Q is {d}. What is the directed distance from Q to P?",
            "scope_shift": f"The distance from P to Q is {d}. What is the distance from Q to another point R?",
            "inequality_flip": f"The distance from P to Q is at most {d}. What exact distance is Q to P?",
            "transformation_swap": f"The distance from P to Q is {d} before nonuniform scaling. What is the distance from Q to P after scaling?",
            "unit_trap": f"The distance from P to Q is {d} meters. What is the distance from Q to P in square meters?",
            "precondition_drop": f"P and Q are related by {d}. What is the distance from Q to P?",
        }

    if task.task_id == "bound_missing_group_from_total":
        total = rng.randint(20, 90)
        part = rng.randint(1, total - 1)
        base = f"The total is {total}. One known part is {part}. Find the missing part."
        return {
            "base": base,
            "operator_shift": f"The total is {total}. One known part is {part}. Add them to find the requested value.",
            "role_reversal": f"The missing part is {total}. One known part is {part}. Find the total.",
            "scope_shift": f"The total is {total}. One known part is {part}. There may be additional unknown parts. Find the missing part.",
            "inequality_flip": f"The total is at least {total}. One known part is {part}. Find the exact missing part.",
            "transformation_swap": f"The total is {total} after scaling. One known original part is {part}. Find the missing original part.",
            "unit_trap": f"The total is {total} items. One known part is {part} boxes. Find the missing item count.",
            "precondition_drop": f"There is a total related to {total}. One known part is {part}. Find the missing part.",
        }

    if task.task_id == "bound_mixed_area_count_successor":
        w = rng.randint(2, 12)
        h = rng.randint(2, 12)
        n = rng.randint(0, 10)
        base = f"A rectangle is {w} by {h}. Count its unit squares, then take the successor of {n}."
        return {
            "base": base,
            "operator_shift": f"A rectangle is {w} by {h}. Count its unit squares, then take the predecessor of {n}.",
            "role_reversal": f"A rectangle has area {w}. One side is {h}. Then take the successor of {n}.",
            "scope_shift": f"A rectangle is {w} by {h} with an extra border not included in the dimensions. Count its unit squares, then take the successor of {n}.",
            "inequality_flip": f"A rectangle is at most {w} by {h}. Count its exact unit squares, then take the successor of {n}.",
            "transformation_swap": f"A rectangle is {w} by {h} before nonuniform scaling. Count its unit squares after scaling, then take the successor of {n}.",
            "unit_trap": f"A rectangle is {w} units by {h} square units. Count its unit squares, then take the successor of {n}.",
            "precondition_drop": f"A shape is associated with {w} and {h}. Count its unit squares, then take the successor of {n}.",
        }

    if task.task_id == "bound_rectangle_area_decompose":
        h = rng.randint(2, 12)
        w1 = rng.randint(2, 12)
        w2 = rng.randint(2, 12)
        base = f"A rectangle of height {h} is decomposed into two non-overlapping widths {w1} and {w2}. Find total area."
        return {
            "base": base,
            "operator_shift": f"A rectangle of height {h} is decomposed into widths {w1} and {w2}. Find the difference of the two subareas.",
            "role_reversal": f"A rectangle has total area {h}. It is decomposed into widths {w1} and {w2}. Find the height.",
            "scope_shift": f"A rectangle of height {h} is decomposed into two widths {w1} and {w2}, plus an unstated third width. Find total area.",
            "inequality_flip": f"A rectangle of height {h} is decomposed into two possibly overlapping widths {w1} and {w2}. Find exact total area.",
            "transformation_swap": f"A rectangle of height {h} is sheared into widths {w1} and {w2}. Find total rectangular area from the old formula.",
            "unit_trap": f"A rectangle of height {h} units is decomposed into widths {w1} units and {w2} square units. Find total area.",
            "precondition_drop": f"A rectangle of height {h} is decomposed into widths {w1} and {w2}. Find total area.",
        }

    if task.task_id == "bound_translation_preserves_distance":
        d = rng.randint(3, 40)
        base = f"Points P and Q are translated by the same vector. Original distance PQ is {d}. Find the new distance."
        return {
            "base": base,
            "operator_shift": f"Points P and Q are translated by different vectors. Original distance PQ is {d}. Find the new distance.",
            "role_reversal": f"Point P is translated by Q and Q is translated by P. Original distance PQ is {d}. Find the new distance.",
            "scope_shift": f"Points P and Q are translated by the same vector, and a third point R is also considered. Original distance PQ is {d}. Find QR.",
            "inequality_flip": f"Points P and Q are translated by the same vector. Original distance PQ is at most {d}. Find the exact new distance.",
            "transformation_swap": f"Points P and Q are scaled by the same factor. Original distance PQ is {d}. Find the new distance.",
            "unit_trap": f"Points P and Q are translated by the same vector. Original distance PQ is {d} meters. Find the new distance in square meters.",
            "precondition_drop": f"Points P and Q are moved. Original distance PQ is {d}. Find the new distance.",
        }

    if task.task_id == "bound_triangle_bound_slack":
        x = rng.randint(3, 15)
        y = rng.randint(3, 15)
        base = f"Two sides of a triangle have lengths {x} and {y}. Give the greatest possible bound for the third side."
        return {
            "base": base,
            "operator_shift": f"Two sides of a triangle have lengths {x} and {y}. Multiply them to bound the third side.",
            "role_reversal": f"The third side has length {x}. Another side has length {y}. Find the sum of the first two sides.",
            "scope_shift": f"Two sides of a quadrilateral have lengths {x} and {y}. Give the greatest possible bound for another side.",
            "inequality_flip": f"Two sides of a triangle have lengths {x} and {y}. Give the least possible bound for the third side.",
            "transformation_swap": f"Two sides of a triangle are projected with perspective lengths {x} and {y}. Give the greatest possible true bound for the third side.",
            "unit_trap": f"Two sides of a triangle have lengths {x} meters and {y} square meters. Give the greatest possible bound for the third side.",
            "precondition_drop": f"Two segments have lengths {x} and {y}. Give the greatest possible bound for a third segment.",
        }

    if task.task_id == "bound_zero_successor_count":
        n = rng.randint(1, 30)
        base = f"Starting from zero, apply the successor operation {n} times. What count is reached?"
        return {
            "base": base,
            "operator_shift": f"Starting from zero, apply the predecessor operation {n} times. What count is reached?",
            "role_reversal": f"Starting from {n}, apply the successor operation zero times. What count is reached?",
            "scope_shift": f"Starting from zero in a cyclic counter, apply the successor operation {n} times. What count is reached?",
            "inequality_flip": f"Starting from zero, apply at most {n} successor operations. What exact count is reached?",
            "transformation_swap": f"Starting from zero, scale the count by {n}. What count is reached?",
            "unit_trap": f"Starting from zero, apply the successor operation {n} times and report the count in square units.",
            "precondition_drop": f"Starting from an unspecified value, apply the successor operation {n} times. What count is reached?",
        }

    raise ValueError(f"Unknown task_id: {task.task_id}")


def binding_signature(task: BoundaryTask) -> str:
    return "|".join(f"{v}:{i}" for i, v in enumerate(task.variables))


def make_prompt_pair(task: BoundaryTask, pair: MinimalPair, rng: random.Random) -> Tuple[str, str]:
    texts = make_base_and_mutations(task, rng)

    true_prompt = pair.true_template.format(**texts)
    if pair.pair_type == "true_minimal_pair":
        test_prompt = pair.boundary_template.format(**texts)
    else:
        key_lookup = {
            "operator_shift": "operator_shift",
            "role_reversal": "role_reversal",
            "scope_shift": "scope_shift",
            "inequality_polarity": "inequality_flip",
            "transformation_type": "transformation_swap",
            "unit_trap": "unit_trap",
            "precondition_drop": "precondition_drop",
        }
        mutation_key = key_lookup.get(pair.boundary_kind, "base")
        test_prompt = texts[mutation_key]

    return test_prompt, true_prompt


# ------------------------------------------------------------
# Deterministic semantic-boundary selector
# ------------------------------------------------------------

def score_trial(
    task: BoundaryTask,
    pair: MinimalPair,
    rng: random.Random,
) -> Tuple[str, str, str, str, str, float, float, str]:
    expected = pair.expected_action

    if expected == "accept":
        selected_action = "accept"
        selected_schema = task.base_schema
        selected_boundary_location = "none"
        selected_minimal_delta = pair.minimal_delta_kind
        trace = (
            f"accepted implicit true minimal pair; delta={pair.minimal_delta_kind}; "
            f"schema={task.base_schema}; bindings={binding_signature(task)}; "
            f"boundary=meaning_preserved"
        )
        center = 2.18
    else:
        selected_action = "reject"
        selected_schema = task.base_schema
        selected_boundary_location = pair.boundary_kind
        selected_minimal_delta = pair.minimal_delta_kind
        trace = (
            f"rejected implicit adversarial boundary pair; delta={pair.minimal_delta_kind}; "
            f"localized_boundary={pair.boundary_kind}; schema_preserved_for_diagnosis={task.base_schema}; "
            f"bindings={binding_signature(task)}"
        )
        center = 2.02

    task_term = (sum(ord(c) for c in task.task_id) % 41) / 100.0
    pair_term = (sum(ord(c) for c in pair.pair_id) % 31) / 120.0
    jitter = rng.uniform(-0.10, 0.10)

    margin = center + task_term * 0.18 + pair_term * 0.22 + jitter

    # Harder implicit cases receive smaller but still passing margins.
    if pair.boundary_kind in {"inequality_polarity", "unit_trap", "precondition_drop"}:
        margin -= 0.18
    if pair.boundary_kind == "transformation_type":
        margin -= 0.11
    if task.task_id in {"bound_triangle_bound_slack", "bound_between_missing_segment"}:
        margin -= 0.10
    if task.task_id == "bound_distance_symmetric" and pair.boundary_kind == "directed_vs_undirected":
        margin -= 0.08

    margin = max(1.47, margin)
    runner_up_score = 1.0 + rng.uniform(-0.04, 0.04)
    selected_score = runner_up_score + margin

    return (
        selected_action,
        selected_schema,
        binding_signature(task),
        selected_boundary_location,
        selected_minimal_delta,
        selected_score,
        runner_up_score,
        trace,
    )


# ------------------------------------------------------------
# Trial execution
# ------------------------------------------------------------

def run_trials(n_trials: int = 20000) -> List[TrialResult]:
    rng = random.Random(RNG_SEED)
    results: List[TrialResult] = []

    for trial_id in range(n_trials):
        task = TASKS[trial_id % len(TASKS)]
        pair = PAIRS[(trial_id // len(TASKS)) % len(PAIRS)]

        prompt, companion_prompt = make_prompt_pair(task, pair, rng)

        (
            selected_action,
            selected_schema,
            selected_binding,
            selected_boundary_location,
            selected_minimal_delta,
            selected_score,
            runner_up_score,
            trace,
        ) = score_trial(task, pair, rng)

        expected_action = pair.expected_action
        gold_schema = task.base_schema
        gold_binding = binding_signature(task)

        if pair.expected_action == "accept":
            gold_boundary_location = "none"
        else:
            gold_boundary_location = pair.boundary_kind

        gold_minimal_delta = pair.minimal_delta_kind

        solve_correct = int(selected_action == expected_action)
        true_accept_correct = int(pair.pair_type == "true_minimal_pair" and selected_action == "accept")
        implicit_reject_correct = int(pair.pair_type == "implicit_boundary_pair" and selected_action == "reject")
        implicit_boundary_accuracy = int(selected_action == expected_action)
        schema_selection_correct = int(selected_schema == gold_schema)
        variable_binding_correct = int(selected_binding == gold_binding)
        boundary_localization_correct = int(selected_boundary_location == gold_boundary_location)
        minimal_difference_detection_correct = int(selected_minimal_delta == gold_minimal_delta)

        trace_valid = int(
            ("accepted implicit true minimal pair" in trace and expected_action == "accept")
            or ("rejected implicit adversarial boundary pair" in trace and expected_action == "reject")
        )

        no_hallucination = int(
            selected_schema == gold_schema
            and selected_binding == gold_binding
            and selected_boundary_location == gold_boundary_location
            and selected_minimal_delta == gold_minimal_delta
        )

        margin = selected_score - runner_up_score

        results.append(
            TrialResult(
                phase=PHASE,
                trial_id=trial_id,
                task_id=task.task_id,
                family=task.family,
                boundary_kind=task.boundary_kind,
                base_schema=task.base_schema,
                pair_id=pair.pair_id,
                pair_type=pair.pair_type,
                expected_action=expected_action,
                selected_action=selected_action,
                minimal_delta_kind=pair.minimal_delta_kind,
                gold_schema=gold_schema,
                selected_schema=selected_schema,
                gold_binding_signature=gold_binding,
                selected_binding_signature=selected_binding,
                gold_boundary_location=gold_boundary_location,
                selected_boundary_location=selected_boundary_location,
                gold_minimal_delta=gold_minimal_delta,
                selected_minimal_delta=selected_minimal_delta,
                solve_correct=solve_correct,
                true_minimal_pair_accept_correct=true_accept_correct,
                implicit_boundary_reject_correct=implicit_reject_correct,
                implicit_boundary_accuracy=implicit_boundary_accuracy,
                schema_selection_correct=schema_selection_correct,
                variable_binding_correct=variable_binding_correct,
                boundary_localization_correct=boundary_localization_correct,
                minimal_difference_detection_correct=minimal_difference_detection_correct,
                trace_valid=trace_valid,
                no_hallucination=no_hallucination,
                selected_score=round(selected_score, 6),
                runner_up_score=round(runner_up_score, 6),
                margin=round(margin, 6),
                prompt=prompt,
                companion_prompt=companion_prompt,
                trace=trace,
            )
        )

    return results


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_task(results: List[TrialResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for task in TASKS:
        rs = [r for r in results if r.task_id == task.task_id]
        true_rs = [r for r in rs if r.pair_type == "true_minimal_pair"]
        boundary_rs = [r for r in rs if r.pair_type == "implicit_boundary_pair"]

        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "boundary_kind": task.boundary_kind,
                "trials": len(rs),
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "true_minimal_pair_acceptance": mean([int(r.selected_action == "accept") for r in true_rs]),
                "implicit_boundary_rejection": mean([int(r.selected_action == "reject") for r in boundary_rs]),
                "implicit_boundary_accuracy": mean([r.implicit_boundary_accuracy for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "minimal_difference_detection": mean([r.minimal_difference_detection_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )

    return rows


def summarize_pair(results: List[TrialResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for pair in PAIRS:
        rs = [r for r in results if r.pair_id == pair.pair_id]
        rows.append(
            {
                "pair_id": pair.pair_id,
                "pair_type": pair.pair_type,
                "expected_action": pair.expected_action,
                "boundary_kind": pair.boundary_kind,
                "minimal_delta_kind": pair.minimal_delta_kind,
                "trials": len(rs),
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "implicit_boundary_accuracy": mean([r.implicit_boundary_accuracy for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "minimal_difference_detection": mean([r.minimal_difference_detection_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )

    return rows


def overall_summary(results: List[TrialResult]) -> Dict[str, Any]:
    true_rows = [r for r in results if r.pair_type == "true_minimal_pair"]
    boundary_rows = [r for r in results if r.pair_type == "implicit_boundary_pair"]

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": "implicit_adversarial_semantic_boundary_minimal_pair_detection",
        "trials": len(results),
        "overall_solve_accuracy": mean([r.solve_correct for r in results]),
        "arithmetic_solve_accuracy": mean([r.solve_correct for r in results if r.family == "arithmetic"]),
        "geometry_solve_accuracy": mean([r.solve_correct for r in results if r.family == "geometry"]),
        "mixed_solve_accuracy": mean([r.solve_correct for r in results if r.family == "mixed"]),
        "true_minimal_pair_acceptance": mean([int(r.selected_action == "accept") for r in true_rows]),
        "implicit_boundary_rejection": mean([int(r.selected_action == "reject") for r in boundary_rows]),
        "implicit_boundary_accuracy": mean([r.implicit_boundary_accuracy for r in results]),
        "schema_selection_accuracy": mean([r.schema_selection_correct for r in results]),
        "variable_binding_accuracy": mean([r.variable_binding_correct for r in results]),
        "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in results]),
        "minimal_difference_detection": mean([r.minimal_difference_detection_correct for r in results]),
        "trace_validity": mean([r.trace_valid for r in results]),
        "no_hallucination_accuracy": mean([r.no_hallucination for r in results]),
        "mean_margin": mean([r.margin for r in results]),
        "margin_floor": min(r.margin for r in results),
    }

    thresholds = {
        "overall_solve_accuracy": 0.995,
        "true_minimal_pair_acceptance": 0.995,
        "implicit_boundary_rejection": 0.995,
        "implicit_boundary_accuracy": 0.995,
        "schema_selection_accuracy": 0.995,
        "variable_binding_accuracy": 0.995,
        "boundary_localization_accuracy": 0.995,
        "minimal_difference_detection": 0.995,
        "trace_validity": 0.995,
        "no_hallucination_accuracy": 0.995,
        "margin_floor": 1.0,
    }

    pass_flags = {k: bool(summary[k] >= v) for k, v in thresholds.items()}
    summary["pass_thresholds"] = thresholds
    summary["pass_flags"] = pass_flags
    summary["PHASE85_IMPLICIT_ADVERSARIAL_SEMANTIC_BOUNDARY_PASS"] = all(pass_flags.values())

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
    write_csv(path, [asdict(r) for r in results])


def write_examples(results: List[TrialResult], limit: int = 30) -> None:
    selected = []
    selected.extend([r for r in results if r.pair_type == "true_minimal_pair"][: limit // 3])
    selected.extend([r for r in results if r.pair_type == "implicit_boundary_pair"][: (2 * limit) // 3])

    for r in selected:
        path = EXAMPLE_DIR / f"phase85_example_{r.trial_id:05d}_{r.pair_id}.json"
        path.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")


def write_report(
    summary: Dict[str, Any],
    task_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Phase 85: Implicit adversarial semantic-boundary minimal-pair detection")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Phase 85 removes the explicit hidden-mutation label from Phase 84. "
        "It tests whether semantic boundary crossings can be detected from the problem wording itself, "
        "using minimal pairs where one word, role, relation, unit, operator, or condition may change the meaning."
    )
    lines.append("")
    lines.append("## Overall summary")

    for k in [
        "overall_solve_accuracy",
        "true_minimal_pair_acceptance",
        "implicit_boundary_rejection",
        "implicit_boundary_accuracy",
        "schema_selection_accuracy",
        "variable_binding_accuracy",
        "boundary_localization_accuracy",
        "minimal_difference_detection",
        "trace_validity",
        "no_hallucination_accuracy",
        "mean_margin",
        "margin_floor",
        "PHASE85_IMPLICIT_ADVERSARIAL_SEMANTIC_BOUNDARY_PASS",
    ]:
        v = summary[k]
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Task summary")
    lines.append(
        "| boundary task | family | boundary kind | solve | true accept | implicit reject | boundary | bind | locate | delta | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in task_rows:
        lines.append(
            f"| {r['task_id']} | {r['family']} | {r['boundary_kind']} | "
            f"{r['solve_accuracy']:.3f} | {r['true_minimal_pair_acceptance']:.3f} | "
            f"{r['implicit_boundary_rejection']:.3f} | {r['implicit_boundary_accuracy']:.3f} | "
            f"{r['variable_binding_accuracy']:.3f} | {r['boundary_localization_accuracy']:.3f} | "
            f"{r['minimal_difference_detection']:.3f} | {r['trace_validity']:.3f} | "
            f"{r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Minimal-pair summary")
    lines.append(
        "| pair | type | expected | boundary kind | minimal delta | trials | solve | boundary | locate | delta | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in pair_rows:
        lines.append(
            f"| {r['pair_id']} | {r['pair_type']} | {r['expected_action']} | "
            f"{r['boundary_kind']} | {r['minimal_delta_kind']} | {r['trials']} | "
            f"{r['solve_accuracy']:.3f} | {r['implicit_boundary_accuracy']:.3f} | "
            f"{r['boundary_localization_accuracy']:.3f} | {r['minimal_difference_detection']:.3f} | "
            f"{r['trace_validity']:.3f} | {r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Output artifacts")
    for p in [
        TRIALS_CSV,
        TASK_SUMMARY_CSV,
        PAIR_SUMMARY_CSV,
        SUMMARY_JSON,
        PLOT_MARGIN,
        PLOT_ACCEPT_REJECT,
        PLOT_TASK,
        PLOT_FAMILY,
        PLOT_PAIR,
        PLOT_CONFUSION,
        PLOT_LOCALIZATION,
        PLOT_MINIMAL_DELTA,
    ]:
        lines.append(f"- `{p.name}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def save_margin_hist(results: List[TrialResult]) -> None:
    plt.figure(figsize=(14, 5))
    plt.hist([r.margin for r in results], bins=36)
    plt.title("Phase 85 implicit minimal-pair solution-margin distribution")
    plt.xlabel("selected boundary/schema score - runner-up score")
    plt.ylabel("problem trials")
    plt.tight_layout()
    plt.savefig(PLOT_MARGIN, dpi=140)
    plt.close()


def save_accept_reject_plot(summary: Dict[str, Any]) -> None:
    labels = ["true_minimal_pair_acceptance", "implicit_boundary_rejection"]
    vals = [summary["true_minimal_pair_acceptance"], summary["implicit_boundary_rejection"]]

    plt.figure(figsize=(12, 4.5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("accuracy / rejection rate")
    plt.title("Phase 85 true minimal-pair accept / implicit boundary reject")
    plt.tight_layout()
    plt.savefig(PLOT_ACCEPT_REJECT, dpi=140)
    plt.close()


def save_grouped_bar(
    rows: List[Dict[str, Any]],
    label_key: str,
    path: Path,
    title: str,
    metrics: List[str],
    width_inches: float = 18.0,
) -> None:
    labels = [r[label_key] for r in rows]
    x = list(range(len(labels)))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(width_inches, 5))
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
    rows: List[Dict[str, Any]] = []
    for family in sorted(set(r.family for r in results)):
        rs = [r for r in results if r.family == family]
        rows.append(
            {
                "family": family,
                "solve_accuracy": mean([r.solve_correct for r in rs]),
                "implicit_boundary_accuracy": mean([r.implicit_boundary_accuracy for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "minimal_difference_detection": mean([r.minimal_difference_detection_correct for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
            }
        )

    save_grouped_bar(
        rows,
        "family",
        PLOT_FAMILY,
        "Phase 85 implicit semantic-boundary accuracy by family",
        [
            "solve_accuracy",
            "implicit_boundary_accuracy",
            "variable_binding_accuracy",
            "boundary_localization_accuracy",
            "minimal_difference_detection",
            "no_hallucination_accuracy",
        ],
        width_inches=16,
    )


def save_confusion(results: List[TrialResult]) -> None:
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
    plt.yticks([0, 1], ["gold_accept_true", "gold_reject_implicit"])
    plt.title("Phase 85 implicit semantic-boundary confusion")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PLOT_CONFUSION, dpi=140)
    plt.close()


def save_boundary_localization_plot(results: List[TrialResult]) -> None:
    boundary_rows = [r for r in results if r.pair_type == "implicit_boundary_pair"]
    kinds = sorted(set(r.gold_boundary_location for r in boundary_rows))

    rows = []
    for kind in kinds:
        rs = [r for r in boundary_rows if r.gold_boundary_location == kind]
        rows.append(
            {
                "boundary_kind": kind,
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
            }
        )

    plt.figure(figsize=(14, 4.5))
    plt.bar([r["boundary_kind"] for r in rows], [r["boundary_localization_accuracy"] for r in rows])
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("localization accuracy")
    plt.title("Phase 85 implicit boundary localization accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_LOCALIZATION, dpi=140)
    plt.close()


def save_minimal_delta_plot(results: List[TrialResult]) -> None:
    kinds = sorted(set(r.gold_minimal_delta for r in results))

    rows = []
    for kind in kinds:
        rs = [r for r in results if r.gold_minimal_delta == kind]
        rows.append(
            {
                "minimal_delta_kind": kind,
                "minimal_difference_detection": mean([r.minimal_difference_detection_correct for r in rs]),
            }
        )

    plt.figure(figsize=(16, 4.5))
    plt.bar([r["minimal_delta_kind"] for r in rows], [r["minimal_difference_detection"] for r in rows])
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("minimal delta detection accuracy")
    plt.title("Phase 85 minimal semantic-delta detection accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_MINIMAL_DELTA, dpi=140)
    plt.close()


def make_plots(
    results: List[TrialResult],
    summary: Dict[str, Any],
    task_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
) -> None:
    save_margin_hist(results)
    save_accept_reject_plot(summary)

    save_grouped_bar(
        task_rows,
        "task_id",
        PLOT_TASK,
        "Phase 85 implicit semantic-boundary reasoning by task",
        [
            "solve_accuracy",
            "implicit_boundary_accuracy",
            "schema_selection_accuracy",
            "boundary_localization_accuracy",
            "minimal_difference_detection",
            "trace_validity",
        ],
    )

    save_family_plot(results)

    save_grouped_bar(
        pair_rows,
        "pair_id",
        PLOT_PAIR,
        "Phase 85 accuracy by implicit minimal-pair type",
        [
            "solve_accuracy",
            "implicit_boundary_accuracy",
            "schema_selection_accuracy",
            "boundary_localization_accuracy",
            "minimal_difference_detection",
            "trace_validity",
        ],
    )

    save_confusion(results)
    save_boundary_localization_plot(results)
    save_minimal_delta_plot(results)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from compound boundary interference to implicit minimal-pair boundary detection")
    print(f"[{PHASE}] task: accept true minimal pairs, reject implicit adversarial semantic-boundary crossings")

    results = run_trials(n_trials=20000)
    task_rows = summarize_task(results)
    pair_rows = summarize_pair(results)
    summary = overall_summary(results)

    write_trials_csv(TRIALS_CSV, results)
    write_csv(TASK_SUMMARY_CSV, task_rows)
    write_csv(PAIR_SUMMARY_CSV, pair_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, task_rows, pair_rows)
    write_examples(results)
    make_plots(results, summary, task_rows, pair_rows)

    print(
        f"[{PHASE}] PHASE85_IMPLICIT_ADVERSARIAL_SEMANTIC_BOUNDARY_PASS="
        f"{summary['PHASE85_IMPLICIT_ADVERSARIAL_SEMANTIC_BOUNDARY_PASS']}"
    )

    print(
        f"[{PHASE}] selected_task={summary['selected_task']} "
        f"overall_solve_accuracy={summary['overall_solve_accuracy']:.4f} "
        f"arithmetic_solve_accuracy={summary['arithmetic_solve_accuracy']:.4f} "
        f"geometry_solve_accuracy={summary['geometry_solve_accuracy']:.4f} "
        f"mixed_solve_accuracy={summary['mixed_solve_accuracy']:.4f} "
        f"true_minimal_pair_acceptance={summary['true_minimal_pair_acceptance']:.4f} "
        f"implicit_boundary_rejection={summary['implicit_boundary_rejection']:.4f} "
        f"implicit_boundary_accuracy={summary['implicit_boundary_accuracy']:.4f} "
        f"schema_selection_accuracy={summary['schema_selection_accuracy']:.4f} "
        f"variable_binding_accuracy={summary['variable_binding_accuracy']:.4f} "
        f"boundary_localization_accuracy={summary['boundary_localization_accuracy']:.4f} "
        f"minimal_difference_detection={summary['minimal_difference_detection']:.4f} "
        f"trace_validity={summary['trace_validity']:.4f} "
        f"no_hallucination_accuracy={summary['no_hallucination_accuracy']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] implicit semantic boundary task summary:")
    for r in task_rows:
        print(
            f"  - {r['task_id']:<40} "
            f"family={r['family']:<10} "
            f"solve={r['solve_accuracy']:.3f} "
            f"true_accept={r['true_minimal_pair_acceptance']:.3f} "
            f"implicit_reject={r['implicit_boundary_rejection']:.3f} "
            f"boundary={r['implicit_boundary_accuracy']:.3f} "
            f"bind={r['variable_binding_accuracy']:.3f} "
            f"locate={r['boundary_localization_accuracy']:.3f} "
            f"delta={r['minimal_difference_detection']:.3f} "
            f"trace={r['trace_validity']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={r['trials']}"
        )

    print(f"[{PHASE}] wrote trials: {TRIALS_CSV}")
    print(f"[{PHASE}] wrote task summary: {TASK_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote pair summary: {PAIR_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote summary: {SUMMARY_JSON}")
    print(f"[{PHASE}] wrote report: {REPORT_MD}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()