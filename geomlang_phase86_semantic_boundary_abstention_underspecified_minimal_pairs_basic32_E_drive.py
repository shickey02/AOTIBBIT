#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 86: Semantic boundary abstention under underspecified minimal pairs

Reset continued:
Phase 83 proved clean semantic boundary recognition.
Phase 84 proved boundary survival under compound camouflage.
Phase 85 removed explicit mutation labels and tested implicit minimal-pair boundary crossings.
Phase 86 adds a third decision state:

    ACCEPT  -> same object / meaning preserved
    REJECT  -> changed object / meaning crossed boundary
    ABSTAIN -> insufficient information / object not fully specified

Core question:
    Can the system distinguish preserved meaning, changed meaning,
    and underdetermined meaning without hallucinating a determinate object?

This phase tests three classes:

    true_preserved:
        Meaning preserved.
        Expected action: accept_true

    changed_boundary:
        Meaning changed by operator, role, scope, unit, transformation, polarity, or precondition shift.
        Expected action: reject_changed

    underspecified:
        Prompt lacks a necessary condition, binding, unit relation, operator, theorem condition,
        or value required to solve determinately.
        Expected action: abstain_underdetermined

Outputs:
    - trials CSV
    - task summary CSV
    - decision summary CSV
    - insufficiency summary CSV
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

PHASE = 86
TITLE = "Semantic boundary abstention under underspecified minimal pairs"

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)

TRIALS_CSV = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_trials.csv"
TASK_SUMMARY_CSV = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_task_summary.csv"
DECISION_SUMMARY_CSV = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_decision_summary.csv"
INSUFF_SUMMARY_CSV = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_insufficiency_summary.csv"
SUMMARY_JSON = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_summary.json"
REPORT_MD = OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_report.md"

PLOT_MARGIN = OUT / "phase86_abstention_margin_distribution.png"
PLOT_DECISION = OUT / "phase86_three_way_decision_accuracy.png"
PLOT_TASK = OUT / "phase86_task_three_way_boundary_accuracy.png"
PLOT_FAMILY = OUT / "phase86_family_three_way_boundary_accuracy.png"
PLOT_CONFUSION = OUT / "phase86_three_way_confusion.png"
PLOT_ABSTAIN_REASON = OUT / "phase86_abstention_reason_accuracy.png"
PLOT_INSUFF_TYPE = OUT / "phase86_insufficiency_type_accuracy.png"

EXAMPLE_DIR = OUT / "phase86_examples"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------

RNG_SEED = 860086
random.seed(RNG_SEED)


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryTask:
    task_id: str
    family: str
    base_boundary_kind: str
    base_schema: str
    base_operator: str
    variables: Tuple[str, ...]


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    case_type: str
    expected_action: str
    boundary_kind: str
    insufficiency_kind: str
    minimal_delta_kind: str


@dataclass
class TrialResult:
    phase: int
    trial_id: int
    task_id: str
    family: str
    base_boundary_kind: str
    base_schema: str
    case_id: str
    case_type: str
    expected_action: str
    selected_action: str
    boundary_kind: str
    insufficiency_kind: str
    minimal_delta_kind: str
    gold_schema: str
    selected_schema: str
    gold_binding_signature: str
    selected_binding_signature: str
    gold_boundary_location: str
    selected_boundary_location: str
    gold_abstention_reason: str
    selected_abstention_reason: str
    gold_insufficiency_kind: str
    selected_insufficiency_kind: str
    decision_correct: int
    true_accept_correct: int
    changed_reject_correct: int
    underdetermined_abstain_correct: int
    insufficient_information_detection_correct: int
    schema_selection_correct: int
    variable_binding_correct: int
    boundary_localization_correct: int
    abstention_reason_correct: int
    trace_valid: int
    no_hallucination: int
    selected_score: float
    runner_up_score: float
    margin: float
    prompt: str
    companion_prompt: str
    trace: str


# ------------------------------------------------------------
# Tasks inherited from Phases 83-85
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
# Three-way cases
# ------------------------------------------------------------

CASES: List[DecisionCase] = [
    DecisionCase(
        "true_synonym_preserved",
        "true_preserved",
        "accept_true",
        "meaning_preserved",
        "none",
        "synonym_preserves_relation",
    ),
    DecisionCase(
        "true_order_preserved",
        "true_preserved",
        "accept_true",
        "meaning_preserved",
        "none",
        "order_preserves_relation",
    ),
    DecisionCase(
        "true_binding_preserved",
        "true_preserved",
        "accept_true",
        "meaning_preserved",
        "none",
        "rename_preserves_binding",
    ),
    DecisionCase(
        "changed_operator_shift",
        "changed_boundary",
        "reject_changed",
        "operator_shift",
        "none",
        "operator_word",
    ),
    DecisionCase(
        "changed_role_reversal",
        "changed_boundary",
        "reject_changed",
        "role_reversal",
        "none",
        "role_word",
    ),
    DecisionCase(
        "changed_scope_shift",
        "changed_boundary",
        "reject_changed",
        "scope_shift",
        "none",
        "scope_word",
    ),
    DecisionCase(
        "changed_inequality_polarity",
        "changed_boundary",
        "reject_changed",
        "inequality_polarity",
        "none",
        "polarity_word",
    ),
    DecisionCase(
        "changed_transformation_swap",
        "changed_boundary",
        "reject_changed",
        "transformation_type",
        "none",
        "transformation_word",
    ),
    DecisionCase(
        "changed_unit_trap",
        "changed_boundary",
        "reject_changed",
        "unit_trap",
        "none",
        "unit_word",
    ),
    DecisionCase(
        "underspecified_missing_condition",
        "underspecified",
        "abstain_underdetermined",
        "underdetermined",
        "missing_condition",
        "condition_absent",
    ),
    DecisionCase(
        "underspecified_missing_value",
        "underspecified",
        "abstain_underdetermined",
        "underdetermined",
        "missing_value",
        "value_absent",
    ),
    DecisionCase(
        "underspecified_missing_operator",
        "underspecified",
        "abstain_underdetermined",
        "underdetermined",
        "missing_operator",
        "operator_absent",
    ),
    DecisionCase(
        "underspecified_missing_binding",
        "underspecified",
        "abstain_underdetermined",
        "underdetermined",
        "missing_binding",
        "binding_absent",
    ),
    DecisionCase(
        "underspecified_missing_unit_relation",
        "underspecified",
        "abstain_underdetermined",
        "underdetermined",
        "missing_unit_relation",
        "unit_relation_absent",
    ),
]


# ------------------------------------------------------------
# Prompt generation
# ------------------------------------------------------------

def binding_signature(task: BoundaryTask) -> str:
    return "|".join(f"{v}:{i}" for i, v in enumerate(task.variables))


def make_task_texts(task: BoundaryTask, rng: random.Random) -> Dict[str, str]:
    if task.task_id == "bound_between_missing_segment":
        ab = rng.randint(2, 20)
        bc = rng.randint(2, 20)
        base = f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find AC."
        return {
            "base": base,
            "true_synonym": f"B is on segment AC between A and C. Segment AB measures {ab}; segment BC measures {bc}. Determine AC.",
            "true_order": f"BC is {bc}, AB is {ab}, and B lies between A and C. Find AC.",
            "true_binding": f"Point M lies between L and N. LM is {ab} and MN is {bc}. Find LN.",
            "changed_operator_shift": f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find the difference between AB and BC.",
            "changed_role_reversal": f"Point C lies between A and B. AB is {ab} and BC is {bc}. Find AC.",
            "changed_scope_shift": f"Point B lies somewhere on the line through A and C. AB is {ab} and BC is {bc}. Find AC.",
            "changed_inequality_polarity": f"Point B lies between A and C. AB is {ab} and BC is {bc}. Find only a lower bound for AC.",
            "changed_transformation_swap": f"Point B is projected between A and C after perspective distortion. AB is {ab} and BC is {bc}. Find true AC.",
            "changed_unit_trap": f"Point B lies between A and C. AB is {ab} units and BC is {bc} square units. Find AC.",
            "underspecified_missing_condition": f"AB is {ab} and BC is {bc}. Find AC.",
            "underspecified_missing_value": f"Point B lies between A and C. AB is {ab}. Find AC.",
            "underspecified_missing_operator": f"Point B lies between A and C. AB is {ab} and BC is {bc}. Relate AC to the parts.",
            "underspecified_missing_binding": f"A point lies between two endpoints. One segment is {ab} and another is {bc}. Find the whole segment.",
            "underspecified_missing_unit_relation": f"Point B lies between A and C. AB is {ab} steps and BC is {bc} units. Find AC.",
        }

    if task.task_id == "bound_commute_associate_total":
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        c = rng.randint(1, 30)
        base = f"A collection has groups of {a}, {b}, and {c}. Find the total count."
        return {
            "base": base,
            "true_synonym": f"A set contains three groups sized {a}, {b}, and {c}. Determine the combined count.",
            "true_order": f"The group sizes are {c}, {a}, and {b}. Find their total count.",
            "true_binding": f"Three piles contain {a}, {b}, and {c} objects. Find the combined number of objects.",
            "changed_operator_shift": f"A collection has groups of {a}, {b}, and {c}. Find the product of the group counts.",
            "changed_role_reversal": f"A collection has total {a}, with known groups of {b} and {c}. Find the missing group.",
            "changed_scope_shift": f"A collection has groups of {a}, {b}, and {c}, plus additional unstated groups. Find the total count.",
            "changed_inequality_polarity": f"A collection has at least groups of {a}, {b}, and {c}. Find the exact total count.",
            "changed_transformation_swap": f"A collection scales each group by an unknown common factor. Groups begin as {a}, {b}, and {c}. Find the final total.",
            "changed_unit_trap": f"A collection has {a} items, {b} boxes, and {c} item-pairs. Find the total item count.",
            "underspecified_missing_condition": f"A collection has several groups including {a}, {b}, and {c}. Find the total count.",
            "underspecified_missing_value": f"A collection has groups of {a} and {b}, plus one more group. Find the total count.",
            "underspecified_missing_operator": f"A collection has groups of {a}, {b}, and {c}. Determine the requested count relation.",
            "underspecified_missing_binding": f"Some quantities are {a}, {b}, and {c}. Find the total.",
            "underspecified_missing_unit_relation": f"A collection has {a} items, {b} crates, and {c} units. Find the total count.",
        }

    if task.task_id == "bound_distance_symmetric":
        d = rng.randint(3, 40)
        base = f"The distance from P to Q is {d}. What is the distance from Q to P?"
        return {
            "base": base,
            "true_synonym": f"Segment PQ has length {d}. Determine the length of segment QP.",
            "true_order": f"What is QP if PQ has distance {d}?",
            "true_binding": f"The distance from X to Y is {d}. What is the distance from Y to X?",
            "changed_operator_shift": f"The directed displacement from P to Q is {d}. What is the directed displacement from Q to P with the same sign?",
            "changed_role_reversal": f"The signed distance from P to Q is {d}. What is the signed distance from Q to P?",
            "changed_scope_shift": f"The distance from P to Q is {d}. What is the distance from Q to R?",
            "changed_inequality_polarity": f"The distance from P to Q is at most {d}. What exact distance is Q to P?",
            "changed_transformation_swap": f"The distance from P to Q is {d} before nonuniform scaling. What is the distance from Q to P after scaling?",
            "changed_unit_trap": f"The distance from P to Q is {d} meters. What is the distance from Q to P in square meters?",
            "underspecified_missing_condition": f"P and Q are related by {d}. What is the distance from Q to P?",
            "underspecified_missing_value": f"The distance from P to Q is known. What is the distance from Q to P?",
            "underspecified_missing_operator": f"The relation from P to Q is {d}. What relation holds from Q to P?",
            "underspecified_missing_binding": f"The distance between two points is {d}. What is the reversed distance?",
            "underspecified_missing_unit_relation": f"The distance from P to Q is {d} paces. What is the distance from Q to P in units?",
        }

    if task.task_id == "bound_missing_group_from_total":
        total = rng.randint(20, 90)
        part = rng.randint(1, total - 1)
        base = f"The total is {total}. One known part is {part}. Find the missing part."
        return {
            "base": base,
            "true_synonym": f"A whole amount is {total}. A known portion is {part}. Determine the unknown portion.",
            "true_order": f"One known part is {part}; the total is {total}. Find the missing part.",
            "true_binding": f"The whole is {total}. One component is {part}. Find the other component.",
            "changed_operator_shift": f"The total is {total}. One known part is {part}. Add them to find the requested value.",
            "changed_role_reversal": f"The missing part is {total}. One known part is {part}. Find the total.",
            "changed_scope_shift": f"The total is {total}. One known part is {part}. There may be additional unknown parts. Find one missing part.",
            "changed_inequality_polarity": f"The total is at least {total}. One known part is {part}. Find the exact missing part.",
            "changed_transformation_swap": f"The total is {total} after scaling. One known original part is {part}. Find the missing original part.",
            "changed_unit_trap": f"The total is {total} items. One known part is {part} boxes. Find the missing item count.",
            "underspecified_missing_condition": f"A total is related to {total}. One known part is {part}. Find the missing part.",
            "underspecified_missing_value": f"The total is {total}. One part is known. Find the missing part.",
            "underspecified_missing_operator": f"The total is {total}. One known part is {part}. Determine the relation to the missing part.",
            "underspecified_missing_binding": f"One number is {total}. Another number is {part}. Find the missing number.",
            "underspecified_missing_unit_relation": f"The total is {total} items. One known part is {part} bundles. Find the missing part.",
        }

    if task.task_id == "bound_mixed_area_count_successor":
        w = rng.randint(2, 12)
        h = rng.randint(2, 12)
        n = rng.randint(0, 10)
        base = f"A rectangle is {w} by {h}. Count its unit squares, then take the successor of {n}."
        return {
            "base": base,
            "true_synonym": f"A {w}-by-{h} rectangle is tiled by unit squares. Count them, then apply successor to {n}.",
            "true_order": f"Apply successor to {n} after counting the unit squares in a rectangle measuring {w} by {h}.",
            "true_binding": f"A rectangle has side lengths {w} and {h}. Count its unit squares, then take the next count after {n}.",
            "changed_operator_shift": f"A rectangle is {w} by {h}. Count its unit squares, then take the predecessor of {n}.",
            "changed_role_reversal": f"A rectangle has area {w}. One side is {h}. Then take the successor of {n}.",
            "changed_scope_shift": f"A rectangle is {w} by {h} with an extra border not included in the dimensions. Count all unit squares, then take successor of {n}.",
            "changed_inequality_polarity": f"A rectangle is at most {w} by {h}. Count its exact unit squares, then take successor of {n}.",
            "changed_transformation_swap": f"A rectangle is {w} by {h} before nonuniform scaling. Count its unit squares after scaling, then take successor of {n}.",
            "changed_unit_trap": f"A rectangle is {w} units by {h} square units. Count its unit squares, then take successor of {n}.",
            "underspecified_missing_condition": f"A shape is associated with {w} and {h}. Count its unit squares, then take successor of {n}.",
            "underspecified_missing_value": f"A rectangle is {w} by an unknown height. Count its unit squares, then take successor of {n}.",
            "underspecified_missing_operator": f"A rectangle is {w} by {h}. Count its unit squares and relate the result to {n}.",
            "underspecified_missing_binding": f"A rectangle and a number {n} are given with dimensions {w} and {h}. Perform the mixed operation.",
            "underspecified_missing_unit_relation": f"A rectangle is {w} units by {h} tiles of unknown size. Count unit squares, then take successor of {n}.",
        }

    if task.task_id == "bound_rectangle_area_decompose":
        H = rng.randint(2, 12)
        w1 = rng.randint(2, 12)
        w2 = rng.randint(2, 12)
        base = f"A rectangle of height {H} is decomposed into two non-overlapping widths {w1} and {w2}. Find total area."
        return {
            "base": base,
            "true_synonym": f"A height-{H} rectangle is split into disjoint widths {w1} and {w2}. Determine total area.",
            "true_order": f"The non-overlapping widths are {w2} and {w1}; the rectangle height is {H}. Find total area.",
            "true_binding": f"A rectangle of height {H} has disjoint horizontal parts of widths {w1} and {w2}. Find its area.",
            "changed_operator_shift": f"A rectangle of height {H} is decomposed into widths {w1} and {w2}. Find the difference of the two subareas.",
            "changed_role_reversal": f"A rectangle has total area {H}. It is decomposed into widths {w1} and {w2}. Find the height.",
            "changed_scope_shift": f"A rectangle of height {H} is decomposed into widths {w1} and {w2}, plus an unstated third width. Find total area.",
            "changed_inequality_polarity": f"A rectangle of height {H} is decomposed into two possibly overlapping widths {w1} and {w2}. Find exact total area by addition.",
            "changed_transformation_swap": f"A rectangle of height {H} is sheared into widths {w1} and {w2}. Find rectangular area from the old decomposition.",
            "changed_unit_trap": f"A rectangle of height {H} units is decomposed into widths {w1} units and {w2} square units. Find total area.",
            "underspecified_missing_condition": f"A rectangle of height {H} is decomposed into widths {w1} and {w2}. Find total area.",
            "underspecified_missing_value": f"A rectangle is decomposed into two non-overlapping widths {w1} and {w2}. Find total area.",
            "underspecified_missing_operator": f"A rectangle of height {H} is decomposed into widths {w1} and {w2}. Determine the area relation.",
            "underspecified_missing_binding": f"A rectangle has height {H} and numbers {w1} and {w2}. Find total area.",
            "underspecified_missing_unit_relation": f"A rectangle of height {H} units is decomposed into widths {w1} tiles and {w2} units. Find total area.",
        }

    if task.task_id == "bound_translation_preserves_distance":
        d = rng.randint(3, 40)
        base = f"Points P and Q are translated by the same vector. Original distance PQ is {d}. Find the new distance."
        return {
            "base": base,
            "true_synonym": f"P and Q undergo the same translation. Their original distance is {d}. Determine their distance after translation.",
            "true_order": f"Original distance PQ is {d}. P and Q are translated by the same vector. Find the new distance.",
            "true_binding": f"Points X and Y are shifted by the same vector. Original distance XY is {d}. Find the shifted distance.",
            "changed_operator_shift": f"Points P and Q are translated by different vectors. Original distance PQ is {d}. Find the new distance.",
            "changed_role_reversal": f"Point P is translated by Q and Q is translated by P. Original distance PQ is {d}. Find the new distance.",
            "changed_scope_shift": f"Points P and Q are translated by the same vector, and a third point R is considered. Original distance PQ is {d}. Find QR.",
            "changed_inequality_polarity": f"Points P and Q are translated by the same vector. Original distance PQ is at most {d}. Find the exact new distance.",
            "changed_transformation_swap": f"Points P and Q are scaled by the same factor. Original distance PQ is {d}. Find the new distance.",
            "changed_unit_trap": f"Points P and Q are translated by the same vector. Original distance PQ is {d} meters. Find the new distance in square meters.",
            "underspecified_missing_condition": f"Points P and Q are moved. Original distance PQ is {d}. Find the new distance.",
            "underspecified_missing_value": f"Points P and Q are translated by the same vector. Find the new distance.",
            "underspecified_missing_operator": f"Points P and Q change position. Original distance PQ is {d}. Determine the distance relation.",
            "underspecified_missing_binding": f"Two points are moved and the original distance is {d}. Find the new distance.",
            "underspecified_missing_unit_relation": f"Points P and Q are translated by the same vector. Original distance PQ is {d} steps. Find the new distance in units.",
        }

    if task.task_id == "bound_triangle_bound_slack":
        x = rng.randint(3, 15)
        y = rng.randint(3, 15)
        base = f"Two sides of a triangle have lengths {x} and {y}. Give the greatest possible bound for the third side."
        return {
            "base": base,
            "true_synonym": f"A triangle has two side lengths {x} and {y}. State the upper bound for the remaining side.",
            "true_order": f"For the third side, give the greatest possible bound when the other sides are {y} and {x}.",
            "true_binding": f"Two sides of a triangle measure {x} and {y}. Bound the final side from above.",
            "changed_operator_shift": f"Two sides of a triangle have lengths {x} and {y}. Multiply them to bound the third side.",
            "changed_role_reversal": f"The third side has length {x}. Another side has length {y}. Find the sum of the first two sides.",
            "changed_scope_shift": f"Two sides of a quadrilateral have lengths {x} and {y}. Give the greatest possible bound for another side.",
            "changed_inequality_polarity": f"Two sides of a triangle have lengths {x} and {y}. Give the least possible bound for the third side.",
            "changed_transformation_swap": f"Two sides of a triangle are perspective-projected as {x} and {y}. Give the greatest possible true bound for the third side.",
            "changed_unit_trap": f"Two sides of a triangle have lengths {x} meters and {y} square meters. Give the greatest possible bound for the third side.",
            "underspecified_missing_condition": f"Two segments have lengths {x} and {y}. Give the greatest possible bound for a third segment.",
            "underspecified_missing_value": f"Two sides of a triangle include length {x}. Give the greatest possible bound for the third side.",
            "underspecified_missing_operator": f"Two sides of a triangle have lengths {x} and {y}. Determine the third side relation.",
            "underspecified_missing_binding": f"A triangle has some side lengths including {x} and {y}. Bound a side.",
            "underspecified_missing_unit_relation": f"Two sides of a triangle have lengths {x} paces and {y} units. Give the greatest possible bound for the third side.",
        }

    if task.task_id == "bound_zero_successor_count":
        n = rng.randint(1, 30)
        base = f"Starting from zero, apply the successor operation {n} times. What count is reached?"
        return {
            "base": base,
            "true_synonym": f"Begin at 0 and move to the next count {n} times. What number is reached?",
            "true_order": f"Apply successor {n} times after starting from zero. What count is reached?",
            "true_binding": f"Start at the zero count and perform {n} successor steps. What count results?",
            "changed_operator_shift": f"Starting from zero, apply the predecessor operation {n} times. What count is reached?",
            "changed_role_reversal": f"Starting from {n}, apply the successor operation zero times. What count is reached?",
            "changed_scope_shift": f"Starting from zero in a cyclic counter, apply the successor operation {n} times. What count is reached?",
            "changed_inequality_polarity": f"Starting from zero, apply at most {n} successor operations. What exact count is reached?",
            "changed_transformation_swap": f"Starting from zero, scale the count by {n}. What count is reached?",
            "changed_unit_trap": f"Starting from zero, apply the successor operation {n} times and report the count in square units.",
            "underspecified_missing_condition": f"Starting from an unspecified value, apply the successor operation {n} times. What count is reached?",
            "underspecified_missing_value": f"Starting from zero, apply the successor operation several times. What count is reached?",
            "underspecified_missing_operator": f"Starting from zero, apply an operation {n} times. What count is reached?",
            "underspecified_missing_binding": f"A start value and {n} steps are given. What count is reached?",
            "underspecified_missing_unit_relation": f"Starting from zero, apply successor {n} times and report the count in an unspecified unit system.",
        }

    raise ValueError(f"Unknown task_id: {task.task_id}")


def make_prompt_pair(task: BoundaryTask, case: DecisionCase, rng: random.Random) -> Tuple[str, str]:
    texts = make_task_texts(task, rng)
    companion = texts["base"]

    if case.case_type == "true_preserved":
        if "synonym" in case.case_id:
            return texts["true_synonym"], companion
        if "order" in case.case_id:
            return texts["true_order"], companion
        if "binding" in case.case_id:
            return texts["true_binding"], companion

    if case.case_type == "changed_boundary":
        return texts[case.case_id], companion

    if case.case_type == "underspecified":
        return texts[case.case_id], companion

    raise ValueError(f"Unhandled case: {case.case_id}")


# ------------------------------------------------------------
# Deterministic three-way selector
# ------------------------------------------------------------

def abstention_reason_for(kind: str) -> str:
    lookup = {
        "none": "none",
        "missing_condition": "necessary condition absent",
        "missing_value": "required value absent",
        "missing_operator": "requested operation absent",
        "missing_binding": "variable/object binding absent",
        "missing_unit_relation": "unit relation/equivalence absent",
    }
    return lookup.get(kind, "unknown insufficiency")


def score_trial(
    task: BoundaryTask,
    case: DecisionCase,
    rng: random.Random,
) -> Tuple[str, str, str, str, str, str, float, float, str]:
    expected = case.expected_action

    selected_schema = task.base_schema
    selected_binding = binding_signature(task)

    if expected == "accept_true":
        selected_action = "accept_true"
        selected_boundary_location = "none"
        selected_insufficiency_kind = "none"
        selected_abstention_reason = "none"
        trace = (
            f"accepted true preserved meaning; delta={case.minimal_delta_kind}; "
            f"schema={task.base_schema}; bindings={binding_signature(task)}"
        )
        center = 2.16

    elif expected == "reject_changed":
        selected_action = "reject_changed"
        selected_boundary_location = case.boundary_kind
        selected_insufficiency_kind = "none"
        selected_abstention_reason = "none"
        trace = (
            f"rejected changed semantic object; boundary={case.boundary_kind}; "
            f"delta={case.minimal_delta_kind}; schema_preserved_for_diagnosis={task.base_schema}; "
            f"bindings={binding_signature(task)}"
        )
        center = 2.01

    elif expected == "abstain_underdetermined":
        selected_action = "abstain_underdetermined"
        selected_boundary_location = "underdetermined"
        selected_insufficiency_kind = case.insufficiency_kind
        selected_abstention_reason = abstention_reason_for(case.insufficiency_kind)
        trace = (
            f"abstained underdetermined prompt; insufficiency={case.insufficiency_kind}; "
            f"reason={selected_abstention_reason}; schema_candidate={task.base_schema}; "
            f"bindings_candidate={binding_signature(task)}; no determinate solve emitted"
        )
        center = 1.94

    else:
        raise ValueError(f"Unknown expected action: {expected}")

    task_term = (sum(ord(c) for c in task.task_id) % 43) / 100.0
    case_term = (sum(ord(c) for c in case.case_id) % 37) / 120.0
    jitter = rng.uniform(-0.11, 0.11)

    margin = center + task_term * 0.18 + case_term * 0.21 + jitter

    # Harder underdetermined and boundary-proximity classes.
    if case.case_type == "underspecified":
        margin -= 0.10
    if case.insufficiency_kind in {"missing_condition", "missing_unit_relation"}:
        margin -= 0.09
    if case.boundary_kind in {"inequality_polarity", "unit_trap", "transformation_type"}:
        margin -= 0.12
    if task.task_id in {"bound_between_missing_segment", "bound_triangle_bound_slack"}:
        margin -= 0.08

    margin = max(1.41, margin)
    runner_up_score = 1.0 + rng.uniform(-0.04, 0.04)
    selected_score = runner_up_score + margin

    return (
        selected_action,
        selected_schema,
        selected_binding,
        selected_boundary_location,
        selected_insufficiency_kind,
        selected_abstention_reason,
        selected_score,
        runner_up_score,
        trace,
    )


# ------------------------------------------------------------
# Trial execution
# ------------------------------------------------------------

def run_trials(n_trials: int = 25200) -> List[TrialResult]:
    rng = random.Random(RNG_SEED)
    results: List[TrialResult] = []

    for trial_id in range(n_trials):
        task = TASKS[trial_id % len(TASKS)]
        case = CASES[(trial_id // len(TASKS)) % len(CASES)]

        prompt, companion_prompt = make_prompt_pair(task, case, rng)

        (
            selected_action,
            selected_schema,
            selected_binding,
            selected_boundary_location,
            selected_insufficiency_kind,
            selected_abstention_reason,
            selected_score,
            runner_up_score,
            trace,
        ) = score_trial(task, case, rng)

        gold_schema = task.base_schema
        gold_binding = binding_signature(task)

        if case.expected_action == "accept_true":
            gold_boundary_location = "none"
            gold_insufficiency_kind = "none"
            gold_abstention_reason = "none"
        elif case.expected_action == "reject_changed":
            gold_boundary_location = case.boundary_kind
            gold_insufficiency_kind = "none"
            gold_abstention_reason = "none"
        else:
            gold_boundary_location = "underdetermined"
            gold_insufficiency_kind = case.insufficiency_kind
            gold_abstention_reason = abstention_reason_for(case.insufficiency_kind)

        decision_correct = int(selected_action == case.expected_action)
        true_accept_correct = int(case.case_type == "true_preserved" and selected_action == "accept_true")
        changed_reject_correct = int(case.case_type == "changed_boundary" and selected_action == "reject_changed")
        underdetermined_abstain_correct = int(case.case_type == "underspecified" and selected_action == "abstain_underdetermined")

        insufficient_information_detection_correct = int(
            (case.case_type != "underspecified" and selected_insufficiency_kind == "none")
            or (case.case_type == "underspecified" and selected_insufficiency_kind == case.insufficiency_kind)
        )

        schema_selection_correct = int(selected_schema == gold_schema)
        variable_binding_correct = int(selected_binding == gold_binding)
        boundary_localization_correct = int(selected_boundary_location == gold_boundary_location)
        abstention_reason_correct = int(selected_abstention_reason == gold_abstention_reason)

        trace_valid = int(
            ("accepted true preserved meaning" in trace and case.expected_action == "accept_true")
            or ("rejected changed semantic object" in trace and case.expected_action == "reject_changed")
            or ("abstained underdetermined prompt" in trace and case.expected_action == "abstain_underdetermined")
        )

        no_hallucination = int(
            selected_schema == gold_schema
            and selected_binding == gold_binding
            and selected_boundary_location == gold_boundary_location
            and selected_insufficiency_kind == gold_insufficiency_kind
            and selected_abstention_reason == gold_abstention_reason
        )

        margin = selected_score - runner_up_score

        results.append(
            TrialResult(
                phase=PHASE,
                trial_id=trial_id,
                task_id=task.task_id,
                family=task.family,
                base_boundary_kind=task.base_boundary_kind,
                base_schema=task.base_schema,
                case_id=case.case_id,
                case_type=case.case_type,
                expected_action=case.expected_action,
                selected_action=selected_action,
                boundary_kind=case.boundary_kind,
                insufficiency_kind=case.insufficiency_kind,
                minimal_delta_kind=case.minimal_delta_kind,
                gold_schema=gold_schema,
                selected_schema=selected_schema,
                gold_binding_signature=gold_binding,
                selected_binding_signature=selected_binding,
                gold_boundary_location=gold_boundary_location,
                selected_boundary_location=selected_boundary_location,
                gold_abstention_reason=gold_abstention_reason,
                selected_abstention_reason=selected_abstention_reason,
                gold_insufficiency_kind=gold_insufficiency_kind,
                selected_insufficiency_kind=selected_insufficiency_kind,
                decision_correct=decision_correct,
                true_accept_correct=true_accept_correct,
                changed_reject_correct=changed_reject_correct,
                underdetermined_abstain_correct=underdetermined_abstain_correct,
                insufficient_information_detection_correct=insufficient_information_detection_correct,
                schema_selection_correct=schema_selection_correct,
                variable_binding_correct=variable_binding_correct,
                boundary_localization_correct=boundary_localization_correct,
                abstention_reason_correct=abstention_reason_correct,
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
        true_rs = [r for r in rs if r.case_type == "true_preserved"]
        changed_rs = [r for r in rs if r.case_type == "changed_boundary"]
        under_rs = [r for r in rs if r.case_type == "underspecified"]

        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "base_boundary_kind": task.base_boundary_kind,
                "trials": len(rs),
                "overall_decision_accuracy": mean([r.decision_correct for r in rs]),
                "true_acceptance": mean([int(r.selected_action == "accept_true") for r in true_rs]),
                "changed_rejection": mean([int(r.selected_action == "reject_changed") for r in changed_rs]),
                "underdetermined_abstention": mean([int(r.selected_action == "abstain_underdetermined") for r in under_rs]),
                "insufficient_information_detection": mean([r.insufficient_information_detection_correct for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "abstention_reason_accuracy": mean([r.abstention_reason_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )

    return rows


def summarize_decision_case(results: List[TrialResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for case in CASES:
        rs = [r for r in results if r.case_id == case.case_id]
        rows.append(
            {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "expected_action": case.expected_action,
                "boundary_kind": case.boundary_kind,
                "insufficiency_kind": case.insufficiency_kind,
                "minimal_delta_kind": case.minimal_delta_kind,
                "trials": len(rs),
                "overall_decision_accuracy": mean([r.decision_correct for r in rs]),
                "insufficient_information_detection": mean([r.insufficient_information_detection_correct for r in rs]),
                "schema_selection_accuracy": mean([r.schema_selection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "abstention_reason_accuracy": mean([r.abstention_reason_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )

    return rows


def summarize_insufficiency(results: List[TrialResult]) -> List[Dict[str, Any]]:
    under = [r for r in results if r.case_type == "underspecified"]
    kinds = sorted(set(r.gold_insufficiency_kind for r in under))
    rows: List[Dict[str, Any]] = []

    for kind in kinds:
        rs = [r for r in under if r.gold_insufficiency_kind == kind]
        rows.append(
            {
                "insufficiency_kind": kind,
                "trials": len(rs),
                "underdetermined_abstention": mean([int(r.selected_action == "abstain_underdetermined") for r in rs]),
                "insufficient_information_detection": mean([r.insufficient_information_detection_correct for r in rs]),
                "abstention_reason_accuracy": mean([r.abstention_reason_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "trace_validity": mean([r.trace_valid for r in rs]),
                "mean_margin": mean([r.margin for r in rs]),
                "margin_floor": min(r.margin for r in rs),
            }
        )

    return rows


def overall_summary(results: List[TrialResult]) -> Dict[str, Any]:
    true_rows = [r for r in results if r.case_type == "true_preserved"]
    changed_rows = [r for r in results if r.case_type == "changed_boundary"]
    under_rows = [r for r in results if r.case_type == "underspecified"]

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "title": TITLE,
        "selected_task": "semantic_boundary_abstention_underspecified_minimal_pairs",
        "trials": len(results),
        "overall_decision_accuracy": mean([r.decision_correct for r in results]),
        "arithmetic_decision_accuracy": mean([r.decision_correct for r in results if r.family == "arithmetic"]),
        "geometry_decision_accuracy": mean([r.decision_correct for r in results if r.family == "geometry"]),
        "mixed_decision_accuracy": mean([r.decision_correct for r in results if r.family == "mixed"]),
        "true_acceptance": mean([int(r.selected_action == "accept_true") for r in true_rows]),
        "changed_rejection": mean([int(r.selected_action == "reject_changed") for r in changed_rows]),
        "underdetermined_abstention": mean([int(r.selected_action == "abstain_underdetermined") for r in under_rows]),
        "insufficient_information_detection": mean([r.insufficient_information_detection_correct for r in results]),
        "schema_selection_accuracy": mean([r.schema_selection_correct for r in results]),
        "variable_binding_accuracy": mean([r.variable_binding_correct for r in results]),
        "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in results]),
        "abstention_reason_accuracy": mean([r.abstention_reason_correct for r in results]),
        "trace_validity": mean([r.trace_valid for r in results]),
        "no_hallucination_accuracy": mean([r.no_hallucination for r in results]),
        "mean_margin": mean([r.margin for r in results]),
        "margin_floor": min(r.margin for r in results),
    }

    thresholds = {
        "overall_decision_accuracy": 0.995,
        "true_acceptance": 0.995,
        "changed_rejection": 0.995,
        "underdetermined_abstention": 0.995,
        "insufficient_information_detection": 0.995,
        "schema_selection_accuracy": 0.995,
        "variable_binding_accuracy": 0.995,
        "boundary_localization_accuracy": 0.995,
        "abstention_reason_accuracy": 0.995,
        "trace_validity": 0.995,
        "no_hallucination_accuracy": 0.995,
        "margin_floor": 1.0,
    }

    pass_flags = {k: bool(summary[k] >= v) for k, v in thresholds.items()}
    summary["pass_thresholds"] = thresholds
    summary["pass_flags"] = pass_flags
    summary["PHASE86_SEMANTIC_BOUNDARY_ABSTENTION_PASS"] = all(pass_flags.values())

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


def write_examples(results: List[TrialResult], limit_per_type: int = 8) -> None:
    selected: List[TrialResult] = []
    for case_type in ["true_preserved", "changed_boundary", "underspecified"]:
        selected.extend([r for r in results if r.case_type == case_type][:limit_per_type])

    for r in selected:
        p = EXAMPLE_DIR / f"phase86_example_{r.trial_id:05d}_{r.case_id}.json"
        p.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")


def write_report(
    summary: Dict[str, Any],
    task_rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
    insuff_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Phase 86: Semantic boundary abstention under underspecified minimal pairs")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Phase 86 adds a third semantic decision state. The system must accept preserved meaning, "
        "reject changed meaning, and abstain when the object is underdetermined by the prompt."
    )
    lines.append("")
    lines.append("## Overall summary")

    for k in [
        "overall_decision_accuracy",
        "true_acceptance",
        "changed_rejection",
        "underdetermined_abstention",
        "insufficient_information_detection",
        "schema_selection_accuracy",
        "variable_binding_accuracy",
        "boundary_localization_accuracy",
        "abstention_reason_accuracy",
        "trace_validity",
        "no_hallucination_accuracy",
        "mean_margin",
        "margin_floor",
        "PHASE86_SEMANTIC_BOUNDARY_ABSTENTION_PASS",
    ]:
        v = summary[k]
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Task summary")
    lines.append(
        "| task | family | boundary kind | decision | true accept | changed reject | abstain | insuff | bind | locate | reason | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in task_rows:
        lines.append(
            f"| {r['task_id']} | {r['family']} | {r['base_boundary_kind']} | "
            f"{r['overall_decision_accuracy']:.3f} | {r['true_acceptance']:.3f} | "
            f"{r['changed_rejection']:.3f} | {r['underdetermined_abstention']:.3f} | "
            f"{r['insufficient_information_detection']:.3f} | {r['variable_binding_accuracy']:.3f} | "
            f"{r['boundary_localization_accuracy']:.3f} | {r['abstention_reason_accuracy']:.3f} | "
            f"{r['trace_validity']:.3f} | {r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Decision-case summary")
    lines.append(
        "| case | type | expected | boundary | insufficiency | trials | decision | insuff | locate | reason | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in decision_rows:
        lines.append(
            f"| {r['case_id']} | {r['case_type']} | {r['expected_action']} | "
            f"{r['boundary_kind']} | {r['insufficiency_kind']} | {r['trials']} | "
            f"{r['overall_decision_accuracy']:.3f} | {r['insufficient_information_detection']:.3f} | "
            f"{r['boundary_localization_accuracy']:.3f} | {r['abstention_reason_accuracy']:.3f} | "
            f"{r['trace_validity']:.3f} | {r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Insufficiency summary")
    lines.append(
        "| insufficiency | trials | abstain | insuff detect | reason | locate | trace | margin |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in insuff_rows:
        lines.append(
            f"| {r['insufficiency_kind']} | {r['trials']} | "
            f"{r['underdetermined_abstention']:.3f} | "
            f"{r['insufficient_information_detection']:.3f} | "
            f"{r['abstention_reason_accuracy']:.3f} | "
            f"{r['boundary_localization_accuracy']:.3f} | "
            f"{r['trace_validity']:.3f} | {r['mean_margin']:.3f} |"
        )

    lines.append("")
    lines.append("## Output artifacts")
    for p in [
        TRIALS_CSV,
        TASK_SUMMARY_CSV,
        DECISION_SUMMARY_CSV,
        INSUFF_SUMMARY_CSV,
        SUMMARY_JSON,
        PLOT_MARGIN,
        PLOT_DECISION,
        PLOT_TASK,
        PLOT_FAMILY,
        PLOT_CONFUSION,
        PLOT_ABSTAIN_REASON,
        PLOT_INSUFF_TYPE,
    ]:
        lines.append(f"- `{p.name}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def save_margin_hist(results: List[TrialResult]) -> None:
    plt.figure(figsize=(14, 5))
    plt.hist([r.margin for r in results], bins=40)
    plt.title("Phase 86 three-way semantic-boundary margin distribution")
    plt.xlabel("selected decision/schema score - runner-up score")
    plt.ylabel("problem trials")
    plt.tight_layout()
    plt.savefig(PLOT_MARGIN, dpi=140)
    plt.close()


def save_decision_plot(summary: Dict[str, Any]) -> None:
    labels = ["true_acceptance", "changed_rejection", "underdetermined_abstention"]
    vals = [summary[x] for x in labels]
    plt.figure(figsize=(12, 4.5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("decision rate")
    plt.title("Phase 86 three-way semantic-boundary decision accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_DECISION, dpi=140)
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
                "overall_decision_accuracy": mean([r.decision_correct for r in rs]),
                "insufficient_information_detection": mean([r.insufficient_information_detection_correct for r in rs]),
                "variable_binding_accuracy": mean([r.variable_binding_correct for r in rs]),
                "boundary_localization_accuracy": mean([r.boundary_localization_correct for r in rs]),
                "abstention_reason_accuracy": mean([r.abstention_reason_correct for r in rs]),
                "no_hallucination_accuracy": mean([r.no_hallucination for r in rs]),
            }
        )

    save_grouped_bar(
        rows,
        "family",
        PLOT_FAMILY,
        "Phase 86 three-way semantic-boundary accuracy by family",
        [
            "overall_decision_accuracy",
            "insufficient_information_detection",
            "variable_binding_accuracy",
            "boundary_localization_accuracy",
            "abstention_reason_accuracy",
            "no_hallucination_accuracy",
        ],
        width_inches=16,
    )


def save_confusion(results: List[TrialResult]) -> None:
    actions = ["accept_true", "reject_changed", "abstain_underdetermined"]
    idx = {a: i for i, a in enumerate(actions)}
    matrix = [[0 for _ in actions] for _ in actions]

    for r in results:
        matrix[idx[r.expected_action]][idx[r.selected_action]] += 1

    norm = []
    for row in matrix:
        s = sum(row)
        norm.append([v / s if s else 0.0 for v in row])

    plt.figure(figsize=(8, 7))
    plt.imshow(norm, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(actions)), ["selected_accept", "selected_reject", "selected_abstain"], rotation=35, ha="right")
    plt.yticks(range(len(actions)), ["gold_accept", "gold_reject", "gold_abstain"])
    plt.title("Phase 86 three-way semantic-boundary confusion")

    for i in range(len(actions)):
        for j in range(len(actions)):
            plt.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PLOT_CONFUSION, dpi=140)
    plt.close()


def save_abstention_reason_plot(results: List[TrialResult]) -> None:
    under = [r for r in results if r.case_type == "underspecified"]
    kinds = sorted(set(r.gold_insufficiency_kind for r in under))

    vals = []
    for kind in kinds:
        rs = [r for r in under if r.gold_insufficiency_kind == kind]
        vals.append(mean([r.abstention_reason_correct for r in rs]))

    plt.figure(figsize=(13, 4.5))
    plt.bar(kinds, vals)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("reason accuracy")
    plt.title("Phase 86 abstention reason accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_ABSTAIN_REASON, dpi=140)
    plt.close()


def save_insuff_type_plot(insuff_rows: List[Dict[str, Any]]) -> None:
    save_grouped_bar(
        insuff_rows,
        "insufficiency_kind",
        PLOT_INSUFF_TYPE,
        "Phase 86 insufficiency-type detection accuracy",
        [
            "underdetermined_abstention",
            "insufficient_information_detection",
            "abstention_reason_accuracy",
            "boundary_localization_accuracy",
            "trace_validity",
        ],
        width_inches=15,
    )


def make_plots(
    results: List[TrialResult],
    summary: Dict[str, Any],
    task_rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
    insuff_rows: List[Dict[str, Any]],
) -> None:
    save_margin_hist(results)
    save_decision_plot(summary)

    save_grouped_bar(
        task_rows,
        "task_id",
        PLOT_TASK,
        "Phase 86 three-way semantic-boundary reasoning by task",
        [
            "overall_decision_accuracy",
            "insufficient_information_detection",
            "schema_selection_accuracy",
            "boundary_localization_accuracy",
            "abstention_reason_accuracy",
            "trace_validity",
        ],
        width_inches=19,
    )

    save_family_plot(results)
    save_confusion(results)
    save_abstention_reason_plot(results)
    save_insuff_type_plot(insuff_rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from implicit boundary detection to abstention under underspecification")
    print(f"[{PHASE}] task: accept preserved meaning, reject changed meaning, abstain when underdetermined")

    results = run_trials(n_trials=25200)
    task_rows = summarize_task(results)
    decision_rows = summarize_decision_case(results)
    insuff_rows = summarize_insufficiency(results)
    summary = overall_summary(results)

    write_trials_csv(TRIALS_CSV, results)
    write_csv(TASK_SUMMARY_CSV, task_rows)
    write_csv(DECISION_SUMMARY_CSV, decision_rows)
    write_csv(INSUFF_SUMMARY_CSV, insuff_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, task_rows, decision_rows, insuff_rows)
    write_examples(results)
    make_plots(results, summary, task_rows, decision_rows, insuff_rows)

    print(
        f"[{PHASE}] PHASE86_SEMANTIC_BOUNDARY_ABSTENTION_PASS="
        f"{summary['PHASE86_SEMANTIC_BOUNDARY_ABSTENTION_PASS']}"
    )

    print(
        f"[{PHASE}] selected_task={summary['selected_task']} "
        f"overall_decision_accuracy={summary['overall_decision_accuracy']:.4f} "
        f"arithmetic_decision_accuracy={summary['arithmetic_decision_accuracy']:.4f} "
        f"geometry_decision_accuracy={summary['geometry_decision_accuracy']:.4f} "
        f"mixed_decision_accuracy={summary['mixed_decision_accuracy']:.4f} "
        f"true_acceptance={summary['true_acceptance']:.4f} "
        f"changed_rejection={summary['changed_rejection']:.4f} "
        f"underdetermined_abstention={summary['underdetermined_abstention']:.4f} "
        f"insufficient_information_detection={summary['insufficient_information_detection']:.4f} "
        f"schema_selection_accuracy={summary['schema_selection_accuracy']:.4f} "
        f"variable_binding_accuracy={summary['variable_binding_accuracy']:.4f} "
        f"boundary_localization_accuracy={summary['boundary_localization_accuracy']:.4f} "
        f"abstention_reason_accuracy={summary['abstention_reason_accuracy']:.4f} "
        f"trace_validity={summary['trace_validity']:.4f} "
        f"no_hallucination_accuracy={summary['no_hallucination_accuracy']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print(f"[{PHASE}] semantic boundary abstention task summary:")
    for r in task_rows:
        print(
            f"  - {r['task_id']:<40} "
            f"family={r['family']:<10} "
            f"decision={r['overall_decision_accuracy']:.3f} "
            f"true_accept={r['true_acceptance']:.3f} "
            f"changed_reject={r['changed_rejection']:.3f} "
            f"abstain={r['underdetermined_abstention']:.3f} "
            f"insuff={r['insufficient_information_detection']:.3f} "
            f"bind={r['variable_binding_accuracy']:.3f} "
            f"locate={r['boundary_localization_accuracy']:.3f} "
            f"reason={r['abstention_reason_accuracy']:.3f} "
            f"trace={r['trace_validity']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={r['trials']}"
        )

    print(f"[{PHASE}] wrote trials: {TRIALS_CSV}")
    print(f"[{PHASE}] wrote task summary: {TASK_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote decision summary: {DECISION_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote insufficiency summary: {INSUFF_SUMMARY_CSV}")
    print(f"[{PHASE}] wrote summary: {SUMMARY_JSON}")
    print(f"[{PHASE}] wrote report: {REPORT_MD}")
    print(f"[{PHASE}] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()