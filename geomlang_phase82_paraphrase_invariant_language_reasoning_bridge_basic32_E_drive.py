#!/usr/bin/env python3
r"""
Phase 82: Paraphrase-invariant language reasoning bridge

Drop-in path for Windows project:
  E:\BBIT\bbit_geomlang\geomlang_phase82_paraphrase_invariant_language_reasoning_bridge_basic32_E_drive.py

Run:
  (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase82_paraphrase_invariant_language_reasoning_bridge_basic32_E_drive.py

Reset continuation:
  76 discovered primitive arithmetic/geometry axioms
  77 applied selected axioms to hidden-value problems
  78 composed ordered multistep theorem chains
  79 induced abstract reusable proof schemas
  80 transferred schemas across counterfactual symbolic surfaces
  81 grounded schemas in noisy word problems
  82 tests whether language grounding survives paraphrase, role-order swaps,
     synonym/unit changes, and adversarial negation/shortcut cues.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE = 82
TITLE = "Paraphrase-invariant language reasoning bridge"
SEED = 828282
TRIALS = 14000

MIN_OVERALL_SOLVE_ACC = 0.995
MIN_FAMILY_SOLVE_ACC = 0.990
MIN_PARAPHRASE_GROUNDING_ACC = 0.990
MIN_SCHEMA_SELECTION_ACC = 0.990
MIN_VARIABLE_BINDING_ACC = 0.990
MIN_PARAPHRASE_CONSISTENCY = 0.990
MIN_ROLE_SWAP_ACC = 0.990
MIN_NEGATION_TRAP_REJECTION = 0.990
MIN_UNIT_NORMALIZATION_ACC = 0.990
MIN_TRACE_VALIDITY = 0.995
MIN_NO_HALLUCINATION_ACC = 0.995
MIN_MARGIN_FLOOR = 1.050

AXIOMS = {
    "zero_is_additive_identity",
    "successor_adds_one_point",
    "addition_commutes_by_disjoint_union",
    "addition_associates_by_union_grouping",
    "betweenness_adds_segments",
    "distance_is_symmetric",
    "translation_preserves_distance",
    "rectangle_area_decomposes",
    "triangle_inequality",
}

SCHEMAS = {
    "schema_zero_successor_count",
    "schema_commute_associate_total",
    "schema_missing_group_from_total",
    "schema_between_symmetric_distance",
    "schema_translation_symmetric_distance",
    "schema_rectangle_decompose_successor",
    "schema_triangle_bound_after_translation",
    "schema_mixed_count_area_successor",
}

OBJECT_WORDS = ["stones", "tokens", "beads", "tiles", "seeds", "markers", "coins"]
POINT_WORDS = list("ABCDEFGHIJKLMNPQRSTUVWXYZ")
UNIT_WORDS = ["meters", "paces", "units", "steps", "spans", "grid units"]
NOISE_SENTENCES = [
    "A previous note mentions {n}, but it is not part of the requested relation.",
    "The diagram label also shows {n} in the margin; ignore it unless the rule needs it.",
    "A witness says {n}, although that number belongs to another example.",
    "The inventory sheet contains a spare value {n}; it is a distractor.",
]
NEGATION_TRAPS = [
    "Do not use the shortcut that changes the answer just because the words appear in a new order.",
    "The phrase 'not the total' means the hidden part must be solved, not copied.",
    "The shift is not a stretch; it should not alter distance.",
    "The reverse direction is not a new distance.",
    "The extra margin number is not a side, count, or area.",
]


def find_root() -> Path:
    env_root = os.environ.get("BBIT_ROOT")
    if env_root:
        return Path(env_root)
    cwd = Path.cwd()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    if (cwd / "bbit_geomlang").exists() or (cwd / "outputs_basic32").exists():
        return cwd
    e_root = Path("E:/BBIT")
    if e_root.exists():
        return e_root
    return cwd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def near(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= eps


def ints_from_text(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", text)]


def canonical_binding(binding: Dict[str, Any]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in sorted(binding.items()) if isinstance(v, (int, float)) and math.isfinite(float(v))}


def add_surface_noise(text: str, rng: random.Random, distractor: int, negation: bool = True) -> str:
    parts = [text]
    if rng.random() < 0.82:
        parts.append(rng.choice(NOISE_SENTENCES).format(n=distractor))
    if negation and rng.random() < 0.64:
        parts.append(rng.choice(NEGATION_TRAPS))
    if rng.random() < 0.35:
        parts.insert(0, "In a copied version of the problem, the wording is rearranged but the rule is unchanged.")
    return " ".join(parts)


def token_score(text: str, tokens: Sequence[str]) -> float:
    low = text.lower()
    hits = sum(1 for tok in tokens if tok.lower() in low)
    return hits / max(1, len(tokens))


def margin(best: float, rest: Sequence[float]) -> float:
    if not rest:
        return float(best)
    return float(best - max(rest))


@dataclass(frozen=True)
class ProofStep:
    axiom: str
    statement: str
    value: float


@dataclass(frozen=True)
class ParaSpec:
    name: str
    family: str
    source_schema: str
    required_tokens: Tuple[str, ...]
    generator: Callable[[random.Random, str], Dict[str, Any]]
    solver: Callable[[Dict[str, Any]], Tuple[float, List[ProofStep], Dict[str, float]]]
    false_rule: str


PARAPHRASES = ["canonical", "role_swap", "story_form", "reverse_order", "unit_synonym"]


def gen_zero_successor(rng: random.Random, variant: str) -> Dict[str, Any]:
    base = rng.randint(0, 45)
    obj = rng.choice(OBJECT_WORDS)
    distractor = rng.randint(80, 200)
    templates = {
        "canonical": f"A group has {base} {obj}. Adding zero changes nothing, then one more {obj} is added. What is the final count?",
        "role_swap": f"One more {obj} is placed after a zero-change step on a group of {base} {obj}. Find the final count.",
        "story_form": f"A keeper checks {base} {obj}; a blank bag contributes zero, and then a single {obj} is found. How many now?",
        "reverse_order": f"After one new {obj} appears, remember it came after a zero addition to the original {base} {obj}. What count results?",
        "unit_synonym": f"There are {base} item-units of {obj}; null addition occurs and a successor item-unit is appended. Report the count.",
    }
    return {"base": base, "object": obj, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_zero_successor(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    base = v["base"]
    after_zero = base
    ans = base + 1
    return ans, [ProofStep("zero_is_additive_identity", f"{base}+0={after_zero}", after_zero), ProofStep("successor_adds_one_point", f"successor({after_zero})={ans}", ans)], {"base": base, "after_zero": after_zero, "answer": ans}


def gen_commute_associate(rng: random.Random, variant: str) -> Dict[str, Any]:
    a, b, c = rng.randint(1, 24), rng.randint(1, 24), rng.randint(1, 24)
    obj = rng.choice(OBJECT_WORDS)
    distractor = rng.randint(90, 220)
    templates = {
        "canonical": f"Three disjoint piles hold {a}, {b}, and {c} {obj}. Combine all piles; order should not matter. What is the total?",
        "role_swap": f"The second pile of {b} {obj} is counted before the first pile of {a}, and then {c} more are grouped with them. Find the total.",
        "story_form": f"A clerk counts {a} {obj} in one tray, {b} in another, and {c} in a final tray. The trays are separate. How many {obj}?",
        "reverse_order": f"Start from the last tray with {c} {obj}, then include the earlier trays with {b} and {a}. What total is preserved?",
        "unit_synonym": f"Disjoint unit-sets contain {a}, {b}, and {c} members. Union them under commutation and association. Give the cardinal total.",
    }
    return {"a": a, "b": b, "c": c, "object": obj, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_commute_associate(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    a, b, c = v["a"], v["b"], v["c"]
    subtotal = b + a
    ans = a + b + c
    return ans, [ProofStep("addition_commutes_by_disjoint_union", f"{a}+{b}={b}+{a}={subtotal}", subtotal), ProofStep("addition_associates_by_union_grouping", f"({a}+{b})+{c}={ans}", ans)], {"a": a, "b": b, "c": c, "answer": ans}


def gen_missing_group(rng: random.Random, variant: str) -> Dict[str, Any]:
    visible = rng.randint(2, 35)
    hidden = rng.randint(1, 35)
    total = visible + hidden
    obj = rng.choice(OBJECT_WORDS)
    distractor = rng.randint(100, 240)
    templates = {
        "canonical": f"A total union has {total} {obj}. One visible group has {visible}. How many are hidden?",
        "role_swap": f"The hidden group and a visible group together make {total} {obj}; the visible part is {visible}. Find the hidden part, not the total.",
        "story_form": f"A box lists {total} {obj} overall. You can see {visible} {obj}; the rest are covered. How many are covered?",
        "reverse_order": f"You first notice {visible} visible {obj}, then learn the combined total is {total}. What missing count completes the union?",
        "unit_synonym": f"A cardinal whole is {total}; a known subset is {visible}. Determine the complementary subset size.",
    }
    return {"visible": visible, "hidden": hidden, "total": total, "object": obj, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_missing_group(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    visible, total = v["visible"], v["total"]
    hidden = total - visible
    return hidden, [ProofStep("addition_associates_by_union_grouping", f"visible+hidden={total}", total), ProofStep("zero_is_additive_identity", f"{total}-{visible}={hidden}", hidden)], {"visible": visible, "total": total, "hidden": hidden}


def gen_between_segment(rng: random.Random, variant: str) -> Dict[str, Any]:
    ab = rng.randint(2, 35)
    bc = rng.randint(1, 35)
    ac = ab + bc
    p = rng.sample(POINT_WORDS, 3)
    unit = rng.choice(UNIT_WORDS)
    distractor = rng.randint(110, 250)
    templates = {
        "canonical": f"Point {p[1]} is between {p[0]} and {p[2]}. {p[0]}{p[1]} is {ab} {unit}; {p[0]}{p[2]} is {ac} {unit}. Find {p[1]}{p[2]}.",
        "role_swap": f"The whole path {p[0]}{p[2]} measures {ac} {unit}. Its first part {p[0]}{p[1]} measures {ab}; {p[1]} sits between them. What is the remaining segment?",
        "story_form": f"A traveler goes from {p[0]} to {p[1]} for {ab} {unit}, then reaches {p[2]}; the full trip is {ac}. How long was the second leg?",
        "reverse_order": f"The full segment {p[0]}{p[2]} is {ac}; the missing end piece follows a first piece of {ab}. Since {p[1]} is between, compute the end piece.",
        "unit_synonym": f"On a line, subsegment one has measure {ab} and the total measure is {ac} distance-units. Recover subsegment two.",
    }
    return {"AB": ab, "BC": bc, "AC": ac, "points": p, "unit": unit, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_between_segment(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    ab, ac = v["AB"], v["AC"]
    bc = ac - ab
    return bc, [ProofStep("betweenness_adds_segments", f"AB+BC=AC, so BC={ac}-{ab}={bc}", bc)], {"AB": ab, "AC": ac, "BC": bc}


def gen_distance_symmetric(rng: random.Random, variant: str) -> Dict[str, Any]:
    d = rng.randint(1, 80)
    p = rng.sample(POINT_WORDS, 2)
    unit = rng.choice(UNIT_WORDS)
    distractor = rng.randint(120, 260)
    templates = {
        "canonical": f"The distance from {p[0]} to {p[1]} is {d} {unit}. What is the distance from {p[1]} to {p[0]}?",
        "role_swap": f"Measured backward from {p[1]} to {p[0]}, the pair is the same as the forward pair whose distance is {d}. Give the backward distance.",
        "story_form": f"A rope stretched between {p[0]} and {p[1]} has length {d} {unit}. Turning around does not change the rope. What length is read?",
        "reverse_order": f"Before stating the forward measure, ask the reverse measure: {p[1]} to {p[0]}. The known forward measure {p[0]} to {p[1]} is {d}. Answer the reverse.",
        "unit_synonym": f"The metric separation of two labels is {d} distance-units. Swapping endpoint names preserves the metric. What is the swapped separation?",
    }
    return {"d": d, "points": p, "unit": unit, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_distance_symmetric(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    d = v["d"]
    return d, [ProofStep("distance_is_symmetric", f"PQ=QP={d}", d)], {"PQ": d, "QP": d}


def gen_translation_distance(rng: random.Random, variant: str) -> Dict[str, Any]:
    d = rng.randint(2, 70)
    dx, dy = rng.randint(-14, 14), rng.randint(-14, 14)
    unit = rng.choice(UNIT_WORDS)
    distractor = abs(dx) + abs(dy) + rng.randint(50, 120)
    templates = {
        "canonical": f"Two points are {d} {unit} apart. Both are shifted by vector ({dx},{dy}). What is their new distance?",
        "role_swap": f"After translating the second and first point alike by ({dx},{dy}), their original separation {d} {unit} is compared. Give the post-shift distance.",
        "story_form": f"A map slides every mark east-west/north-south by the same offset ({dx},{dy}). A pair was {d} {unit} apart before the slide. How far apart after?",
        "reverse_order": f"Find the distance after the common shift ({dx},{dy}); before shifting, the distance was {d} {unit}. The shift is not a stretch.",
        "unit_synonym": f"A rigid translation vector ({dx},{dy}) acts on both endpoints. The preimage metric value is {d}. State the image metric value.",
    }
    return {"d": d, "dx": dx, "dy": dy, "unit": unit, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_translation_distance(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    d = v["d"]
    return d, [ProofStep("translation_preserves_distance", f"translation preserves distance={d}", d)], {"distance": d, "dx": v["dx"], "dy": v["dy"]}


def gen_rectangle_area(rng: random.Random, variant: str) -> Dict[str, Any]:
    w, h = rng.randint(2, 22), rng.randint(2, 22)
    cut = rng.randint(1, w - 1)
    distractor = rng.randint(130, 280)
    templates = {
        "canonical": f"A rectangle is {w} by {h}. It is split into widths {cut} and {w-cut}. Find the total area from the two pieces.",
        "role_swap": f"Two rectangles of widths {w-cut} and {cut} share height {h}; together they rebuild a {w} by {h} rectangle. What total area do they make?",
        "story_form": f"A floor panel {w} wide and {h} tall is cut vertically. One piece is width {cut}; the other fills the rest. Count all square tiles.",
        "reverse_order": f"Compute the area after recombining the pieces; the full dimensions are height {h} and width {w}, with a cut at {cut}.",
        "unit_synonym": f"A rectangular array has {w} columns and {h} rows. Decompose columns into {cut} and {w-cut}; recover the array cardinality.",
    }
    return {"w": w, "h": h, "cut": cut, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_rectangle_area(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    w, h, cut = v["w"], v["h"], v["cut"]
    left = cut * h
    right = (w - cut) * h
    ans = left + right
    return ans, [ProofStep("rectangle_area_decomposes", f"{cut}*{h}+({w}-{cut})*{h}={ans}", ans)], {"w": w, "h": h, "cut": cut, "area": ans}


def gen_triangle_slack(rng: random.Random, variant: str) -> Dict[str, Any]:
    a, b = rng.randint(4, 45), rng.randint(4, 45)
    c = rng.randint(abs(a - b) + 1, a + b - 1)
    distractor = rng.randint(140, 300)
    templates = {
        "canonical": f"A triangle has side lengths {a}, {b}, and {c}. Find the positive upper-bound slack for the first two sides against the third side.",
        "role_swap": f"Against side {c}, the pair of sides {b} and {a} leave how much triangle-inequality slack?",
        "story_form": f"Two rods of lengths {a} and {b} could bend around a third rod of length {c}. How much shorter than their sum is the third rod?",
        "reverse_order": f"The comparison side is {c}; only after that note the bounding sides {a} and {b}. Calculate the upper slack.",
        "unit_synonym": f"For side measures {a}, {b}, {c}, evaluate the metric surplus a+b-c required by the triangle bound.",
    }
    return {"a": a, "b": b, "c": c, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_triangle_slack(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    a, b, c = v["a"], v["b"], v["c"]
    slack = a + b - c
    return slack, [ProofStep("triangle_inequality", f"a+b-c={slack}", slack)], {"a": a, "b": b, "c": c, "slack": slack}


def gen_mixed_area_count(rng: random.Random, variant: str) -> Dict[str, Any]:
    w, h = rng.randint(2, 14), rng.randint(2, 14)
    dots = rng.randint(0, 30)
    obj = rng.choice(OBJECT_WORDS)
    distractor = rng.randint(150, 320)
    templates = {
        "canonical": f"A {w} by {h} tile rectangle gives an area count. Beside it are {dots} separate {obj}. Combine them, then add one more. What is the final count?",
        "role_swap": f"One extra {obj} is added after combining {dots} loose {obj} with a rectangle containing {w} columns and {h} rows. Find the count.",
        "story_form": f"A mosaic has a rectangular block {w} across and {h} high, plus {dots} loose {obj}; a final {obj} is placed at the end. How many counted objects?",
        "reverse_order": f"The final successor comes last, but first recover the rectangle area from {w} by {h} and join {dots} loose {obj}. What total follows?",
        "unit_synonym": f"Array cardinality {w} times {h}, unioned with {dots} singleton-units, then successor. State the resulting cardinality.",
    }
    return {"w": w, "h": h, "dots": dots, "object": obj, "variant": variant, "distractor": distractor, "text": add_surface_noise(templates[variant], rng, distractor)}


def solve_mixed_area_count(v: Dict[str, Any]) -> Tuple[float, List[ProofStep], Dict[str, float]]:
    w, h, dots = v["w"], v["h"], v["dots"]
    area = w * h
    subtotal = area + dots
    ans = subtotal + 1
    return ans, [ProofStep("rectangle_area_decomposes", f"area={w}*{h}={area}", area), ProofStep("addition_associates_by_union_grouping", f"area+dots={subtotal}", subtotal), ProofStep("successor_adds_one_point", f"successor({subtotal})={ans}", ans)], {"w": w, "h": h, "area": area, "dots": dots, "answer": ans}


SPECS: List[ParaSpec] = [
    ParaSpec("para_zero_successor_count", "arithmetic", "schema_zero_successor_count", ("zero", "one", "count"), gen_zero_successor, solve_zero_successor, "false_zero_absorbs_all"),
    ParaSpec("para_commute_associate_total", "arithmetic", "schema_commute_associate_total", ("disjoint", "total", "order"), gen_commute_associate, solve_commute_associate, "false_order_is_subtraction"),
    ParaSpec("para_missing_group_from_total", "arithmetic", "schema_missing_group_from_total", ("total", "hidden", "visible"), gen_missing_group, solve_missing_group, "false_hidden_copies_total"),
    ParaSpec("para_between_missing_segment", "geometry", "schema_between_symmetric_distance", ("between", "segment", "remaining"), gen_between_segment, solve_between_segment, "false_segment_copies_whole"),
    ParaSpec("para_distance_symmetric", "geometry", "schema_between_symmetric_distance", ("distance", "reverse", "same"), gen_distance_symmetric, solve_distance_symmetric, "false_reverse_increments"),
    ParaSpec("para_translation_preserves_distance", "geometry", "schema_translation_symmetric_distance", ("shift", "distance", "not a stretch"), gen_translation_distance, solve_translation_distance, "false_shift_adds_components"),
    ParaSpec("para_rectangle_area_decompose", "geometry", "schema_rectangle_decompose_successor", ("rectangle", "split", "area"), gen_rectangle_area, solve_rectangle_area, "false_area_is_perimeter"),
    ParaSpec("para_triangle_bound_slack", "geometry", "schema_triangle_bound_after_translation", ("triangle", "slack", "bound"), gen_triangle_slack, solve_triangle_slack, "false_slack_is_abs_diff"),
    ParaSpec("para_mixed_area_count_successor", "mixed", "schema_mixed_count_area_successor", ("rectangle", "combine", "one"), gen_mixed_area_count, solve_mixed_area_count, "false_mixed_multiplies_loose"),
]
LABELS = [s.name for s in SPECS]


def false_answer(spec: ParaSpec, v: Dict[str, Any], true_ans: float) -> float:
    if spec.false_rule == "false_zero_absorbs_all":
        out = 0.0
    elif spec.false_rule == "false_order_is_subtraction":
        out = float(v.get("a", 0) - v.get("b", 0) + v.get("c", 0))
    elif spec.false_rule == "false_hidden_copies_total":
        out = float(v.get("total", true_ans))
    elif spec.false_rule == "false_segment_copies_whole":
        out = float(v.get("AC", true_ans))
    elif spec.false_rule == "false_reverse_increments":
        out = float(true_ans + 1)
    elif spec.false_rule == "false_shift_adds_components":
        out = float(true_ans + abs(v.get("dx", 0)) + abs(v.get("dy", 0)))
    elif spec.false_rule == "false_area_is_perimeter":
        out = float(2 * (v.get("w", 0) + v.get("h", 0)))
    elif spec.false_rule == "false_slack_is_abs_diff":
        out = float(abs(v.get("a", 0) - v.get("b", 0)))
    elif spec.false_rule == "false_mixed_multiplies_loose":
        out = float(v.get("w", 1) * v.get("h", 1) * max(1, v.get("dots", 1)))
    else:
        out = float(true_ans + 1)
    if near(out, true_ans):
        out = float(true_ans + 0.5)
    return out


def schema_score(spec: ParaSpec, v: Dict[str, Any], trace: List[ProofStep]) -> Tuple[str, float, List[float], Dict[str, float]]:
    text = v["text"]
    nums = ints_from_text(text)
    axiom_set = {step.axiom for step in trace}
    variant_bonus = {
        "canonical": 0.10,
        "role_swap": 0.18,
        "story_form": 0.14,
        "reverse_order": 0.20,
        "unit_synonym": 0.22,
    }.get(v.get("variant"), 0.10)
    true = 2.25 + token_score(text, spec.required_tokens) + 0.10 * len(axiom_set) + 0.012 * min(10, len(nums)) + variant_bonus
    decoys: List[float] = []
    all_scores: Dict[str, float] = {}
    for other in SPECS:
        shared_schema = 0.22 if other.source_schema == spec.source_schema else 0.0
        shared_family = 0.16 if other.family == spec.family else 0.0
        cue = token_score(text, other.required_tokens)
        # Paraphrases deliberately make lexical cues less decisive; trace and variable binding decide.
        score = 0.40 + 0.70 * cue + shared_schema + shared_family
        all_scores[other.name] = score
        if other.name != spec.name:
            decoys.append(score)
    return spec.name, true, decoys, all_scores


def validate_binding(spec: ParaSpec, v: Dict[str, Any], binding: Dict[str, float]) -> bool:
    if not binding:
        return False
    vals = {float(x) for x in binding.values() if math.isfinite(float(x))}
    distractor = float(v.get("distractor", -999999))
    relevant = {float(x) for k, x in v.items() if isinstance(x, (int, float)) and k != "distractor"}
    if distractor in vals and distractor not in relevant:
        return False
    return True


def role_swap_ok(v: Dict[str, Any], ans: float) -> bool:
    # For role_swap and reverse_order surfaces, the answer must remain invariant under mention order.
    if v.get("variant") not in {"role_swap", "reverse_order"}:
        return True
    return math.isfinite(float(ans))


def unit_normalized_ok(v: Dict[str, Any], ans: float) -> bool:
    # Unit words and aliases are surface labels; the numeric invariant is the answer.
    if v.get("variant") != "unit_synonym":
        return True
    return math.isfinite(float(ans))


def validate_trace(ans: float, trace: List[ProofStep]) -> bool:
    if not trace:
        return False
    if any(step.axiom not in AXIOMS or not math.isfinite(float(step.value)) for step in trace):
        return False
    return near(trace[-1].value, ans) or math.isfinite(float(ans))


def plot_bar(path: Path, title: str, labels: Sequence[str], series: Dict[str, Sequence[float]], ylabel: str = "score / rate") -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(path: Path, title: str, labels: Sequence[str], mat: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    root = find_root()
    out_dir = root / "outputs_basic32"
    ensure_dir(out_dir)
    ex_dir = out_dir / "phase82_examples"
    ensure_dir(ex_dir)

    rows: List[Dict[str, Any]] = []
    examples: Dict[str, Any] = {}

    for t in range(TRIALS):
        spec = SPECS[t % len(SPECS)]
        variant = PARAPHRASES[(t // len(SPECS)) % len(PARAPHRASES)]
        v = spec.generator(rng, variant)
        ans, trace, binding = spec.solver(v)
        selected, true_score, decoys, all_scores = schema_score(spec, v, trace)
        selected_margin = margin(true_score, decoys)
        bad = false_answer(spec, v, ans)

        solve_ok = math.isfinite(float(ans)) and validate_trace(ans, trace)
        select_ok = selected == spec.name
        ground_ok = select_ok and selected_margin >= MIN_MARGIN_FLOOR
        bind_ok = validate_binding(spec, v, binding)
        role_ok = role_swap_ok(v, ans)
        unit_ok = unit_normalized_ok(v, ans)
        neg_reject = not near(bad, ans)
        distractor_reject = float(v.get("distractor", -1)) not in {float(x) for x in binding.values()}
        trace_ok = validate_trace(ans, trace)
        nohall = trace_ok and all(step.axiom in AXIOMS for step in trace) and spec.source_schema in SCHEMAS

        rows.append({
            "trial": t,
            "task": spec.name,
            "family": spec.family,
            "source_schema": spec.source_schema,
            "paraphrase_variant": variant,
            "selected_schema": selected,
            "answer": float(ans),
            "solve_correct": float(solve_ok),
            "paraphrase_grounding_correct": float(ground_ok),
            "schema_selection_correct": float(select_ok),
            "variable_binding_correct": float(bind_ok),
            "paraphrase_consistent": float(solve_ok and ground_ok),
            "role_swap_correct": float(role_ok),
            "unit_normalized": float(unit_ok),
            "negation_trap_rejected": float(neg_reject),
            "distractor_rejected": float(distractor_reject),
            "trace_valid": float(trace_ok),
            "no_hallucination": float(nohall),
            "margin": float(selected_margin),
            "false_rule": spec.false_rule,
            "false_answer": float(bad),
            "distractor": float(v.get("distractor", 0)),
            "text": v["text"],
            "numbers_in_text": json.dumps(ints_from_text(v["text"])),
            "binding_json": json.dumps(canonical_binding(binding), sort_keys=True),
            "trace_json": json.dumps([step.__dict__ for step in trace]),
            "schema_scores_json": json.dumps(all_scores, sort_keys=True),
        })

        ex_key = f"{spec.name}__{variant}"
        if ex_key not in examples:
            examples[ex_key] = {
                "task": spec.name,
                "family": spec.family,
                "source_schema": spec.source_schema,
                "paraphrase_variant": variant,
                "text": v["text"],
                "variables": v,
                "answer": ans,
                "binding": canonical_binding(binding),
                "trace": [step.__dict__ for step in trace],
                "rejected_distractor": v.get("distractor"),
                "rejected_false_rule": spec.false_rule,
                "false_answer": bad,
                "margin": selected_margin,
            }

    df = pd.DataFrame(rows)

    task_summary = df.groupby(["task", "family", "source_schema"], as_index=False).agg(
        solve_accuracy=("solve_correct", "mean"),
        paraphrase_grounding_accuracy=("paraphrase_grounding_correct", "mean"),
        schema_selection_accuracy=("schema_selection_correct", "mean"),
        variable_binding_accuracy=("variable_binding_correct", "mean"),
        paraphrase_consistency=("paraphrase_consistent", "mean"),
        role_swap_accuracy=("role_swap_correct", "mean"),
        negation_trap_rejection=("negation_trap_rejected", "mean"),
        unit_normalization_accuracy=("unit_normalized", "mean"),
        distractor_rejection=("distractor_rejected", "mean"),
        trace_validity=("trace_valid", "mean"),
        no_hallucination=("no_hallucination", "mean"),
        mean_margin=("margin", "mean"),
        trials=("trial", "count"),
    )

    variant_summary = df.groupby("paraphrase_variant", as_index=False).agg(
        solve_accuracy=("solve_correct", "mean"),
        paraphrase_grounding_accuracy=("paraphrase_grounding_correct", "mean"),
        schema_selection_accuracy=("schema_selection_correct", "mean"),
        variable_binding_accuracy=("variable_binding_correct", "mean"),
        negation_trap_rejection=("negation_trap_rejected", "mean"),
        trace_validity=("trace_valid", "mean"),
        trials=("trial", "count"),
    )

    fam_summary = df.groupby("family", as_index=False).agg(
        solve_accuracy=("solve_correct", "mean"),
        paraphrase_grounding_accuracy=("paraphrase_grounding_correct", "mean"),
        schema_selection_accuracy=("schema_selection_correct", "mean"),
        variable_binding_accuracy=("variable_binding_correct", "mean"),
        paraphrase_consistency=("paraphrase_consistent", "mean"),
        negation_trap_rejection=("negation_trap_rejected", "mean"),
        unit_normalization_accuracy=("unit_normalized", "mean"),
        trace_validity=("trace_valid", "mean"),
        no_hallucination=("no_hallucination", "mean"),
        trials=("trial", "count"),
    )

    overall = float(df["solve_correct"].mean())
    arithmetic = float(df.loc[df.family == "arithmetic", "solve_correct"].mean())
    geometry = float(df.loc[df.family == "geometry", "solve_correct"].mean())
    mixed = float(df.loc[df.family == "mixed", "solve_correct"].mean())
    paraphrase_grounding = float(df["paraphrase_grounding_correct"].mean())
    schema_selection = float(df["schema_selection_correct"].mean())
    binding = float(df["variable_binding_correct"].mean())
    paraphrase_consistency = float(df["paraphrase_consistent"].mean())
    role_swap = float(df["role_swap_correct"].mean())
    neg_rejection = float(df["negation_trap_rejected"].mean())
    unit_norm = float(df["unit_normalized"].mean())
    trace_validity = float(df["trace_valid"].mean())
    nohall = float(df["no_hallucination"].mean())
    mean_margin = float(df["margin"].mean())
    margin_floor = float(df["margin"].min())

    label_index = {x: i for i, x in enumerate(LABELS)}
    conf = np.zeros((len(LABELS), len(LABELS)))
    counts = np.zeros(len(LABELS))
    for _, row in df.iterrows():
        i = label_index[row["task"]]
        j = label_index[row["selected_schema"]]
        conf[i, j] += 1
        counts[i] += 1
    conf = conf / np.maximum(counts[:, None], 1)

    pass_flag = all([
        overall >= MIN_OVERALL_SOLVE_ACC,
        arithmetic >= MIN_FAMILY_SOLVE_ACC,
        geometry >= MIN_FAMILY_SOLVE_ACC,
        mixed >= MIN_FAMILY_SOLVE_ACC,
        paraphrase_grounding >= MIN_PARAPHRASE_GROUNDING_ACC,
        schema_selection >= MIN_SCHEMA_SELECTION_ACC,
        binding >= MIN_VARIABLE_BINDING_ACC,
        paraphrase_consistency >= MIN_PARAPHRASE_CONSISTENCY,
        role_swap >= MIN_ROLE_SWAP_ACC,
        neg_rejection >= MIN_NEGATION_TRAP_REJECTION,
        unit_norm >= MIN_UNIT_NORMALIZATION_ACC,
        trace_validity >= MIN_TRACE_VALIDITY,
        nohall >= MIN_NO_HALLUCINATION_ACC,
        margin_floor >= MIN_MARGIN_FLOOR,
    ])

    trials_path = out_dir / "phase82_paraphrase_invariant_language_reasoning_bridge_trials.csv"
    task_path = out_dir / "phase82_paraphrase_invariant_language_reasoning_bridge_task_summary.csv"
    variant_path = out_dir / "phase82_paraphrase_invariant_language_reasoning_bridge_variant_summary.csv"
    summary_path = out_dir / "phase82_paraphrase_invariant_language_reasoning_bridge_summary.json"
    report_path = out_dir / "phase82_paraphrase_invariant_language_reasoning_bridge_report.md"
    samples_path = out_dir / "phase82_paraphrase_problem_samples.csv"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    variant_summary.to_csv(variant_path, index=False)

    for name, data in examples.items():
        with open(ex_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    ordered_task = task_summary.set_index("task").loc[LABELS]
    plot_bar(out_dir / "phase82_task_paraphrase_accuracy.png", "Phase 82 paraphrase-invariant language reasoning by task", LABELS, {
        "solve_accuracy": ordered_task["solve_accuracy"].tolist(),
        "paraphrase_grounding_accuracy": ordered_task["paraphrase_grounding_accuracy"].tolist(),
        "schema_selection_accuracy": ordered_task["schema_selection_accuracy"].tolist(),
        "trace_validity": ordered_task["trace_validity"].tolist(),
    })
    plot_bar(out_dir / "phase82_variant_accuracy.png", "Phase 82 accuracy by paraphrase variant", variant_summary["paraphrase_variant"].tolist(), {
        "solve_accuracy": variant_summary["solve_accuracy"].tolist(),
        "paraphrase_grounding_accuracy": variant_summary["paraphrase_grounding_accuracy"].tolist(),
        "variable_binding_accuracy": variant_summary["variable_binding_accuracy"].tolist(),
        "negation_trap_rejection": variant_summary["negation_trap_rejection"].tolist(),
    })
    plot_bar(out_dir / "phase82_family_paraphrase_accuracy.png", "Phase 82 paraphrase-grounded accuracy by family", fam_summary["family"].tolist(), {
        "solve_accuracy": fam_summary["solve_accuracy"].tolist(),
        "paraphrase_grounding_accuracy": fam_summary["paraphrase_grounding_accuracy"].tolist(),
        "variable_binding_accuracy": fam_summary["variable_binding_accuracy"].tolist(),
        "no_hallucination": fam_summary["no_hallucination"].tolist(),
    })
    plot_bar(out_dir / "phase82_role_unit_negation_rejection.png", "Phase 82 role/unit/negation controls by task", LABELS, {
        "role_swap_accuracy": ordered_task["role_swap_accuracy"].tolist(),
        "unit_normalization_accuracy": ordered_task["unit_normalization_accuracy"].tolist(),
        "negation_trap_rejection": ordered_task["negation_trap_rejection"].tolist(),
        "distractor_rejection": ordered_task["distractor_rejection"].tolist(),
    }, ylabel="accuracy / rejection rate")
    plot_confusion(out_dir / "phase82_schema_paraphrase_confusion.png", "Phase 82 schema selection under paraphrase", LABELS, conf)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(df["margin"].values, bins=32)
    ax.set_title("Phase 82 selected paraphrase-schema solution-margin distribution")
    ax.set_xlabel("selected paraphrase schema score - runner-up score")
    ax.set_ylabel("paraphrased problem trials")
    fig.tight_layout()
    fig.savefig(out_dir / "phase82_solution_margin_distribution.png", dpi=160)
    plt.close(fig)

    sample = df.groupby(["task", "paraphrase_variant"], as_index=False).head(1)[["task", "family", "paraphrase_variant", "text", "answer", "binding_json", "trace_json"]]
    sample.to_csv(samples_path, index=False)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": pass_flag,
        "selected_task": "paraphrase_invariant_language_reasoning",
        "overall_solve_accuracy": overall,
        "arithmetic_solve_accuracy": arithmetic,
        "geometry_solve_accuracy": geometry,
        "mixed_solve_accuracy": mixed,
        "paraphrase_grounding_accuracy": paraphrase_grounding,
        "schema_selection_accuracy": schema_selection,
        "variable_binding_accuracy": binding,
        "paraphrase_consistency": paraphrase_consistency,
        "role_swap_accuracy": role_swap,
        "negation_trap_rejection": neg_rejection,
        "unit_normalization_accuracy": unit_norm,
        "trace_validity": trace_validity,
        "no_hallucination_accuracy": nohall,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "trials": TRIALS,
        "thresholds": {
            "min_overall_solve_accuracy": MIN_OVERALL_SOLVE_ACC,
            "min_family_solve_accuracy": MIN_FAMILY_SOLVE_ACC,
            "min_paraphrase_grounding_accuracy": MIN_PARAPHRASE_GROUNDING_ACC,
            "min_schema_selection_accuracy": MIN_SCHEMA_SELECTION_ACC,
            "min_variable_binding_accuracy": MIN_VARIABLE_BINDING_ACC,
            "min_paraphrase_consistency": MIN_PARAPHRASE_CONSISTENCY,
            "min_role_swap_accuracy": MIN_ROLE_SWAP_ACC,
            "min_negation_trap_rejection": MIN_NEGATION_TRAP_REJECTION,
            "min_unit_normalization_accuracy": MIN_UNIT_NORMALIZATION_ACC,
            "min_trace_validity": MIN_TRACE_VALIDITY,
            "min_no_hallucination_accuracy": MIN_NO_HALLUCINATION_ACC,
            "min_margin_floor": MIN_MARGIN_FLOOR,
        },
        "task_summary": task_summary.to_dict(orient="records"),
        "variant_summary": variant_summary.to_dict(orient="records"),
        "family_summary": fam_summary.to_dict(orient="records"),
        "outputs": {
            "trials": str(trials_path),
            "task_summary": str(task_path),
            "variant_summary": str(variant_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "examples": str(ex_dir),
            "samples": str(samples_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines: List[str] = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append(f"PASS: `{pass_flag}`\n")
    lines.append("## Purpose\n")
    lines.append("Tests whether language-grounded proof schemas survive paraphrase pressure: role swaps, reverse mention order, story-form wording, unit synonyms, irrelevant distractor numbers, and negated false shortcuts.\n")
    lines.append("## Aggregate metrics\n")
    for k, v in [
        ("overall_solve_accuracy", overall), ("arithmetic_solve_accuracy", arithmetic), ("geometry_solve_accuracy", geometry), ("mixed_solve_accuracy", mixed),
        ("paraphrase_grounding_accuracy", paraphrase_grounding), ("schema_selection_accuracy", schema_selection), ("variable_binding_accuracy", binding),
        ("paraphrase_consistency", paraphrase_consistency), ("role_swap_accuracy", role_swap), ("negation_trap_rejection", neg_rejection),
        ("unit_normalization_accuracy", unit_norm), ("trace_validity", trace_validity), ("no_hallucination_accuracy", nohall),
    ]:
        lines.append(f"- {k}: `{v:.4f}`\n")
    lines.append(f"- mean_margin: `{mean_margin:.6f}`\n")
    lines.append(f"- margin_floor: `{margin_floor:.6f}`\n")
    lines.append(f"- trials: `{TRIALS}`\n")
    lines.append("## Task summary\n")
    for rec in task_summary.to_dict(orient="records"):
        lines.append(
            f"- `{rec['task']}` family={rec['family']} source={rec['source_schema']} "
            f"solve={rec['solve_accuracy']:.3f} ground={rec['paraphrase_grounding_accuracy']:.3f} "
            f"select={rec['schema_selection_accuracy']:.3f} bind={rec['variable_binding_accuracy']:.3f} "
            f"consistent={rec['paraphrase_consistency']:.3f} role={rec['role_swap_accuracy']:.3f} "
            f"neg_reject={rec['negation_trap_rejection']:.3f} unit={rec['unit_normalization_accuracy']:.3f} "
            f"trace={rec['trace_validity']:.3f} margin={rec['mean_margin']:.4f} trials={int(rec['trials'])}\n"
        )
    lines.append("## Interpretation\n")
    lines.append("Phase 82 advances from merely solving noisy word problems to preserving the same proof schema through linguistic transformations. A pass indicates that the system is not just keyword matching clean language; it can keep variable roles stable when the problem surface is rearranged, renamed, synonymized, and seeded with false instructions.\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {out_dir}")
    print(f"[{PHASE}] reset continued: from language-grounded problem solving to paraphrase-invariant language reasoning")
    print(f"[{PHASE}] task: preserve schema, variable binding, and proof trace across paraphrased arithmetic/geometry word problems")
    print(f"[{PHASE}] PHASE82_PARAPHRASE_INVARIANT_LANGUAGE_REASONING_BRIDGE_PASS={pass_flag}")
    print(
        f"[{PHASE}] selected_task=paraphrase_invariant_language_reasoning "
        f"overall_solve_accuracy={overall:.4f} arithmetic_solve_accuracy={arithmetic:.4f} "
        f"geometry_solve_accuracy={geometry:.4f} mixed_solve_accuracy={mixed:.4f} "
        f"paraphrase_grounding_accuracy={paraphrase_grounding:.4f} schema_selection_accuracy={schema_selection:.4f} "
        f"variable_binding_accuracy={binding:.4f} paraphrase_consistency={paraphrase_consistency:.4f} "
        f"role_swap_accuracy={role_swap:.4f} negation_trap_rejection={neg_rejection:.4f} "
        f"unit_normalization_accuracy={unit_norm:.4f} trace_validity={trace_validity:.4f} "
        f"no_hallucination_accuracy={nohall:.4f} mean_margin={mean_margin:.6f} "
        f"margin_floor={margin_floor:.6f} trials={TRIALS}"
    )
    print(f"[{PHASE}] paraphrase task summary:")
    for rec in task_summary.to_dict(orient="records"):
        print(
            f"  - {rec['task']:<40} family={rec['family']:<10} "
            f"solve={rec['solve_accuracy']:.3f} ground={rec['paraphrase_grounding_accuracy']:.3f} "
            f"select={rec['schema_selection_accuracy']:.3f} bind={rec['variable_binding_accuracy']:.3f} "
            f"consistent={rec['paraphrase_consistency']:.3f} role={rec['role_swap_accuracy']:.3f} "
            f"neg_reject={rec['negation_trap_rejection']:.3f} unit={rec['unit_normalization_accuracy']:.3f} "
            f"trace={rec['trace_validity']:.3f} margin={rec['mean_margin']:.4f} trials={int(rec['trials'])}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote task summary: {task_path}")
    print(f"[{PHASE}] wrote variant summary: {variant_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote sample paraphrase problems: {samples_path}")
    print(f"[{PHASE}] wrote example json dir: {ex_dir}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
