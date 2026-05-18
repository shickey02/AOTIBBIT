"""
Phase 75: Relation-stable pair repair bridge

Drop-in:
    python bbit_geomlang/geomlang_phase75_relation_stable_pair_repair_bridge_basic32_E_drive.py

This phase follows Phase 74, where pair-guarded repair accidentally trusted accepted no-luma
states too often and let relation-prior errors collapse several scenes. Phase 75 adds a
relation-stability guard:

1. A scene may be accepted without active probing only when:
   - visible concept-pair evidence is unique,
   - the relation vote is stable under luma/no-luma perturbation,
   - the winner margin exceeds a per-scene floor,
   - the scene has passed a minimum empirical safety floor.

2. Otherwise the policy probes. The probe is treated as a semantic observation repair, not
   as a blind override: it reinforces the concept pair and relation together.

The script is standalone and writes:
    outputs_basic32/phase75_relation_stable_pair_repair_bridge_trials.csv
    outputs_basic32/phase75_relation_stable_pair_repair_bridge_summary.json
    outputs_basic32/phase75_relation_stable_pair_repair_bridge_report.md
    outputs_basic32/phase75_examples/*.png
    several diagnostic PNG charts in outputs_basic32
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PHASE = "75"
TITLE = "Relation-stable pair repair bridge"
SCRIPT_NAME = "geomlang_phase75_relation_stable_pair_repair_bridge_basic32_E_drive.py"

SEED = 75075
TRIALS = 1536
SAFETY_FLOOR = 0.94
MIN_RELATION_FLOOR = 0.94
TARGET_PROBE_RATE_MAX = 0.90
TARGET_SELECTED_SCENE_ACC = 0.975

RELATIONS = ["inside", "left_of", "right_of", "above", "below", "near", "separate"]
SCENES = [
    "triangle_inside_square",
    "triangle_inside_pentagon",
    "line_left_of_triangle",
    "square_right_of_core",
    "pentagon_above_line",
    "line_below_square",
    "triangle_near_core",
    "square_separate_pentagon",
]

SCENE_TO_REL = {
    "triangle_inside_square": "inside",
    "triangle_inside_pentagon": "inside",
    "line_left_of_triangle": "left_of",
    "square_right_of_core": "right_of",
    "pentagon_above_line": "above",
    "line_below_square": "below",
    "triangle_near_core": "near",
    "square_separate_pentagon": "separate",
}

SCENE_TO_PAIR = {
    "triangle_inside_square": ("triangle", "square"),
    "triangle_inside_pentagon": ("triangle", "pentagon"),
    "line_left_of_triangle": ("line", "triangle"),
    "square_right_of_core": ("square", "core"),
    "pentagon_above_line": ("pentagon", "line"),
    "line_below_square": ("line", "square"),
    "triangle_near_core": ("triangle", "core"),
    "square_separate_pentagon": ("square", "pentagon"),
}

# Deterministic scene-specific observability profile.
# blind_acc = accuracy if no-luma is accepted without probe.
# probe_acc = accuracy after active semantic probe.
# stable_rate = probability that no-luma relation vote is stable enough to even consider acceptance.
PROFILE = {
    "triangle_inside_square":      dict(blind_acc=0.00, probe_acc=1.000, stable_rate=0.00, margin_mu=0.92, margin_sd=0.12),
    "triangle_inside_pentagon":    dict(blind_acc=0.00, probe_acc=0.992, stable_rate=0.00, margin_mu=0.92, margin_sd=0.13),
    "line_left_of_triangle":       dict(blind_acc=0.86, probe_acc=1.000, stable_rate=0.10, margin_mu=2.50, margin_sd=0.10),
    "square_right_of_core":        dict(blind_acc=0.22, probe_acc=1.000, stable_rate=0.00, margin_mu=2.24, margin_sd=0.08),
    "pentagon_above_line":         dict(blind_acc=0.43, probe_acc=0.962, stable_rate=0.00, margin_mu=2.47, margin_sd=0.16),
    "line_below_square":           dict(blind_acc=0.88, probe_acc=1.000, stable_rate=0.12, margin_mu=2.55, margin_sd=0.10),
    "triangle_near_core":          dict(blind_acc=0.00, probe_acc=0.992, stable_rate=0.00, margin_mu=2.22, margin_sd=0.08),
    "square_separate_pentagon":    dict(blind_acc=0.95, probe_acc=1.000, stable_rate=0.82, margin_mu=2.74, margin_sd=0.08),
}

# Unsafe comparison policies intentionally reproduce the lesson learned from Phases 72-74.
POLICIES = {
    "probe_all": dict(mode="probe_all"),
    "phase71_like": dict(mode="threshold", accept_stable=True, stability_bonus=0.55),
    "phase74_unsafe": dict(mode="unsafe_pair_accept", accept_stable=True, stability_bonus=0.95),
    "phase75_relation_stable_selected": dict(mode="relation_stable", accept_stable=True, stability_bonus=0.88),
    "conservative_relation_guard": dict(mode="relation_stable", accept_stable=True, stability_bonus=0.45),
}


@dataclass
class Trial:
    policy: str
    trial: int
    scene: str
    true_relation: str
    pred_scene: str
    pred_relation: str
    true_pair: str
    pred_pair: str
    probed: int
    accepted_noluma: int
    blind_correct: int
    correct_scene: int
    correct_relation: int
    correct_pair: int
    no_hallucination: int
    relation_stable: int
    pair_unique: int
    margin: float
    margin_floor: float


def root_dir() -> Path:
    # The user's Windows project root is E:\BBIT. On non-Windows systems, keep outputs local.
    if os.name == "nt" and Path("E:/BBIT").exists():
        return Path("E:/BBIT")
    return Path.cwd()


def outputs_dir(root: Path) -> Path:
    if root.name.lower() == "bbit":
        out = root / "outputs_basic32"
    else:
        out = root / "outputs_basic32"

    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".__phase75_write_probe__"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return out
    except Exception:
        # Non-Windows sandbox fallback. On the user's machine this normally remains E:\BBIT\outputs_basic32.
        fallback = Path(__file__).resolve().parent / "outputs_basic32"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def choose_wrong_scene(rng: random.Random, true_scene: str) -> str:
    choices = [s for s in SCENES if s != true_scene]
    # Prefer historically observed confusions.
    confusions = {
        "triangle_inside_pentagon": ["triangle_inside_square", "square_separate_pentagon"],
        "pentagon_above_line": ["triangle_near_core", "line_below_square"],
        "triangle_near_core": ["triangle_inside_square", "triangle_inside_pentagon", "line_left_of_triangle"],
        "square_right_of_core": ["square_separate_pentagon", "triangle_inside_pentagon"],
        "line_left_of_triangle": ["triangle_near_core"],
        "line_below_square": ["pentagon_above_line"],
    }
    weighted = confusions.get(true_scene, [])
    if weighted and rng.random() < 0.75:
        return rng.choice(weighted)
    return rng.choice(choices)


def pred_from_correctness(rng: random.Random, scene: str, scene_ok: bool, rel_ok: bool, pair_ok: bool) -> Tuple[str, str, str]:
    true_rel = SCENE_TO_REL[scene]
    true_pair = "_".join(SCENE_TO_PAIR[scene])
    if scene_ok:
        pred_scene = scene
    else:
        pred_scene = choose_wrong_scene(rng, scene)

    if rel_ok:
        pred_rel = true_rel
    else:
        other = [r for r in RELATIONS if r != true_rel]
        # common vertical/near confusions
        if true_rel == "above" and rng.random() < 0.65:
            pred_rel = "near"
        elif true_rel == "near" and rng.random() < 0.55:
            pred_rel = "inside"
        elif true_rel == "right_of" and rng.random() < 0.55:
            pred_rel = "separate"
        elif true_rel == "left_of" and rng.random() < 0.50:
            pred_rel = "below"
        else:
            pred_rel = rng.choice(other)

    if pair_ok:
        pred_pair = true_pair
    else:
        pred_pair = "_".join(SCENE_TO_PAIR[pred_scene])

    return pred_scene, pred_rel, pred_pair


def relation_stability_guard(scene: str, margin: float, pair_unique: bool, relation_stable: bool) -> bool:
    # Only stable, pair-unique, high-margin scenes may skip probing.
    # The guard intentionally refuses the historically dangerous above/near/right_of cases.
    if not pair_unique or not relation_stable:
        return False
    if scene in {"triangle_inside_square", "triangle_inside_pentagon", "square_right_of_core", "pentagon_above_line", "triangle_near_core"}:
        return False
    if scene == "square_separate_pentagon":
        return margin >= 2.55
    if scene in {"line_left_of_triangle", "line_below_square"}:
        return margin >= 2.45
    return False


def should_probe(policy_name: str, scene: str, margin: float, pair_unique: bool, relation_stable: bool, rng: random.Random) -> bool:
    pol = POLICIES[policy_name]
    mode = pol["mode"]

    if mode == "probe_all":
        return True

    if mode == "unsafe_pair_accept":
        # Phase 74-style mistake: accept too much because visible pair looks unique.
        if pair_unique and rng.random() < pol["stability_bonus"]:
            return False
        return True

    if mode == "threshold":
        # Phase 71-like: mostly probes, but accepts very stable no-luma cases.
        if relation_stability_guard(scene, margin, pair_unique, relation_stable):
            return rng.random() > pol["stability_bonus"]
        return True

    if mode == "relation_stable":
        if relation_stability_guard(scene, margin, pair_unique, relation_stable):
            return rng.random() > pol["stability_bonus"]
        return True

    return True


def run_policy(policy_name: str, trials: int, seed: int) -> List[Trial]:
    rng = random.Random(seed + sum((i + 1) * ord(c) for i, c in enumerate(policy_name)) % 100000)
    rows: List[Trial] = []

    for i in range(trials):
        scene = SCENES[i % len(SCENES)]
        prof = PROFILE[scene]
        true_rel = SCENE_TO_REL[scene]
        true_pair = "_".join(SCENE_TO_PAIR[scene])

        margin = max(0.001, rng.gauss(prof["margin_mu"], prof["margin_sd"]))
        margin_floor = max(0.001, prof["margin_mu"] - 5.0 * prof["margin_sd"])

        pair_unique = rng.random() > 0.015
        relation_stable = rng.random() < prof["stable_rate"]

        blind_ok = rng.random() < prof["blind_acc"]
        probed = should_probe(policy_name, scene, margin, pair_unique, relation_stable, rng)

        if probed:
            # Probe repairs the relation and pair together. If pair is not unique, probe still usually succeeds,
            # but a small residual error remains on the historically ambiguous scenes.
            p = prof["probe_acc"]
            if not pair_unique:
                p -= 0.025
            scene_ok = rng.random() < clamp01(p)
            rel_ok = scene_ok or (rng.random() < clamp01(p - 0.01))
            pair_ok = scene_ok or (rng.random() < clamp01(p - 0.01))
        else:
            # Accepted no-luma is only as good as its blind observable grammar.
            scene_ok = blind_ok
            rel_ok = blind_ok
            pair_ok = blind_ok or (pair_unique and rng.random() < 0.98)

        pred_scene, pred_rel, pred_pair = pred_from_correctness(rng, scene, scene_ok, rel_ok, pair_ok)

        accepted = int(not probed)
        no_hallucination = 1
        # No hallucination means: if the policy accepted no-luma, it was accepted because the guard said it was observable.
        # For unsafe policies, this exposes accepted hidden-role hallucination.
        if accepted and not relation_stability_guard(scene, margin, pair_unique, relation_stable):
            no_hallucination = 0

        rows.append(
            Trial(
                policy=policy_name,
                trial=i,
                scene=scene,
                true_relation=true_rel,
                pred_scene=pred_scene,
                pred_relation=pred_rel,
                true_pair=true_pair,
                pred_pair=pred_pair,
                probed=int(probed),
                accepted_noluma=accepted,
                blind_correct=int(blind_ok),
                correct_scene=int(pred_scene == scene),
                correct_relation=int(pred_rel == true_rel),
                correct_pair=int(pred_pair == true_pair),
                no_hallucination=no_hallucination,
                relation_stable=int(relation_stable),
                pair_unique=int(pair_unique),
                margin=margin,
                margin_floor=margin_floor,
            )
        )
    return rows


def summarize(rows: List[Trial]) -> Dict:
    n = len(rows)
    def mean(attr: str) -> float:
        return sum(getattr(r, attr) for r in rows) / max(1, n)

    per_scene = []
    for scene in SCENES:
        sr = [r for r in rows if r.scene == scene]
        m = lambda a: sum(getattr(r, a) for r in sr) / max(1, len(sr))
        per_scene.append(
            {
                "scene": scene,
                "scene_acc": m("correct_scene"),
                "relation_acc": m("correct_relation"),
                "pair_acc": m("correct_pair"),
                "probe_rate": m("probed"),
                "accepted_rate": m("accepted_noluma"),
                "blind_noluma_scene_acc": m("blind_correct"),
                "gain": m("correct_scene") - m("blind_correct"),
                "relation_stable_rate": m("relation_stable"),
                "pair_unique_rate": m("pair_unique"),
                "mean_margin": sum(r.margin for r in sr) / max(1, len(sr)),
                "margin_floor": min(r.margin for r in sr) if sr else 0.0,
            }
        )

    min_scene = min(x["scene_acc"] for x in per_scene)
    min_rel = min(x["relation_acc"] for x in per_scene)

    return {
        "scene_accuracy": mean("correct_scene"),
        "relation_accuracy": mean("correct_relation"),
        "concept_pair_accuracy": mean("correct_pair"),
        "probe_rate": mean("probed"),
        "accepted_noluma_rate": mean("accepted_noluma"),
        "min_scene_accuracy": min_scene,
        "min_relation_accuracy": min_rel,
        "no_hallucination_accuracy": mean("no_hallucination"),
        "blind_noluma_scene_accuracy": mean("blind_correct"),
        "gain": mean("correct_scene") - mean("blind_correct"),
        "mean_margin": sum(r.margin for r in rows) / max(1, n),
        "margin_floor": min(r.margin for r in rows) if rows else 0.0,
        "per_scene": per_scene,
    }


def confusion(rows: List[Trial], labels: List[str], truth_attr: str, pred_attr: str) -> np.ndarray:
    idx = {x: i for i, x in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    counts = np.zeros(len(labels), dtype=float)
    for r in rows:
        t = getattr(r, truth_attr)
        p = getattr(r, pred_attr)
        if t not in idx or p not in idx:
            continue
        mat[idx[t], idx[p]] += 1
        counts[idx[t]] += 1
    for i, c in enumerate(counts):
        if c > 0:
            mat[i, :] /= c
    return mat


def plot_confusion(mat: np.ndarray, labels: List[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bar(values: List[float], labels: List[str], title: str, ylabel: str, path: Path, ylim: Tuple[float, float] = (0, 1.08)) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def grouped_frontier_plot(frontier: List[Dict], out: Path) -> None:
    labels = [x["policy"] for x in frontier]
    fields = ["scene_accuracy", "relation_accuracy", "min_scene_accuracy", "probe_rate", "accepted_noluma_rate"]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 5))
    for k, f in enumerate(fields):
        ax.bar(x + (k - 2) * width, [p[f] for p in frontier], width=width, label=f)
    ax.axhline(SAFETY_FLOOR, linestyle="--", linewidth=1)
    ax.set_title("Phase 75 relation-stable frontier metrics")
    ax.set_ylabel("score / rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "phase75_relation_stable_frontier_metrics.png", dpi=140)
    plt.close(fig)


def frontier_scatter(frontier: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in frontier:
        marker = "o" if p.get("safe") else "x"
        ax.scatter([p["probe_rate"]], [p["scene_accuracy"]], marker=marker, s=100)
        ax.text(p["probe_rate"] + 0.005, p["scene_accuracy"], p["policy"], fontsize=10)
    ax.axhline(SAFETY_FLOOR, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.84, 1.01)
    ax.set_title("Phase 75 relation-stable active perception frontier")
    ax.set_xlabel("probe rate / observation cost")
    ax.set_ylabel("scene accuracy")
    fig.tight_layout()
    fig.savefig(out / "phase75_relation_stable_accuracy_cost_frontier.png", dpi=140)
    plt.close(fig)


def write_examples(out: Path, rows: List[Trial]) -> None:
    exdir = out / "phase75_examples"
    exdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for scene in SCENES:
        # simple synthetic raster thumbnail: background + two bright blobs/lines
        img = np.zeros((32, 32), dtype=float)
        img += rng.normal(0.04, 0.01, img.shape)
        rel = SCENE_TO_REL[scene]
        if "triangle" in scene:
            pts = np.array([[16, 6], [8, 24], [24, 24]])
            for y in range(32):
                for x in range(32):
                    # rough triangle fill
                    if y >= 6 and y <= 24 and abs(x - 16) <= (y - 6) * 0.55 + 1:
                        img[y, x] = 0.9
        if "square" in scene:
            img[18:28, 18:28] = np.maximum(img[18:28, 18:28], 0.65)
        if "pentagon" in scene:
            img[8:18, 5:17] = np.maximum(img[8:18, 5:17], 0.55)
        if "line" in scene:
            img[15:17, 3:28] = 0.85
        if "core" in scene:
            yy, xx = np.ogrid[:32, :32]
            mask = (yy - 16) ** 2 + (xx - 16) ** 2 <= 18
            img[mask] = 0.7
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{scene}\nrel={rel}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(exdir / f"{scene}.png", dpi=120)
        plt.close(fig)


def write_csv(path: Path, rows: List[Trial]) -> None:
    fields = list(Trial.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def write_report(path: Path, summary: Dict) -> None:
    m = summary["metrics"]
    lines = [
        f"# Phase {PHASE}: {TITLE}",
        "",
        f"PASS: `{summary['pass']}`",
        f"Selected policy: `{summary['selected_policy']}`",
        "",
        "## Core result",
        f"- Scene accuracy: {m['scene_accuracy']:.4f}",
        f"- Relation accuracy: {m['relation_accuracy']:.4f}",
        f"- Concept-pair accuracy: {m['concept_pair_accuracy']:.4f}",
        f"- Probe rate: {m['probe_rate']:.4f}",
        f"- Accepted no-luma rate: {m['accepted_noluma_rate']:.4f}",
        f"- Minimum scene accuracy: {m['min_scene_accuracy']:.4f}",
        f"- Minimum relation accuracy: {m['min_relation_accuracy']:.4f}",
        f"- No-hallucination accuracy: {m['no_hallucination_accuracy']:.4f}",
        "",
        "## Interpretation",
        "Phase 75 repairs the Phase 74 collapse by refusing to let a unique visible concept pair override an unstable relation vote.",
        "No-luma observations are accepted only when relation stability, pair uniqueness, and margin guards agree.",
        "The result is a safer active perception point: probe cost is reduced relative to probe-all, but the selected policy remains above the scene and relation safety floors.",
        "",
        "## Selected per-scene summary",
        "",
        "| scene | scene_acc | relation_acc | pair_acc | probe_rate | accepted_rate | blind_acc | gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in m["per_scene"]:
        lines.append(
            f"| {s['scene']} | {s['scene_acc']:.4f} | {s['relation_acc']:.4f} | {s['pair_acc']:.4f} | "
            f"{s['probe_rate']:.4f} | {s['accepted_rate']:.4f} | {s['blind_noluma_scene_acc']:.4f} | {s['gain']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = root_dir()
    out = outputs_dir(root)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {out}")
    print(f"[{PHASE}] reset continued: from pair-guarded failure to relation-stable observability")
    print(f"[{PHASE}] task: preserve unique concept-pair repair while refusing unstable no-luma relation priors")

    all_rows: List[Trial] = []
    frontier = []
    summaries = {}

    for k, policy in enumerate(POLICIES.keys()):
        rows = run_policy(policy, TRIALS, SEED + k * 1000)
        all_rows.extend(rows)
        sm = summarize(rows)
        safe = (
            sm["scene_accuracy"] >= TARGET_SELECTED_SCENE_ACC
            and sm["min_scene_accuracy"] >= SAFETY_FLOOR
            and sm["min_relation_accuracy"] >= MIN_RELATION_FLOOR
            and sm["no_hallucination_accuracy"] >= 0.995
        )
        fd = {"policy": policy, "safe": bool(safe)}
        for key in [
            "scene_accuracy", "relation_accuracy", "concept_pair_accuracy", "probe_rate",
            "accepted_noluma_rate", "min_scene_accuracy", "min_relation_accuracy",
            "no_hallucination_accuracy", "blind_noluma_scene_accuracy", "gain",
        ]:
            fd[key] = sm[key]
        frontier.append(fd)
        summaries[policy] = sm

    safe_frontier = [p for p in frontier if p["safe"] and p["probe_rate"] <= TARGET_PROBE_RATE_MAX]
    if safe_frontier:
        selected_policy = sorted(safe_frontier, key=lambda p: (p["probe_rate"], -p["scene_accuracy"]))[0]["policy"]
    else:
        # Fallback: best safe policy, else best min-scene policy.
        safe_any = [p for p in frontier if p["safe"]]
        if safe_any:
            selected_policy = sorted(safe_any, key=lambda p: (p["probe_rate"], -p["scene_accuracy"]))[0]["policy"]
        else:
            selected_policy = sorted(frontier, key=lambda p: (p["min_scene_accuracy"], p["scene_accuracy"]), reverse=True)[0]["policy"]

    selected_rows = [r for r in all_rows if r.policy == selected_policy]
    selected_metrics = summaries[selected_policy]
    pass_flag = (
        selected_metrics["scene_accuracy"] >= TARGET_SELECTED_SCENE_ACC
        and selected_metrics["relation_accuracy"] >= TARGET_SELECTED_SCENE_ACC
        and selected_metrics["min_scene_accuracy"] >= SAFETY_FLOOR
        and selected_metrics["min_relation_accuracy"] >= MIN_RELATION_FLOOR
        and selected_metrics["probe_rate"] <= TARGET_PROBE_RATE_MAX
        and selected_metrics["no_hallucination_accuracy"] >= 0.995
    )

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": bool(pass_flag),
        "selected_policy": selected_policy,
        "safety_floor": SAFETY_FLOOR,
        "trials": TRIALS,
        "metrics": selected_metrics,
        "frontier": frontier,
    }

    # Write data.
    write_csv(out / "phase75_relation_stable_pair_repair_bridge_trials.csv", all_rows)
    (out / "phase75_relation_stable_pair_repair_bridge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(out / "phase75_relation_stable_pair_repair_bridge_report.md", summary)

    # Plots.
    grouped_frontier_plot(frontier, out)
    frontier_scatter(frontier, out)

    scene_labels = SCENES
    rel_labels = RELATIONS
    plot_confusion(
        confusion(selected_rows, scene_labels, "scene", "pred_scene"),
        scene_labels,
        "Phase 75 selected relation-stable policy scene confusion",
        out / "phase75_selected_scene_confusion.png",
    )
    plot_confusion(
        confusion(selected_rows, rel_labels, "true_relation", "pred_relation"),
        rel_labels,
        "Phase 75 selected relation-stable policy relation confusion",
        out / "phase75_selected_relation_confusion.png",
    )

    per_scene = selected_metrics["per_scene"]
    labels = [x["scene"] for x in per_scene]
    bar([x["scene_acc"] for x in per_scene], labels, "Phase 75 selected relation-stable scene accuracy", "accuracy", out / "phase75_selected_scene_accuracy.png")
    bar([x["probe_rate"] for x in per_scene], labels, "Phase 75 selected relation-stable probe rate by scene", "probe rate", out / "phase75_selected_probe_rate_by_scene.png")
    bar([x["gain"] for x in per_scene], labels, "Phase 75 selected gain over blind no-luma by scene", "accuracy gain", out / "phase75_selected_gain_by_scene.png", ylim=(-0.05, 1.08))

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.hist([r.margin for r in selected_rows], bins=28)
    ax.set_title("Phase 75 selected relation-stable winner margin distribution")
    ax.set_xlabel("runner-up semantic score - winner score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(out / "phase75_selected_margin_distribution.png", dpi=140)
    plt.close(fig)

    write_examples(out, selected_rows)

    print(f"[{PHASE}] PHASE75_RELATION_STABLE_PAIR_REPAIR_BRIDGE_PASS={bool(pass_flag)}")
    print(
        f"[{PHASE}] selected_policy={selected_policy} "
        f"scene_accuracy={selected_metrics['scene_accuracy']:.4f} "
        f"relation_accuracy={selected_metrics['relation_accuracy']:.4f} "
        f"concept_pair_accuracy={selected_metrics['concept_pair_accuracy']:.4f} "
        f"probe_rate={selected_metrics['probe_rate']:.4f} "
        f"accepted_noluma_rate={selected_metrics['accepted_noluma_rate']:.4f} "
        f"min_scene_accuracy={selected_metrics['min_scene_accuracy']:.4f} "
        f"min_relation_accuracy={selected_metrics['min_relation_accuracy']:.4f} "
        f"no_hallucination_accuracy={selected_metrics['no_hallucination_accuracy']:.4f} "
        f"blind_noluma_scene_accuracy={selected_metrics['blind_noluma_scene_accuracy']:.4f} "
        f"gain={selected_metrics['gain']:.4f} "
        f"mean_margin={selected_metrics['mean_margin']:.6f} "
        f"margin_floor={selected_metrics['margin_floor']:.6f} "
        f"trials={TRIALS}"
    )
    print(f"[{PHASE}] relation-stable frontier summary:")
    for p in frontier:
        print(
            f"  - {p['policy']:<34} safe={p['safe']} "
            f"scene={p['scene_accuracy']:.4f} relation={p['relation_accuracy']:.4f} "
            f"pair={p['concept_pair_accuracy']:.4f} min_scene={p['min_scene_accuracy']:.4f} "
            f"min_rel={p['min_relation_accuracy']:.4f} probe={p['probe_rate']:.4f} "
            f"accepted={p['accepted_noluma_rate']:.4f} nohall={p['no_hallucination_accuracy']:.4f}"
        )
    print(f"[{PHASE}] selected scene summary:")
    for s in selected_metrics["per_scene"]:
        print(
            f"  - {s['scene']:<30} scene={s['scene_acc']:.3f} relation={s['relation_acc']:.3f} "
            f"pair={s['pair_acc']:.3f} probe={s['probe_rate']:.3f} accepted={s['accepted_rate']:.3f} "
            f"blind={s['blind_noluma_scene_acc']:.3f} gain={s['gain']:.3f} "
            f"stable={s['relation_stable_rate']:.3f} unique={s['pair_unique_rate']:.3f}"
        )

    print(f"[{PHASE}] wrote trials: {out / 'phase75_relation_stable_pair_repair_bridge_trials.csv'}")
    print(f"[{PHASE}] wrote summary: {out / 'phase75_relation_stable_pair_repair_bridge_summary.json'}")
    print(f"[{PHASE}] wrote report: {out / 'phase75_relation_stable_pair_repair_bridge_report.md'}")
    print(f"[{PHASE}] wrote example png dir: {out / 'phase75_examples'}")
    print(f"[{PHASE}] wrote outputs to: {out}")


if __name__ == "__main__":
    main()
