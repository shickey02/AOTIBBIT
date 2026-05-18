r"""
Phase 73 — Guarded Pareto observability frontier bridge for Basic32 raster grammar.

Drop this file into:
    E:/BBIT/bbit_geomlang/geomlang_phase73_guarded_pareto_observability_frontier_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase73_guarded_pareto_observability_frontier_bridge_basic32_E_drive.py

Phase 72 was the right conceptual move but the selected Pareto point failed:
it saved probe cost by accepting too much no-luma evidence. The catastrophic
case was line_below_square, where a high-margin no-luma read was confidently
wrong because the luma channel carried the missing role distinction.

Phase 73 repairs this by making the Pareto frontier guarded rather than merely
cheap. A policy is only selectable if it satisfies global accuracy floors AND
per-scene safety floors. In other words: no scene may be sacrificed to lower the
observation bill. The selected policy is the cheapest policy on the safe frontier,
not the cheapest policy overall.

Conceptual move:
    Phase 72: "Attention has a cost; choose an operating point."
    Phase 73: "Do not buy cheap attention by creating blind spots."
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

PHASE = "73"
TITLE = "Guarded Pareto observability frontier bridge"
ROOT = Path("E:/BBIT") if Path("E:/BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase73_examples"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import geomlang_phase68_real_raster_perception_bridge_basic32_E_drive as p68
    import geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive as p69
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Phase 73 expects Phase 68 and Phase 69 to be present beside it in bbit_geomlang. "
        "Copy prior phase scripts into E:/BBIT/bbit_geomlang first."
    ) from exc

SEED = 73073
TRIALS_PER_SCENE = 192
PROBE_REPEATS = 5
SCENES = p68.SCENES
RELATIONS = p68.RELATIONS
SCENE_NAMES = [s[0] for s in SCENES]

# Guard floors: a policy is not selectable if any scene falls below these.
GLOBAL_SCENE_FLOOR = 0.975
GLOBAL_RELATION_FLOOR = 0.975
GLOBAL_PAIR_FLOOR = 0.970
MIN_SCENE_FLOOR = 0.930
MIN_RELATION_FLOOR = 0.930
NO_HALLUCINATION_FLOOR = 0.999

# Candidate policies.  The important change from Phase 72 is that candidates are
# evaluated under per-scene guardrails.  We keep unsafe candidates in the output
# so the failure mode remains visible.
POLICIES: Dict[str, Dict[str, float]] = {
    "probe_all": {},
    "phase71_like": {
        "square_separate_pentagon": 1.85,
    },
    "guarded_square_only": {
        "square_separate_pentagon": 2.10,
    },
    "accept_square_and_pentagon_above": {
        "square_separate_pentagon": 1.85,
        "pentagon_above_line": 2.20,
    },
    "phase72_like_unsafe": {
        "square_separate_pentagon": 1.85,
        "pentagon_above_line": 1.75,
        "line_below_square": 2.35,
    },
    "aggressive_unsafe": {
        "square_separate_pentagon": 1.75,
        "pentagon_above_line": 1.75,
        "line_below_square": 2.25,
        "square_right_of_core": 2.05,
    },
}


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def is_noluma_acceptable(policy_name: str, no_pred: Dict[str, object], obs: Dict[str, object]) -> Tuple[bool, str]:
    """Accept no-luma only when the policy explicitly allows this predicted family.

    Phase 72's bug was treating a high no-luma margin as generally safe.  Phase 73
    treats no-luma margin as locally valid only inside a named accepted family.
    """
    if bool(obs.get("abstain", False)):
        return False, "probe_required_underobservable"

    scene = str(no_pred.get("scene", "unknown"))
    relation = str(no_pred.get("relation", "unknown"))
    margin = safe_float(no_pred.get("margin", 0.0))
    thresholds = POLICIES[policy_name]

    if scene not in thresholds:
        return False, "probe_required_scene_not_whitelisted"

    if margin < thresholds[scene]:
        return False, f"probe_required_margin_lt_{thresholds[scene]:.2f}"

    # Semantic consistency guard.  The scene name contains the relation token in
    # these Basic32 cases; do not accept an internally inconsistent no-luma read.
    expected_relation = None
    for truth_scene, _a, _b, rel in SCENES:
        if truth_scene == scene:
            expected_relation = rel
            break
    if expected_relation is not None and relation != expected_relation:
        return False, "probe_required_inconsistent_scene_relation"

    return True, f"accepted_guarded_{scene}_margin_ge_{thresholds[scene]:.2f}"


def active_probe_prediction(shape_a: str, shape_b: str, relation: str, rng: random.Random) -> Dict[str, object]:
    preds: List[Dict[str, object]] = []
    margins: List[float] = []
    for _ in range(PROBE_REPEATS):
        img, _meta = p68.render_scene(shape_a, shape_b, relation, rng, luma=True)
        pred = p69.semantic_repair_prediction(img, use_luma=True)
        preds.append(pred)
        margins.append(safe_float(pred.get("margin", 0.0)))
    return {
        "scene": Counter([p.get("scene", "unknown") for p in preds]).most_common(1)[0][0],
        "relation": Counter([p.get("relation", "unknown") for p in preds]).most_common(1)[0][0],
        "shape_a": Counter([p.get("shape_a", "unknown") for p in preds]).most_common(1)[0][0],
        "shape_b": Counter([p.get("shape_b", "unknown") for p in preds]).most_common(1)[0][0],
        "margin": float(np.median(margins)),
    }


def run_trial(policy_name: str, scene_name: str, shape_a: str, shape_b: str, relation: str, rng: random.Random) -> Dict[str, object]:
    img_noluma, _meta = p68.render_scene(shape_a, shape_b, relation, rng, luma=False)
    no_pred = p69.semantic_repair_prediction(img_noluma, use_luma=False)
    obs = p69.no_luma_observability(img_noluma, luma_pred=None)

    accepted, reason = is_noluma_acceptable(policy_name, no_pred, obs)
    if accepted:
        final = {
            "source": "accepted_no_luma",
            "probe_requested": 0,
            "pred_scene": no_pred.get("scene", "unknown"),
            "pred_relation": no_pred.get("relation", "unknown"),
            "pred_shape_a": no_pred.get("shape_a", "unknown"),
            "pred_shape_b": no_pred.get("shape_b", "unknown"),
            "margin": safe_float(no_pred.get("margin", 0.0)),
        }
    else:
        probe = active_probe_prediction(shape_a, shape_b, relation, rng)
        final = {
            "source": "active_probe",
            "probe_requested": 1,
            "pred_scene": probe["scene"],
            "pred_relation": probe["relation"],
            "pred_shape_a": probe["shape_a"],
            "pred_shape_b": probe["shape_b"],
            "margin": safe_float(probe["margin"]),
        }

    truth_underdetermined = int(bool(obs.get("abstain", False)))
    scene_ok = int(final["pred_scene"] == scene_name)
    relation_ok = int(final["pred_relation"] == relation)
    pair_ok = int(set([final["pred_shape_a"], final["pred_shape_b"]]) == set([shape_a, shape_b]))

    return {
        "policy": policy_name,
        "scene": scene_name,
        "shape_a": shape_a,
        "shape_b": shape_b,
        "relation": relation,
        "pred_scene": final["pred_scene"],
        "pred_relation": final["pred_relation"],
        "pred_shape_a": final["pred_shape_a"],
        "pred_shape_b": final["pred_shape_b"],
        "scene_ok": scene_ok,
        "relation_ok": relation_ok,
        "concept_pair_ok": pair_ok,
        "source": final["source"],
        "probe_requested": final["probe_requested"],
        "accepted_no_luma": int(final["probe_requested"] == 0),
        "policy_reason": reason,
        "truth_underdetermined": truth_underdetermined,
        "no_hallucination_ok": int((truth_underdetermined == 0) or (final["probe_requested"] == 1)),
        "observable": int(bool(obs.get("observable", False))),
        "abstain": int(bool(obs.get("abstain", False))),
        "component_count": int(obs.get("component_count", 0)),
        "noluma_pred_scene": no_pred.get("scene", "unknown"),
        "noluma_pred_relation": no_pred.get("relation", "unknown"),
        "noluma_scene_ok": int(no_pred.get("scene", "unknown") == scene_name),
        "noluma_relation_ok": int(no_pred.get("relation", "unknown") == relation),
        "margin": safe_float(final["margin"]),
        "noluma_margin": safe_float(no_pred.get("margin", 0.0)),
    }


def scene_summaries(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for scene_name, _a, _b, _rel in SCENES:
        sub = [r for r in rows if r["scene"] == scene_name]
        out.append({
            "scene": scene_name,
            "scene_acc": mean(sub, "scene_ok"),
            "relation_acc": mean(sub, "relation_ok"),
            "concept_pair_acc": mean(sub, "concept_pair_ok"),
            "probe_rate": mean(sub, "probe_requested"),
            "accepted_rate": mean(sub, "accepted_no_luma"),
            "blind_noluma_scene_acc": mean(sub, "noluma_scene_ok"),
            "active_gain": mean(sub, "scene_ok") - mean(sub, "noluma_scene_ok"),
            "mean_margin": mean(sub, "margin"),
            "margin_floor": float(np.min([float(r["margin"]) for r in sub])) if sub else 0.0,
        })
    return out


def summarize_policy(policy_name: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    per_scene = scene_summaries(rows)
    min_scene = min(s["scene_acc"] for s in per_scene)
    min_relation = min(s["relation_acc"] for s in per_scene)
    safe = (
        mean(rows, "scene_ok") >= GLOBAL_SCENE_FLOOR and
        mean(rows, "relation_ok") >= GLOBAL_RELATION_FLOOR and
        mean(rows, "concept_pair_ok") >= GLOBAL_PAIR_FLOOR and
        mean(rows, "no_hallucination_ok") >= NO_HALLUCINATION_FLOOR and
        min_scene >= MIN_SCENE_FLOOR and
        min_relation >= MIN_RELATION_FLOOR
    )
    return {
        "policy": policy_name,
        "trials": len(rows),
        "scene_accuracy": mean(rows, "scene_ok"),
        "relation_accuracy": mean(rows, "relation_ok"),
        "concept_pair_accuracy": mean(rows, "concept_pair_ok"),
        "probe_rate": mean(rows, "probe_requested"),
        "accepted_noluma_rate": mean(rows, "accepted_no_luma"),
        "no_hallucination_accuracy": mean(rows, "no_hallucination_ok"),
        "blind_noluma_scene_accuracy": mean(rows, "noluma_scene_ok"),
        "probe_gain_over_blind": mean(rows, "scene_ok") - mean(rows, "noluma_scene_ok"),
        "mean_margin": mean(rows, "margin"),
        "margin_floor": float(np.min([float(r["margin"]) for r in rows])) if rows else 0.0,
        "min_scene_accuracy": min_scene,
        "min_relation_accuracy": min_relation,
        "safe_frontier_candidate": bool(safe),
        "scene_summary": per_scene,
    }


def confusion(rows: List[Dict[str, object]], labels: List[str], truth_key: str, pred_key: str) -> np.ndarray:
    idx = {x: i for i, x in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for r in rows:
        truth = str(r[truth_key])
        pred = str(r[pred_key])
        if truth in idx and pred in idx:
            mat[idx[truth], idx[pred]] += 1
    return mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)


def bar_plot(path: Path, title: str, labels: List[str], values: List[float], ylabel: str, ylim: Tuple[float, float] = (0, 1.08)) -> None:
    plt.figure(figsize=(16, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def grouped_bar_plot(path: Path, title: str, groups: List[str], series: Dict[str, List[float]], ylabel: str, ylim: Tuple[float, float]) -> None:
    x = np.arange(len(groups))
    width = 0.82 / max(len(series), 1)
    plt.figure(figsize=(16, 5.4))
    for i, (name, vals) in enumerate(series.items()):
        plt.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xticks(x, groups, rotation=28, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def confusion_plot(path: Path, title: str, labels: List[str], mat: np.ndarray) -> None:
    plt.figure(figsize=(9.3, 7.8))
    plt.imshow(mat, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def histogram_plot(path: Path, title: str, values: List[float], xlabel: str) -> None:
    plt.figure(figsize=(16, 4.8))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("trials")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def select_policy(summaries: List[Dict[str, object]]) -> Dict[str, object]:
    safe = [s for s in summaries if bool(s["safe_frontier_candidate"])]
    if not safe:
        # Diagnostic fallback: highest scene accuracy, then lowest probe rate.
        return sorted(summaries, key=lambda s: (float(s["scene_accuracy"]), -float(s["probe_rate"])), reverse=True)[0]
    # Cheapest safe policy = lowest probe rate; tie-break with accuracy.
    return sorted(safe, key=lambda s: (float(s["probe_rate"]), -float(s["scene_accuracy"]), -float(s["relation_accuracy"])))[0]


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from failed Pareto budget to guarded cost/accuracy frontier")
    print(f"[{PHASE}] task: lower probe cost only when every scene remains above the safety floor")

    all_rows: List[Dict[str, object]] = []
    policy_rows: Dict[str, List[Dict[str, object]]] = {}
    summaries: List[Dict[str, object]] = []

    for pi, policy_name in enumerate(POLICIES.keys()):
        rng = random.Random(SEED + pi * 1009)
        rows: List[Dict[str, object]] = []
        example_count = 0
        for scene_name, shape_a, shape_b, relation in SCENES:
            for k in range(TRIALS_PER_SCENE):
                row = run_trial(policy_name, scene_name, shape_a, shape_b, relation, rng)
                rows.append(row)
                if example_count < 32 and k % 48 == 0:
                    img, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=False)
                    plt.figure(figsize=(3, 3))
                    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(EXAMPLE_DIR / f"phase73_{policy_name}_{example_count:03d}_{scene_name}.png", dpi=140)
                    plt.close()
                    example_count += 1
        policy_rows[policy_name] = rows
        all_rows.extend(rows)
        summaries.append(summarize_policy(policy_name, rows))

    selected = select_policy(summaries)
    selected_policy = str(selected["policy"])
    selected_rows = policy_rows[selected_policy]
    passed = bool(selected["safe_frontier_candidate"]) and float(selected["accepted_noluma_rate"]) >= 0.05

    trials_path = OUT / "phase73_guarded_pareto_observability_frontier_bridge_trials.csv"
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": bool(passed),
        "seed": SEED,
        "trials_per_policy": len(selected_rows),
        "selected_policy": selected_policy,
        "probe_repeats_when_used": PROBE_REPEATS,
        "guard_floors": {
            "global_scene_floor": GLOBAL_SCENE_FLOOR,
            "global_relation_floor": GLOBAL_RELATION_FLOOR,
            "global_pair_floor": GLOBAL_PAIR_FLOOR,
            "min_scene_floor": MIN_SCENE_FLOOR,
            "min_relation_floor": MIN_RELATION_FLOOR,
            "no_hallucination_floor": NO_HALLUCINATION_FLOOR,
        },
        "selected_summary": selected,
        "policy_frontier": summaries,
        "notes": [
            "Phase 73 repairs Phase 72 by selecting the cheapest safe policy, not the cheapest policy overall.",
            "A policy is rejected if any scene drops below the per-scene safety floor.",
            "The expected failure mode from Phase 72, line_below_square being accepted without luma, is now exposed as an unsafe-frontier policy rather than selected."
        ],
    }
    summary_path = OUT / "phase73_guarded_pareto_observability_frontier_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = OUT / "phase73_guarded_pareto_observability_frontier_bridge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {PHASE}: {TITLE}\n\n")
        f.write("## Selected guarded operating point\n\n")
        for k in [
            "policy", "trials", "scene_accuracy", "relation_accuracy", "concept_pair_accuracy",
            "probe_rate", "accepted_noluma_rate", "no_hallucination_accuracy", "blind_noluma_scene_accuracy",
            "probe_gain_over_blind", "min_scene_accuracy", "min_relation_accuracy", "mean_margin", "margin_floor",
            "safe_frontier_candidate",
        ]:
            f.write(f"- **{k}**: {selected[k]}\n")
        f.write("\n## Frontier candidates\n\n")
        for s in summaries:
            f.write(
                f"- **{s['policy']}**: safe={s['safe_frontier_candidate']}, scene={s['scene_accuracy']:.4f}, "
                f"relation={s['relation_accuracy']:.4f}, min_scene={s['min_scene_accuracy']:.4f}, "
                f"probe={s['probe_rate']:.4f}, accepted={s['accepted_noluma_rate']:.4f}\n"
            )
        f.write("\n## Selected scene summary\n\n")
        for s in selected["scene_summary"]:
            f.write(
                f"- **{s['scene']}**: scene={s['scene_acc']:.3f}, relation={s['relation_acc']:.3f}, "
                f"probe={s['probe_rate']:.3f}, accepted={s['accepted_rate']:.3f}, blind={s['blind_noluma_scene_acc']:.3f}, "
                f"gain={s['active_gain']:.3f}, margin={s['mean_margin']:.3f}\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "Phase 73 turns Pareto optimization into guarded optimization. The system may save attention only when doing so does not create a local semantic collapse. "
            "This matters because a globally attractive average can hide one destroyed scene family. The new rule is: a cheap perception policy is invalid if it buys cost savings by creating a blind spot.\n"
        )

    # Plots.
    names = [s["policy"] for s in summaries]
    grouped_bar_plot(
        OUT / "phase73_guarded_frontier_metrics.png",
        "Phase 73 guarded Pareto frontier metrics",
        names,
        {
            "scene_acc": [float(s["scene_accuracy"]) for s in summaries],
            "relation_acc": [float(s["relation_accuracy"]) for s in summaries],
            "min_scene_acc": [float(s["min_scene_accuracy"]) for s in summaries],
            "probe_rate": [float(s["probe_rate"]) for s in summaries],
            "accepted_rate": [float(s["accepted_noluma_rate"]) for s in summaries],
        },
        ylabel="score / rate",
        ylim=(0, 1.08),
    )

    plt.figure(figsize=(9.5, 6.0))
    xs = [float(s["probe_rate"]) for s in summaries]
    ys = [float(s["scene_accuracy"]) for s in summaries]
    cs = ["safe" if bool(s["safe_frontier_candidate"]) else "unsafe" for s in summaries]
    for s, x, y, c in zip(summaries, xs, ys, cs):
        marker = "o" if c == "safe" else "x"
        plt.scatter([x], [y], s=100, marker=marker)
        plt.text(x + 0.006, y, str(s["policy"]))
    plt.axhline(GLOBAL_SCENE_FLOOR, linestyle="--", linewidth=1)
    plt.title("Phase 73 guarded active perception frontier")
    plt.xlabel("probe rate / observation cost")
    plt.ylabel("scene accuracy")
    plt.ylim(0.84, 1.01)
    plt.xlim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "phase73_guarded_accuracy_cost_frontier.png", dpi=140)
    plt.close()

    confusion_plot(
        OUT / "phase73_selected_scene_confusion.png",
        "Phase 73 selected guarded policy scene confusion",
        SCENE_NAMES,
        confusion(selected_rows, SCENE_NAMES, "scene", "pred_scene"),
    )
    confusion_plot(
        OUT / "phase73_selected_relation_confusion.png",
        "Phase 73 selected guarded policy relation confusion",
        RELATIONS,
        confusion(selected_rows, RELATIONS, "relation", "pred_relation"),
    )
    histogram_plot(
        OUT / "phase73_selected_margin_distribution.png",
        "Phase 73 selected guarded winner margin distribution",
        [float(r["margin"]) for r in selected_rows],
        "runner-up semantic score - winner score",
    )
    bar_plot(
        OUT / "phase73_selected_probe_rate_by_scene.png",
        "Phase 73 selected guarded probe rate by scene",
        [s["scene"] for s in selected["scene_summary"]],
        [float(s["probe_rate"]) for s in selected["scene_summary"]],
        ylabel="probe rate",
    )
    bar_plot(
        OUT / "phase73_selected_scene_accuracy.png",
        "Phase 73 selected guarded scene accuracy",
        [s["scene"] for s in selected["scene_summary"]],
        [float(s["scene_acc"]) for s in selected["scene_summary"]],
        ylabel="accuracy",
    )

    print(f"[{PHASE}] PHASE73_GUARDED_PARETO_OBSERVABILITY_FRONTIER_BRIDGE_PASS={passed}")
    print(
        f"[{PHASE}] selected_policy={selected_policy} scene_accuracy={selected['scene_accuracy']:.4f} "
        f"relation_accuracy={selected['relation_accuracy']:.4f} concept_pair_accuracy={selected['concept_pair_accuracy']:.4f} "
        f"probe_rate={selected['probe_rate']:.4f} accepted_noluma_rate={selected['accepted_noluma_rate']:.4f} "
        f"min_scene_accuracy={selected['min_scene_accuracy']:.4f} min_relation_accuracy={selected['min_relation_accuracy']:.4f} "
        f"no_hallucination_accuracy={selected['no_hallucination_accuracy']:.4f} "
        f"blind_noluma_scene_accuracy={selected['blind_noluma_scene_accuracy']:.4f} "
        f"gain={selected['probe_gain_over_blind']:.4f} mean_margin={selected['mean_margin']:.6f} "
        f"margin_floor={selected['margin_floor']:.6f} trials={selected['trials']}"
    )
    print(f"[{PHASE}] guarded frontier summary:")
    for s in summaries:
        print(
            f"  - {s['policy']:<30} safe={str(s['safe_frontier_candidate']):<5} "
            f"scene={s['scene_accuracy']:.4f} relation={s['relation_accuracy']:.4f} pair={s['concept_pair_accuracy']:.4f} "
            f"min_scene={s['min_scene_accuracy']:.4f} min_rel={s['min_relation_accuracy']:.4f} "
            f"probe={s['probe_rate']:.4f} accepted={s['accepted_noluma_rate']:.4f}"
        )
    print(f"[{PHASE}] selected scene summary:")
    for s in selected["scene_summary"]:
        print(
            f"  - {s['scene']:<28} scene={s['scene_acc']:.3f} relation={s['relation_acc']:.3f} "
            f"probe={s['probe_rate']:.3f} accepted={s['accepted_rate']:.3f} blind={s['blind_noluma_scene_acc']:.3f} "
            f"gain={s['active_gain']:.3f} margin={s['mean_margin']:.4f} floor={s['margin_floor']:.4f}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
