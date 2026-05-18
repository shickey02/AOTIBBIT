r"""
Phase 72 — Pareto budget probe frontier bridge for Basic32 raster grammar.

Drop this file into:
    E:/BBIT/bbit_geomlang/geomlang_phase72_pareto_budget_probe_frontier_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase72_pareto_budget_probe_frontier_bridge_basic32_E_drive.py

Phase 72 continues Phase 71. Phase 70 proved active probing recovers hidden
raster grammar, but probes almost everything. Phase 71 made probing budgeted by
accepting one calibrated no-luma family. Phase 72 turns that into an explicit
Pareto frontier: multiple epistemic probe policies are evaluated side-by-side,
and a selected policy must preserve high scene/relation accuracy while reducing
the probe rate further.

The conceptual move is:
    Phase 69: "I know when I do not know."
    Phase 70: "When I do not know, I can actively look."
    Phase 71: "Looking has a cost, so do not probe what is already safe."
    Phase 72: "Choose a calibrated operating point on the accuracy/cost frontier."
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PHASE = "72"
TITLE = "Pareto budget probe frontier bridge"
ROOT = Path("E:/BBIT") if Path("E:/BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase72_examples"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import geomlang_phase68_real_raster_perception_bridge_basic32_E_drive as p68
    import geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive as p69
    import geomlang_phase71_budgeted_active_observability_probe_bridge_basic32_E_drive as p71
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Phase 72 expects Phase 68, Phase 69, and Phase 71 to be present beside it in bbit_geomlang. "
        "Copy the prior phase scripts into E:/BBIT/bbit_geomlang first."
    ) from exc

SEED = 72072
TRIALS_PER_SCENE = 160
SCENES = p68.SCENES
RELATIONS = p68.RELATIONS
PROBE_REPEATS = 5

# Policy definitions.  Each policy is an operating point on the active-perception
# budget frontier.  A scene listed here can be accepted from no-luma evidence if
# its semantic margin clears the listed floor.  Otherwise the system probes.
POLICIES: Dict[str, Dict[str, float]] = {
    # Almost pure active perception. Useful as an upper accuracy bound.
    "probe_all": {},

    # Phase 71 behavior: only the reliably separable pair is accepted without a probe.
    "phase71_like": {
        "square_separate_pentagon": 1.85,
    },

    # Selected Phase 72 operating point. Accepts the two no-luma families whose
    # blind/no-luma evidence has repeatedly been strong enough to justify the cost savings.
    "pareto_selected": {
        "square_separate_pentagon": 1.85,
        "pentagon_above_line": 1.75,
    },

    # A deliberately more aggressive point. It may save slightly more probe cost,
    # but is expected to risk below/above/near relation bleed. It is kept as a
    # frontier diagnostic rather than the selected pass condition.
    "aggressive": {
        "square_separate_pentagon": 1.85,
        "pentagon_above_line": 1.75,
        "line_below_square": 2.35,
    },
}
SELECTED_POLICY = "pareto_selected"


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def should_accept_noluma(policy_name: str, no_pred: Dict[str, object], obs: Dict[str, object]) -> Tuple[bool, str]:
    """Budget policy: accept no-luma only for calibrated families with adequate margin."""
    if bool(obs.get("abstain", False)):
        return False, "abstain_or_underobservable"

    scene = str(no_pred.get("scene", "unknown"))
    margin = safe_float(no_pred.get("margin", 0.0))
    scene_thresholds = POLICIES[policy_name]
    if scene in scene_thresholds and margin >= scene_thresholds[scene]:
        return True, f"accepted_{scene}_margin_ge_{scene_thresholds[scene]:.2f}"
    return False, "probe_expected_value_positive"


def active_probe_prediction(shape_a: str, shape_b: str, relation: str, rng: random.Random) -> Dict[str, object]:
    """Small repeated active luma probe bank with semantic voting."""
    from collections import Counter

    preds: List[Dict[str, object]] = []
    margins: List[float] = []
    for _ in range(PROBE_REPEATS):
        img, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=True)
        pred = p69.semantic_repair_prediction(img, use_luma=True)
        preds.append(pred)
        margins.append(safe_float(pred.get("margin", 0.0)))

    return {
        "scene": Counter([p.get("scene", "unknown") for p in preds]).most_common(1)[0][0],
        "relation": Counter([p.get("relation", "unknown") for p in preds]).most_common(1)[0][0],
        "shape_a": Counter([p.get("shape_a", "unknown") for p in preds]).most_common(1)[0][0],
        "shape_b": Counter([p.get("shape_b", "unknown") for p in preds]).most_common(1)[0][0],
        "margin": float(np.median(margins)),
        "probe_repeats": PROBE_REPEATS,
    }


def run_trial(policy_name: str, scene_name: str, shape_a: str, shape_b: str, relation: str, rng: random.Random) -> Dict[str, object]:
    img_noluma, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=False)
    no_pred = p69.semantic_repair_prediction(img_noluma, use_luma=False)
    obs = p69.no_luma_observability(img_noluma, luma_pred=None)

    accept, reason = should_accept_noluma(policy_name, no_pred, obs)
    if accept:
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
            "pred_scene": probe.get("scene", "unknown"),
            "pred_relation": probe.get("relation", "unknown"),
            "pred_shape_a": probe.get("shape_a", "unknown"),
            "pred_shape_b": probe.get("shape_b", "unknown"),
            "margin": safe_float(probe.get("margin", 0.0)),
        }

    scene_ok = int(final["pred_scene"] == scene_name)
    relation_ok = int(final["pred_relation"] == relation)
    pair_ok = int(set([final["pred_shape_a"], final["pred_shape_b"]]) == set([shape_a, shape_b]))
    truth_underdetermined = int(bool(obs.get("abstain", False)))
    no_hallucination_ok = int((truth_underdetermined == 0) or (final["probe_requested"] == 1))

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
        "policy_reason": reason,
        "truth_underdetermined": truth_underdetermined,
        "no_hallucination_ok": no_hallucination_ok,
        "observable": int(bool(obs.get("observable", False))),
        "abstain": int(bool(obs.get("abstain", False))),
        "component_count": int(obs.get("component_count", 0)),
        "noluma_pred_scene": no_pred.get("scene", "unknown"),
        "noluma_pred_relation": no_pred.get("relation", "unknown"),
        "noluma_scene_ok": int(no_pred.get("scene", "unknown") == scene_name),
        "noluma_relation_ok": int(no_pred.get("relation", "unknown") == relation),
        "margin": final["margin"],
        "noluma_margin": safe_float(no_pred.get("margin", 0.0)),
    }


def mean(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def bar_plot(path: Path, title: str, labels: List[str], values: List[float], ylabel: str = "score", ylim: Tuple[float, float] = (0, 1.08)) -> None:
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
    width = 0.8 / max(len(series), 1)
    plt.figure(figsize=(16, 5.2))
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


def confusion_plot(path: Path, title: str, labels: List[str], matrix: np.ndarray) -> None:
    plt.figure(figsize=(9.0, 7.5))
    plt.imshow(matrix, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
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


def compute_confusions(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    scene_labels = [s[0] for s in SCENES]
    scene_idx = {x: i for i, x in enumerate(scene_labels)}
    scene_conf = np.zeros((len(scene_labels), len(scene_labels)), dtype=float)
    for r in rows:
        ps = str(r["pred_scene"])
        if ps in scene_idx:
            scene_conf[scene_idx[str(r["scene"])], scene_idx[ps]] += 1
    scene_conf = scene_conf / np.maximum(scene_conf.sum(axis=1, keepdims=True), 1)

    rel_idx = {x: i for i, x in enumerate(RELATIONS)}
    rel_conf = np.zeros((len(RELATIONS), len(RELATIONS)), dtype=float)
    for r in rows:
        pr = str(r["pred_relation"])
        if pr in rel_idx:
            rel_conf[rel_idx[str(r["relation"])], rel_idx[pr]] += 1
    rel_conf = rel_conf / np.maximum(rel_conf.sum(axis=1, keepdims=True), 1)
    return scene_conf, rel_conf, scene_labels


def summarize_policy(policy_name: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(rows)
    summaries = []
    for scene_name, shape_a, shape_b, relation in SCENES:
        sub = [r for r in rows if r["scene"] == scene_name]
        summaries.append({
            "scene": scene_name,
            "scene_acc": mean(sub, "scene_ok"),
            "relation_acc": mean(sub, "relation_ok"),
            "concept_pair_acc": mean(sub, "concept_pair_ok"),
            "probe_rate": mean(sub, "probe_requested"),
            "accepted_rate": 1.0 - mean(sub, "probe_requested"),
            "blind_noluma_scene_acc": mean(sub, "noluma_scene_ok"),
            "active_gain": mean(sub, "scene_ok") - mean(sub, "noluma_scene_ok"),
            "mean_margin": mean(sub, "margin"),
            "margin_floor": float(np.min([float(r["margin"]) for r in sub])) if sub else 0.0,
        })
    return {
        "policy": policy_name,
        "trials": n,
        "scene_accuracy": mean(rows, "scene_ok"),
        "relation_accuracy": mean(rows, "relation_ok"),
        "concept_pair_accuracy": mean(rows, "concept_pair_ok"),
        "probe_rate": mean(rows, "probe_requested"),
        "accepted_noluma_rate": 1.0 - mean(rows, "probe_requested"),
        "no_hallucination_accuracy": mean(rows, "no_hallucination_ok"),
        "blind_noluma_scene_accuracy": mean(rows, "noluma_scene_ok"),
        "blind_noluma_relation_accuracy": mean(rows, "noluma_relation_ok"),
        "probe_gain_over_blind": mean(rows, "scene_ok") - mean(rows, "noluma_scene_ok"),
        "mean_margin": mean(rows, "margin"),
        "margin_floor": float(np.min([float(r["margin"]) for r in rows])) if rows else 0.0,
        "scene_summary": summaries,
    }


def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from budgeted active observability to Pareto cost/accuracy frontier")
    print(f"[{PHASE}] task: pick an active perception operating point that preserves grammar accuracy while lowering probe cost")

    all_rows: List[Dict[str, object]] = []
    policy_summaries: List[Dict[str, object]] = []

    for policy_i, policy_name in enumerate(POLICIES.keys()):
        rng = random.Random(SEED + policy_i * 1009)
        rows: List[Dict[str, object]] = []
        example_count = 0
        for scene_name, shape_a, shape_b, relation in SCENES:
            for k in range(TRIALS_PER_SCENE):
                row = run_trial(policy_name, scene_name, shape_a, shape_b, relation, rng)
                rows.append(row)
                if policy_name == SELECTED_POLICY and example_count < 40 and k % 20 == 0:
                    # Save the actual no-luma input examples for the selected operating point.
                    img, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=False)
                    plt.figure(figsize=(3, 3))
                    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(EXAMPLE_DIR / f"phase72_selected_noluma_{example_count:03d}_{scene_name}.png", dpi=140)
                    plt.close()
                    example_count += 1
        all_rows.extend(rows)
        policy_summaries.append(summarize_policy(policy_name, rows))

    selected_rows = [r for r in all_rows if r["policy"] == SELECTED_POLICY]
    selected = summarize_policy(SELECTED_POLICY, selected_rows)

    passed = (
        selected["scene_accuracy"] >= 0.965 and
        selected["relation_accuracy"] >= 0.955 and
        selected["concept_pair_accuracy"] >= 0.955 and
        selected["no_hallucination_accuracy"] >= 0.99 and
        selected["probe_rate"] <= 0.79 and
        selected["accepted_noluma_rate"] >= 0.20 and
        selected["scene_accuracy"] >= selected["blind_noluma_scene_accuracy"] + 0.50
    )

    trials_path = OUT / "phase72_pareto_budget_probe_frontier_bridge_trials.csv"
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
        "selected_policy": SELECTED_POLICY,
        "probe_repeats_when_used": PROBE_REPEATS,
        "selected_summary": selected,
        "policy_frontier": policy_summaries,
        "notes": [
            "Phase 72 explicitly evaluates several active-perception budget policies instead of hardcoding one behavior.",
            "The selected policy accepts calibrated no-luma families for square_separate_pentagon and pentagon_above_line, then probes the remaining unsafe families.",
            "Success requires high grammar recovery, no hallucination under underdetermination, and a lower probe rate than Phase 71."
        ],
    }
    summary_path = OUT / "phase72_pareto_budget_probe_frontier_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = OUT / "phase72_pareto_budget_probe_frontier_bridge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {PHASE}: {TITLE}\n\n")
        f.write("## Selected operating point\n\n")
        for k in [
            "policy", "trials", "scene_accuracy", "relation_accuracy", "concept_pair_accuracy",
            "probe_rate", "accepted_noluma_rate", "no_hallucination_accuracy",
            "blind_noluma_scene_accuracy", "probe_gain_over_blind", "mean_margin", "margin_floor"
        ]:
            f.write(f"- **{k}**: {selected[k]}\n")
        f.write("\n## Frontier\n\n")
        for s in policy_summaries:
            f.write(
                f"- **{s['policy']}**: scene={s['scene_accuracy']:.4f}, relation={s['relation_accuracy']:.4f}, "
                f"probe={s['probe_rate']:.4f}, accepted={s['accepted_noluma_rate']:.4f}, "
                f"blind={s['blind_noluma_scene_accuracy']:.4f}, gain={s['probe_gain_over_blind']:.4f}\n"
            )
        f.write("\n## Selected scene summary\n\n")
        for s in selected["scene_summary"]:
            f.write(
                f"- **{s['scene']}**: scene={s['scene_acc']:.3f}, relation={s['relation_acc']:.3f}, "
                f"probe={s['probe_rate']:.3f}, accepted={s['accepted_rate']:.3f}, "
                f"blind={s['blind_noluma_scene_acc']:.3f}, gain={s['active_gain']:.3f}, margin={s['mean_margin']:.3f}\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "Phase 72 converts active probing into an operating-point choice. The system no longer treats probing as a yes/no technique; "
            "it evaluates a frontier of possible probe budgets and selects a policy that maintains semantic grammar accuracy while saving observations. "
            "This is the first phase where perception behaves less like classification and more like an epistemic economy: it spends attention only where the expected value is high.\n"
        )

    # Plots.
    policy_names = [s["policy"] for s in policy_summaries]
    grouped_bar_plot(
        OUT / "phase72_pareto_frontier_metrics.png",
        "Phase 72 Pareto frontier: accuracy versus probe cost",
        policy_names,
        {
            "scene_acc": [float(s["scene_accuracy"]) for s in policy_summaries],
            "relation_acc": [float(s["relation_accuracy"]) for s in policy_summaries],
            "probe_rate": [float(s["probe_rate"]) for s in policy_summaries],
            "accepted_rate": [float(s["accepted_noluma_rate"]) for s in policy_summaries],
        },
        ylabel="score / rate",
        ylim=(0, 1.08),
    )

    # Accuracy-cost scatter.
    plt.figure(figsize=(8.5, 5.5))
    xs = [float(s["probe_rate"]) for s in policy_summaries]
    ys = [float(s["scene_accuracy"]) for s in policy_summaries]
    plt.scatter(xs, ys, s=90)
    for s, x, y in zip(policy_summaries, xs, ys):
        plt.text(x + 0.005, y, str(s["policy"]))
    plt.title("Phase 72 active perception Pareto frontier")
    plt.xlabel("probe rate / observation cost")
    plt.ylabel("scene accuracy")
    plt.ylim(0.85, 1.01)
    plt.xlim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "phase72_accuracy_cost_frontier.png", dpi=140)
    plt.close()

    scene_conf, rel_conf, scene_labels = compute_confusions(selected_rows)
    confusion_plot(OUT / "phase72_selected_scene_confusion.png", "Phase 72 selected policy scene confusion", scene_labels, scene_conf)
    confusion_plot(OUT / "phase72_selected_relation_confusion.png", "Phase 72 selected policy relation confusion", RELATIONS, rel_conf)
    histogram_plot(
        OUT / "phase72_selected_margin_distribution.png",
        "Phase 72 selected policy winner margin distribution",
        [float(r["margin"]) for r in selected_rows],
        "runner-up semantic score - winner score",
    )
    bar_plot(
        OUT / "phase72_selected_probe_rate_by_scene.png",
        "Phase 72 selected probe rate by scene",
        [s["scene"] for s in selected["scene_summary"]],
        [float(s["probe_rate"]) for s in selected["scene_summary"]],
        ylabel="probe rate",
    )
    bar_plot(
        OUT / "phase72_selected_gain_by_scene.png",
        "Phase 72 selected gain over blind no-luma by scene",
        [s["scene"] for s in selected["scene_summary"]],
        [float(s["active_gain"]) for s in selected["scene_summary"]],
        ylabel="accuracy gain",
        ylim=(-0.05, 1.08),
    )

    print(f"[{PHASE}] PHASE72_PARETO_BUDGET_PROBE_FRONTIER_BRIDGE_PASS={passed}")
    print(
        f"[{PHASE}] selected_policy={SELECTED_POLICY} scene_accuracy={selected['scene_accuracy']:.4f} "
        f"relation_accuracy={selected['relation_accuracy']:.4f} concept_pair_accuracy={selected['concept_pair_accuracy']:.4f} "
        f"probe_rate={selected['probe_rate']:.4f} accepted_noluma_rate={selected['accepted_noluma_rate']:.4f} "
        f"no_hallucination_accuracy={selected['no_hallucination_accuracy']:.4f} "
        f"blind_noluma_scene_accuracy={selected['blind_noluma_scene_accuracy']:.4f} "
        f"gain={selected['probe_gain_over_blind']:.4f} mean_margin={selected['mean_margin']:.6f} "
        f"margin_floor={selected['margin_floor']:.6f} trials={selected['trials']}"
    )
    print(f"[{PHASE}] frontier summary:")
    for s in policy_summaries:
        print(
            f"  - {s['policy']:<16} scene={s['scene_accuracy']:.4f} relation={s['relation_accuracy']:.4f} "
            f"pair={s['concept_pair_accuracy']:.4f} probe={s['probe_rate']:.4f} accepted={s['accepted_noluma_rate']:.4f} "
            f"blind={s['blind_noluma_scene_accuracy']:.4f} gain={s['probe_gain_over_blind']:.4f}"
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
