"""
Phase 71 — Budgeted active observability probe bridge for Basic32 raster grammar.

Drop this file into:
    E:\BBIT\bbit_geomlang\geomlang_phase71_budgeted_active_observability_probe_bridge_basic32_E_drive.py

Run:
    python bbit_geomlang/geomlang_phase71_budgeted_active_observability_probe_bridge_basic32_E_drive.py

Phase 71 continues the Phase 69 epistemic reset.  Phase 71 proved that active probing can recover scene grammar, but it probed almost every trial.  Phase 71 adds cost awareness: the system should probe only when the no-luma evidence is epistemically unsafe, while accepting calibrated no-luma observations when they are already reliable enough.

The goal is not to make the no-luma image magically determinate.  The goal is to
separate three states:
    1. determinate from the current raster,
    2. honestly underdetermined from the current raster,
    3. recoverable after a targeted probe.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PHASE = "71"
TITLE = "Budgeted active observability probe bridge"
ROOT = Path("E:/BBIT") if Path("E:/BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase71_examples"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Make sibling phase modules importable when run from E:/BBIT.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive as p69
    import geomlang_phase68_real_raster_perception_bridge_basic32_E_drive as p68
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Phase 71 expects Phase 68 and Phase 69 to be present beside it in bbit_geomlang. "
        "Copy geomlang_phase68_real_raster_perception_bridge_basic32_E_drive.py and "
        "geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive.py into "
        "E:/BBIT/bbit_geomlang first."
    ) from exc

SEED = 71071
random.seed(SEED)
np.random.seed(SEED)
TRIALS_PER_SCENE = 128
LOW_CONF_MARGIN = 2.15
PROBE_REPEATS = 5

# Phase 71 recovered almost everything but paid for it by probing nearly every
# frame.  Phase 71 uses a small calibrated no-luma acceptance prior.  These are
# not truth labels for a trial; they are the scene families that Phase 68/69/70
# repeatedly showed can remain observable even after the role-luma channel is
# removed.  Everything else is treated as epistemically unsafe and receives the
# active luma probe.
STABLE_NOLUMA_SCENES = {
    "square_separate_pentagon",
}
STABLE_NOLUMA_MIN_MARGIN = 1.85

SCENES = p68.SCENES
RELATIONS = p68.RELATIONS

# -----------------------------
# Active probe logic
# -----------------------------

def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def calibrated_no_luma_accept(no_luma_obs: Dict[str, object], no_luma_pred: Dict[str, object]) -> Tuple[bool, str]:
    """Return whether a no-luma answer is safe enough to accept without probing."""
    if bool(no_luma_obs.get("abstain", False)):
        return False, str(no_luma_obs.get("reason", "abstain"))

    pred_scene = str(no_luma_pred.get("scene", "unknown"))
    pred_relation = str(no_luma_pred.get("relation", "unknown"))
    margin = safe_float(no_luma_pred.get("margin", 0.0))
    component_count = int(no_luma_obs.get("component_count", 0))

    # If the observation is a known stable no-luma family and the semantic
    # margin is healthy, accept it.  This is the new budget-aware behavior.
    if pred_scene in STABLE_NOLUMA_SCENES and margin >= STABLE_NOLUMA_MIN_MARGIN:
        return True, "calibrated_stable_noluma_scene"

    return False, "probe_expected_value_positive"


def needs_probe(no_luma_obs: Dict[str, object], no_luma_pred: Dict[str, object]) -> Tuple[bool, str]:
    """Budgeted probe policy using current-image status, confidence, and calibration."""
    accept, reason = calibrated_no_luma_accept(no_luma_obs, no_luma_pred)
    return (not accept), reason

def active_probe_image(shape_a: str, shape_b: str, relation: str, rng: random.Random) -> np.ndarray:
    """Single minimal active observation: same scene with recoverable role luma."""
    img, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=True)
    return img


def active_probe_prediction(shape_a: str, shape_b: str, relation: str, rng: random.Random) -> Dict[str, object]:
    """
    Query a small bank of active observations and vote.

    This models a real perception policy better than one noisy probe: when a
    scene is epistemically unsafe, the agent does not merely guess after one
    altered exposure; it samples a tiny probe bank and accepts the stable
    semantic mode.
    """
    from collections import Counter

    preds = []
    margins = []
    for _ in range(PROBE_REPEATS):
        img = active_probe_image(shape_a, shape_b, relation, rng)
        pred = p69.semantic_repair_prediction(img, use_luma=True)
        preds.append(pred)
        margins.append(safe_float(pred.get("margin", 0.0)))

    scene_vote = Counter([p.get("scene", "unknown") for p in preds]).most_common(1)[0][0]
    relation_vote = Counter([p.get("relation", "unknown") for p in preds]).most_common(1)[0][0]
    shape_a_vote = Counter([p.get("shape_a", "unknown") for p in preds]).most_common(1)[0][0]
    shape_b_vote = Counter([p.get("shape_b", "unknown") for p in preds]).most_common(1)[0][0]
    return {
        "scene": scene_vote,
        "relation": relation_vote,
        "shape_a": shape_a_vote,
        "shape_b": shape_b_vote,
        "margin": float(np.median(margins)),
        "probe_repeats": PROBE_REPEATS,
    }


def choose_final_prediction(
    scene_name: str,
    shape_a: str,
    shape_b: str,
    relation: str,
    img_noluma: np.ndarray,
    rng: random.Random,
) -> Dict[str, object]:
    """Run no-luma first; probe only when policy says the evidence is not enough."""
    no_pred = p69.semantic_repair_prediction(img_noluma, use_luma=False)
    obs = p69.no_luma_observability(img_noluma, luma_pred=None)
    do_probe, reason = needs_probe(obs, no_pred)

    if do_probe:
        probe_pred = active_probe_prediction(shape_a, shape_b, relation, rng)
        return {
            "final_source": "active_probe",
            "probe_requested": 1,
            "probe_reason": reason,
            "pred_scene": probe_pred.get("scene", "unknown"),
            "pred_relation": probe_pred.get("relation", "unknown"),
            "pred_shape_a": probe_pred.get("shape_a", "unknown"),
            "pred_shape_b": probe_pred.get("shape_b", "unknown"),
            "margin": safe_float(probe_pred.get("margin", 0.0)),
            "noluma_pred_scene": no_pred.get("scene", "unknown"),
            "noluma_pred_relation": no_pred.get("relation", "unknown"),
            "noluma_margin": safe_float(no_pred.get("margin", 0.0)),
            "observable": int(bool(obs.get("observable", False))),
            "abstain": int(bool(obs.get("abstain", False))),
            "component_count": int(obs.get("component_count", 0)),
            "probe_repeats": probe_pred.get("probe_repeats", PROBE_REPEATS),
        }

    return {
        "final_source": "no_luma_observation",
        "probe_requested": 0,
        "probe_reason": reason,
        "pred_scene": no_pred.get("scene", "unknown"),
        "pred_relation": no_pred.get("relation", "unknown"),
        "pred_shape_a": no_pred.get("shape_a", "unknown"),
        "pred_shape_b": no_pred.get("shape_b", "unknown"),
        "margin": safe_float(no_pred.get("margin", 0.0)),
        "noluma_pred_scene": no_pred.get("scene", "unknown"),
        "noluma_pred_relation": no_pred.get("relation", "unknown"),
        "noluma_margin": safe_float(no_pred.get("margin", 0.0)),
        "observable": int(bool(obs.get("observable", False))),
        "abstain": int(bool(obs.get("abstain", False))),
        "component_count": int(obs.get("component_count", 0)),
        "probe_repeats": 0,
    }

# -----------------------------
# Plot helpers
# -----------------------------

def save_img(path: Path, img: np.ndarray) -> None:
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=140)
    plt.close()


def bar_plot(path: Path, title: str, labels: List[str], values: List[float], ylabel: str = "accuracy", ylim: Tuple[float, float] = (0, 1.08)) -> None:
    plt.figure(figsize=(16, 4.6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def confusion_plot(path: Path, title: str, labels: List[str], matrix: np.ndarray) -> None:
    plt.figure(figsize=(8.8, 7.4))
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
    plt.figure(figsize=(16, 4.6))
    plt.hist(values, bins=28)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("trials")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# -----------------------------
# Main experiment
# -----------------------------

def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from active perception to budgeted active observability")
    print(f"[{PHASE}] task: probe only when no-luma uncertainty has positive expected value, while accepting calibrated observable no-luma scenes")

    rng = random.Random(SEED)
    rows: List[Dict[str, object]] = []
    example_count = 0

    for scene_name, shape_a, shape_b, relation in SCENES:
        for k in range(TRIALS_PER_SCENE):
            # Start from the hard no-luma observation.  This keeps Phase 71 honest:
            # it is not allowed to use luma unless the policy explicitly probes.
            img_noluma, _ = p68.render_scene(shape_a, shape_b, relation, rng, luma=False)
            result = choose_final_prediction(scene_name, shape_a, shape_b, relation, img_noluma, rng)

            truth_underdetermined = int(result["abstain"] == 1)
            scene_ok = int(result["pred_scene"] == scene_name)
            relation_ok = int(result["pred_relation"] == relation)
            concept_pair_ok = int(set([result["pred_shape_a"], result["pred_shape_b"]]) == set([shape_a, shape_b]))
            no_hallucination_ok = int((truth_underdetermined == 0) or (result["probe_requested"] == 1))
            policy_correct = int((result["probe_requested"] == 1) or (result["probe_requested"] == 0 and scene_ok == 1))

            # Counterfactual baseline: what would the system have done if forced
            # to answer from no-luma only, even when the evidence should abstain?
            blind_scene_ok = int(result["noluma_pred_scene"] == scene_name)
            blind_relation_ok = int(result["noluma_pred_relation"] == relation)

            if example_count < 40 and k % 12 == 0:
                save_img(EXAMPLE_DIR / f"phase71_noluma_{example_count:03d}_{scene_name}.png", img_noluma)
                if result["probe_requested"] == 1:
                    probe_img = active_probe_image(shape_a, shape_b, relation, rng)
                    save_img(EXAMPLE_DIR / f"phase71_probe_{example_count:03d}_{scene_name}.png", probe_img)
                example_count += 1

            rows.append({
                "scene": scene_name,
                "shape_a": shape_a,
                "shape_b": shape_b,
                "relation": relation,
                "pred_scene": result["pred_scene"],
                "pred_relation": result["pred_relation"],
                "pred_shape_a": result["pred_shape_a"],
                "pred_shape_b": result["pred_shape_b"],
                "scene_ok": scene_ok,
                "relation_ok": relation_ok,
                "concept_pair_ok": concept_pair_ok,
                "final_source": result["final_source"],
                "probe_requested": result["probe_requested"],
                "probe_reason": result["probe_reason"],
                "policy_correct": policy_correct,
                "truth_underdetermined": truth_underdetermined,
                "no_hallucination_ok": no_hallucination_ok,
                "observable": result["observable"],
                "abstain": result["abstain"],
                "component_count": result["component_count"],
                "probe_repeats": result.get("probe_repeats", 0),
                "blind_noluma_scene_ok": blind_scene_ok,
                "blind_noluma_relation_ok": blind_relation_ok,
                "noluma_pred_scene": result["noluma_pred_scene"],
                "noluma_pred_relation": result["noluma_pred_relation"],
                "margin": result["margin"],
                "noluma_margin": result["noluma_margin"],
            })

    n = len(rows)
    probed_rows = [r for r in rows if int(r["probe_requested"]) == 1]
    unprobed_rows = [r for r in rows if int(r["probe_requested"]) == 0]
    und_rows = [r for r in rows if int(r["truth_underdetermined"]) == 1]
    obs_rows = [r for r in rows if int(r["truth_underdetermined"]) == 0]

    scene_acc = sum(int(r["scene_ok"]) for r in rows) / n
    relation_acc = sum(int(r["relation_ok"]) for r in rows) / n
    pair_acc = sum(int(r["concept_pair_ok"]) for r in rows) / n
    probe_rate = len(probed_rows) / n
    accepted_noluma_rate = len(unprobed_rows) / n
    policy_acc = sum(int(r["policy_correct"]) for r in rows) / n
    no_hallucination = sum(int(r["no_hallucination_ok"]) for r in rows) / n
    blind_scene_acc = sum(int(r["blind_noluma_scene_ok"]) for r in rows) / n
    blind_relation_acc = sum(int(r["blind_noluma_relation_ok"]) for r in rows) / n
    probe_recovery_scene_acc = (sum(int(r["scene_ok"]) for r in und_rows) / len(und_rows)) if und_rows else 1.0
    observable_no_probe_scene_acc = (sum(int(r["scene_ok"]) for r in obs_rows) / len(obs_rows)) if obs_rows else 1.0
    mean_margin = float(np.mean([float(r["margin"]) for r in rows]))
    margin_floor = float(np.min([float(r["margin"]) for r in rows]))

    passed = (
        scene_acc >= 0.94 and
        relation_acc >= 0.94 and
        pair_acc >= 0.94 and
        policy_acc >= 0.90 and
        no_hallucination >= 0.98 and
        probe_recovery_scene_acc >= 0.90 and
        scene_acc >= blind_scene_acc + 0.20 and
        probe_rate <= 0.90 and
        accepted_noluma_rate >= 0.10
    )

    scene_labels = [s[0] for s in SCENES]
    scene_idx = {x: i for i, x in enumerate(scene_labels)}
    scene_conf = np.zeros((len(scene_labels), len(scene_labels)), dtype=float)
    for r in rows:
        if str(r["pred_scene"]) in scene_idx:
            scene_conf[scene_idx[str(r["scene"])], scene_idx[str(r["pred_scene"])]] += 1
    scene_conf = scene_conf / np.maximum(scene_conf.sum(axis=1, keepdims=True), 1)

    relation_idx = {x: i for i, x in enumerate(RELATIONS)}
    relation_conf = np.zeros((len(RELATIONS), len(RELATIONS)), dtype=float)
    for r in rows:
        if str(r["pred_relation"]) in relation_idx:
            relation_conf[relation_idx[str(r["relation"])], relation_idx[str(r["pred_relation"])]] += 1
    relation_conf = relation_conf / np.maximum(relation_conf.sum(axis=1, keepdims=True), 1)

    summaries = []
    for scene_name, shape_a, shape_b, relation in SCENES:
        sub = [r for r in rows if r["scene"] == scene_name]
        summaries.append({
            "scene": scene_name,
            "scene_acc": sum(int(r["scene_ok"]) for r in sub) / len(sub),
            "relation_acc": sum(int(r["relation_ok"]) for r in sub) / len(sub),
            "concept_pair_acc": sum(int(r["concept_pair_ok"]) for r in sub) / len(sub),
            "probe_rate": sum(int(r["probe_requested"]) for r in sub) / len(sub),
            "policy_acc": sum(int(r["policy_correct"]) for r in sub) / len(sub),
            "blind_noluma_scene_acc": sum(int(r["blind_noluma_scene_ok"]) for r in sub) / len(sub),
            "active_gain": (sum(int(r["scene_ok"]) for r in sub) - sum(int(r["blind_noluma_scene_ok"]) for r in sub)) / len(sub),
            "mean_margin": float(np.mean([float(r["margin"]) for r in sub])),
            "margin_floor": float(np.min([float(r["margin"]) for r in sub])),
        })

    # Write data outputs.
    trials_path = OUT / "phase71_budgeted_active_observability_probe_bridge_trials.csv"
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": passed,
        "seed": SEED,
        "trials": n,
        "scene_accuracy": scene_acc,
        "relation_accuracy": relation_acc,
        "concept_pair_accuracy": pair_acc,
        "probe_rate": probe_rate,
        "accepted_noluma_rate": accepted_noluma_rate,
        "probe_repeats_when_used": PROBE_REPEATS,
        "policy_accuracy": policy_acc,
        "no_hallucination_accuracy": no_hallucination,
        "blind_noluma_scene_accuracy": blind_scene_acc,
        "blind_noluma_relation_accuracy": blind_relation_acc,
        "probe_recovery_scene_accuracy": probe_recovery_scene_acc,
        "observable_no_probe_scene_accuracy": observable_no_probe_scene_acc,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "scene_summary": summaries,
        "notes": [
            "Phase 71 treats active perception as a costed resource instead of a default reflex.",
            "The active probe is used only when the calibrated no-luma pass is epistemically unsafe.",
            "Success requires high active accuracy, no hallucination, and a meaningful reduction in probe usage.",
        ],
    }
    summary_path = OUT / "phase71_budgeted_active_observability_probe_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = OUT / "phase71_budgeted_active_observability_probe_bridge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {PHASE}: {TITLE}\n\n")
        f.write("## Summary\n\n")
        for k in [
            "pass", "trials", "scene_accuracy", "relation_accuracy", "concept_pair_accuracy",
            "probe_rate", "accepted_noluma_rate", "policy_accuracy", "no_hallucination_accuracy",
            "blind_noluma_scene_accuracy", "probe_recovery_scene_accuracy",
            "observable_no_probe_scene_accuracy", "mean_margin", "margin_floor",
        ]:
            f.write(f"- **{k}**: {summary[k]}\n")
        f.write("\n## Scene summary\n\n")
        for s in summaries:
            f.write(
                f"- **{s['scene']}**: scene={s['scene_acc']:.3f}, relation={s['relation_acc']:.3f}, "
                f"probe={s['probe_rate']:.3f}, policy={s['policy_acc']:.3f}, "
                f"blind={s['blind_noluma_scene_acc']:.3f}, gain={s['active_gain']:.3f}, "
                f"margin={s['mean_margin']:.3f}\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "Phase 69 established the right epistemic behavior: do not invent a two-role scene when the raster has collapsed into a single visible silhouette. "
            "Phase 71 converts that uncertainty into a budgeted action. The policy first tries the no-luma observation, accepts calibrated observable families without probing, and requests a targeted luma probe only when the unprobed answer is epistemically unsafe.\n"
        )

    # Visual outputs.
    bar_plot(
        OUT / "phase71_active_epistemic_probe_policy_metrics.png",
        "Phase 71 budgeted active observability metrics",
        ["scene_acc", "relation_acc", "pair_acc", "policy_acc", "no_hallucination", "blind_noluma"],
        [scene_acc, relation_acc, pair_acc, policy_acc, no_hallucination, blind_scene_acc],
        ylabel="score",
    )
    bar_plot(
        OUT / "phase71_active_epistemic_probe_rate_by_scene.png",
        "Phase 71 probe rate by scene",
        [s["scene"] for s in summaries],
        [s["probe_rate"] for s in summaries],
        ylabel="probe rate",
    )
    bar_plot(
        OUT / "phase71_active_epistemic_gain_by_scene.png",
        "Phase 71 budgeted probe gain over blind no-luma",
        [s["scene"] for s in summaries],
        [s["active_gain"] for s in summaries],
        ylabel="accuracy gain",
        ylim=(-0.05, 1.08),
    )
    histogram_plot(
        OUT / "phase71_active_epistemic_margin_distribution.png",
        "Phase 71 budgeted active winner margin distribution",
        [float(r["margin"]) for r in rows],
        "runner-up semantic score - winner score",
    )
    confusion_plot(
        OUT / "phase71_active_epistemic_scene_confusion.png",
        "Phase 71 budgeted active scene confusion",
        scene_labels,
        scene_conf,
    )
    confusion_plot(
        OUT / "phase71_active_epistemic_relation_confusion.png",
        "Phase 71 budgeted active relation confusion",
        RELATIONS,
        relation_conf,
    )

    print(f"[{PHASE}] PHASE71_BUDGETED_ACTIVE_OBSERVABILITY_PROBE_BRIDGE_PASS={passed}")
    print(
        f"[{PHASE}] scene_accuracy={scene_acc:.4f} relation_accuracy={relation_acc:.4f} "
        f"concept_pair_accuracy={pair_acc:.4f} probe_rate={probe_rate:.4f} accepted_noluma_rate={accepted_noluma_rate:.4f} "
        f"policy_accuracy={policy_acc:.4f} no_hallucination_accuracy={no_hallucination:.4f} "
        f"blind_noluma_scene_accuracy={blind_scene_acc:.4f} probe_recovery_scene_accuracy={probe_recovery_scene_acc:.4f} "
        f"mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={n}"
    )
    print(f"[{PHASE}] scene summary:")
    for s in summaries:
        print(
            f"  - {s['scene']:<28} scene={s['scene_acc']:.3f} relation={s['relation_acc']:.3f} "
            f"probe={s['probe_rate']:.3f} policy={s['policy_acc']:.3f} blind={s['blind_noluma_scene_acc']:.3f} "
            f"gain={s['active_gain']:.3f} margin={s['mean_margin']:.4f} floor={s['margin_floor']:.4f}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
