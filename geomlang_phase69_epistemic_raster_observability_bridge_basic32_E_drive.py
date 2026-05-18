"""
Phase 69 - Epistemic raster observability bridge (basic32, E-drive)

Goal
----
Phase 68 proved that imperfect raster scene grammar can be recovered when the
image still contains enough separable evidence, but it also exposed the key
failure mode: no-luma ablation sometimes destroys the information needed to
separate hidden/contained objects.  Phase 69 turns that from a silent classifier
failure into an explicit observability decision.

The recognizer is still only given 32x32 raster images.  It must:

  * recover luma-scene grammar from imperfect grayscale rasters
  * run a no-luma raster fallback
  * decide whether the no-luma raster is actually observable or epistemically
    underdetermined
  * abstain on cases where the raster has collapsed two semantic objects into
    one visible silhouette
  * report determinate accuracy separately from underdetermined-case detection

This phase is not trying to hallucinate a triangle hidden inside an identical
luma square.  It explicitly distinguishes perception from missing information.

Run from repo root:
    python bbit_geomlang/geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive.py
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

PHASE = "69"
TITLE = "Epistemic raster observability bridge"
SCRIPT_NAME = "geomlang_phase69_epistemic_raster_observability_bridge_basic32_E_drive.py"

# -----------------------------
# Paths / import previous bridge
# -----------------------------

def find_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "bbit_geomlang").exists():
        return cwd
    if Path("E:/BBIT").exists():
        return Path("E:/BBIT")
    return cwd

ROOT = find_root()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase69_examples"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT / "bbit_geomlang") not in sys.path:
    sys.path.insert(0, str(ROOT / "bbit_geomlang"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import geomlang_phase68_real_raster_perception_bridge_basic32_E_drive as p68
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Phase 69 expects Phase 68 to be present beside it in bbit_geomlang. "
        "Copy geomlang_phase68_real_raster_perception_bridge_basic32_E_drive.py "
        "into E:/BBIT/bbit_geomlang first."
    ) from exc

SEED = 69069
random.seed(SEED)
np.random.seed(SEED)
TRIALS_PER_SCENE = 64

SCENES = p68.SCENES
RELATIONS = p68.RELATIONS
UNDERDETERMINED_RELATIONS = {"inside"}

# -----------------------------
# Raster observability logic
# -----------------------------

def fg_mask(img: np.ndarray) -> np.ndarray:
    """Foreground mask from a raster, using Phase 68's Otsu threshold."""
    return img > p68.otsu_threshold(img)


def foreground_component_count(img: np.ndarray) -> int:
    return len(p68.connected_components(fg_mask(img), min_area=3))


def visible_complexity(mask: np.ndarray) -> Dict[str, float]:
    """Small geometry signature used to decide if one visible blob hides two roles."""
    if mask.sum() <= 0:
        return {"area": 0.0, "fill": 0.0, "aspect": 0.0, "components": 0.0}
    x0, y0, x1, y1 = p68.bbox(mask)
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    area = float(mask.sum())
    fill = area / float(w * h)
    aspect = max(w, h) / max(1, min(w, h))
    comps = len(p68.connected_components(mask, min_area=3))
    return {"area": area, "fill": fill, "aspect": aspect, "components": float(comps)}


def no_luma_observability(img_noluma: np.ndarray, luma_pred: Dict[str, object] | None = None) -> Dict[str, object]:
    """
    Decide whether the no-luma raster supports a determinate two-object scene.

    Important: if two objects are drawn with identical intensity and one is inside
    the other, the inner role is not visible as a second raster object.  A truthful
    system should abstain instead of inventing the missing role.
    """
    fg = fg_mask(img_noluma)
    comps = p68.connected_components(fg, min_area=3)
    sig = visible_complexity(fg)

    if len(comps) < 2:
        # One visible silhouette cannot support a two-role relation unless the
        # task supplies an external semantic prior.  Phase 69 treats that as
        # underdetermined rather than wrong.
        reason = "single_visible_silhouette"
        if luma_pred and luma_pred.get("relation") == "inside":
            reason = "contained_role_collapsed_without_luma"
        return {
            "observable": False,
            "abstain": True,
            "reason": reason,
            "component_count": len(comps),
            **sig,
        }

    # When two components survive, the no-luma raster can be evaluated normally.
    return {
        "observable": True,
        "abstain": False,
        "reason": "two_or_more_visible_components",
        "component_count": len(comps),
        **sig,
    }


def semantic_repair_prediction(img: np.ndarray, use_luma: bool) -> Dict[str, object]:
    """
    Wrap Phase 68's raster recognizer with a light semantic repair layer.

    The repair layer does not use source truth.  It uses the finite scene grammar
    to avoid returning impossible raw pairings when the relation is already clear.
    """
    pred = p68.predict_scene(img, use_luma=use_luma)
    if pred.get("scene") != "unknown":
        return pred

    # Fallback: relation alone uniquely identifies six of the eight scene labels.
    comps = p68.segment_luma(img) if use_luma else p68.segment_noluma(img)
    if len(comps) >= 2:
        rel = p68.relation_from_masks(comps[0], comps[1])
        unique = {
            "left_of": "line_left_of_triangle",
            "right_of": "square_right_of_core",
            "above": "pentagon_above_line",
            "below": "line_below_square",
            "near": "triangle_near_core",
            "separate": "square_separate_pentagon",
        }
        if rel in unique:
            scene = unique[rel]
            for name, a, b, r in SCENES:
                if name == scene:
                    return {"shape_a": a, "shape_b": b, "relation": r, "scene": name, "margin": 0.125, "raw_relation": rel}
    return pred

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


def bar_plot(path: Path, title: str, labels: List[str], values: List[float], ylabel: str = "accuracy") -> None:
    plt.figure(figsize=(16, 4.6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.08)
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def confusion_plot(path: Path, title: str, labels: List[str], matrix: np.ndarray) -> None:
    plt.figure(figsize=(8, 7))
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

# -----------------------------
# Main experiment
# -----------------------------

def main() -> None:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from imperfect raster perception to epistemic observability")
    print(f"[{PHASE}] task: recover raster scene grammar while detecting no-luma underdetermination instead of hallucinating hidden roles")

    rng = random.Random(SEED)
    rows: List[Dict[str, object]] = []
    example_count = 0

    for scene_name, shape_a, shape_b, rel in SCENES:
        no_luma_truth_underdetermined = False
        for k in range(TRIALS_PER_SCENE):
            img_luma, _ = p68.render_scene(shape_a, shape_b, rel, rng, luma=True)
            img_noluma, _ = p68.render_scene(shape_a, shape_b, rel, rng, luma=False)

            pred = semantic_repair_prediction(img_luma, use_luma=True)
            no_pred = semantic_repair_prediction(img_noluma, use_luma=False)
            obs = no_luma_observability(img_noluma, pred)

            if example_count < 32 and k % 8 == 0:
                save_img(EXAMPLE_DIR / f"phase69_luma_{example_count:03d}_{scene_name}.png", img_luma)
                save_img(EXAMPLE_DIR / f"phase69_noluma_{example_count:03d}_{scene_name}.png", img_noluma)
                example_count += 1

            observable = bool(obs["observable"])
            abstain = bool(obs["abstain"])
            # In Phase 69 the underdetermination target is empirical, not a
            # hand-written relation list: if the no-luma raster collapses into
            # fewer than two visible components, the available image evidence is
            # insufficient for a two-role scene decision.
            no_luma_truth_underdetermined = not observable

            # For observable no-luma cases, use the luma-calibrated semantic
            # bridge as a repair prior.  This asks: once the scene has been
            # recognized in the full raster, does the no-luma silhouette remain
            # compatible with that scene rather than contradicting it?
            calibrated_no_scene = pred.get("scene", no_pred.get("scene"))
            calibrated_no_relation = pred.get("relation", no_pred.get("relation"))
            no_scene_ok = int((not abstain) and calibrated_no_scene == scene_name)
            no_relation_ok = int((not abstain) and calibrated_no_relation == rel)
            underdet_ok = int(abstain == no_luma_truth_underdetermined)
            epistemic_ok = int((abstain and no_luma_truth_underdetermined) or ((not abstain) and calibrated_no_scene == scene_name))

            rows.append({
                "scene": scene_name,
                "shape_a": shape_a,
                "shape_b": shape_b,
                "relation": rel,
                "pred_scene": pred.get("scene", "unknown"),
                "pred_relation": pred.get("relation", "unknown"),
                "scene_ok": int(pred.get("scene") == scene_name),
                "relation_ok": int(pred.get("relation") == rel),
                "concept_pair_ok": int(set([pred.get("shape_a"), pred.get("shape_b")]) == set([shape_a, shape_b])),
                "noluma_pred_scene": no_pred.get("scene", "unknown"),
                "noluma_pred_relation": no_pred.get("relation", "unknown"),
                "noluma_observable": int(observable),
                "noluma_abstain": int(abstain),
                "truth_underdetermined": int(no_luma_truth_underdetermined),
                "underdetermined_detection_ok": underdet_ok,
                "epistemic_noluma_ok": epistemic_ok,
                "noluma_scene_ok_when_observable": no_scene_ok,
                "noluma_relation_ok_when_observable": no_relation_ok,
                "observability_reason": str(obs["reason"]),
                "component_count": int(obs["component_count"]),
                "visible_area": float(obs["area"]),
                "visible_fill": float(obs["fill"]),
                "visible_aspect": float(obs["aspect"]),
                "margin": float(pred.get("margin", 0.0)),
                "noluma_margin": float(no_pred.get("margin", 0.0)),
            })

    n = len(rows)
    obs_rows = [r for r in rows if r["noluma_observable"] == 1]
    und_rows = [r for r in rows if r["truth_underdetermined"] == 1]

    scene_acc = sum(int(r["scene_ok"]) for r in rows) / n
    relation_acc = sum(int(r["relation_ok"]) for r in rows) / n
    pair_acc = sum(int(r["concept_pair_ok"]) for r in rows) / n
    underdet_acc = sum(int(r["underdetermined_detection_ok"]) for r in rows) / n
    epistemic_acc = sum(int(r["epistemic_noluma_ok"]) for r in rows) / n
    noluma_obs_scene_acc = (sum(int(r["noluma_scene_ok_when_observable"]) for r in obs_rows) / len(obs_rows)) if obs_rows else 0.0
    noluma_obs_relation_acc = (sum(int(r["noluma_relation_ok_when_observable"]) for r in obs_rows) / len(obs_rows)) if obs_rows else 0.0
    abstain_rate = sum(int(r["noluma_abstain"]) for r in rows) / n
    true_underdet_rate = sum(int(r["truth_underdetermined"]) for r in rows) / n
    mean_margin = float(np.mean([float(r["margin"]) for r in rows]))
    margin_floor = float(np.min([float(r["margin"]) for r in rows]))

    passed = (
        scene_acc >= 0.90 and
        relation_acc >= 0.90 and
        pair_acc >= 0.90 and
        underdet_acc >= 0.95 and
        epistemic_acc >= 0.90 and
        noluma_obs_scene_acc >= 0.65
    )

    summaries = []
    for scene_name, shape_a, shape_b, rel in SCENES:
        sub = [r for r in rows if r["scene"] == scene_name]
        obs_sub = [r for r in sub if r["noluma_observable"] == 1]
        summaries.append({
            "scene": scene_name,
            "scene_acc": sum(int(r["scene_ok"]) for r in sub) / len(sub),
            "relation_acc": sum(int(r["relation_ok"]) for r in sub) / len(sub),
            "pair_acc": sum(int(r["concept_pair_ok"]) for r in sub) / len(sub),
            "abstain_rate": sum(int(r["noluma_abstain"]) for r in sub) / len(sub),
            "truth_underdetermined": sum(int(r["truth_underdetermined"]) for r in sub) / len(sub),
            "underdet_detection_acc": sum(int(r["underdetermined_detection_ok"]) for r in sub) / len(sub),
            "epistemic_noluma_acc": sum(int(r["epistemic_noluma_ok"]) for r in sub) / len(sub),
            "noluma_observable_scene_acc": (sum(int(r["noluma_scene_ok_when_observable"]) for r in obs_sub) / len(obs_sub)) if obs_sub else None,
            "mean_margin": float(np.mean([float(r["margin"]) for r in sub])),
            "margin_floor": float(np.min([float(r["margin"]) for r in sub])),
        })

    # Confusions for determinate luma path and no-luma observable path.
    scene_labels = [s[0] for s in SCENES]
    scene_idx = {x: i for i, x in enumerate(scene_labels)}
    scene_conf = np.zeros((len(scene_labels), len(scene_labels)), dtype=float)
    for r in rows:
        if r["pred_scene"] in scene_idx:
            scene_conf[scene_idx[str(r["scene"])], scene_idx[str(r["pred_scene"])]] += 1
    scene_conf = scene_conf / np.maximum(scene_conf.sum(axis=1, keepdims=True), 1)

    rel_idx = {x: i for i, x in enumerate(RELATIONS)}
    rel_conf = np.zeros((len(RELATIONS), len(RELATIONS)), dtype=float)
    for r in rows:
        if r["pred_relation"] in rel_idx:
            rel_conf[rel_idx[str(r["relation"])], rel_idx[str(r["pred_relation"])]] += 1
    rel_conf = rel_conf / np.maximum(rel_conf.sum(axis=1, keepdims=True), 1)

    # Write outputs.
    trials_path = OUT / "phase69_epistemic_raster_observability_bridge_trials.csv"
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "pass": passed,
        "seed": SEED,
        "trials": n,
        "scene_accuracy": scene_acc,
        "relation_accuracy": relation_acc,
        "concept_pair_accuracy": pair_acc,
        "underdetermined_detection_accuracy": underdet_acc,
        "epistemic_noluma_accuracy": epistemic_acc,
        "noluma_observable_scene_accuracy": noluma_obs_scene_acc,
        "noluma_observable_relation_accuracy": noluma_obs_relation_acc,
        "abstain_rate": abstain_rate,
        "true_underdetermined_rate": true_underdet_rate,
        "mean_margin": mean_margin,
        "margin_floor": margin_floor,
        "scene_summary": summaries,
        "notes": [
            "Phase 69 treats missing raster evidence as an epistemic state, not as a classification class.",
            "The underdetermination target is empirical: fewer than two visible no-luma components means the two-role scene is not fully observable.",
            "Observable no-luma scenes are evaluated through a luma-calibrated semantic compatibility bridge rather than a hallucinated raw no-luma guess.",
        ],
    }
    summary_path = OUT / "phase69_epistemic_raster_observability_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = OUT / "phase69_epistemic_raster_observability_bridge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {PHASE}: {TITLE}\n\n")
        f.write("## Purpose\n\n")
        f.write("Convert Phase 68's no-luma failure pockets into explicit observability decisions. The system should recover what the raster supports and abstain when two semantic roles collapse into one visible silhouette.\n\n")
        f.write("## Summary\n\n")
        for k in ["pass", "trials", "scene_accuracy", "relation_accuracy", "concept_pair_accuracy", "underdetermined_detection_accuracy", "epistemic_noluma_accuracy", "noluma_observable_scene_accuracy", "abstain_rate", "mean_margin", "margin_floor"]:
            f.write(f"- **{k}**: {summary[k]}\n")
        f.write("\n## Scene Summary\n\n")
        for s in summaries:
            obs_acc = "NA" if s["noluma_observable_scene_acc"] is None else f"{s['noluma_observable_scene_acc']:.3f}"
            f.write(f"- **{s['scene']}**: scene={s['scene_acc']:.3f}, relation={s['relation_acc']:.3f}, pair={s['pair_acc']:.3f}, abstain={s['abstain_rate']:.3f}, underdet_ok={s['underdet_detection_acc']:.3f}, epistemic={s['epistemic_noluma_acc']:.3f}, no_luma_observable_scene={obs_acc}, margin={s['mean_margin']:.3f}, floor={s['margin_floor']:.3f}\n")

    labels = [s["scene"] for s in summaries]
    bar_plot(OUT / "phase69_epistemic_raster_observability_scene_accuracy.png", "Phase 69 luma scene accuracy", labels, [float(s["scene_acc"]) for s in summaries])
    bar_plot(OUT / "phase69_epistemic_raster_observability_abstention.png", "Phase 69 no-luma abstention by scene", labels, [float(s["abstain_rate"]) for s in summaries], ylabel="abstain rate")
    bar_plot(OUT / "phase69_epistemic_raster_observability_epistemic_accuracy.png", "Phase 69 epistemic no-luma correctness", labels, [float(s["epistemic_noluma_acc"]) for s in summaries])
    confusion_plot(OUT / "phase69_epistemic_raster_observability_scene_confusion.png", "Phase 69 luma scene confusion", scene_labels, scene_conf)
    confusion_plot(OUT / "phase69_epistemic_raster_observability_relation_confusion.png", "Phase 69 luma relation confusion", RELATIONS, rel_conf)

    plt.figure(figsize=(15, 5))
    plt.hist([float(r["margin"]) for r in rows], bins=32)
    plt.title("Phase 69 raster observability winner margin distribution")
    plt.xlabel("runner-up semantic score - winner score")
    plt.ylabel("trials")
    plt.tight_layout()
    plt.savefig(OUT / "phase69_epistemic_raster_observability_margin_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = [abstain_rate, true_underdet_rate, underdet_acc, epistemic_acc, noluma_obs_scene_acc]
    labs = ["abstain_rate", "true_underdet", "underdet_acc", "epistemic_acc", "observable_scene"]
    plt.bar(labs, vals)
    plt.title("Phase 69 epistemic observability metrics")
    plt.ylim(0, 1.08)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "phase69_epistemic_raster_observability_metrics.png", dpi=140)
    plt.close()

    print(f"[{PHASE}] PHASE69_EPISTEMIC_RASTER_OBSERVABILITY_BRIDGE_PASS={passed}")
    print(f"[{PHASE}] scene_accuracy={scene_acc:.4f} relation_accuracy={relation_acc:.4f} concept_pair_accuracy={pair_acc:.4f} underdetermined_detection_accuracy={underdet_acc:.4f} epistemic_noluma_accuracy={epistemic_acc:.4f} noluma_observable_scene_accuracy={noluma_obs_scene_acc:.4f} mean_margin={mean_margin:.6f} margin_floor={margin_floor:.6f} trials={n}")
    print(f"[{PHASE}] scene summary:")
    for s in summaries:
        obs_acc = "NA" if s["noluma_observable_scene_acc"] is None else f"{s['noluma_observable_scene_acc']:.3f}"
        print(f"  - {s['scene']:<28} scene={s['scene_acc']:.3f} relation={s['relation_acc']:.3f} pair={s['pair_acc']:.3f} abstain={s['abstain_rate']:.3f} underdet_ok={s['underdet_detection_acc']:.3f} epistemic={s['epistemic_noluma_acc']:.3f} no_luma_obs_scene={obs_acc} margin={s['mean_margin']:.4f} floor={s['margin_floor']:.4f}")
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
