# geomlang_phase31_compositional_geometric_thought_reasoner_basic32_E_drive.py

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


PHASE_TAG = "31"
PHASE_NAME = "Compositional geometric thought reasoner"
OUTPUT_ROOT = Path(r"E:\BBIT\outputs_basic32")
TRIALS_CSV = OUTPUT_ROOT / "phase31_compositional_geometric_trials.csv"
SUMMARY_JSON = OUTPUT_ROOT / "phase31_compositional_geometric_summary.json"
REPORT_MD = OUTPUT_ROOT / "phase31_compositional_geometric_report.md"
EXAMPLE_DIR = OUTPUT_ROOT / "phase31_examples"

N_TRIALS = 700
N_POINTS = 40
N_CANDIDATES = 6
N_EXAMPLE_PNGS = 8
SEED = 31_314159

# We deliberately keep the pass gate focused on actual reasoning success,
# not on human-facing label bookkeeping.
PASS_MIN_ACCURACY = 0.97
PASS_MIN_SCRAMBLE = 0.99
PASS_MAX_TRANSFER_ERROR = 0.020


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FieldModel:
    partition: str
    region_weights: Dict[int, np.ndarray]  # region -> (3, 2) affine weights


@dataclass
class TrialResult:
    trial_id: int
    family: str
    true_partition: str
    inferred_ab_partition: str
    inferred_bc_partition: str
    correct_index: int
    chosen_index: int
    is_correct: bool
    scramble_stable: bool
    source_fit_ab: float
    source_fit_bc: float
    transfer_error: float
    best_score: float
    runner_up_score: float
    margin: float
    candidate_scores: List[float]
    example_png: str


# -----------------------------
# Basic geometry helpers
# -----------------------------

def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale <= 1e-12:
        return pts
    return pts / scale


def make_base_shape(rng: np.random.Generator, n_points: int = N_POINTS) -> np.ndarray:
    """
    Generates an irregular point cloud with enough structure for partitioned reasoning.
    """
    clusters = []
    n1 = n_points // 2
    n2 = n_points // 3
    n3 = n_points - n1 - n2

    c1 = rng.normal(loc=[-0.35, 0.15], scale=[0.16, 0.10], size=(n1, 2))
    c2 = rng.normal(loc=[0.30, -0.10], scale=[0.12, 0.18], size=(n2, 2))

    theta = np.linspace(0, 2 * np.pi, n3, endpoint=False)
    radius = 0.20 + 0.06 * rng.random(n3)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    ring += rng.normal(scale=0.015, size=ring.shape)

    clusters.extend([c1, c2, ring])
    pts = np.concatenate(clusters, axis=0)

    # Small random global warp to diversify source forms.
    jitter = rng.normal(scale=0.015, size=pts.shape)
    pts = pts + jitter
    return normalize_points(pts)


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_affine(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    aug = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return aug @ w


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = pairwise_distances(a, b)
    return float(np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0)))


# -----------------------------
# Partition logic
# -----------------------------

def compute_partition_labels(points: np.ndarray, partition: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    if partition == "global":
        return np.zeros(len(pts), dtype=np.int64)

    if partition == "left_right":
        return (x >= 0.0).astype(np.int64)

    if partition == "top_bottom":
        return (y >= 0.0).astype(np.int64)

    if partition == "quadrants":
        labels = np.zeros(len(pts), dtype=np.int64)
        labels[(x >= 0.0) & (y >= 0.0)] = 0
        labels[(x < 0.0) & (y >= 0.0)] = 1
        labels[(x < 0.0) & (y < 0.0)] = 2
        labels[(x >= 0.0) & (y < 0.0)] = 3
        return labels

    if partition == "core_shell":
        r = np.linalg.norm(pts, axis=1)
        threshold = np.quantile(r, 0.55)
        return (r >= threshold).astype(np.int64)

    raise ValueError(f"Unknown partition: {partition}")


def partition_region_ids(partition: str) -> List[int]:
    if partition == "global":
        return [0]
    if partition in {"left_right", "top_bottom", "core_shell"}:
        return [0, 1]
    if partition == "quadrants":
        return [0, 1, 2, 3]
    raise ValueError(f"Unknown partition: {partition}")


# -----------------------------
# Hidden family generation
# -----------------------------

FAMILIES = [
    "rigid_chain",
    "similarity_chain",
    "shear_chain",
    "left_right_chain",
    "top_bottom_chain",
    "quadrants_chain",
    "core_shell_chain",
]


def family_to_partition(family: str) -> str:
    if family in {"rigid_chain", "similarity_chain", "shear_chain"}:
        return "global"
    if family == "left_right_chain":
        return "left_right"
    if family == "top_bottom_chain":
        return "top_bottom"
    if family == "quadrants_chain":
        return "quadrants"
    if family == "core_shell_chain":
        return "core_shell"
    raise ValueError(f"Unknown family: {family}")


def random_affine_weights_for_mode(rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "rigid":
        theta = rng.uniform(-0.55, 0.55)
        rot = rotation_matrix(theta)
        t = rng.uniform(-0.18, 0.18, size=2)
        w = np.zeros((3, 2), dtype=np.float64)
        w[0:2, :] = rot.T
        w[2, :] = t
        return w

    if mode == "similarity":
        theta = rng.uniform(-0.55, 0.55)
        scale = rng.uniform(0.82, 1.18)
        mat = scale * rotation_matrix(theta)
        t = rng.uniform(-0.18, 0.18, size=2)
        w = np.zeros((3, 2), dtype=np.float64)
        w[0:2, :] = mat.T
        w[2, :] = t
        return w

    if mode == "shear":
        sx = rng.uniform(0.88, 1.15)
        sy = rng.uniform(0.88, 1.15)
        shx = rng.uniform(-0.22, 0.22)
        shy = rng.uniform(-0.22, 0.22)
        theta = rng.uniform(-0.30, 0.30)
        base = np.array([[sx, shx], [shy, sy]], dtype=np.float64)
        mat = rotation_matrix(theta) @ base
        t = rng.uniform(-0.15, 0.15, size=2)
        w = np.zeros((3, 2), dtype=np.float64)
        w[0:2, :] = mat.T
        w[2, :] = t
        return w

    raise ValueError(f"Unknown mode: {mode}")


def random_local_weights(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(-0.28, 0.28)
    scale_x = rng.uniform(0.90, 1.12)
    scale_y = rng.uniform(0.90, 1.12)
    shear = rng.uniform(-0.10, 0.10)
    mat = rotation_matrix(theta) @ np.array(
        [[scale_x, shear], [shear * 0.25, scale_y]],
        dtype=np.float64,
    )
    t = rng.uniform(-0.12, 0.12, size=2)

    w = np.zeros((3, 2), dtype=np.float64)
    w[0:2, :] = mat.T
    w[2, :] = t
    return w


def build_random_hidden_field(rng: np.random.Generator, family: str) -> FieldModel:
    partition = family_to_partition(family)
    region_ids = partition_region_ids(partition)

    weights: Dict[int, np.ndarray] = {}

    if family == "rigid_chain":
        weights[0] = random_affine_weights_for_mode(rng, "rigid")
    elif family == "similarity_chain":
        weights[0] = random_affine_weights_for_mode(rng, "similarity")
    elif family == "shear_chain":
        weights[0] = random_affine_weights_for_mode(rng, "shear")
    else:
        # Local families: each region gets its own mild affine.
        for region_id in region_ids:
            weights[region_id] = random_local_weights(rng)

    return FieldModel(partition=partition, region_weights=weights)


def apply_field(points: np.ndarray, model: FieldModel) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    labels = compute_partition_labels(pts, model.partition)
    out = np.zeros_like(pts)

    for region_id, w in model.region_weights.items():
        mask = labels == region_id
        if np.any(mask):
            out[mask] = apply_affine(pts[mask], w)

    return out


# -----------------------------
# Inference
# -----------------------------

CANDIDATE_PARTITIONS = ["global", "left_right", "top_bottom", "quadrants", "core_shell"]


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.concatenate([src, np.ones((len(src), 1), dtype=np.float64)], axis=1)
    w, _, _, _ = np.linalg.lstsq(x, dst, rcond=None)
    pred = x @ w
    residual = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return w, residual


def fit_field(src: np.ndarray, dst: np.ndarray, partition: str) -> Tuple[FieldModel | None, float]:
    labels = compute_partition_labels(src, partition)
    region_ids = partition_region_ids(partition)

    weights: Dict[int, np.ndarray] = {}
    total_error = 0.0
    total_count = 0

    for region_id in region_ids:
        mask = labels == region_id
        count = int(np.sum(mask))
        if count < 3:
            return None, float("inf")

        w, residual = fit_affine(src[mask], dst[mask])
        weights[region_id] = w
        total_error += residual * count
        total_count += count

    avg_error = total_error / max(total_count, 1)
    return FieldModel(partition=partition, region_weights=weights), float(avg_error)


def infer_best_field(src: np.ndarray, dst: np.ndarray) -> Tuple[FieldModel, float]:
    best_model: FieldModel | None = None
    best_partition = None
    best_residual = float("inf")

    for partition in CANDIDATE_PARTITIONS:
        model, residual = fit_field(src, dst, partition)
        if model is None:
            continue
        if residual < best_residual:
            best_model = model
            best_residual = residual
            best_partition = partition

    if best_model is None or best_partition is None:
        raise RuntimeError("Failed to infer a field for this trial.")

    return best_model, best_residual


# -----------------------------
# Candidate generation
# -----------------------------

def make_decoy_from_noise(target: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    return target + rng.normal(scale=scale, size=target.shape)


def make_decoy_wrong_order(d: np.ndarray, f1: FieldModel, f2: FieldModel) -> np.ndarray:
    return apply_field(apply_field(d, f2), f1)


def make_decoy_single_step(d: np.ndarray, f: FieldModel) -> np.ndarray:
    return apply_field(d, f)


def make_decoy_blend(d: np.ndarray, true_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.35, 0.70)
    noise = rng.normal(scale=0.020, size=true_target.shape)
    return normalize_points((1.0 - alpha) * d + alpha * true_target + noise)


def build_candidates(
    rng: np.random.Generator,
    d: np.ndarray,
    true_f1: FieldModel,
    true_f2: FieldModel,
    inferred_f1: FieldModel,
    inferred_f2: FieldModel,
) -> Tuple[List[np.ndarray], int]:
    true_target = apply_field(apply_field(d, true_f1), true_f2)

    decoys = [
        make_decoy_single_step(d, inferred_f1),
        make_decoy_single_step(d, inferred_f2),
        make_decoy_wrong_order(d, inferred_f1, inferred_f2),
        make_decoy_blend(d, true_target, rng),
        make_decoy_from_noise(true_target, rng, scale=0.040),
    ]

    candidates = [true_target] + decoys
    candidates = [normalize_points(c) for c in candidates]

    # Shuffle candidate order.
    perm = rng.permutation(len(candidates))
    shuffled = [candidates[i] for i in perm]
    correct_index = int(np.where(perm == 0)[0][0])

    # If more candidates requested in the future, pad with extra noisy ones.
    while len(shuffled) < N_CANDIDATES:
        shuffled.append(normalize_points(make_decoy_from_noise(true_target, rng, scale=0.055)))

    return shuffled[:N_CANDIDATES], correct_index


# -----------------------------
# Visualization
# -----------------------------

def plot_trial_example(
    out_path: Path,
    trial: TrialResult,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    predicted: np.ndarray,
    true_target: np.ndarray,
    candidates: List[np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    def draw_points(ax, pts, title, color="tab:blue"):
        ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.9, c=color)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)

    # A
    draw_points(axes[0, 0], a, "A", color="tab:blue")

    # A -> B overlay
    axes[0, 1].scatter(a[:, 0], a[:, 1], s=22, alpha=0.55, c="tab:blue", label="A")
    axes[0, 1].scatter(b[:, 0], b[:, 1], s=22, alpha=0.75, c="tab:orange", label="B")
    for i in range(len(a)):
        axes[0, 1].plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], alpha=0.12, c="gray", linewidth=0.8)
    axes[0, 1].set_title(f"A → B\ninferred partition: {trial.inferred_ab_partition}")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].set_xlim(-1.45, 1.45)
    axes[0, 1].set_ylim(-1.45, 1.45)

    # B -> C overlay
    axes[0, 2].scatter(b[:, 0], b[:, 1], s=22, alpha=0.55, c="tab:orange", label="B")
    axes[0, 2].scatter(c[:, 0], c[:, 1], s=22, alpha=0.75, c="tab:green", label="C")
    for i in range(len(b)):
        axes[0, 2].plot([b[i, 0], c[i, 0]], [b[i, 1], c[i, 1]], alpha=0.12, c="gray", linewidth=0.8)
    axes[0, 2].set_title(f"B → C\ninferred partition: {trial.inferred_bc_partition}")
    axes[0, 2].legend(loc="upper right", fontsize=8)
    axes[0, 2].set_aspect("equal")
    axes[0, 2].grid(True, alpha=0.25)
    axes[0, 2].set_xlim(-1.45, 1.45)
    axes[0, 2].set_ylim(-1.45, 1.45)

    # D
    draw_points(axes[1, 0], d, "D", color="tab:purple")

    # Predicted vs true target
    axes[1, 1].scatter(true_target[:, 0], true_target[:, 1], s=28, alpha=0.75, c="tab:orange", label="true target")
    axes[1, 1].scatter(predicted[:, 0], predicted[:, 1], s=16, alpha=0.9, c="tab:blue", marker="x", label="predicted")
    axes[1, 1].set_title(
        "predicted transfer\n"
        f"transfer_err={trial.transfer_error:.6f}\n"
        f"margin={trial.margin:.6f}"
    )
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].set_xlim(-1.45, 1.45)
    axes[1, 1].set_ylim(-1.45, 1.45)

    # Candidate score bar chart
    idx = np.arange(len(trial.candidate_scores))
    bars = axes[1, 2].bar(idx, trial.candidate_scores)
    for i, bar in enumerate(bars):
        if i == trial.correct_index:
            bar.set_alpha(0.95)
            bar.set_linewidth(2.0)
            bar.set_edgecolor("green")
        if i == trial.chosen_index:
            bar.set_linewidth(2.0)
            bar.set_edgecolor("red")
    axes[1, 2].set_title(
        f"candidate scores\nchosen={trial.chosen_index} correct={trial.correct_index}\n"
        f"{'CORRECT' if trial.is_correct else 'WRONG'}"
    )
    axes[1, 2].set_xlabel("candidate index")
    axes[1, 2].set_ylabel("Chamfer score")

    fig.suptitle(
        f"Phase 31 example #{trial.trial_id} | family={trial.family} | "
        f"true partition={trial.true_partition}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_plots(results: List[TrialResult]) -> None:
    # Accuracy by family
    family_names = sorted(set(r.family for r in results))
    family_acc = []
    for name in family_names:
        fam = [r for r in results if r.family == name]
        family_acc.append(float(np.mean([r.is_correct for r in fam])))

    plt.figure(figsize=(12, 6))
    plt.barh(family_names, family_acc)
    for i, acc in enumerate(family_acc):
        plt.text(acc + 0.01, i, f"{acc:.2f}", va="center")
    plt.xlim(0, 1.05)
    plt.xlabel("accuracy")
    plt.title("Phase 31 compositional geometric thought: accuracy by hidden family")
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "phase31_accuracy_by_family.png", dpi=160)
    plt.close()

    # Margin histogram
    margins = [r.margin for r in results]
    plt.figure(figsize=(12, 6))
    plt.hist(margins, bins=35)
    plt.xlabel("runner-up score - best score")
    plt.ylabel("trials")
    plt.title("Phase 31 answer margin distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "phase31_margin_distribution.png", dpi=160)
    plt.close()

    # Transfer error histogram
    transfer_errors = [r.transfer_error for r in results]
    plt.figure(figsize=(12, 6))
    plt.hist(transfer_errors, bins=35)
    plt.xlabel("Chamfer error: predicted target vs true target")
    plt.ylabel("trials")
    plt.title("Phase 31 transferred compositional-field error distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "phase31_transfer_error_distribution.png", dpi=160)
    plt.close()

    # Partition usage
    partitions = CANDIDATE_PARTITIONS
    counts = []
    for p in partitions:
        counts.append(sum(1 for r in results if r.inferred_ab_partition == p and r.inferred_bc_partition == p))
    plt.figure(figsize=(10, 5))
    plt.bar(partitions, counts)
    plt.ylabel("trials")
    plt.title("Phase 31 inferred same-partition chain counts")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "phase31_partition_chain_counts.png", dpi=160)
    plt.close()


# -----------------------------
# Trial execution
# -----------------------------

def run_trial(trial_id: int, rng: np.random.Generator, save_example: bool) -> TrialResult:
    family = str(rng.choice(FAMILIES))
    true_partition = family_to_partition(family)

    # Base shapes
    a = make_base_shape(rng, N_POINTS)
    d = make_base_shape(rng, N_POINTS)

    # Hidden two-step chain
    true_f1 = build_random_hidden_field(rng, family)
    true_f2 = build_random_hidden_field(rng, family)

    b = normalize_points(apply_field(a, true_f1))
    c = normalize_points(apply_field(b, true_f2))

    # Infer A->B and B->C
    inferred_f1, fit_ab = infer_best_field(a, b)
    inferred_f2, fit_bc = infer_best_field(b, c)

    # Compose inferred relation onto D
    predicted = normalize_points(apply_field(apply_field(d, inferred_f1), inferred_f2))
    true_target = normalize_points(apply_field(apply_field(d, true_f1), true_f2))

    candidates, correct_index = build_candidates(rng, d, true_f1, true_f2, inferred_f1, inferred_f2)

    def score_candidate(candidate: np.ndarray) -> float:
        return chamfer_distance(predicted, candidate)

    candidate_scores = [score_candidate(cand) for cand in candidates]
    chosen_index = int(np.argmin(candidate_scores))
    is_correct = chosen_index == correct_index

    # Scramble stability check
    perm = rng.permutation(len(candidates))
    scrambled = [candidates[i] for i in perm]
    scrambled_scores = [score_candidate(cand) for cand in scrambled]
    scrambled_choice = int(np.argmin(scrambled_scores))
    scramble_correct_index = int(np.where(perm == correct_index)[0][0])
    scramble_stable = scrambled_choice == scramble_correct_index

    ordered_scores = np.sort(np.array(candidate_scores, dtype=np.float64))
    best_score = float(ordered_scores[0])
    runner_up_score = float(ordered_scores[1])
    margin = runner_up_score - best_score
    transfer_error = chamfer_distance(predicted, true_target)

    example_png = ""
    if save_example:
        example_png = f"phase31_example_{trial_id:03d}.png"
        plot_trial_example(
            EXAMPLE_DIR / example_png,
            TrialResult(
                trial_id=trial_id,
                family=family,
                true_partition=true_partition,
                inferred_ab_partition=inferred_f1.partition,
                inferred_bc_partition=inferred_f2.partition,
                correct_index=correct_index,
                chosen_index=chosen_index,
                is_correct=is_correct,
                scramble_stable=scramble_stable,
                source_fit_ab=fit_ab,
                source_fit_bc=fit_bc,
                transfer_error=transfer_error,
                best_score=best_score,
                runner_up_score=runner_up_score,
                margin=margin,
                candidate_scores=candidate_scores,
                example_png=example_png,
            ),
            a=a,
            b=b,
            c=c,
            d=d,
            predicted=predicted,
            true_target=true_target,
            candidates=candidates,
        )

    return TrialResult(
        trial_id=trial_id,
        family=family,
        true_partition=true_partition,
        inferred_ab_partition=inferred_f1.partition,
        inferred_bc_partition=inferred_f2.partition,
        correct_index=correct_index,
        chosen_index=chosen_index,
        is_correct=is_correct,
        scramble_stable=scramble_stable,
        source_fit_ab=fit_ab,
        source_fit_bc=fit_bc,
        transfer_error=transfer_error,
        best_score=best_score,
        runner_up_score=runner_up_score,
        margin=margin,
        candidate_scores=candidate_scores,
        example_png=example_png,
    )


# -----------------------------
# Output writing
# -----------------------------

def write_trials_csv(results: List[TrialResult]) -> None:
    fieldnames = [
        "trial_id",
        "family",
        "true_partition",
        "inferred_ab_partition",
        "inferred_bc_partition",
        "correct_index",
        "chosen_index",
        "is_correct",
        "scramble_stable",
        "source_fit_ab",
        "source_fit_bc",
        "transfer_error",
        "best_score",
        "runner_up_score",
        "margin",
        "example_png",
        "candidate_scores_json",
    ]

    with TRIALS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "trial_id": r.trial_id,
                    "family": r.family,
                    "true_partition": r.true_partition,
                    "inferred_ab_partition": r.inferred_ab_partition,
                    "inferred_bc_partition": r.inferred_bc_partition,
                    "correct_index": r.correct_index,
                    "chosen_index": r.chosen_index,
                    "is_correct": r.is_correct,
                    "scramble_stable": r.scramble_stable,
                    "source_fit_ab": f"{r.source_fit_ab:.9f}",
                    "source_fit_bc": f"{r.source_fit_bc:.9f}",
                    "transfer_error": f"{r.transfer_error:.9f}",
                    "best_score": f"{r.best_score:.9f}",
                    "runner_up_score": f"{r.runner_up_score:.9f}",
                    "margin": f"{r.margin:.9f}",
                    "example_png": r.example_png,
                    "candidate_scores_json": json.dumps([round(x, 9) for x in r.candidate_scores]),
                }
            )


def build_summary(results: List[TrialResult]) -> dict:
    accuracy = float(np.mean([r.is_correct for r in results]))
    scramble_stability = float(np.mean([r.scramble_stable for r in results]))
    mean_fit_ab = float(np.mean([r.source_fit_ab for r in results]))
    mean_fit_bc = float(np.mean([r.source_fit_bc for r in results]))
    mean_transfer_error = float(np.mean([r.transfer_error for r in results]))
    mean_margin = float(np.mean([r.margin for r in results]))
    min_margin = float(np.min([r.margin for r in results]))

    family_summary = []
    for family in sorted(set(r.family for r in results)):
        fam = [r for r in results if r.family == family]
        family_summary.append(
            {
                "family": family,
                "accuracy": float(np.mean([r.is_correct for r in fam])),
                "scramble_stability": float(np.mean([r.scramble_stable for r in fam])),
                "mean_fit_ab": float(np.mean([r.source_fit_ab for r in fam])),
                "mean_fit_bc": float(np.mean([r.source_fit_bc for r in fam])),
                "mean_transfer_error": float(np.mean([r.transfer_error for r in fam])),
                "mean_margin": float(np.mean([r.margin for r in fam])),
                "min_margin": float(np.min([r.margin for r in fam])),
            }
        )

    phase_pass = (
        accuracy >= PASS_MIN_ACCURACY
        and scramble_stability >= PASS_MIN_SCRAMBLE
        and mean_transfer_error <= PASS_MAX_TRANSFER_ERROR
    )

    return {
        "phase": PHASE_TAG,
        "title": PHASE_NAME,
        "trials": len(results),
        "seed": SEED,
        "COMPOSITIONAL_GEOMETRIC_THOUGHT_PASS": phase_pass,
        "accuracy": accuracy,
        "scramble_stability": scramble_stability,
        "mean_source_fit_ab": mean_fit_ab,
        "mean_source_fit_bc": mean_fit_bc,
        "mean_transfer_error": mean_transfer_error,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "example_png_count": sum(1 for r in results if r.example_png),
        "family_summary": family_summary,
    }


def write_summary_json(summary: dict) -> None:
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_report_md(summary: dict, results: List[TrialResult]) -> None:
    lines: List[str] = []
    lines.append(f"# Phase {PHASE_TAG}: {PHASE_NAME}")
    lines.append("")
    lines.append("This phase moves from single-step geometric analogy to **compositional reasoning**.")
    lines.append("")
    lines.append("Each trial asks the system to:")
    lines.append("")
    lines.append("1. infer the relation from `A -> B`")
    lines.append("2. infer the relation from `B -> C`")
    lines.append("3. compose those inferred geometric fields")
    lines.append("4. transfer the composed relation onto `D`")
    lines.append("5. choose the correct answer from candidate point-cloud answers")
    lines.append("")
    lines.append("## Overall results")
    lines.append("")
    lines.append(f"- pass: `{summary['COMPOSITIONAL_GEOMETRIC_THOUGHT_PASS']}`")
    lines.append(f"- trials: `{summary['trials']}`")
    lines.append(f"- accuracy: `{summary['accuracy']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- mean source fit A->B: `{summary['mean_source_fit_ab']:.6f}`")
    lines.append(f"- mean source fit B->C: `{summary['mean_source_fit_bc']:.6f}`")
    lines.append(f"- mean transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append(f"- mean answer margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- min answer margin: `{summary['min_margin']:.6f}`")
    lines.append(f"- example PNGs written: `{summary['example_png_count']}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    for item in summary["family_summary"]:
        lines.append(
            f"- {item['family']}: "
            f"acc={item['accuracy']:.3f}, "
            f"stable={item['scramble_stability']:.3f}, "
            f"fit_ab={item['mean_fit_ab']:.6f}, "
            f"fit_bc={item['mean_fit_bc']:.6f}, "
            f"transfer_err={item['mean_transfer_error']:.6f}, "
            f"min_margin={item['min_margin']:.6f}"
        )

    lines.append("")
    lines.append("## Example files")
    lines.append("")
    for r in results:
        if r.example_png:
            lines.append(f"- `{r.example_png}` | family={r.family} | correct={r.is_correct}")

    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    print(f"[{PHASE_TAG}] {PHASE_NAME}")
    print(f"[{PHASE_TAG}] root: E:\\BBIT")
    print(f"[{PHASE_TAG}] outputs: {OUTPUT_ROOT}")
    print(f"[{PHASE_TAG}] reset continued: from single-step analogy to compositional geometric thought")
    print(f"[{PHASE_TAG}] task: infer A->B, infer B->C, compose both, transfer onto D, choose answer")
    ensure_dirs()

    rng = np.random.default_rng(SEED)
    results: List[TrialResult] = []

    example_trial_ids = set(np.linspace(0, N_TRIALS - 1, N_EXAMPLE_PNGS, dtype=int).tolist())

    for trial_id in range(N_TRIALS):
        save_example = trial_id in example_trial_ids
        result = run_trial(trial_id, rng, save_example)
        results.append(result)

    write_trials_csv(results)
    summary = build_summary(results)
    write_summary_json(summary)
    write_report_md(summary, results)
    write_summary_plots(results)

    print(f"[{PHASE_TAG}] COMPOSITIONAL_GEOMETRIC_THOUGHT_PASS={summary['COMPOSITIONAL_GEOMETRIC_THOUGHT_PASS']}")
    print(
        f"[{PHASE_TAG}] accuracy={summary['accuracy']:.4f} "
        f"scramble_stability={summary['scramble_stability']:.4f} "
        f"trials={summary['trials']}"
    )
    print(
        f"[{PHASE_TAG}] mean_source_fit_ab={summary['mean_source_fit_ab']:.6f} "
        f"mean_source_fit_bc={summary['mean_source_fit_bc']:.6f} "
        f"mean_transfer_error={summary['mean_transfer_error']:.6f}"
    )

    print(f"[{PHASE_TAG}] family summary:")
    for item in summary["family_summary"]:
        print(
            f"  - {item['family']:<16} "
            f"acc={item['accuracy']:.3f} "
            f"stable={item['scramble_stability']:.3f} "
            f"fit_ab={item['mean_fit_ab']:.6f} "
            f"fit_bc={item['mean_fit_bc']:.6f} "
            f"transfer_err={item['mean_transfer_error']:.6f} "
            f"min_margin={item['min_margin']:.6f}"
        )

    print(f"[{PHASE_TAG}] wrote trials: {TRIALS_CSV}")
    print(f"[{PHASE_TAG}] wrote summary: {SUMMARY_JSON}")
    print(f"[{PHASE_TAG}] wrote report: {REPORT_MD}")
    print(f"[{PHASE_TAG}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE_TAG}] wrote outputs to: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())