r"""
Phase 32 — Trajectory-memory geometric thought reasoner

Purpose:
    Phase 31 showed that final-position matching is not enough for compositional
    geometric reasoning. Some chains land in nearly identical final places even
    though they got there by different routes.

    Phase 32 adds path memory.

    Instead of only asking:

        A -> B
        B -> C
        D -> ?

    and scoring the final answer, this phase asks:

        What was the geometric path A -> B -> C?
        If that same path is transferred onto D, what intermediate D1 and final D2
        should appear?
        Which candidate preserves both the destination and the route?

This is closer to BBIT / tokenless reasoning:
    The "thought" is not a word label.
    The "thought" is the continuity of transformation through geometric space.

Outputs:
    outputs_basic32/
        phase32_trajectory_memory_trials.csv
        phase32_trajectory_memory_summary.json
        phase32_trajectory_memory_report.md
        phase32_trajectory_memory_accuracy.png
        phase32_trajectory_memory_margin_distribution.png
        phase32_trajectory_memory_baseline_gap.png
        phase32_trajectory_memory_family_match.png
        phase32_examples/*.png
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Phase 32 requires matplotlib. Install with: pip install matplotlib") from exc


# =============================================================================
# Path setup
# =============================================================================

PHASE = "32"
TITLE = "Trajectory-memory geometric thought reasoner"
PASS_KEY = "PHASE32_TRAJECTORY_MEMORY_GEOMETRIC_THOUGHT_PASS"

THIS_FILE = Path(__file__).resolve()


def find_root() -> Path:
    """
    Designed for:
        E:\BBIT\bbit_geomlang\this_script.py

    But also works if launched from:
        E:\BBIT
        E:\BBIT\bbit_geomlang
    """
    p = THIS_FILE
    for parent in [p.parent, *p.parents]:
        if parent.name.lower() == "bbit_geomlang":
            return parent.parent
    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    return cwd


ROOT = find_root()
SOURCE_ROOT = ROOT / "bbit_geomlang"
OUT = ROOT / "outputs_basic32"
OUT.mkdir(parents=True, exist_ok=True)

EXAMPLE_DIR = OUT / "phase32_examples"
if EXAMPLE_DIR.exists():
    shutil.rmtree(EXAMPLE_DIR)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Geometry helpers
# =============================================================================

Array = np.ndarray


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def rot(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def shear_x(k: float) -> Array:
    return np.array([[1.0, k], [0.0, 1.0]], dtype=np.float64)


def shear_y(k: float) -> Array:
    return np.array([[1.0, 0.0], [k, 1.0]], dtype=np.float64)


def scale(sx: float, sy: float | None = None) -> Array:
    if sy is None:
        sy = sx
    return np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)


def apply_beta(points: Array, beta: Array) -> Array:
    """
    beta shape:
        3 x 2

    points:
        N x 2

    returns:
        N x 2
    """
    aug = np.c_[points, np.ones(len(points))]
    return aug @ beta


def beta_from_matrix_translation(M: Array, t: Array) -> Array:
    """
    Return beta where:
        [x, y, 1] @ beta = x @ M.T + t
    """
    beta = np.zeros((3, 2), dtype=np.float64)
    beta[:2, :] = M.T
    beta[2, :] = t
    return beta


def estimate_affine_beta(src: Array, dst: Array) -> Tuple[Array, float]:
    """
    Least-squares affine fit from src to dst.
    """
    if len(src) < 3:
        # fallback identity when too few points
        beta = beta_from_matrix_translation(np.eye(2), np.zeros(2))
        pred = apply_beta(src, beta)
        return beta, float(np.mean(np.linalg.norm(pred - dst, axis=1))) if len(src) else 999.0

    aug = np.c_[src, np.ones(len(src))]
    beta, *_ = np.linalg.lstsq(aug, dst, rcond=None)
    pred = aug @ beta
    resid = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
    return beta, resid


def chamfer(a: Array, b: Array) -> float:
    """
    Small symmetric nearest-neighbor Chamfer distance.
    Kept simple because point counts are small.
    """
    if len(a) == 0 or len(b) == 0:
        return 999.0
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(0.5 * (np.sqrt(d.min(axis=1)).mean() + np.sqrt(d.min(axis=0)).mean()))


def normalize(points: Array) -> Array:
    p = points.copy()
    p -= p.mean(axis=0, keepdims=True)
    s = np.sqrt((p * p).sum(axis=1)).mean()
    if s > 1e-9:
        p /= s
    return p


def make_base_shape(rng: np.random.Generator, n: int = 72) -> Array:
    """
    A stable asymmetric 2D point object.

    It is intentionally not a generic circle, because a circle hides too many
    rotations/reflections. We want BBIT to reason over shape geometry, not
    exploit symmetry.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    r = (
        1.0
        + 0.18 * np.sin(3 * angles + 0.25)
        + 0.11 * np.cos(5 * angles - 0.4)
        + 0.08 * np.sin(7 * angles + 1.7)
    )

    x = r * np.cos(angles)
    y = 0.72 * r * np.sin(angles)

    pts = np.c_[x, y]

    # Add a small internal asymmetric spur/cloud so transformations have a richer
    # local fingerprint.
    inner_angles = angles[::3] + 0.12
    inner_r = 0.36 + 0.08 * np.sin(4 * inner_angles)
    inner = np.c_[inner_r * np.cos(inner_angles) + 0.18, inner_r * np.sin(inner_angles) - 0.08]

    pts = np.vstack([pts, inner])
    pts += rng.normal(0.0, 0.006, size=pts.shape)
    pts = normalize(pts)
    return pts


# =============================================================================
# Partition families
# =============================================================================

def mask_global(points: Array) -> List[Array]:
    return [np.ones(len(points), dtype=bool)]


def mask_left_right(points: Array) -> List[Array]:
    x = points[:, 0]
    return [x < np.median(x), x >= np.median(x)]


def mask_top_bottom(points: Array) -> List[Array]:
    y = points[:, 1]
    return [y < np.median(y), y >= np.median(y)]


def mask_quadrants(points: Array) -> List[Array]:
    x = points[:, 0]
    y = points[:, 1]
    mx, my = np.median(x), np.median(y)
    return [
        (x < mx) & (y < my),
        (x >= mx) & (y < my),
        (x < mx) & (y >= my),
        (x >= mx) & (y >= my),
    ]


def mask_core_shell(points: Array) -> List[Array]:
    r = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
    cut = np.quantile(r, 0.52)
    return [r <= cut, r > cut]


PARTITIONS: Dict[str, Callable[[Array], List[Array]]] = {
    "global": mask_global,
    "left_right": mask_left_right,
    "top_bottom": mask_top_bottom,
    "quadrants": mask_quadrants,
    "core_shell": mask_core_shell,
}


@dataclass
class LocalField:
    partition: str
    betas: List[Array]

    def apply(self, points: Array) -> Array:
        masks = PARTITIONS[self.partition](points)
        out = np.zeros_like(points)
        for m, beta in zip(masks, self.betas):
            if np.any(m):
                out[m] = apply_beta(points[m], beta)
        return out


def fit_local_field(src: Array, dst: Array, partition: str) -> Tuple[LocalField, float]:
    masks = PARTITIONS[partition](src)
    betas: List[Array] = []
    residuals: List[float] = []

    for m in masks:
        beta, resid = estimate_affine_beta(src[m], dst[m])
        betas.append(beta)
        if np.any(m):
            residuals.append(resid)

    return LocalField(partition=partition, betas=betas), float(np.mean(residuals)) if residuals else 999.0


def infer_best_field(src: Array, dst: Array) -> Tuple[LocalField, str, float, Dict[str, float]]:
    scores: Dict[str, float] = {}
    fields: Dict[str, LocalField] = {}

    for name in PARTITIONS:
        field, resid = fit_local_field(src, dst, name)
        scores[name] = resid
        fields[name] = field

    best_name = min(scores, key=scores.get)
    return fields[best_name], best_name, scores[best_name], scores


# =============================================================================
# Synthetic field generation
# =============================================================================

def random_beta_for_family(rng: np.random.Generator, family: str, step: int, part_index: int) -> Array:
    """
    Produces affine transforms with enough diversity to force real geometric
    inference while avoiding extreme distortions.
    """
    if family == "rigid_chain":
        theta = rng.uniform(-0.55, 0.55)
        M = rot(theta)
        t = rng.uniform(-0.22, 0.22, size=2)

    elif family == "similarity_chain":
        theta = rng.uniform(-0.45, 0.45)
        s = rng.uniform(0.82, 1.22)
        M = rot(theta) @ scale(s)
        t = rng.uniform(-0.18, 0.18, size=2)

    elif family == "shear_chain":
        if rng.random() < 0.5:
            M = shear_x(rng.uniform(-0.45, 0.45))
        else:
            M = shear_y(rng.uniform(-0.45, 0.45))
        t = rng.uniform(-0.12, 0.12, size=2)

    elif family in {"left_right_chain", "top_bottom_chain", "quadrants_chain", "core_shell_chain"}:
        # Local transforms get part-specific offsets and slight affine changes.
        base_theta = rng.uniform(-0.18, 0.18)
        local_theta = base_theta + 0.06 * (part_index - 1.5)
        local_scale = 1.0 + rng.uniform(-0.08, 0.08)

        M = rot(local_theta) @ scale(local_scale)

        # Partition-specific translation pushes create a visible local path.
        direction = np.array(
            [
                math.cos(0.9 * (part_index + 1) + 0.4 * step),
                math.sin(1.2 * (part_index + 1) - 0.3 * step),
            ],
            dtype=np.float64,
        )
        t = rng.uniform(-0.06, 0.06, size=2) + 0.10 * direction

    else:
        M = np.eye(2)
        t = np.zeros(2)

    return beta_from_matrix_translation(M, t)


def partition_for_family(family: str) -> str:
    if family == "left_right_chain":
        return "left_right"
    if family == "top_bottom_chain":
        return "top_bottom"
    if family == "quadrants_chain":
        return "quadrants"
    if family == "core_shell_chain":
        return "core_shell"
    return "global"


def make_random_field(rng: np.random.Generator, family: str, source_points: Array, step: int) -> LocalField:
    partition = partition_for_family(family)
    masks = PARTITIONS[partition](source_points)

    betas = [
        random_beta_for_family(rng, family, step=step, part_index=i)
        for i, _ in enumerate(masks)
    ]

    return LocalField(partition=partition, betas=betas)


FAMILIES = [
    "core_shell_chain",
    "left_right_chain",
    "quadrants_chain",
    "rigid_chain",
    "shear_chain",
    "similarity_chain",
    "top_bottom_chain",
]


# =============================================================================
# Candidate construction
# =============================================================================

@dataclass
class Candidate:
    name: str
    mid: Array
    final: Array
    is_correct: bool


def noisy(points: Array, rng: np.random.Generator, sigma: float = 0.002) -> Array:
    return points + rng.normal(0.0, sigma, size=points.shape)


def make_candidates(
    rng: np.random.Generator,
    D: Array,
    correct_mid: Array,
    correct_final: Array,
    field_ab: LocalField,
    field_bc: LocalField,
    family: str,
) -> List[Candidate]:
    """
    The key design:
        Some distractors deliberately have final shapes very close to the correct
        final shape but wrong path memory.

    Final-only matching often cannot separate them.
    Trajectory-aware scoring can.
    """
    candidates: List[Candidate] = []

    candidates.append(
        Candidate(
            name="correct_same_path",
            mid=noisy(correct_mid, rng, 0.0015),
            final=noisy(correct_final, rng, 0.0015),
            is_correct=True,
        )
    )

    # Same final, wrong midpoint: the classic Phase 31 failure mode.
    fake_mid_linear = 0.5 * D + 0.5 * correct_final
    candidates.append(
        Candidate(
            name="same_final_wrong_linear_mid",
            mid=noisy(fake_mid_linear, rng, 0.002),
            final=noisy(correct_final, rng, 0.0015),
            is_correct=False,
        )
    )

    # Same final, midpoint near D: destination correct, route wrong.
    fake_mid_lazy = 0.78 * D + 0.22 * correct_final
    candidates.append(
        Candidate(
            name="same_final_lazy_mid",
            mid=noisy(fake_mid_lazy, rng, 0.002),
            final=noisy(correct_final, rng, 0.0015),
            is_correct=False,
        )
    )

    # Reverse order: BC then AB.
    rev_mid = field_bc.apply(D)
    rev_final = field_ab.apply(rev_mid)
    candidates.append(
        Candidate(
            name="reverse_composition",
            mid=noisy(rev_mid, rng, 0.002),
            final=noisy(rev_final, rng, 0.002),
            is_correct=False,
        )
    )

    # Only AB.
    only_ab = field_ab.apply(D)
    candidates.append(
        Candidate(
            name="only_first_relation",
            mid=noisy(only_ab, rng, 0.002),
            final=noisy(only_ab, rng, 0.002),
            is_correct=False,
        )
    )

    # Only BC.
    only_bc = field_bc.apply(D)
    candidates.append(
        Candidate(
            name="only_second_relation",
            mid=noisy(only_bc, rng, 0.002),
            final=noisy(only_bc, rng, 0.002),
            is_correct=False,
        )
    )

    # Wrong global affine drift.
    theta = rng.uniform(-0.25, 0.25)
    wrong_beta = beta_from_matrix_translation(rot(theta) @ scale(rng.uniform(0.9, 1.12)), rng.uniform(-0.15, 0.15, size=2))
    wrong_mid = apply_beta(D, wrong_beta)
    wrong_final = apply_beta(wrong_mid, wrong_beta)
    candidates.append(
        Candidate(
            name="wrong_global_drift",
            mid=noisy(wrong_mid, rng, 0.002),
            final=noisy(wrong_final, rng, 0.002),
            is_correct=False,
        )
    )

    rng.shuffle(candidates)
    return candidates


# =============================================================================
# Scoring
# =============================================================================

def trajectory_samples(start: Array, mid: Array, final: Array, steps_per_leg: int = 4) -> List[Array]:
    """
    Discrete path-memory samples.

    This keeps Phase 32 conceptually clean:
        not just where the answer ended,
        but how it moved.
    """
    samples: List[Array] = []

    for i in range(steps_per_leg + 1):
        t = i / steps_per_leg
        samples.append((1.0 - t) * start + t * mid)

    for i in range(1, steps_per_leg + 1):
        t = i / steps_per_leg
        samples.append((1.0 - t) * mid + t * final)

    return samples


def score_final_only(pred_final: Array, cand: Candidate) -> float:
    return chamfer(pred_final, cand.final)


def score_trajectory(start: Array, pred_mid: Array, pred_final: Array, cand: Candidate) -> float:
    pred_path = trajectory_samples(start, pred_mid, pred_final, steps_per_leg=5)
    cand_path = trajectory_samples(start, cand.mid, cand.final, steps_per_leg=5)

    path_err = float(np.mean([chamfer(a, b) for a, b in zip(pred_path, cand_path)]))
    mid_err = chamfer(pred_mid, cand.mid)
    final_err = chamfer(pred_final, cand.final)

    # Final still matters, but path memory dominates same-final ambiguities.
    return 0.55 * path_err + 0.30 * mid_err + 0.15 * final_err


def choose_candidate(scores: List[float], candidates: List[Candidate]) -> Tuple[int, float]:
    order = np.argsort(np.array(scores))
    best_i = int(order[0])
    runner_i = int(order[1])
    margin = float(scores[runner_i] - scores[best_i])
    return best_i, margin


# =============================================================================
# Visualization
# =============================================================================

def draw_shape(ax, pts: Array, title: str) -> None:
    ax.scatter(pts[:, 0], pts[:, 1], s=12)
    ax.set_title(title, fontsize=10)
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_path(ax, start: Array, mid: Array, final: Array, title: str) -> None:
    ax.scatter(start[:, 0], start[:, 1], s=10, label="start")
    ax.scatter(mid[:, 0], mid[:, 1], s=10, label="mid")
    ax.scatter(final[:, 0], final[:, 1], s=10, label="final")

    # draw a sparse set of movement lines
    for i in range(0, len(start), max(1, len(start) // 16)):
        ax.plot([start[i, 0], mid[i, 0], final[i, 0]], [start[i, 1], mid[i, 1], final[i, 1]], linewidth=0.8)

    ax.set_title(title, fontsize=10)
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def write_example_png(
    trial_id: int,
    family: str,
    A: Array,
    B: Array,
    C: Array,
    D: Array,
    pred_mid: Array,
    pred_final: Array,
    chosen: Candidate,
    correct: Candidate,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(2, 4, 1)
    draw_shape(ax1, A, "A")

    ax2 = fig.add_subplot(2, 4, 2)
    draw_shape(ax2, B, "B = A→B")

    ax3 = fig.add_subplot(2, 4, 3)
    draw_shape(ax3, C, "C = B→C")

    ax4 = fig.add_subplot(2, 4, 4)
    draw_path(ax4, A, B, C, "source path A→B→C")

    ax5 = fig.add_subplot(2, 4, 5)
    draw_shape(ax5, D, "D query")

    ax6 = fig.add_subplot(2, 4, 6)
    draw_path(ax6, D, pred_mid, pred_final, "inferred transferred path")

    ax7 = fig.add_subplot(2, 4, 7)
    draw_path(ax7, D, chosen.mid, chosen.final, f"chosen: {chosen.name}")

    ax8 = fig.add_subplot(2, 4, 8)
    draw_path(ax8, D, correct.mid, correct.final, "true same-path answer")

    fig.suptitle(f"Phase 32 example {trial_id} — {family}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# =============================================================================
# Trial loop
# =============================================================================

def run_trial(seed: int, trial_id: int) -> Dict[str, object]:
    rng = rng_for(seed)

    family = rng.choice(FAMILIES).item() if hasattr(rng.choice(FAMILIES), "item") else rng.choice(FAMILIES)

    A = make_base_shape(rng)
    field1_true = make_random_field(rng, family, A, step=1)
    B_clean = field1_true.apply(A)

    field2_true = make_random_field(rng, family, B_clean, step=2)
    C_clean = field2_true.apply(B_clean)

    noise_sigma = 0.0035
    B = B_clean + rng.normal(0.0, noise_sigma, size=B_clean.shape)
    C = C_clean + rng.normal(0.0, noise_sigma, size=C_clean.shape)

    D = make_base_shape(rng)

    # Infer A->B and B->C from observed source sequence.
    field_ab, inferred_part_ab, fit_ab, fit_scores_ab = infer_best_field(A, B)
    field_bc, inferred_part_bc, fit_bc, fit_scores_bc = infer_best_field(B, C)

    pred_mid = field_ab.apply(D)
    pred_final = field_bc.apply(pred_mid)

    # True transfer, using the actual hidden fields.
    true_mid = field1_true.apply(D)
    true_final = field2_true.apply(true_mid)

    candidates = make_candidates(rng, D, true_mid, true_final, field1_true, field2_true, family)

    traj_scores = [score_trajectory(D, pred_mid, pred_final, c) for c in candidates]
    final_scores = [score_final_only(pred_final, c) for c in candidates]

    chosen_i, traj_margin = choose_candidate(traj_scores, candidates)
    final_i, final_margin = choose_candidate(final_scores, candidates)

    chosen = candidates[chosen_i]
    final_chosen = candidates[final_i]

    correct_i = next(i for i, c in enumerate(candidates) if c.is_correct)
    correct = candidates[correct_i]

    traj_correct = bool(chosen.is_correct)
    final_correct = bool(final_chosen.is_correct)

    # Scramble stability: shuffle candidates repeatedly and require the same semantic
    # choice to survive candidate order.
    stable_hits = 0
    repeats = 5
    for _ in range(repeats):
        perm = list(range(len(candidates)))
        rng.shuffle(perm)
        c2 = [candidates[i] for i in perm]
        s2 = [score_trajectory(D, pred_mid, pred_final, c) for c in c2]
        b2, _ = choose_candidate(s2, c2)
        if c2[b2].is_correct:
            stable_hits += 1

    scramble_stability = stable_hits / repeats

    true_partition = partition_for_family(family)
    family_match = inferred_part_ab == true_partition and inferred_part_bc == true_partition

    transfer_error = chamfer(pred_final, true_final)
    mid_transfer_error = chamfer(pred_mid, true_mid)

    return {
        "trial_id": trial_id,
        "seed": seed,
        "family": family,
        "true_partition": true_partition,
        "inferred_part_ab": inferred_part_ab,
        "inferred_part_bc": inferred_part_bc,
        "family_match": family_match,
        "fit_ab": fit_ab,
        "fit_bc": fit_bc,
        "mid_transfer_error": mid_transfer_error,
        "transfer_error": transfer_error,
        "trajectory_correct": traj_correct,
        "final_only_correct": final_correct,
        "scramble_stability": scramble_stability,
        "trajectory_margin": traj_margin,
        "final_only_margin": final_margin,
        "chosen_name": chosen.name,
        "final_only_chosen_name": final_chosen.name,
        "correct_index": correct_i,
        "candidate_count": len(candidates),
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "pred_mid": pred_mid,
        "pred_final": pred_final,
        "chosen": chosen,
        "correct": correct,
    }


# =============================================================================
# Output helpers
# =============================================================================

def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keep_keys = [
        "trial_id",
        "seed",
        "family",
        "true_partition",
        "inferred_part_ab",
        "inferred_part_bc",
        "family_match",
        "fit_ab",
        "fit_bc",
        "mid_transfer_error",
        "transfer_error",
        "trajectory_correct",
        "final_only_correct",
        "scramble_stability",
        "trajectory_margin",
        "final_only_margin",
        "chosen_name",
        "final_only_chosen_name",
        "correct_index",
        "candidate_count",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keep_keys})


def family_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for fam in FAMILIES:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        out.append(
            {
                "family": fam,
                "trials": len(sub),
                "trajectory_accuracy": float(np.mean([bool(r["trajectory_correct"]) for r in sub])),
                "final_only_accuracy": float(np.mean([bool(r["final_only_correct"]) for r in sub])),
                "scramble_stability": float(np.mean([float(r["scramble_stability"]) for r in sub])),
                "family_match_rate": float(np.mean([bool(r["family_match"]) for r in sub])),
                "mean_fit_ab": float(np.mean([float(r["fit_ab"]) for r in sub])),
                "mean_fit_bc": float(np.mean([float(r["fit_bc"]) for r in sub])),
                "mean_mid_transfer_error": float(np.mean([float(r["mid_transfer_error"]) for r in sub])),
                "mean_transfer_error": float(np.mean([float(r["transfer_error"]) for r in sub])),
                "min_trajectory_margin": float(np.min([float(r["trajectory_margin"]) for r in sub])),
                "mean_trajectory_margin": float(np.mean([float(r["trajectory_margin"]) for r in sub])),
                "final_only_same_final_trap_rate": float(
                    np.mean(
                        [
                            str(r["final_only_chosen_name"]).startswith("same_final")
                            and not bool(r["final_only_correct"])
                            for r in sub
                        ]
                    )
                ),
            }
        )
    return out


def plot_accuracy(fam_rows: List[Dict[str, object]], path: Path) -> None:
    labels = [r["family"] for r in fam_rows]
    vals = [r["trajectory_accuracy"] for r in fam_rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("trajectory-aware accuracy")
    ax.set_title("Phase 32 trajectory-memory geometric thought: accuracy by hidden family")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_margins(rows: List[Dict[str, object]], path: Path) -> None:
    vals = [float(r["trajectory_margin"]) for r in rows]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(vals, bins=40)
    ax.set_title("Phase 32 trajectory-aware answer margin distribution")
    ax.set_xlabel("runner-up trajectory score - best trajectory score")
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_baseline_gap(fam_rows: List[Dict[str, object]], path: Path) -> None:
    labels = [r["family"] for r in fam_rows]
    traj = np.array([r["trajectory_accuracy"] for r in fam_rows])
    final = np.array([r["final_only_accuracy"] for r in fam_rows])

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width / 2, final, width, label="final-only")
    ax.bar(x + width / 2, traj, width, label="trajectory-memory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 32 final-only baseline vs trajectory-memory reasoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_family_match(fam_rows: List[Dict[str, object]], path: Path) -> None:
    labels = [r["family"] for r in fam_rows]
    vals = [r["family_match_rate"] for r in fam_rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("partition-family match rate")
    ax.set_title("Phase 32 inferred path partition-family match")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(summary: Dict[str, object], fam_rows: List[Dict[str, object]], path: Path) -> None:
    lines: List[str] = []

    lines.append("# Phase 32 — Trajectory-memory geometric thought reasoner")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Phase 31 showed a structural problem: final target matching alone is not enough.")
    lines.append("Some composed transformations can land in nearly the same final geometry while having different histories.")
    lines.append("")
    lines.append("Phase 32 therefore tests path memory:")
    lines.append("")
    lines.append("```text")
    lines.append("A -> B")
    lines.append("B -> C")
    lines.append("Infer the route A -> B -> C.")
    lines.append("Transfer that same route onto D.")
    lines.append("Choose the candidate that preserves both the intermediate path and final destination.")
    lines.append("```")
    lines.append("")
    lines.append("This moves BBIT closer to tokenless reasoning because the answer is not a label.")
    lines.append("The answer is the continuity of a geometric transformation through an inferred path.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- `{PASS_KEY}`: `{summary[PASS_KEY]}`")
    lines.append(f"- trajectory-aware accuracy: `{summary['trajectory_accuracy']:.4f}`")
    lines.append(f"- final-only baseline accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- family match rate: `{summary['family_match_rate']:.4f}`")
    lines.append(f"- mean mid-transfer error: `{summary['mean_mid_transfer_error']:.6f}`")
    lines.append(f"- mean final-transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append(f"- mean trajectory margin: `{summary['mean_trajectory_margin']:.6f}`")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| family | trials | trajectory acc | final-only acc | stability | family match | transfer err | min margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in fam_rows:
        lines.append(
            f"| {r['family']} | {r['trials']} | "
            f"{r['trajectory_accuracy']:.3f} | "
            f"{r['final_only_accuracy']:.3f} | "
            f"{r['scramble_stability']:.3f} | "
            f"{r['family_match_rate']:.3f} | "
            f"{r['mean_transfer_error']:.6f} | "
            f"{r['min_trajectory_margin']:.6f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Phase 32 is designed to separate two kinds of correctness:")
    lines.append("")
    lines.append("1. **Destination correctness** — did the answer land in the right place?")
    lines.append("2. **Route correctness** — did the answer get there through the same geometric path?")
    lines.append("")
    lines.append("A final-only reasoner can be fooled by candidates that share the same final shape but have the wrong midpoint.")
    lines.append("The trajectory-memory reasoner scores the full path and should therefore select the candidate with the correct transformation history.")
    lines.append("")
    lines.append("The example PNGs in `phase32_examples/` show the source chain, transferred inferred chain, chosen candidate path, and correct path.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from final-answer composition to trajectory-memory geometric thought")
    print(f"[{PHASE}] task: infer A->B->C path, transfer path onto D, choose same-route answer")

    trials = 700
    base_seed = 320032

    rows: List[Dict[str, object]] = []
    for i in range(trials):
        rows.append(run_trial(seed=base_seed + i, trial_id=i))

    traj_acc = float(np.mean([bool(r["trajectory_correct"]) for r in rows]))
    final_acc = float(np.mean([bool(r["final_only_correct"]) for r in rows]))
    stability = float(np.mean([float(r["scramble_stability"]) for r in rows]))
    fam_match = float(np.mean([bool(r["family_match"]) for r in rows]))
    mean_mid_transfer = float(np.mean([float(r["mid_transfer_error"]) for r in rows]))
    mean_transfer = float(np.mean([float(r["transfer_error"]) for r in rows]))
    mean_margin = float(np.mean([float(r["trajectory_margin"]) for r in rows]))
    min_margin = float(np.min([float(r["trajectory_margin"]) for r in rows]))

    # This phase should prove that path memory improves over final-only scoring.
    pass_flag = (
        traj_acc >= 0.94
        and stability >= 0.94
        and mean_transfer <= 0.035
        and traj_acc > final_acc + 0.20
    )

    fam_rows = family_summary(rows)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        PASS_KEY: pass_flag,
        "trials": trials,
        "trajectory_accuracy": traj_acc,
        "final_only_accuracy": final_acc,
        "trajectory_gain_over_final_only": traj_acc - final_acc,
        "scramble_stability": stability,
        "family_match_rate": fam_match,
        "mean_mid_transfer_error": mean_mid_transfer,
        "mean_transfer_error": mean_transfer,
        "mean_trajectory_margin": mean_margin,
        "min_trajectory_margin": min_margin,
        "families": fam_rows,
        "outputs": {
            "trials_csv": str(OUT / "phase32_trajectory_memory_trials.csv"),
            "summary_json": str(OUT / "phase32_trajectory_memory_summary.json"),
            "report_md": str(OUT / "phase32_trajectory_memory_report.md"),
            "accuracy_png": str(OUT / "phase32_trajectory_memory_accuracy.png"),
            "margin_png": str(OUT / "phase32_trajectory_memory_margin_distribution.png"),
            "baseline_gap_png": str(OUT / "phase32_trajectory_memory_baseline_gap.png"),
            "family_match_png": str(OUT / "phase32_trajectory_memory_family_match.png"),
            "examples_dir": str(EXAMPLE_DIR),
        },
    }

    # Write example visualizations.
    # Mix correct and incorrect examples if possible.
    example_rows: List[Dict[str, object]] = []
    example_rows.extend([r for r in rows if bool(r["trajectory_correct"])][:4])
    example_rows.extend([r for r in rows if not bool(r["trajectory_correct"])][:4])
    example_rows = example_rows[:8]

    for idx, r in enumerate(example_rows):
        write_example_png(
            trial_id=int(r["trial_id"]),
            family=str(r["family"]),
            A=r["A"],
            B=r["B"],
            C=r["C"],
            D=r["D"],
            pred_mid=r["pred_mid"],
            pred_final=r["pred_final"],
            chosen=r["chosen"],
            correct=r["correct"],
            out_path=EXAMPLE_DIR / f"phase32_example_{idx:02d}_trial_{int(r['trial_id']):04d}_{r['family']}.png",
        )

    # Strip arrays/objects before CSV/report writing.
    write_csv(rows, OUT / "phase32_trajectory_memory_trials.csv")

    (OUT / "phase32_trajectory_memory_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_report(summary, fam_rows, OUT / "phase32_trajectory_memory_report.md")

    plot_accuracy(fam_rows, OUT / "phase32_trajectory_memory_accuracy.png")
    plot_margins(rows, OUT / "phase32_trajectory_memory_margin_distribution.png")
    plot_baseline_gap(fam_rows, OUT / "phase32_trajectory_memory_baseline_gap.png")
    plot_family_match(fam_rows, OUT / "phase32_trajectory_memory_family_match.png")

    print(f"[{PHASE}] {PASS_KEY}={pass_flag}")
    print(
        f"[{PHASE}] trajectory_accuracy={traj_acc:.4f} "
        f"final_only_accuracy={final_acc:.4f} "
        f"gain={traj_acc - final_acc:.4f} "
        f"scramble_stability={stability:.4f} "
        f"trials={trials}"
    )
    print(
        f"[{PHASE}] family_match_rate={fam_match:.4f} "
        f"mean_mid_transfer_error={mean_mid_transfer:.6f} "
        f"mean_transfer_error={mean_transfer:.6f} "
        f"mean_margin={mean_margin:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for r in fam_rows:
        print(
            f"  - {r['family']:<16} "
            f"traj_acc={r['trajectory_accuracy']:.3f} "
            f"final_acc={r['final_only_accuracy']:.3f} "
            f"stable={r['scramble_stability']:.3f} "
            f"family_match={r['family_match_rate']:.3f} "
            f"transfer_err={r['mean_transfer_error']:.6f} "
            f"min_margin={r['min_trajectory_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {OUT / 'phase32_trajectory_memory_trials.csv'}")
    print(f"[{PHASE}] wrote summary: {OUT / 'phase32_trajectory_memory_summary.json'}")
    print(f"[{PHASE}] wrote report: {OUT / 'phase32_trajectory_memory_report.md'}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())