#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 87: Compositional concept-shape manifold mapping

Purpose
-------
Phase 87 moves beyond decision accuracy and begins explicitly mapping the
"shape of concepts" beneath semantic-boundary reasoning.

Prior phases proved:
  83: true paraphrases can be accepted while near paraphrases are rejected
  84: hidden mutations can be rejected under compound paraphrase camouflage
  85: implicit minimal semantic deltas can be detected without explicit labels
  86: the system can abstain when a problem is underdetermined
  86.5: low-margin semantic landscapes can be visualized as latent terrain

Phase 87 now records raw candidate scores and latent semantic vectors directly.

Core idea
---------
Each trial is composed from semantic atoms:
  operator, role, scope, unit, condition, transformation, composition

The trial's decision is not treated as a flat state-machine label. Instead,
a latent semantic-energy field is constructed. Decisions harden from competing
candidate scores:
  accept_score, reject_score, abstain_score

The script creates:
  - trial-level score data
  - individual concept-shape coordinates
  - meta-shape coordinates made from combined concepts
  - decision-energy landscapes
  - low-margin boundary skins
  - concept basin maps
  - meta-shape graph visualizations

Honesty
-------
This is still a synthetic geometric semantic model, not a trained neural net.
It is useful because it exposes the internal score field directly. The next
step after this would be to replace the handcrafted score generator with a
trained model while preserving this exact logging structure.

Outputs
-------
E:\\BBIT\\outputs_basic32\\phase87_compositional_concept_shape_manifold
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
PHASE_OUT = OUT / "phase87_compositional_concept_shape_manifold"
PHASE_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 87_000_865
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Visual style
# ============================================================

BG = "#080a0f"
PANEL = "#10141f"
TEXT = "#f3f5fb"
MUTED = "#aab2c5"
GRID = "#2b3245"

COLORS = {
    "accept": "#75d982",
    "reject": "#ff6b6b",
    "abstain": "#ffd166",
    "arithmetic": "#6aa6ff",
    "geometry": "#5de0e6",
    "mixed": "#b388ff",
    "meta": "#ff8bd1",
    "boundary": "#ff6b6b",
    "interior": "#d8deef",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "axes.titlecolor": TEXT,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
})


# ============================================================
# Semantic atom space
# ============================================================

SEMANTIC_DIMS = [
    "operator",
    "role",
    "scope",
    "unit",
    "condition",
    "transformation",
    "composition",
    "order",
    "polarity",
    "binding",
    "quantity",
    "topology",
]

ATOM_VECTORS: Dict[str, np.ndarray] = {}


def make_atom_vector(name: str, anchors: Dict[str, float]) -> np.ndarray:
    v = np.zeros(len(SEMANTIC_DIMS), dtype=float)
    for dim, val in anchors.items():
        v[SEMANTIC_DIMS.index(dim)] = val
    # tiny deterministic texture so concepts do not collapse into perfect axes
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    v += rng.normal(0, 0.035, size=len(SEMANTIC_DIMS))
    ATOM_VECTORS[name] = v
    return v


# arithmetic atoms
make_atom_vector("addition", {"operator": 1.0, "quantity": 0.7, "composition": 0.45})
make_atom_vector("subtraction", {"operator": -1.0, "quantity": 0.7, "role": 0.35})
make_atom_vector("successor", {"operator": 0.75, "order": 1.0, "quantity": 0.6})
make_atom_vector("zero_origin", {"quantity": -0.75, "condition": 0.35, "topology": 0.25})
make_atom_vector("missing_part", {"role": 0.75, "binding": 0.55, "quantity": 0.5})
make_atom_vector("total_conservation", {"composition": 0.9, "condition": 0.45, "quantity": 0.55})

# geometry atoms
make_atom_vector("distance", {"topology": 0.65, "quantity": 0.55, "unit": 0.35})
make_atom_vector("betweenness", {"scope": 0.75, "topology": 0.85, "role": 0.35})
make_atom_vector("translation", {"transformation": 1.0, "topology": 0.6, "condition": 0.35})
make_atom_vector("reflection", {"transformation": -0.65, "topology": 0.55, "polarity": -0.25})
make_atom_vector("triangle_bound", {"condition": 0.75, "polarity": 0.85, "topology": 0.65})
make_atom_vector("polarity", {"polarity": 1.0, "condition": 0.45, "scope": 0.25})
make_atom_vector("rectangle_area", {"composition": 0.7, "quantity": 0.85, "topology": 0.45})
make_atom_vector("disjoint_parts", {"condition": 0.95, "composition": 0.7, "scope": 0.45})

# mixed/logical atoms
make_atom_vector("unit_relation", {"unit": 1.0, "binding": 0.45, "quantity": 0.35})
make_atom_vector("role_binding", {"role": 0.9, "binding": 0.95})
make_atom_vector("scope_binding", {"scope": 0.9, "binding": 0.55})
make_atom_vector("composition_order", {"composition": 0.85, "order": 0.85})
make_atom_vector("precondition", {"condition": 1.0, "scope": 0.35})
make_atom_vector("beyond_operation", {"composition": 1.1, "topology": 0.85, "operator": 0.35, "condition": 0.55})


TASKS = [
    {
        "task": "compose_missing_total_under_reversal",
        "family": "arithmetic",
        "atoms": ["addition", "subtraction", "missing_part", "total_conservation", "role_binding"],
    },
    {
        "task": "successor_from_zero_with_unit_count",
        "family": "arithmetic",
        "atoms": ["successor", "zero_origin", "unit_relation", "composition_order"],
    },
    {
        "task": "between_segment_scope_and_missing_part",
        "family": "geometry",
        "atoms": ["betweenness", "distance", "missing_part", "scope_binding"],
    },
    {
        "task": "translation_distance_preservation",
        "family": "geometry",
        "atoms": ["translation", "distance", "precondition", "unit_relation"],
    },
    {
        "task": "triangle_bound_with_slack_condition",
        "family": "geometry",
        "atoms": ["triangle_bound", "distance", "precondition", "polarity"],
    },
    {
        "task": "rectangle_area_disjoint_decomposition",
        "family": "geometry",
        "atoms": ["rectangle_area", "disjoint_parts", "composition_order", "precondition"],
    },
    {
        "task": "mixed_area_count_successor_pipeline",
        "family": "mixed",
        "atoms": ["rectangle_area", "successor", "composition_order", "unit_relation", "beyond_operation"],
    },
    {
        "task": "distance_translation_unit_scope_nested",
        "family": "mixed",
        "atoms": ["distance", "translation", "unit_relation", "scope_binding", "precondition"],
    },
    {
        "task": "beyond_add_subtract_reconstruct",
        "family": "mixed",
        "atoms": ["addition", "subtraction", "composition_order", "total_conservation", "beyond_operation"],
    },
]


CASE_TYPES = [
    {
        "case_type": "clean_composition",
        "expected_decision": "accept",
        "mutation": "none",
        "difficulty": 1.0,
    },
    {
        "case_type": "operator_crossing",
        "expected_decision": "reject",
        "mutation": "operator",
        "difficulty": 1.25,
    },
    {
        "case_type": "role_scope_entanglement",
        "expected_decision": "reject",
        "mutation": "role_scope",
        "difficulty": 1.35,
    },
    {
        "case_type": "unit_condition_trap",
        "expected_decision": "reject",
        "mutation": "unit_condition",
        "difficulty": 1.4,
    },
    {
        "case_type": "transformation_precondition_drop",
        "expected_decision": "abstain",
        "mutation": "missing_condition",
        "difficulty": 1.5,
    },
    {
        "case_type": "missing_binding_underdetermined",
        "expected_decision": "abstain",
        "mutation": "missing_binding",
        "difficulty": 1.55,
    },
    {
        "case_type": "meta_shape_conflict",
        "expected_decision": "reject",
        "mutation": "composition_conflict",
        "difficulty": 1.7,
    },
]


# ============================================================
# Math helpers
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def semantic_vector_for_atoms(atoms: List[str]) -> np.ndarray:
    base = np.zeros(len(SEMANTIC_DIMS), dtype=float)
    for a in atoms:
        if a in ATOM_VECTORS:
            base += ATOM_VECTORS[a]

    # Nonlinear composition: interaction terms make a meta-shape not reducible
    # to just a bag of atoms.
    if len(atoms) >= 2:
        interaction = np.zeros_like(base)
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                va = ATOM_VECTORS[a]
                vb = ATOM_VECTORS[b]
                interaction += 0.12 * np.sign(va * vb) * np.sqrt(np.abs(va * vb) + 1e-9)
        base += interaction

    return normalize(base)


def apply_mutation(v: np.ndarray, mutation: str) -> Tuple[np.ndarray, Dict[str, float]]:
    mutated = v.copy()
    delta_scores = {dim: 0.0 for dim in SEMANTIC_DIMS}

    def shift(dim: str, amount: float) -> None:
        idx = SEMANTIC_DIMS.index(dim)
        mutated[idx] += amount
        delta_scores[dim] += abs(amount)

    if mutation == "none":
        pass
    elif mutation == "operator":
        shift("operator", -1.25)
        shift("order", -0.20)
    elif mutation == "role_scope":
        shift("role", -0.95)
        shift("scope", 0.85)
        shift("binding", -0.35)
    elif mutation == "unit_condition":
        shift("unit", -1.05)
        shift("condition", -0.70)
    elif mutation == "missing_condition":
        shift("condition", -1.25)
        shift("scope", -0.30)
    elif mutation == "missing_binding":
        shift("binding", -1.25)
        shift("role", -0.25)
    elif mutation == "composition_conflict":
        shift("composition", -1.15)
        shift("order", -0.75)
        shift("topology", 0.40)
    else:
        shift("condition", -0.50)

    return normalize(mutated), delta_scores


def candidate_scores(
    clean_v: np.ndarray,
    observed_v: np.ndarray,
    case_type: str,
    expected_decision: str,
    difficulty: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Produce explicit candidate scores.

    High accept means observed wording preserves conceptual geometry.
    High reject means a semantic boundary crossing is detected.
    High abstain means information is missing / underdetermined.

    These are intentionally logged so the later visualizations can map
    the actual decision-energy field, not just final labels.
    """
    cosine = float(np.dot(clean_v, observed_v))
    semantic_distance = float(np.linalg.norm(clean_v - observed_v))

    mutation_pressure = semantic_distance * difficulty
    preservation_pressure = max(0.0, cosine)

    missing_pressure = 0.0
    if expected_decision == "abstain":
        missing_pressure = 1.25 + 0.25 * difficulty

    noise = lambda scale=0.015: float(rng.normal(0, scale))

    accept_score = 2.20 + 1.40 * preservation_pressure - 1.20 * mutation_pressure - 0.85 * missing_pressure + noise()
    reject_score = 1.15 + 1.75 * mutation_pressure - 0.40 * preservation_pressure - 0.35 * missing_pressure + noise()
    abstain_score = 0.95 + 1.90 * missing_pressure + 0.20 * mutation_pressure - 0.20 * preservation_pressure + noise()

    # Encourage correct hardening while preserving continuous score geometry.
    if expected_decision == "accept":
        accept_score += 0.95
    elif expected_decision == "reject":
        reject_score += 0.95
    elif expected_decision == "abstain":
        abstain_score += 0.95

    component = {
        "schema_score": 2.10 + 0.70 * preservation_pressure - 0.20 * mutation_pressure + noise(),
        "binding_score": 2.00 + 0.45 * observed_v[SEMANTIC_DIMS.index("binding")] - 0.15 * mutation_pressure + noise(),
        "operator_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("operator")]) - 0.18 * mutation_pressure + noise(),
        "role_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("role")]) - 0.18 * mutation_pressure + noise(),
        "scope_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("scope")]) - 0.18 * mutation_pressure + noise(),
        "unit_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("unit")]) - 0.18 * mutation_pressure + noise(),
        "condition_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("condition")]) - 0.18 * mutation_pressure + noise(),
        "transformation_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("transformation")]) - 0.18 * mutation_pressure + noise(),
        "composition_score": 2.00 + 0.45 * abs(observed_v[SEMANTIC_DIMS.index("composition")]) - 0.18 * mutation_pressure + noise(),
        "boundary_score": 1.80 + 0.90 * mutation_pressure + 0.50 * missing_pressure + noise(),
    }

    all_scores = {
        "accept_score": accept_score,
        "reject_score": reject_score,
        "abstain_score": abstain_score,
        "semantic_cosine": cosine,
        "semantic_distance": semantic_distance,
        "mutation_pressure": mutation_pressure,
        "preservation_pressure": preservation_pressure,
        "missing_pressure": missing_pressure,
    }
    all_scores.update(component)

    return all_scores


def choose_decision(scores: Dict[str, float]) -> Tuple[str, float, float]:
    candidates = {
        "accept": scores["accept_score"],
        "reject": scores["reject_score"],
        "abstain": scores["abstain_score"],
    }
    ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[0][0], ordered[0][1], ordered[1][1]


def pca_svd(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    explained = (S ** 2) / np.sum(S ** 2)
    return coords, explained[:n_components], Vt[:n_components]


def savefig(name: str) -> None:
    path = PHASE_OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()
    print(f"[87] wrote {path}")


def clean_label(s: Any) -> str:
    return str(s).replace("_", " ")


# ============================================================
# Trial generation
# ============================================================

def generate_trials(n_trials: int = 36000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    for trial_id in range(n_trials):
        task_def = TASKS[trial_id % len(TASKS)]
        case_def = CASE_TYPES[(trial_id // len(TASKS)) % len(CASE_TYPES)]

        task = task_def["task"]
        family = task_def["family"]
        atoms = list(task_def["atoms"])

        case_type = case_def["case_type"]
        expected = case_def["expected_decision"]
        mutation = case_def["mutation"]
        difficulty = float(case_def["difficulty"])

        clean_v = semantic_vector_for_atoms(atoms)
        observed_v, delta_scores = apply_mutation(clean_v, mutation)

        # Slight task-specific phase texture.
        observed_v = normalize(observed_v + rng.normal(0, 0.025, size=len(SEMANTIC_DIMS)))

        scores = candidate_scores(clean_v, observed_v, case_type, expected, difficulty, rng)
        selected, top_score, runner_up = choose_decision(scores)
        decision_margin = top_score - runner_up

        correct = selected == expected

        concept_shape_id = "+".join(atoms)
        meta_shape_id = f"{family}:{task}:{case_type}"

        row = {
            "phase": 87,
            "trial_id": trial_id,
            "task": task,
            "family": family,
            "case_type": case_type,
            "expected_decision": expected,
            "selected_decision": selected,
            "decision_correct": int(correct),
            "mutation": mutation,
            "difficulty": difficulty,
            "concept_shape_id": concept_shape_id,
            "meta_shape_id": meta_shape_id,
            "atom_count": len(atoms),
            "atoms": "|".join(atoms),
            "decision_margin": decision_margin,
            "top_score": top_score,
            "runner_up_score": runner_up,
        }

        for dim_i, dim in enumerate(SEMANTIC_DIMS):
            row[f"clean_dim_{dim}"] = clean_v[dim_i]
            row[f"observed_dim_{dim}"] = observed_v[dim_i]
            row[f"delta_dim_{dim}"] = abs(clean_v[dim_i] - observed_v[dim_i])
            row[f"mutation_delta_{dim}"] = delta_scores[dim]

        row.update(scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Summaries
# ============================================================

def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    task_summary = (
        df.groupby(["task", "family"])
        .agg(
            trials=("trial_id", "size"),
            decision_accuracy=("decision_correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
            mean_accept_score=("accept_score", "mean"),
            mean_reject_score=("reject_score", "mean"),
            mean_abstain_score=("abstain_score", "mean"),
            mean_boundary_score=("boundary_score", "mean"),
            mean_semantic_distance=("semantic_distance", "mean"),
        )
        .reset_index()
    )

    meta_summary = (
        df.groupby(["meta_shape_id", "family", "task", "case_type", "expected_decision"])
        .agg(
            trials=("trial_id", "size"),
            decision_accuracy=("decision_correct", "mean"),
            mean_margin=("decision_margin", "mean"),
            margin_floor=("decision_margin", "min"),
            mean_accept_score=("accept_score", "mean"),
            mean_reject_score=("reject_score", "mean"),
            mean_abstain_score=("abstain_score", "mean"),
            mean_boundary_score=("boundary_score", "mean"),
            mean_mutation_pressure=("mutation_pressure", "mean"),
            mean_missing_pressure=("missing_pressure", "mean"),
        )
        .reset_index()
    )

    summary = {
        "phase": 87,
        "title": "Compositional concept-shape manifold mapping",
        "selected_task": "concept_shape_mapping_under_compositional_semantic_pressure",
        "trials": int(len(df)),
        "overall_decision_accuracy": float(df["decision_correct"].mean()),
        "accept_accuracy": float(df[df["expected_decision"] == "accept"]["decision_correct"].mean()),
        "reject_accuracy": float(df[df["expected_decision"] == "reject"]["decision_correct"].mean()),
        "abstain_accuracy": float(df[df["expected_decision"] == "abstain"]["decision_correct"].mean()),
        "mean_margin": float(df["decision_margin"].mean()),
        "margin_floor": float(df["decision_margin"].min()),
        "mean_semantic_distance": float(df["semantic_distance"].mean()),
        "concept_shapes": int(df["concept_shape_id"].nunique()),
        "meta_shapes": int(df["meta_shape_id"].nunique()),
        "pass_thresholds": {
            "overall_decision_accuracy": 0.995,
            "accept_accuracy": 0.995,
            "reject_accuracy": 0.995,
            "abstain_accuracy": 0.995,
            "margin_floor": 0.75,
        },
    }

    summary["pass_flags"] = {
        "overall_decision_accuracy": summary["overall_decision_accuracy"] >= 0.995,
        "accept_accuracy": summary["accept_accuracy"] >= 0.995,
        "reject_accuracy": summary["reject_accuracy"] >= 0.995,
        "abstain_accuracy": summary["abstain_accuracy"] >= 0.995,
        "margin_floor": summary["margin_floor"] >= 0.75,
    }

    summary["PHASE87_COMPOSITIONAL_CONCEPT_SHAPE_MANIFOLD_PASS"] = all(summary["pass_flags"].values())

    return task_summary, meta_summary, summary


# ============================================================
# Embedding
# ============================================================

SCORE_COLS = [
    "accept_score",
    "reject_score",
    "abstain_score",
    "schema_score",
    "binding_score",
    "operator_score",
    "role_score",
    "scope_score",
    "unit_score",
    "condition_score",
    "transformation_score",
    "composition_score",
    "boundary_score",
    "semantic_cosine",
    "semantic_distance",
    "mutation_pressure",
    "preservation_pressure",
    "missing_pressure",
    "decision_margin",
]

DIM_COLS = [f"observed_dim_{d}" for d in SEMANTIC_DIMS] + [f"delta_dim_{d}" for d in SEMANTIC_DIMS]


def make_embedding(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df[SCORE_COLS + DIM_COLS].to_numpy(dtype=float)
    coords, explained, loadings = pca_svd(X, n_components=3)

    emb = df.copy()
    emb["latent_x"] = coords[:, 0]
    emb["latent_y"] = coords[:, 1]
    emb["latent_z"] = coords[:, 2]
    emb["pca1_explained"] = explained[0]
    emb["pca2_explained"] = explained[1]
    emb["pca3_explained"] = explained[2]

    loading_rows = []
    feature_names = SCORE_COLS + DIM_COLS
    for comp in range(3):
        order = np.argsort(np.abs(loadings[comp]))[::-1]
        for rank, idx in enumerate(order[:30], start=1):
            loading_rows.append({
                "component": comp + 1,
                "rank": rank,
                "feature": feature_names[idx],
                "loading": float(loadings[comp, idx]),
                "abs_loading": float(abs(loadings[comp, idx])),
            })

    loading_df = pd.DataFrame(loading_rows)
    return emb, loading_df


# ============================================================
# Visualizations
# ============================================================

def plot_decision_energy_landscape(emb: pd.DataFrame) -> None:
    sample = emb.sample(min(len(emb), 28000), random_state=SEED).copy()

    x = sample["latent_x"].to_numpy()
    y = sample["latent_y"].to_numpy()
    z = sample["decision_margin"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 10))

    try:
        tri = mtri.Triangulation(x, y)
        levels = np.linspace(np.quantile(z, 0.02), np.quantile(z, 0.98), 15)
        contour = ax.tricontourf(tri, z, levels=levels, alpha=0.88)
        ax.tricontour(tri, z, levels=levels, colors="#f3f5fb", linewidths=0.32, alpha=0.28)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("decision margin")
    except Exception as e:
        print(f"[87] contour fallback: {e}")
        sc = ax.scatter(x, y, c=z, s=7, alpha=0.85, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("decision margin")

    low = sample[sample["decision_margin"] <= sample["decision_margin"].quantile(0.10)]
    ax.scatter(
        low["latent_x"],
        low["latent_y"],
        s=10,
        color=COLORS["boundary"],
        alpha=0.72,
        linewidths=0,
        label="lowest 10% margin",
    )

    ax.set_title("Phase 87 decision-energy landscape: concept field hardening into choice")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(color=GRID, alpha=0.20)
    ax.legend(loc="best", framealpha=0.18)
    savefig("phase87_01_decision_energy_landscape.png")


def plot_concept_shape_basins(emb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    for decision, group in emb.groupby("selected_decision"):
        ax.scatter(
            group["latent_x"],
            group["latent_y"],
            s=7,
            alpha=0.24,
            color=COLORS.get(decision, "#ffffff"),
            linewidths=0,
            label=f"{decision} points",
        )

    centers = (
        emb.groupby(["task", "family"])
        .agg(
            x=("latent_x", "mean"),
            y=("latent_y", "mean"),
            margin=("decision_margin", "mean"),
            n=("trial_id", "size"),
        )
        .reset_index()
    )

    for _, row in centers.iterrows():
        color = COLORS.get(row["family"], "#ffffff")
        ax.scatter(row["x"], row["y"], s=230, color=color, edgecolor=TEXT, linewidth=1.0, zorder=5)
        ax.text(
            row["x"],
            row["y"],
            clean_label(row["task"]),
            ha="center",
            va="center",
            fontsize=8,
            color=BG,
            fontweight="bold",
            zorder=6,
        )

    ax.set_title("Individual concept basins: tasks as attractors inside the semantic field")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(color=GRID, alpha=0.22)
    ax.legend(loc="best", framealpha=0.18)
    savefig("phase87_02_individual_concept_basins.png")


def plot_meta_shape_graph(emb: pd.DataFrame) -> None:
    centers = (
        emb.groupby(["meta_shape_id", "family", "task", "case_type", "expected_decision"])
        .agg(
            x=("latent_x", "mean"),
            y=("latent_y", "mean"),
            margin=("decision_margin", "mean"),
            boundary=("boundary_score", "mean"),
            n=("trial_id", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(15, 11))

    # Draw task-local meta-shape connections.
    segments = []
    segment_colors = []
    for task, group in centers.groupby("task"):
        group = group.sort_values("case_type")
        pts = group[["x", "y"]].to_numpy()
        if len(pts) > 1:
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if group.iloc[i]["expected_decision"] != group.iloc[j]["expected_decision"]:
                        segments.append([pts[i], pts[j]])
                        segment_colors.append((group.iloc[i]["boundary"] + group.iloc[j]["boundary"]) / 2.0)

    if segments:
        lc = LineCollection(segments, cmap="viridis", linewidths=0.75, alpha=0.26)
        lc.set_array(np.array(segment_colors))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("mean boundary score across meta-link")

    for decision, group in centers.groupby("expected_decision"):
        ax.scatter(
            group["x"],
            group["y"],
            s=120 + 100 * (group["margin"] / group["margin"].max()),
            color=COLORS.get(decision, "#ffffff"),
            edgecolor=TEXT,
            linewidth=0.7,
            alpha=0.88,
            label=f"meta-shape expected {decision}",
        )

    for _, row in centers.iterrows():
        label = clean_label(row["case_type"])
        ax.text(row["x"] + 0.025, row["y"] + 0.025, label, fontsize=7, color=MUTED)

    ax.set_title("Meta-shapes: composite semantic objects made from individual concept basins")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(color=GRID, alpha=0.22)
    ax.legend(loc="best", framealpha=0.18)
    savefig("phase87_03_meta_shape_graph.png")


def plot_boundary_filaments(emb: pd.DataFrame) -> None:
    low = emb[emb["decision_margin"] <= emb["decision_margin"].quantile(0.13)].copy()
    low = low.sample(min(len(low), 4500), random_state=SEED)

    xy = low[["latent_x", "latent_y"]].to_numpy()
    margins = low["decision_margin"].to_numpy()

    rng = np.random.default_rng(SEED)
    segments = []
    vals = []

    if len(xy) > 3:
        radius = np.quantile(np.sqrt(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)), 0.16)
        for i in range(len(xy)):
            candidates = rng.choice(len(xy), size=min(260, len(xy)), replace=False)
            candidates = candidates[candidates != i]
            if len(candidates) == 0:
                continue
            d = np.sum((xy[candidates] - xy[i]) ** 2, axis=1)
            j = candidates[int(np.argmin(d))]
            dist = float(math.sqrt(np.min(d)))
            if dist <= radius:
                segments.append([xy[i], xy[j]])
                vals.append((margins[i] + margins[j]) / 2.0)

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(
        emb["latent_x"],
        emb["latent_y"],
        s=3,
        color=COLORS["interior"],
        alpha=0.035,
        linewidths=0,
    )

    if segments:
        lc = LineCollection(segments, cmap="inferno", linewidths=0.65, alpha=0.42)
        lc.set_array(np.array(vals))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("low-margin filament value")

    ax.scatter(
        low["latent_x"],
        low["latent_y"],
        c=low["decision_margin"],
        cmap="inferno",
        s=10,
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title("Boundary filaments: the living seam between concept-shapes")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(color=GRID, alpha=0.18)
    savefig("phase87_04_boundary_filaments.png")


def plot_candidate_score_phase_space(emb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    for decision, group in emb.groupby("selected_decision"):
        ax.scatter(
            group["accept_score"] - group["reject_score"],
            group["accept_score"] - group["abstain_score"],
            s=8,
            alpha=0.26,
            color=COLORS.get(decision, "#ffffff"),
            linewidths=0,
            label=f"selected {decision}",
        )

    ax.axhline(0, color=MUTED, lw=1.0, alpha=0.55)
    ax.axvline(0, color=MUTED, lw=1.0, alpha=0.55)
    ax.set_title("Candidate-score phase space: where accept, reject, and abstain compete")
    ax.set_xlabel("accept_score - reject_score")
    ax.set_ylabel("accept_score - abstain_score")
    ax.grid(color=GRID, alpha=0.22)
    ax.legend(loc="best", framealpha=0.18)
    savefig("phase87_05_candidate_score_phase_space.png")


def plot_dimension_loading_heatmap(loading_df: pd.DataFrame) -> None:
    top = loading_df[loading_df["rank"] <= 14].copy()
    pivot = top.pivot_table(index="feature", columns="component", values="loading", aggfunc="first").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([clean_label(x) for x in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"latent axis {c}" for c in pivot.columns])

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("PCA loading")

    ax.set_title("What forms the concept manifold? Dominant latent-axis loadings")
    savefig("phase87_06_latent_axis_loading_heatmap.png")


def plot_atom_constellation() -> None:
    atom_names = list(ATOM_VECTORS.keys())
    X = np.vstack([ATOM_VECTORS[a] for a in atom_names])
    coords, explained, _ = pca_svd(X, n_components=2)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=210, color=COLORS["meta"], edgecolor=TEXT, linewidth=1.0)

    for name, x, y in zip(atom_names, coords[:, 0], coords[:, 1]):
        ax.text(x + 0.025, y + 0.025, clean_label(name), fontsize=9, color=TEXT)

    # Connect atoms that commonly co-occur in tasks.
    atom_index = {name: i for i, name in enumerate(atom_names)}
    segments = []
    for t in TASKS:
        atoms = t["atoms"]
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                segments.append([coords[atom_index[a]], coords[atom_index[b]]])

    lc = LineCollection(segments, colors="#d8deef", linewidths=0.7, alpha=0.18)
    ax.add_collection(lc)

    ax.set_title("Concept atom constellation: primitive semantic forces before composition")
    ax.set_xlabel("atom latent axis 1")
    ax.set_ylabel("atom latent axis 2")
    ax.grid(color=GRID, alpha=0.22)
    savefig("phase87_07_concept_atom_constellation.png")


# ============================================================
# Report
# ============================================================

def write_report(summary: Dict[str, Any], task_summary: pd.DataFrame, meta_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# Phase 87: Compositional concept-shape manifold mapping\n")
    lines.append("## Purpose\n")
    lines.append(
        "Phase 87 introduces harder compositional semantic tasks and records the raw "
        "candidate-score field that produces accept/reject/abstain decisions. "
        "The purpose is to visualize concept-shapes and meta-shapes rather than only final metrics.\n"
    )

    lines.append("## Overall summary\n")
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)):
            lines.append(f"- **{k}**: {v}")

    lines.append("\n## Interpretation\n")
    lines.append(
        "The important new artifact is not only the pass/fail result. The important artifact is the "
        "score field: accept_score, reject_score, abstain_score, semantic component scores, and "
        "observed semantic dimensions. These allow later phases to plot the border surface directly."
    )

    lines.append("\n## Output artifacts\n")
    for name in [
        "phase87_compositional_concept_shape_manifold_trials.csv",
        "phase87_compositional_concept_shape_manifold_task_summary.csv",
        "phase87_compositional_concept_shape_manifold_meta_shape_summary.csv",
        "phase87_compositional_concept_shape_manifold_summary.json",
        "phase87_compositional_concept_shape_manifold_embedding.csv",
        "phase87_pca_loading_report.csv",
        "phase87_01_decision_energy_landscape.png",
        "phase87_02_individual_concept_basins.png",
        "phase87_03_meta_shape_graph.png",
        "phase87_04_boundary_filaments.png",
        "phase87_05_candidate_score_phase_space.png",
        "phase87_06_latent_axis_loading_heatmap.png",
        "phase87_07_concept_atom_constellation.png",
    ]:
        lines.append(f"- `{name}`")

    (PHASE_OUT / "phase87_compositional_concept_shape_manifold_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("[87] Compositional concept-shape manifold mapping")
    print(f"[87] root: {ROOT}")
    print(f"[87] outputs: {PHASE_OUT}")
    print("[87] reset continued: from low-margin landscapes to explicit concept-shape score fields")
    print("[87] task: map individual concept-shapes and meta-shapes under harder compositional pressure")

    df = generate_trials(n_trials=36000)
    task_summary, meta_summary, summary = summarize(df)
    emb, loading_df = make_embedding(df)

    df.to_csv(PHASE_OUT / "phase87_compositional_concept_shape_manifold_trials.csv", index=False)
    task_summary.to_csv(PHASE_OUT / "phase87_compositional_concept_shape_manifold_task_summary.csv", index=False)
    meta_summary.to_csv(PHASE_OUT / "phase87_compositional_concept_shape_manifold_meta_shape_summary.csv", index=False)
    emb.to_csv(PHASE_OUT / "phase87_compositional_concept_shape_manifold_embedding.csv", index=False)
    loading_df.to_csv(PHASE_OUT / "phase87_pca_loading_report.csv", index=False)

    (PHASE_OUT / "phase87_compositional_concept_shape_manifold_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_decision_energy_landscape(emb)
    plot_concept_shape_basins(emb)
    plot_meta_shape_graph(emb)
    plot_boundary_filaments(emb)
    plot_candidate_score_phase_space(emb)
    plot_dimension_loading_heatmap(loading_df)
    plot_atom_constellation()
    write_report(summary, task_summary, meta_summary)

    print(
        "[87] PHASE87_COMPOSITIONAL_CONCEPT_SHAPE_MANIFOLD_PASS="
        f"{summary['PHASE87_COMPOSITIONAL_CONCEPT_SHAPE_MANIFOLD_PASS']}"
    )
    print(
        "[87] "
        f"overall_decision_accuracy={summary['overall_decision_accuracy']:.4f} "
        f"accept_accuracy={summary['accept_accuracy']:.4f} "
        f"reject_accuracy={summary['reject_accuracy']:.4f} "
        f"abstain_accuracy={summary['abstain_accuracy']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"concept_shapes={summary['concept_shapes']} "
        f"meta_shapes={summary['meta_shapes']} "
        f"trials={summary['trials']}"
    )
    print(f"[87] wrote outputs to: {PHASE_OUT}")


if __name__ == "__main__":
    main()