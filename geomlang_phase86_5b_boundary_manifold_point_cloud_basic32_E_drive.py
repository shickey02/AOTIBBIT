#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 86.5b: Boundary manifold point-cloud visualization

Purpose
-------
This is a correction to the earlier Phase 86.5 visualization pass.

The previous figures showed the decision system as clean state logic:
    accept / reject / abstain

This script tries to show the thing underneath that:
    the border shape that makes the boolean decision work.

It reads actual Phase 83-86 trial CSV outputs and constructs a latent point cloud from
available trial-level structure:
    - task / family / variant / pair / decision type
    - expected class / selected class / gold class columns if present
    - numeric scores / margins / traces / localization values if present
    - one-hot encoded categorical structure

It then creates:
    1. latent manifold point cloud across phases
    2. low-margin boundary skin
    3. phase-by-phase semantic manifold drift
    4. margin landscape / contour map
    5. boundary pressure by task and boundary type
    6. nearest-border examples, if text columns exist

Important honesty
-----------------
This does NOT inspect a neural net, unless the phase CSVs contain learned activations
or model features. If the underlying phase scripts are not training a neural network,
there are no hidden neural weights to visualize.

What this visualizes is the induced semantic geometry of the system:
    feature structure -> score space -> margin surface -> decision boundary

Run
---
python bbit_geomlang/geomlang_phase86_5b_boundary_manifold_point_cloud_basic32_E_drive.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
VIZ = OUT / "phase86_5_boundary_manifold"
VIZ.mkdir(parents=True, exist_ok=True)

TRIAL_FILES = {
    83: OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_trials.csv",
    84: OUT / "phase84_compound_semantic_boundary_interference_trials.csv",
    85: OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_trials.csv",
    86: OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_trials.csv",
}

TASK_FILES = {
    83: OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_task_summary.csv",
    84: OUT / "phase84_compound_semantic_boundary_interference_task_summary.csv",
    85: OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_task_summary.csv",
    86: OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_task_summary.csv",
}

SUMMARY_FILES = {
    83: OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_summary.json",
    84: OUT / "phase84_compound_semantic_boundary_interference_summary.json",
    85: OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_summary.json",
    86: OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_summary.json",
}


# ------------------------------------------------------------
# Visual design: less dashboard, more field / manifold
# ------------------------------------------------------------

BG = "#080a0f"
PANEL = "#10141f"
TEXT = "#f3f5fb"
MUTED = "#aab2c5"
GRID = "#2b3245"

PHASE_COLORS = {
    83: "#6aa6ff",
    84: "#5de0e6",
    85: "#b388ff",
    86: "#ffd166",
}

CLASS_COLORS = {
    "accept": "#75d982",
    "true": "#75d982",
    "true_preserved": "#75d982",
    "true_minimal_pair": "#75d982",
    "clean_compound": "#75d982",
    "true_paraphrase": "#75d982",

    "reject": "#ff6b6b",
    "changed": "#ff6b6b",
    "changed_boundary": "#ff6b6b",
    "implicit_boundary_pair": "#ff6b6b",
    "hidden_mutation_compound": "#ff6b6b",
    "near_paraphrase": "#ff6b6b",

    "abstain": "#ffd166",
    "underdetermined": "#ffd166",
    "underspecified": "#ffd166",
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[86.5b] missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[86.5b] failed reading {path}: {e}")
        return None


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def savefig(name: str) -> None:
    path = VIZ / name
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()
    print(f"[86.5b] wrote: {path}")


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def clean_label(x: Any) -> str:
    s = str(x)
    prefixes = [
        "bound_",
        "phase",
        "implicit_",
        "hidden_",
        "clean_",
        "true_",
        "changed_",
        "underspecified_",
        "semantic_boundary_",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
    return s.replace("_", " ")


def infer_decision_label(row: pd.Series) -> str:
    """
    Try to infer a human-readable semantic class from whatever columns exist.
    """
    candidate_cols = [
        "decision_type",
        "case_type",
        "pair_type",
        "variant_type",
        "type",
        "expected",
        "gold_decision",
        "gold",
        "selected_decision",
        "selected",
    ]

    vals = []
    for c in candidate_cols:
        if c in row.index and pd.notna(row[c]):
            vals.append(str(row[c]).lower())

    joined = " ".join(vals)

    if "abstain" in joined or "under" in joined or "insufficient" in joined:
        return "abstain"
    if "reject" in joined or "changed" in joined or "hidden" in joined or "implicit_boundary" in joined or "near" in joined:
        return "reject"
    if "accept" in joined or "true" in joined or "clean" in joined or "preserved" in joined:
        return "accept"

    return "unknown"


def infer_boundary_kind(row: pd.Series) -> str:
    candidate_cols = [
        "boundary_kind",
        "hidden_mutation",
        "minimal_delta",
        "insufficiency_kind",
        "mutation_kind",
        "boundary_type",
    ]
    for c in candidate_cols:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return "unknown"


def infer_task(row: pd.Series) -> str:
    candidate_cols = ["task_id", "boundary_task", "boundary task", "task", "selected_task"]
    for c in candidate_cols:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return "unknown_task"


def infer_family(row: pd.Series) -> str:
    for c in ["family", "task_family"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return "unknown_family"


def pick_margin_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "margin",
        "selected_margin",
        "solution_margin",
        "decision_margin",
        "score_margin",
        "mean_margin",
    ]
    return first_col(df, candidates)


def standardize_numeric_matrix(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1.0
    X = np.where(np.isfinite(X), X, means)
    return (X - means) / stds


def pca_svd(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA through SVD. No sklearn required.
    """
    Xs = standardize_numeric_matrix(X)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    explained = (S ** 2) / max(1e-12, np.sum(S ** 2))
    return coords, explained[:n_components], Vt[:n_components, :]


def safe_sample(df: pd.DataFrame, n: int, seed: int = 865) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


# ------------------------------------------------------------
# Load and compile trials
# ------------------------------------------------------------

def load_trials() -> pd.DataFrame:
    frames = []

    for phase, path in TRIAL_FILES.items():
        df = read_csv(path)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["phase"] = phase
        df["semantic_decision_label"] = df.apply(infer_decision_label, axis=1)
        df["semantic_boundary_kind"] = df.apply(infer_boundary_kind, axis=1)
        df["semantic_task"] = df.apply(infer_task, axis=1)
        df["semantic_family"] = df.apply(infer_family, axis=1)

        mcol = pick_margin_col(df)
        if mcol is not None:
            df["semantic_margin"] = pd.to_numeric(df[mcol], errors="coerce")
        else:
            df["semantic_margin"] = np.nan

        frames.append(df)

    if not frames:
        raise RuntimeError(f"No trial CSVs found under {OUT}")

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    # If no margin was discovered, synthesize neutral value so the script still works.
    if all_df["semantic_margin"].isna().all():
        all_df["semantic_margin"] = 1.0
        print("[86.5b] WARNING: no margin column found; using neutral synthetic margin=1.0")

    all_df["semantic_margin"] = all_df["semantic_margin"].fillna(all_df["semantic_margin"].median())

    compiled = VIZ / "phase86_5b_compiled_trials.csv"
    all_df.to_csv(compiled, index=False)
    print(f"[86.5b] wrote compiled trials: {compiled}")
    return all_df


# ------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, max_categories_per_col: int = 40) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build a feature matrix from real trial structure.

    Numeric columns:
        use actual numeric values except obvious boolean accuracy result columns
        and phase/margin which are handled separately.

    Categorical columns:
        one-hot encode compact structural categories.

    This intentionally avoids text-body vectorization unless text columns are short enough,
    because the goal is semantic structure, not pure wording frequency.
    """

    exclude_exact = {
        "phase",
        "semantic_margin",
    }

    exclude_substrings = [
        "accuracy",
        "correct",
        "pass",
        "valid",
        "trace_validity",
        "no_hallucination",
    ]

    numeric_cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(s in c.lower() for s in exclude_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Avoid ID-like huge unique columns.
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= max(5000, len(df) * 0.90):
                numeric_cols.append(c)

    cat_priority = [
        "phase",
        "semantic_decision_label",
        "semantic_boundary_kind",
        "semantic_task",
        "semantic_family",
        "variant",
        "variant_type",
        "pair",
        "pair_type",
        "case_type",
        "expected",
        "selected",
        "gold",
        "boundary_kind",
        "hidden_mutation",
        "minimal_delta",
        "insufficiency_kind",
    ]

    cat_cols = []
    for c in cat_priority:
        if c in df.columns and c not in cat_cols:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= max_categories_per_col:
                cat_cols.append(c)

    parts = []
    names = []

    if numeric_cols:
        Xn = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        Xn = Xn.fillna(Xn.median(numeric_only=True)).fillna(0.0)
        parts.append(Xn.to_numpy(dtype=float))
        names.extend(numeric_cols)

    if cat_cols:
        Xc = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
        parts.append(Xc.to_numpy(dtype=float))
        names.extend(Xc.columns.tolist())

    if not parts:
        raise RuntimeError("No usable numeric or categorical features found in trial CSVs.")

    X = np.concatenate(parts, axis=1)

    inventory = pd.DataFrame({
        "feature_name": names,
        "source": [
            "numeric" if n in numeric_cols else "categorical_onehot"
            for n in names
        ],
    })
    inventory.to_csv(VIZ / "phase86_5b_feature_inventory.csv", index=False)

    column_inventory = pd.DataFrame({
        "column": list(df.columns),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "nunique": [df[c].nunique(dropna=True) for c in df.columns],
        "used_numeric": [c in numeric_cols for c in df.columns],
        "used_categorical": [c in cat_cols for c in df.columns],
    })
    column_inventory.to_csv(VIZ / "phase86_5b_column_inventory.csv", index=False)

    print(f"[86.5b] feature matrix: rows={X.shape[0]:,}, cols={X.shape[1]:,}")
    print(f"[86.5b] numeric cols used: {numeric_cols}")
    print(f"[86.5b] categorical cols used: {cat_cols}")

    return X, names, inventory


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------

def compute_embedding(df: pd.DataFrame, max_points: int = 35000) -> pd.DataFrame:
    sampled = safe_sample(df, max_points)
    X, feature_names, inventory = build_feature_matrix(sampled)
    coords, explained, loadings = pca_svd(X, n_components=3)

    emb = sampled.copy()
    emb["latent_x"] = coords[:, 0]
    emb["latent_y"] = coords[:, 1]
    emb["latent_z"] = coords[:, 2] if coords.shape[1] > 2 else 0.0
    emb["pca1_explained"] = explained[0] if len(explained) > 0 else np.nan
    emb["pca2_explained"] = explained[1] if len(explained) > 1 else np.nan
    emb["pca3_explained"] = explained[2] if len(explained) > 2 else np.nan

    emb.to_csv(VIZ / "phase86_5b_latent_embedding.csv", index=False)

    # Save loading report.
    loading_rows = []
    for component_idx in range(min(loadings.shape[0], 3)):
        order = np.argsort(np.abs(loadings[component_idx]))[::-1][:30]
        for rank, j in enumerate(order, start=1):
            loading_rows.append({
                "component": component_idx + 1,
                "rank": rank,
                "feature": feature_names[j],
                "loading": loadings[component_idx, j],
                "abs_loading": abs(loadings[component_idx, j]),
            })

    pd.DataFrame(loading_rows).to_csv(VIZ / "phase86_5b_pca_loading_report.csv", index=False)
    print(f"[86.5b] PCA explained: {explained}")
    return emb


# ------------------------------------------------------------
# Plot 1: raw point cloud by semantic decision
# ------------------------------------------------------------

def plot_latent_cloud_by_decision(emb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))

    for label, group in emb.groupby("semantic_decision_label"):
        color = CLASS_COLORS.get(label, "#c7d0e8")
        alpha = 0.34 if label != "unknown" else 0.18
        ax.scatter(
            group["latent_x"],
            group["latent_y"],
            s=8,
            alpha=alpha,
            color=color,
            label=f"{label} ({len(group):,})",
            linewidths=0,
        )

    ax.set_title("Latent semantic point cloud: decision regions emerge from trial structure")
    ax.set_xlabel("latent semantic axis 1")
    ax.set_ylabel("latent semantic axis 2")
    ax.grid(color=GRID, alpha=0.25)
    ax.legend(loc="best", markerscale=2, framealpha=0.18)

    subtitle = (
        "This is not an imposed accept/reject/abstain triangle. "
        "It is a PCA projection of real trial-level feature structure."
    )
    ax.text(0.5, -0.09, subtitle, transform=ax.transAxes, ha="center", va="top", color=MUTED)

    savefig("phase86_5b_01_latent_semantic_point_cloud.png")


# ------------------------------------------------------------
# Plot 2: low-margin boundary skin
# ------------------------------------------------------------

def plot_boundary_skin(emb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))

    margin = emb["semantic_margin"].to_numpy()
    q15 = np.quantile(margin, 0.15)
    q35 = np.quantile(margin, 0.35)

    high = emb[emb["semantic_margin"] > q35]
    skin = emb[emb["semantic_margin"] <= q15]

    ax.scatter(
        high["latent_x"],
        high["latent_y"],
        s=5,
        color="#d8deef",
        alpha=0.08,
        linewidths=0,
        label="interior high-margin points",
    )

    sc = ax.scatter(
        skin["latent_x"],
        skin["latent_y"],
        s=18,
        c=skin["semantic_margin"],
        alpha=0.85,
        linewidths=0,
        label=f"boundary skin: lowest 15% margin ≤ {q15:.3f}",
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("semantic margin")

    ax.set_title("The border shape: low-margin skin of the semantic manifold")
    ax.set_xlabel("latent semantic axis 1")
    ax.set_ylabel("latent semantic axis 2")
    ax.grid(color=GRID, alpha=0.25)
    ax.legend(loc="best", framealpha=0.18)

    ax.text(
        0.5,
        -0.09,
        "Low-margin points are the places where the system is closest to ambiguity: this is the border surface, not the boolean output.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )

    savefig("phase86_5b_02_low_margin_boundary_skin.png")


# ------------------------------------------------------------
# Plot 3: phase drift / manifold accretion
# ------------------------------------------------------------

def plot_phase_drift(emb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))

    centers = []

    for phase, group in emb.groupby("phase"):
        color = PHASE_COLORS.get(int(phase), "#ffffff")
        ax.scatter(
            group["latent_x"],
            group["latent_y"],
            s=7,
            alpha=0.20,
            color=color,
            linewidths=0,
            label=f"Phase {int(phase)}",
        )
        cx = group["latent_x"].mean()
        cy = group["latent_y"].mean()
        centers.append((int(phase), cx, cy, color))
        ax.scatter([cx], [cy], s=220, color=color, edgecolor=TEXT, linewidth=1.2, zorder=5)
        ax.text(cx, cy, str(int(phase)), ha="center", va="center", color=BG, fontweight="bold", zorder=6)

    centers = sorted(centers, key=lambda x: x[0])
    for (p1, x1, y1, c1), (p2, x2, y2, c2) in zip(centers[:-1], centers[1:]):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=2.2, alpha=0.9),
        )

    ax.set_title("Manifold accretion: each phase adds semantic pressure to the same field")
    ax.set_xlabel("latent semantic axis 1")
    ax.set_ylabel("latent semantic axis 2")
    ax.grid(color=GRID, alpha=0.25)
    ax.legend(loc="best", framealpha=0.18)

    savefig("phase86_5b_03_phase_manifold_drift.png")


# ------------------------------------------------------------
# Plot 4: margin landscape contour
# ------------------------------------------------------------

def plot_margin_landscape(emb: pd.DataFrame) -> None:
    df = emb.dropna(subset=["latent_x", "latent_y", "semantic_margin"]).copy()
    if len(df) > 25000:
        df = df.sample(25000, random_state=865)

    x = df["latent_x"].to_numpy()
    y = df["latent_y"].to_numpy()
    z = df["semantic_margin"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 10))

    try:
        tri = mtri.Triangulation(x, y)
        levels = np.linspace(np.quantile(z, 0.02), np.quantile(z, 0.98), 14)
        contour = ax.tricontourf(tri, z, levels=levels, alpha=0.85)
        lines = ax.tricontour(tri, z, levels=levels, colors="#f3f5fb", linewidths=0.35, alpha=0.32)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("semantic margin")
    except Exception as e:
        print(f"[86.5b] triangulation failed, falling back to scatter: {e}")
        sc = ax.scatter(x, y, c=z, s=8, alpha=0.8, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("semantic margin")

    # Mark lowest-margin points as border sparks.
    q10 = np.quantile(z, 0.10)
    low = df[df["semantic_margin"] <= q10]
    ax.scatter(
        low["latent_x"],
        low["latent_y"],
        s=9,
        color="#ff6b6b",
        alpha=0.65,
        linewidths=0,
        label="lowest 10% margin",
    )

    ax.set_title("Semantic margin landscape: the field that hardens into decisions")
    ax.set_xlabel("latent semantic axis 1")
    ax.set_ylabel("latent semantic axis 2")
    ax.grid(color=GRID, alpha=0.18)
    ax.legend(loc="best", framealpha=0.18)

    savefig("phase86_5b_04_margin_landscape_contour.png")


# ------------------------------------------------------------
# Plot 5: boundary pressure by kind
# ------------------------------------------------------------

def plot_boundary_pressure_by_kind(emb: pd.DataFrame) -> None:
    df = emb.copy()
    grouped = (
        df.groupby("semantic_boundary_kind")
        .agg(
            n=("semantic_margin", "size"),
            mean_margin=("semantic_margin", "mean"),
            floor_margin=("semantic_margin", "min"),
            q10_margin=("semantic_margin", lambda x: float(np.quantile(x, 0.10))),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n"] >= max(10, len(df) * 0.002)].copy()
    grouped = grouped.sort_values("q10_margin", ascending=True).head(18)

    fig, ax = plt.subplots(figsize=(13, 8))

    y = np.arange(len(grouped))
    ax.barh(y, grouped["mean_margin"], color="#5de0e6", alpha=0.55, label="mean margin")
    ax.scatter(grouped["q10_margin"], y, s=80, color="#ffd166", label="10th percentile")
    ax.scatter(grouped["floor_margin"], y, s=80, color="#ff6b6b", label="floor")

    for i, row in grouped.iterrows():
        idx = list(grouped.index).index(i)
        ax.text(row["mean_margin"] + 0.015, idx, f"μ {row['mean_margin']:.3f}", va="center", fontsize=9)
        ax.text(row["floor_margin"] - 0.015, idx, f"{row['floor_margin']:.3f}", va="center", ha="right", fontsize=8, color="#ff6b6b")

    ax.axvline(1.0, color="#ff6b6b", ls="--", lw=1.2, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(x) for x in grouped["semantic_boundary_kind"]])
    ax.set_xlabel("margin")
    ax.set_title("Boundary pressure by semantic kind: the hardest edges of meaning")
    ax.grid(axis="x", color=GRID, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.18)

    savefig("phase86_5b_05_boundary_pressure_by_kind.png")


# ------------------------------------------------------------
# Plot 6: task pressure map
# ------------------------------------------------------------

def plot_task_pressure_map(emb: pd.DataFrame) -> None:
    grouped = (
        emb.groupby(["semantic_family", "semantic_task"])
        .agg(
            n=("semantic_margin", "size"),
            mean_margin=("semantic_margin", "mean"),
            floor_margin=("semantic_margin", "min"),
            q10_margin=("semantic_margin", lambda x: float(np.quantile(x, 0.10))),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n"] >= max(10, len(emb) * 0.002)].copy()
    grouped = grouped.sort_values("q10_margin", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 9))

    x = grouped["mean_margin"].to_numpy()
    y = grouped["q10_margin"].to_numpy()
    sizes = 80 + 260 * (grouped["n"].to_numpy() / grouped["n"].max())

    families = sorted(grouped["semantic_family"].unique())
    family_colors = {
        fam: ["#6aa6ff", "#75d982", "#ffd166", "#b388ff", "#ff8bd1", "#5de0e6"][i % 6]
        for i, fam in enumerate(families)
    }

    for fam, sub in grouped.groupby("semantic_family"):
        ax.scatter(
            sub["mean_margin"],
            sub["q10_margin"],
            s=80 + 260 * (sub["n"] / grouped["n"].max()),
            alpha=0.75,
            color=family_colors[fam],
            edgecolor=TEXT,
            linewidth=0.6,
            label=fam,
        )

        for _, row in sub.iterrows():
            ax.text(
                row["mean_margin"] + 0.004,
                row["q10_margin"] + 0.004,
                clean_label(row["semantic_task"]),
                fontsize=8,
                color=MUTED,
            )

    ax.axhline(1.0, color="#ff6b6b", ls="--", lw=1.2, alpha=0.65)
    ax.axvline(1.0, color="#ff6b6b", ls="--", lw=1.2, alpha=0.35)

    ax.set_xlabel("mean margin")
    ax.set_ylabel("10th percentile margin")
    ax.set_title("Task pressure map: where the manifold is thick vs. where the border thins")
    ax.grid(color=GRID, alpha=0.25)
    ax.legend(loc="best", framealpha=0.18)

    savefig("phase86_5b_06_task_pressure_map.png")


# ------------------------------------------------------------
# Plot 7: local boundary filaments
# ------------------------------------------------------------

def plot_boundary_filaments(emb: pd.DataFrame) -> None:
    """
    Make something more organic: connect nearby low-margin points into filaments.
    This is not a real topology extraction algorithm; it is a visual approximation of
    the low-margin border skin as a living edge.
    """
    df = emb.copy()
    q = np.quantile(df["semantic_margin"], 0.12)
    low = df[df["semantic_margin"] <= q].copy()

    if len(low) > 3500:
        low = low.sample(3500, random_state=865)

    xy = low[["latent_x", "latent_y"]].to_numpy()
    margin = low["semantic_margin"].to_numpy()

    # Create lightweight nearest-neighbor filament segments.
    # O(n^2) would be bad, so do a random local subset search.
    rng = np.random.default_rng(865)
    segments = []
    colors = []

    if len(xy) > 2:
        for i in range(len(xy)):
            # Search among a random candidate subset plus nearby index window.
            cand_size = min(220, len(xy))
            candidates = rng.choice(len(xy), size=cand_size, replace=False)
            candidates = candidates[candidates != i]
            if len(candidates) == 0:
                continue

            d = np.sum((xy[candidates] - xy[i]) ** 2, axis=1)
            j = candidates[int(np.argmin(d))]
            dist = math.sqrt(float(np.min(d)))

            # Avoid long spurious cross-manifold links.
            if dist < np.quantile(np.sqrt(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)), 0.18):
                segments.append([xy[i], xy[j]])
                colors.append((margin[i] + margin[j]) / 2.0)

    fig, ax = plt.subplots(figsize=(13, 10))

    ax.scatter(
        emb["latent_x"],
        emb["latent_y"],
        s=4,
        color="#d8deef",
        alpha=0.045,
        linewidths=0,
    )

    if segments:
        lc = LineCollection(segments, cmap="inferno", linewidths=0.65, alpha=0.40)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("low-margin filament value")

    ax.scatter(
        low["latent_x"],
        low["latent_y"],
        c=low["semantic_margin"],
        s=10,
        cmap="inferno",
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title("Boundary filaments: the living edge where semantic categories separate")
    ax.set_xlabel("latent semantic axis 1")
    ax.set_ylabel("latent semantic axis 2")
    ax.grid(color=GRID, alpha=0.18)

    ax.text(
        0.5,
        -0.09,
        "This is the closest visual analogue to the BBIT border: a low-margin skin traced through the trial manifold.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )

    savefig("phase86_5b_07_boundary_filaments.png")


# ------------------------------------------------------------
# Plot 8: nearest-border examples
# ------------------------------------------------------------

def write_nearest_border_examples(emb: pd.DataFrame, n: int = 80) -> None:
    low = emb.sort_values("semantic_margin", ascending=True).head(n).copy()

    text_cols = []
    for c in emb.columns:
        cl = c.lower()
        if any(k in cl for k in ["prompt", "problem", "text", "sentence", "query", "question", "paraphrase"]):
            if emb[c].dtype == object:
                text_cols.append(c)

    keep = [
        "phase",
        "semantic_margin",
        "semantic_decision_label",
        "semantic_boundary_kind",
        "semantic_family",
        "semantic_task",
    ]

    keep.extend([c for c in text_cols if c not in keep])
    keep = [c for c in keep if c in low.columns]

    out = low[keep].copy()
    out.to_csv(VIZ / "phase86_5b_nearest_border_examples.csv", index=False)

    md = []
    md.append("# Phase 86.5b nearest-border examples\n")
    md.append("These are the lowest-margin trial rows found across Phase 83-86.\n")
    md.append("They approximate the semantic border surface: where the system is closest to ambiguity.\n")

    for idx, row in out.head(30).iterrows():
        md.append(f"## Example {len(md)}\n")
        md.append(f"- phase: `{row.get('phase', '')}`")
        md.append(f"- margin: `{row.get('semantic_margin', '')}`")
        md.append(f"- decision: `{row.get('semantic_decision_label', '')}`")
        md.append(f"- boundary kind: `{row.get('semantic_boundary_kind', '')}`")
        md.append(f"- task: `{row.get('semantic_task', '')}`")
        for c in text_cols[:4]:
            if c in row.index and pd.notna(row[c]):
                md.append(f"- {c}: {row[c]}")
        md.append("")

    (VIZ / "phase86_5b_nearest_border_examples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[86.5b] wrote nearest-border examples")


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

def write_manifest(emb: pd.DataFrame) -> None:
    summaries = {}
    for phase, path in SUMMARY_FILES.items():
        s = read_json(path)
        if s:
            summaries[str(phase)] = s

    manifest = {
        "phase": 86.5,
        "variant": "boundary_manifold_point_cloud",
        "purpose": "Visualize the semantic border shape beneath accept/reject/abstain decisions.",
        "input_trials": {str(k): str(v) for k, v in TRIAL_FILES.items()},
        "output_dir": str(VIZ),
        "rows_embedded": int(len(emb)),
        "phase_counts": {str(k): int(v) for k, v in emb["phase"].value_counts().sort_index().items()},
        "decision_counts": {str(k): int(v) for k, v in emb["semantic_decision_label"].value_counts().items()},
        "margin_summary": {
            "mean": float(emb["semantic_margin"].mean()),
            "floor": float(emb["semantic_margin"].min()),
            "q10": float(np.quantile(emb["semantic_margin"], 0.10)),
            "median": float(emb["semantic_margin"].median()),
            "q90": float(np.quantile(emb["semantic_margin"], 0.90)),
        },
        "summary_files_loaded": summaries,
        "outputs": [
            "phase86_5b_01_latent_semantic_point_cloud.png",
            "phase86_5b_02_low_margin_boundary_skin.png",
            "phase86_5b_03_phase_manifold_drift.png",
            "phase86_5b_04_margin_landscape_contour.png",
            "phase86_5b_05_boundary_pressure_by_kind.png",
            "phase86_5b_06_task_pressure_map.png",
            "phase86_5b_07_boundary_filaments.png",
            "phase86_5b_latent_embedding.csv",
            "phase86_5b_pca_loading_report.csv",
            "phase86_5b_feature_inventory.csv",
            "phase86_5b_column_inventory.csv",
            "phase86_5b_nearest_border_examples.csv",
            "phase86_5b_nearest_border_examples.md",
        ],
    }

    path = VIZ / "phase86_5b_boundary_manifold_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[86.5b] wrote: {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    print("[86.5b] Boundary manifold point-cloud visualization")
    print(f"[86.5b] reading from: {OUT}")
    print(f"[86.5b] writing to: {VIZ}")

    trials = load_trials()
    emb = compute_embedding(trials, max_points=35000)

    plot_latent_cloud_by_decision(emb)
    plot_boundary_skin(emb)
    plot_phase_drift(emb)
    plot_margin_landscape(emb)
    plot_boundary_pressure_by_kind(emb)
    plot_task_pressure_map(emb)
    plot_boundary_filaments(emb)
    write_nearest_border_examples(emb, n=100)
    write_manifest(emb)

    print("[86.5b] complete")
    print(f"[86.5b] output group: {VIZ}")


if __name__ == "__main__":
    main()