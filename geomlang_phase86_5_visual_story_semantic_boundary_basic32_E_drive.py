#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 86.5: Visual story of BBIT semantic-boundary emergence

Purpose:
    Create compelling matplotlib visualizations from the actual Phase 83-86 research outputs.

This script does NOT regenerate research data.
It reads the existing JSON/CSV artifacts in:

    E:\\BBIT\\outputs_basic32

and writes visual outputs to:

    E:\\BBIT\\outputs_basic32\\phase86_5_visual_story

Visual questions:
    1. What capability ladder did Phases 83-86 build?
    2. How did the semantic decision space expand from binary to ternary?
    3. How did margins behave as the task became more conceptually difficult?
    4. Which tasks carried the hardest boundary pressure?
    5. What does abstention look like as a structured semantic capacity?
    6. Can we visualize the architecture of accept / reject / abstain as a state-space?

Run:
    python bbit_geomlang/geomlang_phase86_5_visual_story_semantic_boundary_basic32_E_drive.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(r"E:\BBIT")
OUT = ROOT / "outputs_basic32"
VIZ = OUT / "phase86_5_visual_story"
VIZ.mkdir(parents=True, exist_ok=True)

PHASE_FILES = {
    83: {
        "summary": OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_summary.json",
        "tasks": OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_task_summary.csv",
        "trials": OUT / "phase83_semantic_boundary_near_paraphrase_rejection_bridge_trials.csv",
    },
    84: {
        "summary": OUT / "phase84_compound_semantic_boundary_interference_summary.json",
        "tasks": OUT / "phase84_compound_semantic_boundary_interference_task_summary.csv",
        "trials": OUT / "phase84_compound_semantic_boundary_interference_trials.csv",
    },
    85: {
        "summary": OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_summary.json",
        "tasks": OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_task_summary.csv",
        "trials": OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_trials.csv",
        "pairs": OUT / "phase85_implicit_adversarial_semantic_boundary_minimal_pair_pair_summary.csv",
    },
    86: {
        "summary": OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_summary.json",
        "tasks": OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_task_summary.csv",
        "trials": OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_trials.csv",
        "decisions": OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_decision_summary.csv",
        "insuff": OUT / "phase86_semantic_boundary_abstention_underspecified_minimal_pairs_insufficiency_summary.csv",
    },
}


# ------------------------------------------------------------
# Design
# ------------------------------------------------------------

BG = "#0f1117"
PANEL = "#171b26"
PANEL_2 = "#1e2433"
TEXT = "#f2f4f8"
MUTED = "#aeb6c8"
GRID = "#32384a"

BLUE = "#6aa6ff"
CYAN = "#62d6e8"
GREEN = "#74d680"
YELLOW = "#ffd166"
ORANGE = "#ff9f43"
RED = "#ff6b6b"
PURPLE = "#b388ff"
PINK = "#ff8bd1"
WHITE = "#ffffff"

PHASE_COLORS = {
    83: BLUE,
    84: CYAN,
    85: PURPLE,
    86: YELLOW,
}

CASE_COLORS = {
    "true_preserved": GREEN,
    "changed_boundary": RED,
    "underspecified": YELLOW,
    "true_minimal_pair": GREEN,
    "implicit_boundary_pair": RED,
    "clean_compound": GREEN,
    "hidden_mutation_compound": RED,
    "true_paraphrase": GREEN,
    "near_paraphrase": RED,
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
# IO helpers
# ------------------------------------------------------------

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[86.5] missing JSON: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[86.5] missing CSV: {path}")
        return None
    return pd.read_csv(path)


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_task_name(name: str) -> str:
    name = str(name)
    if name.startswith("bound_"):
        name = name[len("bound_"):]
    return name.replace("_", "\n")


def clean_short(name: str) -> str:
    name = str(name)
    prefixes = [
        "bound_",
        "implicit_",
        "changed_",
        "underspecified_",
        "true_",
        "hidden_",
        "clean_",
    ]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
    return name.replace("_", " ")


def metric(summary: Dict[str, Any], *keys: str, default: float = np.nan) -> float:
    for k in keys:
        if k in summary:
            return float(summary[k])
    return default


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[86.5] wrote: {path}")


# ------------------------------------------------------------
# Load all available data
# ------------------------------------------------------------

def load_data() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], Dict[str, pd.DataFrame]]:
    summaries: Dict[int, Dict[str, Any]] = {}
    tasks: Dict[int, pd.DataFrame] = {}
    trials: Dict[int, pd.DataFrame] = {}
    extras: Dict[str, pd.DataFrame] = {}

    for phase, files in PHASE_FILES.items():
        s = read_json(files["summary"])
        if s is not None:
            summaries[phase] = s

        t = read_csv(files["tasks"])
        if t is not None:
            tasks[phase] = t

        tr = read_csv(files["trials"])
        if tr is not None:
            trials[phase] = tr

        for key in ["pairs", "decisions", "insuff"]:
            if key in files:
                df = read_csv(files[key])
                if df is not None:
                    extras[f"{phase}_{key}"] = df

    return summaries, tasks, trials, extras


# ------------------------------------------------------------
# Derived summary table
# ------------------------------------------------------------

def build_phase_frame(summaries: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in sorted(summaries):
        s = summaries[p]

        rows.append({
            "phase": p,
            "title": str(s.get("title", f"Phase {p}")),
            "trials": int(s.get("trials", 0)),
            "mean_margin": metric(s, "mean_margin"),
            "margin_floor": metric(s, "margin_floor"),
            "overall": metric(s, "overall_decision_accuracy", "overall_solve_accuracy"),
            "true_acceptance": metric(s, "true_acceptance", "true_minimal_pair_acceptance", "clean_compound_acceptance", "true_paraphrase_acceptance"),
            "reject_crossing": metric(s, "changed_rejection", "implicit_boundary_rejection", "hidden_mutation_rejection", "near_paraphrase_rejection"),
            "boundary_accuracy": metric(s, "implicit_boundary_accuracy", "compound_boundary_accuracy", "semantic_boundary_accuracy", "overall_decision_accuracy"),
            "schema": metric(s, "schema_selection_accuracy"),
            "binding": metric(s, "variable_binding_accuracy"),
            "localize": metric(s, "boundary_localization_accuracy", "mutation_localization_accuracy"),
            "abstain": metric(s, "underdetermined_abstention", default=0.0),
            "insuff": metric(s, "insufficient_information_detection", default=0.0),
            "reason": metric(s, "abstention_reason_accuracy", default=0.0),
            "trace": metric(s, "trace_validity"),
            "no_hallucination": metric(s, "no_hallucination_accuracy"),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Visualization 1: Capability ladder
# ------------------------------------------------------------

def plot_capability_ladder(phase_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(82.3, 86.7)
    ax.set_ylim(-0.6, 4.8)
    ax.axis("off")

    fig.suptitle("Phase 83 → 86: semantic boundary becomes a three-way reasoning system", y=0.96)

    capabilities = {
        83: {
            "label": "Boundary",
            "subtitle": "accept true paraphrase\nreject near paraphrase",
            "detail": "The system learns that paraphrase invariance has a boundary.",
        },
        84: {
            "label": "Camouflage",
            "subtitle": "detect hidden mutation\nunder harmless layers",
            "detail": "The boundary survives compound linguistic interference.",
        },
        85: {
            "label": "Implicit delta",
            "subtitle": "minimal word/role/unit\nchange crosses meaning",
            "detail": "No explicit mutation label is needed.",
        },
        86: {
            "label": "Abstention",
            "subtitle": "accept / reject /\nabstain when undefined",
            "detail": "The system refuses to hallucinate a determinate object.",
        },
    }

    y = 2.35
    for _, row in phase_df.iterrows():
        p = int(row["phase"])
        x = p
        color = PHASE_COLORS[p]

        circ = Circle((x, y), radius=0.30, facecolor=color, edgecolor=WHITE, lw=1.5, alpha=0.95)
        ax.add_patch(circ)
        ax.text(x, y, str(p), ha="center", va="center", color=BG, fontsize=18, fontweight="bold")

        ax.text(x, y + 0.63, capabilities[p]["label"], ha="center", va="bottom", fontsize=18, fontweight="bold", color=color)
        ax.text(x, y - 0.62, capabilities[p]["subtitle"], ha="center", va="top", fontsize=11, color=TEXT, linespacing=1.3)

        metric_box = (
            f"trials {int(row['trials']):,}\n"
            f"mean margin {row['mean_margin']:.3f}\n"
            f"floor {row['margin_floor']:.3f}"
        )
        ax.text(x, 0.45, metric_box, ha="center", va="center", fontsize=10, color=MUTED)

    for p in [83, 84, 85]:
        arrow = FancyArrowPatch(
            (p + 0.36, y), (p + 1 - 0.36, y),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.2,
            color=GRID,
        )
        ax.add_patch(arrow)

    ax.text(
        84.5,
        4.25,
        "The important movement is not only higher accuracy. It is a growing ontology of decisions.",
        ha="center",
        va="center",
        fontsize=15,
        color=TEXT,
    )

    ax.text(
        84.5,
        -0.15,
        "Binary boundary recognition becomes ternary semantic judgment: same object / changed object / insufficiently specified object.",
        ha="center",
        va="center",
        fontsize=13,
        color=MUTED,
    )

    savefig(VIZ / "phase86_5_01_capability_ladder.png")


# ------------------------------------------------------------
# Visualization 2: capability matrix
# ------------------------------------------------------------

def plot_capability_matrix(phase_df: pd.DataFrame) -> None:
    caps = [
        ("True meaning accepted", "true_acceptance"),
        ("Boundary crossing rejected", "reject_crossing"),
        ("Boundary localized", "localize"),
        ("Variable binding preserved", "binding"),
        ("Trace valid", "trace"),
        ("No hallucination", "no_hallucination"),
        ("Underdetermined abstained", "abstain"),
        ("Insufficiency classified", "insuff"),
        ("Abstention reason identified", "reason"),
    ]

    phases = phase_df["phase"].astype(int).tolist()
    mat = np.array([[float(phase_df.loc[phase_df["phase"] == p, key].iloc[0]) for p in phases] for _, key in caps])

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels([name for name, _ in caps])

    ax.set_title("Semantic capability matrix: what exists at each phase")
    ax.set_xlabel("research phase")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            label = "—" if val == 0 else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", color=BG if val > 0.65 else TEXT, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("measured capability score")

    savefig(VIZ / "phase86_5_02_capability_matrix.png")


# ------------------------------------------------------------
# Visualization 3: margin ecology / violin plot
# ------------------------------------------------------------

def plot_margin_ecology(trials: Dict[int, pd.DataFrame], phase_df: pd.DataFrame) -> None:
    data = []
    labels = []
    colors = []

    for p in sorted(trials):
        df = trials[p]
        if "margin" not in df.columns:
            continue
        vals = pd.to_numeric(df["margin"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"Phase {p}")
        colors.append(PHASE_COLORS.get(p, BLUE))

    fig, ax = plt.subplots(figsize=(14, 7))

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor(WHITE)
        body.set_alpha(0.50)

    for i, vals in enumerate(data, start=1):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        floor = np.min(vals)
        mean_v = np.mean(vals)

        ax.scatter([i], [mean_v], s=80, color=WHITE, zorder=5, label="mean" if i == 1 else None)
        ax.plot([i - 0.18, i + 0.18], [med, med], color=BG, lw=4, zorder=6)
        ax.plot([i, i], [q1, q3], color=WHITE, lw=3, zorder=5)
        ax.scatter([i], [floor], s=45, color=RED, zorder=7, label="floor" if i == 1 else None)

        ax.text(i, floor - 0.045, f"{floor:.3f}", ha="center", va="top", fontsize=9, color=RED)
        ax.text(i, mean_v + 0.055, f"μ {mean_v:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.axhline(1.0, color=RED, lw=1.5, ls="--", alpha=0.7)
    ax.text(0.55, 1.02, "pass floor threshold", color=RED, fontsize=10, va="bottom")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("decision margin")
    ax.set_title("Margin ecology: harder semantics compress the floor, but do not break it")
    ax.grid(axis="y", color=GRID, alpha=0.35)
    ax.legend(loc="upper right")

    savefig(VIZ / "phase86_5_03_margin_ecology.png")


# ------------------------------------------------------------
# Visualization 4: task x phase margin heatmap
# ------------------------------------------------------------

def normalize_task_summary_df(phase: int, df: pd.DataFrame) -> pd.DataFrame:
    task_col = first_existing_col(df, ["task_id", "boundary task", "boundary_task", "task"])
    family_col = first_existing_col(df, ["family"])
    margin_col = first_existing_col(df, ["mean_margin", "margin"])

    if task_col is None or margin_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "phase": phase,
        "task": df[task_col].astype(str),
        "family": df[family_col].astype(str) if family_col else "unknown",
        "mean_margin": pd.to_numeric(df[margin_col], errors="coerce"),
    })

    return out.dropna(subset=["mean_margin"])


def plot_task_margin_heatmap(tasks: Dict[int, pd.DataFrame]) -> None:
    rows = []
    for p, df in tasks.items():
        rows.append(normalize_task_summary_df(p, df))

    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if all_df.empty:
        print("[86.5] no task margin data available for heatmap")
        return

    pivot = all_df.pivot_table(index="task", columns="phase", values="mean_margin", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([clean_task_name(t) for t in pivot.index], fontsize=9)

    ax.set_xlabel("phase")
    ax.set_title("Task margin heatmap: where semantic pressure concentrates")

    vals = pivot.values
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", color=BG, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("mean margin")

    savefig(VIZ / "phase86_5_04_task_margin_heatmap.png")


# ------------------------------------------------------------
# Visualization 5: phase 86 ternary decision state-space
# ------------------------------------------------------------

def barycentric_to_xy(a: float, r: float, u: float) -> Tuple[float, float]:
    """
    Vertices:
        accept  = (0, 0)
        reject  = (1, 0)
        abstain = (0.5, sqrt(3)/2)
    """
    x = a * 0.0 + r * 1.0 + u * 0.5
    y = a * 0.0 + r * 0.0 + u * (math.sqrt(3) / 2.0)
    return x, y


def plot_phase86_decision_space(extras: Dict[str, pd.DataFrame]) -> None:
    df = extras.get("86_decisions")
    if df is None or df.empty:
        print("[86.5] no Phase 86 decision summary available")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    triangle = np.array([
        barycentric_to_xy(1, 0, 0),
        barycentric_to_xy(0, 1, 0),
        barycentric_to_xy(0, 0, 1),
    ])
    poly = Polygon(triangle, closed=True, facecolor=PANEL_2, edgecolor=GRID, lw=2)
    ax.add_patch(poly)

    # Internal grid
    for t in np.linspace(0.2, 0.8, 4):
        p1 = barycentric_to_xy(t, 1 - t, 0)
        p2 = barycentric_to_xy(t, 0, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, lw=0.8, alpha=0.45)

        p1 = barycentric_to_xy(1 - t, t, 0)
        p2 = barycentric_to_xy(0, t, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, lw=0.8, alpha=0.45)

        p1 = barycentric_to_xy(1 - t, 0, t)
        p2 = barycentric_to_xy(0, 1 - t, t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, lw=0.8, alpha=0.45)

    ax.text(-0.05, -0.06, "ACCEPT\nsame object", ha="center", va="top", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(1.05, -0.06, "REJECT\nchanged object", ha="center", va="top", fontsize=14, color=RED, fontweight="bold")
    ax.text(0.5, math.sqrt(3) / 2 + 0.06, "ABSTAIN\nundefined object", ha="center", va="bottom", fontsize=14, color=YELLOW, fontweight="bold")

    rng = np.random.default_rng(865)
    for _, row in df.iterrows():
        case_type = str(row.get("case_type", "unknown"))
        case_id = str(row.get("case_id", "case"))
        margin = float(row.get("mean_margin", row.get("margin", 2.0)))

        if case_type == "true_preserved":
            base = np.array(barycentric_to_xy(0.88, 0.06, 0.06))
            color = GREEN
        elif case_type == "changed_boundary":
            base = np.array(barycentric_to_xy(0.06, 0.88, 0.06))
            color = RED
        elif case_type == "underspecified":
            base = np.array(barycentric_to_xy(0.06, 0.06, 0.88))
            color = YELLOW
        else:
            base = np.array(barycentric_to_xy(0.33, 0.33, 0.34))
            color = PURPLE

        jitter = rng.normal(0, 0.018, size=2)
        xy = base + jitter
        size = 110 + 95 * max(0.0, margin - 1.5)

        ax.scatter([xy[0]], [xy[1]], s=size, color=color, edgecolor=WHITE, lw=1.1, alpha=0.9)
        ax.text(xy[0], xy[1] - 0.035, clean_short(case_id), ha="center", va="top", fontsize=8, color=TEXT)

    ax.set_title("Phase 86 decision state-space: same / changed / underdetermined")
    ax.text(
        0.5,
        -0.18,
        "The third vertex is the conceptual advance: the system can identify when no determinate object has been supplied.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12,
    )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.22, 1.02)

    savefig(VIZ / "phase86_5_05_three_way_decision_state_space.png")


# ------------------------------------------------------------
# Visualization 6: semantic transition graph
# ------------------------------------------------------------

def plot_semantic_transition_graph(phase_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    nodes = {
        "surface wording": (1.2, 3.0, BLUE),
        "schema candidate": (3.0, 4.2, CYAN),
        "binding signature": (3.0, 1.8, PURPLE),
        "semantic boundary": (5.0, 3.0, YELLOW),
        "accept\nsame object": (7.6, 4.6, GREEN),
        "reject\nchanged object": (7.6, 3.0, RED),
        "abstain\nundefined object": (7.6, 1.4, YELLOW),
        "repair target\nPhase 87": (9.2, 1.4, PINK),
    }

    edges = [
        ("surface wording", "schema candidate"),
        ("surface wording", "binding signature"),
        ("schema candidate", "semantic boundary"),
        ("binding signature", "semantic boundary"),
        ("semantic boundary", "accept\nsame object"),
        ("semantic boundary", "reject\nchanged object"),
        ("semantic boundary", "abstain\nundefined object"),
        ("abstain\nundefined object", "repair target\nPhase 87"),
    ]

    for a, b in edges:
        x1, y1, _ = nodes[a]
        x2, y2, _ = nodes[b]
        arrow = FancyArrowPatch(
            (x1 + 0.55, y1), (x2 - 0.55, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            lw=2.0,
            color=GRID,
            alpha=0.95,
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    for label, (x, y, color) in nodes.items():
        box = FancyBboxPatch(
            (x - 0.75, y - 0.35),
            1.5,
            0.70,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            facecolor=PANEL_2,
            edgecolor=color,
            lw=2.2,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=TEXT, fontweight="bold")

    # Phase badges
    badges = [
        ("83", "paraphrase boundary", 4.7, 5.35, BLUE),
        ("84", "compound camouflage", 5.7, 5.0, CYAN),
        ("85", "implicit minimal delta", 6.5, 4.55, PURPLE),
        ("86", "abstention", 7.0, 0.55, YELLOW),
    ]

    for ph, lab, x, y, color in badges:
        ax.add_patch(Circle((x, y), 0.23, facecolor=color, edgecolor=WHITE, lw=1.2))
        ax.text(x, y, ph, ha="center", va="center", color=BG, fontsize=10, fontweight="bold")
        ax.text(x + 0.32, y, lab, ha="left", va="center", color=MUTED, fontsize=10)

    ax.set_title("Emergent structure: from wording to semantic decision")
    ax.text(
        5.0,
        0.25,
        "Phase 86 makes non-answer a structured answer: undefined inputs are routed to abstention instead of forced solution.",
        ha="center",
        va="center",
        fontsize=13,
        color=MUTED,
    )

    savefig(VIZ / "phase86_5_06_semantic_transition_graph.png")


# ------------------------------------------------------------
# Visualization 7: insufficiency fingerprint
# ------------------------------------------------------------

def plot_insufficiency_fingerprint(extras: Dict[str, pd.DataFrame]) -> None:
    df = extras.get("86_insuff")
    if df is None or df.empty:
        print("[86.5] no Phase 86 insufficiency summary available")
        return

    kind_col = first_existing_col(df, ["insufficiency_kind"])
    margin_col = first_existing_col(df, ["mean_margin", "margin"])
    floor_col = first_existing_col(df, ["margin_floor"])
    reason_col = first_existing_col(df, ["abstention_reason_accuracy"])
    detect_col = first_existing_col(df, ["insufficient_information_detection"])

    if kind_col is None or margin_col is None:
        print("[86.5] insufficiency summary lacks needed columns")
        return

    df = df.copy()
    df["label"] = df[kind_col].map(clean_short)
    df["mean_margin"] = pd.to_numeric(df[margin_col], errors="coerce")
    df["margin_floor"] = pd.to_numeric(df[floor_col], errors="coerce") if floor_col else np.nan
    df["reason"] = pd.to_numeric(df[reason_col], errors="coerce") if reason_col else 1.0
    df["detect"] = pd.to_numeric(df[detect_col], errors="coerce") if detect_col else 1.0

    df = df.sort_values("mean_margin", ascending=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(df))

    ax.barh(y, df["mean_margin"], color=YELLOW, alpha=0.85, label="mean margin")
    if df["margin_floor"].notna().any():
        ax.scatter(df["margin_floor"], y, color=RED, s=70, zorder=5, label="margin floor")

    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        ax.text(row["mean_margin"] + 0.015, idx, f"{row['mean_margin']:.3f}", va="center", ha="left", fontsize=10)
        if np.isfinite(row["margin_floor"]):
            ax.text(row["margin_floor"] - 0.015, idx, f"{row['margin_floor']:.3f}", va="center", ha="right", fontsize=9, color=RED)

    ax.axvline(1.0, color=RED, ls="--", lw=1.3, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("margin")
    ax.set_title("Phase 86 insufficiency fingerprint: what kind of missingness is easiest to stabilize?")
    ax.grid(axis="x", color=GRID, alpha=0.35)
    ax.legend(loc="lower right")

    savefig(VIZ / "phase86_5_07_insufficiency_fingerprint.png")


# ------------------------------------------------------------
# Visualization 8: one-page hero dashboard
# ------------------------------------------------------------

def plot_hero_dashboard(phase_df: pd.DataFrame, trials: Dict[int, pd.DataFrame], extras: Dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], width_ratios=[1.1, 1, 1])

    fig.suptitle("BBIT Phase 83–86: semantic boundary reasoning becomes anti-hallucination structure", y=0.97)

    # Panel 1: phase progression
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    phases = phase_df["phase"].astype(int).values
    floors = phase_df["margin_floor"].values
    means = phase_df["mean_margin"].values
    ax1.plot(phases, means, marker="o", lw=2.5, color=CYAN, label="mean margin")
    ax1.plot(phases, floors, marker="o", lw=2.5, color=RED, label="margin floor")
    ax1.axhline(1.0, color=RED, ls="--", alpha=0.5)
    ax1.set_title("Margins compress, structure holds")
    ax1.set_xlabel("phase")
    ax1.set_ylabel("margin")
    ax1.set_xticks(phases)
    ax1.grid(color=GRID, alpha=0.3)
    ax1.legend()

    for x, y in zip(phases, floors):
        ax1.text(x, y - 0.035, f"{y:.3f}", ha="center", va="top", fontsize=9, color=RED)

    # Panel 2: capability bars
    ax2 = fig.add_subplot(gs[0, 1])
    latest = phase_df[phase_df["phase"] == 86].iloc[0]
    labels = ["accept", "reject", "abstain", "insuff", "reason", "no halluc."]
    vals = [
        latest["true_acceptance"],
        latest["reject_crossing"],
        latest["abstain"],
        latest["insuff"],
        latest["reason"],
        latest["no_hallucination"],
    ]
    ax2.barh(np.arange(len(labels)), vals, color=[GREEN, RED, YELLOW, CYAN, PURPLE, BLUE])
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Phase 86 capability closure")
    ax2.grid(axis="x", color=GRID, alpha=0.3)

    for i, v in enumerate(vals):
        ax2.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=10)

    # Panel 3: decision triangle mini
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_aspect("equal")
    ax3.axis("off")
    tri = np.array([
        barycentric_to_xy(1, 0, 0),
        barycentric_to_xy(0, 1, 0),
        barycentric_to_xy(0, 0, 1),
    ])
    ax3.add_patch(Polygon(tri, closed=True, facecolor=PANEL_2, edgecolor=GRID, lw=2))
    ax3.text(0, -0.06, "accept", ha="center", va="top", color=GREEN, fontweight="bold")
    ax3.text(1, -0.06, "reject", ha="center", va="top", color=RED, fontweight="bold")
    ax3.text(0.5, math.sqrt(3) / 2 + 0.05, "abstain", ha="center", va="bottom", color=YELLOW, fontweight="bold")
    ax3.scatter([0.08, 0.92, 0.5], [0.06, 0.06, 0.75], s=[400, 400, 400], color=[GREEN, RED, YELLOW], edgecolor=WHITE, lw=1.2)
    ax3.set_title("Three-way semantic object space")
    ax3.set_xlim(-0.12, 1.12)
    ax3.set_ylim(-0.14, 1.02)

    # Panel 4: margin distribution
    ax4 = fig.add_subplot(gs[1, 0:2])
    for p in sorted(trials):
        df = trials[p]
        if "margin" not in df.columns:
            continue
        vals = pd.to_numeric(df["margin"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        ax4.hist(vals, bins=45, histtype="step", lw=2.2, color=PHASE_COLORS[p], label=f"Phase {p}", density=True, alpha=0.9)

    ax4.axvline(1.0, color=RED, ls="--", alpha=0.7)
    ax4.set_title("Distribution of decision confidence across phases")
    ax4.set_xlabel("margin")
    ax4.set_ylabel("density")
    ax4.grid(color=GRID, alpha=0.3)
    ax4.legend()

    # Panel 5: insufficiency margins
    ax5 = fig.add_subplot(gs[1, 2])
    insuff = extras.get("86_insuff")
    if insuff is not None and not insuff.empty:
        kind_col = first_existing_col(insuff, ["insufficiency_kind"])
        margin_col = first_existing_col(insuff, ["mean_margin", "margin"])
        if kind_col and margin_col:
            temp = insuff.copy()
            temp["label"] = temp[kind_col].map(clean_short)
            temp["mean_margin"] = pd.to_numeric(temp[margin_col], errors="coerce")
            temp = temp.sort_values("mean_margin", ascending=True)
            ax5.barh(np.arange(len(temp)), temp["mean_margin"], color=YELLOW)
            ax5.set_yticks(np.arange(len(temp)))
            ax5.set_yticklabels(temp["label"], fontsize=9)
            ax5.set_title("Underspecification is typed")
            ax5.set_xlabel("mean margin")
            ax5.grid(axis="x", color=GRID, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No insufficiency CSV found", ha="center", va="center")
        ax5.set_axis_off()

    fig.text(
        0.5,
        0.025,
        "Implication: the system is not merely solving. It is learning when the semantic object is preserved, changed, or not yet defined.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=13,
    )

    savefig(VIZ / "phase86_5_08_hero_dashboard.png")


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

def write_manifest(phase_df: pd.DataFrame) -> None:
    manifest = {
        "phase": 86.5,
        "title": "Visual story of BBIT semantic-boundary emergence",
        "input_root": str(OUT),
        "output_dir": str(VIZ),
        "phases_loaded": phase_df["phase"].astype(int).tolist(),
        "visualizations": [
            "phase86_5_01_capability_ladder.png",
            "phase86_5_02_capability_matrix.png",
            "phase86_5_03_margin_ecology.png",
            "phase86_5_04_task_margin_heatmap.png",
            "phase86_5_05_three_way_decision_state_space.png",
            "phase86_5_06_semantic_transition_graph.png",
            "phase86_5_07_insufficiency_fingerprint.png",
            "phase86_5_08_hero_dashboard.png",
        ],
        "summary_table": phase_df.to_dict(orient="records"),
    }

    p = VIZ / "phase86_5_visual_story_manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[86.5] wrote: {p}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    print("[86.5] Visual story of BBIT semantic-boundary emergence")
    print(f"[86.5] reading from: {OUT}")
    print(f"[86.5] writing to: {VIZ}")

    summaries, tasks, trials, extras = load_data()
    if not summaries:
        raise RuntimeError(f"No phase summary JSON files found in {OUT}")

    phase_df = build_phase_frame(summaries)
    phase_df.to_csv(VIZ / "phase86_5_phase_summary_compiled.csv", index=False)
    print(f"[86.5] compiled phases: {phase_df['phase'].astype(int).tolist()}")
    print(f"[86.5] wrote: {VIZ / 'phase86_5_phase_summary_compiled.csv'}")

    plot_capability_ladder(phase_df)
    plot_capability_matrix(phase_df)
    plot_margin_ecology(trials, phase_df)
    plot_task_margin_heatmap(tasks)
    plot_phase86_decision_space(extras)
    plot_semantic_transition_graph(phase_df)
    plot_insufficiency_fingerprint(extras)
    plot_hero_dashboard(phase_df, trials, extras)
    write_manifest(phase_df)

    print("[86.5] complete")
    print("[86.5] output group:")
    print(f"       {VIZ}")


if __name__ == "__main__":
    main()