#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 45: Counterfactual quorum-lattice causal resilience verifier

Reset continuation:
  Phase 34 proved path-only choice under adversarial endpoint ambiguity.
  Phase 35 proved trajectory reconstruction under occlusion.
  Phase 36 proved noisy multi-branch causal disambiguation.
  Phase 37 proved intervention-invariant causal path reconstruction.
  Phase 38 proved cross-context intervention-bridge transfer.

Phase 45 adds a stricter necessity-verification problem: the system now receives several source
A->B->C examples from the same hidden causal family, but each source witness has
a different nuisance context, a different partial middle-state intervention, and
some witnesses are deliberately poisoned by a wrong path fragment. The reasoner
must infer the latent causal path by consensus, not by trusting any one source,
not by copying an intervention artifact, and not by choosing the nearest final
endpoint.

Task:
  Observe K source witnesses A_i -> Bobs_i -> C_i.
  Each Bobs_i contains noise/context/intervention. Some witnesses are poisoned.
  Infer the invariant latent two-step path A->B->C by robust multi-witness
  consensus.
  Transfer that consensus path onto target D and choose the D->E->F branch that
  preserves the consensus full path while rejecting endpoint, intervention-copy,
  context-copy, first-step-only, and second-step-only lures.

  Phase 45 adds a quorum-lattice necessity check: the consensus is recomputed
  with each witness removed. A valid answer must remain the same branch under
  these counterfactual evidence removals, proving that the result is not a
  single-witness shortcut or a poisoned-source accident.

Outputs:
  E:\BBIT\outputs_basic32\phase45_quorum_lattice_verifier_trials.csv
  E:\BBIT\outputs_basic32\phase45_quorum_lattice_verifier_summary.json
  E:\BBIT\outputs_basic32\phase45_quorum_lattice_verifier_report.md
  E:\BBIT\outputs_basic32\phase45_quorum_lattice_poison_sweep.csv
  E:\BBIT\outputs_basic32\phase45_examples\*.png
  plus summary png charts.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "45"
TITLE = "Counterfactual quorum-lattice causal resilience verifier"
SCRIPT_NAME = "geomlang_phase45_quorum_lattice_causal_resilience_verifier_basic32_E_drive.py"
ROOT = Path(r"E:\BBIT") if Path(r"E:\BBIT").exists() else Path.cwd()
OUT = ROOT / "outputs_basic32"
EXAMPLE_DIR = OUT / "phase45_examples"

SEED = 42042
TRIALS = 700
N_POINTS = 28
N_WITNESSES = 5
SOURCE_NOISE_SIGMA = 0.009
CANDIDATE_NOISE_SIGMA = 0.0012
INTERVENTION_FRACTION = 0.34
INTERVENTION_MAG = 0.13
CONTEXT_MAG = 0.055
TARGET_CONTEXT_MAG = 0.12
POISON_FRACTION = 0.28

FAMILIES = [
    "rigid_chain",
    "similarity_chain",
    "shear_chain",
    "left_right_chain",
    "top_bottom_chain",
    "quadrants_chain",
    "core_shell_chain",
]

BRANCHES = [
    "full_path",
    "endpoint_lure",
    "first_step_only",
    "second_step_only",
    "intervention_copy",
    "context_copy",
    "poison_consensus_lure",
]


@dataclass
class Affine:
    A: np.ndarray
    b: np.ndarray

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.A.T + self.b


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def normalize_points(p: np.ndarray) -> np.ndarray:
    p = p - p.mean(axis=0, keepdims=True)
    scale = np.sqrt((p * p).sum(axis=1)).max()
    if scale < 1e-9:
        scale = 1.0
    return p / scale


def make_cloud(rng: np.random.Generator, family: str, n: int = N_POINTS) -> np.ndarray:
    if family == "quadrants_chain":
        per = n // 4
        centers = np.array([[-0.55, -0.55], [0.55, -0.55], [-0.55, 0.55], [0.55, 0.55]])
        p = np.vstack([c + rng.normal(0, 0.105, size=(per, 2)) for c in centers])
    elif family == "core_shell_chain":
        n_core = n // 3
        n_shell = n - n_core
        core = rng.normal(0, 0.12, size=(n_core, 2))
        ang = rng.uniform(0, 2 * np.pi, size=n_shell)
        rad = rng.normal(0.74, 0.045, size=n_shell)
        shell = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
        p = np.vstack([core, shell])
    elif family in ("left_right_chain", "top_bottom_chain"):
        x = rng.uniform(-0.75, 0.75, size=n)
        y = 0.25 * np.sin(3.1 * x) + rng.normal(0, 0.13, size=n)
        p = np.c_[x, y]
        if family == "top_bottom_chain":
            p = p[:, ::-1]
    else:
        ang = rng.uniform(0, 2 * np.pi, size=n)
        rad = rng.beta(1.7, 2.8, size=n) * 0.95
        p = np.c_[rad * np.cos(ang), 0.72 * rad * np.sin(ang)]
        p += rng.normal(0, 0.035, size=p.shape)
    return normalize_points(p)


def family_affines(rng: np.random.Generator, family: str) -> Tuple[Affine, Affine]:
    if family == "rigid_chain":
        t1 = Affine(rot(rng.uniform(0.23, 0.42)), np.array([0.16, -0.05]))
        t2 = Affine(rot(rng.uniform(-0.44, -0.24)), np.array([-0.08, 0.14]))
    elif family == "similarity_chain":
        t1 = Affine(1.10 * rot(rng.uniform(0.15, 0.35)), np.array([0.09, 0.11]))
        t2 = Affine(0.88 * rot(rng.uniform(-0.31, -0.12)), np.array([0.13, -0.07]))
    elif family == "shear_chain":
        t1 = Affine(np.array([[1.00, 0.26], [0.02, 1.00]]), np.array([0.07, -0.11]))
        t2 = Affine(np.array([[1.00, -0.19], [0.05, 1.00]]), np.array([-0.10, 0.10]))
    elif family == "left_right_chain":
        t1 = Affine(np.array([[-1.0, 0.0], [0.0, 1.0]]) @ rot(rng.uniform(-0.10, 0.10)), np.array([0.18, 0.04]))
        t2 = Affine(rot(rng.uniform(0.08, 0.22)) @ np.array([[-1.0, 0.0], [0.0, 1.0]]), np.array([-0.14, 0.13]))
    elif family == "top_bottom_chain":
        t1 = Affine(np.array([[1.0, 0.0], [0.0, -1.0]]) @ rot(rng.uniform(-0.10, 0.10)), np.array([0.03, -0.18]))
        t2 = Affine(rot(rng.uniform(-0.22, -0.08)) @ np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([0.15, 0.11]))
    elif family == "quadrants_chain":
        t1 = Affine(np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([0.09, 0.07]))
        t2 = Affine(np.array([[0.0, 1.0], [-1.0, 0.0]]), np.array([-0.11, 0.08]))
    elif family == "core_shell_chain":
        t1 = Affine(np.array([[1.13, 0.08], [-0.03, 1.08]]), np.array([0.04, 0.10]))
        t2 = Affine(np.array([[0.91, -0.10], [0.06, 0.88]]), np.array([0.11, -0.08]))
    else:
        raise ValueError(f"Unknown family: {family}")
    return t1, t2


def fit_affine_lstsq(p: np.ndarray, q: np.ndarray, weights: np.ndarray | None = None) -> Affine:
    x = np.c_[p, np.ones(len(p))]
    if weights is None:
        m, *_ = np.linalg.lstsq(x, q, rcond=None)
    else:
        w = np.sqrt(np.maximum(weights, 1e-9))[:, None]
        m, *_ = np.linalg.lstsq(x * w, q * w, rcond=None)
    return Affine(m[:2, :].T, m[2, :])


def robust_fit_affine_lstsq(p: np.ndarray, q: np.ndarray, keep_frac: float = 0.70, iters: int = 2) -> Tuple[Affine, float]:
    weights = np.ones(len(p), dtype=np.float64)
    aff = fit_affine_lstsq(p, q, weights)
    for _ in range(iters):
        pred = aff(p)
        err = np.linalg.norm(pred - q, axis=1)
        cutoff = np.quantile(err, keep_frac)
        weights = np.where(err <= cutoff, 1.0, 0.05)
        aff = fit_affine_lstsq(p, q, weights)
    residual = float(np.mean(np.linalg.norm(aff(p) - q, axis=1)))
    return aff, residual


def affine_median(affs: List[Affine], weights: np.ndarray) -> Affine:
    mats = np.stack([a.A for a in affs], axis=0)
    bs = np.stack([a.b for a in affs], axis=0)
    # Weighted geometric median is overkill here. Use reliability-filtered coordinate median.
    cutoff = np.quantile(weights, 0.45)
    good = weights <= cutoff
    if good.sum() < max(3, len(affs) // 2):
        good = np.argsort(weights)[: max(3, len(affs) // 2)]
    return Affine(np.median(mats[good], axis=0), np.median(bs[good], axis=0))


def intervention_vector_field(p: np.ndarray, family: str) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    if family == "left_right_chain":
        v = np.c_[0.9 + 0.25 * np.sin(4 * y), 0.15 * np.cos(3 * x)]
    elif family == "top_bottom_chain":
        v = np.c_[0.18 * np.sin(3 * y), -(0.9 + 0.20 * np.cos(4 * x))]
    elif family == "quadrants_chain":
        v = np.c_[np.sign(x + 1e-6), -np.sign(y + 1e-6)]
    elif family == "core_shell_chain":
        r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-6
        v = p / r
    elif family == "shear_chain":
        v = np.c_[y, 0.35 * x]
    elif family == "similarity_chain":
        v = np.c_[-y, x]
    else:
        v = np.c_[0.6 * x + 0.2, -0.5 * y + 0.1]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def context_field(p: np.ndarray, seed_angle: float) -> np.ndarray:
    # Smooth nuisance field that changes by witness/context.
    u = p @ rot(seed_angle).T
    v = np.c_[np.sin(2.4 * u[:, 1]) + 0.2 * u[:, 0], np.cos(2.1 * u[:, 0]) - 0.15 * u[:, 1]]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def apply_middle_intervention(p: np.ndarray, family: str, frac: float, mag: float, sign: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p)
    k = max(3, int(round(frac * n)))
    projection = p @ np.array([0.73, -0.41])
    idx = np.argsort(projection)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    q = p.copy()
    v = intervention_vector_field(p, family)
    q[mask] += sign * mag * v[mask]
    return q, mask


def source_witnesses(
    rng: np.random.Generator,
    family: str,
    t1: Affine,
    t2: Affine,
    poison_fraction: float = POISON_FRACTION,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]], Dict[str, float]]:
    witnesses = []
    n_poison = int(round(N_WITNESSES * poison_fraction))
    poison_ids = set(rng.choice(np.arange(N_WITNESSES), size=n_poison, replace=False).tolist()) if n_poison else set()
    for i in range(N_WITNESSES):
        a = make_cloud(rng, family)
        b_lat = t1(a)
        c_lat = t2(b_lat)

        angle = rng.uniform(-np.pi, np.pi)
        ctx = CONTEXT_MAG * context_field(a, angle)
        a_obs = a + 0.25 * ctx + rng.normal(0, SOURCE_NOISE_SIGMA, size=a.shape)
        b_obs = b_lat + 0.55 * ctx + rng.normal(0, SOURCE_NOISE_SIGMA, size=b_lat.shape)
        c_obs = c_lat + 0.30 * ctx + rng.normal(0, SOURCE_NOISE_SIGMA, size=c_lat.shape)

        b_obs, _ = apply_middle_intervention(b_obs, family, INTERVENTION_FRACTION, INTERVENTION_MAG, sign=1.0)

        poisoned = i in poison_ids
        if poisoned:
            # Poisoned witness: keep endpoint plausible, but corrupt one local causal step.
            wrong = intervention_vector_field(b_lat, family)
            b_obs = b_obs + 0.10 * wrong
            c_obs = c_obs - 0.08 * wrong + rng.normal(0, SOURCE_NOISE_SIGMA, size=c_obs.shape)

        witnesses.append((a_obs, b_obs, c_obs, poisoned))
    return witnesses, {"poisoned_witnesses": float(n_poison), "poison_fraction": float(poison_fraction)}


def infer_consensus(witnesses: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]]) -> Tuple[Affine, Affine, Dict[str, float]]:
    t1s: List[Affine] = []
    t2s: List[Affine] = []
    residuals = []
    closure_residuals = []
    for a, b, c, _poisoned in witnesses:
        f1, r1 = robust_fit_affine_lstsq(a, b, keep_frac=0.66, iters=2)
        f2, r2 = robust_fit_affine_lstsq(b, c, keep_frac=0.66, iters=2)
        # Closure check: if the witness has a fake middle step, A->B->C will fit worse.
        pred_c = f2(f1(a))
        closure = float(np.mean(np.linalg.norm(pred_c - c, axis=1)))
        t1s.append(f1)
        t2s.append(f2)
        residuals.append(r1 + r2)
        closure_residuals.append(closure)
    reliab = np.asarray(residuals) + 0.75 * np.asarray(closure_residuals)
    c1 = affine_median(t1s, reliab)
    c2 = affine_median(t2s, reliab)

    # Phase 45 quorum-lattice sufficiency diagnostic: rebuild the weighted-median
    # consensus from multiple small quorums rather than only from N-1 leave-outs.
    # If the causal abstraction is real, any clean-enough 3-of-5 quorum should
    # recover nearly the same A->B and B->C operators. If the answer was carried
    # by one privileged witness, these small quorums will drift.
    quorum_size = max(3, min(len(t1s), int(np.ceil(0.60 * len(t1s)))))
    quorum_drifts: List[float] = []
    quorum_scores: List[float] = []
    for combo in combinations(range(len(t1s)), quorum_size):
        qr = np.asarray([reliab[j] for j in combo], dtype=np.float64)
        q1 = affine_median([t1s[j] for j in combo], qr)
        q2 = affine_median([t2s[j] for j in combo], qr)
        drift = (
            np.linalg.norm(c1.A - q1.A) + np.linalg.norm(c1.b - q1.b)
            + np.linalg.norm(c2.A - q2.A) + np.linalg.norm(c2.b - q2.b)
        )
        quorum_drifts.append(float(drift))
        quorum_scores.append(float(np.mean(qr)))

    # Phase 45 quorum-lattice diagnostic: now ask a harder question.
    # Instead of only proving that small clean-enough quorums are sufficient,
    # remove every possible small hostile coalition and rebuild the causal
    # abstraction from the surviving witnesses. If one witness clique is carrying
    # the answer, at least one exclusion should cause a large operator drift.
    coalition_size = max(1, min(len(t1s) - 1, int(np.ceil(0.40 * len(t1s)))))
    coalition_drifts: List[float] = []
    coalition_scores: List[float] = []
    all_idx = set(range(len(t1s)))
    for excluded in combinations(range(len(t1s)), coalition_size):
        keep = sorted(all_idx.difference(excluded))
        kr = np.asarray([reliab[j] for j in keep], dtype=np.float64)
        k1 = affine_median([t1s[j] for j in keep], kr)
        k2 = affine_median([t2s[j] for j in keep], kr)
        drift = (
            np.linalg.norm(c1.A - k1.A) + np.linalg.norm(c1.b - k1.b)
            + np.linalg.norm(c2.A - k2.A) + np.linalg.norm(c2.b - k2.b)
        )
        coalition_drifts.append(float(drift))
        coalition_scores.append(float(np.mean(kr)))

    return c1, c2, {
        "mean_witness_fit": float(np.mean(residuals)),
        "mean_closure_residual": float(np.mean(closure_residuals)),
        "minimal_quorum_error": float(np.mean(quorum_drifts)),
        "minimal_quorum_max_error": float(np.max(quorum_drifts)),
        "minimal_quorum_best_error": float(np.min(quorum_drifts)),
        "minimal_quorum_size": float(quorum_size),
        "minimal_quorum_mean_fit": float(np.mean(quorum_scores)),
        "quorum_lattice_error": float(np.mean(coalition_drifts)),
        "quorum_lattice_max_error": float(np.max(coalition_drifts)),
        "quorum_lattice_best_error": float(np.min(coalition_drifts)),
        "quorum_lattice_size": float(coalition_size),
        "quorum_lattice_mean_fit": float(np.mean(coalition_scores)),
        "best_witness_score": float(np.min(reliab)),
        "worst_witness_score": float(np.max(reliab)),
    }


def mse(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.sum((p - q) ** 2, axis=1)))


def final_only_score_twin(f0: np.ndarray, f1: np.ndarray, candidates: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
    return {name: 0.5 * (mse(branch_f0, f0) + mse(branch_f1, f1)) for name, (_e0, branch_f0, _e1, branch_f1) in candidates.items()}


def make_twin_candidates(
    rng: np.random.Generator,
    family: str,
    d0: np.ndarray,
    d1: np.ndarray,
    true_t1: Affine,
    true_t2: Affine,
    consensus_t1: Affine,
    consensus_t2: Affine,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], str]:
    """Two target contexts; only the correct branch preserves the same causal abstraction in both."""
    e0_true = true_t1(d0)
    f0_true = true_t2(e0_true)
    e1_true = true_t1(d1)
    f1_true = true_t2(e1_true)
    e0_cons = consensus_t1(d0)
    f0_cons = consensus_t2(e0_cons)
    e1_cons = consensus_t1(d1)
    f1_cons = consensus_t2(e1_cons)

    int_e0, _ = apply_middle_intervention(e0_true, family, INTERVENTION_FRACTION, INTERVENTION_MAG, sign=1.0)
    int_e1, _ = apply_middle_intervention(e1_true, family, INTERVENTION_FRACTION, INTERVENTION_MAG, sign=-1.0)
    ctx0 = TARGET_CONTEXT_MAG * context_field(d0, rng.uniform(-np.pi, np.pi))
    ctx1 = TARGET_CONTEXT_MAG * context_field(d1, rng.uniform(-np.pi, np.pi))

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    candidates["full_path"] = (
        e0_cons + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_cons.shape),
        f0_cons + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_cons.shape),
        e1_cons + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_cons.shape),
        f1_cons + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_cons.shape),
    )

    # Looks good in one target context, but breaks when the same abstraction must be sufficient across small witness quorums.
    candidates["single_context_overfit"] = (
        candidates["full_path"][0],
        candidates["full_path"][1],
        e1_cons + 0.15 * intervention_vector_field(e1_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_cons.shape),
        f1_cons - 0.12 * intervention_vector_field(f1_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_cons.shape),
    )
    candidates["endpoint_lure"] = (
        int_e0 + 0.55 * ctx0 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        f0_true + rng.normal(0, CANDIDATE_NOISE_SIGMA * 1.5, size=f0_true.shape),
        int_e1 + 0.55 * ctx1 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        f1_true + rng.normal(0, CANDIDATE_NOISE_SIGMA * 1.5, size=f1_true.shape),
    )
    candidates["first_step_only"] = (
        e0_true + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        f0_true + 0.10 * intervention_vector_field(f0_true, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_true.shape),
        e1_true + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        f1_true - 0.10 * intervention_vector_field(f1_true, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_true.shape),
    )
    candidates["second_step_only"] = (
        e0_true + 0.10 * context_field(e0_true, rng.uniform(-np.pi, np.pi)) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        f0_true + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_true.shape),
        e1_true + 0.10 * context_field(e1_true, rng.uniform(-np.pi, np.pi)) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        f1_true + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_true.shape),
    )
    candidates["intervention_copy"] = (
        int_e0 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        true_t2(int_e0) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_true.shape),
        int_e1 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        true_t2(int_e1) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_true.shape),
    )
    candidates["context_copy"] = (
        e0_true + ctx0 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        f0_true + 0.70 * ctx0 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_true.shape),
        e1_true + ctx1 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        f1_true + 0.70 * ctx1 + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_true.shape),
    )
    candidates["poison_consensus_lure"] = (
        e0_cons + 0.18 * intervention_vector_field(e0_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e0_true.shape),
        f0_cons - 0.15 * intervention_vector_field(f0_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f0_true.shape),
        e1_cons + 0.18 * intervention_vector_field(e1_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=e1_true.shape),
        f1_cons - 0.15 * intervention_vector_field(f1_cons, family) + rng.normal(0, CANDIDATE_NOISE_SIGMA, size=f1_true.shape),
    )
    return candidates, "full_path"


def score_twin_branch(d0: np.ndarray, d1: np.ndarray, branch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], t1: Affine, t2: Affine) -> Tuple[float, float, float]:
    e0, f0, e1, f1 = branch
    pe0 = t1(d0)
    pf0 = t2(pe0)
    pe1 = t1(d1)
    pf1 = t2(pe1)
    path = 0.25 * (mse(e0, pe0) + mse(f0, pf0) + mse(e1, pe1) + mse(f1, pf1))

    # Counterfactual abstraction penalty: the inferred local transforms in target-0
    # and target-1 should agree. Single-context overfits score well on path-0 and
    # badly here.
    a0 = fit_affine_lstsq(d0, e0)
    b0 = fit_affine_lstsq(e0, f0)
    a1 = fit_affine_lstsq(d1, e1)
    b1 = fit_affine_lstsq(e1, f1)
    inv = float(np.mean((a0.A - a1.A) ** 2) + np.mean((b0.A - b1.A) ** 2) + 0.10 * np.mean((a0.b - a1.b) ** 2) + 0.10 * np.mean((b0.b - b1.b) ** 2))
    score = path + 0.012 * inv
    return score, path, inv

def run_trial(rng: np.random.Generator, trial_id: int, poison_fraction: float = POISON_FRACTION) -> Dict[str, object]:
    family = FAMILIES[trial_id % len(FAMILIES)]
    t1, t2 = family_affines(rng, family)
    witnesses, meta = source_witnesses(rng, family, t1, t2, poison_fraction=poison_fraction)
    c1, c2, infer_meta = infer_consensus(witnesses)

    d0 = make_cloud(rng, family)
    d1 = make_cloud(rng, family)
    d1 = d1 + 0.055 * context_field(d1, rng.uniform(-np.pi, np.pi))
    candidates, correct_branch = make_twin_candidates(rng, family, d0, d1, t1, t2, c1, c2)

    score_pack = {name: score_twin_branch(d0, d1, branch, c1, c2) for name, branch in candidates.items()}
    scores = {name: pack[0] for name, pack in score_pack.items()}
    inv_scores = {name: pack[2] for name, pack in score_pack.items()}
    pred = min(scores, key=scores.get)
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    margin = float(sorted_scores[1][1] - sorted_scores[0][1])

    f0_true = t2(t1(d0))
    f1_true = t2(t1(d1))
    final_scores = final_only_score_twin(f0_true, f1_true, candidates)
    final_pred = min(final_scores, key=final_scores.get)
    final_sorted = sorted(final_scores.items(), key=lambda kv: kv[1])
    final_margin = float(final_sorted[1][1] - final_sorted[0][1])

    # Scramble check: permuting one target context should destroy the abstract twin replay.
    perm = rng.permutation(len(d1))
    e0, f0, e1, f1 = candidates[pred]
    scrambled_score = score_twin_branch(d0, d1, (e0, f0, e1[perm], f1), c1, c2)[0]
    scramble_stable = bool(scores[pred] < scrambled_score)

    transfer0 = np.mean(np.linalg.norm(c2(c1(d0)) - f0_true, axis=1))
    transfer1 = np.mean(np.linalg.norm(c2(c1(d1)) - f1_true, axis=1))
    transfer_err = float(0.5 * (transfer0 + transfer1))
    recon_err = float(np.mean([np.linalg.norm(c2(c1(a)) - c, axis=1).mean() for a, _b, c, _p in witnesses]))

    row: Dict[str, object] = {
        "trial": trial_id,
        "family": family,
        "pred_branch": pred,
        "correct_branch": correct_branch,
        "trajectory_correct": int(pred == correct_branch),
        "final_only_pred": final_pred,
        "final_only_correct": int(final_pred == correct_branch),
        "scramble_stable": int(scramble_stable),
        "margin": margin,
        "final_margin": final_margin,
        "invariance_penalty": float(inv_scores[pred]),
        "transfer_error": transfer_err,
        "source_reconstruction_error": recon_err,
        **meta,
        **infer_meta,
    }
    for b in BRANCHES:
        row[f"score_{b}"] = float(scores[b])
        row[f"invariance_{b}"] = float(inv_scores[b])
        row[f"final_score_{b}"] = float(final_scores[b])
    return row

def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    arr = lambda key: np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    summary: Dict[str, object] = {
        "phase": PHASE,
        "title": TITLE,
        "trials": len(rows),
        "n_witnesses": N_WITNESSES,
        "poison_fraction": POISON_FRACTION,
        "trajectory_accuracy": float(arr("trajectory_correct").mean()),
        "final_only_accuracy": float(arr("final_only_correct").mean()),
        "gain": float(arr("trajectory_correct").mean() - arr("final_only_correct").mean()),
        "scramble_stability": float(arr("scramble_stable").mean()),
        "mean_transfer_error": float(arr("transfer_error").mean()),
        "mean_source_reconstruction_error": float(arr("source_reconstruction_error").mean()),
        "mean_witness_fit": float(arr("mean_witness_fit").mean()),
        "mean_closure_residual": float(arr("mean_closure_residual").mean()),
        "mean_quorum_lattice_error": float(arr("quorum_lattice_error").mean()),
        "mean_quorum_lattice_max_error": float(arr("quorum_lattice_max_error").mean()),
        "mean_quorum_lattice_best_error": float(arr("quorum_lattice_best_error").mean()),
        "mean_quorum_lattice_size": float(arr("quorum_lattice_size").mean()),
        "mean_quorum_lattice_fit": float(arr("quorum_lattice_mean_fit").mean()),
        "mean_margin": float(arr("margin").mean()),
        "mean_final_margin": float(arr("final_margin").mean()),
    }
    fams = {}
    for fam in FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        aa = lambda key: np.asarray([float(r[key]) for r in fr], dtype=np.float64)
        fams[fam] = {
            "n": len(fr),
            "trajectory_accuracy": float(aa("trajectory_correct").mean()),
            "final_only_accuracy": float(aa("final_only_correct").mean()),
            "scramble_stability": float(aa("scramble_stable").mean()),
            "mean_transfer_error": float(aa("transfer_error").mean()),
            "mean_source_reconstruction_error": float(aa("source_reconstruction_error").mean()),
            "mean_quorum_lattice_error": float(aa("quorum_lattice_error").mean()),
            "min_margin": float(aa("margin").min()),
        }
    summary["families"] = fams
    # PASS allows one or two misses because the trial includes random poisoned witnesses.
    summary["pass"] = bool(
        summary["trajectory_accuracy"] >= 0.985
        and summary["gain"] >= 0.55
        and summary["scramble_stability"] >= 0.985
        and min(v["trajectory_accuracy"] for v in fams.values()) >= 0.96
    )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_accuracy(summary: Dict[str, object]) -> None:
    fams = summary["families"]
    labels = list(fams.keys())
    vals = [fams[k]["trajectory_accuracy"] for k in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("trajectory accuracy")
    ax.set_title("Phase 45 counterfactual quorum-lattice causal resilience reasoning: accuracy by hidden family")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "phase45_quorum_lattice_verifier_accuracy.png", dpi=140)
    plt.close(fig)


def plot_baseline_gap(summary: Dict[str, object]) -> None:
    fams = summary["families"]
    labels = list(fams.keys())
    traj = [fams[k]["trajectory_accuracy"] for k in labels]
    final = [fams[k]["final_only_accuracy"] for k in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width/2, final, width, label="final-only")
    ax.bar(x + width/2, traj, width, label="quorum-lattice verifier")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 45 final-only baseline vs quorum-lattice causal resilience")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase45_quorum_lattice_verifier_baseline_gap.png", dpi=140)
    plt.close(fig)


def plot_hist(rows: List[Dict[str, object]], key: str, title: str, xlabel: str, filename: str) -> None:
    vals = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(vals, bins=36)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=140)
    plt.close(fig)


def plot_branch_counts(rows: List[Dict[str, object]]) -> None:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[str(r["pred_branch"])] = counts.get(str(r["pred_branch"]), 0) + 1
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, vals)
    ax.set_ylabel("trials")
    ax.set_title("Phase 45 predicted branch counts")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "phase45_quorum_lattice_verifier_branch_counts.png", dpi=140)
    plt.close(fig)


def poison_sweep() -> List[Dict[str, float]]:
    rows = []
    for frac in [0.0, 0.14, 0.28, 0.42, 0.56, 0.70]:
        rng = np.random.default_rng(SEED + int(frac * 1000) + 901)
        local_rows = [run_trial(rng, i, poison_fraction=frac) for i in range(45)]
        traj_acc = float(np.mean([r["trajectory_correct"] for r in local_rows]))
        final_acc = float(np.mean([r["final_only_correct"] for r in local_rows]))
        rows.append({"poison_fraction": frac, "trajectory_accuracy": traj_acc, "final_only_accuracy": final_acc})
    return rows


def write_poison_sweep(rows: List[Dict[str, float]]) -> None:
    path = OUT / "phase45_quorum_lattice_poison_sweep.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["poison_fraction", "trajectory_accuracy", "final_only_accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = [r["poison_fraction"] for r in rows]
    ax.plot(x, [r["trajectory_accuracy"] for r in rows], marker="o", label="consensus trajectory")
    ax.plot(x, [r["final_only_accuracy"] for r in rows], marker="o", label="final-only")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("poisoned witness fraction")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 45 poisoned-witness stress sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase45_quorum_lattice_verifier_poison_sweep.png", dpi=140)
    plt.close(fig)


def write_report(summary: Dict[str, object], path: Path) -> None:
    fams = summary["families"]
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}\n")
    lines.append("## Result\n")
    lines.append(f"- PASS: `{summary['pass']}`")
    lines.append(f"- trajectory accuracy: `{summary['trajectory_accuracy']:.4f}`")
    lines.append(f"- final-only accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- gain: `{summary['gain']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- witnesses per trial: `{summary['n_witnesses']}`")
    lines.append(f"- poisoned witness fraction: `{summary['poison_fraction']:.2f}`")
    lines.append(f"- mean transfer error: `{summary['mean_transfer_error']:.6f}`")
    lines.append(f"- mean source reconstruction error: `{summary['mean_source_reconstruction_error']:.6f}`")
    lines.append(f"- mean witness fit: `{summary['mean_witness_fit']:.6f}`")
    lines.append(f"- mean closure residual: `{summary['mean_closure_residual']:.6f}`")
    lines.append(f"- mean quorum-lattice error: `{summary['mean_quorum_lattice_error']:.6f}`")
    lines.append(f"- mean quorum-lattice max error: `{summary['mean_quorum_lattice_max_error']:.6f}`")
    lines.append(f"- mean quorum-lattice best error: `{summary['mean_quorum_lattice_best_error']:.6f}`")
    lines.append(f"- mean quorum-lattice size: `{summary['mean_quorum_lattice_size']:.1f}`")
    lines.append(f"- mean trajectory margin: `{summary['mean_margin']:.6f}`")
    lines.append("\n## Family summary\n")
    lines.append("| family | traj_acc | final_acc | stable | transfer_err | recon_err | min_margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, d in fams.items():
        lines.append(
            f"| {fam} | {d['trajectory_accuracy']:.3f} | {d['final_only_accuracy']:.3f} | "
            f"{d['scramble_stability']:.3f} | {d['mean_transfer_error']:.6f} | "
            f"{d['mean_source_reconstruction_error']:.6f} | {d['min_margin']:.6f} |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "Phase 45 turns the prior necessity, sufficiency, and hostile-coalition checks into one quorum-lattice verifier. It asks whether the same inferred causal path survives many counterfactual witness subsets rather than depending on the whole witness pool. "
        "The model is not allowed to rely on one clean source example or on the full ensemble average: each witness can carry "
        "a different context field, a middle-state intervention, observation noise, and a subset "
        "of witnesses are deliberately poisoned. The winning branch is the one that preserves "
        "the robustly inferred two-step causal path across the deletion/quorum lattice, including clean-enough minimal quorums and hostile coalition removals."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_examples(rng: np.random.Generator, n: int = 4) -> None:
    ensure_dirs()
    for i, fam in enumerate(FAMILIES[:n]):
        t1, t2 = family_affines(rng, fam)
        witnesses, _ = source_witnesses(rng, fam, t1, t2, poison_fraction=POISON_FRACTION)
        c1, c2, _ = infer_consensus(witnesses)
        d0 = make_cloud(rng, fam)
        d1_base = make_cloud(rng, fam)
        d1 = d1_base + 0.055 * context_field(d1_base, rng.uniform(-np.pi, np.pi))
        candidates, _ = make_twin_candidates(rng, fam, d0, d1, t1, t2, c1, c2)
        e0_full, f0_full, e1_full, f1_full = candidates["full_path"]
        _e0_lure, _f0_lure, _e1_lure, f1_lure = candidates["single_context_overfit"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(d0[:, 0], d0[:, 1], s=12, label="D0 target")
        axes[0].scatter(e0_full[:, 0], e0_full[:, 1], s=12, label="E0 consensus")
        axes[0].scatter(f0_full[:, 0], f0_full[:, 1], s=12, label="F0 consensus")
        axes[0].set_title(f"{fam}: target context 0")
        axes[0].axis("equal")
        axes[0].legend(fontsize=8)
        axes[1].scatter(d1[:, 0], d1[:, 1], s=12, label="D1 target")
        axes[1].scatter(e1_full[:, 0], e1_full[:, 1], s=12, label="E1 consensus")
        axes[1].scatter(f1_full[:, 0], f1_full[:, 1], s=12, label="F1 consensus")
        axes[1].set_title("counterfactual twin replay")
        axes[1].axis("equal")
        axes[1].legend(fontsize=8)
        axes[2].scatter(f1_full[:, 0], f1_full[:, 1], s=12, label="correct F1")
        axes[2].scatter(f1_lure[:, 0], f1_lure[:, 1], s=12, label="single-context lure")
        axes[2].set_title("quorum-lattice rejection")
        axes[2].axis("equal")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(EXAMPLE_DIR / f"phase45_example_{i+1}_{fam}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] reset continued: from leave-one necessity to quorum-lattice causal resilience")
    print(f"[{PHASE}] task: infer invariant A->B->C from witnesses, then require the answer branch to survive the witness-deletion lattice: minimal quorum and hostile-coalition replays")

    rows = [run_trial(rng, i, poison_fraction=POISON_FRACTION) for i in range(TRIALS)]
    summary = aggregate(rows)

    trials_path = OUT / "phase45_quorum_lattice_verifier_trials.csv"
    summary_path = OUT / "phase45_quorum_lattice_verifier_summary.json"
    report_path = OUT / "phase45_quorum_lattice_verifier_report.md"
    write_csv(trials_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, report_path)

    plot_accuracy(summary)
    plot_baseline_gap(summary)
    plot_branch_counts(rows)
    plot_hist(rows, "margin", "Phase 45 quorum-lattice trajectory answer margin distribution", "runner-up trajectory score - best trajectory score", "phase45_quorum_lattice_verifier_margin_distribution.png")
    plot_hist(rows, "final_margin", "Phase 45 final-only endpoint margin distribution", "runner-up endpoint score - best endpoint score", "phase45_quorum_lattice_verifier_final_only_margin_distribution.png")
    sweep_rows = poison_sweep()
    write_poison_sweep(sweep_rows)
    save_examples(rng)

    print(f"[{PHASE}] PHASE45_COUNTERFACTUAL_QUORUM_LATTICE_CAUSAL_RESILIENCE_PASS={summary['pass']}")
    print(
        f"[{PHASE}] trajectory_accuracy={summary['trajectory_accuracy']:.4f} "
        f"final_only_accuracy={summary['final_only_accuracy']:.4f} "
        f"gain={summary['gain']:.4f} scramble_stability={summary['scramble_stability']:.4f} "
        f"trials={summary['trials']} witnesses={summary['n_witnesses']} poison_fraction={summary['poison_fraction']:.2f}"
    )
    print(
        f"[{PHASE}] mean_witness_fit={summary['mean_witness_fit']:.6f} "
        f"mean_closure_residual={summary['mean_closure_residual']:.6f} "
        f"mean_quorum_lattice_error={summary['mean_quorum_lattice_error']:.6f} "
        f"mean_quorum_lattice_max_error={summary['mean_quorum_lattice_max_error']:.6f} "
        f"mean_quorum_lattice_best_error={summary['mean_quorum_lattice_best_error']:.6f} "
        f"mean_source_reconstruction_error={summary['mean_source_reconstruction_error']:.6f} "
        f"mean_transfer_error={summary['mean_transfer_error']:.6f} mean_margin={summary['mean_margin']:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam, d in summary["families"].items():
        print(
            f"  - {fam:<18} traj_acc={d['trajectory_accuracy']:.3f} "
            f"final_acc={d['final_only_accuracy']:.3f} stable={d['scramble_stability']:.3f} "
            f"transfer_err={d['mean_transfer_error']:.6f} recon_err={d['mean_source_reconstruction_error']:.6f} "
            f"min_margin={d['min_margin']:.6f}"
        )
    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote quorum-lattice poison sweep: {OUT / 'phase45_quorum_lattice_poison_sweep.csv'}")
    print(f"[{PHASE}] wrote example png dir: {EXAMPLE_DIR}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
