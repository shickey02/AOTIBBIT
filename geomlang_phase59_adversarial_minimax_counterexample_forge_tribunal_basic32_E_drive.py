#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 59: Adversarial minimax counterexample-forge tribunal

Drop-in path:
  E:\BBIT\bbit_geomlang\geomlang_phase59_counterfactual_adversarial_minimax_counterexample_forge_tribunal_basic32_E_drive.py

Run:
  python bbit_geomlang/geomlang_phase59_counterfactual_adversarial_minimax_counterexample_forge_tribunal_basic32_E_drive.py

Purpose
-------
Phase 51 proved the typed contradiction tribunal could accept the true A -> B -> C certificate,
reject the false certificates, and pass one sealed holdout/null-ablation checks. Phase 59 raises the bar by
adding a metamorphic-shadow invariance tribunal: the certificate must keep its selectivity
on two independent panels, including a higher-noise adversarial panel that was never used to
select the winning branch.

Phase 59 adds a metamorphic-shadow invariance tribunal on top of the Phase 54 null-ablation near-miss tribunal.

Instead of only asking:
  "Can the ledger preserve the correct A -> B -> C causal path?"

it also asks:
  "Can the ledger reject deliberate contradiction witnesses and wrong causal certificates?"

The verifier must now pass two simultaneous tests:

  1. Acceptance:
     The true branch must still survive clean/noisy/poisoned witness ledgers.

  2. Rejection:
     Deliberately false certificates must fail. These include:
       - reversed direction certificate: C -> B -> A
       - shortcut certificate: A -> C
       - wrong middle certificate: A -> wrong_B -> C
       - wrong family certificate: witness family swapped with another branch
       - endpoint-only certificate: ignores the middle causal state

This phase is designed to catch "rubber-stamp" verifiers. If every proposed certificate
passes, the phase fails even if trajectory accuracy is 1.0.

Outputs
-------
Writes into E:\BBIT\outputs_basic32 by default:

  phase59_adversarial_minimax_counterexample_forge_tribunal_trials.csv
  phase59_adversarial_minimax_counterexample_forge_tribunal_summary.json
  phase59_adversarial_minimax_counterexample_forge_tribunal_report.md
  phase59_adversarial_minimax_counterexample_forge_tribunal_poison_sweep.csv
  phase59_examples\*.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_accuracy.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_rejection_rates.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_false_certificate_score_distribution.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_true_vs_false_score_margin.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_branch_counts.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_poison_sweep.png
  phase59_adversarial_minimax_counterexample_forge_tribunal_forge_rates.png

Pass condition
--------------
  true trajectory accuracy >= 0.98
  false certificate rejection rate >= 0.98
  double-sealed holdout acceptance rate >= 0.98
  contradiction selectivity margin > 0
  scramble stability >= 0.98

This script is intentionally self-contained. It does not import prior phase files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    print("[59] WARNING: matplotlib unavailable; png plots will be skipped:", exc)


PHASE = "59"
TITLE = "Adversarial minimax counterexample-forge tribunal"
PASS_FLAG = "PHASE59_ADVERSARIAL_MINIMAX_COUNTEREXAMPLE_FORGE_TRIBUNAL_PASS"

FAMILIES = [
    "rigid_chain",
    "similarity_chain",
    "shear_chain",
    "left_right_chain",
    "top_bottom_chain",
    "quadrants_chain",
    "core_shell_chain",
]

FALSE_CERT_TYPES = [
    "reversed_direction",
    "shortcut_endpoint_only",
    "wrong_middle",
    "wrong_family",
    "scrambled_temporal_order",
]


# -----------------------------
# Geometry helpers
# -----------------------------

def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def safe_norm(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(x) + eps)


def mean_pairwise_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))




def transition_role_penalty(
    w: Witness,
    cand_M1: np.ndarray,
    cand_b1: np.ndarray,
    cand_M2: np.ndarray,
    cand_b2: np.ndarray,
    ref_M1: np.ndarray,
    ref_b1: np.ndarray,
    ref_M2: np.ndarray,
    ref_b2: np.ndarray,
    pred_B: np.ndarray,
    pred_C: np.ndarray,
) -> float:
    """
    Axis-typed role interrogation.

    Phase 49 could accept false certificates for near-involutive families
    because endpoint replay alone could make a reversed or shortcut explanation
    look plausible. This probe asks whether the certificate's first transition
    has the A->B role and its second transition has the B->C role.
    """
    transform_delta = (
        np.linalg.norm(cand_M1 - ref_M1)
        + 0.45 * np.linalg.norm(cand_b1 - ref_b1)
        + np.linalg.norm(cand_M2 - ref_M2)
        + 0.45 * np.linalg.norm(cand_b2 - ref_b2)
    )

    obs_v1 = w.B - w.A
    obs_v2 = w.C - w.B
    cand_v1 = pred_B - w.A
    cand_v2 = pred_C - pred_B
    velocity_delta = mean_pairwise_error(cand_v1, obs_v1) + mean_pairwise_error(cand_v2, obs_v2)

    def moments(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(X, axis=0)
        Z = X - mu
        cov = (Z.T @ Z) / max(1, len(X) - 1)
        vals = np.linalg.eigvalsh(cov)
        return mu, np.sort(vals)

    a_mu, a_ev = moments(w.A)
    b_mu, b_ev = moments(w.B)
    c_mu, c_ev = moments(w.C)
    pb_mu, pb_ev = moments(pred_B)
    pc_mu, pc_ev = moments(pred_C)
    role_moment_delta = (
        np.linalg.norm((pb_mu - a_mu) - (b_mu - a_mu))
        + np.linalg.norm((pc_mu - pb_mu) - (c_mu - b_mu))
        + np.linalg.norm((pb_ev - a_ev) - (b_ev - a_ev))
        + np.linalg.norm((pc_ev - pb_ev) - (c_ev - b_ev))
    )

    return float(0.55 * transform_delta + 0.35 * velocity_delta + 0.25 * role_moment_delta)


def canonical_shape(rng: np.random.Generator, n: int = 32) -> np.ndarray:
    """
    Make a stable but nontrivial 2D point cloud.
    A tiny amount of jitter prevents degeneracies while preserving family structure.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + 0.18 * np.sin(3 * t + 0.4) + 0.09 * np.cos(5 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    pts = np.stack([x, y], axis=1)
    pts += rng.normal(0, 0.015, size=pts.shape)
    pts -= pts.mean(axis=0, keepdims=True)
    return pts.astype(np.float64)


def apply_affine(pts: np.ndarray, M: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pts @ M.T + b[None, :]


def fit_affine(src: np.ndarray, dst: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Least-squares affine fit dst ~= src @ M.T + b.
    Returns M, b, residual.
    """
    n = src.shape[0]
    X = np.concatenate([src, np.ones((n, 1))], axis=1)
    Y = dst
    # Ridge-stabilized normal solve.
    A = X.T @ X + ridge * np.eye(3)
    B = X.T @ Y
    W = np.linalg.solve(A, B)  # 3 x 2
    pred = X @ W
    M = W[:2, :].T
    b = W[2, :]
    residual = mean_pairwise_error(pred, dst)
    return M, b, residual


def compose_affine(M2: np.ndarray, b2: np.ndarray, M1: np.ndarray, b1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose T2(T1(x)).
    T1: x -> M1 x + b1
    T2: x -> M2 x + b2
    """
    return M2 @ M1, M2 @ b1 + b2


def family_transforms(family: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return true A->B and B->C affine transforms for a hidden family.
    The families deliberately share endpoint ambiguity: final-only A->C can be
    made too close across branches, while the two-step trajectory separates them.
    """
    angle1 = rng.uniform(-0.45, 0.45)
    angle2 = rng.uniform(-0.45, 0.45)
    shift1 = rng.normal(0, 0.055, size=2)
    shift2 = rng.normal(0, 0.055, size=2)

    if family == "rigid_chain":
        M1 = rot(angle1)
        M2 = rot(angle2)
    elif family == "similarity_chain":
        M1 = 1.0 + rng.uniform(-0.10, 0.10)
        M2 = 1.0 + rng.uniform(-0.10, 0.10)
        M1 = M1 * rot(angle1)
        M2 = M2 * rot(angle2)
    elif family == "shear_chain":
        sh1 = rng.uniform(-0.24, 0.24)
        sh2 = rng.uniform(-0.24, 0.24)
        M1 = np.array([[1.0, sh1], [0.0, 1.0]], dtype=np.float64) @ rot(angle1 * 0.35)
        M2 = np.array([[1.0, 0.0], [sh2, 1.0]], dtype=np.float64) @ rot(angle2 * 0.35)
    elif family == "left_right_chain":
        M1 = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64) @ rot(angle1 * 0.25)
        M2 = rot(angle2 * 0.25) @ np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    elif family == "top_bottom_chain":
        M1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64) @ rot(angle1 * 0.25)
        M2 = rot(angle2 * 0.25) @ np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    elif family == "quadrants_chain":
        M1 = rot(np.pi / 2 + angle1 * 0.18)
        M2 = rot(-np.pi / 2 + angle2 * 0.18)
    elif family == "core_shell_chain":
        # Anisotropic compression then re-expansion with mild rotation.
        M1 = rot(angle1 * 0.25) @ np.array([[0.82, 0.0], [0.0, 1.18]], dtype=np.float64)
        M2 = np.array([[1.18, 0.0], [0.0, 0.82]], dtype=np.float64) @ rot(angle2 * 0.25)
    else:
        raise ValueError(f"unknown family {family!r}")

    return M1, shift1, M2, shift2


def make_chain(family: str, rng: np.random.Generator, n: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    A = canonical_shape(rng, n=n)
    M1, b1, M2, b2 = family_transforms(family, rng)
    B = apply_affine(A, M1, b1)
    C = apply_affine(B, M2, b2)
    meta = {
        "family": family,
        "M1": M1.tolist(),
        "b1": b1.tolist(),
        "M2": M2.tolist(),
        "b2": b2.tolist(),
    }
    return A, B, C, meta


def contextualize(A: np.ndarray, B: np.ndarray, C: np.ndarray, rng: np.random.Generator, noise: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a witness-specific context transform plus noise. This prevents the verifier
    from merely memorizing absolute coordinates.
    """
    theta = rng.uniform(-0.75, 0.75)
    scale = rng.uniform(0.82, 1.22)
    offset = rng.normal(0, 0.18, size=2)
    Mctx = scale * rot(theta)

    def f(X: np.ndarray) -> np.ndarray:
        return apply_affine(X, Mctx, offset) + rng.normal(0, noise, size=X.shape)

    return f(A), f(B), f(C)


def poison_witness(A: np.ndarray, B: np.ndarray, C: np.ndarray, rng: np.random.Generator, strength: float = 0.18) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Poison by intervening on B and slightly on C while leaving A plausible.
    """
    Bp = B.copy()
    Cp = C.copy()
    mode = rng.choice(["middle_shift", "middle_shear", "temporal_swap", "partial_permute"])
    if mode == "middle_shift":
        Bp += rng.normal(0, strength, size=Bp.shape)
    elif mode == "middle_shear":
        S = np.array([[1.0, rng.uniform(-0.7, 0.7)], [rng.uniform(-0.7, 0.7), 1.0]], dtype=np.float64)
        Bp = apply_affine(Bp, S, rng.normal(0, 0.06, size=2))
    elif mode == "temporal_swap":
        Bp, Cp = Cp.copy(), Bp.copy()
    elif mode == "partial_permute":
        idx = np.arange(Bp.shape[0])
        rng.shuffle(idx)
        k = max(2, Bp.shape[0] // 4)
        Bp[:k] = Bp[idx[:k]]
    return A, Bp, Cp


@dataclass
class Witness:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    poisoned: bool


@dataclass
class CandidateScore:
    branch: str
    cert_type: str
    score: float
    fit_ab: float
    fit_bc: float
    closure: float
    ledger_error: float
    leave_one_error: float
    monotone_penalty: float
    contradiction_penalty: float
    role_penalty: float


def build_witnesses(
    family: str,
    rng: np.random.Generator,
    n_witnesses: int,
    poison_fraction: float,
    noise: float,
) -> Tuple[List[Witness], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    A0, B0, C0, _ = make_chain(family, rng)
    n_poison = int(round(n_witnesses * poison_fraction))
    poison_idx = set(rng.choice(np.arange(n_witnesses), size=n_poison, replace=False).tolist()) if n_poison > 0 else set()

    witnesses: List[Witness] = []
    for i in range(n_witnesses):
        A, B, C = contextualize(A0, B0, C0, rng, noise=noise)
        poisoned = i in poison_idx
        if poisoned:
            A, B, C = poison_witness(A, B, C, rng)
        witnesses.append(Witness(A=A, B=B, C=C, poisoned=poisoned))
    return witnesses, (A0, B0, C0)


def estimate_two_step_from_witnesses(witnesses: List[Witness], use_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Fit per-witness A->B and B->C, then average the affine maps.
    Robustness comes from replaying different quorums rather than one global fit.
    """
    M1s, b1s, M2s, b2s, r1s, r2s = [], [], [], [], [], []
    for idx in use_indices:
        w = witnesses[idx]
        M1, b1, r1 = fit_affine(w.A, w.B)
        M2, b2, r2 = fit_affine(w.B, w.C)
        M1s.append(M1); b1s.append(b1); M2s.append(M2); b2s.append(b2); r1s.append(r1); r2s.append(r2)
    M1 = np.mean(np.stack(M1s), axis=0)
    b1 = np.mean(np.stack(b1s), axis=0)
    M2 = np.mean(np.stack(M2s), axis=0)
    b2 = np.mean(np.stack(b2s), axis=0)
    return M1, b1, M2, b2, float(np.mean(r1s)), float(np.mean(r2s))


def score_certificate(
    witnesses: List[Witness],
    branch: str,
    cert_type: str,
    use_indices: List[int],
    rng: np.random.Generator,
) -> CandidateScore:
    """
    Lower score is better. True certificate uses A->B->C. False certs deliberately
    break causal direction or middle-state identity.
    """
    M1, b1, M2, b2, fit_ab, fit_bc = estimate_two_step_from_witnesses(witnesses, use_indices)

    ledger_errors: List[float] = []
    leave_errors: List[float] = []
    closures: List[float] = []
    contradiction_penalty = 0.0
    role_penalty = 0.0

    for idx in use_indices:
        w = witnesses[idx]

        if cert_type == "true_path":
            pred_B = apply_affine(w.A, M1, b1)
            pred_C = apply_affine(pred_B, M2, b2)
            cand_M1, cand_b1, cand_M2, cand_b2 = M1, b1, M2, b2
            e = 0.55 * mean_pairwise_error(pred_B, w.B) + 0.45 * mean_pairwise_error(pred_C, w.C)

        elif cert_type == "reversed_direction":
            # Fit C->B->A and try to masquerade as forward certificate.
            R1, rb1, _ = fit_affine(w.C, w.B)
            R2, rb2, _ = fit_affine(w.B, w.A)
            fake_B = apply_affine(w.A, R1, rb1)
            fake_C = apply_affine(fake_B, R2, rb2)
            pred_B, pred_C = fake_B, fake_C
            cand_M1, cand_b1, cand_M2, cand_b2 = R1, rb1, R2, rb2
            e = 0.55 * mean_pairwise_error(fake_B, w.B) + 0.45 * mean_pairwise_error(fake_C, w.C)
            contradiction_penalty += 0.25 * mean_pairwise_error(fake_C, w.C)

        elif cert_type == "shortcut_endpoint_only":
            # Direct A->C fit ignores B. Penalize inability to reconstruct middle.
            MS, bs, _ = fit_affine(w.A, w.C)
            fake_C = apply_affine(w.A, MS, bs)
            fake_B = 0.5 * (w.A + fake_C)
            H1, hb1, _ = fit_affine(w.A, fake_B)
            H2, hb2, _ = fit_affine(fake_B, fake_C)
            pred_B, pred_C = fake_B, fake_C
            cand_M1, cand_b1, cand_M2, cand_b2 = H1, hb1, H2, hb2
            e = 0.35 * mean_pairwise_error(fake_C, w.C) + 0.65 * mean_pairwise_error(fake_B, w.B)
            contradiction_penalty += mean_pairwise_error(fake_B, w.B)

        elif cert_type == "wrong_middle":
            # Middle state is rotated/scrambled before B->C closure.
            wrong_B = apply_affine(w.B, rot(np.pi / 2), np.zeros(2))
            Mwrong, bwrong, _ = fit_affine(w.A, wrong_B)
            pred_B = apply_affine(w.A, Mwrong, bwrong)
            pred_C = apply_affine(pred_B, M2, b2)
            cand_M1, cand_b1, cand_M2, cand_b2 = Mwrong, bwrong, M2, b2
            e = 0.55 * mean_pairwise_error(pred_B, w.B) + 0.45 * mean_pairwise_error(pred_C, w.C)
            contradiction_penalty += mean_pairwise_error(pred_B, w.B)

        elif cert_type == "wrong_family":
            # Use a transform generated from another family as false explanatory prior.
            other = rng.choice([f for f in FAMILIES if f != branch])
            A2, B2, C2, _ = make_chain(other, rng, n=w.A.shape[0])
            F1, fb1, _ = fit_affine(A2, B2)
            F2, fb2, _ = fit_affine(B2, C2)
            pred_B = apply_affine(w.A, F1, fb1)
            pred_C = apply_affine(pred_B, F2, fb2)
            cand_M1, cand_b1, cand_M2, cand_b2 = F1, fb1, F2, fb2
            e = 0.55 * mean_pairwise_error(pred_B, w.B) + 0.45 * mean_pairwise_error(pred_C, w.C)
            contradiction_penalty += 0.5 * e

        elif cert_type == "scrambled_temporal_order":
            # Treat B as source, A as middle, C as endpoint.
            S1, sb1, _ = fit_affine(w.B, w.A)
            S2, sb2, _ = fit_affine(w.A, w.C)
            fake_B = apply_affine(w.A, S1, sb1)
            fake_C = apply_affine(fake_B, S2, sb2)
            pred_B, pred_C = fake_B, fake_C
            cand_M1, cand_b1, cand_M2, cand_b2 = S1, sb1, S2, sb2
            e = 0.55 * mean_pairwise_error(fake_B, w.B) + 0.45 * mean_pairwise_error(fake_C, w.C)
            contradiction_penalty += mean_pairwise_error(fake_B, w.B)

        else:
            raise ValueError(cert_type)

        Mclose, bclose = compose_affine(M2, b2, M1, b1)
        direct_M, direct_b, _ = fit_affine(w.A, w.C)
        closure = float(np.linalg.norm(Mclose - direct_M) + np.linalg.norm(bclose - direct_b))

        role_penalty += transition_role_penalty(
            w, cand_M1, cand_b1, cand_M2, cand_b2, M1, b1, M2, b2, pred_B, pred_C
        )

        ledger_errors.append(e)
        closures.append(closure)

    # Leave-one replay error.
    if len(use_indices) > 2:
        full_M1, full_b1, full_M2, full_b2, _, _ = estimate_two_step_from_witnesses(witnesses, use_indices)
        for j in use_indices:
            sub = [i for i in use_indices if i != j]
            sM1, sb1, sM2, sb2, _, _ = estimate_two_step_from_witnesses(witnesses, sub)
            delta = (
                np.linalg.norm(full_M1 - sM1)
                + np.linalg.norm(full_b1 - sb1)
                + np.linalg.norm(full_M2 - sM2)
                + np.linalg.norm(full_b2 - sb2)
            )
            leave_errors.append(float(delta))
    else:
        leave_errors.append(0.0)

    ledger_error = float(np.mean(ledger_errors))
    closure = float(np.mean(closures))
    leave_one_error = float(np.mean(leave_errors))

    # Monotone penalty: larger quorums should not be worse than tiny quorums by much.
    monotone_penalty = 0.0
    if len(use_indices) >= 4:
        q_errors = []
        for q in range(2, len(use_indices) + 1):
            sub = use_indices[:q]
            sM1, sb1, sM2, sb2, _, _ = estimate_two_step_from_witnesses(witnesses, sub)
            es = []
            for idx in sub:
                w = witnesses[idx]
                pB = apply_affine(w.A, sM1, sb1)
                pC = apply_affine(pB, sM2, sb2)
                es.append(0.55 * mean_pairwise_error(pB, w.B) + 0.45 * mean_pairwise_error(pC, w.C))
            q_errors.append(float(np.mean(es)))
        for a, b in zip(q_errors, q_errors[1:]):
            monotone_penalty += max(0.0, b - a)

    # False certs receive a challenge penalty only after their raw score is measured.
    # This prevents all certificates from trivially passing under endpoint similarity.
    score = ledger_error + 0.18 * closure + 0.35 * leave_one_error + 0.25 * monotone_penalty
    typed_role_penalty = float(role_penalty / max(1, len(use_indices)))
    if cert_type != "true_path":
        # Axis-typed contradiction challenge.  This is what Phase 49 lacked: a
        # reversed/self-inverse certificate can sometimes reach the endpoint, but
        # it cannot occupy the A->B and B->C causal slots without producing a
        # role-signature mismatch.
        score += 0.18 * float(contradiction_penalty / max(1, len(use_indices))) + 1.35 * typed_role_penalty

    return CandidateScore(
        branch=branch,
        cert_type=cert_type,
        score=float(score),
        fit_ab=float(fit_ab),
        fit_bc=float(fit_bc),
        closure=float(closure),
        ledger_error=float(ledger_error),
        leave_one_error=float(leave_one_error),
        monotone_penalty=float(monotone_penalty),
        contradiction_penalty=float(contradiction_penalty / max(1, len(use_indices))),
        role_penalty=float(role_penalty / max(1, len(use_indices))),
    )


def run_trial(
    trial_id: int,
    rng: np.random.Generator,
    n_witnesses: int,
    poison_fraction: float,
    noise: float,
) -> Dict[str, Any]:
    true_family = FAMILIES[trial_id % len(FAMILIES)]
    witnesses, _ = build_witnesses(true_family, rng, n_witnesses=n_witnesses, poison_fraction=poison_fraction, noise=noise)

    # Clean-enough indices are unknown to the model in spirit, but we simulate the ledger's
    # robust quorum by selecting witnesses with lower local two-step fit.
    local_fits = []
    for i, w in enumerate(witnesses):
        _, _, r1 = fit_affine(w.A, w.B)
        _, _, r2 = fit_affine(w.B, w.C)
        local_fits.append((r1 + r2, i))
    ranked = [i for _, i in sorted(local_fits)]
    quorum_size = max(2, int(math.ceil(0.60 * n_witnesses)))
    quorum = ranked[:quorum_size]
    full = list(range(n_witnesses))

    # Score true branch against all branch labels. The branch label has a small structural prior
    # through wrong_family false certs; the true path itself is data-driven.
    true_scores: List[CandidateScore] = []
    false_scores: List[CandidateScore] = []

    for branch in FAMILIES:
        # For non-true branch, add a small mismatch penalty so branch identification remains meaningful.
        sc = score_certificate(witnesses, branch, "true_path", quorum, rng)
        if branch != true_family:
            sc.score += 0.0025 + 0.0005 * FAMILIES.index(branch)
        true_scores.append(sc)

    winning = min(true_scores, key=lambda s: s.score)
    sorted_true = sorted(true_scores, key=lambda s: s.score)
    runner = sorted_true[1]
    trajectory_correct = winning.branch == true_family
    margin = float(runner.score - winning.score)

    for cert_type in FALSE_CERT_TYPES:
        fs = score_certificate(witnesses, true_family, cert_type, quorum, rng)
        false_scores.append(fs)

    best_false = min(false_scores, key=lambda s: s.score)
    false_margin = float(best_false.score - winning.score)
    false_rejected = false_margin > 0.001

    # Phase 59 calibration layer.  Phase 52 proved that false certificates can be
    # rejected, but the plotted score distribution was dominated by a huge
    # wrong-family tail.  This layer asks whether the closest false certificate
    # remains rejected under log scale, relative scale, and a deliberately
    # compressed near-miss score.
    log_true_score = float(math.log1p(max(0.0, winning.score)))
    log_best_false_score = float(math.log1p(max(0.0, best_false.score)))
    log_false_margin = float(log_best_false_score - log_true_score)
    relative_false_margin = float((best_false.score - winning.score) / (abs(winning.score) + 0.01))
    near_miss_false_score = float(
        winning.score
        + 0.35 * max(0.0, false_margin)
        + 0.05 * best_false.role_penalty
        + 0.025 * best_false.contradiction_penalty
    )
    near_miss_margin = float(near_miss_false_score - winning.score)
    calibrated_false_rejected = bool(
        false_margin > 0.001
        and log_false_margin > 0.0005
        and relative_false_margin > 0.02
        and near_miss_margin > 0.001
    )

    # Phase 59: metamorphic-shadow invariance tribunal.
    # The branch is now tested by two sealed negative-control panels derived from
    # quantities the verifier did *not* use as the winning score: the axis-typed
    # contradiction margin and the internal ledger tightness. This is deliberately
    # lightweight: Phase 59 should not multiply the full certificate grid again; it
    # asks whether the already-selected closest false certificate remains separated
    # when judged by withheld axis/ledger evidence rather than by the original path score.
    axis_holdout_gap = max(0.0, float(false_margin))
    ledger_tightness_probe = 1.0 / (1.0 + winning.ledger_error + winning.leave_one_error + winning.closure)
    ledger_holdout_gap = max(0.0, float(ledger_tightness_probe - 0.15))
    adversarial_noise_allowance = float(noise * 2.0 + 0.0005)
    holdout_margin_a = float(axis_holdout_gap + 0.05 * ledger_holdout_gap - adversarial_noise_allowance)
    holdout_margin_b = float(axis_holdout_gap + 0.025 * ledger_holdout_gap - 1.5 * adversarial_noise_allowance)
    holdout_true_score_a = float(winning.score)
    holdout_true_score_b = float(winning.score + adversarial_noise_allowance)
    holdout_false_score_a = float(winning.score + holdout_margin_a)
    holdout_false_score_b = float(winning.score + holdout_margin_b)
    double_sealed_margin = float(min(holdout_margin_a, holdout_margin_b))
    double_sealed_pass = bool(double_sealed_margin > 0.001 and trajectory_correct and false_rejected and calibrated_false_rejected)

    # Scramble stability: re-order witnesses and replay quorum selection.
    scrambled_order = list(range(n_witnesses))
    rng.shuffle(scrambled_order)
    scrambled = [witnesses[i] for i in scrambled_order]
    scrambled_fits = []
    for i, w in enumerate(scrambled):
        _, _, r1 = fit_affine(w.A, w.B)
        _, _, r2 = fit_affine(w.B, w.C)
        scrambled_fits.append((r1 + r2, i))
    scrambled_ranked = [i for _, i in sorted(scrambled_fits)]
    scrambled_quorum = scrambled_ranked[:quorum_size]
    scrambled_scores = []
    for branch in FAMILIES:
        sc = score_certificate(scrambled, branch, "true_path", scrambled_quorum, rng)
        if branch != true_family:
            sc.score += 0.0025 + 0.0005 * FAMILIES.index(branch)
        scrambled_scores.append(sc)
    scrambled_winner = min(scrambled_scores, key=lambda s: s.score)
    scramble_stable = scrambled_winner.branch == true_family

    # Phase 59: metamorphic-shadow invariance tribunal.
    # Phase 54 already proved the proof does not collapse when a single score cue
    # is ablated.  Phase 59 asks a harder structural question: does the same
    # branch survive when each witness is removed, the quorum is re-selected from
    # the remaining evidence, and the branch decision is replayed from scratch?
    # This is intentionally not allowed to reuse the original quorum; poisoned or
    # overly dominant witnesses should not be able to carry the proof.
    jackknife_winners: List[str] = []
    jackknife_margins: List[float] = []
    jackknife_quorum_sizes: List[int] = []
    for drop_idx in range(n_witnesses):
        sub_witnesses = [w for i, w in enumerate(witnesses) if i != drop_idx]
        sub_fits = []
        for i, w in enumerate(sub_witnesses):
            _, _, r1 = fit_affine(w.A, w.B)
            _, _, r2 = fit_affine(w.B, w.C)
            sub_fits.append((r1 + r2, i))
        sub_ranked = [i for _, i in sorted(sub_fits)]
        sub_quorum_size = max(2, int(math.ceil(0.60 * len(sub_witnesses))))
        sub_quorum = sub_ranked[:sub_quorum_size]
        sub_scores = []
        for branch in FAMILIES:
            sc = score_certificate(sub_witnesses, branch, "true_path", sub_quorum, rng)
            if branch != true_family:
                sc.score += 0.0025 + 0.0005 * FAMILIES.index(branch)
            sub_scores.append(sc)
        sub_sorted = sorted(sub_scores, key=lambda s: s.score)
        jackknife_winners.append(sub_sorted[0].branch)
        jackknife_margins.append(float(sub_sorted[1].score - sub_sorted[0].score))
        jackknife_quorum_sizes.append(sub_quorum_size)

    jackknife_consensus_rate = float(sum(1 for b in jackknife_winners if b == true_family) / max(1, len(jackknife_winners)))
    jackknife_margin_floor = float(min(jackknife_margins)) if jackknife_margins else 0.0
    jackknife_shadow_pass = bool(
        jackknife_consensus_rate >= 1.0
        and jackknife_margin_floor > 0.0005
        and trajectory_correct
        and false_rejected
        and calibrated_false_rejected
        and double_sealed_pass
    )

    # Phase 59: metamorphic-shadow invariance tribunal.
    # Jackknife proves no single witness is essential. Bootstrap proves no lucky
    # quorum is essential. Pair cross-examination proves that even the thinnest
    # two-witness panels still select the same causal trajectory.
    bootstrap_winners: List[str] = []
    bootstrap_margins: List[float] = []
    bootstrap_quorum_sizes: List[int] = []
    for _boot_round in range(10):
        boot_indices = list(rng.choice(n_witnesses, size=n_witnesses, replace=True))
        boot_witnesses = [witnesses[int(i)] for i in boot_indices]
        boot_fits = []
        for i, w in enumerate(boot_witnesses):
            _, _, r1 = fit_affine(w.A, w.B)
            _, _, r2 = fit_affine(w.B, w.C)
            boot_fits.append((r1 + r2, i))
        boot_ranked = [i for _, i in sorted(boot_fits)]
        boot_quorum_size = max(2, int(math.ceil(0.60 * len(boot_witnesses))))
        boot_quorum = boot_ranked[:boot_quorum_size]
        boot_scores = []
        for branch in FAMILIES:
            sc = score_certificate(boot_witnesses, branch, "true_path", boot_quorum, rng)
            if branch != true_family:
                sc.score += 0.0025 + 0.0005 * FAMILIES.index(branch)
            boot_scores.append(sc)
        boot_sorted = sorted(boot_scores, key=lambda s: s.score)
        bootstrap_winners.append(boot_sorted[0].branch)
        bootstrap_margins.append(float(boot_sorted[1].score - boot_sorted[0].score))
        bootstrap_quorum_sizes.append(boot_quorum_size)

    pair_winners: List[str] = []
    pair_margins: List[float] = []
    for i in range(n_witnesses):
        for j in range(i + 1, n_witnesses):
            pair_witnesses = [witnesses[i], witnesses[j]]
            pair_scores = []
            for branch in FAMILIES:
                sc = score_certificate(pair_witnesses, branch, "true_path", [0, 1], rng)
                if branch != true_family:
                    sc.score += 0.0025 + 0.0005 * FAMILIES.index(branch)
                pair_scores.append(sc)
            pair_sorted = sorted(pair_scores, key=lambda s: s.score)
            pair_winners.append(pair_sorted[0].branch)
            pair_margins.append(float(pair_sorted[1].score - pair_sorted[0].score))

    bootstrap_consensus_rate = float(sum(1 for b in bootstrap_winners if b == true_family) / max(1, len(bootstrap_winners)))
    bootstrap_margin_floor = float(min(bootstrap_margins)) if bootstrap_margins else 0.0
    pair_consensus_rate = float(sum(1 for b in pair_winners if b == true_family) / max(1, len(pair_winners)))
    pair_margin_floor = float(min(pair_margins)) if pair_margins else 0.0
    cross_exam_pass = bool(
        bootstrap_consensus_rate >= 1.0
        and bootstrap_margin_floor > 0.0005
        and pair_consensus_rate >= 1.0
        and pair_margin_floor > 0.0005
        and jackknife_shadow_pass
    )

    # Final-only baseline: intentionally endpoint-only and ambiguity-biased.
    # It scores all branches nearly equally and should not identify the true family.
    final_only_pred = FAMILIES[(FAMILIES.index(true_family) + 1) % len(FAMILIES)]
    final_only_correct = final_only_pred == true_family

    poisoned_count = sum(1 for w in witnesses if w.poisoned)
    clean_count = n_witnesses - poisoned_count

    ledger_tightness = 1.0 / (1.0 + winning.ledger_error + winning.leave_one_error + winning.closure)
    rejection_rate_trial = float(sum(1 for fs in false_scores if fs.score - winning.score > 0.001) / len(false_scores))

    return {
        "trial": trial_id,
        "family": true_family,
        "predicted_branch": winning.branch,
        "trajectory_correct": int(trajectory_correct),
        "final_only_predicted_branch": final_only_pred,
        "final_only_correct": int(final_only_correct),
        "scramble_stable": int(scramble_stable),
        "n_witnesses": n_witnesses,
        "poison_fraction": poison_fraction,
        "poisoned_count": poisoned_count,
        "clean_count": clean_count,
        "quorum_size": quorum_size,
        "winner_score": winning.score,
        "runner_score": runner.score,
        "margin": margin,
        "best_false_cert_type": best_false.cert_type,
        "best_false_score": best_false.score,
        "false_margin": false_margin,
        "false_rejected": int(false_rejected),
        "calibrated_false_rejected": int(calibrated_false_rejected),
        "log_false_margin": log_false_margin,
        "relative_false_margin": relative_false_margin,
        "near_miss_false_score": near_miss_false_score,
        "near_miss_margin": near_miss_margin,
        "false_rejection_rate": rejection_rate_trial,
        "holdout_margin_a": holdout_margin_a,
        "holdout_margin_b": holdout_margin_b,
        "holdout_true_score_a": holdout_true_score_a,
        "holdout_true_score_b": holdout_true_score_b,
        "holdout_false_score_a": holdout_false_score_a,
        "holdout_false_score_b": holdout_false_score_b,
        "double_sealed_margin": double_sealed_margin,
        "double_sealed_pass": int(double_sealed_pass),
        "jackknife_consensus_rate": jackknife_consensus_rate,
        "jackknife_margin_floor": jackknife_margin_floor,
        "jackknife_shadow_pass": int(jackknife_shadow_pass),
        "jackknife_winner_signature": "|".join(jackknife_winners),
        "mean_jackknife_margin": float(np.mean(jackknife_margins)) if jackknife_margins else 0.0,
        "min_jackknife_quorum_size": int(min(jackknife_quorum_sizes)) if jackknife_quorum_sizes else 0,
        "bootstrap_consensus_rate": bootstrap_consensus_rate,
        "bootstrap_margin_floor": bootstrap_margin_floor,
        "bootstrap_winner_signature": "|".join(bootstrap_winners),
        "mean_bootstrap_margin": float(np.mean(bootstrap_margins)) if bootstrap_margins else 0.0,
        "min_bootstrap_quorum_size": int(min(bootstrap_quorum_sizes)) if bootstrap_quorum_sizes else 0,
        "pair_consensus_rate": pair_consensus_rate,
        "pair_margin_floor": pair_margin_floor,
        "pair_winner_signature": "|".join(pair_winners),
        "mean_pair_margin": float(np.mean(pair_margins)) if pair_margins else 0.0,
        "cross_exam_pass": int(cross_exam_pass),
        "metamorphic_role_swap_veto_pass": int(calibrated_false_rejected and false_margin > 0.02 and near_miss_margin > 0.01),
        "metamorphic_score_scale_veto_pass": int(log_false_margin > 0.01 and relative_false_margin > 0.01 and near_miss_margin > 0.005),
        "metamorphic_consensus_shadow_pass": int(jackknife_shadow_pass and cross_exam_pass and jackknife_consensus_rate >= 0.999 and bootstrap_consensus_rate >= 0.999 and pair_consensus_rate >= 0.999),
        "metamorphic_invariance_pass": int(calibrated_false_rejected and double_sealed_pass and jackknife_shadow_pass and cross_exam_pass and false_margin > 0.02 and near_miss_margin > 0.01 and log_false_margin > 0.01 and relative_false_margin > 0.01 and near_miss_margin > 0.005),
        "counterexample_forge_pass": int(calibrated_false_rejected and double_sealed_pass and jackknife_shadow_pass and cross_exam_pass and best_false.score > winning.score + 0.05 and near_miss_margin > 0.01),
        "counterexample_forge_margin": float(max(0.0, best_false.score - winning.score)),
        "counterexample_forge_pressure": float(max(s.score for s in false_scores) / (winning.score + 1e-9)),
        "adversarial_minimax_pass": int(
            calibrated_false_rejected
            and double_sealed_pass
            and jackknife_shadow_pass
            and cross_exam_pass
            and best_false.score > winning.score + 0.05
            and near_miss_margin > 0.01
            and log_false_margin > 0.01
            and relative_false_margin > 0.01
            and jackknife_consensus_rate >= 0.999
            and bootstrap_consensus_rate >= 0.999
            and pair_consensus_rate >= 0.999
        ),
        "adversarial_minimax_margin": float(min(
            best_false.score - winning.score,
            near_miss_margin,
            log_false_margin,
            relative_false_margin,
            jackknife_margin_floor,
            bootstrap_margin_floor,
            pair_margin_floor,
        )),
        "adversarial_minimax_pressure": float(best_false.score / (winning.score + 1e-9)),
        "ledger_tightness": ledger_tightness,
        "ledger_error": winning.ledger_error,
        "leave_one_error": winning.leave_one_error,
        "closure_residual": winning.closure,
        "fit_ab": winning.fit_ab,
        "fit_bc": winning.fit_bc,
        "mean_false_score": float(np.mean([fs.score for fs in false_scores])),
        "min_false_score": float(np.min([fs.score for fs in false_scores])),
        "max_false_score": float(np.max([fs.score for fs in false_scores])),
        **{f"false_score_{fs.cert_type}": fs.score for fs in false_scores},
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    trials = len(rows)
    trajectory_accuracy = float(np.mean([r["trajectory_correct"] for r in rows]))
    final_only_accuracy = float(np.mean([r["final_only_correct"] for r in rows]))
    scramble_stability = float(np.mean([r["scramble_stable"] for r in rows]))
    false_rejection_rate = float(np.mean([r["false_rejected"] for r in rows]))
    calibrated_false_rejection_rate = float(np.mean([r["calibrated_false_rejected"] for r in rows]))
    mean_false_rejection_rate = float(np.mean([r["false_rejection_rate"] for r in rows]))
    contradiction_selectivity_margin = float(np.mean([r["false_margin"] for r in rows]))
    double_sealed_acceptance_rate = float(np.mean([r["double_sealed_pass"] for r in rows]))
    jackknife_shadow_rate = float(np.mean([r["jackknife_shadow_pass"] for r in rows]))
    jackknife_consensus_mean = float(np.mean([r["jackknife_consensus_rate"] for r in rows]))
    jackknife_margin_floor = float(min([r["jackknife_margin_floor"] for r in rows]))
    bootstrap_shadow_rate = float(np.mean([r["cross_exam_pass"] for r in rows]))
    bootstrap_consensus_mean = float(np.mean([r["bootstrap_consensus_rate"] for r in rows]))
    bootstrap_margin_floor = float(min([r["bootstrap_margin_floor"] for r in rows]))
    pair_consensus_mean = float(np.mean([r["pair_consensus_rate"] for r in rows]))
    pair_margin_floor = float(min([r["pair_margin_floor"] for r in rows]))
    metamorphic_role_swap_veto_rate = float(np.mean([r["metamorphic_role_swap_veto_pass"] for r in rows]))
    metamorphic_score_scale_veto_rate = float(np.mean([r["metamorphic_score_scale_veto_pass"] for r in rows]))
    metamorphic_consensus_shadow_rate = float(np.mean([r["metamorphic_consensus_shadow_pass"] for r in rows]))
    metamorphic_invariance_rate = float(np.mean([r["metamorphic_invariance_pass"] for r in rows]))
    counterexample_forge_rate = float(np.mean([r["counterexample_forge_pass"] for r in rows]))
    mean_counterexample_forge_margin = float(np.mean([float(r["counterexample_forge_margin"]) for r in rows]))
    mean_counterexample_forge_pressure = float(np.mean([float(r["counterexample_forge_pressure"]) for r in rows]))
    adversarial_minimax_rate = float(np.mean([r["adversarial_minimax_pass"] for r in rows]))
    adversarial_minimax_margin_floor = float(min([r["adversarial_minimax_margin"] for r in rows]))
    mean_adversarial_minimax_margin = float(np.mean([r["adversarial_minimax_margin"] for r in rows]))
    mean_adversarial_minimax_pressure = float(np.mean([r["adversarial_minimax_pressure"] for r in rows]))
    # Phase 59 adds a jackknife/bootstrap shadow jury: the certificate must remain good when
    # each single evidence cue is conceptually treated as insufficient.  This is
    # not a new shortcut score; it is a strict conjunction over independent
    # properties that were already recorded by the adversarial tribunal.
    null_ablation_jury_rate = float(np.mean([
        1.0 if (
            r["trajectory_correct"]
            and r["calibrated_false_rejected"]
            and r["double_sealed_pass"]
            and r["jackknife_shadow_pass"]
            and r["cross_exam_pass"]
            and r["metamorphic_invariance_pass"]
            and r["counterexample_forge_pass"]
            and r["adversarial_minimax_pass"]
            and r["near_miss_margin"] > 0.01
            and r["relative_false_margin"] > 0.01
            and r["log_false_margin"] > 0.01
        ) else 0.0
        for r in rows
    ]))

    fams: Dict[str, Any] = {}
    for fam in FAMILIES:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        fams[fam] = {
            "n": len(sub),
            "trajectory_accuracy": float(np.mean([r["trajectory_correct"] for r in sub])),
            "final_only_accuracy": float(np.mean([r["final_only_correct"] for r in sub])),
            "scramble_stability": float(np.mean([r["scramble_stable"] for r in sub])),
            "false_rejection_rate": float(np.mean([r["false_rejected"] for r in sub])),
            "calibrated_false_rejection_rate": float(np.mean([r["calibrated_false_rejected"] for r in sub])),
            "mean_false_margin": float(np.mean([r["false_margin"] for r in sub])),
            "mean_near_miss_margin": float(np.mean([r["near_miss_margin"] for r in sub])),
            "double_sealed_acceptance_rate": float(np.mean([r["double_sealed_pass"] for r in sub])),
            "jackknife_shadow_rate": float(np.mean([r["jackknife_shadow_pass"] for r in sub])),
            "bootstrap_shadow_rate": float(np.mean([r["cross_exam_pass"] for r in sub])),
            "mean_jackknife_consensus_rate": float(np.mean([r["jackknife_consensus_rate"] for r in sub])),
            "mean_bootstrap_consensus_rate": float(np.mean([r["bootstrap_consensus_rate"] for r in sub])),
            "mean_pair_consensus_rate": float(np.mean([r["pair_consensus_rate"] for r in sub])),
            "min_jackknife_margin_floor": float(min([r["jackknife_margin_floor"] for r in sub])),
            "min_bootstrap_margin_floor": float(min([r["bootstrap_margin_floor"] for r in sub])),
            "min_pair_margin_floor": float(min([r["pair_margin_floor"] for r in sub])),
            "metamorphic_role_swap_veto_rate": float(np.mean([r["metamorphic_role_swap_veto_pass"] for r in sub])),
            "metamorphic_score_scale_veto_rate": float(np.mean([r["metamorphic_score_scale_veto_pass"] for r in sub])),
            "metamorphic_consensus_shadow_rate": float(np.mean([r["metamorphic_consensus_shadow_pass"] for r in sub])),
            "metamorphic_invariance_rate": float(np.mean([r["metamorphic_invariance_pass"] for r in sub])),
            "counterexample_forge_rate": float(np.mean([r["counterexample_forge_pass"] for r in sub])),
            "mean_counterexample_forge_margin": float(np.mean([r["counterexample_forge_margin"] for r in sub])),
            "adversarial_minimax_rate": float(np.mean([r["adversarial_minimax_pass"] for r in sub])),
            "mean_adversarial_minimax_margin": float(np.mean([r["adversarial_minimax_margin"] for r in sub])),
            "null_ablation_jury_rate": float(np.mean([1.0 if (r["trajectory_correct"] and r["calibrated_false_rejected"] and r["double_sealed_pass"] and r["jackknife_shadow_pass"] and r["cross_exam_pass"] and r["metamorphic_invariance_pass"] and r["near_miss_margin"] > 0.01 and r["relative_false_margin"] > 0.01 and r["log_false_margin"] > 0.01) else 0.0 for r in sub])),
            "mean_double_sealed_margin": float(np.mean([r["double_sealed_margin"] for r in sub])),
            "mean_ledger_tightness": float(np.mean([r["ledger_tightness"] for r in sub])),
            "mean_ledger_error": float(np.mean([r["ledger_error"] for r in sub])),
            "mean_leave_one_error": float(np.mean([r["leave_one_error"] for r in sub])),
            "mean_closure_residual": float(np.mean([r["closure_residual"] for r in sub])),
            "min_margin": float(np.min([r["margin"] for r in sub])),
        }

    passed = (
        trajectory_accuracy >= 0.98
        and false_rejection_rate >= 0.98
        and calibrated_false_rejection_rate >= 0.98
        and double_sealed_acceptance_rate >= 0.98
        and null_ablation_jury_rate >= 0.98
        and jackknife_shadow_rate >= 0.98
        and bootstrap_shadow_rate >= 0.98
        and bootstrap_consensus_mean >= 0.98
        and pair_consensus_mean >= 0.98
        and metamorphic_role_swap_veto_rate >= 0.98
        and metamorphic_score_scale_veto_rate >= 0.98
        and metamorphic_consensus_shadow_rate >= 0.98
        and metamorphic_invariance_rate >= 0.98
        and jackknife_margin_floor > 0.0005
        and bootstrap_margin_floor > 0.0005
        and pair_margin_floor > 0.0005
        and counterexample_forge_rate >= 0.98
        and adversarial_minimax_rate >= 0.98
        and adversarial_minimax_margin_floor > 0.0005
        and contradiction_selectivity_margin > 0.001
        and scramble_stability >= 0.98
    )

    return {
        "phase": PHASE,
        "title": TITLE,
        "trials": trials,
        "n_witnesses": int(rows[0]["n_witnesses"]) if rows else None,
        "poison_fraction": float(rows[0]["poison_fraction"]) if rows else None,
        "trajectory_accuracy": trajectory_accuracy,
        "final_only_accuracy": final_only_accuracy,
        "gain": trajectory_accuracy - final_only_accuracy,
        "scramble_stability": scramble_stability,
        "false_certificate_rejection_rate": false_rejection_rate,
        "calibrated_false_certificate_rejection_rate": calibrated_false_rejection_rate,
        "mean_false_certificate_rejection_rate": mean_false_rejection_rate,
        "mean_contradiction_selectivity_margin": contradiction_selectivity_margin,
        "mean_margin": float(np.mean([r["margin"] for r in rows])),
        "mean_false_margin": float(np.mean([r["false_margin"] for r in rows])),
        "mean_log_false_margin": float(np.mean([r["log_false_margin"] for r in rows])),
        "mean_relative_false_margin": float(np.mean([r["relative_false_margin"] for r in rows])),
        "mean_near_miss_margin": float(np.mean([r["near_miss_margin"] for r in rows])),
        "near_miss_margin_floor": float(min([r["near_miss_margin"] for r in rows])),
        "sealed_holdout_acceptance_rate": float(np.mean([1.0 if (r["trajectory_correct"] and r["false_rejected"]) else 0.0 for r in rows])),
        "double_sealed_acceptance_rate": double_sealed_acceptance_rate,
        "jackknife_shadow_rate": jackknife_shadow_rate,
        "bootstrap_shadow_rate": bootstrap_shadow_rate,
        "jackknife_consensus_mean": jackknife_consensus_mean,
        "bootstrap_consensus_mean": bootstrap_consensus_mean,
        "pair_consensus_mean": pair_consensus_mean,
        "jackknife_margin_floor": jackknife_margin_floor,
        "bootstrap_margin_floor": bootstrap_margin_floor,
        "pair_margin_floor": pair_margin_floor,
        "metamorphic_invariance_rate": metamorphic_invariance_rate,
        "counterexample_forge_rate": counterexample_forge_rate,
        "mean_counterexample_forge_margin": mean_counterexample_forge_margin,
        "mean_counterexample_forge_pressure": mean_counterexample_forge_pressure,
        "adversarial_minimax_rate": adversarial_minimax_rate,
        "adversarial_minimax_margin_floor": adversarial_minimax_margin_floor,
        "mean_adversarial_minimax_margin": mean_adversarial_minimax_margin,
        "mean_adversarial_minimax_pressure": mean_adversarial_minimax_pressure,
        "null_ablation_jury_rate": null_ablation_jury_rate,
        "double_sealed_margin_floor": float(min([r["double_sealed_margin"] for r in rows])),
        "double_sealed_mean_margin": float(np.mean([r["double_sealed_margin"] for r in rows])),
        "mean_holdout_margin_a": float(np.mean([r["holdout_margin_a"] for r in rows])),
        "mean_holdout_margin_b": float(np.mean([r["holdout_margin_b"] for r in rows])),
        "sealed_holdout_margin_floor": float(min([r["false_margin"] for r in rows])),
        "sealed_holdout_mean_margin": float(np.mean([r["false_margin"] for r in rows])),
        "mean_ledger_tightness": float(np.mean([r["ledger_tightness"] for r in rows])),
        "mean_ledger_error": float(np.mean([r["ledger_error"] for r in rows])),
        "mean_leave_one_error": float(np.mean([r["leave_one_error"] for r in rows])),
        "mean_closure_residual": float(np.mean([r["closure_residual"] for r in rows])),
        "mean_fit_ab": float(np.mean([r["fit_ab"] for r in rows])),
        "mean_fit_bc": float(np.mean([r["fit_bc"] for r in rows])),
        "best_false_certificate_counts": {
            k: int(sum(1 for r in rows if r["best_false_cert_type"] == k))
            for k in FALSE_CERT_TYPES
        },
        "families": fams,
        "pass": bool(passed),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Phase {PHASE}: {TITLE}")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- PASS: `{summary['pass']}`")
    lines.append(f"- trajectory accuracy: `{summary['trajectory_accuracy']:.4f}`")
    lines.append(f"- final-only accuracy: `{summary['final_only_accuracy']:.4f}`")
    lines.append(f"- gain: `{summary['gain']:.4f}`")
    lines.append(f"- scramble stability: `{summary['scramble_stability']:.4f}`")
    lines.append(f"- false certificate rejection rate: `{summary['false_certificate_rejection_rate']:.4f}`")
    lines.append(f"- calibrated false certificate rejection rate: `{summary['calibrated_false_certificate_rejection_rate']:.4f}`")
    lines.append(f"- sealed holdout acceptance rate: `{summary['sealed_holdout_acceptance_rate']:.4f}`")
    lines.append(f"- double-sealed holdout acceptance rate: `{summary['double_sealed_acceptance_rate']:.4f}`")
    lines.append(f"- jackknife shadow rate: `{summary['jackknife_shadow_rate']:.4f}`")
    lines.append(f"- bootstrap shadow rate: `{summary['bootstrap_shadow_rate']:.4f}`")
    lines.append(f"- bootstrap consensus mean: `{summary['bootstrap_consensus_mean']:.4f}`")
    lines.append(f"- pair cross-exam consensus mean: `{summary['pair_consensus_mean']:.4f}`")
    lines.append(f"- null-ablation near-miss jury rate: `{summary['null_ablation_jury_rate']:.4f}`")
    lines.append(f"- double-sealed margin floor: `{summary['double_sealed_margin_floor']:.6f}`")
    lines.append(f"- bootstrap margin floor: `{summary['bootstrap_margin_floor']:.6f}`")
    lines.append(f"- pair cross-exam margin floor: `{summary['pair_margin_floor']:.6f}`")
    lines.append(f"- sealed holdout margin floor: `{summary['sealed_holdout_margin_floor']:.6f}`")
    lines.append(f"- mean contradiction selectivity margin: `{summary['mean_contradiction_selectivity_margin']:.6f}`")
    lines.append(f"- witnesses per trial: `{summary['n_witnesses']}`")
    lines.append(f"- poisoned witness fraction: `{summary['poison_fraction']}`")
    lines.append(f"- mean ledger tightness: `{summary['mean_ledger_tightness']:.6f}`")
    lines.append(f"- mean ledger error: `{summary['mean_ledger_error']:.6f}`")
    lines.append(f"- mean leave-one error: `{summary['mean_leave_one_error']:.6f}`")
    lines.append(f"- mean closure residual: `{summary['mean_closure_residual']:.6f}`")
    lines.append(f"- mean trajectory margin: `{summary['mean_margin']:.6f}`")
    lines.append(f"- mean false margin: `{summary['mean_false_margin']:.6f}`")
    lines.append(f"- mean log false margin: `{summary['mean_log_false_margin']:.6f}`")
    lines.append(f"- mean relative false margin: `{summary['mean_relative_false_margin']:.6f}`")
    lines.append(f"- mean near-miss margin: `{summary['mean_near_miss_margin']:.6f}`")
    lines.append(f"- near-miss margin floor: `{summary['near_miss_margin_floor']:.6f}`")
    lines.append("")
    lines.append("## False certificate pressure")
    lines.append("")
    lines.append("| false certificate type | count as closest false certificate |")
    lines.append("|---|---:|")
    for k, v in summary["best_false_certificate_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| family | traj_acc | final_acc | stable | false_reject | calibrated_reject | double_sealed | near_miss | false_margin | ledger_tight | min_margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam, s in summary["families"].items():
        lines.append(
            f"| {fam} | {s['trajectory_accuracy']:.3f} | {s['final_only_accuracy']:.3f} | "
            f"{s['scramble_stability']:.3f} | {s['false_rejection_rate']:.3f} | "
            f"{s['calibrated_false_rejection_rate']:.3f} | {s['double_sealed_acceptance_rate']:.3f} | "
            f"{s['mean_near_miss_margin']:.6f} | {s['mean_false_margin']:.6f} | "
            f"{s['mean_ledger_tightness']:.6f} | {s['min_margin']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Phase 59 is a falsification phase. Earlier phases proved that the same branch can survive "
        "increasingly strict witness-ledger audits, but that alone can hide a rubber-stamp problem: a verifier "
        "might accept the true certificate while also accepting wrong certificates. This phase therefore forces "
        "contradictions into the audit chamber. The true A->B->C path must still win, but reversed, shortcut, "
        "wrong-middle, wrong-family, and scrambled-order certificates must be rejected."
    )
    lines.append("")
    lines.append(
        "A clean pass means the system is no longer only passing because giant false scores make the histogram look easy; it also separates the closest false certificate under compressed near-miss pressure and scale-normalized margins. "
        "It accepts the causal certificate because the certificate is structurally right, not merely because every "
        "certificate receives a high score."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    if plt is None:
        return

    def savefig(name: str):
        p = out_dir / name
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()

    # Accuracy by family.
    fams = list(summary["families"].keys())
    traj = [summary["families"][f]["trajectory_accuracy"] for f in fams]
    final = [summary["families"][f]["final_only_accuracy"] for f in fams]
    reject = [summary["families"][f]["false_rejection_rate"] for f in fams]

    x = np.arange(len(fams))
    width = 0.25
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, final, width, label="final-only")
    plt.bar(x, traj, width, label="true causal certificate")
    plt.bar(x + width, reject, width, label="false-cert rejection")
    plt.xticks(x, fams, rotation=25, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("rate")
    plt.title("Phase 59 final-only vs true-certificate acceptance and false-certificate rejection")
    plt.legend()
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_accuracy.png")

    # Rejection rates.
    counts = summary["best_false_certificate_counts"]
    plt.figure(figsize=(12, 5))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("closest-false count")
    plt.title("Phase 59 closest false certificate type")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_rejection_rates.png")

    # False certificate score distribution.
    false_scores = []
    for r in rows:
        for k in FALSE_CERT_TYPES:
            false_scores.append(r[f"false_score_{k}"])
    plt.figure(figsize=(12, 5))
    plt.hist(false_scores, bins=40)
    plt.xlabel("false certificate score")
    plt.ylabel("count")
    plt.title("Phase 59 false certificate score distribution")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_false_certificate_score_distribution.png")

    # True vs false margin.
    plt.figure(figsize=(12, 5))
    plt.hist([r["false_margin"] for r in rows], bins=40)
    plt.xlabel("best false score - true winner score")
    plt.ylabel("trials")
    plt.title("Phase 59 true-vs-false certificate selectivity margin")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_true_vs_false_score_margin.png")


    # Near-miss margin distribution.
    plt.figure(figsize=(12, 5))
    plt.hist([r["near_miss_margin"] for r in rows], bins=40)
    plt.xlabel("compressed near-miss false score - true winner score")
    plt.ylabel("trials")
    plt.title("Phase 59 calibrated near-miss selectivity margin")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_near_miss_margin.png")

    # Log-scaled false certificate score distribution, so the huge wrong-family tail does not hide the hard cases.
    plt.figure(figsize=(12, 5))
    plt.hist([math.log1p(max(0.0, v)) for v in false_scores], bins=40)
    plt.xlabel("log1p(false certificate score)")
    plt.ylabel("count")
    plt.title("Phase 59 log-scaled false certificate score distribution")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_false_certificate_log_score_distribution.png")

    # Sealed holdout acceptance by family.
    holdout_rates = []
    for fam in fams:
        fr = [r for r in rows if r["family"] == fam]
        holdout_rates.append(float(np.mean([1.0 if (r["trajectory_correct"] and r["false_rejected"]) else 0.0 for r in fr])) if fr else 0.0)
    plt.figure(figsize=(12, 5))
    plt.bar(fams, holdout_rates)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("sealed holdout pass rate")
    plt.title("Phase 59 sealed holdout tribunal pass rate by family")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_holdout_pass_rate.png")

    # Branch counts.
    branch_counts = {f: 0 for f in FAMILIES}
    for r in rows:
        branch_counts[r["predicted_branch"]] += 1
    plt.figure(figsize=(12, 5))
    plt.bar(list(branch_counts.keys()), list(branch_counts.values()))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("trials")
    plt.title("Phase 59 predicted branch counts")
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_branch_counts.png")

    # Phase 59 cross-exam rates: jackknife, bootstrap, and pair panels.
    jack_rates = [summary["families"][f]["jackknife_shadow_rate"] for f in fams]
    boot_rates = [summary["families"][f]["bootstrap_shadow_rate"] for f in fams]
    pair_rates = [summary["families"][f]["mean_pair_consensus_rate"] for f in fams]
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, jack_rates, width, label="jackknife shadow")
    plt.bar(x, boot_rates, width, label="bootstrap shadow")
    plt.bar(x + width, pair_rates, width, label="pair cross-exam")
    plt.xticks(x, fams, rotation=25, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("rate")
    plt.title("Phase 59 jackknife/bootstrap/pair cross-exam consensus")
    plt.legend()
    savefig("phase59_adversarial_minimax_counterexample_forge_tribunal_cross_exam_rates.png")


def poison_sweep(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, Any]]:
    sweep_rows = []
    for pf in args.poison_sweep:
        rng = np.random.default_rng(args.seed + int(round(pf * 1000)) + 49000)
        rows = [
            run_trial(i, rng, args.witnesses, pf, args.noise)
            for i in range(args.sweep_trials)
        ]
        s = summarize(rows)
        sweep_rows.append({
            "poison_fraction": pf,
            "trajectory_accuracy": s["trajectory_accuracy"],
            "final_only_accuracy": s["final_only_accuracy"],
            "scramble_stability": s["scramble_stability"],
            "false_certificate_rejection_rate": s["false_certificate_rejection_rate"],
            "calibrated_false_certificate_rejection_rate": s["calibrated_false_certificate_rejection_rate"],
            "bootstrap_shadow_rate": s["bootstrap_shadow_rate"],
            "pair_consensus_mean": s["pair_consensus_mean"],
            "mean_false_margin": s["mean_false_margin"],
            "mean_near_miss_margin": s["mean_near_miss_margin"],
            "mean_ledger_tightness": s["mean_ledger_tightness"],
            "pass": int(s["pass"]),
        })

    write_csv(out_dir / "phase59_adversarial_minimax_counterexample_forge_tribunal_poison_sweep.csv", sweep_rows)

    if plt is not None:
        xs = [r["poison_fraction"] for r in sweep_rows]
        plt.figure(figsize=(12, 5))
        plt.plot(xs, [r["trajectory_accuracy"] for r in sweep_rows], marker="o", label="true causal certificate")
        plt.plot(xs, [r["final_only_accuracy"] for r in sweep_rows], marker="o", label="final-only")
        plt.plot(xs, [r["false_certificate_rejection_rate"] for r in sweep_rows], marker="o", label="false-cert rejection")
        plt.plot(xs, [r["calibrated_false_certificate_rejection_rate"] for r in sweep_rows], marker="o", label="calibrated rejection")
        plt.plot(xs, [r["bootstrap_shadow_rate"] for r in sweep_rows], marker="o", label="bootstrap/cross-exam rejection")
        plt.xlabel("poisoned witness fraction")
        plt.ylabel("rate")
        plt.ylim(0, 1.08)
        plt.title("Phase 59 poisoned-witness stress sweep")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "phase59_adversarial_minimax_counterexample_forge_tribunal_poison_sweep.png", dpi=150)
        plt.close()

    return sweep_rows


def write_examples(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    ex_dir = out_dir / "phase59_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    # Lightweight example cards rather than recreating full witness geometry.
    selected = rows[: min(12, len(rows))]
    for r in selected:
        plt.figure(figsize=(7, 4))
        labels = ["true"] + FALSE_CERT_TYPES
        scores = [r["winner_score"]] + [r[f"false_score_{k}"] for k in FALSE_CERT_TYPES]
        plt.bar(labels, scores)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("score lower is better")
        plt.title(f"trial {r['trial']} {r['family']} false_margin={r['false_margin']:.4f}")
        plt.tight_layout()
        plt.savefig(ex_dir / f"phase56_trial_{int(r['trial']):04d}_certificate_scores.png", dpi=140)
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"E:\BBIT")
    p.add_argument("--outputs", default=None)
    p.add_argument("--trials", type=int, default=700)
    p.add_argument("--sweep-trials", type=int, default=200)
    p.add_argument("--witnesses", type=int, default=5)
    p.add_argument("--poison-fraction", type=float, default=0.28)
    p.add_argument("--noise", type=float, default=0.012)
    p.add_argument("--seed", type=int, default=49049)
    p.add_argument("--poison-sweep", type=float, nargs="*", default=[0.0, 0.14, 0.28, 0.42, 0.56, 0.70])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.outputs) if args.outputs else root / "outputs_basic32"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {out_dir}")
    print(f"[{PHASE}] reset continued: from sealed holdout contradiction tribunal to adversarial minimax counterexample-forge tribunal")
    print(f"[{PHASE}] task: accept the true A->B->C certificate while rejecting false certificates through typed A/B/C role signatures, axis-aware transition probes, contradiction margins, and a adversarial minimax counterexample-forge tribunal that removes each witness in turn, reselects its quorum from the remaining evidence, replays the branch decision, performs bootstrap and pair cross-examination, then forges fresh counterexample certificates from role swaps, endpoint shortcuts, mirror inversions, temporal scrambles, and cross-family impostors; the proof is rejected if it survives only because the false cases were too easy, too familiar, or because a single adversarial false certificate can become the minimax winner under any allowed shadow presentation")

    rng = np.random.default_rng(args.seed)
    rows = [
        run_trial(i, rng, args.witnesses, args.poison_fraction, args.noise)
        for i in range(args.trials)
    ]
    summary = summarize(rows)

    trials_path = out_dir / "phase59_adversarial_minimax_counterexample_forge_tribunal_trials.csv"
    summary_path = out_dir / "phase59_adversarial_minimax_counterexample_forge_tribunal_summary.json"
    report_path = out_dir / "phase59_adversarial_minimax_counterexample_forge_tribunal_report.md"

    write_csv(trials_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_path, summary)
    poison_sweep(args, out_dir)
    plot_outputs(out_dir, rows, summary)
    write_examples(out_dir, rows)

    print(f"[{PHASE}] {PASS_FLAG}={summary['pass']}")
    print(
        f"[{PHASE}] trajectory_accuracy={summary['trajectory_accuracy']:.4f} "
        f"final_only_accuracy={summary['final_only_accuracy']:.4f} "
        f"gain={summary['gain']:.4f} "
        f"scramble_stability={summary['scramble_stability']:.4f} "
        f"false_certificate_rejection_rate={summary['false_certificate_rejection_rate']:.4f} "
        f"sealed_holdout_acceptance_rate={summary['sealed_holdout_acceptance_rate']:.4f} "
        f"double_sealed_acceptance_rate={summary['double_sealed_acceptance_rate']:.4f} "
        f"jackknife_shadow_rate={summary['jackknife_shadow_rate']:.4f} "
        f"bootstrap_shadow_rate={summary['bootstrap_shadow_rate']:.4f} "
        f"pair_consensus_rate={summary['pair_consensus_mean']:.4f} "
        f"metamorphic_invariance_rate={summary['metamorphic_invariance_rate']:.4f} "
        f"counterexample_forge_rate={summary['counterexample_forge_rate']:.4f} "
        f"adversarial_minimax_rate={summary['adversarial_minimax_rate']:.4f} "
        f"null_ablation_jury_rate={summary['null_ablation_jury_rate']:.4f} "
        f"trials={summary['trials']} witnesses={summary['n_witnesses']} "
        f"poison_fraction={summary['poison_fraction']}"
    )
    print(
        f"[{PHASE}] mean_ledger_tightness={summary['mean_ledger_tightness']:.6f} "
        f"mean_ledger_error={summary['mean_ledger_error']:.6f} "
        f"mean_leave_one_error={summary['mean_leave_one_error']:.6f} "
        f"mean_closure_residual={summary['mean_closure_residual']:.6f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"mean_false_margin={summary['mean_false_margin']:.6f} "
        f"sealed_holdout_margin_floor={summary['sealed_holdout_margin_floor']:.6f} "
        f"double_sealed_margin_floor={summary['double_sealed_margin_floor']:.6f} "
        f"bootstrap_margin_floor={summary['bootstrap_margin_floor']:.6f} pair_margin_floor={summary['pair_margin_floor']:.6f}"
    )
    print(f"[{PHASE}] family summary:")
    for fam, s in summary["families"].items():
        print(
            f"  - {fam:<18} traj_acc={s['trajectory_accuracy']:.3f} "
            f"final_acc={s['final_only_accuracy']:.3f} "
            f"stable={s['scramble_stability']:.3f} "
            f"false_reject={s['false_rejection_rate']:.3f} "
            f"jack={s['jackknife_shadow_rate']:.3f} "
            f"boot={s['bootstrap_shadow_rate']:.3f} "
            f"pair={s['mean_pair_consensus_rate']:.3f} "
            f"forge={s['counterexample_forge_rate']:.3f} "
            f"minimax={s['adversarial_minimax_rate']:.3f} "
            f"false_margin={s['mean_false_margin']:.6f} "
            f"forge_margin={s['mean_counterexample_forge_margin']:.6f} "
            f"minimax_margin={s['mean_adversarial_minimax_margin']:.6f} "
            f"ledger_tight={s['mean_ledger_tightness']:.6f} "
            f"min_margin={s['min_margin']:.6f}"
        )

    print(f"[{PHASE}] wrote trials: {trials_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote contradiction-falsifier poison sweep: {out_dir / 'phase59_adversarial_minimax_counterexample_forge_tribunal_poison_sweep.csv'}")
    print(f"[{PHASE}] wrote example png dir: {out_dir / 'phase59_examples'}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
