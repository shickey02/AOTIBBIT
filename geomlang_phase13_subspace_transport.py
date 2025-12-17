#!/usr/bin/env python3
# geomlang_phase13_subspace_transport.py
#
# Computes subspace transport stability using principal angles between:
#   S_bo = span(v_between_clean, v_overlap_clean)
#   S_lt = span(v_lr, v_tproj)
#
# This is the right “global factor stability” measure when single directions twist.

import os, json
import numpy as np

OUTDIR = "outputs_edges_relternary256_phase13"
REPORT = os.path.join(OUTDIR, "phase13_transport_report_v2.json")

def orth(A, eps=1e-12):
    """Orthonormalize columns of A (D,k) via QR."""
    Q, R = np.linalg.qr(A)
    # keep only non-degenerate columns
    keep = []
    for i in range(Q.shape[1]):
        if np.linalg.norm(Q[:, i]) > eps:
            keep.append(i)
    if not keep:
        return Q[:, :0]
    return Q[:, keep]

def principal_angles(Q1, Q2):
    """
    Q1, Q2: orthonormal bases (D,k1) and (D,k2).
    Returns angles in radians (min(k1,k2)).
    """
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([])
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)  # 0 = identical subspace, pi/2 = orthogonal

def subspace_similarity(Q1, Q2):
    """Mean cos(principal angles) in [0,1]."""
    ang = principal_angles(Q1, Q2)
    if ang.size == 0:
        return 0.0
    return float(np.mean(np.cos(ang)))

def deg(x): return float(x * 180.0 / np.pi)

def main():
    assert os.path.exists(REPORT), "Run phase13_direction_transport_v2.py first."
    with open(REPORT, "r") as f:
        rep = json.load(f)

    vec_npz = rep["vectors_npz"]
    V = np.load(vec_npz)

    anchors = list(rep["anchors"].keys())

    # build subspace bases per anchor
    subspaces = {}
    for a in anchors:
        vb = V[f"{a}__v_between_clean"].astype(np.float64)
        vo = V[f"{a}__v_overlap_clean"].astype(np.float64)
        vl = V[f"{a}__v_lr"].astype(np.float64)
        vt = V[f"{a}__v_tproj"].astype(np.float64)

        Q_bo = orth(np.stack([vb, vo], axis=1))
        Q_lt = orth(np.stack([vl, vt], axis=1))

        subspaces[a] = {"Q_bo": Q_bo, "Q_lt": Q_lt}

    # pairwise report
    out = {"bo": {}, "lt": {}, "notes": [
        "Similarity is mean cos(principal angles). 1.0=identical subspace, 0.0=orthogonal.",
        "Angles are listed in degrees; smaller is more stable under transport.",
        "If subspaces are stable but single vectors are not, the factor exists globally but twists within that subspace."
    ]}

    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            a = anchors[i]; b = anchors[j]

            Q1 = subspaces[a]["Q_bo"]; Q2 = subspaces[b]["Q_bo"]
            ang = principal_angles(Q1, Q2)
            out["bo"][f"{a}__vs__{b}"] = {
                "similarity": subspace_similarity(Q1, Q2),
                "angles_deg": [deg(x) for x in ang.tolist()]
            }

            Q1 = subspaces[a]["Q_lt"]; Q2 = subspaces[b]["Q_lt"]
            ang = principal_angles(Q1, Q2)
            out["lt"][f"{a}__vs__{b}"] = {
                "similarity": subspace_similarity(Q1, Q2),
                "angles_deg": [deg(x) for x in ang.tolist()]
            }

    out_path = os.path.join(OUTDIR, "phase13_subspace_transport_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase13-subspace] saved ->", out_path)

if __name__ == "__main__":
    main()
