#!/usr/bin/env python3
# geomlang_phase10_discover_directions.py
#
# Phase 10A: Discover latent directions for LR, betweenness, overlap, tproj
# from exported latents + targets (phase7/phase9 compatible).
#
# Outputs:
#   outputs_edges_relternary256_phase10/phase10_directions.npz
#   outputs_edges_relternary256_phase10/phase10_directions_report.json

import os, json, math
import numpy as np

OUTDIR = "outputs_edges_relternary256_phase10"
os.makedirs(OUTDIR, exist_ok=True)

# Prefer Phase9 derived exports if present; fallback to Phase7 exports.
PHASE9_DIR = "outputs_edges_relternary256_phase9"
PHASE7_DIR = "outputs_edges_relternary256_phase7"

LATENTS_FALLBACK = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS_FALLBACK = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS_FALLBACK   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")

LABELS = [
    "A_left_of_BtoC",      # 0
    "A_right_of_BtoC",     # 1
    "A_between_clear",     # 2
    "A_between_overlap",   # 3
    "A_overlap_only",      # 4
]

def _load_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _find_in_dir(dirpath, exts=(".npy", ".npz", ".json")):
    if not os.path.isdir(dirpath):
        return []
    out = []
    for fn in os.listdir(dirpath):
        if fn.lower().endswith(exts):
            out.append(os.path.join(dirpath, fn))
    return sorted(out)

def _npz_get(npz, wanted, aliases=()):
    keys = list(npz.files)
    for k in [wanted] + list(aliases):
        if k in npz.files:
            return npz[k]
    raise KeyError(f"Missing '{wanted}'. Tried { [wanted]+list(aliases) }. Available: {keys}")

def _ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D (N,) but got {x.shape}")
    return x

def _maybe_sigmoid_if_logits(x, name):
    x = np.asarray(x, dtype=np.float32)
    # Heuristic: if values fall well outside [0,1], treat as logits.
    if (np.nanmin(x) < -0.2) or (np.nanmax(x) > 1.2):
        return 1.0 / (1.0 + np.exp(-x))
    return x

def _standardize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)

def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + eps)
    return v / n

def _ridge_direction(Z, y, lam=1e-2):
    """
    Solve (Z^T Z + lam I) w = Z^T y for w, where y is standardized.
    Returns a unit vector direction in latent space.
    """
    Z = np.asarray(Z, dtype=np.float32)
    y = _standardize(y)
    # center Z to remove mean direction bias
    Zc = Z - Z.mean(axis=0, keepdims=True)
    A = Zc.T @ Zc
    A += lam * np.eye(A.shape[0], dtype=np.float32)
    b = Zc.T @ y
    w = np.linalg.solve(A, b)
    return _unit(w)

def _mean_diff_direction(Z, mask_pos, mask_neg):
    Z = np.asarray(Z, dtype=np.float32)
    mpos = Z[mask_pos].mean(axis=0)
    mneg = Z[mask_neg].mean(axis=0)
    return _unit(mpos - mneg)

def _corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# -------------------------
# Load latents + targets
# -------------------------
# Phase9: try to locate y_true/y_pred and maybe factors inside its outputs folder.
phase9_files = _find_in_dir(PHASE9_DIR, exts=(".npy", ".npz", ".json"))
phase9_npzs  = [p for p in phase9_files if p.lower().endswith(".npz")]
phase9_npys  = [p for p in phase9_files if p.lower().endswith(".npy")]

Z_path = _load_first_existing([
    os.path.join(PHASE9_DIR, "encoded_latents_seed123_N6000.npy"),
    os.path.join(PHASE9_DIR, "Z.npy"),
    os.path.join(PHASE9_DIR, "latents.npy"),
    LATENTS_FALLBACK,
])

targets_path = _load_first_existing([
    os.path.join(PHASE9_DIR, "encoded_targets_seed123_N6000.npz"),
    os.path.join(PHASE9_DIR, "targets.npz"),
    TARGETS_FALLBACK,
])

preds_path = _load_first_existing([
    os.path.join(PHASE9_DIR, "encoded_preds_seed123_N6000.npz"),
    os.path.join(PHASE9_DIR, "preds.npz"),
    PREDS_FALLBACK,
])

print("[phase10A] loading latents:", Z_path)
print("[phase10A] loading targets:", targets_path)
print("[phase10A] loading preds  :", preds_path)

Z = np.load(Z_path)
targets = np.load(targets_path)
preds   = np.load(preds_path)

print("[phase10A] target keys:", list(targets.files))
print("[phase10A] pred keys  :", list(preds.files))
print("[phase10A] Z shape    :", Z.shape)

# -------------------------
# Get supervision signals
# -------------------------
between = _npz_get(targets, "between_score", aliases=("between", "between_gt"))
tproj   = _npz_get(targets, "t_on_BC", aliases=("tproj", "t_on_bc", "t"))
overlap_any = _npz_get(targets, "overlap_any", aliases=("overlap", "o_any", "oany", "overlap_gt"))
lr_sign = _npz_get(targets, "lr_sign", aliases=("lr", "lr_gt", "left_right", "lr_label"))

between = _ensure_1d(between, "between_score")
tproj   = _ensure_1d(tproj,   "t_on_BC")
overlap_any = _ensure_1d(overlap_any, "overlap_any")
lr_sign = _ensure_1d(lr_sign, "lr_sign")

between = _maybe_sigmoid_if_logits(between, "between_score")
tproj   = _maybe_sigmoid_if_logits(tproj,   "t_on_BC")
overlap_any = _maybe_sigmoid_if_logits(overlap_any, "overlap_any")
lr_sign = _maybe_sigmoid_if_logits(lr_sign, "lr_sign")

# Derived discrete labels (Phase9 export should have them; otherwise derive from Phase8 thresholds if present)
y_true = None
for k in ("y_true", "derived_y_true", "derived_label", "labels", "y"):
    if k in targets.files:
        y_true = targets[k]
        break

if y_true is None:
    # fallback: try phase9 npy files for y_true
    for cand in [
        os.path.join(PHASE9_DIR, "y_true.npy"),
        os.path.join(PHASE9_DIR, "derived_y_true.npy"),
        os.path.join(PHASE9_DIR, "labels.npy"),
    ]:
        if os.path.exists(cand):
            y_true = np.load(cand)
            break

if y_true is not None:
    y_true = _ensure_1d(y_true, "y_true").astype(int)
    print("[phase10A] found y_true with counts:", np.bincount(y_true, minlength=len(LABELS)))
else:
    # last-resort: build a rough y_true from targets only
    # (no phase8 thresholds here; just basic default)
    Tb = 0.55
    To = 0.60
    y = []
    for b, o, lr in zip(between, overlap_any, lr_sign):
        if b >= Tb:
            y.append(3 if o >= To else 2)
        else:
            if o >= To:
                y.append(4)
            else:
                y.append(0 if lr < 0.5 else 1)
    y_true = np.array(y, dtype=int)
    print("[phase10A] y_true missing; using fallback Tb/To -> counts:", np.bincount(y_true, minlength=len(LABELS)))

# -------------------------
# Build directions
# -------------------------
# LR: difference of means between left vs right, but only on non-overlap and non-between region (cleaner)
mask_lr_pool = (y_true == 0) | (y_true == 1)
mask_left = (y_true == 0)
mask_right = (y_true == 1)
if mask_left.sum() < 10 or mask_right.sum() < 10:
    # fallback to lr_sign threshold
    mask_left = lr_sign < 0.5
    mask_right = lr_sign >= 0.5

v_lr = _mean_diff_direction(Z, mask_right, mask_left)  # direction that increases "rightness"

# Between: ridge regression on between_score
v_between = _ridge_direction(Z, between, lam=1e-2)

# Overlap: ridge regression on overlap_any
v_overlap = _ridge_direction(Z, overlap_any, lam=1e-2)

# Tproj: ridge regression on t_on_BC
v_tproj = _ridge_direction(Z, tproj, lam=1e-2)

# Orthogonality report
dirs = {
    "v_lr": v_lr,
    "v_between": v_between,
    "v_overlap": v_overlap,
    "v_tproj": v_tproj,
}
names = list(dirs.keys())

dots = {}
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        dots[f"{a}·{b}"] = float(np.dot(dirs[a], dirs[b]))

# Alignment correlations (projection vs signal)
Zc = Z - Z.mean(axis=0, keepdims=True)
proj_lr = Zc @ v_lr
proj_between = Zc @ v_between
proj_overlap = Zc @ v_overlap
proj_tproj = Zc @ v_tproj

metrics = {
    "corr(proj_lr, lr_sign)": _corr(proj_lr, lr_sign),
    "corr(proj_between, between_score)": _corr(proj_between, between),
    "corr(proj_overlap, overlap_any)": _corr(proj_overlap, overlap_any),
    "corr(proj_tproj, t_on_BC)": _corr(proj_tproj, tproj),
}

report = {
    "Z_shape": list(Z.shape),
    "dot_products": dots,
    "alignment_corrs": metrics,
    "notes": [
        "v_lr is mean-diff direction (right - left).",
        "v_between/v_overlap/v_tproj are ridge regression directions on standardized signals.",
        "Dot products near 0 suggest orthogonality / separability.",
    ],
}

# Save
np.savez(
    os.path.join(OUTDIR, "phase10_directions.npz"),
    v_lr=v_lr, v_between=v_between, v_overlap=v_overlap, v_tproj=v_tproj,
    proj_lr=proj_lr, proj_between=proj_between, proj_overlap=proj_overlap, proj_tproj=proj_tproj,
)

with open(os.path.join(OUTDIR, "phase10_directions_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("[phase10A] saved ->", OUTDIR)
print(json.dumps(report, indent=2))
