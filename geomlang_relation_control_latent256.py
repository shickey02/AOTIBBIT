#!/usr/bin/env python3
# geomlang_relation_control_latent256.py
#
# Train a logistic classifier on geometric features (here we *use only* t_x, t_y)
# and then use it as a "critic" to control the autoencoder latent
# towards a target relation via generator moves in z-space.
#
# This version:
#   - Trains the classifier on (t_x, t_y) only.
#   - Uses a probability-based objective:
#       J = P(target) - lambda * ||z - z_seed||
#     (so we *maximize* J in the controller).
#   - Runs in **axis-following mode**:
#       * For right_of / left_of  -> x-axis rail (g_x, with fixed sign)
#       * For above / below       -> y-axis rail (g_y, with fixed sign)
#       * Controller actions: {stay, +axis} only (no -axis).
#   - Keeps 2-step lookahead over all move sequences on that rail.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[ctrl256]"
IMG_SIZE   = 64
LATENT_DIM = 256
N_SAMPLES  = 6000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

# geometry / PCA config
RESID_DIM  = 32
CURV_IDXS  = [0, 1, 2, 5]

# controller config
CTRL_EPS   = 1.5   # latent step size for control moves
N_STEPS    = 32    # steps per trajectory
LOOKAHEAD  = 2     # 2-step lookahead
SEED_PER_TARGET = 4

RNG_SEED   = 0


# ----------------- helpers: dataset & geometry -----------------

def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    all_imgs = []
    all_rel  = []
    all_z    = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    imgs = torch.cat(all_imgs, dim=0)   # (N, C, H, W)
    rel  = torch.cat(all_rel,  dim=0)   # (N,)
    z    = torch.cat(all_z,    dim=0)   # (N, D)

    print(f"{TAG} Latent shape: {tuple(z.shape)}")
    return imgs, rel, z


def compute_generators(z, rel):
    """Return g_x, g_y, and relation means mu_r (all torch CPU tensors)."""
    mus = []
    rel_np = rel.numpy()
    for rid in range(5):
        mask = (rel_np == rid)
        z_r = z[mask]
        mu_r = z_r.mean(dim=0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={mask.sum()}, ||mu_r||={mu_r.norm():.3f}")
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    v_LR = mu_right - mu_left
    v_AB = mu_below - mu_above

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    cos_xy = torch.dot(g_x, g_y).item()
    angle = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {v_LR.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {v_AB.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x, g_y, mus


def compute_residual_pca(z, mu_overlap, g_x, g_y):
    """
    Subtract overlap mean, project out generator plane, run PCA on residuals.
    Returns:
        H_full: curvature basis (D, k)
        t_x, t_y, r_perp (for diagnostics)
    """
    z = z.to(DEVICE)
    mu0 = mu_overlap.to(DEVICE)

    z_c = z - mu0.unsqueeze(0)
    gx = g_x.to(DEVICE)
    gy = g_y.to(DEVICE)

    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    if denom <= 0:
        raise RuntimeError("Gram matrix near-singular in compute_residual_pca")

    a = torch.matmul(z_c, gx)  # <z_c, g_x>
    b = torch.matmul(z_c, gy)  # <z_c, g_y>
    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    proj_plane = (
        t_x.unsqueeze(1) * gx.unsqueeze(0)
        + t_y.unsqueeze(1) * gy.unsqueeze(0)
    )

    r_perp = z_c - proj_plane          # (N, D)
    norms = torch.norm(r_perp, dim=1)
    print(f"{TAG} Residual ||r_perp|| stats: mean={norms.mean():.4f}, "
          f"std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

    r_np = r_perp.cpu().numpy()

    from sklearn.decomposition import PCA
    print(f"{TAG} Running residual PCA on shape {r_np.shape} with RESID_DIM={RESID_DIM}")
    pca = PCA(n_components=RESID_DIM)
    pca.fit(r_np)
    comps = pca.components_  # (RESID_DIM, D)

    print(f"{TAG} Residual PCA variance (first 10 PCs):")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"   PC{i+1:02d}: {var*100:5.2f}%")

    H_full = comps[CURV_IDXS, :].T  # (D, k)
    print(f"{TAG} Using residual PCs {CURV_IDXS} as curvature basis (dim={H_full.shape[1]})")

    return H_full, t_x.cpu(), t_y.cpu(), r_perp.cpu()


def extract_features_from_z(
    z_vec: torch.Tensor,
    mu_overlap: torch.Tensor,
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    H_full: np.ndarray,
):
    """
    Single-sample feature extractor for controller:
    Input:
        z_vec: (D,) torch on DEVICE
    Output:
        feat: np.array of shape (2 + k,) = [t_x, t_y, c1..ck]
    """
    z_vec = z_vec.to(DEVICE)
    mu0   = mu_overlap.to(DEVICE)
    gx    = g_x.to(DEVICE)
    gy    = g_y.to(DEVICE)
    H     = torch.from_numpy(H_full).to(DEVICE)  # (D, k)

    z_c = z_vec - mu0

    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    a = torch.dot(z_c, gx).item()
    b = torch.dot(z_c, gy).item()
    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    plane = t_x * gx + t_y * gy
    r_perp = z_c - plane

    coeffs = torch.matmul(H.t(), r_perp)  # (k,)
    c = coeffs.detach().cpu().numpy()

    feat = np.concatenate([[t_x, t_y], c])
    return feat


def build_feature_matrix(z, rel, mus, g_x, g_y, H_full):
    """
    Build (N, 2 + k) matrix [t_x, t_y, c1..ck] for all z.
    """
    mu_overlap = mus[REL_OVERLAP]

    N = z.shape[0]
    k = H_full.shape[1]
    feats = np.zeros((N, 2 + k), dtype=np.float32)

    for i in range(N):
        feats[i] = extract_features_from_z(
            z[i], mu_overlap, g_x, g_y, H_full
        )

    # Diagnostics (assume k >= 4 for naming)
    names = ["t_x", "t_y"] + [f"c{j+1}" for j in range(k)]
    for j, name in enumerate(names[:6]):  # print first 6 dims: t_x, t_y, c1..c4
        col = feats[:, j]
        print(f"{TAG} Feature {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"min={col.min():.4f}, max={col.max():.4f}")

    return feats


# ----------------- classifier training -----------------

def train_logistic_features(X_full, y):
    """
    Train multinomial logistic regression on geometric features.
    We only use the first two dimensions (t_x, t_y) for classification.
    Uses 80/20 random train/test split.
    """
    # Use (t_x, t_y) only
    X = X_full[:, :2]

    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    split = int(0.8 * N)
    train_idx = idx[:split]
    test_idx  = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100.0
    test_acc  = clf.score(X_test,  y_test) * 100.0
    print(f"{TAG} Logistic on [t_x,t_y] | train_acc={train_acc:.2f}% | "
          f"test_acc={test_acc:.2f}%")

    # Confusion on test set
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
    print("\n======================================================================")
    print(f"{TAG} Confusion matrix (rows=true, cols=pred) on held-out 20%:")
    header = "         " + " ".join([f"{name:>7}" for name in REL_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = f" {REL_NAMES[i]:>7} " + "".join([f"{v:7d}" for v in row])
        print(row_str)
    print("======================================================================\n")

    return clf, (X_test, y_test)


# ----------------- controller -----------------

def flow_step(model, z_vec, direction, eps):
    """
    One latent move + decode/encode re-projection.
    z_vec: (D,) torch on DEVICE
    direction: (D,) torch on DEVICE
    """
    z_prop = z_vec + eps * direction
    with torch.no_grad():
        x = model.decode(z_prop.unsqueeze(0))
        z_new = model.encode(x).squeeze(0)
    return z_new


def objective_for_z(model, z_seed, z_vec, clf, target_id, mu_overlap, g_x, g_y, H_full):
    """
    Compute a probability-based objective:
        J = P(target) - lambda * ||z_vec - z_seed||
    where J is *maximized* by the controller.
    Only (t_x, t_y) are passed to the classifier; curvature is for logging.
    """
    feat_full = extract_features_from_z(z_vec, mu_overlap, g_x, g_y, H_full)
    feat_xy   = feat_full[:2].reshape(1, -1)

    probs = clf.predict_proba(feat_xy)[0]  # shape (5,)
    P_target = float(probs[target_id])
    pred_id  = int(np.argmax(probs))

    # Small regularizer to discourage huge latent drift from the seed
    loss_reg = float(torch.norm(z_vec - z_seed).item())
    LAMBDA_REG = 1e-5
    J = P_target - LAMBDA_REG * loss_reg

    return J, P_target, pred_id


def enumerate_move_sequences(actions, depth):
    """
    All sequences of given depth from 'actions' list.
    actions: list of labels, e.g. ["stay","+axis"]
    """
    if depth == 1:
        return [[a] for a in actions]
    out = []
    def rec(prefix, d):
        if d == 0:
            out.append(prefix[:])
            return
        for a in actions:
            prefix.append(a)
            rec(prefix, d-1)
            prefix.pop()
    rec([], depth)
    return out


def run_controller_for_seed(
    model, clf,
    imgs, z, rel,
    idx_seed,
    target_id,
    g_x, g_y, mus, H_full,
    out_dir,
    axis_name,
    axis_dir,
    axis_desc,
    max_steps=N_STEPS,
    eps=CTRL_EPS,
    lookahead=LOOKAHEAD,
):
    """
    Run control with multi-step lookahead from a single seed index
    along a **fixed-sign axis rail**.
    Saves a trajectory strip of decoded images.
    """
    assert axis_dir is not None, "axis_dir must be provided for axis-following mode"
    mu_overlap = mus[REL_OVERLAP]

    # Axis-following: actions are stay / +axis only
    move_dirs = {
        "stay":  torch.zeros(LATENT_DIM, device=DEVICE),
        "+axis": axis_dir.to(DEVICE),
    }
    move_labels = list(move_dirs.keys())
    sequences = enumerate_move_sequences(move_labels, lookahead)

    # Seed latent
    z_seed = z[idx_seed].to(DEVICE)
    z_cur  = z_seed.clone()
    rel_true = int(rel[idx_seed].item())

    # Initial classifier readout
    feat0_full = extract_features_from_z(z_cur, mu_overlap, g_x, g_y, H_full)
    feat0_xy   = feat0_full[:2].reshape(1, -1)
    probs0     = clf.predict_proba(feat0_xy)[0]
    P0         = float(probs0[target_id])
    pred0      = int(np.argmax(probs0))

    print(f"{TAG}   start: rel_true={REL_NAMES[rel_true]}, "
          f"pred={REL_NAMES[pred0]}, "
          f"P(target={REL_NAMES[target_id]})={P0:.3f} "
          f"(axis={axis_name}, {axis_desc})")

    frames = []
    with torch.no_grad():
        x0 = model.decode(z_cur.unsqueeze(0)).cpu()  # (1,C,H,W)
        frames.append(x0)

    for step in range(1, max_steps + 1):
        # choose best move sequence by lookahead (maximize J)
        best_J = -1e9
        best_seq = None

        for seq in sequences:
            z_tmp = z_cur.clone()
            for mv in seq:
                dir_vec = move_dirs[mv]
                z_tmp = flow_step(model, z_tmp, dir_vec, eps)

            J, _, _ = objective_for_z(
                model, z_seed, z_tmp, clf, target_id, mu_overlap, g_x, g_y, H_full
            )
            if J > best_J:
                best_J = J
                best_seq = seq

        first_move = best_seq[0]
        dir_first = move_dirs[first_move]
        z_cur = flow_step(model, z_cur, dir_first, eps)

        J_cur, P_tgt, pred_id = objective_for_z(
            model, z_seed, z_cur, clf, target_id, mu_overlap, g_x, g_y, H_full
        )

        print(f"{TAG}   step {step:02d}: move={first_move:>5}, "
              f"pred={REL_NAMES[pred_id]:>7}, "
              f"P(target={REL_NAMES[target_id]})={P_tgt:.3f}, J={J_cur:.3f}")

        with torch.no_grad():
            x = model.decode(z_cur.unsqueeze(0)).cpu()
            frames.append(x)

    # save strip
    frames_t = torch.cat(frames, dim=0)  # (T+1, C, H, W)
    from torchvision.utils import make_grid, save_image
    grid = make_grid(frames_t, nrow=len(frames), padding=2)
    fname = os.path.join(
        out_dir,
        f"relation_control_axis_target_{REL_NAMES[target_id]}_seed{idx_seed}_latent256.png"
    )
    save_image(grid, fname)
    print(f"{TAG}   Saved trajectory strip -> {fname}")

    return


# ----------------- main -----------------

def main():
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    model.to(DEVICE)
    model.eval()

    imgs, rel, z = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mus = compute_generators(z, rel)

    # residual PCA + curvature basis
    H_full, t_x_all, t_y_all, r_perp = compute_residual_pca(z, mus[REL_OVERLAP], g_x, g_y)

    # build feature matrix
    X = build_feature_matrix(z, rel, mus, g_x, g_y, H_full)
    y = rel.numpy()

    print(f"{TAG} Feature matrix shape: {X.shape} (N={X.shape[0]}, F={X.shape[1]})")

    clf, (X_test, y_test) = train_logistic_features(X, y)

    # -------------- controller demos --------------

    print("======================================================================")
    print(f"{TAG} Running controller demos (axis-following mode, fixed-sign rails)...")
    print("======================================================================")

    rel_np = rel.numpy()

    # two targets to show: right_of, above
    targets = [REL_RIGHT, REL_ABOVE]

    for target in targets:
        # Determine axis and direction based on target relation
        if target == REL_RIGHT:
            axis_name = "x"
            axis_dir  = g_x           # +x -> right_of
            axis_desc = "towards right_of (+g_x)"
        elif target == REL_LEFT:
            axis_name = "x"
            axis_dir  = -g_x          # -x -> left_of
            axis_desc = "towards left_of (-g_x)"
        elif target == REL_ABOVE:
            axis_name = "y"
            axis_dir  = -g_y          # -g_y -> above (since g_y = mu_below - mu_above)
            axis_desc = "towards above (-g_y)"
        elif target == REL_BELOW:
            axis_name = "y"
            axis_dir  = g_y           # +g_y -> below
            axis_desc = "towards below (+g_y)"
        else:
            # Default: no axis (shouldn't happen in this script)
            axis_name = "?"
            axis_dir  = None
            axis_desc = "no axis"

        print(f"{TAG} === Target relation: {REL_NAMES[target]} "
              f"(axis={axis_name}, {axis_desc}) ===")

        # choose seeds that are NOT already target relation
        mask_other = (rel_np != target)
        idx_candidates = np.nonzero(mask_other)[0]
        if idx_candidates.size == 0:
            print(f"{TAG}   No non-target seeds available for {REL_NAMES[target]}")
            continue

        np.random.shuffle(idx_candidates)
        seeds = idx_candidates[:SEED_PER_TARGET]
        print(f"{TAG} Seed indices: {list(seeds)}")

        for idx_seed in seeds:
            print(f"{TAG} -- Seed (global idx={idx_seed}, "
                  f"true_rel={REL_NAMES[int(rel_np[idx_seed])]}) --")
            run_controller_for_seed(
                model, clf,
                imgs, z, rel,
                idx_seed,
                target,
                g_x, g_y, mus, H_full,
                OUT_DIR,
                axis_name,
                axis_dir,
                axis_desc,
                max_steps=N_STEPS,
                eps=CTRL_EPS,
                lookahead=LOOKAHEAD,
            )

    print("======================================================================")
    print(f"{TAG} Done.")
    print("======================================================================")


if __name__ == "__main__":
    main()
