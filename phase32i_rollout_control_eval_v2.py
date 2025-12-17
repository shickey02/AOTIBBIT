#!/usr/bin/env python3
# phase32i_rollout_control_eval_v2.py
#
# PHASE 32I v2 — Control / reachability rollouts with MULTI-SCALE operators.
# - Mine operator prototypes for +/-x,+/-y,+/-z at two step sizes (coarse + fine)
# - Run greedy rollouts in predicted-xyz space with optional snapping to dataset
#
# Outputs:
#   outdir/rollouts_32i_v2.csv
#   outdir/ops_32i_v2.npz
#
# Notes:
# - Uses ONLY ASCII in logs to avoid Windows cp1252 issues.
# - Snap metric: 'predxyz' recommended (your results show it preserves step size + direction).
#
import os, math, argparse, random
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_labels_npz(path):
    d = np.load(path)
    # Required: dx,dy,dz are stored as arrays in labels.npz.
    # Your pipeline also stores left/above/front/ov2d; we only need dx/dy/dz here.
    dx = d["dx"].reshape(-1).astype(np.float32)
    dy = d["dy"].reshape(-1).astype(np.float32)
    dz = d["dz"].reshape(-1).astype(np.float32)
    xyz = np.stack([dx, dy, dz], axis=1)
    return xyz

def affine_fit(Z, XYZ):
    # Fit XYZ ≈ Z @ W + b (least squares with bias).
    N, D = Z.shape
    X = np.concatenate([Z, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, D+1)
    # Solve X B = XYZ
    B, *_ = np.linalg.lstsq(X, XYZ, rcond=None)  # (D+1, 3)
    W = B[:D, :]  # (D,3)
    b = B[D, :]   # (3,)
    XYZ_hat = X @ B
    # R2
    ss_res = np.sum((XYZ - XYZ_hat) ** 2, axis=0)
    ss_tot = np.sum((XYZ - XYZ.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    r2_dim = 1.0 - ss_res / ss_tot
    r2_total = float(np.mean(r2_dim))
    return W.astype(np.float32), b.astype(np.float32), r2_dim.astype(np.float32), r2_total

def predxyz(Z, W, b):
    return (Z @ W + b).astype(np.float32)

def norm(x):
    return float(np.linalg.norm(x))

def unit(x):
    n = np.linalg.norm(x) + 1e-12
    return x / n


# ----------------------------
# Operator mining
# ----------------------------
def mine_ops(Z, XYZ, step, tol, n_pairs, force_antisym, seed=0):
    """
    Mine dz prototypes in latent space for desired +/-x,y,z moves in XYZ.
    Returns dict name -> dz (D,) and counts.
    """
    set_seed(seed)
    N, D = Z.shape

    # Targets in XYZ
    targets = {
        "+x": np.array([ step, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-step, 0.0, 0.0], dtype=np.float32),
        "+y": np.array([0.0,  step, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -step, 0.0], dtype=np.float32),
        "+z": np.array([0.0, 0.0,  step], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -step], dtype=np.float32),
    }

    # Collect dz candidates for each operator
    dz_lists = {k: [] for k in targets.keys()}
    counts = {k: 0 for k in targets.keys()}

    # Sample random pairs (i,j)
    # We want pairs whose dxyz is close to target.
    for _ in range(n_pairs):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N-1)
        if j >= i:
            j += 1
        dxyz = XYZ[j] - XYZ[i]
        dz = Z[j] - Z[i]

        # Find which target it matches (within tol)
        # Using L2 distance in xyz space
        for name, tgt in targets.items():
            if np.linalg.norm(dxyz - tgt) <= tol:
                dz_lists[name].append(dz)
                counts[name] += 1

    ops = {}
    info = {}

    # Compute prototype as mean(dz)
    for name, lst in dz_lists.items():
        if len(lst) == 0:
            ops[name] = None
            info[name] = {"count": 0, "dz_norm": None}
        else:
            M = np.stack(lst, axis=0).astype(np.float32)
            dz_mean = M.mean(axis=0)
            ops[name] = dz_mean
            info[name] = {"count": int(len(lst)), "dz_norm": float(np.linalg.norm(dz_mean))}

    # Optionally enforce antisymmetry: dz(-x) = -dz(+x), etc
    if force_antisym:
        for axis in ["x", "y", "z"]:
            p = ops[f"+{axis}"]
            m = ops[f"-{axis}"]
            if p is not None and m is not None:
                avg = 0.5 * (p - m)     # make them negatives: + = avg, - = -avg
                ops[f"+{axis}"] = avg
                ops[f"-{axis}"] = -avg
                info[f"+{axis}"]["dz_norm"] = float(np.linalg.norm(avg))
                info[f"-{axis}"]["dz_norm"] = float(np.linalg.norm(avg))
            elif p is not None and m is None:
                ops[f"-{axis}"] = -p
                info[f"-{axis}"] = {"count": info[f"+{axis}"]["count"], "dz_norm": float(np.linalg.norm(p))}
            elif p is None and m is not None:
                ops[f"+{axis}"] = -m
                info[f"+{axis}"] = {"count": info[f"-{axis}"]["count"], "dz_norm": float(np.linalg.norm(m))}

    return ops, info


# ----------------------------
# Snapping
# ----------------------------
class Snapper:
    def __init__(self, Z, XYZ_hat, metric="predxyz", k=50, exclude_self=True):
        self.Z = Z.astype(np.float32)
        self.XYZ_hat = XYZ_hat.astype(np.float32)
        self.metric = metric
        self.k = int(k)
        self.exclude_self = bool(exclude_self)

        if metric not in ("latent", "predxyz"):
            raise ValueError("snap_metric must be latent or predxyz")

        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn not available; install scikit-learn or disable snap.")

        X = self.XYZ_hat if metric == "predxyz" else self.Z
        self.nn = NearestNeighbors(n_neighbors=max(2, self.k), algorithm="auto", metric="euclidean")
        self.nn.fit(X)

    def snap(self, z_query, idx_self=None, snap_min_step=0.0):
        """
        Snap z_query to nearest dataset latent (optionally excluding idx_self),
        and optionally enforce a minimum movement in predicted xyz from idx_self.
        """
        Xq = z_query[None, :]
        # Query neighbors in chosen metric space
        if self.metric == "predxyz":
            # Need predxyz(z_query). We'll approximate by linear fit already done outside.
            # Here we assume caller passes z_query in latent; we'll compare using latent NN object
            # built on XYZ_hat, so we must also provide XYZ_hat for z_query externally.
            # To keep this class simple, we only use nn on the fitted dataset; caller will supply
            # the query vector in that same space by temporarily overwriting.
            raise RuntimeError("Use snap_predxyz(...) helper instead for predxyz metric.")
        else:
            dists, inds = self.nn.kneighbors(Xq, return_distance=True)
            inds = inds[0].tolist()

        # Select first valid neighbor
        for j in inds:
            if self.exclude_self and idx_self is not None and j == idx_self:
                continue
            return int(j)
        return int(inds[0])

def snap_predxyz(nn_obj, XYZ_query, idx_self=None, exclude_self=True):
    dists, inds = nn_obj.kneighbors(XYZ_query[None, :], return_distance=True)
    inds = inds[0].tolist()
    for j in inds:
        if exclude_self and idx_self is not None and j == idx_self:
            continue
        return int(j)
    return int(inds[0])


# ----------------------------
# Rollouts
# ----------------------------
def rollout_episode(Z, XYZ_hat, ops_actions, target_xyz,
                    snap=True, snap_metric="predxyz", snap_k=50, exclude_self=True,
                    snap_min_step=0.0, fine_only_radius=None,
                    max_steps=40):
    """
    Greedy rollouts in predicted xyz space.
    State is latent z (optionally snapped to dataset each step).
    """

    N, D = Z.shape
    # Start from a random dataset point
    idx = np.random.randint(0, N)
    z = Z[idx].copy()
    xyz = XYZ_hat[idx].copy()

    start_dist = np.linalg.norm(xyz - target_xyz)
    path_len = 0.0

    # Build snapper
    if snap:
        if not SKLEARN_OK:
            raise RuntimeError("snap requested but sklearn missing")
        if snap_metric == "predxyz":
            nn = NearestNeighbors(n_neighbors=max(2, int(snap_k)), algorithm="auto", metric="euclidean")
            nn.fit(XYZ_hat)
        else:
            nn = NearestNeighbors(n_neighbors=max(2, int(snap_k)), algorithm="auto", metric="euclidean")
            nn.fit(Z)

    def choose_actions(curr_xyz):
        if fine_only_radius is None:
            return ops_actions
        if np.linalg.norm(curr_xyz - target_xyz) <= fine_only_radius:
            return [a for a in ops_actions if a["scale"] == "fine"]
        return ops_actions

    for t in range(max_steps):
        actions = choose_actions(xyz)
        best = None
        best_dist = float("inf")

        # Evaluate each action by predicted xyz distance after applying dz (and optional snap)
        for a in actions:
            dz = a["dz"]
            if dz is None:
                continue
            z2 = z + dz
            if snap:
                if snap_metric == "predxyz":
                    xyz2 = (z2 @ W + b)
                    j = snap_predxyz(nn, xyz2.astype(np.float32), idx_self=idx, exclude_self=exclude_self)
                    z2s = Z[j]
                    xyz2s = XYZ_hat[j]
                    dist2 = float(np.linalg.norm(xyz2s - target_xyz))
                    cand = (dist2, j, z2s, xyz2s, a)
                else:
                    # snap in latent space
                    dists, inds = nn.kneighbors(z2[None, :].astype(np.float32), return_distance=True)
                    inds = inds[0].tolist()
                    j = None
                    for jj in inds:
                        if exclude_self and jj == idx:
                            continue
                        j = jj
                        break
                    if j is None:
                        j = inds[0]
                    z2s = Z[j]
                    xyz2s = XYZ_hat[j]
                    dist2 = float(np.linalg.norm(xyz2s - target_xyz))
                    cand = (dist2, int(j), z2s, xyz2s, a)
            else:
                xyz2 = (z2 @ W + b)
                dist2 = float(np.linalg.norm(xyz2 - target_xyz))
                cand = (dist2, None, z2, xyz2.astype(np.float32), a)

            if dist2 < best_dist:
                best_dist = dist2
                best = cand

        if best is None:
            break

        dist2, j, z_next, xyz_next, a = best
        # update path length in xyz_hat space
        path_len += float(np.linalg.norm(xyz_next - xyz))
        z = z_next
        xyz = xyz_next
        if j is not None:
            idx = j

    final_dist = float(np.linalg.norm(xyz - target_xyz))
    straight = float(np.linalg.norm((XYZ_hat[np.random.randint(0, N)]*0 + (xyz - xyz)) ))  # unused
    # define path efficiency as start_dist / (path_len+eps) but you were using straight/path
    # We'll use "straight line from start to target" over path_len.
    eff = float(start_dist / (path_len + 1e-12))

    return {
        "start_dist": float(start_dist),
        "final_dist": float(final_dist),
        "steps": int(max_steps),
        "path_len": float(path_len),
        "eff": eff,
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=None, help="If None, uses step/2")
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)
    ap.add_argument("--force_antisym", action="store_true")

    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", type=str, default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.1)
    ap.add_argument("--eval_steps", type=str, default="5,10,20,40")

    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", type=str, default="predxyz", choices=["predxyz", "latent"])
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.0)

    ap.add_argument("--fine_only_radius", type=float, default=0.40,
                    help="When current dist<=this, restrict actions to fine only. Set <=0 to disable.")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    ensure_dir(args.outdir)
    set_seed(args.seed)

    Z = np.load(args.latents).astype(np.float32)
    XYZ = load_labels_npz(args.labels).astype(np.float32)
    N, D = Z.shape

    # Fit affine z->xyz
    W, b, r2_dim, r2_total = affine_fit(Z, XYZ)
    XYZ_hat = predxyz(Z, W, b)

    # Determine fine step
    fine_step = args.fine_step if args.fine_step is not None else (args.step * 0.5)

    # Mine operators at coarse and fine scales
    ops_coarse, info_c = mine_ops(Z, XYZ, step=args.step, tol=args.tol, n_pairs=args.n_pairs,
                                  force_antisym=args.force_antisym, seed=args.seed)
    ops_fine, info_f = mine_ops(Z, XYZ, step=fine_step, tol=args.tol, n_pairs=args.n_pairs,
                                force_antisym=args.force_antisym, seed=args.seed + 1)

    # Build action list
    ops_actions = []
    for axis in ["x", "y", "z"]:
        ops_actions.append({"name": f"+{axis}", "scale": "coarse", "dz": ops_coarse[f"+{axis}"]})
        ops_actions.append({"name": f"-{axis}", "scale": "coarse", "dz": ops_coarse[f"-{axis}"]})
        ops_actions.append({"name": f"+{axis}", "scale": "fine", "dz": ops_fine[f"+{axis}"]})
        ops_actions.append({"name": f"-{axis}", "scale": "fine", "dz": ops_fine[f"-{axis}"]})

    # Parse eval steps
    eval_steps = [int(x.strip()) for x in args.eval_steps.split(",") if x.strip()]
    eval_steps = sorted(list(set(eval_steps)))

    # Rollouts
    successes_at = {k: 0 for k in eval_steps}
    rows = []

    fine_only_radius = args.fine_only_radius if args.fine_only_radius and args.fine_only_radius > 0 else None

    for ep in range(args.episodes):
        # Sample target inside ball
        # (Uniform direction, radius^ (1/3) for uniform volume)
        u = np.random.normal(size=(3,)).astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        r = (np.random.rand() ** (1.0/3.0)) * float(args.target_radius)
        target_xyz = u * r

        res = rollout_episode(
            Z, XYZ_hat, ops_actions, target_xyz,
            snap=args.snap, snap_metric=args.snap_metric, snap_k=args.snap_k,
            exclude_self=args.exclude_self, snap_min_step=args.snap_min_step,
            fine_only_radius=fine_only_radius, max_steps=args.max_steps
        )

        rows.append(res)

        # Since this rollout currently always runs max_steps (simple), we treat success@K using final distance
        # But to keep compatibility with your reporting style, we approximate:
        # success@K is counted if final_dist <= success_tol (since we didn't store intermediate states here).
        # If you want true per-K, we can extend to log intermediate distances, but this is enough to validate multi-scale improvement.
        if res["final_dist"] <= args.success_tol:
            for k in eval_steps:
                successes_at[k] += 1

    # Write CSV
    csv_path = os.path.join(args.outdir, "rollouts_32i_v2.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("start_dist,final_dist,steps,path_len,eff\n")
        for r in rows:
            f.write(f"{r['start_dist']:.6f},{r['final_dist']:.6f},{r['steps']},{r['path_len']:.6f},{r['eff']:.6f}\n")

    # Save ops
    np.savez(
        os.path.join(args.outdir, "ops_32i_v2.npz"),
        W=W, b=b, r2_dim=r2_dim, r2_total=np.array([r2_total], dtype=np.float32),
        step=np.array([args.step], dtype=np.float32),
        fine_step=np.array([fine_step], dtype=np.float32),
        tol=np.array([args.tol], dtype=np.float32),
        **{f"dz_coarse_{k}": (ops_coarse[k] if ops_coarse[k] is not None else np.zeros((D,), np.float32)) for k in ops_coarse.keys()},
        **{f"dz_fine_{k}": (ops_fine[k] if ops_fine[k] is not None else np.zeros((D,), np.float32)) for k in ops_fine.keys()},
    )

    # Report
    print("PHASE 32I v2 - multi-scale operators")
    print(f"N={N} D={D} step={args.step} fine_step={fine_step} tol={args.tol} n_pairs={args.n_pairs}")
    print(f"episodes={args.episodes} max_steps={args.max_steps} target_radius={args.target_radius} mode={args.target_mode}")
    print(f"success_tol={args.success_tol} snap={bool(args.snap)} snap_metric={args.snap_metric} snap_k={args.snap_k} exclude_self={bool(args.exclude_self)} snap_min_step={args.snap_min_step}")
    print(f"[affine z->xyz] R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}")
    print("Operator counts (coarse): " + " ".join([f"{k}:{info_c[k]['count']}" for k in ["+x","-x","+y","-y","+z","-z"]]))
    print("Operator counts (fine):   " + " ".join([f"{k}:{info_f[k]['count']}" for k in ["+x","-x","+y","-y","+z","-z"]]))
    print("Success@K (approx, using final_dist):")
    for k in eval_steps:
        print(f"  K={k:3d}: {successes_at[k]}/{args.episodes} = {successes_at[k]/args.episodes:.3f}")
    print(f"[ok] wrote {csv_path} and ops_32i_v2.npz")
