#!/usr/bin/env python3
import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    
    # Mining
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=0.0875)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)
    
    # Beam
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--beam_horizon", type=int, default=3)
    
    # Rollout
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.10)
    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed")
    
    # Snap
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--snap_min_step", type=float, default=0.1)
    ap.add_argument("--near_radius", type=float, default=0.50)
    ap.add_argument("--snap_min_step_near", type=float, default=0.0)
    
    # Misc
    ap.add_argument("--force_antisym", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Load & Affine
    print(f"Loading data from {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    N, D = Z.shape

    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)
    W_affine = A[:D]
    b_affine = A[D]

    def z_to_xyz_batch(z_batch):
        return z_batch @ W_affine + b_affine

    # 2. Mine Operators
    print("Mining operators...")
    idx_i = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)
    valid_mask = np.abs(norms - args.step) <= args.tol
    
    dXYZ = dXYZ[valid_mask]
    u = dXYZ / (norms[valid_mask, None] + 1e-8)
    dir_vecs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    dir_keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    
    dots = u @ dir_vecs.T
    best_dir = np.argmax(dots, axis=1)
    final_i = idx_i[valid_mask][np.max(dots, axis=1) > 0.95]
    final_j = idx_j[valid_mask][np.max(dots, axis=1) > 0.95]
    final_dir = best_dir[np.max(dots, axis=1) > 0.95]
    
    dZ = Z[final_j] - Z[final_i]
    proto_coarse = {}
    for i, k in enumerate(dir_keys):
        dz_k = dZ[final_dir == i]
        dz_k = dz_k[np.linalg.norm(dz_k, axis=1) > 1e-6]
        if len(dz_k) > 0: proto_coarse[k] = np.mean(dz_k, axis=0)

    if args.force_antisym:
        for ax in ["x", "y", "z"]:
            p, n = "+" + ax, "-" + ax
            if p in proto_coarse and n in proto_coarse:
                avg = (proto_coarse[p] - proto_coarse[n]) / 2.0
                proto_coarse[p], proto_coarse[n] = avg, -avg

    scale = float(args.fine_step / args.step)
    proto_fine = {k: proto_coarse[k] * scale for k in dir_keys}
    ops_matrix = np.stack([proto_coarse[k] for k in dir_keys] + [proto_fine[k] for k in dir_keys])
    NUM_OPS = 12

    # 3. Snap Setup (Optimized n_jobs=1 for small queries)
    nbrs = None
    if args.snap:
        print(f"Building KDTree (k={args.snap_k})...")
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1, algorithm='auto', n_jobs=1).fit(snap_space)

    # Global Oracle (for validation)
    print("Building Oracle KDTree (Global XYZ search)...")
    oracle_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=1).fit(XYZ)

    def do_snap_batch(z_batch, min_step):
        if not args.snap: return z_batch
        q = z_batch if args.snap_metric == "latent" else z_to_xyz_batch(z_batch)
        _, inds = nbrs.kneighbors(q)
        z_out = z_batch.copy()
        for i in range(len(z_batch)):
            for k_idx in range(inds.shape[1]):
                idx = inds[i, k_idx]
                if np.linalg.norm(z_batch[i] - Z[idx]) >= min_step:
                    z_out[i] = Z[idx]; break
        return z_out

    # 4. Beam Search with Deduplication
    def run_beam_search(z_start, target, width, horizon, snap_min_step):
        # Beam items: {"z": vector, "first_op": int, "dist": float}
        # Initial expansion
        z_cands = z_start + ops_matrix
        if args.snap: z_cands = do_snap_batch(z_cands, snap_min_step)
        xyz_cands = z_to_xyz_batch(z_cands)
        dists = np.linalg.norm(xyz_cands - target, axis=1)
        
        # Init beam
        beam = []
        for i in np.argsort(dists)[:width]:
            beam.append({"z": z_cands[i], "first_op": i, "dist": dists[i]})
            
        for h in range(horizon - 1):
            next_cands = []
            # Batch expand
            zs = np.stack([b["z"] for b in beam]) # (K, D)
            expanded = (zs[:, None, :] + ops_matrix[None, :, :]).reshape(-1, D)
            
            if args.snap: expanded = do_snap_batch(expanded, snap_min_step)
            ex_xyz = z_to_xyz_batch(expanded)
            ex_dists = np.linalg.norm(ex_xyz - target, axis=1)
            
            # Deduplicate by hashing XYZ (approx)
            seen_xyz = set()
            
            # Collect
            for i in range(len(expanded)):
                # Simple spatial hash for dedup (round to 3 decimals)
                h_val = tuple(np.round(ex_xyz[i], 3))
                if h_val in seen_xyz: continue
                seen_xyz.add(h_val)
                
                parent = i // NUM_OPS
                next_cands.append({
                    "z": expanded[i],
                    "first_op": beam[parent]["first_op"],
                    "dist": ex_dists[i]
                })
            
            # Prune
            next_cands.sort(key=lambda x: x["dist"])
            beam = next_cands[:width]
            
        best = beam[0]
        # Re-execute best first move
        z_next = z_start + ops_matrix[best["first_op"]]
        if args.snap: z_next = do_snap_batch(z_next[None,:], snap_min_step)[0]
        xyz_next = z_to_xyz_batch(z_next[None,:])[0]
        return z_next, xyz_next

    # 5. Episodes
    rows = []
    success_at = {k:0 for k in [5,10,20,40]}
    oracle_successes = 0
    oracle_dists = []
    
    print(f"\nStarting {args.episodes} episodes...")
    iterator = tqdm(range(args.episodes), desc="Rollouts")
    
    for ep in iterator:
        s = int(rng.integers(0, N))
        z, xyz = Z[s].copy(), z_to_xyz_batch(Z[s][None,:])[0]

        if args.target_sampling == "fixed":
            v = rng.normal(size=3).astype(np.float32); v /= np.linalg.norm(v)
            tgt = xyz + v * args.target_radius
        else:
            tgt = xyz + rng.normal(size=3).astype(np.float32)
            tgt = xyz + (tgt/np.linalg.norm(tgt)) * args.target_radius

        # ORACLE CHECK
        oracle_dist, _ = oracle_nbrs.kneighbors(tgt.reshape(1, -1))
        od = float(oracle_dist[0][0])
        oracle_dists.append(od)
        if od <= args.success_tol: oracle_successes += 1

        start_dist = np.linalg.norm(xyz - tgt)
        hit_step = None
        path_len = 0.0
        
        for t in range(1, args.max_steps + 1):
            dist = np.linalg.norm(xyz - tgt)
            if dist <= args.success_tol:
                hit_step = t - 1; break
                
            min_step = args.snap_min_step_near if dist < args.near_radius else args.snap_min_step
            z, xyz = run_beam_search(z, tgt, args.beam_width, args.beam_horizon, min_step)
            
            if np.linalg.norm(xyz - tgt) <= args.success_tol:
                hit_step = t; break
                
        final_dist = np.linalg.norm(xyz - tgt)
        if hit_step:
            for k in success_at: 
                if hit_step <= k: success_at[k] += 1
        
        rows.append([ep, start_dist, final_dist, od, hit_step if hit_step else -1, 1 if hit_step else 0])
        
        # Stats update
        succ_rate = success_at[40]/(ep+1)
        oracle_rate = oracle_successes/(ep+1)
        iterator.set_postfix({"Succ": f"{succ_rate:.1%}", "Oracle": f"{oracle_rate:.1%}"})

    # Report
    sd = np.array(rows)[:, 1]
    fd = np.array(rows)[:, 2]
    od = np.array(oracle_dists)
    
    print(f"\nFinal Results:")
    print(f"  Agent Success: {success_at[40]}/{args.episodes} ({success_at[40]/args.episodes:.3f})")
    print(f"  Oracle Success: {oracle_successes}/{args.episodes} ({oracle_successes/args.episodes:.3f})")
    print(f"  Final Dist Mean: {fd.mean():.3f} (Oracle Mean: {od.mean():.3f})")
    
    out_csv = os.path.join(args.outdir, f"rollouts_beam_w{args.beam_width}_h{args.beam_horizon}_oracle.csv")
    np.savetxt(out_csv, np.array(rows), delimiter=",", header="ep,start,final,oracle,hit,succ", comments="")
    print(f"[ok] Saved {out_csv}")

if __name__ == "__main__":
    main()