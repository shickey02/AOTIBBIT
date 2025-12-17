#!/usr/bin/env python3
# phase32h_operator_group_tests.py
#
# Phase 32H: Operator composition & "group-ish" structure tests in latent space.
#
# Writes UTF-8 reports (Windows-safe).

import os, argparse
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)

def load_labels_npz(path: str):
    lab = np.load(path)
    need = ["dx","dy","dz"]
    for k in need:
        if k not in lab:
            raise ValueError(f"labels.npz missing key '{k}'. Has: {list(lab.keys())[:20]}")
    xyz = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)  # [N,3]
    return xyz, lab

def maybe_zscore(Z: np.ndarray, zscore_stats_path: str|None):
    if not zscore_stats_path:
        return Z, None
    st = np.load(zscore_stats_path)
    mean_key = "mean" if "mean" in st else ("mu" if "mu" in st else None)
    std_key  = "std"  if "std"  in st else ("sigma" if "sigma" in st else None)
    if mean_key is None or std_key is None:
        raise ValueError(f"zscore_stats missing mean/std keys. Keys: {list(st.keys())}")
    mu = st[mean_key].astype(np.float32).reshape(1,-1)
    sd = st[std_key].astype(np.float32).reshape(1,-1)
    Zs = (Z - mu) / (sd + 1e-8)
    return Zs, (mu, sd)

def ridge_fit(Z: np.ndarray, Y: np.ndarray, ridge=1e-4):
    N, D = Z.shape
    Z1 = np.concatenate([Z, np.ones((N,1), dtype=Z.dtype)], axis=1)  # [N,D+1]
    A = Z1.T @ Z1
    lam = ridge * np.eye(D+1, dtype=Z.dtype)
    lam[-1,-1] = 0.0
    A += lam
    B = Z1.T @ Y
    theta = np.linalg.solve(A, B)  # [D+1,3]
    W = theta[:-1,:]
    b = theta[-1,:]
    return W.astype(np.float32), b.astype(np.float32)

def affine_predict(Z: np.ndarray, W: np.ndarray, b: np.ndarray):
    return (Z @ W + b).astype(np.float32)

def l2(x, axis=-1):
    return np.sqrt(np.maximum(1e-12, np.sum(x*x, axis=axis)))

def summarize_err(name, e: np.ndarray):
    e = e.reshape(-1).astype(np.float64)
    return f"{name}: mean={e.mean():.4f} med={np.median(e):.4f} p90={np.percentile(e,90):.4f} p99={np.percentile(e,99):.4f} max={e.max():.4f}"

def build_operator_prototypes(Z: np.ndarray, XYZ: np.ndarray,
                              step: float, tol: float,
                              n_pairs: int, seed: int):
    set_seed(seed)
    N, D = Z.shape
    i = np.random.randint(0, N, size=(n_pairs,), dtype=np.int64)
    j = np.random.randint(0, N, size=(n_pairs,), dtype=np.int64)

    dxyz = (XYZ[j] - XYZ[i]).astype(np.float32)  # [P,3]
    dz   = (Z[j]   - Z[i]).astype(np.float32)    # [P,D]

    ops = {}
    meta = {}

    def sel_axis(axis: int, sign: float):
        target = sign * step
        a = dxyz[:,axis]
        ok_main = np.abs(a - target) <= tol
        other = [k for k in [0,1,2] if k != axis]
        ok_off = (np.abs(dxyz[:,other[0]]) <= tol) & (np.abs(dxyz[:,other[1]]) <= tol)
        return ok_main & ok_off

    names = [
        ("+x", 0, +1.0), ("-x", 0, -1.0),
        ("+y", 1, +1.0), ("-y", 1, -1.0),
        ("+z", 2, +1.0), ("-z", 2, -1.0),
    ]

    for nm, ax, sgn in names:
        sel = sel_axis(ax, sgn)
        cnt = int(sel.sum())
        if cnt < 200:
            ops[nm] = None
            meta[nm] = {"count": cnt}
            continue
        proto = dz[sel].mean(axis=0)
        ops[nm] = proto.astype(np.float32)
        meta[nm] = {"count": cnt, "dz_norm": float(l2(proto))}
    return ops, meta

def nn_snapper(Z: np.ndarray):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(Z)
    def snap(zq: np.ndarray):
        dist, idx = nn.kneighbors(zq, return_distance=True)
        return Z[idx[:,0]], idx[:,0], dist[:,0]
    return snap

def apply_op(Z0: np.ndarray, op_dz: np.ndarray, snap_fn=None):
    zt = (Z0 + op_dz.reshape(1,-1)).astype(np.float32)
    if snap_fn is None:
        return zt, None, None
    zn, idx, dist = snap_fn(zt)
    return zn.astype(np.float32), idx, dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_pairs", type=int, default=600000)
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.06)

    ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--zscore_stats", default=None)

    ap.add_argument("--ridge", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    Z = np.load(args.latents).astype(np.float32)       # [N,D]
    XYZ, lab = load_labels_npz(args.labels)            # [N,3]
    N, D = Z.shape
    print(f"N={N} D={D}")

    Zs, zstats = maybe_zscore(Z, args.zscore_stats)
    print("[zscore] enabled" if zstats is not None else "[zscore] disabled")

    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = max(2000, int(0.10 * N))
    tr = idx[n_val:]
    va = idx[:n_val]

    W, b = ridge_fit(Zs[tr], XYZ[tr], ridge=args.ridge)
    pred_va = affine_predict(Zs[va], W, b)
    r2 = 1.0 - (np.sum((pred_va-XYZ[va])**2) / (np.sum((XYZ[va]-XYZ[va].mean(axis=0, keepdims=True))**2)+1e-8))
    print(f"[affine z->xyz] R2_total={float(r2):.4f}")

    ops, meta = build_operator_prototypes(
        Zs, XYZ,
        step=args.step, tol=args.tol,
        n_pairs=args.n_pairs, seed=args.seed
    )

    snap_fn = nn_snapper(Zs) if args.snap else None
    print("[snap] enabled (nearest dataset latent)" if args.snap else "[snap] disabled")

    eval_idx = np.random.randint(0, N, size=(args.n_eval,), dtype=np.int64)
    Z0 = Zs[eval_idx]
    X0hat = affine_predict(Z0, W, b)

    def eval_two_step(op1, op2):
        z1, _, _ = apply_op(Z0, op1, snap_fn=snap_fn)
        z2, _, _ = apply_op(z1, op2, snap_fn=snap_fn)
        return affine_predict(z2, W, b)

    def eval_one_step(op):
        z1, _, _ = apply_op(Z0, op, snap_fn=snap_fn)
        return affine_predict(z1, W, b)

    lines = []
    lines.append("PHASE 32H — Operator group-ish tests")
    lines.append(f"N={N} D={D}  step={args.step} tol={args.tol}  n_pairs={args.n_pairs}  n_eval={args.n_eval}  snap={args.snap}")
    lines.append("")

    lines.append("Operator prototypes (counts / ||dz||):")
    ok_ops = {}
    for k in ["+x","-x","+y","-y","+z","-z"]:
        m = meta.get(k, {})
        if ops.get(k) is None:
            lines.append(f"  {k}: MISSING (count={m.get('count',0)})  -> increase --n_pairs or loosen --tol")
        else:
            lines.append(f"  {k}: count={m.get('count',0)}  dz_norm={m.get('dz_norm',0.0):.4f}")
            ok_ops[k] = ops[k]
    lines.append("")

    if "+x" not in ok_ops or "-x" not in ok_ops or "+y" not in ok_ops or "-y" not in ok_ops:
        lines.append("ERROR: missing required x/y operators. Try: --n_pairs 1200000  and/or --tol 0.08")
        with open(os.path.join(args.outdir, "report_32h.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("[ok] wrote report_32h.txt (but missing ops)")
        return

    op_xy = (ok_ops["+x"] + ok_ops["+y"]).astype(np.float32)
    x_seq = eval_two_step(ok_ops["+x"], ok_ops["+y"])
    x_one = eval_one_step(op_xy)
    err_add = l2((x_seq - x_one), axis=1)
    lines.append("Additivity: apply(+x then +y) vs apply(+(x+y) combined)")
    lines.append("  " + summarize_err("norm(diff)", err_add))
    lines.append("")

    x_xy = eval_two_step(ok_ops["+x"], ok_ops["+y"])
    x_yx = eval_two_step(ok_ops["+y"], ok_ops["+x"])
    err_comm = l2((x_xy - x_yx), axis=1)
    lines.append("Commutativity: apply(+x then +y) vs apply(+y then +x)")
    lines.append("  " + summarize_err("norm(diff)", err_comm))
    lines.append("")

    x_back = eval_two_step(ok_ops["+x"], ok_ops["-x"])
    err_cycle = l2((x_back - X0hat), axis=1)
    lines.append("Cycle: apply(+x then -x) vs identity (in predicted xyz)")
    lines.append("  " + summarize_err("norm(diff)", err_cycle))
    lines.append("")

    x_plus = eval_one_step(ok_ops["+x"])
    d_hat = l2((x_plus - X0hat), axis=1)
    lines.append("Step magnitude consistency for +x (predicted xyz displacement)")
    lines.append("  " + summarize_err("norm(dxyz_hat)", d_hat))
    lines.append(f"  target step={args.step}")
    lines.append("")

    P = 200000
    ii = np.random.randint(0, N, size=(P,), dtype=np.int64)
    jj = np.random.randint(0, N, size=(P,), dtype=np.int64)
    dz = Zs[jj] - Zs[ii]
    dxyz = XYZ[jj] - XYZ[ii]
    nz = l2(dz, axis=1)
    nw = l2(dxyz, axis=1)
    nzm = nz - nz.mean()
    nwm = nw - nw.mean()
    corr = float((nzm @ nwm) / (np.sqrt((nzm @ nzm) * (nwm @ nwm)) + 1e-12))
    lines.append("Metric correlation on random pairs: corr(||dz||, ||dxyz||)")
    lines.append(f"  corr={corr:.4f}")
    lines.append("")

    out_npz = {
        "step": np.array([args.step], dtype=np.float32),
        "tol": np.array([args.tol], dtype=np.float32),
        "W": W, "b": b,
        "snap": np.array([1 if args.snap else 0], dtype=np.int32),
    }
    for k,v in ok_ops.items():
        out_npz[f"op_{k.replace('+','p').replace('-','m')}"] = v.astype(np.float32)
        out_npz[f"count_{k.replace('+','p').replace('-','m')}"] = np.array([meta[k]["count"]], dtype=np.int32)

    np.savez(os.path.join(args.outdir, "operators_32h.npz"), **out_npz)

    with open(os.path.join(args.outdir, "report_32h.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[ok] wrote report_32h.txt and operators_32h.npz")

if __name__ == "__main__":
    main()
