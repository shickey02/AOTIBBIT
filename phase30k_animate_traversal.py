#!/usr/bin/env python3
# phase30k_animate_traversal.py  (v2 drop-in)
#
# Dark-mode traversal GIFs with:
# - white opaque nodes/edges
# - glowing traversal trail
# - on completion: erase the rest of the net, flash ONLY the path
# - invert to light mode and REVEAL a grey/black manifold (tube mesh) built from the path
# - batch mode: generate long random-walk paths and export multiple GIFs
# - optional mesh export (OBJ/PLY) per GIF
#
# Requires: numpy, pandas, matplotlib
# (No ffmpeg needed; outputs GIF via PillowWriter)

import os, math, argparse, random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless + avoids interactive memory issues

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------------------------
# Utils
# ------------------------------

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

def ensure_dir(p):
    if p and os.path.dirname(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)

def read_nodes3d(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["node", "x", "y", "z"]
    for k in need:
        if k not in cols:
            raise ValueError(f"nodes3d missing required column '{k}'. have={list(df.columns)}")
    node_col = cols["node"]
    xcol, ycol, zcol = cols["x"], cols["y"], cols["z"]

    nodes = df[node_col].astype(str).tolist()
    XYZ = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)
    return nodes, XYZ, df

def read_edges(path):
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    # common patterns
    for cand in [("src","dst"), ("u","v"), ("a","b"), ("from","to")]:
        if cand[0] in lower and cand[1] in lower:
            s = df[lower[cand[0]]].astype(str).tolist()
            t = df[lower[cand[1]]].astype(str).tolist()
            return list(zip(s, t)), df

    # fallback: first two columns
    if df.shape[1] >= 2:
        s = df.iloc[:,0].astype(str).tolist()
        t = df.iloc[:,1].astype(str).tolist()
        return list(zip(s, t)), df

    raise ValueError(f"edges csv must have at least 2 columns. got={list(df.columns)}")

def resolve_node_names(wanted, all_nodes):
    """
    Resolve user-supplied names by:
    1) exact match
    2) prefix match (e.g. 'left_clean' -> 'left_clean__00')
    3) contains match
    """
    all_nodes = list(all_nodes)
    out = []
    for w in wanted:
        if w in all_nodes:
            out.append(w); continue
        # prefix
        pref = [n for n in all_nodes if n.startswith(w)]
        if len(pref) > 0:
            out.append(pref[0])
            print(f"[resolve] '{w}' -> '{pref[0]}' (prefix match)")
            continue
        # contains
        mid = [n for n in all_nodes if w in n]
        if len(mid) > 0:
            out.append(mid[0])
            print(f"[resolve] '{w}' -> '{mid[0]}' (contains match)")
            continue
        print(f"[warn] could not resolve node '{w}'")
    return out

def build_adjacency(edges, keep_nodes_set):
    adj = {n: [] for n in keep_nodes_set}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    # de-dup
    for k in adj:
        if adj[k]:
            adj[k] = sorted(list(set(adj[k])))
    return adj

def random_walk(adj, start, length=200, allow_revisit=True, rng=None):
    if rng is None:
        rng = random.Random()
    if start not in adj:
        raise ValueError(f"start node '{start}' not in adjacency.")
    path = [start]
    seen = {start}
    cur = start
    for _ in range(length-1):
        nbrs = adj.get(cur, [])
        if not nbrs:
            break
        if allow_revisit:
            nxt = rng.choice(nbrs)
        else:
            choices = [n for n in nbrs if n not in seen]
            if not choices:
                break
            nxt = rng.choice(choices)
            seen.add(nxt)
        path.append(nxt)
        cur = nxt
    return path

# ------------------------------
# Tube mesh from polyline
# ------------------------------

def build_tube_from_polyline(P, radius=0.22, sides=16, cap_ends=True):
    """
    Build a watertight tube mesh around a 3D polyline.
    Returns (V, F) where V: (Nv,3), F: (Nf,3) ints (0-based).
    """
    P = np.asarray(P, dtype=np.float64)
    N = P.shape[0]
    if N < 2:
        raise ValueError("Need >=2 points for tube.")

    # Tangents
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0]    = P[1] - P[0]
    T[-1]   = P[-1] - P[-2]
    T = np.array([normalize(t) for t in T], dtype=np.float64)

    # Choose a stable reference up
    up0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up0, T[0]))) > 0.9:
        up0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Build a smoothly varying normal/binormal frame
    Nrm = np.zeros_like(P)
    Bin = np.zeros_like(P)

    n0 = np.cross(T[0], up0)
    if np.linalg.norm(n0) < 1e-9:
        n0 = np.cross(T[0], np.array([1.0, 0.0, 0.0]))
    n0 = normalize(n0)
    b0 = normalize(np.cross(T[0], n0))
    Nrm[0], Bin[0] = n0, b0

    # Parallel transport-ish
    for i in range(1, N):
        ti_prev = T[i-1]
        ti = T[i]
        axis = np.cross(ti_prev, ti)
        axn = np.linalg.norm(axis)
        if axn < 1e-9:
            Nrm[i] = Nrm[i-1]
            Bin[i] = Bin[i-1]
            continue
        axis = axis / axn
        angle = math.acos(np.clip(float(np.dot(ti_prev, ti)), -1.0, 1.0))

        # Rodrigues rotate previous frame vectors around axis by angle
        def rot(v):
            v = np.asarray(v, dtype=np.float64)
            return (v*math.cos(angle) +
                    np.cross(axis, v)*math.sin(angle) +
                    axis*float(np.dot(axis, v))*(1.0-math.cos(angle)))

        n = rot(Nrm[i-1])
        b = rot(Bin[i-1])

        # Re-orthonormalize
        n = normalize(n - float(np.dot(n, ti))*ti)
        b = normalize(np.cross(ti, n))
        Nrm[i], Bin[i] = n, b

    # Build rings
    angles = np.linspace(0, 2*np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (sides,2)

    V = []
    for i in range(N):
        for j in range(sides):
            off = radius * (ring[j,0]*Nrm[i] + ring[j,1]*Bin[i])
            V.append(P[i] + off)
    V = np.asarray(V, dtype=np.float64)

    # Stitch triangles between rings
    F = []
    def vid(i, j):  # ring i, side j
        return i*sides + (j % sides)

    for i in range(N-1):
        for j in range(sides):
            a = vid(i, j)
            b = vid(i, j+1)
            c = vid(i+1, j)
            d = vid(i+1, j+1)
            # two tris (a,c,b) and (b,c,d)
            F.append([a, c, b])
            F.append([b, c, d])

    # Caps (fan triangulation)
    if cap_ends:
        # add centers
        c0 = len(V); V = np.vstack([V, P[0]])
        c1 = len(V); V = np.vstack([V, P[-1]])

        # start cap (note orientation)
        for j in range(sides):
            a = vid(0, j)
            b = vid(0, j+1)
            F.append([c0, b, a])

        # end cap
        base = (N-1)*sides
        for j in range(sides):
            a = base + j
            b = base + ((j+1) % sides)
            F.append([c1, a, b])

    F = np.asarray(F, dtype=np.int32)
    return V, F

def face_shades(V, F, view_dir=np.array([0.4, 0.2, 1.0], dtype=np.float64)):
    """
    Grey shade per face based on normal·view_dir (angle-dependent).
    Returns array (M,) in [0,1], where 0=black, 1=light grey.
    """
    view_dir = normalize(view_dir)
    tri = V[F]  # (M,3,3)
    n = np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0])
    n = np.array([normalize(x) for x in n], dtype=np.float64)
    d = np.clip(n @ view_dir, -1.0, 1.0)
    # map to nice contrast (front faces lighter)
    s = (d + 1.0) * 0.5  # [0,1]
    s = 0.10 + 0.80 * (s**0.8)  # avoid pure black/white
    return np.clip(s, 0.0, 1.0)

def write_obj(path, V, F):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write("# phase30k path tube\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            a, b, c = tri.tolist()
            f.write(f"f {a+1} {b+1} {c+1}\n")

def write_ply(path, V, F):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in V:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            a, b, c = tri.tolist()
            f.write(f"3 {a} {b} {c}\n")

# ------------------------------
# Rendering / Animation
# ------------------------------

def set_dark(ax):
    ax.set_facecolor((0,0,0))
    ax.figure.set_facecolor((0,0,0))
    ax.w_xaxis.set_pane_color((0,0,0,0))
    ax.w_yaxis.set_pane_color((0,0,0,0))
    ax.w_zaxis.set_pane_color((0,0,0,0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

def set_light(ax):
    ax.set_facecolor((1,1,1))
    ax.figure.set_facecolor((1,1,1))
    ax.w_xaxis.set_pane_color((1,1,1,0))
    ax.w_yaxis.set_pane_color((1,1,1,0))
    ax.w_zaxis.set_pane_color((1,1,1,0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

def set_equal_3d(ax, X):
    # X: (N,3)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    c = 0.5*(mins+maxs)
    r = 0.5*np.max(maxs-mins)
    if not np.isfinite(r) or r <= 0:
        r = 1.0
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)

def build_path_xyz(path_nodes, node_to_idx, XYZ):
    idxs = [node_to_idx[n] for n in path_nodes]
    return XYZ[idxs]

def draw_base_network(ax, XYZ, edge_pairs_idx,
                      node_size=10, edge_lw=0.75,
                      node_alpha=0.85, edge_alpha=0.22):
    # edges
    for i, j in edge_pairs_idx:
        a = XYZ[i]; b = XYZ[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                linewidth=edge_lw, alpha=edge_alpha, color=(1,1,1))
    # nodes
    ax.scatter(XYZ[:,0], XYZ[:,1], XYZ[:,2],
               s=node_size, alpha=node_alpha, c=[(1,1,1)])

def draw_glow_path(ax, P, upto, trail=80, glow_strength=3.0):
    """
    P: (L,3), upto: current index (inclusive)
    """
    lo = max(0, upto-trail)
    seg = P[lo:upto+1]
    if len(seg) < 2:
        return

    # "glow" as multiple strokes
    # outer
    for k in [5.0, 3.5, 2.2]:
        a = 0.06 * glow_strength * (6.0/k)
        ax.plot(seg[:,0], seg[:,1], seg[:,2], linewidth=k, alpha=a, color=(1,1,1))
    # core
    ax.plot(seg[:,0], seg[:,1], seg[:,2], linewidth=1.8, alpha=0.95, color=(1,1,1))

def add_mesh_collection(ax, V, F, light_mode=True, azim=None, elev=None):
    # Compute per-face greys depending on view direction
    # Use an approximate view direction based on current camera angles
    if azim is None: azim = ax.azim
    if elev is None: elev = ax.elev

    # crude view dir from angles (matplotlib camera convention-ish)
    az = math.radians(azim)
    el = math.radians(elev)
    view_dir = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)], dtype=np.float64)

    shades = face_shades(V, F, view_dir=view_dir)
    # convert to RGB greys
    if light_mode:
        # darker in light mode (so it reads)
        greys = np.stack([shades*0.35]*3, axis=1)
    else:
        greys = np.stack([shades*0.75]*3, axis=1)

    tris = V[F]  # (M,3,3)
    poly = Poly3DCollection(tris, linewidths=0.0)
    poly.set_facecolor(greys)
    poly.set_edgecolor((0,0,0,0))
    poly.set_alpha(0.98)
    ax.add_collection3d(poly)
    return poly

def make_flash_reveal_gif(
    out_path,
    XYZ, edge_pairs_idx,
    path_xyz,
    interval_ms=95,
    trail=85,
    flash_frames=14,
    reveal_frames=55,
    tube_radius=0.20,
    tube_sides=14,
    erase_net_on_complete=True,
    spin_reveal=True
):
    ensure_dir(out_path)

    # Build tube mesh once
    V, F = build_tube_from_polyline(path_xyz, radius=tube_radius, sides=tube_sides, cap_ends=True)

    L = path_xyz.shape[0]
    travel_frames = max(2, L)  # one per node step
    total_frames = travel_frames + flash_frames + reveal_frames

    fig = plt.figure(figsize=(8.0, 6.5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # initial camera
    ax.view_init(elev=18, azim=38)

    # bounds should include both net and mesh
    all_pts = np.vstack([XYZ, V])
    set_equal_3d(ax, all_pts)

    # speed: pre-store base net? (we redraw per frame but keep it light)
    def frame_draw(t):
        ax.cla()

        # PHASE 1: traversal (dark mode, net visible)
        if t < travel_frames:
            set_dark(ax)
            if not erase_net_on_complete:
                draw_base_network(ax, XYZ, edge_pairs_idx)
            else:
                draw_base_network(ax, XYZ, edge_pairs_idx)

            upto = min(L-1, t)
            draw_glow_path(ax, path_xyz, upto=upto, trail=trail, glow_strength=3.2)

        # PHASE 2: flash (dark -> light) ONLY path
        elif t < travel_frames + flash_frames:
            k = t - travel_frames
            # alternate dark/light quickly
            if (k % 2) == 0:
                set_dark(ax)
                # show ONLY path (no net)
                draw_glow_path(ax, path_xyz, upto=L-1, trail=10_000, glow_strength=5.0)
            else:
                set_light(ax)
                # in light mode, invert: path becomes black-ish
                ax.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], linewidth=3.0, alpha=0.95, color=(0,0,0))

        # PHASE 3: reveal (light mode, show mesh, optionally spin)
        else:
            set_light(ax)

            # Hide net; only mesh + faint path silhouette
            # (you asked to erase the rest of the net when complete)
            # optional slow camera spin
            if spin_reveal:
                u = (t - (travel_frames + flash_frames)) / max(1, reveal_frames-1)
                ax.view_init(elev=18 + 4.0*math.sin(2*math.pi*u), azim=38 + 85*u)

            # Add mesh (angle-dependent greys)
            add_mesh_collection(ax, V, F, light_mode=True)

            # subtle path outline (black)
            ax.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], linewidth=1.2, alpha=0.35, color=(0,0,0))

        # re-apply bounds every frame (matplotlib sometimes resets)
        set_equal_3d(ax, all_pts)

    ani = FuncAnimation(fig, lambda i: frame_draw(i), frames=total_frames, interval=interval_ms, blit=False)

    writer = PillowWriter(fps=max(1, int(1000.0/interval_ms)))
    ani.save(out_path, writer=writer)

    plt.close(fig)
    return V, F

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes3d", required=True, help="CSV with node,x,y,z (e.g. phase30b_nodes_3d.csv)")
    ap.add_argument("--edges", required=True, help="CSV with src,dst (e.g. phase30b_edges_knn.csv)")
    ap.add_argument("--out", default=None, help="Output GIF path (single mode)")
    ap.add_argument("--outdir", default=None, help="Output directory (batch mode)")
    ap.add_argument("--interval_ms", type=int, default=95)
    ap.add_argument("--trail", type=int, default=85)
    ap.add_argument("--flash_frames", type=int, default=14)
    ap.add_argument("--reveal_frames", type=int, default=55)
    ap.add_argument("--tube_radius", type=float, default=0.20)
    ap.add_argument("--tube_sides", type=int, default=14)

    # explicit path
    ap.add_argument("--path_nodes", nargs="*", default=None, help="List of node names in order (prefix ok)")

    # batch random walk
    ap.add_argument("--batch", type=int, default=0, help="Number of GIFs to export")
    ap.add_argument("--walk_len", type=int, default=220, help="Random walk length (batch mode)")
    ap.add_argument("--allow_revisit", action="store_true", help="Allow revisiting nodes in random walk")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for batch")
    ap.add_argument("--start_node", default=None, help="Optional start node for batch (prefix ok)")
    ap.add_argument("--spin_reveal", action="store_true", help="Slow camera spin during reveal")

    # mesh export per gif
    ap.add_argument("--export_mesh", action="store_true", help="Export OBJ+PLY alongside each GIF")
    ap.add_argument("--mesh_basename", default="path_mesh", help="Basename for mesh files (no ext)")

    args = ap.parse_args()

    nodes, XYZ, ndf = read_nodes3d(args.nodes3d)
    edges, edf = read_edges(args.edges)

    keep = set(nodes)
    edges_f = [(a,b) for (a,b) in edges if (a in keep and b in keep)]
    if not edges_f:
        raise ValueError("No edges remain after filtering to nodes3d set.")

    node_to_idx = {n:i for i,n in enumerate(nodes)}
    edge_pairs_idx = [(node_to_idx[a], node_to_idx[b]) for (a,b) in edges_f]

    adj = build_adjacency(edges_f, keep)

    def run_one(path_nodes, out_gif, mesh_dir_for_this=None):
        path_nodes = resolve_node_names(path_nodes, nodes)
        # filter invalid
        path_nodes = [n for n in path_nodes if n in node_to_idx]
        if len(path_nodes) < 2:
            raise ValueError("After filtering, path has <2 valid nodes.")
        path_xyz = build_path_xyz(path_nodes, node_to_idx, XYZ)

        V, F = make_flash_reveal_gif(
            out_path=out_gif,
            XYZ=XYZ,
            edge_pairs_idx=edge_pairs_idx,
            path_xyz=path_xyz,
            interval_ms=args.interval_ms,
            trail=args.trail,
            flash_frames=args.flash_frames,
            reveal_frames=args.reveal_frames,
            tube_radius=args.tube_radius,
            tube_sides=args.tube_sides,
            erase_net_on_complete=True,
            spin_reveal=args.spin_reveal
        )

        if args.export_mesh and mesh_dir_for_this:
            os.makedirs(mesh_dir_for_this, exist_ok=True)
            obj_path = os.path.join(mesh_dir_for_this, args.mesh_basename + ".obj")
            ply_path = os.path.join(mesh_dir_for_this, args.mesh_basename + ".ply")
            write_obj(obj_path, V, F)
            write_ply(ply_path, V, F)
            print("[ok] mesh:", obj_path)
            print("[ok] mesh:", ply_path)

    # -------- single mode --------
    if args.batch <= 0:
        if not args.out:
            raise ValueError("Single mode requires --out (GIF path).")
        if not args.path_nodes or len(args.path_nodes) < 2:
            raise ValueError("Single mode requires --path_nodes with >=2 nodes.")
        run_one(args.path_nodes, args.out, mesh_dir_for_this=os.path.dirname(args.out))
        print("[ok] wrote:", args.out)
        return

    # -------- batch mode --------
    if not args.outdir:
        raise ValueError("Batch mode requires --outdir.")
    os.makedirs(args.outdir, exist_ok=True)

    rng = random.Random(args.seed if args.seed is not None else 0)

    # choose start
    start = args.start_node
    if start is None:
        # bias toward higher-degree nodes so walks go longer
        degrees = [(n, len(adj.get(n, []))) for n in nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top = [n for (n,d) in degrees[:max(5, len(degrees)//6)] if d > 0]
        start = rng.choice(top) if top else rng.choice(nodes)
    else:
        start = resolve_node_names([start], nodes)
        start = start[0] if start else rng.choice(nodes)

    for i in range(args.batch):
        # vary start slightly each time for diversity
        if i > 0:
            nbrs = adj.get(start, [])
            if nbrs:
                start_i = rng.choice(nbrs)
            else:
                start_i = start
        else:
            start_i = start

        path_nodes = random_walk(adj, start_i, length=args.walk_len, allow_revisit=args.allow_revisit, rng=rng)
        if len(path_nodes) < 2:
            print(f"[warn] batch {i:03d}: path too short, skipping")
            continue

        out_gif = os.path.join(args.outdir, f"phase30k_traversal_{i:03d}.gif")
        mesh_dir = os.path.join(args.outdir, f"mesh_{i:03d}") if args.export_mesh else None

        try:
            run_one(path_nodes, out_gif, mesh_dir_for_this=mesh_dir)
            print("[ok] wrote:", out_gif, f"(len={len(path_nodes)})")
        except Exception as e:
            print(f"[warn] batch {i:03d} failed:", repr(e))

if __name__ == "__main__":
    main()
