#!/usr/bin/env python3
# phase30a_extract_node_vectors.py
#
# Robust extraction of node base vectors from BBIT transport cache (phase15f schema).
# Handles bases[node] as list OR dict-wrapped vector OR dict-of-index->value.

import os, json, argparse, csv

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def coerce_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "item"):  # numpy scalar
        return float(x.item())
    if isinstance(x, list) and len(x) == 1:
        return coerce_float(x[0])
    if isinstance(x, dict):
        for k in ("value", "v", "val"):
            if k in x:
                return coerce_float(x[k])
    raise TypeError(f"cannot coerce {type(x)} -> float")

def unwrap_vector(obj, ctx=""):
    """
    Returns a python list of floats from many possible schema shapes.
    """
    # Case 1: already a list
    if isinstance(obj, list):
        return [coerce_float(x) for x in obj]

    # Case 2: dict wrapper with an obvious vector key
    if isinstance(obj, dict):
        # Most common wrappers
        for k in ("vec", "vector", "basis", "base", "values", "data", "coeffs"):
            if k in obj:
                return unwrap_vector(obj[k], ctx=f"{ctx}.{k}" if ctx else k)

        # Sometimes nested one more level: bases[node] = {"something": {"vec":[...]}}
        # Try to find the first value that looks vector-like.
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                try:
                    vec = unwrap_vector(v, ctx=f"{ctx}.{k}" if ctx else k)
                    # sanity: vectors should be "long" (e.g. 256)
                    if len(vec) >= 8:
                        return vec
                except Exception:
                    pass

        # Case 3: dict-of-index->value (keys "0","1",... or ints)
        # If keys look numeric, sort by index.
        keys = list(obj.keys())
        def is_numkey(s):
            if isinstance(s, int):
                return True
            if isinstance(s, str) and s.strip().lstrip("-").isdigit():
                return True
            return False

        if keys and all(is_numkey(k) for k in keys):
            # sort numerically
            def to_int(k):
                return int(k) if isinstance(k, (str, int)) else int(str(k))
            items = sorted(obj.items(), key=lambda kv: to_int(kv[0]))
            return [coerce_float(v) for _, v in items]

    raise TypeError(f"[unwrap_vector] {ctx}: unsupported type {type(obj)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    if "bases" not in cache or not isinstance(cache["bases"], dict):
        raise SystemExit("[error] cache missing dict key 'bases'")

    bases = cache["bases"]
    nodes = sorted(bases.keys())

    vectors = {}
    dim = None

    for n in nodes:
        raw = bases[n]
        try:
            vec = unwrap_vector(raw, ctx=f"bases.{n}")
        except Exception as e:
            # print helpful detail
            raise SystemExit(f"[error] cannot unwrap bases.{n}: {e}")

        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            raise SystemExit(f"[error] inconsistent dims at {n}: {len(vec)} vs {dim}")

        vectors[n] = vec

    # ---- write nodes.csv
    nodes_csv = os.path.join(args.outdir, "phase30a_nodes.csv")
    with open(nodes_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node"] + [f"d{i}" for i in range(dim)])
        for n in nodes:
            w.writerow([n] + vectors[n])

    # ---- write edges.csv
    edges_csv = os.path.join(args.outdir, "phase30a_edges.csv")
    planes_seen = set()
    n_edges = 0

    with open(edges_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "plane", "has_map"])

        for k, v in cache.get("transport_maps", {}).items():
            if "__to__" not in k:
                continue
            src, dst = k.split("__to__", 1)
            if not isinstance(v, dict):
                continue
            for plane, obj in v.items():
                planes_seen.add(plane)
                w.writerow([src, dst, plane, 1 if obj is not None else 0])
                n_edges += 1

    # ---- summary
    summary = os.path.join(args.outdir, "phase30a_summary.txt")
    with open(summary, "w") as f:
        f.write(f"nodes={len(nodes)}\n")
        f.write(f"dim={dim}\n")
        f.write(f"edges(rows)={n_edges}\n")
        f.write(f"planes={sorted(list(planes_seen))}\n")
        f.write(f"nodes={nodes}\n")

    print(f"[ok] wrote: {nodes_csv}")
    print(f"[ok] wrote: {edges_csv}")
    print(f"[ok] wrote: {summary}")
    print(f"[info] nodes={len(nodes)} dim={dim} planes={sorted(list(planes_seen))}")

if __name__ == "__main__":
    main()
