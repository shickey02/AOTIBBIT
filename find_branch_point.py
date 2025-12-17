import csv

base_path = r"E:\BBIT\outputs_edges_relternary256_phase15\phase26g_bifurcate\phase26g_rollouts_wN=0.000.csv"
nov_path  = r"E:\BBIT\outputs_edges_relternary256_phase15\phase26g_bifurcate_w035\phase26g_rollouts_wN=0.350.csv"

def split_path(p):
    return [x.strip() for x in (p or "").split("->") if x.strip()]

def load_by_start(path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            start = (r.get("start") or "").strip()
            d[start] = r
    return d

base = load_by_start(base_path)
nov  = load_by_start(nov_path)

starts = sorted(set(base.keys()) & set(nov.keys()))
print("[starts]", starts)

for s in starts:
    bp = split_path(base[s].get("path"))
    np = split_path(nov[s].get("path"))

    # Find first index where they differ (node-by-node)
    m = min(len(bp), len(np))
    j = None
    for i in range(m):
        if bp[i] != np[i]:
            j = i
            break

    print("\n== start:", s, "==")
    print("BASE cycle_key:", base[s].get("cycle_key"))
    print("NOV  cycle_key:", nov[s].get("cycle_key"))

    if j is None:
        if len(bp) == len(np):
            print("DIVERGENCE: none (paths identical)")
        else:
            print("DIVERGENCE: none in common prefix, but path lengths differ")
        print("base_path:", "->".join(bp))
        print("nov_path :", "->".join(np))
        continue

    # Show local branching context
    print(f"DIVERGENCE at step i={j}")
    print("base_prefix:", "->".join(bp[:j]))
    print("nov_prefix :", "->".join(np[:j]))
    print("base_next  :", bp[j] if j < len(bp) else "(end)")
    print("nov_next   :", np[j] if j < len(np) else "(end)")

    # If you want the *branch edge* (prev -> next):
    prev = bp[j-1] if j-1 >= 0 else None
    if prev:
        print("base_edge :", f"{prev} -> {bp[j]}")
        print("nov_edge  :", f"{prev} -> {np[j]}")
