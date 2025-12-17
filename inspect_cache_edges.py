import json, math
from collections import Counter, defaultdict

def _is_number(x):
    try:
        float(x); return True
    except: return False

def _as_float(x, default=0.0):
    try: return float(x)
    except: return default

def extract_edges_from_transport_maps(data):
    edges = []
    tms = data.get("transport_maps", {})
    if not isinstance(tms, dict):
        return edges

    for plane, tm in tms.items():
        plane = str(plane)

        # adjacency dict-of-dicts
        if isinstance(tm, dict):
            meta = {"nodes","names","labels","mat","M","matrix","transport","T","info","notes"}
            node_keys = [k for k in tm.keys() if isinstance(k,str) and k not in meta]
            if node_keys and sum(isinstance(tm[k], dict) for k in node_keys) >= max(1, len(node_keys)//2):
                for a in node_keys:
                    nbrs = tm[a]
                    if not isinstance(nbrs, dict): 
                        continue
                    for b, val in nbrs.items():
                        if a == b or not _is_number(val):
                            continue
                        transport = _as_float(val, 0.0)
                        # same mapping as your current loader
                        if 0.0 <= transport <= 1.0:
                            sim = 1.0 - transport
                        else:
                            sim = math.exp(-max(0.0, transport))
                            transport = 1.0 - sim
                        edges.append((str(a), str(b), plane, sim, transport))
    return edges

def main():
    path = r"E:\BBIT\outputs_edges_relternary256_phase15\phase15f_transport_cache.json"
    with open(path, "r") as f:
        data = json.load(f)

    edges = extract_edges_from_transport_maps(data)
    print("[edges]", len(edges))
    if not edges:
        print("[err] no edges derived from transport_maps")
        print("transport_maps keys:", list(data.get("transport_maps", {}).keys()))
        return

    plane_counts = Counter(p for _,_,p,_,_ in edges)
    print("\n[planes]")
    for p,c in plane_counts.most_common(50):
        print(f"  {p}: {c}")

    # how many outgoing per node (raw, no filtering)
    out_counts = Counter(a for a,_,_,_,_ in edges)
    print("\n[top outgoing nodes]")
    for a,c in out_counts.most_common(30):
        print(f"  {a}: {c}")

    # specifically check left_clean exact match vs fuzzy
    print("\n[left_clean exact outgoing]", out_counts.get("left_clean", 0))
    near = [a for a in out_counts.keys() if "left" in a.lower()]
    if near:
        print("[nodes containing 'left']")
        for a in sorted(near)[:50]:
            print(" ", repr(a))

    # sim distribution (so we know if min_edge_sim is killing everything)
    sims = [sim for *_, sim, _ in edges]
    print("\n[sim stats]")
    print("  min:", min(sims), "max:", max(sims))
    # quick histogram-ish
    buckets = defaultdict(int)
    for s in sims:
        buckets[int(max(0,min(9, s*10)))] += 1
    print("  buckets (0.0-0.1 ... 0.9-1.0):")
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        print(f"   {lo:.1f}-{hi:.1f}: {buckets[i]}")

if __name__ == "__main__":
    main()
