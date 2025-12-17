import csv

path = r"E:\BBIT\outputs_edges_relternary256_phase15\phase26g_bifurcate_w035\phase26g_rollouts_wN=0.350.csv"
target_cycle = "between_boundary_clear|between_deep_clear|overlap_deep_only|overlap_boundary_only|left_clean"

def split_path(p):
    return [x.strip() for x in (p or "").split("->") if x.strip()]

hits = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        if (r.get("cycle_key") or "").strip() == target_cycle:
            hits.append(r)

print(f"[found] {len(hits)} hits for target_cycle")
for r in hits:
    nodes = split_path(r.get("path"))
    cyc = target_cycle.split("|")

    # find where the cycle starts in the path (first occurrence of cyc[0] that aligns)
    # simplest: locate the last node which repeats to close the loop
    # in your case, path ends with left_clean, so cycle is everything from first left_clean to end-1
    if len(nodes) >= 2 and nodes[0] == nodes[-1]:
        prefix = []
        cycle_nodes = nodes[:-1]
    else:
        # fallback: just show full path
        prefix = nodes
        cycle_nodes = []

    print("\n--- HIT ---")
    print("start:", r.get("start"))
    print("status:", r.get("status"))
    print("steps:", r.get("steps"))
    print("cycle_len:", r.get("cycle_len"))
    print("cycle_key:", r.get("cycle_key"))
    print("path:", r.get("path"))
    print("prefix:", "->".join(prefix) if prefix else "(none)")
    print("cycle_nodes:", "->".join(cycle_nodes) if cycle_nodes else "(unknown)")
