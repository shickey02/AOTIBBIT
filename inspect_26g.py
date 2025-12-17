import csv
from collections import Counter

path = r"E:\BBIT\outputs_edges_relternary256_phase15\phase26g_bifurcate_w035\phase26g_rollouts_wN=0.350.csv"

# Nodes you care about (the 5-cycle you saw in the sweep signature)
needle_nodes = {
    "between_boundary_clear",
    "between_deep_clear",
    "overlap_deep_only",
    "overlap_boundary_only",
    "left_clean",
}

def get_cycle_field(row):
    # Try the likely column names in order
    for k in ("attractor_key", "attractor", "cycle", "attractor_cycle", "cycle_key", "attractor_sig"):
        v = row.get(k)
        if v:
            return k, v.strip()
    return None, ""

def parse_nodes_from_cycle_str(s):
    # If it's a signature like: "a|b|c:2|d|e:1", strip counts
    parts = []
    for chunk in s.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # remove ":<count>" if present
        if ":" in chunk:
            chunk = chunk.split(":", 1)[0].strip()
        parts.append(chunk)
    return parts

def is_same_set(nodes, target_set):
    return set(nodes) == set(target_set)

# Load and analyze
cycle_counter = Counter()
raw_examples = {}
contains_needle = Counter()

with open(path, newline="") as f:
    rdr = csv.DictReader(f)
    cols = rdr.fieldnames or []
    print("[columns]", cols)

    for row in rdr:
        colname, cyc = get_cycle_field(row)
        if not cyc:
            continue

        cycle_counter[cyc] += 1
        raw_examples.setdefault(cyc, row)

        nodes = parse_nodes_from_cycle_str(cyc)
        if needle_nodes.issubset(set(nodes)):
            contains_needle[cyc] += 1

print("\n[unique attractors]", len(cycle_counter))
print("\nTOP 25 attractors by frequency:")
for cyc, c in cycle_counter.most_common(25):
    print(f"  count={c:4d}  {cyc}")

print("\nAttractors that CONTAIN all 5 needle nodes (any order / may include extras):")
if not contains_needle:
    print("  (none)")
else:
    for cyc, c in contains_needle.most_common(25):
        print(f"  count={c:4d}  {cyc}")

# If you want to print example paths for any matching attractor:
print("\nExample path rows for any 'contains needle' attractor:")
for cyc in list(contains_needle.keys())[:5]:
    r = raw_examples[cyc]
    print("\n--- EXAMPLE ---")
    print("cycle:", cyc)
    print("start:", r.get("start"))
    print("status:", r.get("status"))
    print("steps:", r.get("steps"))
    print("path:", r.get("path"))
