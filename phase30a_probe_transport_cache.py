#!/usr/bin/env python3
# phase30a_probe_transport_cache.py
#
# Prints the internal schema of phase15f_transport_cache.json so we can locate
# the per-node neighbor/candidate lists used by 28h/29c.

import json, argparse
from collections import deque

def short(x, n=140):
    s = repr(x)
    return s if len(s) <= n else s[:n] + " ..."

def summarize(obj, max_keys=8):
    if isinstance(obj, dict):
        ks = list(obj.keys())
        return f"dict(len={len(ks)}) keys={ks[:max_keys]}" + (" ..." if len(ks) > max_keys else "")
    if isinstance(obj, list):
        return f"list(len={len(obj)}) first={short(obj[0]) if obj else 'EMPTY'}"
    return f"{type(obj).__name__}: {short(obj)}"

def find_candidate_like_paths(root, max_nodes=60000):
    """
    Walk JSON and find paths that look like:
      - list of dicts with float-ish fields
      - dict-of-node -> list(...)
    Returns top hits.
    """
    hits = []
    q = deque([("", root)])
    seen = 0

    while q and seen < max_nodes:
        path, obj = q.popleft()
        seen += 1

        # Heuristic A: dict mapping -> list
        if isinstance(obj, dict) and obj:
            # if values look like lists and keys look like node names
            vals = list(obj.values())
            if all(isinstance(v, list) for v in vals[:10]):
                # sample first non-empty list
                sample_list = next((v for v in vals if isinstance(v, list) and len(v) > 0), None)
                if sample_list is not None:
                    hits.append((path, "dict->list", len(obj), summarize(sample_list)))

            for k, v in obj.items():
                q.append((f"{path}.{k}" if path else str(k), v))

        # Heuristic B: list of dicts
        elif isinstance(obj, list) and obj:
            if isinstance(obj[0], dict):
                # count float-like fields in first dict
                d0 = obj[0]
                floatish = 0
                for kk, vv in d0.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        floatish += 1
                if floatish >= 1:
                    hits.append((path, "list[dict]", len(obj), f"first_keys={list(d0.keys())[:12]} floatish={floatish}"))
            # keep walking into a few elements
            for i, v in enumerate(obj[:5]):
                q.append((f"{path}[{i}]", v))

    # prioritize likely useful:
    # - dict->list paths (node->cands) and list[dict] paths with many entries
    hits_sorted = sorted(hits, key=lambda x: (0 if x[1]=="dict->list" else 1, -x[2]))
    return hits_sorted[:40]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    with open(args.cache, "r") as f:
        cache = json.load(f)

    print("[top] keys:", list(cache.keys()))
    for k in ["anchors", "bases", "transport_maps", "notes"]:
        if k in cache:
            print(f"[top] {k}: {summarize(cache[k])}")

    # Print a little inside transport_maps if present
    if "transport_maps" in cache and isinstance(cache["transport_maps"], dict):
        tm = cache["transport_maps"]
        print("\n[transport_maps] keys:", list(tm.keys()))
        for plane in list(tm.keys())[:6]:
            print(f"[transport_maps.{plane}] {summarize(tm[plane])}")

    print("\n[scan] searching for candidate-like paths...")
    hits = find_candidate_like_paths(cache)
    for path, kind, size, detail in hits:
        print(f" - {kind:9s} size={size:6d} path={path} :: {detail}")

if __name__ == "__main__":
    main()
