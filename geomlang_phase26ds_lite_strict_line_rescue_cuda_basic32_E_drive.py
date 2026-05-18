# geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py
# Phase 26DS-LITE — strict-line rescue, fast CUDA-friendly smoke/audit run.
#
# Why this exists:
#   Full 26DS can stall for hours because it screens hundreds/thousands of candidates
#   and then sends many candidates through the full strict envelope. DS-LITE keeps the
#   same scoring logic and inherited helpers, but intentionally evaluates a tiny pool
#   and audits only a tiny promoted set. Use this to get a directional result quickly,
#   then send the best overrides into a larger run later.
#
# Typical run:
#   python bbit_geomlang/geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py --device cuda
#
# Faster smoke test:
#   python bbit_geomlang/geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py --device cuda --max-candidates 24 --audit-top 3 --audit-repeats 2

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "26DS-LITE"
TITLE = "Lite strict-line rescue CUDA screen + micro-audit"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DS_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"


def import_by_path(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def micro_envelope(drfast: Any, dr: Any, ov: Dict[str, float], seed: int, repeats: int, use_full_tail: bool) -> List[Any]:
    """
    Build a small audit envelope. The important speed choice is that this does
    NOT evaluate the whole DR/DQ strict envelope by default.
    """
    env: List[Any] = []
    seen = set()

    for r in range(max(1, repeats)):
        for item in drfast.make_screen_envelope(dr, ov, seed=seed + 1009 * r, mode="turbo"):
            key = repr(item)
            if key not in seen:
                env.append(item)
                seen.add(key)

    if use_full_tail:
        # This can still be expensive, so only sample a short, evenly-spaced tail.
        try:
            full = list(dr.envelope_cases(ov, seed=seed + 77777))
            if full:
                idx = np.linspace(0, len(full) - 1, min(8, len(full))).round().astype(int).tolist()
                for j in idx:
                    item = full[j]
                    key = repr(item)
                    if key not in seen:
                        env.append(item)
                        seen.add(key)
        except Exception as e:
            print(f"[{PHASE}] WARNING: skipped sampled full-tail envelope: {e!r}")

    return env


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    # Maximize capture, minimize tangent. Include score as tie-breaker.
    rows = []
    for i, row in df.iterrows():
        cap = fnum(row.get("worst_capture_rate"), 0.0)
        tan = fnum(row.get("worst_tangent_ratio"), 999.0)
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            ocap = fnum(other.get("worst_capture_rate"), 0.0)
            otan = fnum(other.get("worst_tangent_ratio"), 999.0)
            if (ocap >= cap and otan <= tan) and (ocap > cap or otan < tan):
                dominated = True
                break
        if not dominated:
            rows.append(row)
    if not rows:
        return df.sort_values("ds_lite_score", ascending=False).head(10).copy()
    return pd.DataFrame(rows).sort_values(["worst_capture_rate", "worst_tangent_ratio"], ascending=[False, True])


def plot_bar(df: pd.DataFrame, out_path: Path, col: str, title: str, n: int = 25) -> None:
    if df.empty or col not in df.columns:
        return
    top = df.sort_values(col, ascending=False).head(n).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.44 * len(top))))
    plt.barh(top["case"], top[col])
    plt.xlabel(col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_scatter(df: pd.DataFrame, pf: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    c = df["ds_lite_score"] if "ds_lite_score" in df.columns else df.get("ds_rescue_score", 0.0)
    sc = plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=c, s=78, alpha=0.78)
    if not pf.empty:
        plt.scatter(pf["worst_capture_rate"], pf["worst_tangent_ratio"], facecolors="none", edgecolors="black", s=175, linewidths=1.7, label="Pareto front")
        plt.legend()
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45)
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DS-LITE score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DS-LITE strict-line rescue: capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)

    ds = import_by_path(DS_FILE, "phase26ds_full")
    drfast = import_by_path(DR_FAST_FILE, "phase26dr_fast")

    torch_info = drfast.setup_torch(args.device)
    dr = drfast.import_phase26dr()

    target_tan = float(getattr(dr, "TARGET_WORST_TAN", getattr(ds, "DEFAULT_TARGET_WORST_TAN", 2.10)))
    target_cap = float(getattr(dr, "TARGET_WORST_CAP", getattr(ds, "DEFAULT_TARGET_WORST_CAP", 0.285)))
    target_q20 = float(getattr(dr, "TARGET_Q20_DP", getattr(ds, "DEFAULT_TARGET_Q20_DP", 2.35)))

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: worstTan<={target_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] max_candidates={args.max_candidates}, audit_top={args.audit_top}, audit_repeats={args.audit_repeats}, sampled_full_tail={args.sample_full_tail}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    candidates = ds.build_candidates(drfast, "turbo", max_candidates=args.max_candidates, seed=args.seed)
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.max_candidates]
    name_to_ov = {name: ov for name, ov in candidates}
    print(f"[{PHASE}] Stage-1 tiny screen candidates: {len(candidates)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    screen_rows: List[Dict[str, Any]] = []
    screen_env_rows: List[Dict[str, Any]] = []

    for i, (case, ov) in enumerate(candidates, 1):
        env = micro_envelope(drfast, dr, ov, seed=args.seed + i * 17, repeats=1, use_full_tail=False)
        agg, rows = drfast.evaluate_envelope(dr, case, ov, basins, seed=args.seed + i * 17, envelope=env, cache=cache, phase_label="lite_screen")
        agg["ds_lite_score"] = ds.rescue_score(agg, target_cap, target_tan, target_q20)
        screen_rows.append(agg)
        screen_env_rows.extend(rows)
        if i <= 5 or i % args.print_every == 0:
            print(
                f"[{PHASE}] screen {i:03d}/{len(candidates)} {case[:62]:62s} "
                f"LITE={agg['ds_lite_score']:.3f} DR={agg['dr_score']:.3f} "
                f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
                f"worstCap={agg['worst_capture_rate']:.3f} worstTan={agg['worst_tangent_ratio']:.3f}"
            )

    screen_df = pd.DataFrame(screen_rows).sort_values("ds_lite_score", ascending=False) if screen_rows else pd.DataFrame()
    screen_env_df = pd.DataFrame(screen_env_rows)
    screen_df.to_csv(ROOT / "phase26ds_lite_screen_results.csv", index=False)
    screen_env_df.to_csv(ROOT / "phase26ds_lite_screen_envelope_results.csv", index=False)

    promote_names = screen_df.head(args.audit_top)["case"].tolist() if not screen_df.empty else []
    audit_candidates = [(name, name_to_ov[name]) for name in promote_names if name in name_to_ov]
    print(f"[{PHASE}] Stage-2 micro-audit candidates: {len(audit_candidates)}")

    audit_rows: List[Dict[str, Any]] = []
    audit_env_rows: List[Dict[str, Any]] = []
    for i, (case, ov) in enumerate(audit_candidates, 1):
        env = micro_envelope(drfast, dr, ov, seed=args.seed + 5000 + i * 31, repeats=args.audit_repeats, use_full_tail=args.sample_full_tail)
        agg, rows = drfast.evaluate_envelope(dr, case, ov, basins, seed=args.seed + 5000 + i * 31, envelope=env, cache=cache, phase_label="lite_audit")
        agg["ds_lite_score"] = ds.rescue_score(agg, target_cap, target_tan, target_q20)
        audit_rows.append(agg)
        audit_env_rows.extend(rows)
        print(
            f"[{PHASE}] audit  {i:03d}/{len(audit_candidates)} {case[:62]:62s} "
            f"LITE={agg['ds_lite_score']:.3f} DR={agg['dr_score']:.3f} "
            f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
            f"worstCap={agg['worst_capture_rate']:.3f} worstTan={agg['worst_tangent_ratio']:.3f} "
            f"strict={agg.get('strict_survival_rate', 0.0):.2f}"
        )

    audit_df = pd.DataFrame(audit_rows).sort_values("ds_lite_score", ascending=False) if audit_rows else pd.DataFrame()
    audit_env_df = pd.DataFrame(audit_env_rows)
    audit_df.to_csv(ROOT / "phase26ds_lite_case_results.csv", index=False)
    audit_env_df.to_csv(ROOT / "phase26ds_lite_envelope_results.csv", index=False)

    pf = pareto_front(audit_df) if not audit_df.empty else pd.DataFrame()
    pf.to_csv(ROOT / "phase26ds_lite_pareto_front.csv", index=False)

    plot_bar(screen_df, ROOT / "phase26ds_lite_screen_top_scores.png", "ds_lite_score", "26DS-LITE screen scores")
    plot_bar(audit_df, ROOT / "phase26ds_lite_top_scores.png", "ds_lite_score", "26DS-LITE micro-audit scores")
    plot_scatter(audit_df, pf, ROOT / "phase26ds_lite_capture_vs_tangent_pareto.png", target_cap, target_tan)

    best = audit_df.iloc[0] if len(audit_df) else (screen_df.iloc[0] if len(screen_df) else pd.Series(dtype=float))
    strict_df = audit_df[(audit_df["worst_capture_rate"] >= target_cap) & (audit_df["worst_tangent_ratio"] <= target_tan)].copy() if len(audit_df) else pd.DataFrame()
    best_strict = strict_df.sort_values("ds_lite_score", ascending=False).iloc[0] if len(strict_df) else pd.Series(dtype=float)

    params = getattr(ds, "PARAMS", [
        "BOWL_RADIUS_FRAC", "BOWL_SEAT_AXIS_GAIN", "BOWL_DIRECTIONAL_BLEND",
        "BOWL_NORM_CAP_MULT", "BOWL_SHELL_RADIAL_GAIN", "BOWL_TANGENT_KILL",
    ])

    def row_overrides(row: pd.Series) -> Dict[str, float] | None:
        if not len(row):
            return None
        return {k: fnum(row[k]) for k in params if k in row and pd.notna(row[k])}

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "torch_info": torch_info,
        "targets": {
            "target_worst_tangent_ratio": target_tan,
            "target_worst_capture_rate": target_cap,
            "target_q20_dp_score": target_q20,
        },
        "num_screen_candidates": int(len(candidates)),
        "num_screen_envelope_evals": int(len(screen_env_df)),
        "num_audit_candidates": int(len(audit_candidates)),
        "num_audit_envelope_evals": int(len(audit_env_df)),
        "num_cached_eval_entries": int(len(cache)),
        "num_pareto": int(len(pf)),
        "num_strict_line_pass": int(len(strict_df)),
        "elapsed_sec": float(time.time() - t0),
        "best_case": None if not len(best) else str(best.get("case")),
        "best_ds_lite_score": None if not len(best) else fnum(best.get("ds_lite_score")),
        "best_dr_score": None if not len(best) else fnum(best.get("dr_score")),
        "best_q20_dp_score": None if not len(best) else fnum(best.get("q20_dp_score")),
        "best_worst_dp_score": None if not len(best) else fnum(best.get("worst_dp_score")),
        "best_worst_capture_rate": None if not len(best) else fnum(best.get("worst_capture_rate")),
        "best_worst_tangent_ratio": None if not len(best) else fnum(best.get("worst_tangent_ratio")),
        "best_overrides": row_overrides(best),
        "best_strict_case": None if not len(best_strict) else str(best_strict.get("case")),
        "best_strict_ds_lite_score": None if not len(best_strict) else fnum(best_strict.get("ds_lite_score")),
        "best_strict_overrides": row_overrides(best_strict),
        "screen_top10": screen_df.head(10).to_dict(orient="records") if len(screen_df) else [],
        "audit_top10": audit_df.head(10).to_dict(orient="records") if len(audit_df) else [],
        "outputs": [
            "phase26ds_lite_screen_results.csv",
            "phase26ds_lite_screen_envelope_results.csv",
            "phase26ds_lite_case_results.csv",
            "phase26ds_lite_envelope_results.csv",
            "phase26ds_lite_pareto_front.csv",
            "phase26ds_lite_summary.json",
            "phase26ds_lite_screen_top_scores.png",
            "phase26ds_lite_top_scores.png",
            "phase26ds_lite_capture_vs_tangent_pareto.png",
        ],
    }
    with open(ROOT / "phase26ds_lite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best):
        print(
            f"[{PHASE}] BEST: {summary['best_case']} | "
            f"LITE={summary['best_ds_lite_score']:.4f} DR={summary['best_dr_score']:.4f} "
            f"q20DP={summary['best_q20_dp_score']:.4f} worstDP={summary['best_worst_dp_score']:.4f} "
            f"worstCap={summary['best_worst_capture_rate']:.3f} worstTan={summary['best_worst_tangent_ratio']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    if len(best_strict):
        print(f"[{PHASE}] BEST STRICT-LINE PASS: {summary['best_strict_case']} | LITE={summary['best_strict_ds_lite_score']:.4f}")
        print(f"[{PHASE}] BEST STRICT overrides: {summary['best_strict_overrides']}")
    else:
        print(f"[{PHASE}] No micro-audit candidate passed strict line worstCap>={target_cap:.3f} and worstTan<={target_tan:.2f}.")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26ds_lite_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-candidates", type=int, default=48, help="Tiny screen pool. Try 24 for smoke test, 96 for a slower pass.")
    p.add_argument("--audit-top", type=int, default=6, help="How many screen winners to micro-audit.")
    p.add_argument("--audit-repeats", type=int, default=3, help="Number of cheap screen-envelope repeats per audited candidate.")
    p.add_argument("--sample-full-tail", action="store_true", help="Add up to 8 sampled full-envelope cases per audited candidate. Slower, off by default.")
    p.add_argument("--print-every", type=int, default=6)
    p.add_argument("--seed", type=int, default=26020)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
