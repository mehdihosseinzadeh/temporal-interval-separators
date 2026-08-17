#!/usr/bin/env python3
"""
One-click batch runner for a city dataset 

You only change:
  CITY = "kuopio"

Then the script:
  1) loads the UVT file that contains "# combo ..." lines + edges u v t
  2) runs each combo separately (or only selected ones)
  3) prints which combo is currently being solved
  4) reports per-combo runtime (wall clock)
  5) writes per-combo TXT + summary TXT

"""

import os
import sys
import time
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from collections import defaultdict
import psutil

# -----------------------------
# EDIT HERE ONLY
# -----------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "real")
CITY = "venice"
INPUT_NAME = "network_temporal_day_uvt_first2h.txt"   # change if needed
LP_ONLY = False  # True = LP relaxation, False = ILP

# Choose which combos to solve:
#   "all"  -> run all combos found in header
#   3      -> run only combo 3
#   [1,4]  -> run combos 1 and 4 only
#   "1,4,7"-> run combos 1,4,7
COMBOS_TO_RUN: Union[str, int, List[int]] = "all"
# -----------------------------

# Add the pure_ilp_solver directory to the path (expects: ./pure_ilp_solver/pure_ilp_temporal_separator.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "pure_ilp_solver"))
from pure_ilp_temporal_separator import PureILPTemporalSeparator  # noqa

import gurobipy as gp  # noqa
from gurobipy import GRB  # noqa


# -----------------------------
# Solver wrapper (LP + ILP)
# -----------------------------
class OptimizedTemporalSeparator(PureILPTemporalSeparator):
    def solve_separator_lp(self):
        """
        LP relaxation: x_{v,t} in [0,1].

        IMPORTANT:
        deadline is travel-time bound, so variables are over full timeline t=1..T.
        The travel-time restriction is handled in find_temporal_paths().
        """
        t0 = time.time()

        paths = self.find_temporal_paths()
        if not paths:
            return None, float("inf"), {
                "status": "no_paths",
                "solve_time": time.time() - t0,
                "num_paths": 0,
            }

        model = gp.Model("temporal_separator_lp")
        model.setParam("OutputFlag", 0)

        x = {}
        separator_vertices = [v for v in self.vertices if v != self.source and v != self.target]
        T = int(self.max_timestamp)

        for v in separator_vertices:
            for t in range(1, T + 1):
                x[v, t] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{v}_{t}"
                )

        model.setObjective(
            gp.quicksum(x[v, t] for v in separator_vertices for t in range(1, T + 1)),
            GRB.MINIMIZE,
        )

        for i, path in enumerate(paths):
            path_vars = []
            for v, t in path:
                if v in separator_vertices and t >= 1:
                    path_vars.append(x[v, t])
            if path_vars:
                model.addConstr(gp.quicksum(path_vars) >= 1, name=f"path_{i}")

        # Contiguity constraints -- MUST mirror the exact ILP's solve_separator()
        # exactly (same pairs, same inequality), only with continuous x. Omitting
        # these would relax more than just integrality (it would also drop the
        # "single contiguous blocking interval per vertex" requirement), giving an
        # invalid / overly loose "LP bound" and a fragmented, non-interval solution.
        for v in separator_vertices:
            for t1 in range(1, T):
                for t2 in range(t1 + 2, T + 1):
                    model.addConstr(
                        x[v, t1] + x[v, t2] - 1 <= x[v, t1 + 1],
                        name=f"contiguity_{v}_{t1}_{t2}",
                    )

        model.optimize()

        stats = {
            "status": model.status,
            "solve_time": time.time() - t0,
            "num_variables": model.numVars,
            "num_constraints": model.numConstrs,
            "num_paths": len(paths),
            "objective_value": float("inf"),
        }

        if model.status == GRB.OPTIMAL:
            sep = {}
            for v in separator_vertices:
                for t in range(1, T + 1):
                    val = x[v, t].X
                    if val > 1e-6:
                        sep[(v, t)] = val
            stats["objective_value"] = model.objVal
            return sep, model.objVal, stats

        return None, float("inf"), stats


# -----------------------------
# Separator summarizer
# -----------------------------
def summarize_separator(sep: Dict[Tuple[int, int], Any]) -> Dict[str, Any]:
    vertex_times: Dict[int, List[int]] = defaultdict(list)

    for (v, t), val in sep.items():
        keep = True
        if isinstance(val, (int, float)):
            keep = val > 0.5 if val in (0, 1) else val > 1e-6
        if keep:
            vertex_times[int(v)].append(int(t))

    vertex_intervals: Dict[int, List[Tuple[int, int]]] = {}
    for v, times in vertex_times.items():
        times = sorted(set(times))
        if not times:
            continue
        intervals: List[Tuple[int, int]] = []
        s = e = times[0]
        for tt in times[1:]:
            if tt == e + 1:
                e = tt
            else:
                intervals.append((s, e))
                s = e = tt
        intervals.append((s, e))
        vertex_intervals[v] = intervals

    return {
        "vertex_times": {k: sorted(set(v)) for k, v in vertex_times.items()},
        "vertex_intervals": vertex_intervals,
        "total_vertices_in_sep": len(vertex_times),
        "total_pairs_kept": sum(len(ts) for ts in vertex_times.values()),
    }


# -----------------------------
# Input parsing (edges + combos)
# -----------------------------
_COMBO_RE = re.compile(
    r"combo\s*(\d+)\s*:\s*source\s*=\s*(\d+)\s+target\s*=\s*(\d+)\s+deadline\s*=\s*(\d+)",
    re.IGNORECASE,
)

def read_uvt_temporal_graph_with_combos(path: str):
    edges: List[Tuple[int, int, int]] = []
    nodes: Set[int] = set()
    max_t = 0
    combos: List[Dict[str, Any]] = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                m = _COMBO_RE.search(s)
                if m:
                    combos.append({
                        "combo_id": int(m.group(1)),
                        "source": int(m.group(2)),
                        "target": int(m.group(3)),
                        "deadline": int(m.group(4)),
                        "raw_line": s,
                    })
                continue

            parts = s.split()
            if len(parts) < 3:
                continue
            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, t))
            nodes.add(u)
            nodes.add(v)
            if t > max_t:
                max_t = t

    if not edges:
        raise ValueError(f"No edges parsed from: {path}")

    combos = sorted(combos, key=lambda d: d["combo_id"])

    meta = {
        "num_nodes": len(nodes),
        "nodes_min": min(nodes),
        "nodes_max": max(nodes),
        "max_timestamp": max_t,
        "temporal_edges": len(edges),
        "nodes_set": nodes,
        "num_combos": len(combos),
    }
    return edges, meta, combos


# -----------------------------
# TXT writer
# -----------------------------
def write_result_txt(
    out_path: str,
    *,
    input_file: str,
    city: str,
    combo_id: Optional[int],
    source: int,
    target: int,
    deadline: int,
    max_timestamp: int,
    lp_only: bool,
    meta: Dict[str, Any],
    sep: Optional[Dict[Tuple[int, int], Any]],
    obj: float,
    stats: Dict[str, Any],
    solve_wall_time: float,
):
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("TEMPORAL SEPARATOR RESULT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"City: {city}\n")
        f.write(f"Input file: {input_file}\n")
        if combo_id is not None:
            f.write(f"Combo: {combo_id}\n")

        f.write("\nINSTANCE\n")
        f.write(f"  source: {source}\n")
        f.write(f"  target: {target}\n")
        f.write(f"  deadline(travel_time): {deadline}\n")
        f.write(f"  max_timestamp(T): {max_timestamp}\n")
        f.write(f"  lp_only: {lp_only}\n")

        f.write("\nGRAPH\n")
        f.write(f"  total_nodes: {meta['num_nodes']}\n")
        f.write(f"  total_temporal_edges: {meta['temporal_edges']}\n")

        f.write("\nSOLVER\n")
        f.write(f"  status: {stats.get('status')}\n")
        if "num_variables" in stats:
            f.write(f"  num_variables: {stats.get('num_variables')}\n")
        if "num_constraints" in stats:
            f.write(f"  num_constraints: {stats.get('num_constraints')}\n")
        if "num_paths" in stats:
            f.write(f"  num_paths: {stats.get('num_paths')}\n")
        if "solve_time" in stats:
            f.write(f"  solver_reported_solve_time: {stats.get('solve_time')}\n")

        f.write("\nOBJECTIVE\n")
        f.write(f"  objective_value: {obj}\n")

        f.write("\nTIMING\n")
        f.write(f"  solve_wall_time_seconds: {solve_wall_time:.6f}\n")

        f.write("\nSEPARATOR\n")
        if sep is None:
            f.write("  NO SOLUTION / FAILED\n")
            return

        info = summarize_separator(sep)
        f.write(f"  separator_size_pairs: {len(sep)}\n")
        f.write(f"  vertices_in_separator: {info['total_vertices_in_sep']}\n")
        f.write(f"  total_kept_pairs: {info['total_pairs_kept']}\n")

        f.write("\n  intervals_per_vertex:\n")
        for v in sorted(info["vertex_intervals"].keys()):
            intervals = info["vertex_intervals"][v]
            intervals_str = ", ".join([f"[{a},{b}]" if a != b else f"[{a}]" for a, b in intervals])
            f.write(f"    {v}: {intervals_str}\n")


# -----------------------------
# Run one combo
# -----------------------------
def run_one_combo(temporal_edges, meta, *, s, z, d, lp_only: bool):
    nodes_set: Set[int] = meta["nodes_set"]
    Tmax = meta["max_timestamp"]

    if s not in nodes_set:
        raise ValueError(f"source {s} not present in graph.")
    if z not in nodes_set:
        raise ValueError(f"target {z} not present in graph.")
    if s == z:
        raise ValueError("source==target invalid.")
    # IMPORTANT: travel-time deadline only needs to be >= 1 (NOT <= Tmax)
    if d < 1:
        raise ValueError(f"deadline must be >= 1 (travel-time), got {d}")

    solver = OptimizedTemporalSeparator(temporal_edges, s, z, d, Tmax)

    t0 = time.time()
    if lp_only:
        sep, obj, stats = solver.solve_separator_lp()
    else:
        sep, obj, stats = solver.solve_separator(10**9)  # effectively "no limit"
    t1 = time.time()

    return sep, obj, stats, (t1 - t0)


# -----------------------------
# Combo filtering
# -----------------------------
def normalize_combos_to_run(spec: Union[str, int, List[int]]) -> Optional[Set[int]]:
    """
    Returns:
      - None if spec == "all" (meaning: run all)
      - otherwise a set of combo_ids to run
    """
    if isinstance(spec, str):
        if spec.strip().lower() == "all":
            return None
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if parts:
            return set(int(p) for p in parts)
        raise ValueError(f"Invalid COMBOS_TO_RUN string: {spec!r}")

    if isinstance(spec, int):
        return {spec}

    if isinstance(spec, list):
        return set(int(x) for x in spec)

    raise ValueError(f"Invalid COMBOS_TO_RUN type: {type(spec)}")


def main():
    input_file = os.path.join(BASE_DIR, CITY, INPUT_NAME)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")

    print("=" * 80)
    print("CITY BATCH SOLVER (NO CLI)")
    print("=" * 80)
    print(f"City: {CITY}")
    print(f"Input: {input_file}")
    print(f"Mode: {'LP' if LP_ONLY else 'ILP'}")
    print(f"Combos: {COMBOS_TO_RUN}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[PHASE] Load graph + combos...")
    temporal_edges, meta, combos = read_uvt_temporal_graph_with_combos(input_file)

    print("\n[INFO] Graph:")
    print(f"  nodes: {meta['num_nodes']:,}")
    print(f"  temporal edges: {meta['temporal_edges']:,}")
    print(f"  Tmax (timestamp horizon): {meta['max_timestamp']}")
    print(f"  combos in header: {len(combos)}")

    mem = psutil.virtual_memory()
    print(f"\n[INFO] Available memory: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

    # Results directory
    stem = os.path.splitext(os.path.basename(input_file))[0]
    results_dir = os.path.join(BASE_DIR, CITY, f"results_{stem}")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.txt")
    summary_lines = []
    summary_lines.append("SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"City: {CITY}")
    summary_lines.append(f"Input: {input_file}")
    summary_lines.append(f"Mode: {'LP' if LP_ONLY else 'ILP'}")
    summary_lines.append(f"Combos request: {COMBOS_TO_RUN}")
    summary_lines.append(f"Graph: nodes={meta['num_nodes']} edges={meta['temporal_edges']} Tmax={meta['max_timestamp']}")
    summary_lines.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")

    if not combos:
        # fallback single inferred
        s = meta["nodes_min"]
        z = meta["nodes_max"]
        d = 1  # travel-time deadline minimal fallback; better to define explicitly in header
        combos = [{"combo_id": 1, "source": s, "target": z, "deadline": d, "raw_line": "# inferred"}]
        print("\n[WARN] No combos found in header -> using inferred single combo with d=1.")

    wanted = normalize_combos_to_run(COMBOS_TO_RUN)
    if wanted is not None:
        combos = [c for c in combos if c["combo_id"] in wanted]
        missing = sorted(wanted - {c["combo_id"] for c in combos})
        if missing:
            print(f"\n[WARN] These requested combos were not found in header: {missing}")

    if not combos:
        raise RuntimeError("After filtering, there are NO combos to run. Check COMBOS_TO_RUN.")

    print(f"\n[PHASE] Solve {len(combos)} combos (each separately)...")

    for idx, c in enumerate(combos, start=1):
        cid = c["combo_id"]
        s = c["source"]
        z = c["target"]
        d = c["deadline"]

        print("\n" + "-" * 80)
        print(f"[RUN {idx}/{len(combos)}] COMBO {cid:02d}  source={s} target={z} deadline(trt)={d}")
        print("-" * 80)
        print("[STEP] Build & solve model...")

        sep, obj, stats, solve_wall = run_one_combo(
            temporal_edges, meta, s=s, z=z, d=d, lp_only=LP_ONLY
        )

        status = stats.get("status")
        print(f"[DONE] combo {cid:02d} finished | status={status} | obj={obj} | time={solve_wall:.3f}s")

        out_txt = os.path.join(results_dir, f"combo_{cid:02d}_s{s}_z{z}_d{d}.txt")
        write_result_txt(
            out_txt,
            input_file=input_file,
            city=CITY,
            combo_id=cid,
            source=s,
            target=z,
            deadline=d,
            max_timestamp=meta["max_timestamp"],
            lp_only=LP_ONLY,
            meta=meta,
            sep=sep,
            obj=obj,
            stats=stats,
            solve_wall_time=solve_wall,
        )

        summary_lines.append(
            f"combo={cid:02d}  s={s}  z={z}  d(trt)={d}  status={status}  obj={obj}  time={solve_wall:.3f}s"
        )

    summary_lines.append("")
    summary_lines.append(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n[PHASE] Saved summary:")
    print(f"  {summary_path}")
    print(f"[DONE] Results dir: {results_dir}")


if __name__ == "__main__":
    main()