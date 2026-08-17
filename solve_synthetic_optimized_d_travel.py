#!/usr/bin/env python3
"""
One-click runner for a SINGLE synthetic UVT instance.

Assumptions (synthetic file header):
  # source = ...
  # target = ...
  # deadline = ...
  # max_timestamp = ...
  # Max timestamps (horizon T): ...
  (edges) u v t

Output:
  - Writes results into:  <synthetic_graphs_dir>/results/
  - Output TXT name:      <city>_TS<max_timestamps>_d<deadline>.txt
    Example: munich_TS100_d50.txt

"""

import os
import sys
import time
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict
import psutil

# -----------------------------
# EDIT HERE ONLY
# -----------------------------
SYNTHETIC_INPUT = os.path.join(os.path.dirname(__file__), "data", "synthetic", "synthetic_temporal_graph_EMA.txt")
LP_ONLY = False  # True = LP relaxation, False = ILP
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
# Synthetic input parsing (single instance)
# -----------------------------
_SYN_SOURCE_RE = re.compile(r"^\s*#\s*source\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_TARGET_RE = re.compile(r"^\s*#\s*target\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_RE = re.compile(r"^\s*#\s*deadline\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_MAXT_RE = re.compile(r"^\s*#\s*max_timestamp\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_HORIZON_RE = re.compile(r"^\s*#\s*Max\s+timestamps\s*\(horizon\s*T\)\s*:\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_NOTE_RE = re.compile(r"^\s*#\s*deadline_note\s*=\s*(.*)\s*$", re.IGNORECASE)


def parse_city_from_filename(path: str) -> str:
    """
    Extract city name from synthetic filename like:
      synthetic_temporal_graph_munich.txt
    Fallback: file stem.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"synthetic_temporal_graph_(.+)$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def read_synthetic_uvt(path: str):
    edges: List[Tuple[int, int, int]] = []
    nodes: Set[int] = set()
    max_t_seen = 0

    source: Optional[int] = None
    target: Optional[int] = None
    deadline: Optional[int] = None
    max_timestamp_header: Optional[int] = None
    horizon_T: Optional[int] = None
    deadline_note: Optional[str] = None

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                m = _SYN_SOURCE_RE.match(s)
                if m:
                    source = int(m.group(1))
                    continue
                m = _SYN_TARGET_RE.match(s)
                if m:
                    target = int(m.group(1))
                    continue
                m = _SYN_DEADLINE_RE.match(s)
                if m:
                    deadline = int(m.group(1))
                    continue
                m = _SYN_MAXT_RE.match(s)
                if m:
                    max_timestamp_header = int(m.group(1))
                    continue
                m = _SYN_HORIZON_RE.match(s)
                if m:
                    horizon_T = int(m.group(1))
                    continue
                m = _SYN_DEADLINE_NOTE_RE.match(s)
                if m:
                    deadline_note = m.group(1).strip()
                    continue
                continue

            parts = s.split()
            if len(parts) < 3:
                continue
            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, t))
            nodes.add(u)
            nodes.add(v)
            if t > max_t_seen:
                max_t_seen = t

    if not edges:
        raise ValueError(f"No edges parsed from: {path}")

    if source is None or target is None or deadline is None:
        raise ValueError(
            "Synthetic file header missing required fields. Need:\n"
            "  # source = ...\n  # target = ...\n  # deadline = ...\n"
            f"Parsed: source={source}, target={target}, deadline={deadline}"
        )

    # Variable horizon T: prefer explicit header max_timestamp, else horizon_T, else max_t_seen
    Tmax = max_timestamp_header if max_timestamp_header is not None else (horizon_T if horizon_T is not None else max_t_seen)

    meta = {
        "num_nodes": len(nodes),
        "temporal_edges": len(edges),
        "nodes_set": nodes,
        "max_timestamp": int(Tmax),          # variable horizon
        "max_timestamp_seen": int(max_t_seen),
        "deadline_note": deadline_note,
    }
    return edges, meta, int(source), int(target), int(deadline)


# -----------------------------
# TXT writer
# -----------------------------
def write_result_txt(
    out_path: str,
    *,
    input_file: str,
    city: str,
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
        f.write("TEMPORAL SEPARATOR RESULT (SYNTHETIC)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"City: {city}\n")
        f.write(f"Input file: {input_file}\n")

        f.write("\nINSTANCE\n")
        f.write(f"  source: {source}\n")
        f.write(f"  target: {target}\n")
        f.write(f"  deadline(travel_time): {deadline}\n")
        f.write(f"  max_timestamp(T): {max_timestamp}\n")
        f.write(f"  lp_only: {lp_only}\n")
        if meta.get("deadline_note"):
            f.write(f"  deadline_note: {meta.get('deadline_note')}\n")

        f.write("\nGRAPH\n")
        f.write(f"  total_nodes: {meta['num_nodes']}\n")
        f.write(f"  total_temporal_edges: {meta['temporal_edges']}\n")
        f.write(f"  max_timestamp_seen_in_edges: {meta['max_timestamp_seen']}\n")

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
# Run synthetic instance
# -----------------------------
def run_synthetic(temporal_edges, meta, *, s: int, z: int, d: int, lp_only: bool):
    nodes_set: Set[int] = meta["nodes_set"]
    Tmax = meta["max_timestamp"]

    if s not in nodes_set:
        raise ValueError(f"source {s} not present in graph.")
    if z not in nodes_set:
        raise ValueError(f"target {z} not present in graph.")
    if s == z:
        raise ValueError("source==target invalid.")
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


def main():
    input_file = SYNTHETIC_INPUT
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")

    city = parse_city_from_filename(input_file)

    print("=" * 80)
    print("SYNTHETIC UVT SOLVER (NO CLI)")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"City: {city}")
    print(f"Mode: {'LP' if LP_ONLY else 'ILP'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[PHASE] Load synthetic UVT instance...")
    temporal_edges, meta, s, z, d = read_synthetic_uvt(input_file)

    print("\n[INFO] Graph:")
    print(f"  nodes: {meta['num_nodes']:,}")
    print(f"  temporal edges: {meta['temporal_edges']:,}")
    print(f"  Tmax (variable horizon): {meta['max_timestamp']}")
    print(f"  max t seen in edges: {meta['max_timestamp_seen']}")
    print(f"  source={s}, target={z}, deadline(trt)={d}")
    if meta.get("deadline_note"):
        print(f"  deadline_note={meta.get('deadline_note')}")

    mem = psutil.virtual_memory()
    print(f"\n[INFO] Available memory: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

    # Output folder: <synthetic_graphs_parent>/results
    syn_dir = os.path.dirname(input_file)               # .../synthetic_graphs/ts100
    syn_parent = os.path.dirname(syn_dir)               # .../synthetic_graphs
    results_dir = os.path.join(syn_parent, "results")   # .../synthetic_graphs/results
    os.makedirs(results_dir, exist_ok=True)

    # Filename: city_TS<max_timestamps>_d<deadline>.txt
    # Here TS should be the synthetic horizon (meta['max_timestamp'])
    TS = int(meta["max_timestamp"])
    out_stem = f"{city}_TS{TS}_d{d}"
    out_txt = os.path.join(results_dir, f"{out_stem}.txt")
    summary_path = os.path.join(results_dir, f"{out_stem}_summary.txt")

    print("\n" + "-" * 80)
    print(f"[RUN] source={s} target={z} deadline(trt)={d} | T={TS}")
    print("-" * 80)
    print("[STEP] Build & solve model...")

    sep, obj, stats, solve_wall = run_synthetic(temporal_edges, meta, s=s, z=z, d=d, lp_only=LP_ONLY)

    status = stats.get("status")
    print(f"[DONE] status={status} | obj={obj} | time={solve_wall:.3f}s")
    print(f"[OUT ] {out_txt}")

    write_result_txt(
        out_txt,
        input_file=input_file,
        city=city,
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

    # Minimal summary file (so you can batch grep later)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("SUMMARY (SYNTHETIC)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {input_file}\n")
        f.write(f"City: {city}\n")
        f.write(f"source={s} target={z} deadline(trt)={d} T={TS}\n")
        if meta.get("deadline_note"):
            f.write(f"deadline_note={meta.get('deadline_note')}\n")
        f.write(f"Mode: {'LP' if LP_ONLY else 'ILP'}\n")
        f.write(f"status={status} obj={obj} time={solve_wall:.6f}s\n")

    print(f"[OUT ] {summary_path}")


if __name__ == "__main__":
    main()