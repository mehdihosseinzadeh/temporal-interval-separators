#!/usr/bin/env python3
"""
One-click batch runner: ILP vs. Greedy-heuristic comparison on ONE real-world
GTFS transportation dataset (no CLI needed) -- mirrors
run_lp_relaxation_comparison_real.py structurally.

"""

import os
import sys
import csv
import glob
import time
import re
import bisect
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from collections import defaultdict, deque
import psutil

# -----------------------------
# EDIT HERE ONLY
# -----------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "real")
CITY = "berlin"
INPUT_NAME = "network_temporal_day_uvt_first2h.txt"

# Choose which combos to evaluate:
#   "all"  -> evaluate all combos found in the header
#   3      -> only combo 3
#   [1,4]  -> only combos 1 and 4
#   "1,4,7" -> only combos 1,4,7
COMBOS_TO_RUN: Union[str, int, List[int]] = "all"

REUSE_EXISTING_ILP_RESULTS = True
EXISTING_ILP_RESULT_DIR_NAMES = [
    "results_first_submitt",
    "results_{stem}",
]
ILP_TIME_LIMIT = 10 ** 9  # only used if a fresh ILP solve is needed as a fallback
# -----------------------------

# Add the pure_ilp_solver directory to the path (expects: ./pure_ilp_solver/pure_ilp_temporal_separator.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "pure_ilp_solver"))
from pure_ilp_temporal_separator import PureILPTemporalSeparator  # noqa

import gurobipy as gp  # noqa  (only needed for the ILP fallback path)
from gurobipy import GRB  # noqa


# -----------------------------
# Reuse existing ILP results (same logic as run_lp_relaxation_comparison_real.py)
# -----------------------------
_RESULT_OBJ_RE = re.compile(r"^\s*objective_value:\s*([\d.]+|inf)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_TIME_RE = re.compile(r"^\s*solve_wall_time_seconds:\s*([\d.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_PATHS_RE = re.compile(r"^\s*num_paths:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_VARS_RE = re.compile(r"^\s*num_variables:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_CONSTR_RE = re.compile(r"^\s*num_constraints:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_STATUS_RE = re.compile(r"^\s*status:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def find_existing_ilp_result(base_dir: str, city: str, stem: str, s: int, z: int, d: int) -> Optional[str]:
    pattern_suffix = f"_s{s}_z{z}_d{d}.txt"
    for dir_template in EXISTING_ILP_RESULT_DIR_NAMES:
        dir_name = dir_template.format(stem=stem)
        candidate_dir = os.path.join(base_dir, city, dir_name)
        if not os.path.isdir(candidate_dir):
            continue
        matches = sorted(glob.glob(os.path.join(candidate_dir, f"combo_*{pattern_suffix}")))
        if matches:
            return matches[0]
    return None


def parse_existing_ilp_result(path: str) -> Tuple[float, float, Dict[str, Any]]:
    text = open(path, "r", encoding="utf-8").read()

    m_obj = _RESULT_OBJ_RE.search(text)
    m_time = _RESULT_TIME_RE.search(text)
    m_paths = _RESULT_PATHS_RE.search(text)
    m_vars = _RESULT_VARS_RE.search(text)
    m_constr = _RESULT_CONSTR_RE.search(text)
    m_status = _RESULT_STATUS_RE.search(text)

    if not m_obj or not m_time:
        raise ValueError(f"Could not parse objective/time from existing result file: {path}")

    obj = float("inf") if m_obj.group(1).lower() == "inf" else float(m_obj.group(1))
    ilp_time = float(m_time.group(1))

    stats = {
        "status": m_status.group(1) if m_status else "reused",
        "num_paths": int(m_paths.group(1)) if m_paths else None,
        "num_variables": int(m_vars.group(1)) if m_vars else None,
        "num_constraints": int(m_constr.group(1)) if m_constr else None,
    }
    return obj, ilp_time, stats


# -----------------------------
# Solver wrapper: exact ILP fallback (only used if no existing result found) +
# Greedy-TimeSep heuristic (pure Python, no Gurobi).
# -----------------------------
class GreedyTemporalSeparator(PureILPTemporalSeparator):
    def _static_degree(self) -> Dict[int, int]:
        """
        Degree of each vertex in the underlying static directed graph D,
        counting each distinct arc (u,v) once regardless of how many
        timestamps it appears at in the temporal graph.
        """
        seen_arcs: Set[Tuple[int, int]] = set()
        degree: Dict[int, int] = defaultdict(int)
        for (u, v, _t) in self.temporal_edges:
            if (u, v) in seen_arcs:
                continue
            seen_arcs.add((u, v))
            degree[u] += 1
            degree[v] += 1
        return degree

    def find_one_unseparated_path(
        self, intervals: Dict[int, List[int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find ONE unseparated temporal path without full enumeration
        (outgoing-arc semantics, per Definition 1). See
        run_greedy_comparison_synthetic.py for the full docstring; identical
        implementation."""
        def is_blocked(v: int, t: int) -> bool:
            iv = intervals.get(v)
            return iv is not None and iv[0] <= t <= iv[1]

        times_sorted = sorted(self.temporal_graph.keys())

        first_times = sorted({
            t for t in times_sorted
            for (u, _v) in self.temporal_graph[t] if u == self.source
        })

        for t_first in first_times:
            start = (self.source, 0)
            visited: Set[Tuple[int, int]] = {start}
            parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
            queue = deque([start])

            while queue:
                v, cur_t = queue.popleft()

                if v == self.source and cur_t == 0:
                    candidate_times = [t_first]
                else:
                    idx = bisect.bisect_right(times_sorted, cur_t)
                    candidate_times = times_sorted[idx:]

                for next_time in candidate_times:
                    if next_time - t_first + 1 > self.deadline:
                        break  # further times only increase trt
                    # v's own departing arc at next_time is blocked if
                    # next_time falls inside v's current interval.
                    if is_blocked(v, next_time):
                        continue
                    for u, vv in self.temporal_graph[next_time]:
                        if u != v:
                            continue
                        if vv == self.source:
                            # A temporal path visits the source exactly once,
                            # at the very start; never re-enter it (see
                            # run_greedy_comparison_synthetic.py for why).
                            continue
                        state = (vv, next_time)
                        if state in visited:
                            continue
                        visited.add(state)
                        parent[state] = (v, cur_t)
                        if vv == self.target:
                            # Reconstruct the arrival-time chain first, then
                            # shift to the outgoing-arc pairing: each vertex
                            # is paired with the NEXT chain entry's
                            # timestamp, i.e. its own departing arc's time.
                            chain = [state]
                            back = (v, cur_t)
                            while back != start:
                                chain.append(back)
                                back = parent[back]
                            chain.append(start)
                            chain.reverse()
                            path = [
                                (chain[i][0], chain[i + 1][1])
                                for i in range(len(chain) - 1)
                            ]
                            return path
                        queue.append(state)

        return None

    def solve_greedy(self):
        """Greedy-TimeSep heuristic (updated design: no full enumeration of
        P_{s,z,d}). See run_greedy_comparison_synthetic.py for the full
        docstring; identical implementation."""
        t0 = time.time()

        degree = self._static_degree()
        intervals: Dict[int, List[int]] = {}  # v -> [l, r]
        num_iterations = 0
        num_path_searches = 0
        status = "greedy_complete"

        while True:
            path = self.find_one_unseparated_path(intervals)
            num_path_searches += 1
            if path is None:
                break

            candidates = [v for (v, t) in path if v != self.source and v != self.target]
            if not candidates:
                # A direct source->target temporal arc within the deadline
                # can never be blocked by any vertex interval -- same
                # limitation the exact ILP has. Nothing to select; stop.
                status = "infeasible_direct_arc"
                break

            v_star = max(candidates, key=lambda v: degree.get(v, 0))
            t_star = next(t for (v, t) in path if v == v_star)

            if v_star in intervals:
                l, r = intervals[v_star]
                intervals[v_star] = [min(l, t_star), max(r, t_star)]
            else:
                intervals[v_star] = [t_star, t_star]
            num_iterations += 1

        solve_time = time.time() - t0

        if status == "infeasible_direct_arc":
            # See run_greedy_comparison_synthetic.py: report this the same
            # way solve_separator() reports INFEASIBLE (sep=None, obj=inf).
            stats = {
                "status": status,
                "solve_time": solve_time,
                "num_path_searches": num_path_searches,
                "num_iterations": num_iterations,
            }
            return None, float("inf"), stats

        sep: Dict[Tuple[int, int], int] = {}
        for v, (l, r) in intervals.items():
            for t in range(l, r + 1):
                sep[(v, t)] = 1

        obj = sum((r - l + 1) for (l, r) in intervals.values())

        stats = {
            "status": status,
            "solve_time": solve_time,
            "num_path_searches": num_path_searches,
            "num_iterations": num_iterations,
            "objective_value": obj,
        }
        return sep, obj, stats


# -----------------------------
# Input parsing (edges + combos), same as the other real-network runners
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


def normalize_combos_to_run(spec: Union[str, int, List[int]]) -> Optional[Set[int]]:
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


# -----------------------------
# Separator summarizer + detail writer
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


def write_detail_txt(
    out_path: str,
    *,
    input_file: str,
    city: str,
    combo_id: int,
    source: int,
    target: int,
    deadline: int,
    max_timestamp: int,
    mode: str,
    meta: Dict[str, Any],
    sep: Optional[Dict[Tuple[int, int], Any]],
    obj: float,
    stats: Dict[str, Any],
    solve_wall_time: float,
):
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"TEMPORAL SEPARATOR RESULT ({mode})\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"City: {city}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Combo: {combo_id}\n")

        f.write("\nINSTANCE\n")
        f.write(f"  source: {source}\n")
        f.write(f"  target: {target}\n")
        f.write(f"  deadline(travel_time): {deadline}\n")
        f.write(f"  max_timestamp(T): {max_timestamp}\n")
        f.write(f"  mode: {mode}\n")

        f.write("\nGRAPH\n")
        f.write(f"  total_nodes: {meta['num_nodes']}\n")
        f.write(f"  total_temporal_edges: {meta['temporal_edges']}\n")

        f.write("\nSOLVER\n")
        f.write(f"  status: {stats.get('status')}\n")
        if "num_paths" in stats:
            f.write(f"  num_paths: {stats.get('num_paths')}\n")
        if "num_path_searches" in stats:
            f.write(f"  num_path_searches: {stats.get('num_path_searches')}\n")
        if "num_iterations" in stats:
            f.write(f"  num_iterations: {stats.get('num_iterations')}\n")

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
# Run one combo: ILP (reuse) + Greedy
# -----------------------------
def run_combo(temporal_edges, meta, *, city, input_file, base_dir, stem, results_dir,
              combo: Dict[str, Any]) -> Dict[str, Any]:
    cid, s, z, d = combo["combo_id"], combo["source"], combo["target"], combo["deadline"]
    nodes_set: Set[int] = meta["nodes_set"]
    Tmax = meta["max_timestamp"]

    if s not in nodes_set or z not in nodes_set:
        raise ValueError(f"combo {cid}: source/target not present in graph (s={s}, z={z}).")
    if s == z:
        raise ValueError(f"combo {cid}: source==target invalid.")
    if d < 1:
        raise ValueError(f"combo {cid}: deadline must be >= 1 (travel-time), got {d}")

    print(f"\n[COMBO {cid:02d}] source={s} target={z} deadline(trt)={d}")

    # ---- ILP: reuse existing result if available, otherwise solve ----
    ilp_source = "resolved_now"
    existing_path = (
        find_existing_ilp_result(base_dir, city, stem, s, z, d)
        if REUSE_EXISTING_ILP_RESULTS else None
    )

    if existing_path is not None:
        print(f"  [REUSE] Found existing ILP result: {existing_path}")
        obj_ilp, ilp_time, stats_ilp = parse_existing_ilp_result(existing_path)
        ilp_source = f"reused:{existing_path}"
        print(f"  [DONE] ILP (reused) status={stats_ilp.get('status')} SL={obj_ilp} time={ilp_time:.3f}s")
    else:
        if REUSE_EXISTING_ILP_RESULTS:
            print(f"  [WARN] No existing ILP result found for combo *_s{s}_z{z}_d{d}.txt under "
                  f"{[os.path.join(base_dir, city, n.format(stem=stem)) for n in EXISTING_ILP_RESULT_DIR_NAMES]} "
                  "-- solving ILP now.")
        print("  [STEP] Solving exact ILP ...")
        solver_ilp = GreedyTemporalSeparator(temporal_edges, s, z, d, Tmax)
        t0 = time.time()
        sep_ilp, obj_ilp, stats_ilp = solver_ilp.solve_separator(ILP_TIME_LIMIT)
        ilp_time = time.time() - t0
        print(f"  [DONE] ILP status={stats_ilp.get('status')} SL={obj_ilp} time={ilp_time:.3f}s")

    # ---- Greedy heuristic (fresh solver instance) ----
    print("  [STEP] Running greedy heuristic ...")
    solver_greedy = GreedyTemporalSeparator(temporal_edges, s, z, d, Tmax)
    assert solver_greedy.source == s and solver_greedy.target == z and solver_greedy.deadline == d, (
        f"Greedy solver instance mismatch for combo {cid:02d}: "
        f"built with source={solver_greedy.source} target={solver_greedy.target} deadline={solver_greedy.deadline}, "
        f"expected source={s} target={z} deadline={d}"
    )
    print(f"  [CHECK] Greedy solver confirmed for THIS combo: source={solver_greedy.source} "
          f"target={solver_greedy.target} deadline={solver_greedy.deadline} (combo {cid:02d} expects s={s} z={z} d={d})")
    sep_greedy, obj_greedy, stats_greedy = solver_greedy.solve_greedy()
    greedy_time = stats_greedy.get("solve_time", 0.0)
    print(f"  [DONE] Greedy status={stats_greedy.get('status')} SL={obj_greedy} "
          f"iterations={stats_greedy.get('num_iterations')} time={greedy_time:.3f}s")

    write_detail_txt(
        os.path.join(results_dir, f"combo_{cid:02d}_s{s}_z{z}_d{d}_greedy.txt"),
        input_file=input_file, city=city, combo_id=cid, source=s, target=z, deadline=d,
        max_timestamp=Tmax, mode="GREEDY", meta=meta, sep=sep_greedy, obj=obj_greedy,
        stats=stats_greedy, solve_wall_time=greedy_time,
    )

    ratio = None
    if obj_ilp not in (None, float("inf")) and obj_ilp > 0 and obj_greedy not in (None, float("inf")):
        ratio = obj_greedy / obj_ilp
    speedup = None
    if greedy_time > 0:
        speedup = ilp_time / greedy_time

    return {
        "combo_id": cid,
        "source": s,
        "target": z,
        "deadline": d,
        "ilp_status": stats_ilp.get("status"),
        "ilp_SL": obj_ilp,
        "ilp_time_s": round(ilp_time, 3),
        "ilp_source": ilp_source,
        "greedy_status": stats_greedy.get("status"),
        "greedy_SL": obj_greedy,
        "greedy_time_s": round(greedy_time, 3),
        "greedy_num_iterations": stats_greedy.get("num_iterations"),
        "ratio_greedy_over_ilp": round(ratio, 3) if ratio is not None else None,
        "speedup_ilp_over_greedy": round(speedup, 2) if speedup is not None else None,
    }


CSV_FIELDS = [
    "combo_id", "source", "target", "deadline",
    "ilp_SL", "ilp_time_s", "greedy_SL", "greedy_time_s",
    "ratio_greedy_over_ilp", "speedup_ilp_over_greedy",
    "greedy_num_iterations", "ilp_status", "greedy_status", "ilp_source", "selected",
]


def write_csv(rows: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})


def main():
    input_file = os.path.join(BASE_DIR, CITY, INPUT_NAME)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")

    print("=" * 80)
    print("ILP vs GREEDY BATCH COMPARISON (REAL CITY, NO CLI)")
    print("=" * 80)
    print(f"City: {CITY}")
    print(f"Input: {input_file}")
    print(f"Combos: {COMBOS_TO_RUN}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    mem = psutil.virtual_memory()
    print(f"[INFO] Available memory: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

    temporal_edges, meta, combos = read_uvt_temporal_graph_with_combos(input_file)
    print(f"\n[INFO] Graph: nodes={meta['num_nodes']:,} edges={meta['temporal_edges']:,} "
          f"Tmax={meta['max_timestamp']} combos_in_header={len(combos)}")

    if not combos:
        s, z = meta["nodes_min"], meta["nodes_max"]
        d = 1
        combos = [{"combo_id": 1, "source": s, "target": z, "deadline": d, "raw_line": "# inferred"}]
        print("[WARN] No combos found in header -> using inferred single combo with d=1.")

    wanted = normalize_combos_to_run(COMBOS_TO_RUN)
    if wanted is not None:
        combos = [c for c in combos if c["combo_id"] in wanted]
    if not combos:
        raise RuntimeError("After filtering, there are NO combos to run. Check COMBOS_TO_RUN.")

    stem = os.path.splitext(os.path.basename(input_file))[0]
    results_dir = os.path.join(BASE_DIR, CITY, f"results_{stem}_greedy_comparison")
    os.makedirs(results_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, combo in enumerate(combos, start=1):
        print("\n" + "-" * 80)
        print(f"[RUN {idx}/{len(combos)}]")
        try:
            row = run_combo(temporal_edges, meta, city=CITY, input_file=input_file,
                             base_dir=BASE_DIR, stem=stem,
                             results_dir=results_dir, combo=combo)
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] combo {combo.get('combo_id')} failed: {e}")

    finite_rows = [r for r in rows if isinstance(r.get("ilp_SL"), (int, float)) and r["ilp_SL"] != float("inf")]
    selected_id = None
    if finite_rows:
        selected_id = max(finite_rows, key=lambda r: r["ilp_SL"])["combo_id"]
    for r in rows:
        r["selected"] = (r["combo_id"] == selected_id)

    csv_path = os.path.join(results_dir, "comparison_summary.csv")
    write_csv(rows, csv_path)

    print("\n" + "=" * 80)
    print("[DONE] Batch comparison finished.")
    print(f"  CSV: {csv_path}")
    if selected_id is not None:
        sel = next(r for r in rows if r["combo_id"] == selected_id)
        print(f"  Selected combo (max ILP SL, per paper methodology): combo {selected_id:02d} "
              f"-> ILP_SL={sel['ilp_SL']} Greedy_SL={sel['greedy_SL']} ratio={sel['ratio_greedy_over_ilp']} "
              f"ILP_time={sel['ilp_time_s']}s Greedy_time={sel['greedy_time_s']}s")
    print(f"  Detail files in: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
