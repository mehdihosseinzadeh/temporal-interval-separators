#!/usr/bin/env python3
"""
One-click batch runner: ILP vs. Greedy-heuristic comparison on the synthetic
transportation instances (no CLI needed). Mirrors
run_lp_relaxation_comparison_synthetic.py structurally.

"""

import os
import sys
import csv
import time
import re
import glob
import bisect
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict, deque
import psutil

# -----------------------------
# EDIT HERE ONLY
# -----------------------------
SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic")

# None -> auto-discover every "synthetic_temporal_graph_*.txt" directly under SYNTHETIC_DIR
DATASET_FILES: Optional[List[str]] = None

# If True (default), reuse the ILP objective/time already computed for the
# paper instead of re-solving the ILP here. Only the greedy heuristic is run.
REUSE_EXISTING_ILP_RESULTS = True
EXISTING_ILP_RESULT_DIRS = [
    os.path.join(SYNTHETIC_DIR, "..", "results", "1"),  # synthetic_graphs/results/1
    os.path.join(SYNTHETIC_DIR, "..", "results"),        # synthetic_graphs/results
]
ILP_TIME_LIMIT = 10 ** 9  # only used if a fresh ILP solve is needed as a fallback
# -----------------------------

# Add the pure_ilp_solver directory to the path (expects: ./pure_ilp_solver/pure_ilp_temporal_separator.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "pure_ilp_solver"))
from pure_ilp_temporal_separator import PureILPTemporalSeparator  # noqa

import gurobipy as gp  # noqa  (only needed for the ILP fallback path)
from gurobipy import GRB  # noqa


# -----------------------------
# Display names matching the paper's tables
# -----------------------------
DATASET_DISPLAY_NAMES = {
    "anaheim": "Anaheim",
    "barcelona": "Barcelona",
    "friedrichshain-center": "Berlin--Friedrichshain",
    "berlin-prenzlauerberg-center": "Berlin--Prenzlauerberg",
    "chicagosketch": "ChicagoSketch",
    "ema": "Eastern--Massachusetts",
    "munich": "Munich",
}


def display_name(city_key: str) -> str:
    return DATASET_DISPLAY_NAMES.get(city_key.lower(), city_key)


# -----------------------------
# Reuse existing ILP results (same logic as run_lp_relaxation_comparison_synthetic.py)
# -----------------------------
_RESULT_OBJ_RE = re.compile(r"^\s*objective_value:\s*([\d.]+|inf)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_TIME_RE = re.compile(r"^\s*solve_wall_time_seconds:\s*([\d.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_PATHS_RE = re.compile(r"^\s*num_paths:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_VARS_RE = re.compile(r"^\s*num_variables:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_CONSTR_RE = re.compile(r"^\s*num_constraints:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_STATUS_RE = re.compile(r"^\s*status:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_SOURCE_RE = re.compile(r"^\s*source:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_RESULT_TARGET_RE = re.compile(r"^\s*target:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def find_existing_ilp_result(city: str, Tmax: int, d: int) -> Optional[str]:
    fname = f"{city}_TS{Tmax}_d{d}.txt"
    for base in EXISTING_ILP_RESULT_DIRS:
        candidate = os.path.normpath(os.path.join(base, fname))
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_existing_ilp_result(path: str, expected_source: int, expected_target: int) -> Tuple[float, float, Dict[str, Any]]:
    text = open(path, "r", encoding="utf-8").read()

    m_obj = _RESULT_OBJ_RE.search(text)
    m_time = _RESULT_TIME_RE.search(text)
    m_paths = _RESULT_PATHS_RE.search(text)
    m_vars = _RESULT_VARS_RE.search(text)
    m_constr = _RESULT_CONSTR_RE.search(text)
    m_status = _RESULT_STATUS_RE.search(text)
    m_source = _RESULT_SOURCE_RE.search(text)
    m_target = _RESULT_TARGET_RE.search(text)

    if not m_obj or not m_time:
        raise ValueError(f"Could not parse objective/time from existing result file: {path}")

    if m_source and m_target:
        found_s, found_z = int(m_source.group(1)), int(m_target.group(1))
        if found_s != expected_source or found_z != expected_target:
            raise ValueError(
                f"Existing result file {path} has source={found_s} target={found_z}, "
                f"but the current synthetic header has source={expected_source} "
                f"target={expected_target} -- refusing to reuse a mismatched instance."
            )

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
        """
        Find ONE temporal path from source to target with trt(P) <= deadline
        that is NOT separated by `intervals` (a dict v -> [l, r]), without
        enumerating the full path set P_{s,z,d}.

        Under Definition 1's separation rule -- "(v,I_v) separates each
        temporal path that traverses an OUTGOING temporal arc (vu,t) with
        t in I_v" -- a vertex v cannot depart via an arc at time t if t is
        already inside I_v (such a departure could only ever be part of an
        already-separated path). So the blocking check here tests the
        DEPARTING vertex, not the arriving one.

        For each distinct timestamp at which the source has an outgoing arc
        (each candidate t_first), runs a breadth-first search over
        (vertex, arrival-time) states -- bounded by the size of the temporal
        graph, not by the number of paths. It returns as soon as the target
        is reached, so for the given t_first the returned path has the
        fewest possible hops; t_first candidates are tried in increasing
        order. Returns None if every temporal path within the deadline is
        already separated by `intervals`.

        The returned path pairs each vertex with the timestamp of ITS OWN
        outgoing arc on the path (matching
        PureILPTemporalSeparator.find_temporal_paths()'s convention), e.g.
        [(s,t1), (u1,t2), ..., (uh,t_{h+1})] -- the target never appears,
        since it has no outgoing arc on this path.
        """
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
                            # at the very start; never re-enter it. Without
                            # this guard, a later return to the source would
                            # be treated as a fresh, unrestricted launch
                            # point (not limited to this branch's t_first),
                            # which can both violate the "each vertex once"
                            # path requirement and admit shortcuts that
                            # shouldn't be reachable under this t_first.
                            continue
                        state = (vv, next_time)
                        if state in visited:
                            continue
                        visited.add(state)
                        parent[state] = (v, cur_t)
                        if vv == self.target:
                            # Reconstruct the arrival-time chain first (the
                            # search states), then shift to the
                            # outgoing-arc pairing: each vertex is paired
                            # with the NEXT chain entry's timestamp, i.e.
                            # its own departing arc's time.
                            chain = [state]
                            back = (v, cur_t)
                            while back != start:
                                chain.append(back)
                                back = parent[back]
                            chain.append(start)
                            chain.reverse()
                            # chain = [(s,0), (u1,t1), ..., (uh,th), (z,t_{h+1})]
                            path = [
                                (chain[i][0], chain[i + 1][1])
                                for i in range(len(chain) - 1)
                            ]
                            return path
                        queue.append(state)

        return None

    def solve_greedy(self):
        """
        Greedy-TimeSep heuristic. See Algorithm "Greedy-TimeSep" in the paper
        (design agreed with the coauthor: no full enumeration of P_{s,z,d}).

        Repeats: find ONE currently-unseparated temporal path p via
        find_one_unseparated_path(); if none exists, stop. Otherwise select
        the vertex v* on p (excluding s,z) with maximum degree in the
        underlying static graph D, and extend I_{v*} directly to include the
        timestamp t* of v*'s arc on p (taking min/max with the current
        interval, or [t*,t*] if I_{v*} was empty). This separates p and
        keeps I_{v*} contiguous by construction. No optimality guarantee is
        claimed for MinTCover.
        """
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
                # can never be blocked by any vertex interval -- the same
                # limitation the exact ILP has (its constraint builder simply
                # omits such paths, since they never yield a usable
                # constraint). Nothing to select here; stop.
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
            # No separator can ever block an unblockable direct s-z arc, so
            # report this the same way solve_separator() reports INFEASIBLE
            # (sep=None, obj=inf) rather than a misleadingly "successful"
            # empty/partial separator with a finite objective.
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
# Synthetic input parsing (same header format as the other runners)
# -----------------------------
_SYN_SOURCE_RE = re.compile(r"^\s*#\s*source\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_TARGET_RE = re.compile(r"^\s*#\s*target\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_RE = re.compile(r"^\s*#\s*deadline\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_MAXT_RE = re.compile(r"^\s*#\s*max_timestamp\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_HORIZON_RE = re.compile(r"^\s*#\s*Max\s+timestamps\s*\(horizon\s*T\)\s*:\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_NOTE_RE = re.compile(r"^\s*#\s*deadline_note\s*=\s*(.*)\s*$", re.IGNORECASE)


def parse_city_from_filename(path: str) -> str:
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

    Tmax = max_timestamp_header if max_timestamp_header is not None else (horizon_T if horizon_T is not None else max_t_seen)

    meta = {
        "num_nodes": len(nodes),
        "temporal_edges": len(edges),
        "nodes_set": nodes,
        "max_timestamp": int(Tmax),
        "max_timestamp_seen": int(max_t_seen),
        "deadline_note": deadline_note,
    }
    return edges, meta, int(source), int(target), int(deadline)


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

        f.write("\nINSTANCE\n")
        f.write(f"  source: {source}\n")
        f.write(f"  target: {target}\n")
        f.write(f"  deadline(travel_time): {deadline}\n")
        f.write(f"  max_timestamp(T): {max_timestamp}\n")
        f.write(f"  mode: {mode}\n")
        if meta.get("deadline_note"):
            f.write(f"  deadline_note: {meta.get('deadline_note')}\n")

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
# Discover input files
# -----------------------------
def discover_dataset_files() -> List[str]:
    if DATASET_FILES:
        return [os.path.join(SYNTHETIC_DIR, f) if not os.path.isabs(f) else f for f in DATASET_FILES]

    pattern = os.path.join(SYNTHETIC_DIR, "synthetic_temporal_graph_*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No 'synthetic_temporal_graph_*.txt' files found under: {SYNTHETIC_DIR}")
    return files


# -----------------------------
# Run one dataset: ILP (reuse) + Greedy
# -----------------------------
def run_dataset(input_file: str, results_dir: str) -> Dict[str, Any]:
    city = parse_city_from_filename(input_file)
    temporal_edges, meta, s, z, d = read_synthetic_uvt(input_file)
    Tmax = meta["max_timestamp"]

    print(f"\n[DATASET] {display_name(city)}  (file: {os.path.basename(input_file)})")
    print(f"  nodes={meta['num_nodes']:,}  temporal_edges={meta['temporal_edges']:,}  T={Tmax}  d={d}  s={s} z={z}")

    row: Dict[str, Any] = {
        "dataset": display_name(city),
        "dataset_key": city,
        "vertices": meta["num_nodes"],
        "temporal_edges": meta["temporal_edges"],
        "TS": Tmax,
        "deadline": d,
    }

    # ---- ILP: reuse existing result if available, otherwise solve ----
    ilp_source = "resolved_now"
    existing_path = find_existing_ilp_result(city, Tmax, d) if REUSE_EXISTING_ILP_RESULTS else None

    if existing_path is not None:
        print(f"  [REUSE] Found existing ILP result: {existing_path}")
        try:
            obj_ilp, ilp_time, stats_ilp = parse_existing_ilp_result(existing_path, expected_source=s, expected_target=z)
            ilp_source = f"reused:{existing_path}"
            print(f"  [DONE] ILP (reused) status={stats_ilp.get('status')} SL={obj_ilp} time={ilp_time:.3f}s")
        except ValueError as e:
            print(f"  [WARN] {e}\n  [WARN] Falling back to solving ILP now.")
            existing_path = None

    if existing_path is None:
        if REUSE_EXISTING_ILP_RESULTS:
            print(f"  [WARN] No usable existing ILP result found for {city}_TS{Tmax}_d{d}.txt in "
                  f"{EXISTING_ILP_RESULT_DIRS} -- solving ILP now.")
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
        f"Greedy solver instance mismatch for {city}: "
        f"built with source={solver_greedy.source} target={solver_greedy.target} deadline={solver_greedy.deadline}, "
        f"expected source={s} target={z} deadline={d}"
    )
    print(f"  [CHECK] Greedy solver confirmed for THIS dataset: source={solver_greedy.source} "
          f"target={solver_greedy.target} deadline={solver_greedy.deadline} ({display_name(city)} expects s={s} z={z} d={d})")
    sep_greedy, obj_greedy, stats_greedy = solver_greedy.solve_greedy()
    greedy_time = stats_greedy.get("solve_time", 0.0)
    print(f"  [DONE] Greedy status={stats_greedy.get('status')} SL={obj_greedy} "
          f"iterations={stats_greedy.get('num_iterations')} time={greedy_time:.3f}s")

    write_detail_txt(
        os.path.join(results_dir, f"{city}_greedy.txt"),
        input_file=input_file, city=city, source=s, target=z, deadline=d,
        max_timestamp=Tmax, mode="GREEDY", meta=meta, sep=sep_greedy, obj=obj_greedy,
        stats=stats_greedy, solve_wall_time=greedy_time,
    )

    # ---- Comparison metrics ----
    ratio = None
    if obj_ilp not in (None, float("inf")) and obj_ilp > 0 and obj_greedy not in (None, float("inf")):
        ratio = obj_greedy / obj_ilp
    speedup = None
    if greedy_time > 0:
        speedup = ilp_time / greedy_time

    row.update({
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
    })
    return row


# -----------------------------
# Output writers
# -----------------------------
CSV_FIELDS = [
    "dataset", "vertices", "temporal_edges", "TS", "deadline",
    "ilp_SL", "ilp_time_s", "greedy_SL", "greedy_time_s",
    "ratio_greedy_over_ilp", "speedup_ilp_over_greedy",
    "greedy_num_iterations", "ilp_status", "greedy_status", "ilp_source",
]


def write_csv(rows: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})


def write_txt_table(rows: List[Dict[str, Any]], out_path: str):
    headers = ["Dataset", "SL(ILP)", "SL(Greedy)", "Ratio", "ILP time(s)", "Greedy time(s)", "Speedup"]
    lines = []
    lines.append("ILP vs Greedy-heuristic comparison (synthetic transportation networks)")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    col_w = [24, 10, 12, 8, 12, 15, 10]
    lines.append("".join(h.ljust(w) for h, w in zip(headers, col_w)))
    lines.append("-" * sum(col_w))
    for r in rows:
        vals = [
            str(r["dataset"]),
            f"{r['ilp_SL']}",
            f"{r['greedy_SL']}",
            f"{r['ratio_greedy_over_ilp']}" if r["ratio_greedy_over_ilp"] is not None else "n/a",
            f"{r['ilp_time_s']}",
            f"{r['greedy_time_s']}",
            f"{r['speedup_ilp_over_greedy']}" if r["speedup_ilp_over_greedy"] is not None else "n/a",
        ]
        lines.append("".join(v.ljust(w) for v, w in zip(vals, col_w)))
    lines.append("")
    lines.append("ILP source per dataset:")
    for r in rows:
        src = r.get("ilp_source", "unknown")
        tag = "reused (original paper run)" if src.startswith("reused:") else "resolved in this run"
        lines.append(f"  {r['dataset']}: {tag}" + (f"  [{src[len('reused:'):]}]" if src.startswith("reused:") else ""))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_snippet(rows: List[Dict[str, Any]], out_path: str):
    lines = []
    lines.append("% Auto-generated by run_greedy_comparison_synthetic.py")
    lines.append("% Paste rows into a table with columns:")
    lines.append("% Dataset & SL (ILP) & SL (Greedy) & Ratio & ILP time (s) & Greedy time (s) & Speedup")
    for r in rows:
        ratio = f"{r['ratio_greedy_over_ilp']:.2f}" if r["ratio_greedy_over_ilp"] is not None else "--"
        speedup = f"{r['speedup_ilp_over_greedy']:.1f}" if r["speedup_ilp_over_greedy"] is not None else "--"
        lines.append(
            f"{r['dataset']} & {r['ilp_SL']} & {r['greedy_SL']} & {ratio} & "
            f"{r['ilp_time_s']:.1f} & {r['greedy_time_s']:.3f} & {speedup} \\\\"
        )
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("=" * 80)
    print("ILP vs GREEDY BATCH COMPARISON (SYNTHETIC, NO CLI)")
    print("=" * 80)
    print(f"Synthetic dir: {SYNTHETIC_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    mem = psutil.virtual_memory()
    print(f"[INFO] Available memory: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

    files = discover_dataset_files()
    print(f"\n[PHASE] Discovered {len(files)} dataset(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    results_dir = os.path.join(SYNTHETIC_DIR, "results", "greedy_comparison")
    os.makedirs(results_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, input_file in enumerate(files, start=1):
        print("\n" + "-" * 80)
        print(f"[RUN {idx}/{len(files)}]")
        try:
            row = run_dataset(input_file, results_dir)
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] Failed on {input_file}: {e}")

    csv_path = os.path.join(results_dir, "comparison_summary.csv")
    txt_path = os.path.join(results_dir, "comparison_summary.txt")
    tex_path = os.path.join(results_dir, "comparison_table_snippet.tex")

    write_csv(rows, csv_path)
    write_txt_table(rows, txt_path)
    write_latex_snippet(rows, tex_path)

    print("\n" + "=" * 80)
    print("[DONE] Batch comparison finished.")
    print(f"  CSV : {csv_path}")
    print(f"  TXT : {txt_path}")
    print(f"  TEX : {tex_path}")
    print(f"  Detail files in: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
