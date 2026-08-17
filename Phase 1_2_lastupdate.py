#!/usr/bin/env python3
"""
Synthetic Temporal Directed Graph Generator

"""

import os
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ======================================================================
# CONFIG (CHANGE EVERYTHING YOU WANT HERE ONLY)
# ======================================================================
CONFIG = {
    # I/O
    "STATIC_GRAPH_FILE": "/Users/mehdi/Desktop/Network_Analysis/TSD/data/friedrichshain-center_edges.txt",
    "OUTPUT_DIR": "/Users/mehdi/Desktop/Network_Analysis/TSD/python/synthetic_graphs/ts50",

    # Horizon / iterations
    "T_MAX": 50,
    "MAX_ITERATIONS": 50,  # safety cap for Phase 1 loop

    # -------- Phase 1 timestamps (aligned-run schedule) --------
    "STEP": 2,                          # fixed to 2
    "ALIGNED_RUNS_RANGE": (4, 6),       # inclusive; randomized PER Phase-1 path (>=2)
    "START_TIME_BIAS": "first_third",   # "first_third" or "full"
    "PHASE1_ADD_MISSING_TIMES_ON_SHARED_EDGES": True,

    # -------- Deadline rule (paper sentence) --------
    "DEADLINE_MULTIPLIER": 3,
    "DEADLINE_MIN_FRAC_OF_T": 0.50,     # raise to 50% of T if below

    # -------- Phase 2: noise edges --------
    "TEMPORALIZE_ALL_REMAINING_EDGES": True,
    "MAX_RANDOM_EDGES": None,           # ignored if TEMPORALIZE_ALL_REMAINING_EDGES=True

    "NOISE_TIMESTAMPS_PER_EDGE_RANGE": (2, 5),  # inclusive
    "NOISE_TIME_WINDOW": None,                 # None => full [1..T_MAX]

    # Reproducibility
    "SEED": 42,
}
# ======================================================================


# ============================================================
# helpers: load graph + degree-based s,z selection
# ============================================================
def load_static_graph(path: str) -> Tuple[Dict[int, Set[int]], Set[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Static graph file not found: {path}")

    g = defaultdict(set)
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            g[u].add(v)
            nodes.add(u)
            nodes.add(v)

    return dict(g), nodes


def select_source_target_high_degree(g: Dict[int, Set[int]], nodes: Set[int]) -> Tuple[int, int, int, int]:
    """
    Select:
      s = argmax out-degree (ties -> smallest id)
      z = argmax in-degree, z != s (ties -> smallest id)
    Degrees computed on ORIGINAL static graph.
    Returns: (s, z, outdeg[s], indeg[z])
    """
    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    for u, nbrs in g.items():
        outdeg[u] += len(nbrs)
        for v in nbrs:
            indeg[v] += 1

    for v in nodes:
        _ = indeg[v]
        _ = outdeg[v]

    max_out = max(outdeg[v] for v in nodes)
    s_candidates = sorted([v for v in nodes if outdeg[v] == max_out])
    s = s_candidates[0]

    remaining = [v for v in nodes if v != s]
    if not remaining:
        raise ValueError("Graph has only one node; cannot choose distinct source/target.")

    max_in = max(indeg[v] for v in remaining)
    z_candidates = sorted([v for v in remaining if indeg[v] == max_in])
    z = z_candidates[0]

    return s, z, outdeg[s], indeg[z]


def bfs_shortest_path(g: Dict[int, Set[int]], s: int, z: int) -> Optional[List[int]]:
    if s == z:
        return [s]
    if s not in g:
        return None

    q = deque([s])
    parent = {s: None}

    while q:
        u = q.popleft()
        if u == z:
            path = []
            cur = z
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        for v in g.get(u, ()):
            if v not in parent:
                parent[v] = u
                q.append(v)

    return None


def remove_internal_vertices_and_incident_arcs(g: Dict[int, Set[int]], path: List[int]) -> int:
    """
    Remove internal vertices (excluding endpoints) and all incident arcs.
    Returns number of internal vertices removed.
    """
    if len(path) < 3:
        return 0

    internal = set(path[1:-1])
    if not internal:
        return 0

    # remove outgoing arcs
    for v in list(internal):
        if v in g:
            del g[v]

    # remove incoming arcs
    for u in list(g.keys()):
        nbrs = g[u]
        new_nbrs = nbrs - internal
        if new_nbrs:
            g[u] = new_nbrs
        else:
            del g[u]

    return len(internal)


class SyntheticTemporalGraphGenerator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        random.seed(cfg["SEED"])
        np.random.seed(cfg["SEED"])

        self.T = int(cfg["T_MAX"])
        self.max_iterations = int(cfg["MAX_ITERATIONS"])

        if int(cfg["STEP"]) != 2:
            raise ValueError("STEP must be 2 (per your generator design).")
        self.step = 2

        rmin, rmax = cfg["ALIGNED_RUNS_RANGE"]
        rmin = int(rmin)
        rmax = int(rmax)
        if rmin < 2 or rmax < rmin:
            raise ValueError("ALIGNED_RUNS_RANGE must be (>=2, >=min).")
        self.R_range = (rmin, rmax)

        # Load original graph
        self.original_graph, self.all_nodes = load_static_graph(cfg["STATIC_GRAPH_FILE"])

        # Choose s,z on ORIGINAL graph
        self.source, self.target, s_out, z_in = select_source_target_high_degree(self.original_graph, self.all_nodes)

        # Working graph for phase-1 removals
        self.working_graph = {u: set(vs) for u, vs in self.original_graph.items()}

        total_edges = sum(len(neighbors) for neighbors in self.original_graph.values())
        print(f"Loaded graph with {len(self.all_nodes)} nodes and {total_edges} edges")
        print(f"Source: {self.source} (max out-degree={s_out})")
        print(f"Target: {self.target} (max in-degree={z_in})")
        print(f"Phase1 schedule: step={self.step}, aligned_runs_range={self.R_range}")

        # temporal edges + dedup
        self.temporal_edges: List[Tuple[int, int, int]] = []
        self.edge_to_times: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        # Phase 1 bookkeeping
        self.phase1_paths: List[List[int]] = []
        self.phase1_tp1: List[int] = []
        self.phase1_new_instances: List[int] = []
        self.phase1_added_temporal_edges: List[List[Tuple[int, int, int]]] = []
        self.phase1_R_used: List[int] = []

        # Deadline
        self.deadline: int = self.T
        self.deadline_note: str = "UNCOMPUTED"

    def _path_edges(self, path: List[int]) -> List[Tuple[int, int]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # -------------------------------
    # Phase 1 timestamp assignment (aligned-run schedule)
    # -------------------------------
    def _assign_timestamps_for_path(self, path: List[int], idx: int) -> int:
        """
        aligned-run schedule:
            t = tp1 + offset + 2*i, offset in [0..R-1]
        Add only missing times per static edge (u,v).
        """
        if len(path) < 2:
            # keep bookkeeping aligned by idx
            self.phase1_tp1.append(-1)
            self.phase1_new_instances.append(0)
            self.phase1_added_temporal_edges.append([])
            self.phase1_R_used.append(0)
            return 0

        edges = self._path_edges(path)
        L = len(edges)

        # R is randomized PER PATH
        R = random.randint(self.R_range[0], self.R_range[1])
        self.phase1_R_used.append(R)

        step = self.step

        # last timestamp on last edge: tp1 + 2*(L-1) + (R-1)
        max_start = self.T - ((R - 1) + step * (L - 1))
        if max_start < 1:
            print(f"[WARN] Phase1 path {idx}: too long (L={L}) for T={self.T} with R={R}, step=2")
            self.phase1_tp1.append(-1)
            self.phase1_new_instances.append(0)
            self.phase1_added_temporal_edges.append([])
            return 0

        if self.cfg["START_TIME_BIAS"] == "first_third":
            upper = min(max_start, max(1, self.T // 3))
        else:
            upper = max_start

        tp1 = random.randint(1, upper)
        add_missing_on_shared = bool(self.cfg["PHASE1_ADD_MISSING_TIMES_ON_SHARED_EDGES"])

        added = 0
        added_list: List[Tuple[int, int, int]] = []

        for i, (u, v) in enumerate(edges):
            base_t = tp1 + step * i
            desired_times = [base_t + o for o in range(R)]

            already = self.edge_to_times[(u, v)]
            if already and (not add_missing_on_shared):
                continue

            for t in desired_times:
                if 1 <= t <= self.T and (t not in already):
                    self.edge_to_times[(u, v)].add(t)
                    triple = (u, v, t)
                    self.temporal_edges.append(triple)
                    added_list.append(triple)
                    added += 1

        self.phase1_tp1.append(tp1)
        self.phase1_new_instances.append(added)
        self.phase1_added_temporal_edges.append(added_list)
        return added

    # -------------------------------
    # Deadline (paper sentence)
    # -------------------------------
    def _compute_deadline_from_first_path(self) -> Tuple[int, str]:
        """
        d = 3 * L0
        cap at T
        floor to ceil(0.5*T) if too small
        """
        if not self.phase1_paths:
            return self.T, "NO_PHASE1_PATH -> deadline=T"

        L0 = len(self.phase1_paths[0]) - 1
        if L0 <= 0:
            return self.T, "FIRST_PATH_EMPTY -> deadline=T"

        mult = int(self.cfg["DEADLINE_MULTIPLIER"])
        d = mult * L0

        if d > self.T:
            d = self.T

        min_frac = float(self.cfg["DEADLINE_MIN_FRAC_OF_T"])
        floor_val = int(math.ceil(min_frac * self.T))
        if d < floor_val:
            d = floor_val

        note = f"paper_deadline: L0={L0}, mult={mult}, cap_T={self.T}, min_frac={min_frac}"
        return int(d), note

    # -------------------------------
    # Phase 1: shortest path + removals until separated
    # -------------------------------
    def _phase1_extract_paths(self):
        print("\n=== PHASE 1: SHORTEST-PATH + REMOVE INTERNAL VERTICES ===")
        it = 0
        while it < self.max_iterations:
            path = bfs_shortest_path(self.working_graph, self.source, self.target)
            if path is None:
                print(f"Stopped: s and z separated after {it} iterations (no more s->z path).")
                break

            it += 1
            self.phase1_paths.append(path)

            new_instances = self._assign_timestamps_for_path(path, idx=it)
            removed = remove_internal_vertices_and_incident_arcs(self.working_graph, path)
            remaining_E = sum(len(vs) for vs in self.working_graph.values())

            print(
                f"Iter {it}: path_len={len(path)-1} | new_temporal_instances={new_instances} | "
                f"removed_internal_vertices={removed} | remaining_E={remaining_E}"
            )

        print(f"\nPhase 1 complete! Paths extracted: {len(self.phase1_paths)}")

    # -------------------------------
    # Phase 2: temporalize remaining edges as noise
    # -------------------------------
    def _add_noise_edges(self):
        remaining = []
        for u, nbrs in self.working_graph.items():
            for v in nbrs:
                if (u, v) not in self.edge_to_times:
                    remaining.append((u, v))

        if not remaining:
            print("\nNo remaining edges to add as noise")
            return

        print("\n=== PHASE 2: ADD NOISE EDGES ===")
        print(f"[PHASE2] Remaining static edges (not yet temporalized): {len(remaining)}")

        if self.cfg["TEMPORALIZE_ALL_REMAINING_EDGES"]:
            edges_to_process = remaining
            print("  Using ALL remaining edges")
        else:
            cap = self.cfg["MAX_RANDOM_EDGES"]
            if cap is None or cap >= len(remaining):
                edges_to_process = remaining
                print(f"  Using all {len(remaining)} remaining edges")
            else:
                idx = np.random.choice(len(remaining), size=int(cap), replace=False)
                edges_to_process = [remaining[i] for i in idx]
                print(f"  Using {len(edges_to_process)} out of {len(remaining)} remaining edges")

        tw = self.cfg["NOISE_TIME_WINDOW"]
        if tw is None:
            a, b = 1, self.T
        else:
            a, b = int(tw[0]), int(tw[1])
            a = max(1, a)
            b = min(self.T, b)
            if a > b:
                a, b = 1, self.T

        available = np.arange(a, b + 1, dtype=int)

        kmin, kmax = self.cfg["NOISE_TIMESTAMPS_PER_EDGE_RANGE"]
        kmin = int(kmin)
        kmax = int(kmax)
        if kmin < 1 or kmax < kmin:
            raise ValueError("Invalid NOISE_TIMESTAMPS_PER_EDGE_RANGE")

        total_added = 0
        for (u, v) in edges_to_process:
            k = int(np.random.randint(kmin, kmax + 1))  # inclusive via +1
            if k > len(available):
                k = len(available)

            ts = np.random.choice(available, size=k, replace=False)
            for t in ts:
                t = int(t)
                if t not in self.edge_to_times[(u, v)]:
                    self.edge_to_times[(u, v)].add(t)
                    self.temporal_edges.append((u, v, t))
                    total_added += 1

        print(f"[PHASE2] Added noise temporal instances: {total_added}")
        print(f"[PHASE2] Noise timestamps window: [{a}, {b}] | avg per edge ~ {(total_added / max(1, len(edges_to_process))):.2f}")

    # -------------------------------
    # Public API
    # -------------------------------
    def generate(self):
        print("\nGenerating synthetic temporal graph...")
        print(f"T={self.T} | max_iterations={self.max_iterations}")
        print(f"Phase1: step={self.step} | aligned_runs_range={self.R_range} | start_bias={self.cfg['START_TIME_BIAS']}")

        # Phase 1
        self._phase1_extract_paths()

        # Deadline from first path
        self.deadline, self.deadline_note = self._compute_deadline_from_first_path()
        print(f"\nDeadline = {self.deadline}  ({self.deadline_note})")

        # Phase 2
        self._add_noise_edges()

        self.temporal_edges.sort(key=lambda e: (e[2], e[0], e[1]))
        return self.temporal_edges

    def save(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        max_t_present = max((t for _, _, t in self.temporal_edges), default=self.T)

        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Synthetic Temporal Directed Graph\n")
            f.write(f"# Generated from: {self.cfg['STATIC_GRAPH_FILE']}\n")
            f.write(f"# source = {self.source}\n")
            f.write(f"# target = {self.target}\n")
            f.write(f"# deadline = {self.deadline}\n")
            f.write(f"# deadline_note = {self.deadline_note}\n")
            f.write(f"# max_timestamp = {max_t_present}\n")
            f.write(f"# T (horizon) = {self.T}\n")
            f.write(f"# phase1_paths_extracted = {len(self.phase1_paths)}\n")
            f.write(f"# phase1_aligned_runs_range = {self.R_range}\n")
            f.write(f"# phase1_step = {self.step}\n")
            f.write(f"# phase2_noise_k_range = {self.cfg['NOISE_TIMESTAMPS_PER_EDGE_RANGE']}\n")
            f.write("# u v t\n")
            for (u, v, t) in self.temporal_edges:
                f.write(f"{u} {v} {t}\n")

        print(f"\n[OUT] {out_path}")

    def save_paths_info(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Synthetic Paths Info (Phase 1 extracted paths)\n")
            f.write(f"# Generated from: {self.cfg['STATIC_GRAPH_FILE']}\n")
            f.write(f"# source = {self.source}\n")
            f.write(f"# target = {self.target}\n")
            f.write(f"# T (horizon) = {self.T}\n")
            f.write(f"# aligned_runs_range = {self.R_range}\n")
            f.write(f"# step = {self.step}\n")
            f.write(f"# deadline = {self.deadline}\n")
            f.write(f"# deadline_note = {self.deadline_note}\n")
            f.write(f"# paths_extracted = {len(self.phase1_paths)}\n\n")

            for idx, path in enumerate(self.phase1_paths, start=1):
                L = len(path) - 1
                tp1 = self.phase1_tp1[idx - 1]
                new_inst = self.phase1_new_instances[idx - 1]
                added_edges = self.phase1_added_temporal_edges[idx - 1]
                R_used = self.phase1_R_used[idx - 1] if idx - 1 < len(self.phase1_R_used) else None

                f.write(f"Path {idx}\n")
                f.write(f"  length_edges = {L}\n")
                f.write(f"  tp1 = {tp1}\n")
                f.write(f"  aligned_runs_used = {R_used}\n")
                f.write(f"  new_temporal_instances_added = {new_inst}\n")
                f.write(f"  path = {' -> '.join(map(str, path))}\n")

                if added_edges:
                    ts = [t for _, _, t in added_edges]
                    f.write(f"  added_time_range = {min(ts)} - {max(ts)}\n")
                else:
                    f.write("  added_time_range = N/A\n")

                f.write("\n")

        print(f"[OUT] {out_path}")


def main():
    cfg = CONFIG
    gen = SyntheticTemporalGraphGenerator(cfg)

    edges = gen.generate()

    dataset_name = os.path.basename(cfg["STATIC_GRAPH_FILE"]).replace("_edges.txt", "").replace(".txt", "")
    out_dir = cfg["OUTPUT_DIR"]
    out_graph = os.path.join(out_dir, f"synthetic_temporal_graph_{dataset_name}.txt")
    out_paths = os.path.join(out_dir, f"synthetic_paths_info_{dataset_name}.txt")

    gen.save(out_graph)
    gen.save_paths_info(out_paths)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"source = {gen.source}")
    print(f"target = {gen.target}")
    print(f"deadline = {gen.deadline}  ({gen.deadline_note})")
    print(f"T (horizon) = {cfg['T_MAX']}")
    print(f"phase1_paths_extracted = {len(gen.phase1_paths)}")
    print(f"temporal_edges total = {len(edges)}")
    print(f"temporal_edges sample = {edges[:5]}")


if __name__ == "__main__":
    main()