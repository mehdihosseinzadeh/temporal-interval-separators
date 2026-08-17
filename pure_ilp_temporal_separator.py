#!/usr/bin/env python3
"""
Pure ILP-Based Temporal Separator Solver

Implements the mathematical formulation for temporal separator computation using
Integer Linear Programming (ILP) with Gurobi optimizer.

"""

import bisect
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional, Any
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureILPTemporalSeparator:
    """
    Pure ILP-based temporal separator solver implementing the academic formulation.
    """

    def __init__(
        self,
        temporal_edges: List[Tuple[int, int, int]],
        source: int,
        target: int,
        deadline: int,
        max_timestamp: int,
    ):
        """
        Initialize the temporal separator solver.

        Args:
            temporal_edges: List of (u, v, t) temporal edges
            source: Source vertex s
            target: Target vertex z
            deadline: Maximum TRAVEL TIME d (trt(P) <= d)
            max_timestamp: Maximum timestamp T
        """
        self.temporal_edges = temporal_edges
        self.source = source
        self.target = target
        self.deadline = int(deadline)
        self.max_timestamp = int(max_timestamp)

        # Build temporal graph structure
        self.vertices = set()
        self.temporal_graph = defaultdict(list)  # t -> [(u, v)]

        for u, v, t in temporal_edges:
            self.vertices.add(u)
            self.vertices.add(v)
            self.temporal_graph[int(t)].append((int(u), int(v)))

        self.vertices = sorted(list(self.vertices))

        logger.info(f"Initialized with {len(self.vertices)} vertices, {len(temporal_edges)} edges")
        logger.info(
            f"Source: {self.source}, Target: {self.target}, "
            f"Deadline(travel-time): {self.deadline}, Max timestamp: {self.max_timestamp}"
        )

    def find_temporal_paths(self, max_paths: int = None) -> List[List[Tuple[int, int]]]:
        """
        Find temporal paths from source to target with:
          - strictly increasing times
          - travel-time constraint: trt(P) = t_last - t_first + 1 <= self.deadline

        Returns:
            paths: list of paths, each path is a list of (vertex, timestamp) pairs,
                   where each vertex is paired with the timestamp of ITS OWN
                   outgoing temporal arc on the path (matching Definition 1 /
                   the paper's separation rule: "(v,I_v) separates each temporal
                   path that traverses an outgoing temporal arc (vu,t) with
                   t in I_v"). E.g. for P = (s,(s u1,t1),u1,(u1 u2,t2),u2,...,
                   (uh z,t_{h+1}),z), the returned path is
                   [(s,t1), (u1,t2), (u2,t3), ..., (uh,t_{h+1})] -- the target
                   itself never appears, since it has no outgoing arc on P.
        """
        paths: List[List[Tuple[int, int]]] = []

        # Only iterate over timestamps that actually exist
        times_sorted = sorted(self.temporal_graph.keys())

        def dfs(
            current_vertex: int,
            current_time: int,
            t_first: Optional[int],  # timestamp of first edge in path
            path: List[Tuple[int, int]],
            visited_vt: Set[Tuple[int, int]],
        ):
            if max_paths is not None and len(paths) >= max_paths:
                return

            # If we reached target, accept only if travel time bound holds
            if current_vertex == self.target:
                if t_first is None:
                    # would only happen if source==target; disallowed elsewhere
                    return
                trt = current_time - t_first + 1
                if trt <= self.deadline:
                    paths.append(path.copy())
                return

            # If path already started, prune by travel time
            if t_first is not None:
                trt_now = current_time - t_first + 1
                if trt_now > self.deadline:
                    return

            # Next time must be strictly greater than current_time
            idx = bisect.bisect_right(times_sorted, current_time)
            for next_time in times_sorted[idx:]:
                # If already started, prune future expansions as soon as travel-time would exceed d
                if t_first is not None:
                    trt_next = next_time - t_first + 1
                    if trt_next > self.deadline:
                        break  # further next_time only increases trt

                for u, v in self.temporal_graph[next_time]:
                    if u != current_vertex:
                        continue
                    if (v, next_time) in visited_vt:
                        continue

                    new_t_first = next_time if t_first is None else t_first

                    # Redundant safety check
                    if (next_time - new_t_first + 1) > self.deadline:
                        continue

                    # Pair the DEPARTING vertex (current_vertex) with this
                    # arc's own timestamp -- current_vertex's outgoing-arc
                    # time -- per Definition 1's outgoing-arc separation
                    # rule. (visited_vt still dedupes by arrival state, to
                    # prevent revisiting the same physical (vertex, time).)
                    path.append((current_vertex, next_time))
                    visited_vt.add((v, next_time))
                    dfs(v, next_time, new_t_first, path, visited_vt)
                    path.pop()
                    visited_vt.remove((v, next_time))

        # Start DFS from source at "time 0" (no edge taken yet). The path
        # list starts empty -- source's own entry (source, t_first) is
        # appended naturally as soon as its first outgoing arc is taken.
        initial_path: List[Tuple[int, int]] = []
        initial_visited = {(self.source, 0)}
        dfs(self.source, 0, None, initial_path, initial_visited)

        if max_paths is not None and len(paths) >= max_paths:
            logger.warning(
                f"Path enumeration stopped at limit {max_paths}. This may lead to incomplete ILP constraints!"
            )

        logger.info(
            f"Found {len(paths)} temporal paths from {self.source} to {self.target} "
            f"(travel-time deadline d={self.deadline})"
        )
        return paths

    def solve_separator(self, time_limit: int = 300) -> Tuple[Optional[Dict[Tuple[int, int], int]], float, Dict]:
        """
        Solve the ILP formulation for temporal separator.

        NOTE: deadline is travel-time bound, so ILP variables are defined over full timeline t=1..T.

        Returns:
            - separator: Dict mapping (vertex, timestamp) to 1 if in separator, None if infeasible
            - objective_value: Total separator timeline length
            - stats: Solving statistics
        """
        start_time = time.time()

        # Find feasible temporal paths (travel-time bounded)
        paths = self.find_temporal_paths()
        if not paths:
            logger.warning("No temporal paths found (within travel-time deadline)")
            return None, float("inf"), {
                "status": "no_paths",
                "solve_time": time.time() - start_time,
                "num_paths": 0,
            }

        # Create ILP model
        model = gp.Model("temporal_separator")
        model.setParam("OutputFlag", 0)  # Suppress output
        model.setParam("TimeLimit", int(time_limit))

        # Decision variables x_{v,t}, for ALL t in 1..T (deadline is NOT a timestamp cap)
        x: Dict[Tuple[int, int], Any] = {}
        separator_vertices = [v for v in self.vertices if v != self.source and v != self.target]

        for v in separator_vertices:
            for t in range(1, self.max_timestamp + 1):
                x[v, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{t}")

        # Objective: minimize sum x_{v,t} over full timeline
        model.setObjective(
            gp.quicksum(x[v, t] for v in separator_vertices for t in range(1, self.max_timestamp + 1)),
            GRB.MINIMIZE,
        )

        # Path separation constraints: each feasible path must be intersected
        # Only include variables for separator vertices (exclude s,z) and times t>=1
        for i, path in enumerate(paths):
            constraint_vars = []
            for v, t in path:
                if t >= 1 and v in separator_vertices:
                    constraint_vars.append(x[v, t])

            if constraint_vars:
                model.addConstr(gp.quicksum(constraint_vars) >= 1, name=f"path_separation_{i}")

        # Contiguity constraints (as in your original code), over full timeline 1..T
        T = self.max_timestamp
        for v in separator_vertices:
            for t1 in range(1, T):
                for t2 in range(t1 + 2, T + 1):
                    model.addConstr(
                        x[v, t1] + x[v, t2] - 1 <= x[v, t1 + 1],
                        name=f"contiguity_{v}_{t1}_{t2}",
                    )

        logger.info(f"Created ILP with {len(x)} variables and {len(paths)} path constraints")

        # Solve
        solve_start = time.time()
        model.optimize()
        solve_time = time.time() - solve_start

        stats = {
            "status": model.status,
            "solve_time": solve_time,
            "total_time": time.time() - start_time,
            "num_variables": len(x),
            "num_paths": len(paths),
            "num_constraints": model.NumConstrs,
        }

        if model.status == GRB.OPTIMAL:
            separator: Dict[Tuple[int, int], int] = {}
            for v in separator_vertices:
                for t in range(1, T + 1):
                    if x[v, t].X > 0.5:
                        separator[(v, t)] = 1

            objective_value = model.objVal
            stats["objective_value"] = objective_value

            logger.info(f"Optimal solution found: objective = {objective_value}")
            logger.info(f"Separator size: {len(separator)} vertex-time pairs")
            return separator, objective_value, stats

        if model.status == GRB.INFEASIBLE:
            logger.error("Model is infeasible")
            return None, float("inf"), stats

        if model.status == GRB.TIME_LIMIT:
            logger.warning("Time limit reached")
            if model.SolCount > 0:
                separator = {}
                for v in separator_vertices:
                    for t in range(1, T + 1):
                        if x[v, t].X > 0.5:
                            separator[(v, t)] = 1
                return separator, model.objVal, stats
            return None, float("inf"), stats

        logger.error(f"Unexpected solver status: {model.status}")
        return None, float("inf"), stats

    def verify_separator(self, separator: Dict[Tuple[int, int], int]) -> bool:
        """
        Verify that the separator blocks all feasible temporal paths
        (i.e., all s->z paths with travel time <= deadline).
        """
        paths = self.find_temporal_paths()

        logger.info(f"Found {len(paths)} temporal paths from {self.source} to {self.target} (for verification)")

        for path in paths:
            blocked = False
            for v, t in path:
                if (v, t) in separator:
                    blocked = True
                    break
            if not blocked:
                logger.error(f"Path not blocked: {path}")
                return False

        logger.info(f"Separator verified: blocks all {len(paths)} feasible paths")
        return True

    def print_separator_summary(self, separator: Dict[Tuple[int, int], int], stats: Dict):
        if separator is None:
            print("No feasible separator found")
            return

        print("\n" + "=" * 60)
        print("TEMPORAL SEPARATOR SOLUTION SUMMARY")
        print("=" * 60)

        print(f"Problem: s={self.source} -> z={self.target}, travel-time deadline d={self.deadline}")
        print(f"Graph: {len(self.vertices)} vertices, {len(self.temporal_edges)} temporal edges")
        print(f"Time horizon: 1 to {self.max_timestamp}")

        print("\nSolver Statistics:")
        print(f"  Status: {stats.get('status', 'unknown')}")
        print(f"  Solve time: {stats.get('solve_time', 0):.3f}s")
        print(f"  Total time: {stats.get('total_time', 0):.3f}s")
        print(f"  Variables: {stats.get('num_variables', 0)}")
        print(f"  Constraints: {stats.get('num_constraints', 0)}")
        print(f"  Feasible paths found: {stats.get('num_paths', 0)}")

        print("\nSeparator Solution:")
        print(f"  Objective value: {stats.get('objective_value', 0)}")
        print(f"  Separator size: {len(separator)} vertex-time pairs")

        # Group by vertex
        vertex_times = defaultdict(list)
        for (v, t), val in separator.items():
            if val == 1:
                vertex_times[v].append(t)

        print("\nVertex Times:")
        for v in sorted(vertex_times.keys()):
            times = sorted(vertex_times[v])
            print(f"  Vertex {v}: {times}")

        print("=" * 60)


def main():
    # Tiny example
    temporal_edges = [
        (1, 2, 10),
        (2, 3, 20),
        (1, 3, 50),
        (3, 4, 60),
    ]

    source = 1
    target = 4
    max_timestamp = 60

    # Travel-time deadline:
    # Path 1->2 (10), 2->3 (20), 3->4 (60) has trt = 60 - 10 + 1 = 51
    deadline = 55

    print("Pure ILP Temporal Separator Solver (travel-time deadline)")
    print("========================================================")

    solver = PureILPTemporalSeparator(temporal_edges, source, target, deadline, max_timestamp)
    separator, objective_value, stats = solver.solve_separator(time_limit=60)

    solver.print_separator_summary(separator, stats)

    if separator is not None:
        ok = solver.verify_separator(separator)
        print(f"\nSeparator validation: {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()