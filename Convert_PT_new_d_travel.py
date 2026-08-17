#!/usr/bin/env python3
from pathlib import Path
import csv
import math
from collections import defaultdict

# ------------------------------------------------------------
# Paths (change ONLY the city name here)
# ------------------------------------------------------------
CITY   = "paris"
INPUT  = Path(f"/Users/mehdi/Desktop/Network_Analysis/TSD/data_new/{CITY}/network_temporal_day.csv")
OUTPUT = Path(f"/Users/mehdi/Desktop/Network_Analysis/TSD/data_new/{CITY}/network_temporal_day_uvt_first2h.txt")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DELIM = ";"
WINDOW_SECONDS = 2 * 60 * 60  # first 2 hours

# ============================================================
# NEW PARAMETER (only change this)
# ============================================================
NUM_SOURCES = 5
NUM_TARGETS = 5
# combos will always be full cartesian product => NUM_SOURCES * NUM_TARGETS
# ============================================================


# ============================================================
# shortest-travel-time temporal path
# ============================================================
def shortest_travel_time_path_stats(edges_sorted_by_t, source: int, target: int):
    """
    Find a temporal path (strictly increasing t) minimizing:
        travel_time = arrival_t - departure_t + 1

    Returns:
        (travel_time, departure_t, arrival_t, hops)
    If no path exists: returns (None, None, None, None)

    Notes:
    - We process edges grouped by timestamp t and apply updates after each t
      so same-t chaining is impossible => STRICT increase.
    - For each node v, we keep the best reachable state in terms of:
        latest departure time (start)  [dominates travel_time]
        tie-break: earlier arrival time
    """
    INF = 10**18

    best_start = defaultdict(lambda: -INF)  # latest departure time used to reach v
    best_arr   = defaultdict(lambda: INF)  # arrival time associated to best_start[v]
    best_hops  = defaultdict(lambda: INF)

    best_travel = INF
    best_dep_t = None
    best_arr_t = None
    best_target_hops = None

    i = 0
    n = len(edges_sorted_by_t)

    while i < n:
        t = edges_sorted_by_t[i][2]

        # Snapshot "before time t" to avoid same-t chaining (strict increase)
        snap_start = best_start.copy()
        snap_arr   = best_arr.copy()
        snap_hops  = best_hops.copy()

        # candidates discovered at this timestamp t
        cand = {}  # v -> (start, arr, hops)

        while i < n and edges_sorted_by_t[i][2] == t:
            u, v, _ = edges_sorted_by_t[i]

            # Start a new path from source with first edge at time t
            if u == source:
                start0 = t
                hops0 = 1
                prev = cand.get(v)
                if (prev is None) or (start0 > prev[0]) or (start0 == prev[0] and t < prev[1]):
                    cand[v] = (start0, t, hops0)

                if v == target:
                    travel = t - start0 + 1  # = 1
                    if travel < best_travel:
                        best_travel = travel
                        best_dep_t = start0
                        best_arr_t = t
                        best_target_hops = hops0

            # Extend an existing path: must have arrived at u strictly before t
            if snap_start[u] != -INF and snap_arr[u] < t:
                start0 = snap_start[u]
                hops0 = snap_hops[u] + 1
                prev = cand.get(v)
                if (prev is None) or (start0 > prev[0]) or (start0 == prev[0] and t < prev[1]):
                    cand[v] = (start0, t, hops0)

                if v == target:
                    travel = t - start0 + 1
                    if (travel < best_travel) or (travel == best_travel and t < (best_arr_t or INF)):
                        best_travel = travel
                        best_dep_t = start0
                        best_arr_t = t
                        best_target_hops = hops0

            i += 1

        # Apply updates after finishing all edges at time t
        for vv, (start0, arr0, hops0) in cand.items():
            if (start0 > best_start[vv]) or (start0 == best_start[vv] and arr0 < best_arr[vv]):
                best_start[vv] = start0
                best_arr[vv] = arr0
                best_hops[vv] = hops0

    if best_travel == INF:
        return None, None, None, None

    return int(best_travel), int(best_dep_t), int(best_arr_t), int(best_target_hops)


def top_k_nodes(counter_dict, k=3, exclude=None):
    """
    Return up to k node ids with highest counts, excluding any in `exclude`.
    Sort: degree desc, then node id asc.
    """
    exclude = set() if exclude is None else set(exclude)
    items = sorted(counter_dict.items(), key=lambda x: (-x[1], x[0]))
    out = []
    for node, deg in items:
        if node in exclude:
            continue
        out.append(node)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------
# Pass 1) Find min/max dep_time_ut and define first-2h window
# ------------------------------------------------------------
min_dep = None
max_dep = None
bad_rows_1 = 0

with INPUT.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter=DELIM)
    for row in reader:
        try:
            dep = int(row["dep_time_ut"])
        except (KeyError, ValueError, TypeError):
            bad_rows_1 += 1
            continue
        if min_dep is None or dep < min_dep:
            min_dep = dep
        if max_dep is None or dep > max_dep:
            max_dep = dep

if min_dep is None:
    raise RuntimeError("No valid dep_time_ut found in input file.")

cutoff_dep = min_dep + WINDOW_SECONDS

print("=== dep_time_ut window ===")
print(f"City: {CITY}")
print(f"Input: {INPUT}")
print(f"First dep_time_ut: {min_dep}")
print(f"Last  dep_time_ut: {max_dep}")
print(f"Keeping first 2 hours: dep_time_ut in [{min_dep}, {cutoff_dep}]")
print(f"Malformed rows (pass1): {bad_rows_1}")
print()

# ------------------------------------------------------------
# Pass 2) Collect unique dep_time_ut inside first-2h window
#         Build mapping dep_time_ut -> t (rank starting at 1)
# ------------------------------------------------------------
dep_times = set()
bad_rows_2 = 0

with INPUT.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter=DELIM)
    for row in reader:
        try:
            dep = int(row["dep_time_ut"])
        except (KeyError, ValueError, TypeError):
            bad_rows_2 += 1
            continue
        if min_dep <= dep <= cutoff_dep:
            dep_times.add(dep)

dep_times_sorted = sorted(dep_times)
dep_time_to_t = {d: i + 1 for i, d in enumerate(dep_times_sorted)}
T_max = len(dep_times_sorted)

print("=== timestamps (ranked) inside first 2 hours ===")
print(f"Max timestamp T = {T_max}")
print(f"Malformed rows (pass2): {bad_rows_2}")
print()

if T_max == 0:
    raise RuntimeError("No timestamps found inside the first 2 hours window.")

# Define windows in rank space
t10_end = max(1, math.ceil(0.10 * T_max))     # first 10% window: t in [1, t10_end]
t50_start = (T_max // 2) + 1                  # last 50% window: t in [t50_start, T_max]
half_cap = math.ceil(0.50 * T_max)            # kept (not used now, but leaving intact)

# ------------------------------------------------------------
# Pass 3) Build UVT edges for first 2 hours + collect degrees
# ------------------------------------------------------------
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

nodes_set = set()
skipped = 0

uvt_edges = []
outdeg_first10 = defaultdict(int)
indeg_last50 = defaultdict(int)

with INPUT.open("r", encoding="utf-8", newline="") as fin:
    reader = csv.DictReader(fin, delimiter=DELIM)
    for row in reader:
        try:
            u = int(row["from_stop_I"])
            v = int(row["to_stop_I"])
            dep = int(row["dep_time_ut"])
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue

        if not (min_dep <= dep <= cutoff_dep):
            continue

        t = dep_time_to_t.get(dep)
        if t is None:
            skipped += 1
            continue

        uvt_edges.append((u, v, t))
        nodes_set.add(u)
        nodes_set.add(v)

        if t <= t10_end:
            outdeg_first10[u] += 1
        if t >= t50_start:
            indeg_last50[v] += 1

total_nodes = len(nodes_set)
total_temporal_edges = len(uvt_edges)

print("=== UVT build report ===")
print(f"Total nodes: {total_nodes}")
print(f"Total temporal edges: {total_temporal_edges}")
print(f"Skipped malformed/missing-map rows (pass3): {skipped}")
print(f"Rank windows: first10% t<= {t10_end}, last50% t>= {t50_start}, T={T_max}")
print()

if total_temporal_edges == 0:
    raise RuntimeError("No edges were written for the first 2 hours window.")

uvt_edges_sorted = sorted(uvt_edges, key=lambda e: e[2])

# ------------------------------------------------------------
# Phase 1) Pick NUM_SOURCES sources
# ------------------------------------------------------------
sources = top_k_nodes(outdeg_first10, NUM_SOURCES)
if len(sources) < NUM_SOURCES:
    sources = sorted(nodes_set)[:NUM_SOURCES]

if len(sources) != NUM_SOURCES:
    raise RuntimeError(f"Could not pick {NUM_SOURCES} sources. sources={sources}")

# ------------------------------------------------------------
# Pick NUM_TARGETS targets that avoid direct-edge (travel_time=1)
# for ANY of the chosen sources.
# ------------------------------------------------------------
all_targets_ranked = sorted(
    [(n, indeg_last50.get(n, 0)) for n in nodes_set if n not in set(sources)],
    key=lambda x: (-x[1], x[0])
)
target_candidates = [n for (n, deg) in all_targets_ranked]

# cache: (s,z) -> (travel, dep, arr, hops)
cache = {}

def get_stats(s, z):
    key = (s, z)
    if key not in cache:
        cache[key] = shortest_travel_time_path_stats(uvt_edges_sorted, s, z)
    return cache[key]

targets = []
for z in target_candidates:
    ok = True
    for s in sources:
        travel, dep_t, arr_t, hops = get_stats(s, z)
        # direct edge implies travel_time==1 (dep==arr==some t)
        if travel == 1:
            ok = False
            break
    if ok:
        targets.append(z)
    if len(targets) == NUM_TARGETS:
        break

if len(targets) < NUM_TARGETS:
    raise RuntimeError(
        f"Could not find {NUM_TARGETS} targets that avoid travel_time=1 (direct-edge) for all sources.\n"
        f"Sources={sources}\n"
        f"Found targets={targets}\n"
        "Try increasing WINDOW_SECONDS or relaxing the rule, but your ILP can't cut 1-arc paths."
    )

print("=== Selected nodes ===")
print(f"Sources (top out-degree in first 10%): {sources}")
print(f"Targets (top in-degree in last 50%, excluding sources, avoiding travel_time=1): {targets}")
print()

# ============================================================
# deadline = 2 * (shortest travel time)
# combos = full cartesian product => NUM_SOURCES * NUM_TARGETS
# ============================================================
combos = []
for s in sources:
    for z in targets:
        if s == z:
            continue

        travel, dep_t, arr_t, hops = get_stats(s, z)

        if travel is None:
            deadline = T_max
            note = "NO_PATH -> deadline=T"
        else:
            deadline = 2 * travel
            if deadline > T_max:
                deadline = T_max
            note = f"travel={travel}, dep_t={dep_t}, arr_t={arr_t}, hops={hops}"

        combos.append((s, z, int(deadline), note))

expected = NUM_SOURCES * NUM_TARGETS
if len(combos) != expected:
    raise RuntimeError(
        f"Expected {expected} combos ({NUM_SOURCES}x{NUM_TARGETS}), got {len(combos)}. "
        f"sources={sources}, targets={targets}"
    )

print(f"=== {expected} combinations with deadlines ===")
for (s, z, d, note) in combos:
    print(f"s={s:>6}  z={z:>6}  deadline={d:>4}   ({note})")
print()

# ------------------------------------------------------------
# Write output file: header with combos + UVT edges
# ------------------------------------------------------------
with OUTPUT.open("w", encoding="utf-8", newline="\n") as fout:
    fout.write("# UVT temporal graph (first 2 hours)\n")
    fout.write(f"# city={CITY}\n")
    fout.write(f"# T_max={T_max}  total_nodes={total_nodes}  total_temporal_edges={total_temporal_edges}\n")
    fout.write(f"# first10_window_end={t10_end}  last50_window_start={t50_start}\n")
    fout.write("#\n")
    fout.write(f"# SOURCE-TARGET-DEADLINE ({expected} combos):\n")
    for i, (s, z, d, note) in enumerate(combos, start=1):
        fout.write(f"# combo {i}: source={s} target={z} deadline={d}  ({note})\n")
    fout.write("#\n")
    fout.write("# u v t\n")
    for (u, v, t) in uvt_edges_sorted:
        fout.write(f"{u} {v} {t}\n")

print("=== OUTPUT ===")
print(f"Output file: {OUTPUT}")
print(f"Total nodes: {total_nodes}")
print(f"Total temporal edges: {total_temporal_edges}")
print(f"Max timestamp T: {T_max}")
