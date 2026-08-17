# TimeSep (d-MinIntSep) a Code for "Testing Robustness of Temporal Transportation Networks via Interval Separators"

This folder contains the implementation used to produce the experimental results
reported in the paper.

## Files

| File | Paper section | What it does |
|---|---|---|
| `pure_ilp_temporal_separator.py` | Core ILP formulation: builds and solves the temporal-path-separation ILP with Gurobi (objective, path-separation constraints, contiguity constraints, iterative constraint generation via `find_temporal_paths`). Imported by all the scripts below. |
| `solve_synthetic_optimized_d_travel.py` |  ILP on a synthetic instance
| `solve_real_optimized_d_travel.py` | ILP on a real-world (GTFS-derived) instance; 
| `run_greedy_comparison_synthetic.py` | greedy heuristic (Algorithm 1) on a synthetic instance
| `run_greedy_comparison_real.py` | greedy heuristic for real-world instances. 
| `scalability_analysis.py` | Scalability Analysis) 
| `Phase 1_2_lastupdate.py` | we transformed each static network into a temporal graph via two phases" | Generates a synthetic temporal graph from a static transportation network (Phase 1: shortest-path extraction + timestamp assignment; Phase 2: sparse temporal background). |
| `Convert_PT_new_d_travel.py` | restricted... to the first two hours...Converts a day-long GTFS CSV export into the first-two-hours temporal network used in the experiments, and generates the `(s,z,d)` combos (source/target selected by out-/in-degree, deadline = 2x shortest travel time). 

## Requirements

- Python 3.13 (as used for the reported running times)
- [Gurobi](https://www.gurobi.com) + a valid license, and the `gurobipy` package
- `numpy`, `scipy`, `matplotlib`, `psutil`

```bash
pip install gurobipy numpy scipy matplotlib psutil
```

## Usage

Each script has an "EDIT HERE ONLY" block near the top with the input file path(s)
and parameters (city, deadline, which combos to run, etc.) â edit that block and run
the script directly, e.g.:

```bash
python3 solve_synthetic_optimized_d_travel.py
python3 run_greedy_comparison_synthetic.py
python3 scalability_analysis.py
```

`pure_ilp_temporal_separator.py` must stay in the same directory as the `solve_*`
and `run_greedy_comparison_*` scripts (they import it directly).

## Data

`data/` contains the exact problem instances solved 

| Folder | Contents | Size |
|---|---|---|
| `data/synthetic/` | The 7 synthetic temporal graphs  (`synthetic_temporal_graph_*.txt`, one per city) |
| `data/real/<city>/` | The first-two-hours temporal network + `(s,z,d)` combos for each of the 5 real cities |

These are small because they are already the two-hour-restricted,
degree-filtered instances actually solved â not the raw source data.


