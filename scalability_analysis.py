#!/usr/bin/env python3
"""
Scalability analysis (ILP only): quantifies how ILP solve time relates to
graph size (vertices, temporal arcs), the number of feasible temporal
paths within the deadline
"""

import os
import re
import glob
import csv
from typing import Dict, Any, List

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# EDIT HERE ONLY
# -----------------------------
PYTHON_ACS_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/python_ACS"
DATA_NEW_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/data_new"

SYNTHETIC_ILP_DIR = os.path.join(PYTHON_ACS_DIR, "synthetic_graphs", "results")

REAL_CITIES = ["berlin", "grenoble", "helsinki", "luxembourg", "venice"]
REAL_ILP_SUBDIR = "results_network_temporal_day_uvt_first2h"

SCALABILITY_OUT_DIR = os.path.join(PYTHON_ACS_DIR, "scalability_analysis")
# -----------------------------

# -----------------------------
# Result-file field parser (same regex style as the other batch runners
# in this project)
# -----------------------------
_FIELD_RES = {
    "city": re.compile(r"^\s*City:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE),
    "combo": re.compile(r"^\s*Combo:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "deadline": re.compile(r"^\s*deadline\(travel_time\):\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "T": re.compile(r"^\s*max_timestamp\(T\):\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "vertices": re.compile(r"^\s*total_nodes:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "temporal_edges": re.compile(r"^\s*total_temporal_edges:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "status": re.compile(r"^\s*status:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE),
    "num_paths": re.compile(r"^\s*num_paths:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    "solve_time": re.compile(r"^\s*solve_wall_time_seconds:\s*([\d.]+)\s*$", re.IGNORECASE | re.MULTILINE),
}


def parse_result_file(path: str) -> Dict[str, Any]:
    text = open(path, "r", encoding="utf-8").read()
    out: Dict[str, Any] = {}
    for key, rgx in _FIELD_RES.items():
        m = rgx.search(text)
        if not m:
            out[key] = None
            continue
        val = m.group(1)
        if key in ("deadline", "T", "vertices", "temporal_edges", "combo", "num_paths"):
            out[key] = int(val)
        elif key == "solve_time":
            out[key] = float(val)
        else:
            out[key] = val
    return out


def load_synthetic_instances() -> List[Dict[str, Any]]:
    rows = []
    ilp_files = [
        f for f in glob.glob(os.path.join(SYNTHETIC_ILP_DIR, "*.txt"))
        if not f.endswith("_summary.txt")
    ]
    for ilp_path in sorted(ilp_files):
        ilp = parse_result_file(ilp_path)
        if ilp.get("city") is None or ilp.get("solve_time") is None:
            continue
        rows.append({
            "dataset": ilp["city"],
            "category": "synthetic",
            "vertices": ilp.get("vertices"),
            "temporal_edges": ilp.get("temporal_edges"),
            "T": ilp.get("T"),
            "d": ilp.get("deadline"),
            "num_paths": ilp.get("num_paths") or 0,
            "ilp_status": ilp.get("status"),
            "ilp_time": ilp.get("solve_time"),
        })
    return rows


def load_real_instances() -> List[Dict[str, Any]]:
    rows = []
    for city in REAL_CITIES:
        ilp_dir = os.path.join(DATA_NEW_DIR, city, REAL_ILP_SUBDIR)
        ilp_files = [
            f for f in glob.glob(os.path.join(ilp_dir, "combo_*.txt"))
            if os.path.basename(f) != "summary.txt"
        ]
        for ilp_path in sorted(ilp_files):
            ilp = parse_result_file(ilp_path)
            combo = ilp.get("combo")
            if combo is None or ilp.get("solve_time") is None:
                continue
            rows.append({
                "dataset": f"{city}_c{combo}",
                "category": "real",
                "city": city,
                "vertices": ilp.get("vertices"),
                "temporal_edges": ilp.get("temporal_edges"),
                "T": ilp.get("T"),
                "d": ilp.get("deadline"),
                "num_paths": ilp.get("num_paths") or 0,
                "ilp_status": ilp.get("status"),
                "ilp_time": ilp.get("solve_time"),
            })
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = stats.pearsonr(x, y)
    return {"r": r, "p": p}


def main():
    os.makedirs(SCALABILITY_OUT_DIR, exist_ok=True)

    rows = load_synthetic_instances() + load_real_instances()
    print(f"[INFO] Loaded {len(rows)} ILP instances "
          f"({sum(1 for r in rows if r['category']=='synthetic')} synthetic, "
          f"{sum(1 for r in rows if r['category']=='real')} real)")

    # ---- merged CSV ----
    csv_path = os.path.join(SCALABILITY_OUT_DIR, "scalability_data.csv")
    fieldnames = ["dataset", "category", "city", "vertices", "temporal_edges", "T", "d",
                  "num_paths", "ilp_status", "ilp_time"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"[OUT] {csv_path}")

    log_time = np.log10(np.array([max(r["ilp_time"], 1e-4) for r in rows]))
    log_paths = np.log10(np.array([r["num_paths"] for r in rows]) + 1)
    log_vertices = np.log10(np.array([r["vertices"] for r in rows], dtype=float))
    log_edges = np.log10(np.array([r["temporal_edges"] for r in rows], dtype=float))
    d_vals = np.array([r["d"] for r in rows], dtype=float)
    T_vals = np.array([r["T"] for r in rows], dtype=float)

    # (predictor values, label for output/plot, x-axis label, log-scale?)
    predictors = [
        (log_paths, "num_paths (log10)", "Number of temporal paths within deadline, #P (+1)", True, np.array([r["num_paths"] for r in rows]) + 1),
        (log_vertices, "vertices (log10)", "Vertices", True, np.array([r["vertices"] for r in rows])),
        (log_edges, "temporal_edges (log10)", "Temporal arcs", True, np.array([r["temporal_edges"] for r in rows])),
        (d_vals, "deadline d", "Deadline d", False, d_vals),
        (T_vals, "horizon T", "Timestamp horizon T", False, T_vals),
    ]

    lines = [f"N = {len(rows)} ILP instances "
             f"({sum(1 for r in rows if r['category']=='synthetic')} synthetic, "
             f"{sum(1 for r in rows if r['category']=='real')} real)", ""]
    lines.append("ILP solve time vs each predictor (Pearson r on log10(time); "
                  "log10(predictor) for #paths/vertices/edges, raw scale for d/T):")

    results = []
    for x, label, _, _, _ in predictors:
        res = pearson(x, log_time)
        results.append(res)
        lines.append(f"  {label:24s}: r={res['r']:+.3f}  p={res['p']:.2e}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    txt_path = os.path.join(SCALABILITY_OUT_DIR, "scalability_correlations.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n[OUT] {txt_path}")

    # ---- figure: one panel per predictor ----
    plt.rcParams.update({"font.size": 10, "font.family": "serif",
                          "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    categories = np.array([r["category"] for r in rows])
    ilp_time_vals = np.array([max(r["ilp_time"], 1e-4) for r in rows])

    for ax, (x, label, xlabel, logscale, x_plot), res in zip(axes, predictors, results):
        for cat, marker, color in [("synthetic", "o", "#1f77b4"), ("real", "^", "#d62728")]:
            mask = categories == cat
            ax.scatter(x_plot[mask], ilp_time_vals[mask], marker=marker, color=color,
                       s=40, alpha=0.85, edgecolor="black", linewidth=0.4, label=cat)
        if logscale:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ILP solve time (s)")
        ax.set_title(f"r={res['r']:.2f}")

    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    pdf_path = os.path.join(SCALABILITY_OUT_DIR, "scalability_figure.pdf")
    png_path = os.path.join(SCALABILITY_OUT_DIR, "scalability_figure.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"[OUT] {pdf_path}")
    print(f"[OUT] {png_path}")


if __name__ == "__main__":
    main()
