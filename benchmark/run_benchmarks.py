"""Task 6 — runs the mandated 9-instance benchmark suite, cross-checks with
brute force (small instances) and exhaustive CSP search (infeasible
instances), writes results.csv/.md and two charts."""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import instance_from_generator
from src.scheduler import solve
from src.brute_force import brute_force_optimal
from benchmark.csp_check import list_coloring_feasible

HERE = os.path.dirname(os.path.abspath(__file__))

SUITE = [
    # (label, n, K, density, seed, group)
    ("small-1", 8, 3, 0.30, 1, "small"),
    ("small-2", 10, 4, 0.40, 2, "small"),
    ("small-3", 12, 4, 0.50, 3, "small"),
    ("medium-1", 50, 8, 0.25, 10, "medium"),
    ("medium-2", 100, 10, 0.30, 11, "medium"),
    ("medium-3", 150, 12, 0.35, 12, "medium"),
    ("stress-dense", 200, 15, 0.40, 20, "stress"),
    ("stress-tightK", 200, 5, 0.60, 21, "stress"),
    ("stress-sparse", 200, 20, 0.10, 22, "stress"),
]


def run():
    rows = []
    for label, n, K, density, seed, group in SUITE:
        inst = instance_from_generator(n, K, density, seed)
        t0 = time.time()
        result = solve(inst)
        wall_ms = int((time.time() - t0) * 1000)

        row = dict(label=label, n=n, K=K, density=density, seed=seed, group=group,
                   feasible=result.feasible, penalty=result.penalty if result.feasible else None,
                   runtime_ms=result.runtime_ms, wall_ms=wall_ms,
                   violation_reason=result.violation_reason, opt_penalty=None, ratio=None,
                   csp_verified=None, csp_calls=None)

        if group == "small":
            bf_assign, bf_pen, timed_out = brute_force_optimal(inst, time_budget_s=60)
            if bf_assign is not None and not timed_out:
                row["opt_penalty"] = bf_pen
                if result.feasible and bf_pen > 0:
                    row["ratio"] = result.penalty / bf_pen
                elif result.feasible and bf_pen == 0:
                    row["ratio"] = 1.0 if result.penalty == 0 else float("inf")

        if not result.feasible:
            csp_result, calls = list_coloring_feasible(inst.n, inst.windows, inst.neighbors, time_budget_s=90)
            row["csp_verified"] = csp_result  # False (exhaustive, proven infeasible) or 'timeout'
            row["csp_calls"] = calls

        rows.append(row)
        print(f"{label}: n={n} K={K} density={density} seed={seed} "
              f"feasible={result.feasible} penalty={row['penalty']} runtime_ms={result.runtime_ms} "
              f"ratio={row['ratio']} csp={row['csp_verified']}({row['csp_calls']})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    csv_path = os.path.join(HERE, "results", "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(HERE, "results", "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote {csv_path} and {json_path}")
    return rows


if __name__ == "__main__":
    run()
