# Credit Pipeline Scheduler — ScoreMe Solutions Capstone Assignment

Assignment of the MSME Credit Pipeline Scheduling problem: assign `n` credit-pipeline
tasks (bureau pulls, OCR, fraud scoring, ...) to `K` processing slots subject to
conflict (F1), multi-dimensional capacity (F2), and SLA-window (F3) constraints,
minimizing a custom penalty function. Proven NP-hard; solved with an original
polynomial-time heuristic, **WD-VTR** (Window-DSATUR with Variance-guided Tabu
Repacking).

## Deliverables

| Task | File |
|---|---|
| T1 — NP-hardness proof | [`docs/T1_np_hardness.tex`](docs/T1_np_hardness.tex) |
| T2 — Penalty function design | [`docs/T2_penalty_function.md`](docs/T2_penalty_function.md) |
| T3 — Algorithm design (WD-VTR) | [`docs/T3_algorithm_design.md`](docs/T3_algorithm_design.md) |
| T4 — Feasibility & approximation proof | [`docs/T4_approximation_proof.md`](docs/T4_approximation_proof.md) |
| T5 — Implementation | [`src/`](src/), [`run.py`](run.py), [`tests/`](tests/) |
| T6 — Benchmarking | [`docs/T6_benchmarking.md`](docs/T6_benchmarking.md), [`benchmark/`](benchmark/) |
| T7 — Design journal | [`docs/T7_design_journal.md`](docs/T7_design_journal.md) |

## Project layout

```
src/
  generator.py     # instance generator, provided by the assignment, unmodified
  models.py        # Instance data class (note: slots are 0-indexed, see module docstring)
  penalty.py        # P_base, P_imbalance, lambda, P(sigma)
  scheduler.py       # WD-VTR: preprocess / construct / repack / local_search / diagnose / solve
  brute_force.py     # exact DFS+pruning solver, small instances only (ground truth for T4/T6)
  io_utils.py        # JSON I/O matching the Task 5 schema
run.py                # CLI entry point
tests/
  test_scheduler.py   # required edge cases + a soundness sweep over generated instances
benchmark/
  run_benchmarks.py   # runs the 9 mandated instances from Section 6
  make_charts.py       # penalty vs n, runtime vs n charts
  csp_check.py          # independent exhaustive F1+F3 checker, used to verify infeasibility claims
  results/, charts/      # generated output
docs/                     # T1-T7 write-ups
```

## Requirements

Python 3.10+. Standard library plus `matplotlib` (used only for the Task 6 charts):

```
pip install matplotlib
```

No forbidden solvers (OR-Tools, PuLP, CPLEX, Gurobi, Z3, networkx.coloring, or any SAT
solver) are used anywhere in this codebase.

## Running it

**Generate an instance and solve it:**
```
python run.py --n 8 --K 3 --density 0.3 --seed 1
```

**Solve a saved instance JSON, write result to a file:**
```
python run.py --input instance.json --output result.json
```

Output JSON matches the Task 5 schema: `assignment`, `penalty`, `runtime_ms`,
`feasible`, `violation_reason`.

**Run the unit tests:**
```
python -m unittest tests.test_scheduler -v
```

**Run the full Task 6 benchmark suite and regenerate charts:**
```
python benchmark/run_benchmarks.py
python benchmark/make_charts.py
```

## Headline results (full analysis in `docs/T6_benchmarking.md`)

- Small instances (n=8, 10, 12): all feasible, 2 of 3 exactly optimal vs. brute force,
  1 at 1.206x optimal.
- Medium/stress instances (n=50-200, all 6): **genuinely infeasible**, independently
  verified with an exhaustive CSP search (`benchmark/csp_check.py`), not assumed from
  the heuristic's own output.
- No constant approximation ratio exists for WD-VTR in the worst case — proved
  constructively in `docs/T4_approximation_proof.md`, with both a synthetic
  unbounded family and a real instance from the mandated generator (2.151x optimal).
