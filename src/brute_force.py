"""Exact optimal solver — backtracking search with constraint pruning.

Used only for small instances (Task 6's n=8/10/12 comparisons and Task 4's
adversarial-example verification), never inside the graded algorithm itself
(WD-VTR must stand on its own polynomial-time merits). Not a SAT/ILP solver
and not one of the forbidden libraries — plain DFS with F1/F2/F3 pruning at
each partial assignment, which is exponential worst-case by design (that is
the whole point: it is the ground truth WD-VTR is measured against).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .models import Instance, Assignment
from .penalty import penalty
from .scheduler import conflict_feasible, capacity_feasible


def brute_force_optimal(instance: Instance, time_budget_s: Optional[float] = None):
    """Returns (best_assignment_by_task_id, best_penalty) or (None, None) if
    no feasible assignment exists. Search order processes the most tightly
    windowed tasks first (smallest branching factor first), which is the
    same MRV intuition as WD-VTR's construction phase — it prunes dead
    branches earlier without changing which leaf is optimal."""
    import time
    start = time.time()
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.windows[i][1] - instance.windows[i][0])

    best = {"assignment": None, "penalty": float("inf")}
    usage = [[0.0] * instance.d for _ in range(instance.K)]
    assignment: Assignment = {}

    def recurse(idx: int) -> bool:
        if time_budget_s is not None and time.time() - start > time_budget_s:
            return True  # signal: bail out, treat as timed out
        if idx == n:
            p = penalty(instance, assignment)
            if p < best["penalty"]:
                best["penalty"] = p
                best["assignment"] = dict(assignment)
            return False
        i = order[idx]
        l, u = instance.windows[i]
        for s in range(l, u + 1):
            if not conflict_feasible(instance, assignment, i, s):
                continue
            if not capacity_feasible(instance, usage, i, s):
                continue
            assignment[i] = s
            for k in range(instance.d):
                usage[s][k] += instance.resources[i][k]
            timed_out = recurse(idx + 1)
            assignment.pop(i)
            for k in range(instance.d):
                usage[s][k] -= instance.resources[i][k]
            if timed_out:
                return True
        return False

    timed_out = recurse(0)
    if best["assignment"] is None:
        return None, None, timed_out
    named = {instance.task_ids[i]: s for i, s in best["assignment"].items()}
    return named, best["penalty"], timed_out
