"""Exhaustive F1+F3 (conflict + window) list-coloring feasibility checker.

Diagnostic tool only — not part of the graded WD-VTR pipeline. Used to tell
apart "the instance really is infeasible" from "WD-VTR gave a false
negative" (Task 4 proves the latter is possible in principle since
Feasibility is NP-complete; this checker is how Task 6 verifies which case
each benchmark instance actually falls into). Plain backtracking + forward
checking + MRV ordering — a from-scratch CSP search, not a SAT/ILP solver.
"""
from __future__ import annotations
import time


def list_coloring_feasible(n, windows, neighbors, time_budget_s=60):
    """Returns (result, calls) where result is True / False / 'timeout'.
    False means an EXHAUSTIVE search proved no assignment satisfies F1+F3 —
    not a heuristic failure to find one."""
    domains = [set(range(l, u + 1)) for l, u in windows]
    assign = {}
    start = time.time()
    calls = [0]

    def pick_mrv():
        unassigned = [i for i in range(n) if i not in assign]
        return min(unassigned, key=lambda i: len(domains[i]))

    def backtrack():
        calls[0] += 1
        if time.time() - start > time_budget_s:
            return "timeout"
        if len(assign) == n:
            return True
        i = pick_mrv()
        if not domains[i]:
            return False
        for s in sorted(domains[i]):
            removed = []
            ok = True
            for j in neighbors[i]:
                if j not in assign and s in domains[j]:
                    domains[j].discard(s)
                    removed.append(j)
                    if not domains[j]:
                        ok = False
            if ok:
                assign[i] = s
                result = backtrack()
                if result == "timeout":
                    for j in removed:
                        domains[j].add(s)
                    del assign[i]
                    return "timeout"
                if result is True:
                    return True
                del assign[i]
            for j in removed:
                domains[j].add(s)
        return False

    result = backtrack()
    return result, calls[0]
