"""Task 2 — penalty function: P(sigma) = P_base(sigma) + lambda * P_imbalance(sigma).

Implements docs/T2_penalty_function.md exactly. Kept separate from the
scheduler so both the algorithm and the brute-force checker score solutions
through the identical, single source of truth (avoids the classic bug class
where the heuristic optimizes one objective and the report evaluates
another).
"""
from __future__ import annotations

from typing import Dict, List

from .models import Instance, Assignment

# Scarcity weights alpha_k, indexed to match resources[i] = [CPU, RAM, GPU, Net].
# GPU dominates because it is the only dimension with hard physical scarcity
# (fixed accelerator count) in the ScoreMe cluster — see T2 doc for the full
# justification. Sum to 1 by construction.
ALPHA = [0.25, 0.15, 0.50, 0.10]  # CPU, RAM, GPU, Net


def p_base(instance: Instance, assignment: Assignment) -> float:
    """Weighted-delay term: sum_i w_i * sigma(i). Only counts placed tasks —
    an unresolved/unplaced task contributes to infeasibility reporting, not
    to a penalty value (there is no well-defined P(sigma) for a partial
    assignment)."""
    return sum(instance.weights[i] * s for i, s in assignment.items())


def slot_utilization(instance: Instance, assignment: Assignment) -> List[List[float]]:
    """u_k(s) for every slot s and dimension k, per the T2 definition.
    Returns u[s][k]. u_k(s) := 0 when C_k(s) == 0, matching the T2 doc's
    convention (only reachable under a feasible sigma when no task with
    r_k(i) > 0 is assigned to s, since F2 forbids the alternative)."""
    K, d = instance.K, instance.d
    usage = [[0.0] * d for _ in range(K)]
    for i, s in assignment.items():
        for k in range(d):
            usage[s][k] += instance.resources[i][k]
    u = [[0.0] * d for _ in range(K)]
    for s in range(K):
        for k in range(d):
            cap = instance.capacities[s][k]
            u[s][k] = usage[s][k] / cap if cap > 0 else 0.0
    return u


def p_imbalance(instance: Instance, assignment: Assignment) -> float:
    """Scarcity-weighted population-variance-of-utilization penalty (T2 doc,
    section 'Formal definition'). O(K*d)."""
    K, d = instance.K, instance.d
    u = slot_utilization(instance, assignment)
    total = 0.0
    for k in range(d):
        col = [u[s][k] for s in range(K)]
        mean = sum(col) / K
        var = sum((x - mean) ** 2 for x in col) / K
        total += ALPHA[k] * var
    return total


def lam(instance: Instance) -> float:
    """lambda = (sum_i w_i) / K — rescales P_imbalance into P_base's order of
    magnitude regardless of instance size, per the T2 doc, so neither term
    dominates by construction."""
    return sum(instance.weights) / instance.K


def penalty(instance: Instance, assignment: Assignment) -> float:
    """P(sigma) = P_base(sigma) + lambda * P_imbalance(sigma)."""
    return p_base(instance, assignment) + lam(instance) * p_imbalance(instance, assignment)


def delta_imbalance(instance: Instance, usage: List[List[float]], i: int, s: int) -> float:
    """P_imbalance(sigma + {i->s}) - P_imbalance(sigma), computed from a running
    usage[s][k] table (NOT from an assignment dict) so the construction phase
    can score every candidate slot for task i in O(K*d) without rebuilding
    the whole utilization table from scratch each time. Used as Phase 1's
    tie-break and Phase 2's exact move/swap delta."""
    K, d = instance.K, instance.d

    def imbalance_of(usage_table):
        total = 0.0
        for k in range(d):
            col = []
            for slot in range(K):
                cap = instance.capacities[slot][k]
                col.append(usage_table[slot][k] / cap if cap > 0 else 0.0)
            mean = sum(col) / K
            total += ALPHA[k] * (sum((x - mean) ** 2 for x in col) / K)
        return total

    before = imbalance_of(usage)
    usage[s] = [usage[s][k] + instance.resources[i][k] for k in range(d)]
    after = imbalance_of(usage)
    usage[s] = [usage[s][k] - instance.resources[i][k] for k in range(d)]
    return after - before
