"""Task 3 — Window-DSATUR with Variance-Guided Tabu Repacking (WD-VTR).

Direct implementation of docs/T3_algorithm_design.md. Function names and
phase numbering mirror the pseudocode 1:1 (Phase 0 / 1 / 1b / 2 / 3) so the
code can be read side-by-side with the design doc during the viva.

Every constraint check (conflict_feasible / capacity_feasible / window
membership) is centralized in the three helpers below and re-run, never
cached-and-trusted, at every point a task is placed or moved — that
repetition is what Task 4's soundness proof relies on (see
docs/T4_approximation_proof.md, Claim A): every mutation to `assignment` is
individually gated by a feasibility check, so F1/F2/F3 can never be silently
violated by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import Instance, Assignment
from .penalty import penalty, delta_imbalance, lam

EPS = 1e-9


@dataclass
class SchedulerResult:
    assignment: Dict[str, int]     # task_id -> slot (0-indexed), only if feasible
    penalty: float
    runtime_ms: int
    feasible: bool
    violation_reason: str = ""


# --------------------------------------------------------------------------
# Phase 0 — preprocessing
# --------------------------------------------------------------------------

def preprocess(instance: Instance) -> Tuple[List[int], List[int], List[float]]:
    """deg(i), W(i) = window width, urgency(i) = w(i)/W(i).

    urgency is the SLA-window-specific departure from stock DSATUR: a
    high-weight task in a wide window is not urgent (room to place it well
    later); a low-weight task jammed into a width-1 window is urgent
    regardless of weight, because it has exactly one legal slot. See T3 §2.
    """
    deg = [len(instance.neighbors[i]) for i in range(instance.n)]
    W = [u - l + 1 for (l, u) in instance.windows]
    urgency = [instance.weights[i] / W[i] for i in range(instance.n)]
    return deg, W, urgency


# --------------------------------------------------------------------------
# Feasibility primitives — the single source of truth for F1/F2/F3
# --------------------------------------------------------------------------

def conflict_feasible(instance: Instance, assignment: Assignment, i: int, s: int) -> bool:
    """F1: no already-placed neighbor of i occupies slot s."""
    for j in instance.neighbors[i]:
        if assignment.get(j) == s:
            return False
    return True


def capacity_feasible(instance: Instance, usage: List[List[float]], i: int, s: int) -> bool:
    """F2: adding task i's demand to slot s does not exceed any dimension's
    capacity. Uses a running usage[s][k] table rather than recomputing from
    the full assignment dict, so this stays O(d) per call instead of O(n)."""
    for k in range(instance.d):
        if usage[s][k] + instance.resources[i][k] > instance.capacities[s][k] + EPS:
            return False
    return True


def in_window(instance: Instance, i: int, s: int) -> bool:
    """F3: s within [l_i, u_i]."""
    l, u = instance.windows[i]
    return l <= s <= u


def feasible_slot(instance: Instance, assignment: Assignment, usage, i: int, s: int) -> bool:
    return in_window(instance, i, s) and conflict_feasible(instance, assignment, i, s) \
        and capacity_feasible(instance, usage, i, s)


def _place(assignment: Assignment, usage, instance: Instance, i: int, s: int) -> None:
    assignment[i] = s
    for k in range(instance.d):
        usage[s][k] += instance.resources[i][k]


def _unplace(assignment: Assignment, usage, instance: Instance, i: int) -> None:
    s = assignment.pop(i)
    for k in range(instance.d):
        usage[s][k] -= instance.resources[i][k]


# --------------------------------------------------------------------------
# Phase 1 — Construction: Window-DSATUR
# --------------------------------------------------------------------------

def _select_most_constrained(instance: Instance, unassigned, assignment, usage, urgency) -> int:
    """Lexicographic argmax over (satDeg, -rem, urgency). See T3 §3 for why
    this order (saturation first, then MRV, then urgency) rather than any
    single criterion alone."""
    best_i, best_key = None, None
    for i in unassigned:
        l, u = instance.windows[i]
        sat_deg = len({assignment[j] for j in instance.neighbors[i]
                       if j in assignment and l <= assignment[j] <= u})
        rem = sum(1 for s in range(l, u + 1)
                  if conflict_feasible(instance, assignment, i, s)
                  and capacity_feasible(instance, usage, i, s))
        key = (sat_deg, -rem, urgency[i])
        if best_key is None or key > best_key:
            best_key, best_i = key, i
    return best_i


def construct(instance: Instance, R: int = 3, tau: int = 5):
    """Phase 1 + 1b combined driver loop. Returns (assignment, usage, unresolved)."""
    n = instance.n
    _, W, urgency = preprocess(instance)
    assignment: Assignment = {}
    usage = [[0.0] * instance.d for _ in range(instance.K)]
    unassigned = set(range(n))
    unresolved: List[int] = []
    tabu_evicted: Dict[int, int] = {}   # task -> iteration it becomes evictable again
    it = 0

    while unassigned:
        it += 1
        i_star = _select_most_constrained(instance, unassigned, assignment, usage, urgency)
        l, u = instance.windows[i_star]
        candidates = [s for s in range(l, u + 1)
                      if conflict_feasible(instance, assignment, i_star, s)
                      and capacity_feasible(instance, usage, i_star, s)]

        if not candidates:
            success = _repack(instance, i_star, assignment, usage, W, tabu_evicted, it, tau, R)
            if not success:
                unresolved.append(i_star)
                unassigned.discard(i_star)
                continue
            candidates = [s for s in range(l, u + 1)
                          if conflict_feasible(instance, assignment, i_star, s)
                          and capacity_feasible(instance, usage, i_star, s)]
            if not candidates:
                unresolved.append(i_star)
                unassigned.discard(i_star)
                continue

        best_s, best_cost = None, None
        for s in candidates:
            cost = instance.weights[i_star] * s + lam(instance) * delta_imbalance(instance, usage, i_star, s)
            if best_cost is None or cost < best_cost:
                best_cost, best_s = cost, s

        _place(assignment, usage, instance, i_star, best_s)
        unassigned.discard(i_star)

    return assignment, usage, unresolved


# --------------------------------------------------------------------------
# Phase 1b — Bounded repacking
# --------------------------------------------------------------------------

def _repack(instance, i, assignment, usage, W, tabu_evicted, it, tau, R) -> bool:
    """Fires only when i has no capacity-feasible slot but at least one
    conflict-feasible slot in its window (if every slot is conflict-blocked,
    repacking cannot help — F1 obstruction, not F2). Evicts the
    cheapest-to-delay (lowest-weight, flexible) occupant first: minimum
    disruption principle, see T3 §4."""
    l, u = instance.windows[i]
    blocked_slots = []
    for s in range(l, u + 1):
        if not conflict_feasible(instance, assignment, i, s):
            continue
        if capacity_feasible(instance, usage, i, s):
            continue
        over = sum(max(0.0, usage[s][k] + instance.resources[i][k] - instance.capacities[s][k])
                    for k in range(instance.d))
        blocked_slots.append((over, s))
    blocked_slots.sort(key=lambda x: x[0])

    for _, s in blocked_slots:
        evictable = [j for j, sl in assignment.items()
                     if sl == s and W[j] > 1 and tabu_evicted.get(j, 0) <= it]
        evictable.sort(key=lambda j: instance.weights[j])
        for j in evictable[:R]:
            lj, uj = instance.windows[j]
            for sp in range(lj, uj + 1):
                if sp == s:
                    continue
                _unplace(assignment, usage, instance, j)
                if conflict_feasible(instance, assignment, j, sp) and capacity_feasible(instance, usage, j, sp):
                    _place(assignment, usage, instance, j, sp)
                    tabu_evicted[j] = it + tau
                    if capacity_feasible(instance, usage, i, s):
                        return True
                    # didn't free enough room for i: undo and try the next candidate
                    _unplace(assignment, usage, instance, j)
                    _place(assignment, usage, instance, j, s)
                else:
                    _place(assignment, usage, instance, j, s)  # revert, sp was infeasible
    return False


# --------------------------------------------------------------------------
# Phase 2 — Local search: variance-guided hill-climb with tabu list
# --------------------------------------------------------------------------

def local_search(instance: Instance, assignment: Assignment, usage, c: int = 5, tau: int = 5,
                  time_budget_s: Optional[float] = None):
    """Steepest-descent hill-climb over single moves and pairwise swaps,
    T_max = c*n iterations, tabu on immediate reversal. See T3 §5 for why
    steepest-descent (not SA) and why tabu is needed even with delta<0-only
    acceptance (P_imbalance is state-dependent, so a move and its exact
    reverse can each show delta<0 at different times).

    c defaults to 5 rather than the design doc's illustrative 20: at n=200
    the full O(n^2) neighborhood scan per iteration makes c=20 too slow for
    a benchmark run in pure Python (see docs/T6 anomaly notes) without any
    change to solution quality guarantees, since every intermediate state
    is already independently feasible and the loop still terminates early
    on a true local optimum.
    """
    n = instance.n
    T_max = c * n
    tabu: Dict[Tuple[int, int], int] = {}   # (task, from_slot) -> expiry iteration
    start = time.time()

    for t in range(1, T_max + 1):
        if time_budget_s is not None and time.time() - start > time_budget_s:
            break
        best_move = None
        best_delta = 0.0

        placed = list(assignment.items())
        for i, si in placed:
            l, u = instance.windows[i]
            if tabu.get((i, si), 0) <= t:
                for s in range(l, u + 1):
                    if s == si:
                        continue
                    if not (conflict_feasible(instance, assignment, i, s)
                            and capacity_feasible(instance, usage, i, s)):
                        continue
                    delta = _delta_move(instance, assignment, usage, i, s)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("move", i, si, s)

            for j, sj in placed:
                if sj == si or j <= i:
                    continue
                if tabu.get((i, si), 0) > t or tabu.get((j, sj), 0) > t:
                    continue
                if not in_window(instance, i, sj) or not in_window(instance, j, si):
                    continue
                delta = _delta_swap(instance, assignment, usage, i, si, j, sj)
                if delta is None or delta >= best_delta:
                    continue
                best_delta = delta
                best_move = ("swap", i, si, j, sj)

        if best_move is None:
            break
        _apply_move(instance, assignment, usage, best_move, tabu, t, tau)

    return assignment


def _delta_move(instance, assignment, usage, i, s) -> float:
    si = assignment[i]
    before = penalty(instance, assignment)
    _unplace(assignment, usage, instance, i)
    _place(assignment, usage, instance, i, s)
    after = penalty(instance, assignment)
    _unplace(assignment, usage, instance, i)
    _place(assignment, usage, instance, i, si)
    return after - before


def _delta_swap(instance, assignment, usage, i, si, j, sj) -> Optional[float]:
    _unplace(assignment, usage, instance, i)
    _unplace(assignment, usage, instance, j)
    if not (conflict_feasible(instance, assignment, i, sj) and capacity_feasible(instance, usage, i, sj)):
        _place(assignment, usage, instance, i, si)
        _place(assignment, usage, instance, j, sj)
        return None
    _place(assignment, usage, instance, i, sj)
    if not (conflict_feasible(instance, assignment, j, si) and capacity_feasible(instance, usage, j, si)):
        _unplace(assignment, usage, instance, i)
        _place(assignment, usage, instance, i, si)
        _place(assignment, usage, instance, j, sj)
        return None
    before_usage_i, before_usage_j = si, sj
    _place(assignment, usage, instance, j, si)
    after = penalty(instance, assignment)
    _unplace(assignment, usage, instance, i)
    _unplace(assignment, usage, instance, j)
    _place(assignment, usage, instance, i, si)
    _place(assignment, usage, instance, j, sj)
    before = penalty(instance, assignment)
    return after - before


def _apply_move(instance, assignment, usage, move, tabu, t, tau):
    if move[0] == "move":
        _, i, si, s = move
        _unplace(assignment, usage, instance, i)
        _place(assignment, usage, instance, i, s)
        tabu[(i, s)] = t + tau
    else:
        _, i, si, j, sj = move
        _unplace(assignment, usage, instance, i)
        _unplace(assignment, usage, instance, j)
        _place(assignment, usage, instance, i, sj)
        _place(assignment, usage, instance, j, si)
        tabu[(i, sj)] = t + tau
        tabu[(j, si)] = t + tau


# --------------------------------------------------------------------------
# Phase 3 — Infeasibility diagnosis
# --------------------------------------------------------------------------

def diagnose(instance: Instance, unresolved: List[int], assignment: Assignment) -> str:
    reasons = []
    for i in unresolved:
        l, u = instance.windows[i]
        tid = instance.task_ids[i]
        if l > u or u < 0 or l >= instance.K:
            reasons.append(f"{tid}: SLA window [{l},{u}] invalid or outside slot range "
                            f"[0,{instance.K - 1}] (F3)")
            continue
        all_conflict_blocked = all(
            not conflict_feasible(instance, assignment, i, s) for s in range(l, u + 1)
        )
        if all_conflict_blocked:
            reasons.append(f"{tid}: every slot in window [{l},{u}] conflict-blocked (F1) - "
                            f"conflict degree {len(instance.neighbors[i])} too high relative to "
                            f"window width {u - l + 1}")
        else:
            reasons.append(f"{tid}: no slot in window [{l},{u}] has residual capacity in some "
                            f"dimension after repacking attempts (F2)")
    return "; ".join(reasons)


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def solve(instance: Instance, local_search_c: int = 5, time_budget_s: Optional[float] = None) -> SchedulerResult:
    start = time.time()
    assignment, usage, unresolved = construct(instance)

    if unresolved:
        reason = diagnose(instance, unresolved, assignment)
        runtime_ms = int((time.time() - start) * 1000)
        return SchedulerResult(assignment={}, penalty=float("inf"), runtime_ms=runtime_ms,
                                feasible=False, violation_reason=reason)

    assignment = local_search(instance, assignment, usage, c=local_search_c, time_budget_s=time_budget_s)
    p = penalty(instance, assignment)
    runtime_ms = int((time.time() - start) * 1000)
    named = {instance.task_ids[i]: s for i, s in assignment.items()}
    return SchedulerResult(assignment=named, penalty=p, runtime_ms=runtime_ms, feasible=True)
