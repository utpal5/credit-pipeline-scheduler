# Task 3 — Algorithm Design

## Name: **Window-DSATUR with Variance-Guided Tabu Repacking (WD-VTR)**

## 1. Why this problem structure demands a hybrid

This is not plain graph coloring and not plain bin-packing: it is **list-coloring**
(SLA window `[l_i, u_i]` restricts each vertex to a *subset* of colors, not all `K`) **+**
**multi-dimensional bin-packing** (F2) **+** a **custom two-term objective** (Task 2).
Three off-the-shelf algorithms each solve one piece and ignore the others:

- Plain DSATUR colors optimally-ish for *unconstrained* palettes but has no notion of a
  restricted color set per vertex, and no notion of capacity at all.
- Bin-packing heuristics (first-fit-decreasing, etc.) ignore conflicts entirely.
- LP relaxation + rounding handles the objective smoothly but rounding does not respect F1
  (two tasks in the same slot after independent rounding is common) or F2, so it still needs
  a repair pass — see rejected alternative #1 below.

WD-VTR is built as **construction (respects F1/F2/F3 by structural invariant, never
violates them) + repair (recovers from local dead-ends) + local search (spends remaining
time budget lowering `P(σ)` from Task 2)**, so feasibility is never traded away for
objective quality — this is what Task 4's feasibility proof leans on.

## 2. Phase 0 — Preprocessing (once, O(n·d + m))

```
for each task i:
    deg(i)     = |N(i)|                      # static conflict degree
    W(i)       = u_i - l_i + 1               # SLA window width
    urgency(i) = w(i) / W(i)                 # weight per available slot
```

`urgency(i)` is the key departure from stock DSATUR: a high-weight task in a wide window is
*not* urgent (it has room to be placed well later); a low-weight task jammed into a
width-1 window *is* urgent regardless of weight, because it has exactly one legal slot and
must be placed before that slot fills up. Pure `w(i)` or pure `deg(i)` tie-breaking (stock
DSATUR) misses this — it is the SLA-window-specific adaptation the assignment brief asks for.

## 3. Phase 1 — Construction: Window-DSATUR

```
UNASSIGNED = all tasks
σ = {}                                        # partial assignment
UNRESOLVED = []

while UNASSIGNED is not empty:
    # --- 3a. Pick the most constrained task (generalized DSATUR selection) ---
    for i in UNASSIGNED:
        satDeg(i) = | { σ(j) : j ∈ N(i), σ(j) assigned, l_i ≤ σ(j) ≤ u_i } |
        rem(i)    = | { s ∈ [l_i,u_i] : s conflict-feasible(i) AND capacity-feasible(i,s) } |
    i* = argmax over UNASSIGNED of:
             ( satDeg(i),  -rem(i),  urgency(i) )      # lexicographic compare
         # ties broken in this order: most saturated first (classic DSATUR rationale:
         # these vertices' options only shrink further if left for later); among equal
         # saturation, fewest remaining legal slots first (CSP minimum-remaining-values
         # heuristic — fail fast / place the hardest task while options still exist);
         # among equal remainder, highest urgency (protects tight-window high-weight
         # bureau/fraud tasks from being crowded out by flexible low-priority ones)

    # --- 3b. Choose the best feasible slot for i*, scored by the REAL objective ---
    candidates = { s ∈ [l_{i*}, u_{i*}] : conflict-feasible(i*,s) AND capacity-feasible(i*,s) }
    if candidates is empty:
        success = REPACK(i*, σ)                        # Phase 1b, see below
        if not success:
            UNRESOLVED.append(i*)
            UNASSIGNED.remove(i*)
            continue
        candidates = { s ∈ [l_{i*},u_{i*}] : feasible(i*,s) }   # recheck after repack

    s* = argmin_{s in candidates} [ w(i*)·s + λ·ΔP_imbalance(σ, i*, s) ]
         # greedy w.r.t. the ACTUAL Task-2 objective, not slot index alone — a tie between
         # two legal slots is broken by which one keeps utilization more even, not
         # arbitrarily

    σ(i*) = s*
    UNASSIGNED.remove(i*)

if UNRESOLVED is not empty:
    return INFEASIBLE(UNRESOLVED, reason=diagnose(UNRESOLVED, σ))
```

**Why recompute `satDeg`/`rem` every iteration instead of a lazy priority queue:** with
`n ≤ 200` a full O(n) rescan per placement is `O(n²)` worst case — for `n=200` that's 40,000
basic comparisons, negligible in wall-clock terms, and it guarantees `satDeg`/`rem` are never
stale (a lazy heap risks operating on outdated saturation counts after neighbors are placed,
which would silently degrade the heuristic quality with no correctness net). Simplicity and
provable freshness win over the marginal complexity saving here.

## 4. Phase 1b — Bounded Repacking (triggered only on construction dead-end)

Fires when task `i` has **no** capacity-feasible slot in its window, but at least one
*conflict*-feasible slot exists (i.e., the obstruction is F2, not F1 — if every slot in the
window is conflict-blocked, repacking cannot help and we go straight to `UNRESOLVED`).

```
REPACK(i, σ):
    for s in window(i) sorted by (least-over-capacity first):
        if s is only capacity-infeasible (not conflict-infeasible):
            evictable = { j : σ(j)=s, W(j) > 1, j ∉ tabu_evicted_recently }
            sort evictable by w(j) ascending           # evict cheapest-to-delay task first
            for j in evictable[:R]:                    # R = 3, bounds worst-case work
                for s' in window(j), s' ≠ s:
                    if feasible(j, s'):                 # standard F1+F2 check
                        move j: σ(j) = s'
                        tabu_evicted_recently.add(j, expires=τ)   # prevent immediate bounce-back
                        if capacity-feasible(i, s):
                            return True
                        # else undo and keep trying next candidate
                        σ(j) = s   # revert, try next eviction candidate
    return False
```

**Why lowest-weight, most-flexible occupant first:** evicting a high-weight or tight-window
task risks *cascading* infeasibility (that task then can't be replaced either). Evicting the
least important, most flexible occupant is the cheapest possible perturbation — this is the
same "minimum disruption" principle behind min-conflicts local search, applied at
displacement-selection time rather than after the fact.

**Why `R = O(1)` eviction attempts, not exhaustive:** an exhaustive repack could itself
cascade recursively across the whole instance, breaking the polynomial bound. Capping at a
constant `R` per blocked task keeps Phase 1b's total cost at
`O(n · R · K)` — polynomial, and empirically (Task 6) sufficient because SLA windows are
narrow (`≤ K`), so few slots need to be tried per blocked task.

## 5. Phase 2 — Local Search: Variance-Guided Hill-Climb with Tabu List

Runs after Phase 1 produces a (possibly partial) assignment, using the remaining time
budget to reduce `P(σ)` further — this is where the imbalance term actually gets optimized,
since Phase 1 only sees it as a tie-breaker.

```
tabu = {}                                     # (task, from_slot) -> expiry_iteration
for t in 1..T_max:                            # T_max = c · n  (c constant, e.g. 20)
    best_move = None; best_delta = 0
    for i in all placed tasks:
        for s in window(i), s ≠ σ(i):
            if (i, σ(i)) in tabu and tabu-not-expired: continue
            if not feasible(i, s): continue              # F1/F2/F3 re-checked, never relaxed
            δ = ΔP(σ, move i→s)                            # exact recompute of Task-2 objective
            if δ < best_delta: best_move = ("move", i, s); best_delta = δ

        for j in all placed tasks, slot(j) ≠ slot(i):     # pairwise swap candidates
            if swap(i,j) feasible for BOTH under F1/F2/F3:
                δ = ΔP(σ, swap i,j)
                if δ < best_delta: best_move = ("swap", i, j); best_delta = δ

    if best_move is None:
        break                                             # local optimum reached, stop early
    apply(best_move)
    tabu[reverse(best_move)] = t + τ                       # forbid immediately undoing it
return σ
```

**Why steepest-descent-with-tabu, not full simulated annealing:** see rejected
alternative #2 — SA's stochastic acceptance of worsening moves has no mechanism that
guarantees it stays inside the feasible region carved out by Phase 1, and its cooling
schedule needs per-instance retuning to behave consistently from `n=8` to `n=200`
(benchmark suite spans both). Steepest-descent only ever accepts moves with `δ < 0` *and*
that are independently re-verified feasible, so **every intermediate state during Phase 2 is
itself a valid solution** — the algorithm can be stopped at any iteration (e.g. on a time
budget) and still return something submittable. That property is worth more here than SA's
theoretical ability to escape deeper local optima, given the grading suite has no time
budget defined and predictability across 9 very differently-sized instances matters more
than squeezing out the last few points of `P(σ)`.

**Why the tabu list at all, given moves are already `δ<0`-only (can't cycle on value
alone):** a move and its exact reverse can still both show `δ<0` at different times because
`ΔP_imbalance` is state-dependent — moving `i` from `s` to `s'` can lower imbalance given the
*current* occupants of `s,s'`, and once other moves change those occupants, moving `i` back
can *also* show `δ<0` under the new state. Without a tabu list this can oscillate
indefinitely on adversarial occupancy patterns; forbidding the immediate reversal for `τ`
iterations breaks that cycle while still allowing the move back later if the state has
genuinely changed enough to justify it.

## 6. Phase 3 — Infeasibility diagnosis

```
diagnose(UNRESOLVED, σ):
    for i in UNRESOLVED:
        if all s in [l_i,u_i] are conflict-blocked by already-placed neighbors:
            reason = f"{i}: every slot in window [{l_i},{u_i}] conflict-blocked (F1) — " +
                     f"conflict degree {deg(i)} too high relative to window width {W(i)}"
        elif [l_i,u_i] ∩ [1,K] is empty or l_i > u_i:
            reason = f"{i}: SLA window [{l_i},{u_i}] invalid or outside slot range [1,{K}] (F3)"
        else:
            reason = f"{i}: no slot in window [{l_i},{u_i}] has residual capacity in some " +
                     f"dimension after repacking attempts (F2), even after evicting up to " +
                     f"{R} lower-priority occupants per slot"
    return (feasible=False, violation_reason=join(reasons))
```
This is a genuine attempt at certifying *why*, not just *that*, the instance failed —
required by the Task 5 output schema (`violation_reason`).

## 7. Overall complexity

| Phase | Cost | Bound |
|---|---|---|
| 0 Preprocess | O(n·d + m) | polynomial |
| 1 Construction | O(n) iterations × O(n) selection × O(K·(d + deg_avg)) feasibility = O(n²·K·(d+deg_avg)) | polynomial |
| 1b Repack | O(n) blocked tasks × O(R·K) attempts × O(n/K) eviction-candidate scan = O(R·n²) | polynomial |
| 2 Local search | O(T_max) × O(n·K·d) per iteration = O(c·n²·K·d) | polynomial |

All phases are polynomial in `n, K, d`; the algorithm runs in worst-case
`O(n²·K·d)` — well within budget for `n ≤ 200, K ≤ 20, d = 4`.

## 8. Two rejected alternatives

**Rejected #1 — LP relaxation + randomized rounding.**
Forbidden solvers (OR-Tools/PuLP/CPLEX/Gurobi) rule out using an off-the-shelf LP engine, so
this would mean implementing a simplex method from scratch just to reach a fractional
solution that *still* needs a rounding-repair pass to fix F1 violations (two tasks
independently rounded into the same slot) and F2 violations (rounded capacity overshoot) —
i.e., it needs the same repair machinery WD-VTR already has, but pays the extra cost and
implementation risk of a from-scratch LP solver first. For the tight-K stress instance in
the benchmark suite (`n=200, K=5`, density 0.6) the LP relaxation is expected to be far from
integral (very few feasible integral points exist at all), so the rounding step would fail
often and lean almost entirely on the repair pass anyway — the LP step would be doing
little useful work for its cost.

**Rejected #2 — Simulated Annealing from a random initial assignment.**
Starting SA from a uniformly random `σ` over a graph-coloring + bin-packing + list-coloring
feasible region is, for the tight instances in the benchmark suite, starting almost
certainly *outside* the feasible region (random slot assignment ignoring F1/F2/F3
essentially never satisfies all three simultaneously once `n` is large and `K` is small).
SA would then spend an unbounded, un-analyzable fraction of its iteration budget just
searching for feasibility, with no structural guarantee it ever gets there — which directly
conflicts with the Task 4 feasibility-guarantee requirement ("if a valid assignment exists,
your algorithm always finds one"). A cooling schedule tuned to reliably reach feasibility
for `n=8` would be far too slow to also finish in reasonable time at `n=200`, and tuning it
per-instance is not viable for a fixed graded benchmark suite. WD-VTR sidesteps this by
guaranteeing a feasible region is reached *structurally* (construction never places a task
outside F1/F2/F3) before any objective-driven search begins.
