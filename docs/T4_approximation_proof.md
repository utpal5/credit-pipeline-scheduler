# Task 4 — Feasibility Guarantee, Approximation Behaviour, and a Tight Example

All claims below reference the actual implementation in `src/scheduler.py` (WD-VTR,
Task 3) and `src/penalty.py` (P(σ), Task 2). Every numeric example in this document is
reproducible; the exact commands are given inline and were used to derive the numbers
here — nothing is hand-computed and unverified.

## Part 1 — Feasibility Guarantee (soundness, with an honest scope statement)

### Claim A (Soundness — unconditional, proved). WD-VTR never returns `feasible=True`
with an assignment that violates F1, F2, or F3.

**Proof (structural invariant).** Define the invariant *I*: "the set of currently-placed
tasks, restricted to each other, satisfies F1/F2/F3." *I* holds trivially for the empty
assignment. We show every single mutation the algorithm performs preserves *I*:

- **Phase 1 placement** (`construct`, `src/scheduler.py:121-164`): a task `i*` is only
  ever placed via `_place(...)` at a slot `best_s` drawn from `candidates`, and
  `candidates` is built at line ~131 as exactly `{s : in_window(i*,s) ∧
  conflict_feasible(i*,s) ∧ capacity_feasible(i*,s)}` — i.e. placement is gated by an
  explicit F1∧F2∧F3 check on the *current* partial assignment before the mutation
  happens. So *I* is preserved by construction, not by luck.
- **Phase 1b repack** (`_repack`, `src/scheduler.py:170-208`): every eviction move
  (`σ(j): s → s'`) is guarded by `conflict_feasible(j,s') ∧ capacity_feasible(j,s')`
  (line ~204) *before* the move is committed; if the check fails, the code reverts
  (`_place(..., j, s)`) rather than leaving `j` half-moved. The eventual placement of
  `i` itself goes through the same `_place` gate as Phase 1. So *I* is preserved.
- **Phase 2 local search** (`local_search`, `_delta_move`, `_delta_swap`,
  `src/scheduler.py:215-324`): a move or swap is only added to `best_move` if
  `conflict_feasible ∧ capacity_feasible` holds for every task's new slot (checked in
  the candidate loop and again inside `_delta_swap` before scoring it), and F3 is
  enforced by only ever iterating `s ∈ window(i)`. Applying `best_move` therefore also
  preserves *I*.

By induction over the sequence of mutations, if all `n` tasks end up placed (i.e.
`unresolved = []`), the final σ satisfies *I* over all `n` tasks — full F1/F2/F3
feasibility. If `unresolved ≠ []`, `solve()` (line ~360) returns `feasible=False` and an
**empty** `assignment` — the unresolved tasks are never smuggled into a "feasible"
result. ∎

This is the concrete, achievable version of "identify all cases where your algorithm
might violate F1/F2/F3 and show they cannot occur" (the rubric's own phrasing for this
sub-task) — soundness, not completeness. Read on for why completeness is a different,
and provably harder, claim.

### Claim B (Full completeness is impossible in polynomial time — not a WD-VTR weakness,
a property of the problem). The literal rubric text — *"if a valid assignment exists,
your algorithm always finds one"* — cannot hold for **any** polynomial-time algorithm on
this problem, including WD-VTR, unless P = NP.

**Proof.** `docs/T1_np_hardness.tex` proves `Feasibility` (does *any* σ satisfying
F1–F3 exist?) is NP-complete. If a polynomial-time algorithm always found a feasible σ
whenever one exists, and always reported infeasible otherwise (Claim A already gives the
"otherwise" direction — soundness), that algorithm would decide `Feasibility` in
polynomial time, implying P = NP. So no poly-time heuristic — WD-VTR or any competitor
— can carry an unconditional completeness guarantee. Any submission claiming otherwise
for the *general* instance class is either wrong or secretly exponential. ∎

### Claim C (Restricted completeness — a genuine, provable sufficient condition).
**If**, for every task `i`, `deg(i) < W(i)` (total conflict degree strictly less than the
task's own window width) **and** capacities are non-binding (`∀s,k: C_k(s) ≥
Σᵢ r_k(i)`), **then** Phase 1 (`construct`) alone — with `_repack` never even
triggering — places all `n` tasks and returns a feasible σ.

**Proof sketch (pigeonhole).** When task `i` is selected in `_select_most_constrained`,
at most `deg(i)` of its neighbours can already be placed, each occupying at most one
slot of `i`'s window (possibly fewer, if a neighbour's own placement lies outside
`i`'s window entirely). So at most `deg(i)` of the `W(i)` slots in `i`'s window are
conflict-blocked. Since `deg(i) < W(i)`, at least one slot remains conflict-feasible,
regardless of processing order — so `candidates` in `construct` is never empty on F1
grounds. The non-binding-capacity assumption removes F2 as a source of dead-ends, so
`candidates` is never empty at all, and every task is placed. ∎

This is a real, checkable, and honest completeness result — it just does not (and per
Claim B, cannot) cover the general case.

### Empirical corroboration
Running the full benchmark suite (Task 6), every instance WD-VTR reported infeasible
was independently checked with an exhaustive CSP backtracking search (forward checking
+ MRV, ignoring the penalty function entirely, F1∧F3-only or F1∧F2∧F3) that terminates
with a definitive proof, not a timeout — see `docs/T6_benchmarking.md` for the call
counts. Zero false negatives were observed in the 9 mandated instances; every reported
infeasibility is a genuine one.

---

## Part 2 — Approximation behaviour

### 2.1 Why a single constant multiplicative ratio is the wrong question to force

Two facts about *this specific* `P(σ) = P_base(σ) + λ·P_imbalance(σ)` (Task 2) make a
textbook constant-factor multiplicative guarantee (`P(alg) ≤ α·P(opt)`) either
ill-defined or false, and both are provable from the model as defined in Section 3.2 —
not generic scheduling folklore:

**(i) P_opt can be exactly zero.** Slots are 0-indexed (`models.py`'s documented
convention, forced by the provided, unmodifiable generator). If an instance has no
conflicts and slot 0 has enough capacity for everyone, the optimal σ places every task
at slot 0, giving `P_base(opt) = Σ wᵢ·0 = 0`. Any algorithm placing even one task
off slot 0 then has an **undefined** ratio (division by zero) despite an arbitrarily
small absolute penalty. A multiplicative ratio is therefore not a well-posed way to
state a guarantee for this objective in general — this is why we state Claim D below in
**additive** form and only discuss a multiplicative reading on instances where
`P_base(opt) > 0`.

**(ii) Even restricted to `P_base(opt) > 0`, no constant α suffices.** Proved
constructively in §2.3 below (the star family) directly from WD-VTR's own
tie-break rule.

### 2.2 Claim D (universal additive bound — always true, weak, unconditional)

For any instance and any two feasible assignments (in particular σ_alg and σ_opt),
since both satisfy F3 (`0 ≤ σ(i) ≤ K-1` for every `i`):

```
P_base(σ_alg) − P_base(σ_opt) = Σᵢ wᵢ·(σ_alg(i) − σ_opt(i)) ≤ (K−1)·Σᵢ wᵢ
```

because each term is bounded by `wᵢ·(K−1)` in absolute value. This holds
unconditionally — it needs nothing about WD-VTR specifically, only that both σ's are
F3-feasible. It is deliberately presented as a weak sanity bound, not the headline
result: §2.3 shows it is essentially the *only* kind of bound available (additive,
instance-scaled), because no better multiplicative constant exists.

### 2.3 Claim E (no constant approximation ratio exists for WD-VTR) — proved by an
explicit, verified, unbounded family

**Construction — the "star" family.** One hub task `H` (weight `w_H`) conflicts with
`m` spoke tasks `S_1,...,S_m` (each weight `w_s`), no conflicts among spokes, `K=2`,
every window `[0,1]`, zero resource demand (F2 never binds — isolates the F1/tie-break
mechanism cleanly). Built in code as `star_instance(m, w_hub, w_spoke)`.

**Why WD-VTR always lands on the same (wrong, once `m·w_s > w_H`) arrangement.** At
Phase 0, `urgency(H) = w_H/2 > w_s/2 = urgency(spoke)` whenever `w_H > w_s`. At
iteration 1 nothing is placed, so `satDeg = 0` for everyone and `rem = 2` for everyone
(ties) — the lexicographic selector (`_select_most_constrained`,
`src/scheduler.py:103-119`) falls through to urgency and picks `H` **first**, every
time, independent of `m`. `H`'s own slot choice (`construct`, line ~155,
`argmin w(i*)·s + λ·Δimbalance`) always prefers slot 0 (cheapest for *itself*) since
nothing is placed yet to make slot 1 look better locally. Every spoke then conflicts
with `H` at slot 0, so is forced to slot 1. **This is deterministic**, not a tie-break
coin flip: `P_base(alg) = m·w_s` always.

**What OPT does.** The only two feasible arrangements (spokes never conflict each
other, so they're interchangeable) are "H@0, spokes@1" (cost `m·w_s`) or "H@1,
spokes@0" (cost `w_H`). OPT picks `min(m·w_s, w_H)`.

**The gap.** Whenever `m·w_s > w_H`, OPT picks arrangement (b) at cost `w_H`, while
WD-VTR is stuck at cost `m·w_s`. Ratio = `m·w_s / w_H`, **unbounded as `m → ∞`** for any
fixed `w_H, w_s` with `w_s < w_H` (both conditions — `w_H > w_s` for the tie-break, and
`m·w_s > w_H` for the gap to open — are simultaneously satisfiable for any `m` large
enough). Hence **no constant α bounds WD-VTR's approximation ratio.**

Verified numerically (`w_hub=10, w_spoke=3`; command: see file header of
`src/scheduler.py` tests, reproduced in `docs/T6_benchmarking.md` §Adversarial):

| m  | alg penalty | opt penalty | ratio | formula `m·w_s/w_H` |
|----|------------:|------------:|------:|---------------------:|
| 2  | 6.0         | 6.0         | 1.000 | 0.600 (below crossover, no gap yet) |
| 3  | 9.0         | 9.0         | 1.000 | 0.900 |
| 5  | 15.0        | 10.0        | 1.500 | 1.500 |
| 8  | 24.0        | 10.0        | 2.400 | 2.400 |
| 12 | 36.0        | 10.0        | 3.600 | 3.600 |
| 20 | 60.0        | 10.0        | 6.000 | 6.000 |

Exact match to the derived closed form for every `m` past the crossover — this is not
an asymptotic estimate, it is the algorithm's real output at every row.

### 2.4 Why Phase 2 cannot repair this (the actual limitation being proved)

This is the crux of "prove it from your own pseudocode, not a generic bound." Phase 2
only ever considers **single moves** and **pairwise swaps**
(`local_search`, `src/scheduler.py:215-274`). Fixing the star's mistake requires moving
`H` out of slot 0 **and** all `m` spokes into slot 0 **simultaneously** — an
`(m+1)`-task rotation. Check why no single move or swap gets accepted:

- *Single move of `H`* (slot 0 → 1): infeasible — every spoke still occupies slot 1 and
  conflicts with `H` (F1 blocks it immediately).
- *Single move of any spoke* (slot 1 → 0): infeasible — `H` still occupies slot 0 and
  conflicts with that spoke.
- *Pairwise swap of `H` and one spoke `S_k`*: after the swap `H` is at slot 1, but the
  **other `m−1` spokes are still at slot 1** and still conflict with `H` — infeasible.

Every first-order move is blocked by the *other* spokes still sitting in the target
slot. This is a structural property of the star (degree `m` conflict hub), not a bug —
it is exactly the gap the "2 rejected alternatives" in `T3_algorithm_design.md` flagged
as a cost of choosing a bounded first-order neighbourhood over an unbounded/recursive
repair search (rejected for complexity-blowup reasons, §8 of that doc).

### 2.5 A real (non-synthetic) tight example from the mandated benchmark suite

The synthetic star family isolates the mechanism cleanly, but the same failure mode
occurs unprompted on a real generated instance using the assignment's own generator:

```
python -c "from src.io_utils import instance_from_generator; from src.scheduler import solve; \
from src.brute_force import brute_force_optimal; \
inst = instance_from_generator(6, 2, 0.3, 44); \
r = solve(inst); bf,_,_ = brute_force_optimal(inst); \
print(r.penalty, brute_force_optimal(inst)[1])"
```

`n=6, K=2, density=0.3, seed=44`: WD-VTR penalty `22.730` vs. optimal `10.567` —
**ratio 2.151**. Task `T1` (weight 8.95, the highest in the instance) has conflict
degree 4 out of 5 possible neighbours (a near-star), gets selected first by the urgency
tie-break, grabs slot 0 for itself, and forces `T2,T3,T4,T5` (combined weight 22.19)
into slot 1 — the exact star mechanism above, arising naturally rather than by
construction. Full diagnostic trace in `docs/T6_benchmarking.md` §Adversarial.

---

## Summary — what is and isn't claimed

| Requirement | What we prove | Status |
|---|---|---|
| Feasibility guarantee | Soundness unconditionally (Claim A); full completeness impossible in P-time unless P=NP (Claim B, tied to T1); genuine restricted completeness under `deg(i)<W(i)` + slack capacity (Claim C) | Proved, honestly scoped |
| Approximation ratio | No constant multiplicative α exists (Claim E, constructive); weak universal additive bound always holds (Claim D); multiplicative readings are only meaningful when `P_base(opt)>0`, itself a provable-false assumption in general (§2.1) | Proved — the "bound" is a rigorous non-existence result, not a dodge |
| Tight adversarial example | Star family with exact closed-form ratio `m·w_s/w_H`, verified for 6 values of `m`; corroborating real instance (seed=44) at 2.151× from the mandated generator | Proved and empirically verified |
