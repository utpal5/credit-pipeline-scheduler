# Task 2 — Penalty Function Design

## Chosen extension: Scarcity-Weighted Load-Imbalance Penalty

### Motivation (ScoreMe context)

`P_base` only prices *when* a task runs (`w_i · σ(i)`), not *how the cluster looks* once
everything is placed. In the real ScoreMe cluster this matters: two assignments can have
identical `P_base` while one packs Slot 1 to 95% GPU utilization and leaves Slot 3 at 10%,
and the other spreads load evenly. The first assignment is operationally worse because:

- **Retry/backfill risk.** NiFi pipelines commonly re-queue a failed OCR or fraud-scoring
  task into the *same* slot window. A saturated slot has no headroom to absorb that retry,
  forcing an SLA breach; an idle slot absorbs it for free.
- **Capital utilization.** GPU accelerators are the scarcest, most expensive resource in the
  cluster (only OCR and fraud/credit-scoring tasks use them at all). A GPU sitting idle in
  Slot 3 while Slot 1 queues GPU work is wasted capital *right now*, not a future risk.
- **Blast radius of a bad estimate.** `r(tᵢ)` is an estimate. A slot near 100% utilization
  has zero tolerance for underestimation; a balanced slot does.

So the extension prices **variance of utilization across slots, per resource dimension**,
weighted by how scarce/expensive that dimension is.

### Formal definition

For assignment `σ`, dimension `k ∈ {CPU, RAM, GPU, Net}`, slot `s ∈ [K]`, define the
**fractional utilization**:

```
u_k(s) = ( Σ_{i : σ(i)=s} r_k(i) ) / C_k(s)
```

(`u_k(s) := 0` if `C_k(s) = 0`; under a feasible `σ` this only occurs when no task with
`r_k(i) > 0` is assigned to `s`, since F2 forbids the alternative.)

Mean utilization of dimension `k` across all slots:

```
ū_k = (1/K) Σ_{s=1}^{K} u_k(s)
```

**Imbalance penalty:**

```
P_imbalance(σ) = Σ_{k=1}^{d} α_k · ( (1/K) Σ_{s=1}^{K} (u_k(s) − ū_k)² )
```

i.e. `α_k`-weighted sum, over resource dimensions, of the population variance of that
dimension's slot utilizations. `α_k` are fixed **scarcity weights** reflecting relative
resource cost, normalized so `Σ_k α_k = 1`:

```
α_GPU = 0.50   α_CPU = 0.25   α_RAM = 0.15   α_Net = 0.10
```

GPU dominates the weighting because it is the only dimension with hard physical scarcity
(fixed accelerator count) rather than elastic capacity — an idle GPU-slot is unrecoverable
waste for that 30-second window, whereas idle CPU/RAM/Net is cheap.

**Extended objective:**

```
P(σ) = P_base(σ) + λ · P_imbalance(σ),      λ = ( Σᵢ w(tᵢ) ) / K
```

`λ` rescales the imbalance term into the same order of magnitude as `P_base` (whose scale
grows with `Σwᵢ` and shrinks with more slots to spread delay over), so neither term
dominates by construction regardless of instance size — no manual per-instance tuning.

### Why this satisfies the requirements

- **Formally defined:** closed-form expression above, no ambiguity.
- **Polynomial time:** computing all `u_k(s)` is `O(n·d)`; computing all variances is
  `O(K·d)`. Total `O(nd + Kd)`, linear in `n` for fixed `d ≤ 4`.
- **Monotonically meaningful:** driving `P_imbalance → 0` drives every slot toward the same
  per-dimension utilization ratio, which is exactly "no slot at 95% while another sits at
  10%" — the concern named directly in the assignment brief. It is minimized (not
  maximized) because concentration, not spread, is the operational risk.
- **Non-trivial:** it is a genuine function of `σ` (changes whenever any task moves between
  slots with different residual capacity), not a constant, and it does not collapse to
  `P_base` — a `σ` that is optimal for `P_base` alone (pack everything into the earliest
  legal slots to minimize `Σ w_i σ(i)`) is typically *far* from optimal for
  `P_imbalance`, so the two terms genuinely trade off against each other. This tension is
  deliberate: it is what makes Task 3's local-search phase non-degenerate — a pure
  delay-minimizing greedy has no incentive to ever move a task once placed, whereas under
  `λ·P_imbalance` there is a real, computable reason to relocate a task even after a
  feasible placement is found.

### Considered and rejected: a separate "Resource Waste" term

An earlier draft of this design used two extra terms instead of one — the imbalance
term above (`L`) plus a separate `W = Σ_s Σ_d (C_{s,d} - Used_{s,d})`, "total idle
capacity across the cluster," on the reasoning that idle GPUs/CPUs are expensive and
should be penalized independently of whether utilization is *balanced*.

`W` as defined turns out to be **a constant, independent of σ** — not a design
weakness but a hard mathematical fact about this problem shape. Every task is
assigned to exactly one slot, so summing `Used_{s,d}` over *all* slots recovers the
total demand across all `n` tasks for dimension `d`, a quantity fixed by the instance:

```
Σ_s Used_{s,d} = Σ_s Σ_{i: σ(i)=s} r_d(i) = Σᵢ r_d(i)     (independent of σ)
```

so `W(σ) = (Σ_s Σ_d C_{s,d}) − (Σ_d Σᵢ r_d(i))` is the same number for *every*
feasible (or infeasible) `σ`. Concretely, with `K=2`, one dimension, `C₁=C₂=10`, and
two tasks each demanding 4 units: packing both into slot 1 gives waste
`(10−8)+(10−0)=12`; splitting one per slot gives `(10−4)+(10−4)=12` — identical,
despite one arrangement being maximally unbalanced and the other perfectly balanced.
`W` cannot distinguish them. This is exactly the failure mode the assignment brief
warns against ("adding a constant... is rejected") — just reached by an intuitive-
looking formula rather than a literal `+0`.

The general lesson, not specific to this one attempt: **any term that linearly sums
"waste" or "usage" across all slots is doomed the same way**, because total usage is
conserved under a partition (every task placed exactly once). Making a waste-style
term respond to `σ` at all requires a *nonlinear* aggregation across slots — variance,
max, or sum-of-squares. But once nonlinear, it stops being an independently
justifiable second concern: minimizing `Σ_s (C_{s,d}-Used_{s,d})²` for a fixed total
demand is minimized exactly when utilization is balanced across slots — the same
optimizer as `P_imbalance` above. So a corrected `W` collapses into a restatement of
the imbalance term rather than a genuinely distinct third idea, which is why the final
design uses one term, not two: `P_imbalance` already *is* the properly-normalized,
correctly-nonlinear version of "penalize idle/uneven capacity," per-dimension-weighted
by scarcity (`α_k`) rather than treated as interchangeable idle CPU-vs-GPU-vs-RAM
units, which a flat waste sum would otherwise conflate.

### Interaction with Task 3

The imbalance term is not decorative — Task 3's algorithm uses `ΔP_imbalance(s)` directly as
the tie-breaking cost when the construction phase picks a slot, and as the objective the
local-search repair phase hill-climbs on. This keeps Task 2 and Task 3 coupled: the
algorithm is provably working to minimize *this specific* `P(σ)`, not a generic proxy.
