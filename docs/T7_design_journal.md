# Task 7 — Design Journal

## The hardest design decision: what to do when Phase 2 provably can't fix Phase 1's mistake

The single decision I keep coming back to is in `local_search` (`src/scheduler.py`,
Phase 2): the neighbourhood is limited to single moves and pairwise swaps. I knew from
the start this was a simplification — the "2 rejected alternatives" section of
`T3_algorithm_design.md` already flags that a full/recursive repair search would blow
up the polynomial bound. But I didn't understand *how bad* the gap it leaves behind
could be until I built the star-conflict adversarial example for Task 4: one high-weight
"hub" task, greedily grabbing the cheap slot for itself because it wins the urgency
tie-break, forces every one of its conflicting neighbours into the expensive slot — even
when the neighbours' combined weight dwarfs the hub's. Fixing it needs the hub and *all*
its neighbours to move at once. A pairwise-swap neighbourhood can never execute that,
because every intermediate single-swap state is still infeasible (the other neighbours
are still sitting in the slot the hub needs).

The trade-off I actually faced: widen the neighbourhood (e.g. allow 3-cycles or small
group rotations) and risk blowing the `O(n^2 K d)` bound Task 4 needs for a clean
feasibility/runtime story, or keep the bound clean and accept that WD-VTR has *no*
constant approximation ratio in the worst case. I chose to keep the bound clean and
instead *prove* the ratio is unbounded (the star family, verified for m up to 20),
rather than quietly ship a wider neighbourhood I couldn't fully analyze. In hindsight I
think that was the right call for this assignment specifically — Task 4 rewards a
proof that's honestly derived from my own pseudocode over one that's hand-waved — but
it means the algorithm's practical quality on high-degree, near-star conflict
structures is genuinely weaker than on the mostly-uniform random graphs the generator
produces, and I'd flag that as a real limitation to a reviewer, not just a proof
artifact.

## Where it failed empirically, and what I'd change with a week

Two distinct failure modes showed up when I actually ran the mandated benchmark suite,
and they surprised me in different ways.

**`small-2` (n=10, K=4, density=0.4, seed=2):** WD-VTR landed 20.6% above the
brute-force optimum. I traced it task-by-task (`docs/T4_approximation_proof.md` §2.5)
and it's the same star mechanism, just spread across six of the ten tasks instead of
a single clean hub — T9 (weight 9.42) and T7 (weight 4.36) don't even conflict with
each other directly, but both get pushed off their individually-preferred slots by a
cascade of greedy decisions on T1/T2/T3. If I had another week, I'd prototype a
bounded look-ahead in the construction phase — before committing task `i*` to its
locally-cheapest slot, check whether that choice forecloses a cheaper aggregate
placement for `i*`'s unplaced neighbours, and only commit if the local gain outweighs
the projected neighbour cost. That's a real algorithmic change, not a parameter tweak,
and I didn't have confidence I could get its complexity bound right before the
deadline, so I left it as a documented limitation instead of shipping something I
couldn't prove.

**The medium/stress tier (`medium-1` through `stress-sparse`): all six infeasible.**
This one genuinely surprised me — I expected maybe the tight-K stress case
(`n=200, K=5, density=0.6`) to be infeasible by design (that's clearly what the "tight
K" label is testing for), but not all six, including the sparse one
(`n=200, K=20, density=0.1`). My first reaction was that I'd broken something in
Phase 1. I spent a while re-checking `_select_most_constrained` and `_repack` for an
off-by-one before I thought to check the *instance* rather than the *algorithm*: I
relaxed capacities to effectively infinite and reran, and it was still infeasible on
every one of the six, which meant the obstruction was pure F1+F3 (conflict + window),
not F2. Then I wrote a from-scratch exhaustive CSP checker
(`benchmark/csp_check.py` — forward checking with MRV, deliberately not using any of
the forbidden solvers) to get a ground-truth answer instead of trusting my own
heuristic to grade itself, and it proved all six genuinely infeasible, in as few as 3
backtracking calls for some of them. That was the moment I actually understood, rather
than just having proved on paper, why Task 1 insists F3 isn't redundant with F1: for
`medium-1` specifically, the *unrestricted* conflict graph has a valid 7-coloring well
within K=8, so plain graph coloring would call it easy — it's only once you add the
per-task SLA window restriction that it becomes infeasible. I had written that
sentence in the T1 proof's "necessity of F3" paragraph as an abstract claim; watching
it actually happen on a random generated instance is a different kind of
understanding. With another week I'd want to characterize *how* narrow the generator's
windows need to be, as a function of K and density, before this kind of list-coloring
infeasibility becomes near-certain — right now I only know it happens at these
particular seeds, not the general threshold.

## Where this shows up at ScoreMe

The clearest match is the **OCR GPU cluster feeding the bureau-pull pipeline**. GPU
memory bus contention is exactly the F1 conflict relation in this model (two OCR jobs
that would fight over the same GPU memory bus can't share a processing slot), GPU unit
count is one dimension of F2's capacity vector, and the SLA window is the real
constraint a bureau pull is under — RBI/bureau API contracts and downstream
underwriting SLAs impose a hard "must complete within N cycles of submission" window,
which is precisely `[l_i, u_i]`. The finding that matters most for that system,
concretely: if the real job mix ever has a small number of high-priority,
high-conflict-degree jobs (e.g. a burst of large-file OCR jobs that all contend for
the same GPU memory bus around a batch cutoff), WD-VTR's construction phase will
systematically front-load the *individually* most urgent job and shove everything it
conflicts with into the next cycle — even when that's collectively worse for total
weighted delay. Given the imbalance penalty (Task 2) already prices "one slot
overloaded while another idles," I'd specifically watch GPU-scarce, high-fan-in
conflict windows (many jobs contending for the same bus around a submission deadline)
as the place this algorithm is most likely to under-perform in production, not the
average case the random benchmark suite mostly represents.

## What surprised me

Going in, I expected the hard part of Task 4 to be the algebra of deriving a
approximation constant. It turned out the harder and more useful realization was that
a *constant* multiplicative ratio isn't even a well-posed guarantee for this objective
in general — `P_base(opt)` can be exactly zero (everyone fits at slot 0), which makes
"how many times worse than optimal" undefined by division, not just hard to bound. I
didn't anticipate that going in; I only found it because I sat down to *use* the
formula on a trivial instance and hit a division by zero, not because I reasoned my
way to it abstractly. That changed how I think about "prove an approximation ratio" as
a task: the interesting content wasn't picking a number, it was figuring out what
question is actually well-posed to ask about this specific objective, given how I
chose to index slots and define the penalty. The other thing that surprised me is how
fast the exhaustive CSP checker proved infeasibility on the stress instances (3–12
backtracking calls for five of the six) — I expected proving infeasibility to be the
expensive direction, and instead it turned out these particular random instances have
a tiny, almost immediately-discoverable "unsat core," while the sparsest instance
(`stress-sparse`) needed nearly 2,000 calls despite having the *lowest* conflict
density — density alone was a bad predictor of search difficulty; window width turned
out to matter more.
