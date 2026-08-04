"""Core data structures for the credit-pipeline scheduling problem.

SLOT INDEXING — deliberate design decision, not an oversight:
The assignment brief's toy example (Section 3.3) writes SLA windows 1-indexed
(e.g. T1:[1,3]). But the *provided, unmodifiable* generator (generator.py)
produces windows via `random.randint(0, K-2)` / `random.randint(lo+1, K-1)`,
which is 0-indexed over {0, ..., K-1}. Since the generator is authoritative
for every graded instance (including held-out seeds), this codebase treats
slots as 0-indexed throughout: slot values live in {0, ..., K-1}, and
P_base(sigma) = sum(w_i * sigma(i)) is computed on those 0-indexed values.
This is consistent internally; it only shifts the additive constant of
P_base relative to a 1-indexed reading of the toy example, which does not
change which assignment is optimal (arg min is shift-invariant except the
implicit "slot 0 is free" vs "slot 1 is free" pricing — see docs/VIVA_PREP.md
for the one-line justification to give live if asked).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Instance:
    n: int
    K: int
    d: int
    task_ids: List[str]
    conflicts: List[Tuple[int, int]]       # 0-indexed task-index pairs
    resources: List[List[float]]           # resources[i][k]
    capacities: List[List[float]]          # capacities[s][k], s in [0, K)
    windows: List[Tuple[int, int]]         # (lo, hi) inclusive, slots in [0, K)
    weights: List[float]
    neighbors: List[Set[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.neighbors:
            self.neighbors = [set() for _ in range(self.n)]
            for i, j in self.conflicts:
                self.neighbors[i].add(j)
                self.neighbors[j].add(i)

    @staticmethod
    def from_dict(data: dict) -> "Instance":
        """Build an Instance from the dict shape produced by generate_instance()
        (also the required JSON input shape for Task 5)."""
        n = len(data["tasks"])
        K = data["K"]
        resources = data["resources"]
        d = len(resources[0]) if resources else 4
        conflicts = [tuple(pair) for pair in data["conflicts"]]
        windows = [tuple(w) for w in data["windows"]]
        return Instance(
            n=n, K=K, d=d,
            task_ids=list(data["tasks"]),
            conflicts=conflicts,
            resources=[list(r) for r in resources],
            capacities=[list(c) for c in data["capacities"]],
            windows=windows,
            weights=list(data["weights"]),
        )


# Assignment: task index -> slot index (0-indexed). A task absent from the
# dict means "unassigned / could not be placed" (only possible when the
# scheduler reports feasible=False).
Assignment = Dict[int, int]
