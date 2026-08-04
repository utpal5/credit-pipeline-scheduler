"""Unit tests required by Task 5: all-conflict graph (chromatic number > K),
zero-capacity slot, tight SLA windows, single-task instance — plus a
soundness sweep over generated instances (Task 4's Claim A, checked
empirically here and proven structurally in docs/T4_approximation_proof.md).
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Instance
from src.scheduler import solve, conflict_feasible, capacity_feasible, in_window
from src.io_utils import instance_from_generator
from src.penalty import penalty


def make_instance(n, K, d, conflicts, resources, capacities, windows, weights):
    return Instance(
        n=n, K=K, d=d,
        task_ids=[f"T{i}" for i in range(n)],
        conflicts=conflicts,
        resources=resources,
        capacities=capacities,
        windows=windows,
        weights=weights,
    )


class TestSingleTask(unittest.TestCase):
    def test_single_task_trivially_feasible(self):
        inst = make_instance(
            n=1, K=3, d=4,
            conflicts=[],
            resources=[[1, 1, 1, 1]],
            capacities=[[10, 10, 10, 10]] * 3,
            windows=[(0, 2)],
            weights=[5.0],
        )
        result = solve(inst)
        self.assertTrue(result.feasible)
        self.assertEqual(set(result.assignment.keys()), {"T0"})
        self.assertIn(result.assignment["T0"], range(3))
        # optimal placement for a single task with no competition is the
        # earliest legal slot (minimizes w*s with nothing else to trade off)
        self.assertEqual(result.assignment["T0"], 0)


class TestAllConflictExceedsK(unittest.TestCase):
    def test_clique_larger_than_K_is_infeasible(self):
        # K4 clique (chromatic number 4) squeezed into K=3 slots -> must be
        # reported infeasible, never a silently-broken F1 assignment.
        n, K = 4, 3
        conflicts = [(i, j) for i in range(n) for j in range(i + 1, n)]
        inst = make_instance(
            n=n, K=K, d=4,
            conflicts=conflicts,
            resources=[[1, 1, 1, 1]] * n,
            capacities=[[10, 10, 10, 10]] * K,
            windows=[(0, K - 1)] * n,
            weights=[1.0] * n,
        )
        result = solve(inst)
        self.assertFalse(result.feasible)
        self.assertIn("conflict-blocked", result.violation_reason)
        self.assertEqual(result.assignment, {})


class TestZeroCapacitySlot(unittest.TestCase):
    def test_zero_capacity_slot_never_overfilled(self):
        # Slot 0 has zero GPU capacity; any task needing GPU must avoid it
        # even when slot 0 is otherwise attractive (earliest / cheapest).
        n, K = 3, 2
        capacities = [[10, 10, 0, 10], [10, 10, 10, 10]]
        resources = [[1, 1, 2, 1]] * n  # every task needs 2 GPU units
        inst = make_instance(
            n=n, K=K, d=4,
            conflicts=[],
            resources=resources,
            capacities=capacities,
            windows=[(0, 1)] * n,
            weights=[1.0] * n,
        )
        result = solve(inst)
        self.assertTrue(result.feasible)
        for tid, s in result.assignment.items():
            self.assertNotEqual(s, 0, f"{tid} illegally placed in the zero-GPU-capacity slot")

    def test_zero_capacity_forces_infeasible_when_only_legal_slot(self):
        # Task needs GPU but its ONLY legal (window-restricted) slot has 0
        # GPU capacity -> must be reported infeasible (F2), not silently
        # dropped or wrongly placed.
        inst = make_instance(
            n=1, K=2, d=4,
            conflicts=[],
            resources=[[1, 1, 3, 1]],
            capacities=[[10, 10, 0, 10], [10, 10, 10, 10]],
            windows=[(0, 0)],  # only slot 0 legal, and slot 0 has 0 GPU capacity
            weights=[1.0],
        )
        result = solve(inst)
        self.assertFalse(result.feasible)
        self.assertNotEqual(result.violation_reason, "")


class TestTightSLAWindows(unittest.TestCase):
    def test_width_one_windows_are_forced(self):
        # Three non-conflicting tasks each pinned to a distinct width-1
        # window -> must land exactly there, no choice for the algorithm.
        inst = make_instance(
            n=3, K=3, d=4,
            conflicts=[],
            resources=[[1, 1, 1, 1]] * 3,
            capacities=[[10, 10, 10, 10]] * 3,
            windows=[(0, 0), (1, 1), (2, 2)],
            weights=[1.0, 1.0, 1.0],
        )
        result = solve(inst)
        self.assertTrue(result.feasible)
        self.assertEqual(result.assignment["T0"], 0)
        self.assertEqual(result.assignment["T1"], 1)
        self.assertEqual(result.assignment["T2"], 2)

    def test_conflicting_width_one_windows_same_slot_infeasible(self):
        # Two tasks conflict but both are window-pinned to the SAME slot ->
        # no valid assignment can exist; must be reported, not fudged.
        inst = make_instance(
            n=2, K=3, d=4,
            conflicts=[(0, 1)],
            resources=[[1, 1, 1, 1]] * 2,
            capacities=[[10, 10, 10, 10]] * 3,
            windows=[(1, 1), (1, 1)],
            weights=[1.0, 1.0],
        )
        result = solve(inst)
        self.assertFalse(result.feasible)


class TestSoundnessSweep(unittest.TestCase):
    """Empirical companion to T4's Claim A: for a spread of generated
    instances, whatever the algorithm returns as 'feasible' must actually
    satisfy F1/F2/F3 for every task, and vice versa for reported UNRESOLVED
    tasks not silently appearing in the assignment."""

    def test_soundness_over_generated_instances(self):
        cases = [
            (8, 3, 0.3, 1), (10, 4, 0.4, 2), (12, 4, 0.5, 3),
            (20, 5, 0.2, 7), (30, 6, 0.5, 9),
        ]
        for n, K, density, seed in cases:
            inst = instance_from_generator(n, K, density, seed)
            result = solve(inst)
            with self.subTest(n=n, K=K, seed=seed):
                if not result.feasible:
                    self.assertEqual(result.assignment, {})
                    continue
                by_id = {tid: s for tid, s in result.assignment.items()}
                self.assertEqual(len(by_id), n, "every task must be placed when feasible=True")
                idx_of = {tid: i for i, tid in enumerate(inst.task_ids)}
                # F3
                for tid, s in by_id.items():
                    i = idx_of[tid]
                    l, u = inst.windows[i]
                    self.assertTrue(l <= s <= u, f"{tid} slot {s} outside window [{l},{u}]")
                # F1
                for a, b in inst.conflicts:
                    ta, tb = inst.task_ids[a], inst.task_ids[b]
                    self.assertNotEqual(by_id[ta], by_id[tb], f"conflict {ta}-{tb} share a slot")
                # F2
                usage = [[0.0] * inst.d for _ in range(inst.K)]
                for tid, s in by_id.items():
                    i = idx_of[tid]
                    for k in range(inst.d):
                        usage[s][k] += inst.resources[i][k]
                for s in range(inst.K):
                    for k in range(inst.d):
                        self.assertLessEqual(usage[s][k], inst.capacities[s][k] + 1e-6,
                                              f"slot {s} dim {k} over capacity")


if __name__ == "__main__":
    unittest.main()
