"""JSON I/O matching the Task 5 schema."""
from __future__ import annotations

import json

from .generator import generate_instance
from .models import Instance
from .scheduler import SchedulerResult


def instance_from_generator(n, K, density, seed, d=4) -> Instance:
    return Instance.from_dict(generate_instance(n, K, d=d, conflict_density=density, seed=seed))


def load_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        return Instance.from_dict(json.load(f))


def save_instance(instance: Instance, path: str) -> None:
    data = dict(
        tasks=instance.task_ids,
        conflicts=instance.conflicts,
        resources=instance.resources,
        capacities=instance.capacities,
        windows=instance.windows,
        weights=instance.weights,
        K=instance.K,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def result_to_dict(result: SchedulerResult) -> dict:
    return dict(
        assignment=result.assignment,
        penalty=result.penalty,
        runtime_ms=result.runtime_ms,
        feasible=result.feasible,
        violation_reason=result.violation_reason,
    )


def save_result(result: SchedulerResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_dict(result), f, indent=2)
