from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def difficulty_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    *,
    expected_samples: int = 4,
) -> dict[str, Any]:
    by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_spec[str(row["spec_id"])].append(row)
    spec_ids = sorted(by_spec)
    if not spec_ids:
        raise ValueError("difficulty results are empty")
    for spec_id, values in by_spec.items():
        actual = {
            (str(value["arm"]), str(value["generator_model"]), int(value["sample_index"]))
            for value in values
        }
        expected = {
            (arm, generator, sample_index)
            for arm in ("thinking", "instruct")
            for generator in ("generator_20b", "generator_120b")
            for sample_index in range(expected_samples)
        }
        if actual != expected or len(values) != len(expected):
            raise ValueError(
                f"difficulty matched records are incomplete for {spec_id}: rows={len(values)}, unique={len(actual)}"
            )

    def delta(ids: list[str], arm: str | None = None) -> float:
        selected = [
            item for spec_id in ids for item in by_spec[spec_id]
            if arm is None or item["arm"] == arm
        ]
        accuracy = {}
        for generator in ("generator_20b", "generator_120b"):
            values = [float(item["correct"]) for item in selected if item["generator_model"] == generator]
            if not values:
                raise ValueError(f"missing difficulty values for {generator}, arm={arm}")
            accuracy[generator] = float(np.mean(values))
        return accuracy["generator_20b"] - accuracy["generator_120b"]

    observed = delta(spec_ids)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = [spec_ids[value] for value in rng.integers(0, len(spec_ids), size=len(spec_ids))]
        boot[index] = delta(sampled)
    return {
        "pooled_delta": observed,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "thinking_delta": delta(spec_ids, "thinking"),
        "instruct_delta": delta(spec_ids, "instruct"),
        "spec_count": len(spec_ids),
        "bootstrap_samples": bootstrap_samples,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted
