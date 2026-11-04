"""Run a small, entirely synthetic example without network or data downloads."""
from itertools import combinations
import json

from . import Instance, exact_design, FloorInstance, exact_design_with_floor


def main():
    labels = ["a", "b", "c"]
    raw = {
        "instance_id": "synthetic_three_labels",
        "candidates": [{"id": f"synthetic_{i}", "labels": [t]} for i, t in enumerate(labels)],
        "pairs": [list(p) for p in combinations(labels, 2)],
    }
    instance = Instance(raw)
    answer = exact_design(instance, budget=2, gamma=1, seconds=10)
    assert answer["certified"] and answer["value"] == answer["upper"] == 1

    # A separate F separator is necessary: H's worst completion need not be F's.
    trap = {
        "instance_id": "synthetic_floor_trap",
        "candidates": [{"id": "synthetic_x", "labels": ["a", "b"]}],
        "pairs": [list(p) for p in combinations(["a", "b", "c", "d"], 2)],
    }
    floor = exact_design_with_floor(FloorInstance(trap), budget=1, gamma=1, f_floor=4,
                                    seconds=10, seeds=[[0]])
    assert floor["infeasibility_certified"] and floor["value"] is None
    result = {
        "scope": "Synthetic labels only; no measured chemical or biological data",
        "guaranteed_additional_classes": answer["value"],
        "upper_bound": answer["upper"],
        "primary_optimum_certified": answer["certified"],
        "selected_ids": [raw["candidates"][i]["id"] for i in answer["selected"]],
        "adverse_cells": [list(instance.cells[j]) for j in answer["attack"]],
        "pair_floor_trap_certified_infeasible": floor["infeasibility_certified"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

