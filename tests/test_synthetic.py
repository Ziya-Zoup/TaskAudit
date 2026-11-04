"""Independent synthetic enumeration; no real annotations or production reference values."""
from itertools import combinations
import unittest

from taskaudit_core import (
    Instance, PairInstance, FloorInstance, oracle, pair_oracle,
    exact_design, exact_design_with_floor,
)


def subsets(n, budget=None):
    for size in range(min(n, n if budget is None else budget) + 1):
        yield from combinations(range(n), size)


def fixture(name, groups, annotations):
    groups = [set(g) for g in groups]
    rows = [set(a) for a in annotations]
    raw = {
        "instance_id": "synthetic_" + name,
        "candidates": [{"id": f"synthetic_{i}", "labels": sorted(a)} for i, a in enumerate(rows)],
        "pairs": [list(p) for g in groups for p in combinations(sorted(g), 2)],
    }
    return groups, rows, raw


def examples():
    return [
        fixture("empty", ["abc"], []),
        fixture("no_class", [], ["z"]),
        fixture("all_zero", ["abc"], ["", ""]),
        fixture("three_singletons", ["abc"], ["a", "b", "c"]),
        fixture("duplicates", ["abc"], ["a", "a", "b", "b"]),
        fixture("two_classes", ["abc", "def"], ["a", "bd", "e", "cf"]),
        fixture("crossing_cuts", ["abcd"], ["ab", "ac"]),
    ]


def raw_scores(groups, rows, selected, attack=()):
    completed = [set(a) for a in rows]
    for i, label in attack:
        completed[i].add(label)
    h = 0
    for group in groups:
        partition = [group]
        for i in selected:
            partition = [part for block in partition
                         for part in (block & completed[i], block - completed[i]) if part]
        h += len(partition) - 1
    f = sum(any((a in completed[i]) != (b in completed[i]) for i in selected)
            for group in groups for a, b in combinations(sorted(group), 2))
    return h, f


def raw_reference(groups, rows):
    """All missing selected-row cells, including initially all-zero class blocks."""
    labels = set().union(*groups) if groups else set()
    result = {}
    for selected in subsets(len(rows)):
        missing = [(i, label) for i in selected for label in sorted(labels - rows[i])]
        h0, f0 = raw_scores(groups, rows, selected)
        h, f = [h0] * 3, [f0] * 3
        for size in (1, 2):
            for attack in combinations(missing, size):
                hv, fv = raw_scores(groups, rows, selected, attack)
                for gamma in range(size, 3):
                    h[gamma] = min(h[gamma], hv)
                    f[gamma] = min(f[gamma], fv)
        result[selected] = (h, f)
    return result


class SyntheticCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = [(g, rows, raw, raw_reference(g, rows)) for g, rows, raw in examples()]

    def test_fixed_oracles_against_all_missing_cells(self):
        for groups, rows, raw, expected in self.fixtures:
            h, f = Instance(raw), PairInstance(raw)
            for selected, (hv, fv) in expected.items():
                for gamma in range(3):
                    with self.subTest(name=raw["instance_id"], selected=selected, gamma=gamma):
                        self.assertEqual(oracle(h, selected, gamma)["value"], hv[gamma])
                        self.assertEqual(pair_oracle(f, selected, gamma)["value"], fv[gamma])

    def test_kernel_symmetry_and_primary_certificates(self):
        for groups, rows, raw, expected in self.fixtures:
            instance = Instance(raw)
            for budget in sorted({0, min(2, len(rows)), len(rows)}):
                for gamma in range(3):
                    optimum = max(h[gamma] for selected, (h, _) in expected.items() if len(selected) <= budget)
                    for reduction in ("none", "kernel"):
                        for symmetry in (False, True):
                            with self.subTest(name=raw["instance_id"], budget=budget, gamma=gamma,
                                              reduction=reduction, symmetry=symmetry):
                                result = exact_design(instance, budget, gamma, reduction=reduction,
                                                      symmetry=symmetry, seconds=10)
                                actual = expected[tuple(result["selected"])][0][gamma]
                                self.assertLessEqual(len(result["selected"]), budget)
                                self.assertEqual(result["value"], actual)
                                self.assertLessEqual(result["value"], optimum)
                                self.assertGreaterEqual(result["upper"], optimum)
                                self.assertTrue(result["certified"])
                                self.assertEqual(result["value"], optimum)

    def test_zero_solver_allowance_retains_valid_bounds(self):
        for groups, rows, raw, expected in self.fixtures:
            instance = Instance(raw)
            for gamma in range(3):
                budget = min(2, len(rows))
                optimum = max(h[gamma] for selected, (h, _) in expected.items() if len(selected) <= budget)
                result = exact_design(instance, budget, gamma, seconds=0)
                self.assertEqual(result["value"], expected[tuple(result["selected"])][0][gamma])
                self.assertLessEqual(result["value"], optimum)
                self.assertGreaterEqual(result["upper"], optimum)

    def test_floor_against_joint_raw_reference(self):
        for groups, rows, raw, expected in self.fixtures:
            instance = FloorInstance(raw)
            budget = min(2, len(rows))
            allowed = {s: v for s, v in expected.items() if len(s) <= budget}
            for gamma in range(3):
                fstar = max(f[gamma] for h, f in allowed.values())
                for floor in sorted({0, fstar, fstar + 1}):
                    optimum = max((h[gamma] for h, f in allowed.values() if f[gamma] >= floor), default=None)
                    result = exact_design_with_floor(instance, budget, gamma, floor, seconds=10)
                    if optimum is None:
                        self.assertTrue(result["infeasibility_certified"])
                        self.assertIsNone(result["value"])
                    else:
                        h, f = allowed[tuple(result["selected"])]
                        self.assertGreaterEqual(f[gamma], floor)
                        self.assertEqual(result["value"], h[gamma])
                        self.assertTrue(result["certified"])
                        self.assertEqual(result["value"], optimum)
                        self.assertEqual(result["upper"], optimum)

    def test_pair_floor_requires_its_own_separator(self):
        groups, rows, raw = fixture("floor_separator_trap", ["abcd"], ["ab"])
        h, f = oracle(Instance(raw), [0], 1), pair_oracle(PairInstance(raw), [0], 1)
        self.assertEqual(h["value"], 1)
        self.assertEqual(h["attack"], [])
        self.assertEqual(f["value"], 3)
        self.assertTrue(f["attack"])
        answer = exact_design_with_floor(FloorInstance(raw), 1, 1, 4, seconds=10, seeds=[[0]])
        self.assertTrue(answer["infeasibility_certified"])
        self.assertGreaterEqual(answer["f_cuts"], 1)

    def test_non_submodular_nominal_and_invalid_domain(self):
        groups, rows, raw = fixture("crossing", ["abcd"], ["ab", "ac"])
        self.assertEqual([raw_scores(groups, rows, s)[0] for s in [(), (0,), (1,), (0, 1)]], [0, 1, 1, 3])
        with self.assertRaises(ValueError):
            Instance({"candidates": [], "pairs": [["a", "b"], ["b", "c"]]})
        with self.assertRaises(ValueError):
            oracle(Instance(raw), [0], 3)
        with self.assertRaises(ValueError):
            oracle(Instance(raw), [0, 0], 1)


if __name__ == "__main__":
    unittest.main()
