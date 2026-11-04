# TaskAudit core

TaskAudit core selects a small set of annotation coordinates to refine fixed task
classes while allowing a declared number of missing positive cells to be added.
It returns a selected set, a worst completion and a primary optimality interval.

This repository is the original **numerical core and synthetic examples**, version
0.1.0. It is distinct from the earlier task-manifest package version 0.2.0. It does
not contain compound annotation tables, biological measurements, empirical paper
results or the manuscript. **It does not reproduce all paper experiments.** Full
empirical reproduction materials remain subject to unresolved source-data rights
and access questions. A working demo is not evidence of biological effectiveness.

## Install and run

Python 3.11 or later is required. From this repository:

```sh
python verify_publication.py
python -m pip install .
python -m taskaudit_core.demo
python -m unittest discover -s tests -v
```

Verify a clean checkout before installation or building. Python packaging tools
may create ignored `build` or `*.egg-info` folders; these generated files are not
part of the publication allowlist and a later strict source-tree check will flag
them. Build outputs should stay outside any folder being uploaded as source.

The demo needs no network access after dependencies are installed. It uses three
synthetic letter labels and reports a certified additional-class value and upper
bound of 1, together with a separately certified infeasible pair-floor example.
Selected identifiers can differ between solver versions when several optima tie.
The frozen modules are bundled in a wheel as well as in the source tree.

The recorded preparation environment was CPython 3.13.3, NumPy 2.3.5 and SciPy
1.15.2 on Windows. Exact numerical dependency pins are in `requirements-lock.txt`.
Other platforms have not been tested for this release. Dependencies are installed
separately and retain their own licences.

## Minimal API

```python
from taskaudit_core import Instance, exact_design, oracle

raw = {
    "instance_id": "synthetic_example",
    "candidates": [
        {"id": "synthetic_0", "labels": ["a"]},
        {"id": "synthetic_1", "labels": ["b"]},
        {"id": "synthetic_2", "labels": ["c"]},
    ],
    "pairs": [["a", "b"], ["a", "c"], ["b", "c"]],
}
instance = Instance(raw)
answer = exact_design(instance, budget=2, gamma=1, seconds=10)
fixed_panel = oracle(instance, answer["selected"], gamma=1)
print(answer["value"], answer["upper"], answer["certified"])
```

`selected` contains zero-based candidate indices. Attack entries are indices into
`instance.cells`, each a `(candidate_index, missing_label)` pair. The public API
also exposes `PairInstance`, `pair_oracle` and `exact_pair_design` for the full
separated-pair objective. `FloorInstance` and `exact_design_with_floor` maximize
task refinement while enforcing a robust pair-coverage floor, using separate
exact separators for the two objectives.

The complete signatures, base eligibility and unique entity identities must be
established by the caller before constructing the numerical instance. This core
does not load a source database or infer biological identities. The `pairs` field
must be the union of complete pair cliques within the fixed initial classes;
arbitrary pair graphs are rejected. The caller must keep recorded positives and
the base fixed, and must not silently equate unrecorded labels with verified
biological negatives.

## Objective and guarantees

For an initial class C and selected set S, group labels by their completed
annotation columns on S. H is the sum, over initial classes, of the number of
resulting groups minus one. The robust objective minimizes H over at most Gamma
candidate-specific zero-to-one cell additions, then maximizes it over sets with
at most B candidates. Gamma is a sensitivity budget, not an estimated error rate.

The core supports Gamma in {0, 1, 2}, unit candidate costs and one global unit-cell
budget. It includes exact fixed-panel completion enumeration, safe full-pattern
multiplicity reduction, prefix symmetry and a representative-based scenario
master. An integer lower/upper equality certifies the primary robust objective
only. A timeout preserves a valid interval; requested solver time is not a hard
end-to-end deadline. The nominal secondary tie-break is not generally certified.

H and full pair coverage F are different utilities. The optional F-floor solver
separates H-worst and F-worst completions independently. A cell addition can create
as well as destroy pair distinctions. Nominal H is not generally submodular, so
ordinary submodular-coverage guarantees must not be transferred to this problem.
Distinct-trace selection and robust scenario generation are established ideas;
this code is a specialization with explicit assumptions, not a new general
optimization framework.

## Integrity and tests

The three source cores are preserved byte-for-byte under
`src/taskaudit_core/_frozen/`; their relative directory structure is retained so
the floor extension can verify its frozen dependencies. `FROZEN_CORE_MANIFEST.json`
records their hashes. Public imports and packaging do not rewrite those modules.

The tests use only original synthetic letter-label fixtures. Their reference
solutions enumerate all selected subsets and all selected-row missing cells,
including all-zero blocks, using raw set partitions rather than the production
oracle or master to compute expected values. They check the H/F oracles, safe
reduction, symmetry, bounded outputs and the necessity of separate F separation.
They do not claim validation of unknown biological labels.

`PUBLISH_ALLOWLIST.txt` is the explicit publication file list.
`PUBLICATION_MANIFEST.json` records member hashes and sizes, with a separate
manifest-hash sidecar. Only listed files belong to this core source release.
Do not copy manuscript, real dataset, cache, private configuration or credential
folders into it. `verify_publication.py` verifies the listed bytes and rejects
unlisted files except local Git metadata and Python bytecode caches.

## Authors and licence

Pengyuan Zou and Liming Liu. Original code, documentation and synthetic examples
in this repository are licensed under MIT. No third-party data or software is
vendored. No archive DOI, journal acceptance or public package-index publication
is asserted by this prepared source tree.

