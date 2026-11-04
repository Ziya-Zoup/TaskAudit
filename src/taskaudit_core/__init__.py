"""Public imports for the original, byte-preserved TaskAudit numerical cores.

This package does not contain compound annotations or empirical paper results.
"""
from ._frozen.research_validation_20261104.class_design.class_design import (
    Instance, exact_design, greedy, oracle,
)
from ._frozen.research_method_20261104.full_design import (
    Instance as PairInstance,
    exact_design as exact_pair_design,
    oracle as pair_oracle,
)
from ._frozen.research_validation_20261104.class_design.floor_extension import (
    Instance as FloorInstance,
    exact_design as exact_design_with_floor,
)

__version__ = "0.1.0"
__all__ = [
    "Instance", "exact_design", "greedy", "oracle", "PairInstance",
    "exact_pair_design", "pair_oracle", "FloorInstance", "exact_design_with_floor",
]

