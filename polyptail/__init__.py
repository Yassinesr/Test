"""POT-TC: extreme-value tail calibration for polyp segmentation.

Implements Candidate 1 of the research-verification brief -- a peaks-over-
threshold / generalized-Pareto training objective that shapes the left tail of
the per-image Dice distribution -- together with the protocol infrastructure
(frozen manifests, a duplicate-collision audit, one evaluation implementation,
multi-seed paired statistics) that the brief marks as blocking before any
comparison is meaningful.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
