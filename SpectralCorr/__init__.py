"""Public package surface used by the shipped examples."""

from .version import __version__
from .ar_processes import AR1_process
from .correlations import cross_correlation, maximum_cross_correlation
from .plotting import plot_cross_correlation

__all__ = [
    "__version__",
    "AR1_process",
    "cross_correlation",
    "maximum_cross_correlation",
    "plot_cross_correlation",
]
