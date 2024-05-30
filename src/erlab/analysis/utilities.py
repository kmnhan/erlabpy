import warnings

from erlab.analysis.utils import correct_with_edge, shift  # noqa: F401

warnings.warn(
    "`erlab.analysis.utilities` has been moved to `erlab.analysis.utils` "
    "and will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)
