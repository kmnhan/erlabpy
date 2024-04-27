import warnings

from erlab.io.characterization import resistance, xrd  # noqa: F401

warnings.warn(
    "`erlab.characterization` is deprecated. Use `erlab.io.characterization` instead",
    DeprecationWarning,
    stacklevel=2,
)
