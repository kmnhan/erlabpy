import warnings

from erlab.interactive.utils import *  # noqa: F403

warnings.warn(
    "`erlab.interactive.utilities` has been moved to `erlab.interactive.utils` "
    "and will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)
