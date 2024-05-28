import warnings

from erlab.io.utils import *  # noqa: F403

warnings.warn(
    "`erlab.io.utilities` has been moved to `erlab.io.utils` "
    "and will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)
