import typing
from importlib.metadata import version as _version

import lazy_loader as _lazy
import xarray_lmfit  # noqa: F401

# Register xarray accessors
import erlab.accessors.fit
import erlab.accessors.general
import erlab.accessors.kspace  # noqa: F401

# Lazy load submodules according to SPEC 1
__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    submodules={
        "analysis",
        "constants",
        "interactive",
        "io",
        "lattice",
        "plotting",
        "utils",
    },
)

try:
    __version__ = _version("erlab")
except Exception:
    __version__ = "0.0.0"


if typing.TYPE_CHECKING:
    from erlab import (  # noqa: F401
        analysis,
        constants,
        interactive,
        io,
        lattice,
        plotting,
        utils,
    )
