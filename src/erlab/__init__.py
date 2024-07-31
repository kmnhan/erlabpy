from importlib.metadata import version as _version

import erlab.accessors.fit
import erlab.accessors.general
import erlab.accessors.kspace  # noqa: F401

try:
    __version__ = _version("erlab")
except Exception:
    __version__ = "0.0.0"
