"""Watch DataArray changes and synchronize them with ImageTool manager."""

__all__ = [
    "WatcherMagics",
    "_core",
    "_ipython",
    "enable_ipython_auto_push",
    "maybe_push",
    "shutdown",
    "watch",
    "watched_variables",
]

from erlab.interactive.imagetool.manager._watcher import _core, _ipython
from erlab.interactive.imagetool.manager._watcher._core import (
    maybe_push,
    shutdown,
    watch,
    watched_variables,
)
from erlab.interactive.imagetool.manager._watcher._ipython import (
    WatcherMagics,
    enable_ipython_auto_push,
)
