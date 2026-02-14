"""Watch DataArray changes and synchronize them with ImageTool manager."""

__all__ = [
    "maybe_push",
    "shutdown",
    "watch",
    "watched_variables",
]

from erlab.interactive.imagetool.manager._watcher._core import (
    maybe_push,
    shutdown,
    watch,
    watched_variables,
)
