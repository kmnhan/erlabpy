from __future__ import annotations

import sys
import types
from typing import Any

from ._window import PeriodicTableWindow, ptable

__all__ = ["PeriodicTableWindow", "ptable"]


class _CallableModule(types.ModuleType):
    # Importing the submodule first can cause ``erlab.interactive.ptable`` on the
    # parent package to resolve to this module object instead of the launcher
    # function. Keeping the module callable preserves ``erlab.interactive.ptable()``
    # regardless of import order.
    def __call__(self, *args: Any, **kwargs: Any) -> PeriodicTableWindow:
        return ptable(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
