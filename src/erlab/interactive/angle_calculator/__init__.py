"""Angle calculator for measurement geometry and coverage."""

from __future__ import annotations

import sys
import types
import typing

if typing.TYPE_CHECKING:
    from erlab.interactive.angle_calculator._window import AngleCalculatorWindow

__all__ = ["angle_calculator"]


def angle_calculator(*, execute: bool | None = None) -> AngleCalculatorWindow:
    """Open the Angle calculator.

    .. versionadded:: 3.28.0

    Parameters
    ----------
    execute
        Start the Qt event loop. If `None`, use the current interactive environment.
        See :func:`erlab.interactive.utils.setup_qapp`.

    Returns
    -------
    AngleCalculatorWindow
        The calculator window. See :ref:`guide-angle-calculator` for controls
        and conventions.
    """
    from erlab.interactive.angle_calculator._window import AngleCalculatorWindow
    from erlab.interactive.utils import setup_qapp

    with setup_qapp(execute):
        window = AngleCalculatorWindow()
        window.show()
        window.raise_()
        window.activateWindow()
    return window


class _CallableModule(types.ModuleType):
    # Importing this package can replace the parent package's launcher attribute.
    # Keep the launcher callable in either import order, like ptable.
    def __call__(self, *, execute: bool | None = None) -> AngleCalculatorWindow:
        return angle_calculator(execute=execute)


sys.modules[__name__].__class__ = _CallableModule
