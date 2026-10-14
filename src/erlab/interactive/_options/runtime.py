"""Apply process-wide plot defaults at GUI startup."""

from collections.abc import Sequence

import pyqtgraph as pg

from erlab.interactive._options.core import options
from erlab.interactive._options.schema import (
    _DEFAULT_CURSOR_COLORS,
    _LIGHT_CURSOR_COLORS,
)

_plot_light_mode: bool | None = None


def initialize_plot_defaults() -> bool:
    """Apply saved plot defaults once and return the active light mode.

    Settings edits take effect in the next process. When light mode is disabled,
    preserve the host application's pyqtgraph configuration.
    """
    global _plot_light_mode

    if _plot_light_mode is None:
        light_mode = options.model.colors.plot_light_mode
        if light_mode:
            pg.setConfigOptions(background="w", foreground="k")
        _plot_light_mode = light_mode
    return _plot_light_mode


def effective_cursor_colors(colors: Sequence[str]) -> tuple[str, ...]:
    """Use the light alternatives only for the default cursor set."""
    palette = tuple(colors)
    if initialize_plot_defaults() and palette == _DEFAULT_CURSOR_COLORS:
        return _LIGHT_CURSOR_COLORS
    return palette
