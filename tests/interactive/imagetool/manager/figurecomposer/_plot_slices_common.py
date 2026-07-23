"""Shared support for the Figure Composer Plot Slices tests."""

import typing
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import style as mpl_style
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets

import erlab.accessors.general as accessor_general
import erlab.interactive._figurecomposer._norms as figurecomposer_norms
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._text as figurecomposer_text
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._stylesheets
import erlab.interactive.imagetool._figurecomposer_adapter as figurecomposer_adapter
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._exceptions import (
    FigureComposerPlotSlicesSelectionError,
)
from erlab.interactive._figurecomposer._model._document import FigureDocument
from erlab.interactive._figurecomposer._operations._method._catalog import _method_spec
from erlab.interactive._figurecomposer._operations._method._editor import (
    _method_float_pair_args,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _codegen as plot_slices_codegen,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _editor as plot_slices_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _model as plot_slices_model,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _panel_style_editor as plot_slices_panel_style_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _render as plot_slices_render,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _spec as plot_slices_spec,
)
from erlab.interactive._figurecomposer._seeding import (
    plot_slices_operation_with_source_styles,
)
from erlab.interactive._figurecomposer._ui import (
    _toolbar_dialogs as figurecomposer_toolbar_dialogs,
)
from erlab.interactive._options import options
from erlab.interactive.imagetool._provenance._code import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
)
from erlab.interactive.imagetool._provenance._graph import _validate_script_code_names

from ._common import (
    _activate_combo_index,
    _activate_combo_text,
    _drag_widget,
    _expected_line_colormap_colors,
    _figure_composer_image_source,
    _operation_section_button,
    _operation_section_buttons,
    _plot_source_checks,
    _plot_source_move_buttons,
    _render_figure_composer_rgba,
    _select_operation_rows,
    _set_figure_stylesheets,
    _set_unsupported_plot_slices_cursor_state,
    _unsupported_plot_slices_data,
)

__all__ = (
    "_SCRIPT_REPLAY_ALLOWED_BUILTINS",
    "Figure",
    "FigureAxesSelectionState",
    "FigureComposerPlotSlicesSelectionError",
    "FigureComposerTool",
    "FigureDataSelectionState",
    "FigureDocument",
    "FigureMethodFamily",
    "FigureOperationKind",
    "FigureOperationState",
    "FigurePlotSlicesPanelStyleState",
    "FigureRecipeState",
    "FigureSourceState",
    "FigureSubplotsState",
    "Path",
    "QtCore",
    "QtWidgets",
    "_activate_combo_index",
    "_activate_combo_text",
    "_drag_widget",
    "_expected_line_colormap_colors",
    "_figure_composer_image_source",
    "_method_float_pair_args",
    "_method_spec",
    "_operation_section_button",
    "_operation_section_buttons",
    "_plot_slices_selection_migration_data",
    "_plot_source_checks",
    "_plot_source_move_buttons",
    "_render_figure_composer_rgba",
    "_select_operation_rows",
    "_set_figure_stylesheets",
    "_set_unsupported_plot_slices_cursor_state",
    "_unsupported_plot_slices_data",
    "_validate_script_code_names",
    "accessor_general",
    "eplt",
    "erlab",
    "figurecomposer_adapter",
    "figurecomposer_norms",
    "figurecomposer_rendering",
    "figurecomposer_text",
    "figurecomposer_tool_module",
    "figurecomposer_toolbar_dialogs",
    "mcolors",
    "mpl",
    "mpl_style",
    "np",
    "options",
    "plot_slices_codegen",
    "plot_slices_editor",
    "plot_slices_model",
    "plot_slices_operation_with_source_styles",
    "plot_slices_panel_style_editor",
    "plot_slices_render",
    "plot_slices_spec",
    "plt",
    "pytest",
    "typing",
    "warnings",
    "xr",
)


def _plot_slices_selection_migration_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("x", "hv", "y"),
        coords={"x": [0.0, 1.0], "hv": [10.0, 20.0, 30.0], "y": [-1.0, 0.0, 1.0, 2.0]},
        name="first",
    )
