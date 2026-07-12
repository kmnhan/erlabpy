# ruff: noqa: F401

import ast
import builtins
import contextlib
import functools
import gc
import json
import sys
import types
import typing
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
import pytest
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import style as mpl_style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.accessors.general as accessor_general
import erlab.interactive._figurecomposer._axes as figurecomposer_axes
import erlab.interactive._figurecomposer._code as figurecomposer_code
import erlab.interactive._figurecomposer._defaults as figurecomposer_defaults
import erlab.interactive._figurecomposer._gridspec as figurecomposer_gridspec
import erlab.interactive._figurecomposer._line_colormap as figurecomposer_line_colormap
import erlab.interactive._figurecomposer._line_style as figurecomposer_line_style
import erlab.interactive._figurecomposer._norms as figurecomposer_norms
import erlab.interactive._figurecomposer._provenance as figurecomposer_provenance
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._seeding as figurecomposer_seeding
import erlab.interactive._figurecomposer._sources as figurecomposer_sources
import erlab.interactive._figurecomposer._text as figurecomposer_text
import erlab.interactive._figurecomposer._tick_params as figurecomposer_tick_params
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._figurecomposer._widgets as figurecomposer_widgets
import erlab.interactive._stylesheets
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._workspace as manager_workspace
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.plot_items as imagetool_plot_items
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureExportState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    _editor_controls,
    _line_transform,
)
from erlab.interactive._figurecomposer import (
    _source_inspector as figurecomposer_source_inspector,
)
from erlab.interactive._figurecomposer import (
    _subplot_adjust as figurecomposer_subplot_adjust,
)
from erlab.interactive._figurecomposer import (
    _toolbar_dialogs as figurecomposer_toolbar_dialogs,
)
from erlab.interactive._figurecomposer._exceptions import (
    FigureComposerPlotSlicesSelectionError,
)
from erlab.interactive._figurecomposer._operations import (
    _bz_overlay as figurecomposer_bz_overlay,
)
from erlab.interactive._figurecomposer._operations import (
    _custom_code as figurecomposer_custom_code,
)
from erlab.interactive._figurecomposer._operations import (
    _line_profile as figurecomposer_line_profile,
)
from erlab.interactive._figurecomposer._operations import (
    _method as figurecomposer_method,
)
from erlab.interactive._figurecomposer._operations import (
    _photon_energy as figurecomposer_photon_energy,
)
from erlab.interactive._figurecomposer._operations import (
    _plot_array as figurecomposer_plot_array,
)
from erlab.interactive._figurecomposer._operations import (
    _plot_slices as figurecomposer_plot_slices,
)
from erlab.interactive._figurecomposer._operations import (
    _set_palette as figurecomposer_set_palette,
)
from erlab.interactive._figurecomposer._seeding import (
    bz_overlay_operation_from_ktool,
    bz_overlay_operation_from_momentum_data,
    plot_slices_operation_with_source_styles,
)
from erlab.interactive._figurecomposer._toolbar_dialogs import (
    _connect_panel_editor_signal,
)
from erlab.interactive._options import options
from erlab.interactive._options.schema import AppOptions, FigureOptions
from erlab.interactive.imagetool import (
    _provenance_framework,
    _replay_graph,
    itool,
    provenance,
)
from erlab.io.exampledata import generate_hvdep_cuts
from tests.interactive.imagetool.manager.helpers import (
    InMemoryClipboard,
    _exec_generated_code,
    activate_widget_shortcut,
    install_in_memory_clipboard,
    select_child_tool,
    select_tools,
    trigger_menu_action,
)

_COLLAPSED_LAYOUT_WARNING = (
    "constrained_layout not applied because axes sizes collapsed to zero.  "
    "Try making figure larger or Axes decorations smaller."
)


@pytest.fixture(autouse=True)
def restore_interactive_options():
    old_options = options.model
    options.model = AppOptions()
    try:
        yield
    finally:
        options.model = old_options
        plt.close("all")


@pytest.fixture(autouse=True)
def isolate_qt_clipboard(monkeypatch: pytest.MonkeyPatch) -> InMemoryClipboard:
    return install_in_memory_clipboard(monkeypatch)


def _set_figure_stylesheets(stylesheets: list[str]) -> None:
    options.model = options.model.model_copy(
        update={"figure": FigureOptions(stylesheets=stylesheets)}
    )


def _send_mouse_event(
    widget: QtWidgets.QWidget,
    event_type: QtCore.QEvent.Type,
    pos: QtCore.QPoint,
    *,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.NoButton,
    buttons: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.NoButton,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    global_pos = widget.mapToGlobal(pos)
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(pos),
        QtCore.QPointF(global_pos),
        button,
        buttons,
        modifiers,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    QtWidgets.QApplication.processEvents()


def _send_mouse_move(
    widget: QtWidgets.QWidget,
    pos: QtCore.QPoint,
    *,
    buttons: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.NoButton,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    _send_mouse_event(
        widget,
        QtCore.QEvent.Type.MouseMove,
        pos,
        buttons=buttons,
        modifiers=modifiers,
    )


def _drag_widget(
    widget: QtWidgets.QWidget,
    start: QtCore.QPoint,
    end: QtCore.QPoint,
    *,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    _send_mouse_event(
        widget,
        QtCore.QEvent.Type.MouseButtonPress,
        start,
        button=QtCore.Qt.MouseButton.LeftButton,
        buttons=QtCore.Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )
    _send_mouse_move(
        widget,
        end,
        buttons=QtCore.Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )
    _send_mouse_event(
        widget,
        QtCore.QEvent.Type.MouseButtonRelease,
        end,
        button=QtCore.Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )


def _expected_layout_from_rcparams() -> str | None:
    if mpl.rcParams["figure.constrained_layout.use"]:
        return "constrained"
    if mpl.rcParams["figure.autolayout"]:
        return "tight"
    return None


def _unsupported_plot_slices_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(120.0).reshape(2, 3, 4, 5),
        dims=("a", "b", "c", "d"),
        coords={
            "a": [0.0, 1.0],
            "b": [0.0, 1.0, 2.0],
            "c": [0.0, 1.0, 2.0, 3.0],
            "d": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        name="map",
    )


def _set_unsupported_plot_slices_cursor_state(
    tool: erlab.interactive.imagetool.ImageTool,
) -> None:
    tool.slicer_area.add_cursor()
    tool.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    tool.slicer_area.set_value(axis=3, value=0.0, cursor=0)
    tool.slicer_area.set_value(axis=2, value=2.0, cursor=1)
    tool.slicer_area.set_value(axis=3, value=1.0, cursor=1)


def _file_load_provenance(path: Path) -> provenance.ToolProvenanceSpec:
    return provenance.file_load(
        start_label=f"Load data from file '{path.name}'",
        seed_code=f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )


def _select_operation_rows(tool: FigureComposerTool, rows: tuple[int, ...]) -> None:
    was_blocked = tool.operation_list.blockSignals(True)
    try:
        tool.operation_list.clearSelection()
        if rows:
            tool.operation_list.setCurrentItem(
                tool.operation_list.topLevelItem(rows[0])
            )
            for row in rows:
                item = tool.operation_list.topLevelItem(row)
                assert item is not None
                item.setSelected(True)
    finally:
        tool.operation_list.blockSignals(was_blocked)
    tool._operation_selection_changed()


def _selected_operation_rows(tool: FigureComposerTool) -> tuple[int, ...]:
    return tuple(
        row
        for row in range(tool.operation_list.topLevelItemCount())
        if tool.operation_list.topLevelItem(row).isSelected()
    )


def _operation_status_codes(tool: FigureComposerTool, row: int) -> tuple[str, ...]:
    item = tool.operation_list.topLevelItem(row)
    assert item is not None
    value = item.data(
        figurecomposer_tool_module._OPERATION_LIST_STATUS_COLUMN,
        figurecomposer_tool_module._OPERATION_LIST_STATUS_ROLE,
    )
    assert isinstance(value, tuple)
    return value


def _clear_clipboard() -> InMemoryClipboard:
    clipboard = QtWidgets.QApplication.clipboard()
    clipboard.clear()
    return clipboard


def _custom_order_step(label: str) -> FigureOperationState:
    return FigureOperationState.custom(
        label=label,
        code=f"fig.__dict__['_order'] = fig.__dict__.get('_order', []) + [{label!r}]",
        trusted=True,
    )


def _method_operations(
    tool: FigureComposerTool,
    family: FigureMethodFamily,
    name: str,
) -> tuple[FigureOperationState, ...]:
    return tuple(
        operation
        for operation in tool.tool_status.operations
        if operation.kind == FigureOperationKind.METHOD
        and operation.method_family == family
        and operation.method_name == name
    )


def _activate_combo_text(combo: QtWidgets.QComboBox, text: str) -> None:
    combo.setCurrentText(text)
    combo.activated.emit(combo.currentIndex())


def _activate_combo_index(combo: QtWidgets.QComboBox, index: int) -> None:
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def _click_tick_params_segment(
    editor: figurecomposer_tick_params.TickParamsEditorWidget,
    object_name: str,
    value: object,
) -> None:
    combo = editor.findChild(QtWidgets.QComboBox, object_name)
    assert combo is not None
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            combo.activated.emit(index)
            return
    raise AssertionError(f"No tick params combo {object_name!r} for {value!r}")


def _set_tick_params_button(
    editor: figurecomposer_tick_params.TickParamsEditorWidget,
    object_name: str,
    value: object,
) -> None:
    check = editor.findChild(QtWidgets.QCheckBox, object_name)
    assert check is not None
    state = {
        True: QtCore.Qt.CheckState.Checked,
        False: QtCore.Qt.CheckState.Unchecked,
        None: QtCore.Qt.CheckState.PartiallyChecked,
    }[value]
    check.setCheckState(state)


def _finish_tick_params_edit(
    editor: figurecomposer_tick_params.TickParamsEditorWidget,
    object_name: str,
    text: str,
) -> QtWidgets.QLineEdit:
    edit = editor.findChild(QtWidgets.QLineEdit, object_name)
    assert edit is not None
    edit.setText(text)
    edit.setModified(True)
    edit.editingFinished.emit()
    return edit


def _plot_source_checks(tool: FigureComposerTool) -> dict[str, QtWidgets.QCheckBox]:
    return {
        str(source_name): check
        for check in tool.step_source_controls.findChildren(QtWidgets.QCheckBox)
        if (source_name := check.property("figure_source_name")) is not None
    }


def _assert_step_editor_section(
    page: QtWidgets.QWidget, object_name: str
) -> QtWidgets.QWidget:
    section = page.findChild(QtWidgets.QWidget, object_name)
    assert section is not None
    assert section.property("figureComposerSectionHeader") is True
    assert section.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    section.ensurePolished()
    assert section.sizeHint().height() > 0
    assert section.sizeHint().width() > 0

    label = next(
        (
            child
            for child in section.findChildren(QtWidgets.QLabel)
            if child.property("figureComposerSectionHeaderLabel") is True
        ),
        None,
    )
    assert label is not None
    assert label.font().bold()

    line = section.findChild(QtWidgets.QFrame, f"{object_name}Line")
    assert line is not None
    assert line.frameShape() == QtWidgets.QFrame.Shape.HLine
    assert line.frameShadow() == QtWidgets.QFrame.Shadow.Sunken
    return section


def _plot_source_move_buttons(
    tool: FigureComposerTool,
) -> dict[tuple[str, str], QtWidgets.QToolButton]:
    return {
        (
            str(source_name),
            str(direction),
        ): button
        for button in tool.step_source_controls.findChildren(QtWidgets.QToolButton)
        if (source_name := button.property("figure_source_name")) is not None
        and (direction := button.property("figure_source_move")) is not None
    }


def _render_figure_composer_rgba(tool: FigureComposerTool) -> np.ndarray:
    with figurecomposer_defaults._figure_style_context():
        figure = Figure(
            figsize=tool.tool_status.setup.figsize,
            dpi=tool.tool_status.setup.dpi,
            layout=tool.tool_status.setup.layout,
        )
        canvas = FigureCanvasAgg(figure)
        figurecomposer_rendering._render_into_figure(
            tool,
            figure,
            sync_visible=False,
        )
        assert tool._operation_render_errors == {}
        with figurecomposer_defaults._figure_draw_context():
            canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()


def _restored_figure_composer_from_netcdf(
    tool: FigureComposerTool,
    qtbot,
    tmp_path,
) -> FigureComposerTool:
    filename = tmp_path / "figure-composer-tool.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are writing invalid netcdf features.*",
            category=UserWarning,
        )
        tool.to_dataset().to_netcdf(
            filename,
            engine="h5netcdf",
            invalid_netcdf=True,
        )
    saved = xr.load_dataset(filename, engine="h5netcdf")
    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    return restored


def _assert_serialized_plot_restores_exactly(
    tool: FigureComposerTool,
    qtbot,
    tmp_path,
) -> FigureComposerTool:
    before = _render_figure_composer_rgba(tool)
    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)
    assert restored.tool_status.model_dump(mode="json") == tool.tool_status.model_dump(
        mode="json"
    )
    after = _render_figure_composer_rgba(restored)
    np.testing.assert_array_equal(after, before)
    return restored


def _figure_composer_image_source(name: str = "image") -> xr.DataArray:
    eV = np.array([-0.5, 0.0, 0.5])
    beta = np.linspace(-1.0, 1.0, 8)
    alpha = np.linspace(-0.75, 0.75, 7)
    values = (
        np.cos(np.pi * eV[:, None, None])
        + 0.4 * np.sin(np.pi * beta[None, :, None])
        + 0.2 * alpha[None, None, :] ** 2
    )
    return xr.DataArray(
        values,
        dims=("eV", "beta", "alpha"),
        coords={"eV": eV, "beta": beta, "alpha": alpha},
        name=name,
    )


def _figure_composer_profile_source(name: str = "profile") -> xr.DataArray:
    kx = np.linspace(-1.0, 1.0, 21)
    values = np.exp(-4.0 * kx**2) + 0.15 * kx
    return xr.DataArray(values, dims=("kx",), coords={"kx": kx}, name=name)


def _figure_composer_line_slice_source(name: str = "line_map") -> xr.DataArray:
    eV = np.array([-0.5, 0.5])
    kx = np.linspace(-1.0, 1.0, 17)
    values = np.vstack(
        (
            np.exp(-8.0 * (kx + 0.25) ** 2),
            np.exp(-8.0 * (kx - 0.25) ** 2),
        )
    )
    return xr.DataArray(
        values,
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name=name,
    )


def _figure_composer_replay_source_state(
    name: str,
    label: str | None = None,
) -> FigureSourceState:
    source_spec = provenance.script(
        start_label=f"Build {label or name}",
        seed_code="derived = xr.DataArray([0.0], dims=('x',))",
        active_name="derived",
    )
    return FigureSourceState(
        name=name,
        label=label or name,
        provenance_spec=source_spec.model_dump(mode="json"),
    )


def _assert_figure_composer_provenance_replayable(
    tool: FigureComposerTool,
    *,
    case_label: str,
) -> str:
    spec = tool.current_provenance_spec()
    assert spec is not None
    try:
        graph = _replay_graph.compile_replay_graph(spec, display=True)
        return _replay_graph.emit_replay_code(graph, output_name="fig")
    except _replay_graph.ReplayGraphError as exc:
        pytest.fail(
            f"{case_label} generated Figure Composer provenance is not replayable: "
            f"{exc}\n\nGenerated code:\n{tool.generated_code()}"
        )


def _expected_line_colormap_colors(
    values: Sequence[float],
    cmap: str,
    *,
    trim: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        norm = mcolors.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(values)
    trim_lower, trim_upper = trim
    if trim_lower or trim_upper:
        normalized = trim_lower + (1.0 - trim_lower - trim_upper) * normalized
    return plt.get_cmap(cmap)(normalized)


def _assert_errorbar_xerr(
    container: ErrorbarContainer,
    x: Sequence[float],
    y: Sequence[float],
    xerr: Sequence[float],
) -> None:
    barlinecols = container.lines[2]
    assert barlinecols
    segments = barlinecols[0].get_segments()
    assert len(segments) == len(x)
    expected = [
        ((x_value - err, y_value), (x_value + err, y_value))
        for x_value, y_value, err in zip(x, y, xerr, strict=True)
    ]
    np.testing.assert_allclose(segments, expected)


def _assert_errorbar_yerr(
    container: ErrorbarContainer,
    x: Sequence[float],
    y: Sequence[float],
    yerr: Sequence[float],
) -> None:
    barlinecols = container.lines[2]
    assert len(barlinecols) >= 2
    segments = barlinecols[1].get_segments()
    assert len(segments) == len(x)
    expected = [
        ((x_value, y_value - err), (x_value, y_value + err))
        for x_value, y_value, err in zip(x, y, yerr, strict=True)
    ]
    np.testing.assert_allclose(segments, expected)


def _assert_errorbar_capsize(container: ErrorbarContainer, capsize: float) -> None:
    caplines = container.lines[1]
    assert caplines
    for capline in caplines:
        assert capline.get_markersize() == pytest.approx(2.0 * capsize)


def _bz_tool(operation: FigureOperationState) -> FigureComposerTool:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-1.0, 1.0]},
        name="data",
    )
    return FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )


def _assert_bz_lines_match_segments(
    lines: Sequence[typing.Any], segments: np.ndarray
) -> None:
    assert len(lines) == len(segments)
    for line, segment in zip(lines, segments, strict=True):
        np.testing.assert_allclose(line.get_xdata(), segment[:, 0])
        np.testing.assert_allclose(line.get_ydata(), segment[:, 1])


@functools.cache
def _cached_photon_energy_source(configuration: int) -> xr.DataArray:
    data = generate_hvdep_cuts((5, 24, 20), seed=1)
    data.kspace.inner_potential = 10.0
    data.attrs["configuration"] = configuration
    return data.kspace.convert(silent=True)


def _photon_energy_source(
    configuration: int = 1, *, name: str = "hvdep_kconv"
) -> xr.DataArray:
    data = _cached_photon_energy_source(configuration).copy(deep=True)
    data.name = name
    return data


def _photon_energy_tool(
    operation: FigureOperationState,
    data: xr.DataArray,
    *,
    setup: FigureSubplotsState | None = None,
    extra_source_data: dict[str, xr.DataArray] | None = None,
) -> FigureComposerTool:
    source_name = str(data.name or "hvdep_kconv")
    source_data = {source_name: data}
    if extra_source_data:
        source_data.update(extra_source_data)
    sources = tuple(FigureSourceState(name=name, label=name) for name in source_data)
    return FigureComposerTool.from_sources(
        source_data,
        sources=sources,
        operations=(operation,),
        setup=setup or FigureSubplotsState(),
        primary_source=source_name,
    )


def _assert_photon_energy_lines_match(
    lines: Sequence[typing.Any], expected: xr.DataArray, x_dim: str
) -> None:
    assert len(lines) == expected.sizes["hv"]
    for index, line in enumerate(lines):
        kz = expected.isel(hv=index)
        np.testing.assert_allclose(line.get_xdata(), kz[x_dim].values)
        np.testing.assert_allclose(line.get_ydata(), kz.values)
        assert line.get_label() == f"$h\\nu = {kz.hv:g}$ eV"


def _selection_shortcut_sequences(widget: QtWidgets.QWidget) -> set[str]:
    return {
        shortcut.key().toString(QtGui.QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QtWidgets.QShortcut)
        if shortcut.parent() is widget
    }


__all__ = tuple(name for name in globals() if not name.startswith("__"))
