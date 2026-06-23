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
from erlab.interactive.imagetool import itool, provenance
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
    if not rows:
        tool.operation_list.clearSelection()
        tool._operation_selection_changed()
        return
    tool.operation_list.setCurrentRow(rows[0])
    tool.operation_list.clearSelection()
    for row in rows:
        item = tool.operation_list.item(row)
        assert item is not None
        item.setSelected(True)
    tool._operation_selection_changed()


def _selected_operation_rows(tool: FigureComposerTool) -> tuple[int, ...]:
    return tuple(
        row
        for row in range(tool.operation_list.count())
        if tool.operation_list.item(row).isSelected()
    )


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


def _source_refresh_buttons(
    tool: FigureComposerTool,
) -> dict[str, QtWidgets.QToolButton]:
    return {
        str(source_name): button
        for button in tool.source_list.findChildren(
            QtWidgets.QToolButton, "figureComposerRefreshSourceButton"
        )
        if (source_name := button.property("figure_source_name")) is not None
    }


def _source_remove_buttons(
    tool: FigureComposerTool,
) -> dict[str, QtWidgets.QToolButton]:
    return {
        str(source_name): button
        for button in tool.source_list.findChildren(
            QtWidgets.QToolButton, "figureComposerRemoveSourceButton"
        )
        if (source_name := button.property("figure_source_name")) is not None
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


def test_figure_composer_plot_source_move_button_uses_disabled_icon_color(
    qtbot, monkeypatch
) -> None:
    records: list[tuple[str, dict[str, typing.Any]]] = []

    def record_icon(name: str, **kwargs: typing.Any) -> QtGui.QIcon:
        records.append((name, kwargs))
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        return QtGui.QIcon(pixmap)

    monkeypatch.setattr(erlab.interactive.utils.qtawesome, "icon", record_icon)
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    button = figurecomposer_plot_slices._PlotSourceMoveButton("up", parent)
    palette = button.palette()
    disabled_text = QtGui.QColor("#6f7782")
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.ButtonText,
        disabled_text,
    )
    button.setPalette(palette)
    button.setEnabled(False)

    assert records[-1][0] == "mdi6.arrow-up"
    assert records[-1][1]["color_disabled"] == disabled_text


def test_figure_composer_operation_modules_use_editor_signal_contract() -> None:
    modules = (
        figurecomposer_custom_code,
        figurecomposer_line_profile,
        figurecomposer_method,
        figurecomposer_photon_energy,
        figurecomposer_plot_slices,
        figurecomposer_set_palette,
    )
    direct_connects: list[str] = []
    for module in modules:
        module_file = module.__file__
        assert module_file is not None
        tree = ast.parse(Path(module_file).read_text())
        direct_connects.extend(
            f"{Path(module_file).name}:{node.lineno}"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
            )
        )
    assert direct_connects == []


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


def test_figure_composer_secondary_source_roundtrip_ignores_stale_backend_encoding(
    qtbot,
    tmp_path,
) -> None:
    primary = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="primary",
    )
    secondary = xr.DataArray(
        np.arange(3.0) + 10.0,
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="secondary",
    )
    secondary.encoding["compression"] = "unknown"
    secondary.encoding["source"] = "stale-source.nc"
    secondary.coords["x"].encoding["compression"] = "unknown"
    recipe = FigureRecipeState(
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="secondary", label="secondary"),
        ),
        primary_source="primary",
    )
    tool = FigureComposerTool(
        primary,
        recipe=recipe,
        source_data={"primary": primary, "secondary": secondary},
    )
    qtbot.addWidget(tool)

    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)

    xr.testing.assert_equal(restored.source_data()["secondary"], secondary)
    assert secondary.encoding["compression"] == "unknown"
    assert secondary.coords["x"].encoding["compression"] == "unknown"


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


def test_figure_composer_set_palette_editor_preview_and_controls(qtbot) -> None:
    sns = pytest.importorskip("seaborn")
    profile = _figure_composer_profile_source("profile")
    operation = FigureOperationState.set_palette().model_copy(
        update={"palette_name": "deep"}
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(operation,),
        primary_source="profile",
    )
    tool = FigureComposerTool(profile, recipe=recipe, source_data={"profile": profile})
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    page = tool.step_editor_stack.currentWidget()
    assert page is not None

    combo = page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo")
    assert combo is not None
    assert combo.isEnabled()
    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    assert preview is not None
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    assert swatches
    expected_first = QtGui.QColor.fromRgbF(*sns.color_palette("deep")[0]).name()
    assert swatches[0].property("palette_color") == expected_first

    _activate_combo_text(combo, "colorblind")
    assert tool.tool_status.operations[0].palette_name == "colorblind"

    count_spin = page.findChild(QtWidgets.QSpinBox, "figureComposerSetPaletteCountSpin")
    assert count_spin is not None
    count_spin.setValue(3)
    QtWidgets.QApplication.processEvents()
    assert tool.tool_status.operations[0].palette_n_colors == 3
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    assert len(swatches) == 3
    expected_first = QtGui.QColor.fromRgbF(
        *sns.color_palette("colorblind", n_colors=3)[0]
    ).name()
    assert swatches[0].property("palette_color") == expected_first

    desat_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerSetPaletteSaturationSpin"
    )
    assert desat_spin is not None
    desat_spin.setValue(0.7)
    assert tool.tool_status.operations[0].palette_desat == pytest.approx(0.7)

    color_codes = page.findChild(
        QtWidgets.QCheckBox, "figureComposerSetPaletteColorCodesCheck"
    )
    assert color_codes is not None
    color_codes.click()
    assert tool.tool_status.operations[0].palette_color_codes is True


def test_figure_composer_set_palette_editor_disables_without_seaborn(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(figurecomposer_set_palette, "_import_seaborn", lambda: None)
    profile = _figure_composer_profile_source("profile")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(FigureOperationState.set_palette(),),
        primary_source="profile",
    )
    tool = FigureComposerTool(profile, recipe=recipe, source_data={"profile": profile})
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    page = tool.step_editor_stack.currentWidget()
    assert page is not None

    combo = page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo")
    count_spin = page.findChild(QtWidgets.QSpinBox, "figureComposerSetPaletteCountSpin")
    desat_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerSetPaletteSaturationSpin"
    )
    color_codes = page.findChild(
        QtWidgets.QCheckBox, "figureComposerSetPaletteColorCodesCheck"
    )
    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    message = page.findChild(
        QtWidgets.QLabel, "figureComposerSetPaletteUnavailableLabel"
    )

    assert combo is not None
    assert not combo.isEnabled()
    assert count_spin is not None
    assert not count_spin.isEnabled()
    assert desat_spin is not None
    assert not desat_spin.isEnabled()
    assert color_codes is not None
    assert not color_codes.isEnabled()
    assert preview is not None
    assert not preview.isEnabled()
    assert message is not None
    assert message.property("missing_dependency") == "seaborn"
    item = tool.operation_list.item(0)
    assert item is not None
    assert item.text().startswith("Skipped Set Palette:")

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}


def test_figure_composer_set_palette_render_and_generated_code(qtbot) -> None:
    sns = pytest.importorskip("seaborn")
    profile = _figure_composer_profile_source("profile")
    palette_operation = FigureOperationState.set_palette().model_copy(
        update={
            "palette_name": "colorblind",
            "palette_n_colors": 3,
            "palette_desat": 0.8,
            "palette_color_codes": True,
        }
    )
    line_operation = FigureOperationState.line(label="line", source="profile")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(palette_operation, line_operation),
        primary_source="profile",
    )
    tool = FigureComposerTool(profile, recipe=recipe, source_data={"profile": profile})
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    expected = sns.color_palette("colorblind", n_colors=3, desat=0.8)[0]
    np.testing.assert_allclose(
        mcolors.to_rgb(tool.figure.axes[0].lines[0].get_color()),
        expected,
    )

    code = tool.generated_code()
    assert "import seaborn as sns" in code
    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(mcolors.to_rgb(generated_line.get_color()), expected)


def test_figure_composer_set_palette_generated_code_skips_without_seaborn(
    qtbot,
) -> None:
    profile = _figure_composer_profile_source("profile")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(
            FigureOperationState.set_palette(),
            FigureOperationState.line(label="line", source="profile"),
        ),
        primary_source="profile",
    )
    tool = FigureComposerTool(profile, recipe=recipe, source_data={"profile": profile})
    qtbot.addWidget(tool)

    real_import = builtins.__import__

    def import_without_seaborn(
        name: str,
        globals_: dict[str, typing.Any] | None = None,
        locals_: dict[str, typing.Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> typing.Any:
        if name == "seaborn" or name.startswith("seaborn."):
            raise ImportError("seaborn is not installed")
        return real_import(name, globals_, locals_, fromlist, level)

    namespace: dict[str, typing.Any] = {
        "profile": profile,
        "__builtins__": {
            **vars(builtins),
            "__import__": import_without_seaborn,
        },
    }
    exec(tool.generated_code(), namespace)  # noqa: S102

    assert len(namespace["fig"].axes[0].lines) == 1


def test_figure_composer_plot_slices_source_selector_updates_sources(
    qtbot, monkeypatch
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="first_source", label="first"),
                FigureSourceState(name="second_source", label="second"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("first_source",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="first_source",
        ),
        source_data={"first_source": first, "second_source": second},
    )
    qtbot.addWidget(tool)
    tool._select_step_section("sources")

    selector = tool.step_source_controls.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesSourceSelector"
    )
    assert selector is not None
    assert not tool.step_source_controls.findChildren(QtWidgets.QLineEdit)
    checks = _plot_source_checks(tool)
    assert set(checks) == {"first_source", "second_source"}
    assert checks["first_source"].checkState() == QtCore.Qt.CheckState.Checked
    assert checks["second_source"].checkState() == QtCore.Qt.CheckState.Unchecked
    assert tool.source_status_label.text() == ""
    assert tool.source_status_label.isHidden()

    checks["second_source"].setCheckState(QtCore.Qt.CheckState.Checked)

    assert tool.tool_status.operations[0].sources == (
        "first_source",
        "second_source",
    )
    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("second_source", "up")].isEnabled(),
        timeout=1000,
    )
    buttons = _plot_source_move_buttons(tool)
    assert buttons[("first_source", "up")].isEnabled() is False
    assert buttons[("first_source", "down")].isEnabled() is True
    assert buttons[("second_source", "up")].isEnabled() is True
    assert buttons[("second_source", "down")].isEnabled() is False

    buttons[("second_source", "up")].click()

    assert tool.tool_status.operations[0].sources == (
        "second_source",
        "first_source",
    )
    captured_maps: list[tuple[str, ...]] = []

    def capture_plot_slices(maps, **_kwargs):
        captured_maps.append(tuple(data.name for data in maps))

    monkeypatch.setattr(eplt, "plot_slices", capture_plot_slices)
    namespace: dict[str, typing.Any] = {
        "first_source": first,
        "second_source": second,
    }
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert captured_maps == [("second", "first")]


def test_figure_composer_plot_slices_reuses_selection_cache_per_render(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data")
    axes = FigureAxesSelectionState(axes=((0, 0), (0, 1)))
    first_operation = FigureOperationState.plot_slices(
        label="first",
        sources=("data",),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0, 0.5),
    ).model_copy(update={"cmap": "viridis"})
    second_operation = FigureOperationState.plot_slices(
        label="second",
        sources=("data",),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0, 0.5),
    ).model_copy(update={"cmap": "magma"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(first_operation, second_operation),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    calls: list[dict[str, object]] = []
    original_qsel = accessor_general.SelectionAccessor.__call__

    def counted_qsel(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_qsel(self, *args, **kwargs)

    monkeypatch.setattr(
        accessor_general.SelectionAccessor,
        "__call__",
        counted_qsel,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 1
    assert calls[0]["eV"] == [0.0, 0.5]
    assert [len(axis.images) for axis in tool.figure.axes] == [2, 2]
    assert [image.get_cmap().name for image in tool.figure.axes[0].images] == [
        "viridis",
        "magma",
    ]

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 2


def test_figure_composer_source_ui_keeps_aliases_as_internal_keys(qtbot) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: sample_map"),
                FigureSourceState(name="data_1", label="ImageTool 1: reference_map"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": first, "data_1": second},
    )
    qtbot.addWidget(tool)

    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "data_0"
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is True
    assert first_item.font(0).bold()
    assert first_item.text(1) == "data_0"
    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    assert second_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is False
    assert second_item.text(1) == "data_1"

    tool._select_step_section("sources")
    checks = _plot_source_checks(tool)
    assert checks["data_0"].property("figure_source_name") == "data_0"

    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert "data_0" in tool.generated_code()


def test_figure_composer_source_ui_uses_shared_shape_formatter(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data")
    calls: list[tuple[tuple[str, ...], bool, str | None]] = []

    def _format_shape(darr: xr.DataArray, show_size: bool = False) -> str:
        calls.append((tuple(str(dim) for dim in darr.dims), show_size, darr.name))
        return "<p>formatted shape</p>"

    monkeypatch.setattr(
        erlab.utils.formatting,
        "format_darr_shape_html",
        _format_shape,
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    shape_widget = tool.source_list.itemWidget(first_item, 2)
    assert isinstance(shape_widget, QtWidgets.QLabel)
    assert shape_widget.text() == "<p>formatted shape</p>"
    assert tool.source_list.toolTip() == ""
    assert first_item.toolTip(0) == ""
    assert first_item.toolTip(1) == ""
    assert first_item.toolTip(2) == ""
    assert shape_widget.toolTip() == ""
    assert (tuple(str(dim) for dim in data.dims), False, None) in calls
    assert any(call[1] for call in calls)


def test_figure_composer_source_refresh_controls_use_live_source_callbacks(
    qtbot,
) -> None:
    image = _figure_composer_image_source("image")
    profile = _figure_composer_profile_source("profile")
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="ImageTool 0: image",
                    node_uid="node-0",
                ),
                FigureSourceState(name="profile", label="Detached"),
            ),
            operations=(),
            primary_source="data_0",
        ),
        source_data={"data_0": image, "profile": profile},
    )
    qtbot.addWidget(tool)

    buttons = _source_refresh_buttons(tool)
    assert set(buttons) == {"data_0", "profile"}
    assert buttons["data_0"].property("figure_source_name") == "data_0"
    assert buttons["data_0"].accessibleName() == "Refresh Source"
    assert not buttons["data_0"].isEnabled()
    assert (
        buttons["data_0"].toolTip() == "This source is not linked to an open ImageTool"
    )
    assert tool.refresh_sources_button.accessibleName() == "Refresh Sources"
    assert not tool.refresh_sources_button.isEnabled()
    assert tool.refresh_sources_button.toolTip() == (
        "No sources are linked to open ImageTools"
    )
    assert tool._source_refresh_label("data_0") is None
    tool._refresh_source_from_button("data_0")
    tool._refresh_sources_from_button()

    refreshed: list[tuple[str, str | tuple[str, ...]]] = []

    def can_refresh_source(name: str) -> bool:
        return name == "data_0"

    def refresh_source(name: str) -> bool:
        refreshed.append(("one", name))
        return True

    def refresh_sources(names: Sequence[str]) -> int:
        refreshed.append(("many", tuple(names)))
        return len(names)

    def source_label(name: str) -> str | None:
        return "ImageTool 0: image" if name == "data_0" else None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=source_label,
    )

    buttons = _source_refresh_buttons(tool)
    assert buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == (
        "Refresh “ImageTool 0: image” from ImageTool 0: image"
    )
    assert not buttons["profile"].isEnabled()
    assert buttons["profile"].toolTip() == (
        "This source is not linked to an open ImageTool"
    )
    assert tool.refresh_sources_button.isEnabled()
    assert tool.refresh_sources_button.toolTip() == (
        "Refresh all sources linked to open ImageTools"
    )

    status_before_refresh = (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    )
    buttons["data_0"].click()
    assert refreshed[-1] == ("one", "data_0")
    assert (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    ) == status_before_refresh

    tool.refresh_sources_button.click()
    assert refreshed[-1] == ("many", ("data_0",))
    assert (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    ) == status_before_refresh

    def refresh_no_sources(names: Sequence[str]) -> int:
        refreshed.append(("none", tuple(names)))
        return 0

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_no_sources,
        source_label=source_label,
    )
    tool._set_source_status_text("diagnostic")
    tool.refresh_sources_button.click()
    assert refreshed[-1] == ("none", ("data_0",))
    assert tool.source_status_label.text() == "diagnostic"

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: False,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=source_label,
    )
    refreshed.clear()
    tool._refresh_source_from_button("data_0")
    tool._refresh_sources_from_button()
    assert refreshed == []

    def raise_lookup(name: str) -> bool:
        raise LookupError(name)

    def raise_lookup_label(name: str) -> str | None:
        raise LookupError(name)

    tool._set_source_refresh_callbacks(
        can_refresh_source=raise_lookup,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=raise_lookup_label,
    )
    assert not tool._source_refresh_available("data_0")
    assert tool._source_refresh_label("data_0") is None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=lambda _name: "",
    )
    assert tool._source_refresh_label("data_0") is None

    direct_item = QtWidgets.QTreeWidgetItem(["direct", "", "", ""])
    tool.source_list.addTopLevelItem(direct_item)
    direct_button = QtWidgets.QToolButton(tool.source_list)
    direct_button.setObjectName("figureComposerRefreshSourceButton")
    tool.source_list.setItemWidget(direct_item, 3, direct_button)
    assert (
        tool._source_list_row_button(direct_item, "figureComposerRefreshSourceButton")
        is direct_button
    )
    assert (
        tool._source_list_row_button(direct_item, "figureComposerRemoveSourceButton")
        is None
    )
    tool._refresh_source_controls()


def test_figure_composer_source_remove_controls_disable_used_sources(qtbot) -> None:
    image = _figure_composer_image_source("image")
    profile = _figure_composer_profile_source("profile")
    unused = xr.DataArray(
        np.array([10.0, 11.0]),
        dims=("q",),
        coords={"q": [0.0, 1.0]},
        name="unused",
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: image"),
                FigureSourceState(name="profile", label="Profile"),
                FigureSourceState(name="unused", label="Unused"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": image, "profile": profile, "unused": unused},
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    buttons = _source_remove_buttons(tool)
    assert set(buttons) == {"data_0", "profile", "unused"}
    assert buttons["unused"].property("figure_source_name") == "unused"
    assert buttons["unused"].accessibleName() == "Remove Source"
    assert not buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == "This source is used by one or more steps"
    assert not buttons["profile"].isEnabled()
    assert buttons["profile"].toolTip() == "This source is used by one or more steps"
    assert buttons["unused"].isEnabled()
    assert buttons["unused"].toolTip() == "Remove “Unused” from this figure"

    before_status = tool.tool_status
    before_source_data = tool.source_data()
    buttons["profile"].click()
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    tool._remove_source_from_button("profile")
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    assert not tool.remove_source("profile")
    assert not tool.remove_source("missing")

    buttons["unused"].click()
    assert "unused" not in tool.source_data()
    assert tool.source_status_label.text() == ""
    assert tool.source_status_label.isHidden()


def test_figure_composer_remove_source_updates_state_history_and_code(qtbot) -> None:
    image = _figure_composer_image_source("image")
    extra = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("q",),
        coords={"q": [0.0, 1.0]},
        name="extra",
    )
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: image"),
                FigureSourceState(name="extra", label="Extra"),
            ),
            operations=(operation,),
            primary_source="extra",
        ),
        source_data={"data_0": image, "extra": extra},
    )
    qtbot.addWidget(tool)
    data_changed: list[None] = []
    info_changed: list[None] = []
    tool.sigDataChanged.connect(lambda: data_changed.append(None))
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))

    assert not tool.remove_source("data_0")
    assert tool.remove_source("extra")

    assert data_changed == [None]
    assert info_changed == [None]
    assert tuple(source.name for source in tool.source_states()) == ("data_0",)
    assert set(tool.source_data()) == {"data_0"}
    assert tool.tool_status.operations == (operation,)
    assert tool.tool_status.primary_source == "data_0"
    assert tool.source_status_label.text() == ""
    buttons = _source_remove_buttons(tool)
    assert set(buttons) == {"data_0"}
    assert not buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == "This source is used by one or more steps"

    _render_figure_composer_rgba(tool)
    assert tool._operation_render_errors == {}
    namespace = _exec_generated_code(tool.generated_code(), {"data_0": image})
    assert isinstance(namespace["fig"], Figure)
    restored = FigureComposerTool(
        image,
        recipe=tool.tool_status,
        source_data=tool.source_data(),
    )
    qtbot.addWidget(restored)
    assert tuple(source.name for source in restored.source_states()) == ("data_0",)
    assert set(restored.source_data()) == {"data_0"}

    assert tool.undoable
    tool.undo()
    assert tuple(source.name for source in tool.source_states()) == ("data_0", "extra")
    assert tool.tool_status.primary_source == "extra"
    xr.testing.assert_identical(tool.source_data()["extra"], extra)
    assert tool.redoable
    tool.redo()
    assert tuple(source.name for source in tool.source_states()) == ("data_0",)
    assert set(tool.source_data()) == {"data_0"}
    assert tool.tool_status.primary_source == "data_0"
    tool._refresh_source_list()
    assert set(_source_remove_buttons(tool)) == {"data_0"}


def test_figure_composer_source_display_helpers_keep_alias_secondary() -> None:
    source = FigureSourceState(name="data_0", label="ImageTool 0: sample_map")
    assert (
        figurecomposer_sources._source_display_label(source, "data_0")
        == "ImageTool 0: sample_map"
    )
    assert (
        figurecomposer_sources._source_display_tooltip(source, "data_0")
        == "ImageTool 0: sample_map\nAlias: data_0"
    )

    duplicates = figurecomposer_sources._source_duplicate_labels(
        (
            FigureSourceState(name="data_0", label="ImageTool"),
            FigureSourceState(name="data_1", label="ImageTool"),
        )
    )
    assert duplicates == frozenset(("ImageTool",))
    assert (
        figurecomposer_sources._source_display_label(
            FigureSourceState(name="data_0", label="ImageTool"),
            "data_0",
            disambiguate=True,
        )
        == "ImageTool (data_0)"
    )


def test_figure_composer_plot_slices_selection_error_details() -> None:
    plain = str(FigureComposerPlotSlicesSelectionError())
    assert "Details:" not in plain

    detailed = str(FigureComposerPlotSlicesSelectionError("bad key"))
    assert plain in detailed
    assert "Details: bad key" in detailed


def test_figure_composer_replace_source_preserves_alias_and_generated_code(
    qtbot, monkeypatch
) -> None:
    original = _figure_composer_image_source("original")
    replacement = original.copy(
        data=np.asarray(original.data) + 10.0,
        deep=True,
    )
    replacement.name = "replacement"
    old_snapshot_id = "old-snapshot"
    new_snapshot_id = "new-snapshot"
    tool = FigureComposerTool(
        original,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="ImageTool 0: original",
                    node_uid="old-node",
                    node_snapshot_token=old_snapshot_id,
                    provenance_spec={"source": "old"},
                ),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="data_0",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "line_selection": {
                            "eV": 0.0,
                            "beta": float(original.coords["beta"].values[0]),
                        },
                        "line_x": "alpha",
                    }
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": original},
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    assert "ImageTool 0: original" in tool.operation_list.item(0).text()
    assert "ImageTool 0: original" in tool.operation_list.item(1).text()
    assert tool.step_section_buttons["sources"].text() == (
        "Sources: ImageTool 0: original"
    )

    replaced = tool.replace_source(
        "data_0",
        FigureSourceState(
            name="data_1",
            label="ImageTool 1: replacement",
            node_uid="new-node",
            node_snapshot_token=new_snapshot_id,
            provenance_spec={"source": "new"},
        ),
        replacement,
    )

    assert replaced is True
    [source] = tool.source_states()
    assert source.name == "data_0"
    assert source.label == "ImageTool 1: replacement"
    assert source.node_uid == "new-node"
    assert source.node_snapshot_token == new_snapshot_id
    assert source.provenance_spec == {"source": "new"}
    xr.testing.assert_identical(tool.source_data()["data_0"], replacement)
    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert tool.tool_status.operations[1].line_source == "data_0"
    assert "ImageTool 1: replacement" in tool.operation_list.item(0).text()
    assert "ImageTool 0: original" not in tool.operation_list.item(0).text()
    assert "ImageTool 1: replacement" in tool.operation_list.item(1).text()
    assert "ImageTool 0: original" not in tool.operation_list.item(1).text()
    assert tool.step_section_buttons["sources"].text() == (
        "Sources: ImageTool 1: replacement"
    )

    _render_figure_composer_rgba(tool)
    assert tool._operation_render_errors == {}

    captured_maps: list[tuple[xr.DataArray, ...]] = []

    def capture_plot_slices(maps, **_kwargs):
        captured_maps.append(tuple(maps))

    monkeypatch.setattr(eplt, "plot_slices", capture_plot_slices)
    exec(tool.generated_code(), {"data_0": replacement})  # noqa: S102

    assert captured_maps
    captured = captured_maps[0][0]
    xr.testing.assert_identical(
        captured,
        replacement.sel(eV=float(captured.coords["eV"])),
    )

    assert not tool.replace_source(
        "missing",
        FigureSourceState(name="data_2", label="Missing"),
        replacement,
    )
    tool._source_data["orphan"] = original
    assert tool.replace_source(
        "orphan",
        FigureSourceState(name="data_2", label="Orphan"),
        original,
    )
    assert tuple(source.name for source in tool.source_states()) == (
        "data_0",
        "orphan",
    )
    assert tool.source_states()[-1].label == "Orphan"


def test_figure_composer_text_helpers_parse_user_inputs() -> None:
    assert figurecomposer_text._float_pair_from_text("") is None
    assert figurecomposer_text._float_pair_from_text("1, 2.5") == (1.0, 2.5)
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1, bad")

    assert figurecomposer_text._plot_limit_from_text("") is None
    assert figurecomposer_text._plot_limit_from_text("1.5") == 1.5
    assert figurecomposer_text._plot_limit_from_text("None") is None
    assert figurecomposer_text._plot_limit_from_text("[2]") == 2.0
    assert figurecomposer_text._plot_limit_from_text("(1, 2)") == (1.0, 2.0)
    assert figurecomposer_text._plot_limit_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._plot_limit_from_text("(None, 2)") == (None, 2.0)
    assert figurecomposer_text._limit_pair_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._limit_pair_from_text("") is None
    assert figurecomposer_text._format_plot_limit((0.0, None)) == "0, None"
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 2, 3)")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 'bad')")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1,)")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1, 'bad')")

    assert figurecomposer_text._float_tuple_from_text("1, 2") == (1.0, 2.0)
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="numbers"):
        figurecomposer_text._float_tuple_from_text("1, bad")
    assert figurecomposer_text._literal_sequence_from_text("") == ()
    assert figurecomposer_text._literal_sequence_from_text("[1, 2]") == (1, 2)
    assert figurecomposer_text._literal_sequence_from_text("(1)") == (1,)
    assert figurecomposer_text._literal_sequence_from_text("'x'") == ("x",)
    assert figurecomposer_text._literal_sequence_from_text("1, 2") == (1, 2)

    assert figurecomposer_text._string_tuple_from_text("") == ()
    assert figurecomposer_text._string_tuple_from_text("alpha, beta") == (
        "alpha",
        "beta",
    )
    assert figurecomposer_text._string_tuple_from_text("('alpha')") == ("alpha",)
    assert figurecomposer_text._string_tuple_from_text("['alpha', 2]") == (
        "alpha",
        "2",
    )
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="text"):
        figurecomposer_text._string_tuple_from_text("(1)")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb") == ("a", "b")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb", preserve_empty=True) == (
        "a",
        "",
        "b",
    )

    assert figurecomposer_text._dict_from_text("") == {}
    assert figurecomposer_text._dict_from_text(
        "a=1, b=slice(0, 2)", allow_slice=True
    ) == {
        "a": 1,
        "b": slice(0, 2),
    }
    assert figurecomposer_text._dict_from_text("{'a': 1}") == {"a": 1}
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=object()")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{1, 2}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("**{'a': 1}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("{**{'a': 1}}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{alpha: 1}")

    assert figurecomposer_text._format_pair(None) == ""
    assert figurecomposer_text._format_limit_pair(None) == ""
    assert figurecomposer_text._format_plot_limit(2.0) == "2"
    assert (
        figurecomposer_text._format_dim_sizes(
            xr.DataArray(np.zeros((2, 3)), dims=("alpha", "beta"))
        )
        == "alpha=2, beta=3"
    )
    assert figurecomposer_text._format_axes_tuple((), nrows=2, ncols=2) == "none"
    assert figurecomposer_text._selection_value_count(np.arange(6).reshape(2, 3)) == 6
    assert figurecomposer_text._selection_value_count([1, 2]) == 2
    assert figurecomposer_text._selection_value_count(slice(None)) is None


def test_figure_composer_source_helpers_cover_selection_contract() -> None:
    unnamed = xr.DataArray(np.arange(2.0), dims=("x",), name=None)
    invalid_name = xr.DataArray(np.arange(2.0), dims=("x",), name="bad name")
    assert figurecomposer_sources._source_name(unnamed) == "data"
    assert figurecomposer_sources._source_label(unnamed) == "data"
    assert figurecomposer_sources._source_name(invalid_name) == "data"
    assert figurecomposer_sources._source_display_tooltip(None, "data_0") == (
        "Alias: data_0"
    )
    with pytest.raises(ValueError, match="not a valid variable"):
        figurecomposer_sources._valid_source_variable("bad name")
    assert figurecomposer_plot_slices._source_names(
        FigureOperationState.plot_slices(
            label="mapped",
            sources=(),
            map_selections=(FigureDataSelectionState(source="extra"),),
        )
    ) == ("extra",)

    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
        name="map",
    )
    assert figurecomposer_sources._available_source_dims(
        {"data": data}, ("missing", "data")
    ) == ["x", "y"]
    assert (
        figurecomposer_sources._selected_data(
            {"data": data}, FigureDataSelectionState(source="missing")
        )
        is None
    )
    selected = figurecomposer_sources._selected_data(
        {"data": data},
        FigureDataSelectionState(
            source="data",
            isel={"x": {"kind": "slice", "start": 0, "stop": 1}},
            qsel={"y": 20.0},
            mean_dims=("x",),
        ),
    )
    assert selected is not None
    assert selected.ndim == 0
    assert float(selected) == 1.0

    selected_multi_mean = figurecomposer_sources._selected_data(
        {"data": data},
        FigureDataSelectionState(source="data", mean_dims=("x", "y")),
    )
    assert selected_multi_mean is not None
    assert selected_multi_mean.ndim == 0

    nonuniform_public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    nonuniform_internal = erlab.interactive.imagetool.slicer.make_dims_uniform(
        nonuniform_public
    )
    assert nonuniform_internal.dims == ("alpha", "eV", "sample_temp_idx")
    assert figurecomposer_sources._available_source_dims(
        {"data": nonuniform_internal}, ("data",)
    ) == ["alpha", "eV", "sample_temp"]
    nonuniform_selected = figurecomposer_sources._selected_data(
        {"data": nonuniform_internal},
        FigureDataSelectionState(
            source="data",
            isel={"sample_temp": {"kind": "slice", "start": 1, "stop": 3}},
            mean_dims=("sample_temp",),
        ),
    )
    assert nonuniform_selected is not None
    xr.testing.assert_identical(
        nonuniform_selected,
        nonuniform_public.isel(sample_temp=slice(1, 3)).qsel.mean("sample_temp"),
    )

    assert (
        figurecomposer_sources._middle_coord_value(
            xr.DataArray([], dims=("x",), coords={"x": []}), "x"
        )
        is None
    )
    assert (
        figurecomposer_sources._middle_coord_value(
            xr.DataArray([1, 2], dims=("x",), coords={"x": ["a", "b"]}), "x"
        )
        is None
    )


def test_figure_composer_opens_plot_slices_selection_on_nonuniform_data(
    qtbot,
) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        map_selections=(
            FigureDataSelectionState(source="data", isel={"sample_temp": 1}),
        ),
    )

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    assert tool.operation_list.count() == 1
    assert "sample_temp_idx" not in tool.generated_code()


def test_imagetool_main_image_seeds_nonuniform_plot_slices_selection(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    tool = erlab.interactive.itool(public, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(tool)

    tool.slicer_area.set_value(axis=0, value=30.0, cursor=0)
    operation = tool.slicer_area.images[2].figure_composer_operation(source_name="data")

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.map_selections == ()
    assert operation.slice_dim == "sample_temp"
    assert operation.slice_values == (30.0,)
    assert operation.slice_kwargs == {}
    assert "sample_temp_idx" not in operation.model_dump_json()


def test_imagetool_line_profile_seeds_public_nonuniform_coordinate(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    source_tool = erlab.interactive.itool(public, manager=False, execute=False)
    assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(source_tool)

    operation = source_tool.slicer_area.profiles[0].figure_composer_operation(
        source_name="data"
    )

    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_x == "sample_temp"
    assert "sample_temp_idx" not in operation.model_dump_json()

    composer = FigureComposerTool.from_sources(
        {"data": source_tool.slicer_area._tool_source_parent_data()},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(composer)

    figure = plt.figure()
    try:
        figurecomposer_rendering._render_into_figure(
            composer, figure, sync_visible=False
        )
        assert composer._operation_render_errors == {}
        assert any(axis.lines for axis in figure.axes)
    finally:
        plt.close(figure)


def test_imagetool_main_image_seeds_plot_slices_selection_with_spaced_dim(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("Track Shift", "kx", "ky"),
        coords={
            "Track Shift": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0],
            "ky": [0.0, 1.0],
        },
        name="map",
    )
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(tool)

    tool.slicer_area.set_value(axis=0, value=1.0, cursor=0)
    operation = tool.slicer_area.images[2].figure_composer_operation(source_name="data")

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.map_selections == ()
    assert operation.slice_dim == "Track Shift"
    assert operation.slice_values == (1.0,)
    assert operation.slice_kwargs == {}


def test_imagetool_rejects_uneditable_plot_slices_selection(qtbot) -> None:
    tool = erlab.interactive.imagetool.ImageTool(_unsupported_plot_slices_data())
    qtbot.addWidget(tool)

    _set_unsupported_plot_slices_cursor_state(tool)

    with pytest.raises(
        FigureComposerPlotSlicesSelectionError,
        match="Cannot plot when more than one dimension",
    ):
        tool.slicer_area.images[0].figure_composer_operation(source_name="data")


def test_imagetool_plot_slices_selection_warning_shows_error_detail(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    messages: list[tuple[QtWidgets.QWidget | None, str, str]] = []

    def warning(
        parent: QtWidgets.QWidget | None, title: str, text: str
    ) -> QtWidgets.QMessageBox.StandardButton:
        messages.append((parent, title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", warning)

    imagetool_plot_items.ItoolPlotItem._show_plot_slices_selection_error(
        parent,
        FigureComposerPlotSlicesSelectionError(
            "Cannot plot when more than one dimension"
        ),
    )

    assert len(messages) == 1
    assert messages[0][0] is parent
    assert messages[0][1] == "Cannot Create Plot Slices Figure"
    assert "Details:" in messages[0][2]
    assert "Cannot plot when more than one dimension" in messages[0][2]


def test_figure_composer_raw_sources_use_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
    )

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    shape = figurecomposer_plot_slices._plot_slices_shape(tool, operation)
    assert "sample_temp" in shape.source_text
    assert "sample_temp_idx" not in shape.source_text
    shape_item = tool.source_list.topLevelItem(0)
    assert shape_item is not None
    shape_label = tool.source_list.itemWidget(shape_item, 2)
    assert isinstance(shape_label, QtWidgets.QLabel)
    assert "sample_temp_idx" not in shape_label.text()


def test_figure_composer_source_inspector_is_sources_tab_compact(
    qtbot, monkeypatch
) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.1], "kx": [0.0, 0.5, 1.0]},
        attrs={"sample": "TiSe2", "long": "x" * 200},
        name="map",
    )
    second = xr.DataArray(
        np.arange(4.0),
        dims=("delay",),
        coords={"delay": [0.0, 1.0, 2.0, 3.0]},
        attrs={"scan": 12},
        name="profile",
    )
    format_calls: list[tuple[tuple[str, ...], dict[str, typing.Any]]] = []

    def fake_format_darr_html(
        data: xr.DataArray,
        **kwargs: typing.Any,
    ) -> str:
        format_calls.append((tuple(str(dim) for dim in data.dims), kwargs))
        return "<p>formatted details</p>"

    monkeypatch.setattr(
        erlab.utils.formatting,
        "format_darr_html",
        fake_format_darr_html,
    )
    tool = FigureComposerTool.from_sources(
        {"map": first, "profile": second},
        sources=(
            FigureSourceState(name="map", label="Map source", node_uid="node-map"),
            FigureSourceState(name="profile", label="Profile source"),
        ),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("map",)),),
        primary_source="map",
    )
    qtbot.addWidget(tool)

    assert tool.source_inspector.parentWidget() is tool.step_sources_page
    assert not hasattr(tool, "step_detail_splitter")
    assert tool.source_inspector.source_name() == "map"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "map"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("eV", "kx")
    assert tool.source_inspector.property("figureComposerSourceDtype") == "float64"
    assert tool.source_inspector.property("figureComposerSourceUsedByStep") is True
    assert not tool.source_inspector.details_button.isChecked()
    assert not tool.source_inspector.details_scroll.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.details_html() == "<p>formatted details</p>"
    assert format_calls[-1] == (
        ("eV", "kx"),
        {"show_size": True, "show_summary": False, "load_values": False},
    )
    assert (
        tool.source_list.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )

    tool.source_inspector.details_button.setChecked(True)
    assert tool.source_inspector.details_scroll.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.property("figureComposerSourceDetailsExpanded") is True

    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    tool.source_list.setCurrentItem(second_item)
    assert tool.source_inspector.source_name() == "profile"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "profile"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("delay",)


def test_figure_composer_source_inspector_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("data",)),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    dims = tool.source_inspector.property("figureComposerSourceDims")
    assert dims == ("alpha", "eV", "sample_temp")
    assert "sample_temp_idx" not in tool.source_inspector.details_html()
    assert "sample_temp" in tool.source_inspector.details_html()


def test_figure_composer_source_inspector_tracks_source_tab_selection(qtbot) -> None:
    first = xr.DataArray([1.0, 2.0], dims=("x",), name="first")
    second = xr.DataArray([3.0, 4.0], dims=("x",), name="second")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.line(label="first line", source="first"),
            FigureOperationState.line(label="second line", source="second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentRow(1)
    assert tool.source_inspector.source_name() == "second"
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    tool.source_list.setCurrentItem(first_item)
    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentRow(0)
    assert tool.source_inspector.source_name() == "first"


def test_figure_composer_source_inspector_updates_without_render_or_history(
    qtbot, monkeypatch
) -> None:
    x_data = xr.DataArray([0.0, 1.0, 2.0], dims=("x",), name="x")
    y_data = xr.DataArray([1.0, 2.0], dims=("y",), name="y")
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(source="x", kind="data"),
            "method_plot_y": FigureMethodPlotValueState(source="y", kind="data"),
        }
    )
    tool = FigureComposerTool.from_sources(
        {"x": x_data, "y": y_data},
        sources=(
            FigureSourceState(name="x", label="x"),
            FigureSourceState(name="y", label="y"),
        ),
        operations=(operation,),
        primary_source="y",
    )
    qtbot.addWidget(tool)
    render_calls: list[object] = []
    write_calls: list[object] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(tool, "_write_state", lambda: write_calls.append(None))

    y_item = tool.source_list.topLevelItem(1)
    assert y_item is not None
    tool.source_list.setCurrentItem(y_item)

    assert render_calls == []
    assert write_calls == []
    assert tool.source_inspector.source_name() == "y"
    assert tool.source_inspector.property("figureComposerSourceUsedByStep") is True
    assert tool.source_inspector.property("figureComposerSourceAlias") == "y"


def test_figure_composer_source_inspector_helper_edges(qtbot) -> None:
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("x",),
        coords={"x": [0.0, 1.0], 7: ("x", [10.0, 20.0])},
        attrs={"note": "kept"},
        name="source",
    )
    source = FigureSourceState(name="data", label="display")

    missing_metadata = figurecomposer_source_inspector.source_metadata_tooltip(
        source, "data", None
    )
    assert "Alias: data" in missing_metadata
    assert "unavailable" in missing_metadata

    metadata = figurecomposer_source_inspector.source_metadata_tooltip(
        source, "data", data
    )
    assert "Dims: x: 2" in metadata
    assert "Shape: 2" in metadata
    assert "Dtype: float64" in metadata

    assert "source is missing" in figurecomposer_source_inspector.source_value_tooltip(
        None, ("data", None), axis="x"
    )
    assert "Use DataArray values" in (
        figurecomposer_source_inspector.source_value_tooltip(
            data, ("data", None), axis="y"
        )
    )
    assert "Coordinate 'missing' is not available" in (
        figurecomposer_source_inspector.source_value_tooltip(
            data, ("coord", "missing"), axis="x"
        )
    )
    numeric_coord_tooltip = figurecomposer_source_inspector.source_value_tooltip(
        data, ("coord", "7"), axis="x"
    )
    assert "Coord dims: x" in numeric_coord_tooltip

    inspector = figurecomposer_source_inspector.SourceInspectorWidget()
    qtbot.addWidget(inspector)
    inspector.set_context(
        source_name=None,
        source_state=None,
        data=None,
        operation_source_names=(),
    )
    assert inspector.source_name() is None
    assert inspector.property("figureComposerSourceDims") == ()
    assert inspector.property("figureComposerSourceUsedByStep") is False
    assert not inspector.details_button.isEnabled()

    inspector.set_context(
        source_name="data",
        source_state=source,
        data=None,
        operation_source_names=("data",),
    )
    assert inspector.source_name() == "data"
    assert inspector.property("figureComposerSourceUsedByStep") is True
    assert inspector.property("figureComposerSourceDims") == ()
    assert not inspector.details_button.isEnabled()


def test_figure_composer_redraw_and_preview_cache_edges(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    render_calls: list[dict[str, object]] = []
    single_shot_calls: list[tuple[int, Callable[[], None]]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda _tool, **kwargs: render_calls.append(kwargs),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda _owner, delay, callback: single_shot_calls.append((delay, callback)),
    )

    tool.auto_redraw_check.setChecked(False)
    assert not tool._maybe_redraw_plot(show_window=True)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool._queue_preview_render_update()
    assert tool._auto_redraw_dirty
    assert single_shot_calls == []

    tool._run_queued_preview_render_update(tool._preview_render_update_generation)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool.auto_redraw_check.setChecked(True)
    assert render_calls == [{}]
    assert not tool._auto_redraw_dirty

    tool._redraw_plot_requested()
    assert render_calls[-1] == {}

    assert not tool._saved_tool_window_visible(xr.Dataset())
    assert tool._saved_tool_window_visible(xr.Dataset(attrs={"tool_visible": True}))

    tool.auto_redraw_check.setChecked(False)
    visible_ds = xr.Dataset(attrs={"tool_visible": True})
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls == []
    tool.auto_redraw_check.setChecked(True)
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls[-1][0] == 0
    single_shot_calls[-1][1]()
    assert render_calls[-1] == {"show_window": True}

    assert tool._persisted_preview_cache_pixmap() is None
    preview = QtGui.QPixmap(
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width() + 20,
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.height() + 20,
    )
    preview.fill(QtGui.QColor("red"))
    tool._preview_pixmap_cache = preview
    persisted = tool._persisted_preview_cache_pixmap()
    assert persisted is not None
    assert (
        persisted.width()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width()
    )
    saved = tool._append_persistence_payload(xr.Dataset())
    assert figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR in saved.attrs

    restored = FigureComposerTool(data)
    qtbot.addWidget(restored)
    restored._restore_persisted_preview_cache(
        xr.Dataset(
            attrs={figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR: "bad"}
        )
    )
    assert restored._preview_pixmap_cache is None
    restored._restore_persisted_preview_cache(saved)
    assert restored._preview_pixmap_cache is not None
    assert (
        restored._preview_pixmap_stale
        == saved.attrs[figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_STALE_ATTR]
    )


def test_figure_composer_source_inspector_target_fallbacks(qtbot) -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="saved", label="Saved"),),
            primary_source="missing",
        ),
        source_data={},
    )
    qtbot.addWidget(tool)

    tool._source_data.clear()
    tool._select_source_list_row_silent(None)
    assert tool._default_source_inspector_target() == "saved"
    tool._refresh_source_inspector()
    assert tool.source_inspector.source_name() == "saved"

    tool._select_source_list_row_silent(None)
    assert tool.source_list.currentItem() is None


def test_figure_composer_line_profile_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(8.0).reshape(4, 2),
        dims=("sample_temp", "alpha"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
        },
        name="profile",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.line(
        label="line",
        source="data",
    ).model_copy(update={"line_selection": {"sample_temp": 15.0}, "line_x": "alpha"})

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="profile"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    coordinate_names = figurecomposer_line_profile._available_line_coordinate_names(
        tool, operation
    )
    assert "alpha" in coordinate_names
    assert "sample_temp_idx" not in coordinate_names
    line_items = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(line_items) == 1
    assert line_items[0].dims == ("alpha",)


def test_manager_default_figure_seed_uses_public_nonuniform_dims(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)

    with manager_context() as manager:
        operation = manager._make_figure_operations_for_sources(
            {"data": internal},
            setup=FigureSubplotsState(),
        )[0]

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "sample_temp"
    assert "sample_temp_idx" not in operation.model_dump_json()


def test_figure_composer_line_style_helpers_update_recipe(qtbot) -> None:
    assert "" not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert " " not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "None" not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "none" in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "" not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert " " not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert "None" not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert "none" in figurecomposer_line_style.LINE_MARKER_OPTIONS

    assert figurecomposer_line_style.color_kw_value_from_text("") is None
    assert figurecomposer_line_style.color_kw_value_from_text("tab:blue") == "tab:blue"
    assert figurecomposer_line_style.color_kw_value_from_text("['red', 'blue']") == (
        "red",
        "blue",
    )
    assert figurecomposer_line_style.color_kw_value_from_text("('red', 'blue')") == (
        "red",
        "blue",
    )
    assert figurecomposer_line_style.color_kw_value_from_text("[bad") == "[bad"

    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    combo = QtWidgets.QComboBox(parent)
    figurecomposer_line_style.configure_style_combo(
        combo,
        figurecomposer_line_style.LINE_STYLE_OPTIONS,
        None,
    )
    assert combo.itemData(0) is None
    assert figurecomposer_line_style.style_combo_value(combo) is None
    figurecomposer_line_style.set_style_combo_value(combo, "")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, " ")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, "None")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, "custom-dash")
    assert figurecomposer_line_style.style_combo_value(combo) == "custom-dash"
    assert combo.itemText(combo.count() - 1) == "custom-dash"

    spinbox = figurecomposer_line_style.optional_positive_spinbox(None, parent=parent)
    assert (
        figurecomposer_line_style.optional_positive_spinbox_value(spinbox.value())
        is None
    )
    spinbox.setValue(2.5)
    assert (
        figurecomposer_line_style.optional_positive_spinbox_value(spinbox.value())
        == 2.5
    )

    operation = FigureOperationState.plot_slices(
        label="line-slices",
        sources=("data",),
    ).model_copy(update={"line_kw": {"lw": "2", "custom": 1}, "cmap": "magma"})
    assert figurecomposer_line_style.line_kw_text(operation, "linewidth", "lw") == "2"
    assert figurecomposer_line_style.line_kw_float(operation, "linewidth", "lw") == 2.0
    assert figurecomposer_line_style.extra_line_kw(operation) == {"custom": 1}
    bad_operation = operation.model_copy(update={"line_kw": {"linewidth": "bad"}})
    assert figurecomposer_line_style.line_kw_float(bad_operation, "linewidth") is None

    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    tool._updating_controls = True
    figurecomposer_line_style.update_current_line_kw(
        tool, "color", "red", clear_stale_cmap=True
    )
    tool._updating_controls = False
    assert tool.tool_status.operations[0].cmap == "magma"

    figurecomposer_line_style.update_current_line_kw(
        tool,
        "color",
        "red",
        aliases=("c",),
        clear_stale_cmap=True,
    )
    updated = tool.tool_status.operations[0]
    assert updated.line_kw["color"] == "red"
    assert "c" not in updated.line_kw
    assert updated.cmap is None

    tool._updating_controls = True
    figurecomposer_line_style.update_current_extra_line_kw(tool, {"zorder": 5})
    tool._updating_controls = False
    assert "zorder" not in tool.tool_status.operations[0].line_kw

    figurecomposer_line_style.update_current_extra_line_kw(tool, {"zorder": 5})
    assert tool.tool_status.operations[0].line_kw == {
        "lw": "2",
        "color": "red",
        "zorder": 5,
    }


def test_figure_composer_step_editor_section_headers_are_native_subgroups(
    qtbot,
) -> None:
    image = _figure_composer_image_source("image")
    line_map = _figure_composer_line_slice_source("line_map")
    recipe = FigureRecipeState(
        sources=(
            FigureSourceState(name="image", label="image"),
            FigureSourceState(name="line_map", label="line map"),
        ),
        operations=(
            FigureOperationState.line(label="profiles", source="line_map").model_copy(
                update={"line_iter_dim": "eV", "line_color_mode": "coordinate"}
            ),
            FigureOperationState.plot_slices(
                label="line slices",
                sources=("line_map",),
                slice_dim="eV",
                slice_values=(-0.5, 0.5),
            ).model_copy(update={"gradient": True}),
            FigureOperationState.plot_slices(
                label="image slices",
                sources=("image",),
                slice_dim="eV",
                slice_values=(0.0,),
            ).model_copy(update={"colorbar": "right"}),
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
                axes=FigureAxesSelectionState(axes=((0, 0),)),
            ),
        ),
        primary_source="image",
    )
    tool = FigureComposerTool(
        image,
        recipe=recipe,
        source_data={"image": image, "line_map": line_map},
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._update_operation_editor()
    tool._select_step_section("selection")
    line_selection_page = tool.step_editor_stack.currentWidget()
    assert line_selection_page is not None
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionDataSection"
    )
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionProfilesSection"
    )
    tool._select_step_section("view")
    line_view_page = tool.step_editor_stack.currentWidget()
    assert line_view_page is not None
    assert (
        line_view_page.findChild(
            QtWidgets.QWidget, "figureComposerLineViewPlacementSection"
        )
        is None
    )
    tool._select_step_section("style")
    line_style_page = tool.step_editor_stack.currentWidget()
    assert line_style_page is not None
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLegendSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleColorSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLineSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleFillSection")
    tool._select_step_section("other")
    line_other_page = tool.step_editor_stack.currentWidget()
    assert line_other_page is not None
    assert tool.step_section_buttons["other"].property("section_title") == "Transform"
    assert (
        line_other_page.findChild(
            QtWidgets.QWidget, "figureComposerLineOtherTransformSection"
        )
        is None
    )

    tool.operation_list.setCurrentRow(1)
    tool._update_operation_editor()
    tool._select_step_section("selection")
    line_slices_selection_page = tool.step_editor_stack.currentWidget()
    assert line_slices_selection_page is not None
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionDimensionsSection",
    )
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionValuesSection",
    )
    tool._select_step_section("view")
    line_slices_view_page = tool.step_editor_stack.currentWidget()
    assert line_slices_view_page is not None
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewPanelsSection"
    )
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewAxesSection"
    )
    tool._select_step_section("colors")
    line_slices_colors_page = tool.step_editor_stack.currentWidget()
    assert line_slices_colors_page is not None
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleLegendSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleLineSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleFillSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStylePanelOverridesSection",
    )
    tool._select_step_section("transform")
    line_slices_transform_page = tool.step_editor_stack.currentWidget()
    assert line_slices_transform_page is not None
    assert (
        line_slices_transform_page.objectName()
        == "figureComposerPlotSlicesTransformPage"
    )
    assert (
        line_slices_transform_page.findChild(
            QtWidgets.QWidget, "figureComposerPlotSlicesColorsTransformSection"
        )
        is None
    )

    tool.operation_list.setCurrentRow(2)
    tool._update_operation_editor()
    tool._select_step_section("colors")
    image_slices_colors_page = tool.step_editor_stack.currentWidget()
    assert image_slices_colors_page is not None
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsImageColorSection",
    )
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsColorbarSection",
    )
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsPanelOverridesSection",
    )

    tool.operation_list.setCurrentRow(3)
    tool._update_operation_editor()
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    assert method_page is not None
    _assert_step_editor_section(method_page, "figureComposerMethodCallSection")
    _assert_step_editor_section(method_page, "figureComposerMethodValuesSection")
    _assert_step_editor_section(method_page, "figureComposerMethodAdvancedSection")


def test_figure_composer_norm_helpers_cover_structured_and_custom_norms(
    monkeypatch,
) -> None:
    assert figurecomposer_norms._norm_module_prefix("Normalize") == "mcolors"
    assert figurecomposer_norms._norm_module_prefix("CenteredPowerNorm") == "eplt"
    assert figurecomposer_norms._norm_combo_choices("CustomNorm")[-1] == "CustomNorm"
    assert figurecomposer_norms._norm_kwarg_fields("CustomNorm") == ("gamma",)
    assert figurecomposer_norms._norm_float_value(None) is None
    assert figurecomposer_norms._norm_updates_from_kwargs(
        {"gamma": 2, "halfrange": 1, "clip": None, "extra": "value"}
    ) == {
        "norm_gamma": 2.0,
        "halfrange": 1.0,
        "norm_clip": None,
        "norm_kwargs": {"extra": "value"},
    }
    assert figurecomposer_norms._cmap_base_and_reverse("magma_r") == ("magma", True)
    assert figurecomposer_norms._cmap_with_reverse("", False) is None
    assert figurecomposer_norms._cmap_with_reverse("magma_r", False) == "magma"

    power_from_plot_gamma = FigureOperationState.plot_slices(
        label="power",
        sources=("data",),
    ).model_copy(update={"gamma": 0.5})
    assert (
        figurecomposer_norms._norm_constructor_kwargs(power_from_plot_gamma)["gamma"]
        == 0.5
    )

    two_slope = FigureOperationState.plot_slices(
        label="two-slope",
        sources=("data",),
    ).model_copy(update={"norm_name": "TwoSlopeNorm"})
    assert figurecomposer_norms._norm_constructor_kwargs(two_slope)["vcenter"] == 0.0

    custom_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    class CustomNorm:
        def __init__(self, *args, **kwargs) -> None:
            custom_calls.append((args, kwargs))

    monkeypatch.setattr(eplt, "CustomNorm", CustomNorm, raising=False)
    custom = FigureOperationState.plot_slices(
        label="custom",
        sources=("data",),
    ).model_copy(
        update={
            "norm_name": "CustomNorm",
            "norm_gamma": 2.0,
            "norm_kwargs": {"alpha": 3},
        }
    )
    figurecomposer_norms._norm_object(custom)
    assert custom_calls[-1] == ((2.0,), {"alpha": 3})
    assert figurecomposer_norms._norm_code(custom) == "eplt.CustomNorm(2.0, alpha=3)"

    custom_no_gamma = custom.model_copy(update={"norm_gamma": None})
    figurecomposer_norms._norm_object(custom_no_gamma)
    assert custom_calls[-1] == ((), {"alpha": 3})
    assert figurecomposer_norms._norm_code(custom_no_gamma) == (
        "eplt.CustomNorm(alpha=3)"
    )


def test_figure_composer_axes_helpers_parse_safe_expressions() -> None:
    axs = np.empty((2, 2), dtype=object)
    axs[0, 0] = "ax00"
    axs[0, 1] = "ax01"
    axs[1, 0] = "ax10"
    axs[1, 1] = "ax11"

    assert figurecomposer_axes._compact_axes_code(()) is None
    assert figurecomposer_axes._compact_axes_iterable_code((), nrows=2, ncols=2) is None
    assert figurecomposer_axes._axes_expression_value("axs", axs) is axs
    assert figurecomposer_axes._axes_expression_value("ax", axs) == "ax00"
    assert figurecomposer_axes._axes_expression_value("axs[-1, 0]", axs) == "ax10"
    assert figurecomposer_axes._axes_expression_value(
        "axs[[0, 1], 0]", axs
    ).tolist() == [
        "ax00",
        "ax10",
    ]
    with pytest.raises(ValueError, match="Unsupported axes name"):
        figurecomposer_axes._axes_expression_value("figure", axs)
    with pytest.raises(ValueError, match="integer indices"):
        figurecomposer_axes._axes_expression_value("axs[1.5]", axs)
    with pytest.raises(ValueError, match="integer indices"):
        figurecomposer_axes._axes_expression_value("axs[-None]", axs)
    with pytest.raises(ValueError, match="Unsupported axes expression"):
        figurecomposer_axes._axes_expression_value("axs + axs", axs)


def test_figure_composer_line_transform_helpers_cover_edge_cases() -> None:
    profile = xr.DataArray(
        [0.0, 0.0],
        dims=("kx",),
        coords={"kx": [0.0, 1.0], "center": 2.0},
    )
    assert _line_transform.line_normalize_from_text("unknown") == "none"
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        _line_transform.normalize_line_data(profile, "max")
    assert _line_transform.line_transform_values((), 0, default=1.0) == ()
    assert _line_transform.line_transform_values((), 2, default=1.0) == (1.0, 1.0)
    assert _line_transform.line_transform_values((2.0,), 2, default=1.0) == (
        2.0,
        2.0,
    )
    assert _line_transform.line_transform_values((1.0, 2.0), 2, default=1.0) == (
        1.0,
        2.0,
    )
    with pytest.raises(ValueError, match="one value or one per profile"):
        _line_transform.line_transform_values((1.0, 2.0, 3.0), 2, default=1.0)

    coordinate_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(update={"line_offset_source": "coordinate"})
    with pytest.raises(ValueError, match="One profile per"):
        _line_transform.line_offset_coordinate_name(coordinate_operation)

    associated_operation = coordinate_operation.model_copy(
        update={"line_offset_source": "associated"}
    )
    with pytest.raises(ValueError, match="require a coordinate"):
        _line_transform.line_offset_coordinate_name(associated_operation)
    with pytest.raises(ValueError, match="no coordinate"):
        _line_transform.profile_scalar_coord_value(profile, "missing")
    with pytest.raises(ValueError, match="not scalar"):
        _line_transform.profile_scalar_coord_value(profile, "kx")

    offset_operation = associated_operation.model_copy(
        update={"line_offset_coord": "center", "line_offset_scale": 0.5}
    )
    assert _line_transform.line_offsets_for_profiles(offset_operation, (profile,)) == (
        1.0,
    )

    code_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(
        update={
            "line_scales": (2.0,),
            "line_offsets": (1.0,),
            "line_normalize": "none",
        }
    )
    assert _line_transform.profile_transform_code_lines(code_operation) == [
        "profiles = [",
        "    1.0 + 2.0 * profile",
        "    for profile in profiles",
        "]",
    ]
    normalized_operation = code_operation.model_copy(update={"line_normalize": "max"})
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        _line_transform.profile_transform_code_lines(
            normalized_operation,
            profiles=(profile,),
        )

    stack_data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("cut", "kx"),
        coords={"cut": [1.0, 2.0], "kx": [-1.0, 0.0, 1.0]},
    )
    stack_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(
        update={
            "line_iter_dim": "cut",
            "line_normalize": "mean",
            "line_scales": (0.5,),
            "line_offset_source": "coordinate",
            "line_offset_scale": 2.0,
        }
    )
    assert _line_transform.profile_stack_transform_code(
        stack_operation,
        data_name="profile_data",
        line_data=stack_data,
    ) == (
        "2.0 * profile_data['cut'] + "
        "0.5 * (profile_data / profile_data.mean('kx', skipna=True))"
    )
    multi_dim_stack = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("cut", "eV", "kx"),
        coords={"cut": [1.0, 2.0], "eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0]},
    )
    assert (
        _line_transform.profile_stack_transform_code(
            stack_operation,
            data_name="profile_data",
            line_data=multi_dim_stack,
        )
        is None
    )
    stack_code_operation = code_operation.model_copy(update={"line_iter_dim": "cut"})
    assert (
        _line_transform.profile_stack_transform_code(
            stack_code_operation,
            data_name="profile_data",
            line_data=stack_data,
        )
        == "1.0 + 2.0 * profile_data"
    )
    manual_multi_offset = code_operation.model_copy(
        update={"line_scales": (), "line_offsets": (1.0, 2.0)}
    )
    assert (
        _line_transform.profile_stack_transform_code(
            manual_multi_offset,
            data_name="profile_data",
            line_data=stack_data,
        )
        is None
    )
    assert _line_transform.profile_transform_code_lines(manual_multi_offset) == [
        "profiles = [",
        "    offset + profile",
        "    for profile, offset in zip(",
        "        profiles,",
        "        [1.0, 2.0],",
        "        strict=True,",
        "    )",
        "]",
    ]
    index_offset = code_operation.model_copy(
        update={
            "line_scales": (),
            "line_offsets": (),
            "line_offset_source": "index",
            "line_offset_scale": 3.0,
        }
    )
    assert (
        _line_transform.profile_stack_transform_code(
            index_offset,
            data_name="profile_data",
            line_data=stack_data,
        )
        is None
    )
    assert _line_transform.profile_transform_code_lines(index_offset)[4] == (
        "        [3.0 * float(index) for index in range(len(profiles))],"
    )


def test_figure_composer_codegen_axes_helpers_cover_invalid_targets(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="outside the current layout"):
        figurecomposer_code._axes_sequence_code(
            tool, FigureAxesSelectionState(axes=((1, 0),))
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(tool, FigureAxesSelectionState(axes=()))

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)

    assert (
        figurecomposer_code._axes_code(
            grid_tool,
            FigureAxesSelectionState(axes_ids=("axis-a", "axis-b")),
            for_plot_slices=False,
        )
        == "[ax0, ax1]"
    )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(
            grid_tool, FigureAxesSelectionState(axes_ids=())
        )


def test_figure_composer_gridspec_helpers_cover_naming_and_region_edges() -> None:
    invalid_child = FigureGridSpecGridState(
        grid_id="child",
        nrows=1,
        ncols=1,
        span=None,
        axes=(
            FigureGridSpecAxesState(
                axes_id="nested-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                label="custom_axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
        child_grids=(invalid_child,),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(root=root),
    )

    assert (
        figurecomposer_gridspec._gridspec_axis_variable_name_error(setup, "axis-b", "")
        == ""
    )
    assert (
        figurecomposer_gridspec._gridspec_axis_variable_name_error(
            setup, "axis-b", "ax0"
        )
        == "Variable name conflicts with an autogenerated name."
    )
    overlap_child = FigureGridSpecGridState(
        grid_id="overlap-child",
        nrows=1,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    overlap_root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=1,
        child_grids=(overlap_child,),
    )
    assert figurecomposer_gridspec._gridspec_region_overlaps(
        overlap_root,
        FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    assert tuple(figurecomposer_gridspec._iter_valid_axes_with_grid(root)) == (
        (root, root.axes[0]),
        (root, root.axes[1]),
    )
    assert (
        figurecomposer_gridspec._axis_code_name_for_validation(
            setup,
            "axis-b",
            FigureGridSpecAxesState(
                axes_id="missing-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            0,
            reserved_names=(),
        )
        == "ax0_1"
    )
    assert (
        figurecomposer_gridspec._unique_axis_code_name("ax", {"ax", "ax_1"}) == "ax_2"
    )


def test_figure_composer_editor_control_adapters_cover_mixed_states(qtbot) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    edit = QtWidgets.QLineEdit(parent)
    line_adapter = _editor_controls.LineEditControlAdapter(edit)
    assert line_adapter.mixed_row_widget(mixed=False) is edit
    mixed_widget = line_adapter.mixed_row_widget(mixed=True, parent=parent)
    assert (
        mixed_widget.findChild(QtWidgets.QLabel, "figureComposerMixedValueMarker")
        is not None
    )
    line_adapter.set_mixed(True)
    assert line_adapter.unchanged_mixed()
    edit.setText("edited")
    edit.setModified(True)
    assert not line_adapter.unchanged_mixed()

    plain = QtWidgets.QPlainTextEdit(parent)
    plain_adapter = _editor_controls.PlainTextControlAdapter(plain)
    plain_adapter.set_mixed(True)
    assert plain_adapter.unchanged_mixed()
    plain.document().setModified(True)
    assert not plain_adapter.unchanged_mixed()

    combo = QtWidgets.QComboBox(parent)
    combo.addItems(["a", "b"])
    combo_adapter = _editor_controls.ComboBoxControlAdapter(combo)
    combo_adapter.set_mixed(True)
    assert combo.currentData() is _editor_controls.MIXED_VALUE
    assert not combo.model().item(0).isEnabled()

    committed_values: list[str] = []
    commit_combo = QtWidgets.QComboBox(parent)
    commit_combo.addItems(["a", "b"])
    _editor_controls.ComboBoxControlAdapter(commit_combo).connect_commit(
        lambda _widget, signal, callback: signal.connect(callback),
        committed_values.append,
    )
    commit_combo.setCurrentIndex(1)
    assert committed_values == []
    commit_combo.activated.emit(1)
    assert committed_values == ["b"]

    check = QtWidgets.QCheckBox(parent)
    check_adapter = _editor_controls.CheckBoxControlAdapter(check)
    check_adapter.set_mixed(True)
    assert check.checkState() == QtCore.Qt.CheckState.PartiallyChecked

    destroyed = QtWidgets.QLineEdit()
    destroyed_adapter = _editor_controls.LineEditControlAdapter(destroyed)
    del destroyed
    gc.collect()
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = destroyed_adapter.widget


def test_figure_composer_state_validators_cover_invalid_values() -> None:
    with pytest.raises(ValueError, match="before zero"):
        FigureGridSpecSpanState(
            row_start=-1,
            row_stop=1,
            col_start=0,
            col_stop=1,
        )
    with pytest.raises(ValueError, match="at least one cell"):
        FigureGridSpecSpanState(
            row_start=1,
            row_stop=1,
            col_start=0,
            col_stop=1,
        )
    with pytest.raises(ValueError, match="at least one row"):
        FigureGridSpecGridState(nrows=0, ncols=1)
    with pytest.raises(ValueError, match="ratios must be positive"):
        FigureGridSpecGridState(nrows=1, ncols=1, width_ratios=(0.0,))
    with pytest.raises(ValueError, match="at least one row"):
        FigureSubplotsState(nrows=0)
    with pytest.raises(ValueError, match="figsize values"):
        FigureSubplotsState(figsize=(0.0, 1.0))
    with pytest.raises(ValueError, match="dpi must be positive"):
        FigureSubplotsState(dpi=0.0)
    with pytest.raises(ValueError, match="ratios must be positive"):
        FigureSubplotsState(width_ratios=(-1.0,))
    with pytest.raises(ValueError, match="export dpi must be positive"):
        FigureExportState(dpi=0.0)

    empty_selection = FigureAxesSelectionState(axes=())
    assert empty_selection.bounded(FigureSubplotsState()).axes == ((0, 0),)


def test_figure_composer_rendering_helpers_cover_selection_edges(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1, sharex=False, sharey=False),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    assert "sharex" not in figurecomposer_rendering._setup_kwargs(tool)
    assert "sharey" not in figurecomposer_rendering._setup_kwargs(tool)

    fig, axs = plt.subplots(1, 1, squeeze=False)
    with pytest.raises(ValueError, match="outside the current figure"):
        figurecomposer_rendering._axes_from_selection(
            tool,
            FigureAxesSelectionState(axes=((2, 0),)),
            axs,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_rendering._axes_from_selection(
            tool,
            FigureAxesSelectionState(axes=()),
            axs,
            for_plot_slices=False,
        )
    plt.close(fig)

    grid_axis = FigureGridSpecAxesState(
        axes_id="axis-a",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(grid_axis,),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_fig, grid_axs = plt.subplots(1, 1)
    axes_by_id = {"axis-a": grid_axs}
    with pytest.raises(ValueError, match="Advanced axes expressions"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(expression="axs"),
            axes_by_id,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="outside the current GridSpec"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(axes_ids=("missing",)),
            axes_by_id,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(axes_ids=()),
            axes_by_id,
            for_plot_slices=False,
        )
    assert figurecomposer_rendering._iter_axes({"axis-a": grid_axs}) == (grid_axs,)
    assert figurecomposer_rendering._iter_axes([grid_axs]) == (grid_axs,)
    assert figurecomposer_rendering._render_error_text(RuntimeError()) == "RuntimeError"
    plt.close(grid_fig)


def test_figure_composer_custom_code_helpers_cover_codegen_paths(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    operation = FigureOperationState.custom(
        label="code",
        code=(
            "values = np.asarray(data.values)\n"
            "arr = xr.DataArray(values)\n"
            "eplt.clean_labels(ax)"
        ),
        trusted=True,
    )
    assert figurecomposer_custom_code._section_summary(tool, "missing", operation) == ""
    assert figurecomposer_custom_code._required_imports(tool, operation) == (
        "import numpy as np",
        "import xarray as xr",
        "import erlab.plotting as eplt",
    )
    assert figurecomposer_custom_code._custom_code_names("bad code !!") == frozenset()
    assert figurecomposer_custom_code._custom_axes_alias_lines(tool) == []
    assert (
        figurecomposer_custom_code._code_lines(
            tool, operation.model_copy(update={"trusted": False})
        )
        == []
    )
    assert (
        figurecomposer_custom_code._required_imports(
            tool, operation.model_copy(update={"code": ""})
        )
        == ()
    )

    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="axis-a",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    assert figurecomposer_custom_code._custom_axes_alias_lines(grid_tool) == [
        "axs = {",
        "    'axis-a': ax0,",
        "}",
    ]
    assert figurecomposer_custom_code._custom_first_axis_code(grid_tool) == "ax0"


def test_figure_composer_custom_code_editor_is_multiline_and_debounced(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None
    assert isinstance(code_edit.highlighter, erlab.interactive.utils.PythonHighlighter)
    assert code_edit.lineWrapMode() == QtWidgets.QTextEdit.LineWrapMode.NoWrap

    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    first_code = "ax.set_title('first')"
    second_code = "ax.set_title('second')\nax.set_xlabel('energy')"
    code_edit.setPlainText(first_code)
    qtbot.wait(300)
    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    assert render_calls == []

    code_edit.setPlainText(second_code)

    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    assert render_calls == []

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == second_code,
        timeout=1000,
    )
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert render_calls == [(tool,)]


def test_figure_composer_custom_code_editor_skips_render_until_valid_python(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    invalid_code = "ax.set_title("
    code_edit.setPlainText(invalid_code)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == invalid_code,
        timeout=2000,
    )
    assert render_calls == []
    assert info_changed == [None]

    valid_code = "ax.set_title('valid')"
    code_edit.setPlainText(valid_code)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == valid_code,
        timeout=2000,
    )
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert render_calls == [(tool,)]
    assert info_changed == [None, None, None]


def test_figure_composer_custom_code_pending_edit_survives_step_switch(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    first_operation = FigureOperationState.custom(
        label="first",
        code="ax.set_title('old')",
        trusted=True,
    )
    second_operation = FigureOperationState.custom(
        label="second",
        code="ax.set_xlabel('other')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(first_operation, second_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    new_code = "ax.set_title('pending')\nax.set_ylabel('counts')"
    code_edit.setPlainText(new_code)
    tool.operation_list.setCurrentRow(1)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == new_code,
        timeout=1000,
    )
    assert tool.tool_status.operations[1].code == "ax.set_xlabel('other')"


def test_figure_composer_custom_code_pending_edit_flushes_on_close(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    new_code = "ax.set_title('pending close')\nax.set_ylabel('counts')"
    code_edit.setPlainText(new_code)

    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    tool.close()

    assert tool.tool_status.operations[0].code == new_code


def test_figure_composer_custom_code_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(8.0).reshape(4, 2),
        dims=("sample_temp", "alpha"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.custom(
        label="code",
        code=(
            "assert 'sample_temp' in data.dims\n"
            "assert 'sample_temp_idx' not in data.dims"
        ),
        trusted=True,
    )
    tool = FigureComposerTool(
        internal,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="map"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_preview(tool, show_window=False)

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" not in item.text()


def test_figure_composer_plot_slices_panel_helpers_cover_style_contract(
    qtbot,
) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={
            "eV": [0.0, 1.0],
            "kx": [-1.0, 0.0, 1.0],
            "temperature": ("eV", [20.0, 30.0]),
        },
        name="line_map",
    )
    tool = FigureComposerTool(
        source,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="line_map"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    operation = FigureOperationState.plot_slices(
        label="selection",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "cmap": "viridis",
            "line_kw": {"linewidth": 1.5},
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="magma",
                    norm_name="Normalize",
                    vmin=0.0,
                    vmax=5.0,
                    line_kw={"color": "red"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="plasma",
                    norm_name="PowerNorm",
                    norm_gamma=0.5,
                    line_kw={"color": "blue"},
                ),
            ),
            "line_normalize": "max",
            "line_offset_source": "coordinate",
            "line_iter_dim": "eV",
        }
    )

    keys = figurecomposer_plot_slices._plot_slices_panel_keys(tool, operation)
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert figurecomposer_plot_slices._plot_slices_slice_count(tool, operation) == 2
    assert figurecomposer_plot_slices._plot_slices_slice_labels(operation, 2) == (
        "eV=0",
        "eV=1",
    )
    assert figurecomposer_plot_slices._panel_cmap_argument(tool, operation) == [
        ["magma", "plasma"]
    ]
    assert figurecomposer_plot_slices._panel_line_kw_argument(tool, operation) == [
        [{"linewidth": 1.5, "color": "red"}, {"linewidth": 1.5, "color": "blue"}]
    ]
    assert figurecomposer_plot_slices._has_panel_line_kw_overrides(tool, operation)
    norm_argument = figurecomposer_plot_slices._panel_norm_argument(tool, operation)
    assert isinstance(norm_argument, list)
    assert figurecomposer_plot_slices._panel_norm_uses_matplotlib_colors(
        tool, operation
    )
    assert "mcolors.Normalize" in (
        figurecomposer_plot_slices._panel_norm_code(tool, operation) or ""
    )

    profiles, profile_keys = figurecomposer_plot_slices._plot_slices_line_profiles(
        tool,
        operation,
        maps=(source,),
    )
    assert len(profiles) == 2
    assert [(key.map_index, key.slice_index) for key in profile_keys] == [
        (0, 0),
        (0, 1),
    ]
    assert figurecomposer_plot_slices._plot_slices_uses_transformed_line_maps(
        tool, operation
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="In a future version of xarray the default value for .*",
            category=FutureWarning,
        )
        transformed_maps = figurecomposer_plot_slices._plot_slices_transformed_maps(
            tool,
            operation,
            (source,),
        )
    assert len(transformed_maps) == 1
    assert transformed_maps[0].dims == ("eV", "kx")
    assert set(
        figurecomposer_plot_slices._available_plot_slices_offset_coords(tool, operation)
    ) >= {"eV", "temperature"}

    code_operation = operation.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 0), (0, 1)))}
    )
    fig, axs = plt.subplots(1, 2, squeeze=False)
    namespace: dict[str, typing.Any] = {
        "data": source,
        "eplt": eplt,
        "xr": xr,
        "axs": axs,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="In a future version of xarray the default value for .*",
            category=FutureWarning,
        )
        exec(  # noqa: S102
            "\n".join(
                figurecomposer_plot_slices._plot_slices_transformed_code_lines(
                    tool, code_operation
                )
            ),
            namespace,
        )
    assert len(axs[0, 0].lines) == 1
    assert len(axs[0, 1].lines) == 1
    plt.close(fig)

    single_panel_operation = operation.model_copy(update={"slice_values": (0.0,)})
    assert (
        figurecomposer_plot_slices._panel_cmap_argument(tool, single_panel_operation)
        == "magma"
    )
    single_norm = figurecomposer_plot_slices._panel_norm_argument(
        tool, single_panel_operation
    )
    assert isinstance(single_norm, mpl.colors.Normalize)
    assert figurecomposer_plot_slices._panel_line_kw_argument(
        tool, single_panel_operation
    ) == {"linewidth": 1.5, "color": "red"}

    no_override_operation = operation.model_copy(
        update={"panel_styles_enabled": False, "panel_styles": ()}
    )
    assert (
        figurecomposer_plot_slices._panel_norm_argument(tool, no_override_operation)
        is None
    )
    assert (
        figurecomposer_plot_slices._panel_norm_code(tool, no_override_operation) is None
    )
    assert figurecomposer_plot_slices._panel_line_kw_argument(
        tool, no_override_operation
    ) == {"linewidth": 1.5}

    same_cmap_operation = operation.model_copy(
        update={
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=0, cmap="magma"
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, cmap="magma"
                ),
            )
        }
    )
    assert (
        figurecomposer_plot_slices._panel_cmap_argument(tool, same_cmap_operation)
        == "magma"
    )
    same_line_operation = operation.model_copy(
        update={
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=0, line_kw={"color": "red"}
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, line_kw={"color": "red"}
                ),
            )
        }
    )
    assert figurecomposer_plot_slices._panel_line_kw_argument(
        tool, same_line_operation
    ) == {"linewidth": 1.5, "color": "red"}

    selection_operation = operation.model_copy(
        update={
            "slice_dim": None,
            "slice_values": (),
            "slice_kwargs": {"kx": [-1.0, 0.0], "kx_width": 0.1},
        }
    )
    assert figurecomposer_plot_slices._plot_slices_slice_labels(
        selection_operation, 2
    ) == ("kx[0]", "kx[1]")

    no_override_operation = operation.model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0),
            ),
        }
    )
    assert (
        figurecomposer_plot_slices._panel_cmap_argument(tool, no_override_operation)
        == "viridis"
    )
    assert (
        figurecomposer_plot_slices._panel_norm_argument(tool, no_override_operation)
        is None
    )
    assert (
        figurecomposer_plot_slices._panel_norm_code(tool, no_override_operation) is None
    )
    assert not figurecomposer_plot_slices._panel_norm_uses_matplotlib_colors(
        tool, no_override_operation
    )
    assert figurecomposer_plot_slices._panel_line_kw_argument(
        tool, no_override_operation
    ) == {"linewidth": 1.5}


def test_figure_composer_plot_slices_edge_helper_contracts(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *_args, **_kwargs: None,
    )
    image = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0], "ky": range(4)},
        name="image",
    )
    line = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0]},
        name="line",
    )
    other = xr.DataArray(
        np.arange(8.0).reshape(2, 4),
        dims=("eV", "phi"),
        coords={"eV": [0.0, 1.0], "phi": range(4)},
        name="other",
    )
    line_operation = FigureOperationState.plot_slices(
        label="line",
        sources=("line",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "line_kw": {"linewidth": 1.5},
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    line_kw={"color": "red"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"color": "blue"},
                ),
            ),
            "order": "F",
            "gradient": True,
            "gradient_kw": {"alpha": 0.2},
            "line_normalize": "mean",
        }
    )
    image_operation = FigureOperationState.plot_slices(
        label="image",
        sources=("image",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
        axes=FigureAxesSelectionState(expression="axs[0, :]"),
    ).model_copy(
        update={
            "transpose": True,
            "xlim": (-1.0, None),
            "ylim": 0.5,
            "crop": False,
            "same_limits": True,
            "axis": "x",
            "show_all_labels": True,
            "colorbar": "right",
            "hide_colorbar_ticks": False,
            "annotate": False,
            "cmap": "magma",
            "norm_name": "PowerNorm",
            "norm_gamma": 0.5,
            "vmin": 0.0,
            "vmax": 10.0,
            "order": "F",
            "cmap_order": "F",
            "norm_order": "F",
            "subplot_kw": {"sharex": True},
            "annotate_kw": {"fontsize": 8},
            "colorbar_kw": {"ticks": [0.0, 1.0]},
            "extra_kwargs": {"alpha": 0.9},
        }
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="image", label="image"),
                FigureSourceState(name="line", label="line"),
                FigureSourceState(name="other", label="other"),
            ),
            operations=(line_operation, image_operation),
            primary_source="image",
        ),
        source_data={"image": image, "line": line, "other": other},
    )
    qtbot.addWidget(tool)

    with monkeypatch.context() as context:
        context.setattr(
            tool,
            "_editable_operations",
            lambda: ((0, line_operation), (1, image_operation)),
        )
        assert (
            figurecomposer_plot_slices._plot_slices_batch_panel_kind(
                tool, line_operation
            )
            == "mixed"
        )
    with monkeypatch.context() as context:
        context.setattr(tool, "_editable_operations", lambda: ())
        assert (
            figurecomposer_plot_slices._plot_slices_batch_panel_kind(
                tool, line_operation
            )
            == "line"
        )

    keys = figurecomposer_plot_slices._plot_slices_panel_keys(tool, line_operation)
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert figurecomposer_plot_slices._plot_slices_slice_labels(
        line_operation.model_copy(update={"slice_values": ()}),
        2,
    ) == ("slice 1", "slice 2")
    slice_kwarg_operation = line_operation.model_copy(
        update={
            "slice_dim": None,
            "slice_values": (),
            "slice_kwargs": {"eV": [0.0, 1.0], "eV_width": 0.2},
        }
    )
    assert (
        figurecomposer_plot_slices._plot_slices_slice_count(tool, slice_kwarg_operation)
        == 2
    )
    shape = figurecomposer_plot_slices._plot_slices_shape(tool, slice_kwarg_operation)
    assert shape.valid
    assert shape.panel_count == 2
    range_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        line_operation.model_copy(
            update={
                "slice_dim": None,
                "slice_values": (),
                "slice_kwargs": {"kx": slice(-1.0, 1.0), "eV": 0.0},
            }
        ),
    )
    assert range_shape.valid

    missing_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        FigureOperationState.plot_slices(label="missing", sources=("missing",)),
    )
    assert not missing_shape.valid
    mismatched_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        FigureOperationState.plot_slices(label="mixed", sources=("line", "other")),
    )
    assert not mismatched_shape.valid
    invalid_cut_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        line_operation.model_copy(update={"slice_dim": "missing", "slice_values": ()}),
    )
    assert invalid_cut_shape.valid
    incomplete_cut_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        line_operation.model_copy(update={"slice_values": ()}),
    )
    assert incomplete_cut_shape.valid

    image_kwargs = figurecomposer_plot_slices._plot_slices_kwargs(tool, image_operation)
    assert image_kwargs["transpose"] is True
    assert image_kwargs["xlim"] == (-1.0, None)
    assert image_kwargs["ylim"] == 0.5
    assert image_kwargs["crop"] is False
    assert image_kwargs["same_limits"] is True
    assert image_kwargs["axis"] == "x"
    assert image_kwargs["show_all_labels"] is True
    assert image_kwargs["colorbar"] == "right"
    assert image_kwargs["hide_colorbar_ticks"] is False
    assert image_kwargs["annotate"] is False
    assert image_kwargs["cmap"] == "magma"
    assert image_kwargs["gamma"] == 0.5
    assert image_kwargs["vmin"] == 0.0
    assert image_kwargs["vmax"] == 10.0
    assert image_kwargs["order"] == "F"
    assert image_kwargs["cmap_order"] == "F"
    assert image_kwargs["norm_order"] == "F"

    set_xlim_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_xlim",
        args=(0.0, None),
    )
    assert figurecomposer_method._method_float_pair_args(
        tool,
        set_xlim_operation,
        figurecomposer_method._method_spec(set_xlim_operation),
    ) == (0.0, None)
    assert image_kwargs["subplot_kw"] == {"sharex": True}
    assert image_kwargs["annotate_kw"] == {"fontsize": 8}
    assert image_kwargs["colorbar_kw"] == {"ticks": [0.0, 1.0]}
    assert image_kwargs["alpha"] == 0.9

    explicit_norm_kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool,
        image_operation.model_copy(
            update={"norm_name": "Normalize", "norm_gamma": None}
        ),
    )
    assert "norm" in explicit_norm_kwargs
    panel_norm_kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool,
        image_operation.model_copy(
            update={
                "panel_styles_enabled": True,
                "panel_styles": (
                    FigurePlotSlicesPanelStyleState(
                        map_index=0,
                        slice_index=0,
                        norm_name="Normalize",
                    ),
                ),
            }
        ),
    )
    assert "norm" in panel_norm_kwargs

    line_kwargs = figurecomposer_plot_slices._plot_slices_kwargs(tool, line_operation)
    assert line_kwargs["line_kw"] == [
        [{"linewidth": 1.5, "color": "red"}],
        [{"linewidth": 1.5, "color": "blue"}],
    ]
    assert line_kwargs["line_order"] == "F"
    assert line_kwargs["gradient"] is True
    assert line_kwargs["gradient_kw"] == {"alpha": 0.2}
    transformed_kwargs = figurecomposer_plot_slices._plot_slices_transformed_kwargs(
        tool,
        line_operation,
    )
    assert "eV_width" not in transformed_kwargs
    assert transformed_kwargs["eV"] == [0.0, 1.0]

    flat_axes = np.empty(4, dtype=object)
    reshaped_axes = figurecomposer_plot_slices._plot_slices_axes(
        line_operation.model_copy(update={"sources": ("line", "other")}),
        (line, other),
        flat_axes,
    )
    assert isinstance(reshaped_axes, np.ndarray)
    assert reshaped_axes.shape == (2, 2)
    mismatched_axes = np.empty(3, dtype=object)
    assert (
        figurecomposer_plot_slices._plot_slices_axes(
            line_operation,
            (line,),
            mismatched_axes,
        )
        is mismatched_axes
    )
    assert (
        figurecomposer_plot_slices._plot_slices_axes(line_operation, (line,), object())
        is not flat_axes
    )

    selection_operation = FigureOperationState.plot_slices(
        label="selection",
        sources=("image",),
        map_selections=(
            FigureDataSelectionState(source="image", isel={"eV": 0}),
            FigureDataSelectionState(source="image", qsel={"eV": 1.0}),
        ),
    )
    assert (
        len(figurecomposer_plot_slices._operation_maps(tool, selection_operation)) == 2
    )
    selection_lines = figurecomposer_plot_slices._plot_slices_code_lines(
        tool,
        selection_operation,
    )
    assert selection_lines[0] == "selected_maps = ["
    assert any("eplt.plot_slices" in line for line in selection_lines)
    single_selection_operation = FigureOperationState.plot_slices(
        label="single_selection",
        sources=("image",),
        map_selections=(FigureDataSelectionState(source="image", qsel={"eV": 1.0}),),
    )
    single_selection_lines = figurecomposer_plot_slices._plot_slices_code_lines(
        tool,
        single_selection_operation,
    )
    assert len(single_selection_lines) == 1
    assert "selected_maps" not in single_selection_lines[0]
    assert "eplt.plot_slices(image.qsel(eV=1.0)" in single_selection_lines[0]
    assert (
        figurecomposer_plot_slices._plot_slices_code_lines(
            tool,
            FigureOperationState.plot_slices(label="empty", sources=()),
        )
        == []
    )

    transform_lines = figurecomposer_plot_slices._plot_slices_transformed_code_lines(
        tool,
        line_operation,
    )
    assert transform_lines[0] == "profiles = ["
    assert any("eplt.plot_slices" in line for line in transform_lines)
    no_slice_map_lines, no_slice_maps_code = (
        figurecomposer_plot_slices._plot_slices_transformed_maps_code(
            line_operation.model_copy(update={"slice_dim": None, "slice_values": ()}),
            keys[:1],
        )
    )
    assert no_slice_map_lines == []
    assert no_slice_maps_code == "profiles[0]"

    assert figurecomposer_plot_slices._bool_or_text("True") is True
    assert figurecomposer_plot_slices._bool_or_text("False") is False
    assert figurecomposer_plot_slices._bool_or_text("row") == "row"
    assert figurecomposer_plot_slices._optional_number_or_text("vmin", "") is None
    assert (
        figurecomposer_plot_slices._optional_number_or_text("cmap", "magma") == "magma"
    )
    assert figurecomposer_plot_slices._optional_number_or_text("vmax", "1.5") == 1.5
    assert (
        figurecomposer_plot_slices._norm_field_placeholder(
            image_operation.model_copy(update={"norm_name": "CenteredPowerNorm"}),
            "vcenter",
        )
        == "0"
    )
    assert (
        figurecomposer_plot_slices._norm_field_placeholder(
            image_operation.model_copy(update={"vcenter": 1.0}),
            "vcenter",
        )
        == ""
    )
    placeholder_operation = FigureOperationState.plot_slices(
        label="image",
        sources=("image",),
        slice_dim="eV",
        slice_values=(0.0,),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    )
    placeholder_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="image", label="image"),),
            operations=(placeholder_operation,),
            primary_source="image",
        ),
        source_data={"image": image},
    )
    qtbot.addWidget(placeholder_tool)
    placeholder_tool.show_figure_window(activate=False)
    figurecomposer_rendering._render_preview(placeholder_tool, show_window=True)
    assert figurecomposer_plot_slices._plot_slices_color_limit_placeholders(
        placeholder_tool,
        placeholder_operation,
    ) == {"vmin": "0", "vmax": "11"}
    assert (
        figurecomposer_plot_slices._norm_gamma_value(
            image_operation.model_copy(update={"norm_gamma": None, "gamma": None})
        )
        == 1.0
    )
    assert figurecomposer_plot_slices._norm_clip_text(None) == "default"
    assert figurecomposer_plot_slices._norm_clip_from_text("True") is True
    assert figurecomposer_plot_slices._norm_clip_from_text("False") is False
    assert figurecomposer_plot_slices._norm_clip_from_text("default") is None

    tool.operation_list.setCurrentRow(1)
    figurecomposer_plot_slices._update_current_norm_name(tool, "CenteredPowerNorm")
    assert tool.tool_status.operations[1].norm_name == "CenteredPowerNorm"
    figurecomposer_plot_slices._update_current_norm_gamma(tool, 0.75)
    assert tool.tool_status.operations[1].norm_gamma == 0.75
    figurecomposer_plot_slices._update_current_norm_kwargs(
        tool,
        "halfrange=2.0, clip=True, custom=1",
    )
    assert tool.tool_status.operations[1].halfrange == 2.0
    assert tool.tool_status.operations[1].norm_clip is True
    assert tool.tool_status.operations[1].norm_kwargs == {"custom": 1}
    figurecomposer_plot_slices._update_current_slice_kwargs(
        tool,
        "eV=[0, 1], eV_width=0.2",
    )
    assert tool.tool_status.operations[1].slice_dim == "eV"
    assert tool.tool_status.operations[1].slice_width == 0.2
    figurecomposer_plot_slices._update_current_extra_kwargs(
        tool,
        "kx=0.0, alpha=0.5",
    )
    assert tool.tool_status.operations[1].slice_kwargs["kx"] == 0.0
    assert tool.tool_status.operations[1].extra_kwargs == {"alpha": 0.5}
    figurecomposer_plot_slices._update_current_cmap(tool, base="viridis", reverse=True)
    assert tool.tool_status.operations[1].cmap == "viridis_r"
    figurecomposer_plot_slices._update_current_panel_styles_enabled(tool, False)
    assert not tool.tool_status.operations[1].panel_styles_enabled
    figurecomposer_plot_slices._update_current_panel_styles(
        tool,
        (FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0, cmap="plasma"),),
    )
    assert tool.tool_status.operations[1].panel_styles_enabled
    assert tool.tool_status.operations[1].panel_styles[0].cmap == "plasma"
    figurecomposer_plot_slices._update_current_panel_styles(tool, ())
    assert not tool.tool_status.operations[1].panel_styles_enabled
    assert tool.tool_status.operations[1].panel_styles == ()


def test_figure_composer_plot_slices_all_coordinate_values_with_thin(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(5 * 3 * 2.0).reshape(5, 3, 2),
        dims=("eV", "kx", "ky"),
        coords={
            "eV": np.linspace(-1.0, 1.0, 5),
            "kx": [0.0, 1.0, 2.0],
            "ky": [10.0, 20.0],
        },
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
        axes=FigureAxesSelectionState(expression="axs"),
    ).model_copy(
        update={
            "slice_values_mode": "all",
            "slice_values_thin": 2,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    expected_values = data.thin({"eV": 2}).coords["eV"].values
    kwargs = figurecomposer_plot_slices._plot_slices_kwargs(tool, operation)
    np.testing.assert_allclose(kwargs["eV"], expected_values)
    shape = figurecomposer_plot_slices._plot_slices_shape(tool, operation)
    assert shape.panel_count == expected_values.size
    assert shape.plot_dims == ("kx", "ky")

    code_kwargs = figurecomposer_plot_slices._plot_slices_code_kwargs(tool, operation)
    assert isinstance(code_kwargs["eV"], figurecomposer_text._RawCode)
    captured: list[dict[str, typing.Any]] = []

    class PlotSlicesCapture:
        @staticmethod
        def plot_slices(_maps, **plot_kwargs):
            captured.append(plot_kwargs)

    exec(  # noqa: S102
        "\n".join(figurecomposer_plot_slices._plot_slices_code_lines(tool, operation)),
        {
            "data": data,
            "eplt": PlotSlicesCapture,
            "axs": object(),
        },
    )
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0]["eV"], expected_values)

    full_values_operation = operation.model_copy(update={"slice_values_thin": 1})
    full_kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool,
        full_values_operation,
    )
    np.testing.assert_allclose(full_kwargs["eV"], data.coords["eV"].values)

    tool.operation_list.setCurrentRow(0)
    tool._update_operation_editor()
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    assert selection_page is not None
    values_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    assert values_edit is None
    coordinate_summary = selection_page.findChild(
        QtWidgets.QLabel, "figureComposerPlotSlicesCoordinateSummary"
    )
    assert coordinate_summary is not None
    thin_spin = selection_page.findChild(
        QtWidgets.QAbstractSpinBox, "figureComposerPlotSlicesValuesThinSpin"
    )
    assert thin_spin is not None
    assert thin_spin.isEnabled()
    assert typing.cast("typing.Any", thin_spin).value() == 2


def test_figure_composer_plot_slices_shape_and_source_editor_contracts(
    qtbot,
    monkeypatch,
) -> None:
    first = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0, 2.0], "ky": range(4)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(12.0).reshape(2, 6),
        dims=("eV", "kz"),
        coords={"eV": [0.0, 1.0], "kz": range(6)},
        name="second",
    )
    first_operation = FigureOperationState.plot_slices(
        label="first",
        sources=("first",),
        axes=FigureAxesSelectionState(axes=((0, 0),), expression="axs[0, 0]"),
    ).model_copy(
        update={
            "transpose": True,
            "slice_kwargs": {
                "eV": slice(0.0, 1.0),
                "kx": 1.0,
                "ky": [0.0, 1.0],
                "ky_width": 0.2,
            },
        }
    )
    second_operation = FigureOperationState.plot_slices(
        label="second",
        sources=("second",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    )
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="first", label="first"),
                FigureSourceState(name="second", label="second"),
            ),
            operations=(first_operation, second_operation),
            primary_source="first",
        ),
        source_data={"first": first, "second": second},
    )
    qtbot.addWidget(tool)

    shape = figurecomposer_plot_slices._plot_slices_shape(tool, first_operation)
    assert shape.source_text == "eV, kx, ky"
    assert shape.panel_text == "eV (1D line)"
    assert shape.selection_text == ""
    assert shape.plot_ndim == 1
    assert shape.panel_count == 2
    assert shape.valid
    invalid_shape = figurecomposer_plot_slices._plot_slices_shape(
        tool,
        first_operation.model_copy(
            update={
                "slice_kwargs": {"eV": 0.0, "kx": 1.0, "ky": 2.0},
            }
        ),
    )
    assert invalid_shape.plot_ndim == 0
    assert not invalid_shape.valid
    assert (
        figurecomposer_plot_slices._section_summary(tool, "selection", first_operation)
        == "additional"
    )
    assert (
        figurecomposer_plot_slices._section_summary(tool, "view", first_operation)
        == "auto"
    )
    assert (
        figurecomposer_plot_slices._section_summary(tool, "advanced", first_operation)
        == ""
    )
    assert (
        figurecomposer_plot_slices._section_summary(tool, "unknown", first_operation)
        == ""
    )

    mixed_operation = first_operation.model_copy(
        update={"sources": ("first", "second")}
    )
    mixed_shape = figurecomposer_plot_slices._plot_slices_shape(tool, mixed_operation)
    assert not mixed_shape.valid
    assert mixed_shape.plot_ndim is None

    empty_tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(),
            operations=(
                FigureOperationState.plot_slices(label="missing", sources=("missing",)),
            ),
            primary_source="missing",
        ),
        source_data={},
    )
    qtbot.addWidget(empty_tool)
    empty_tool._source_data = {}
    empty_shape = figurecomposer_plot_slices._plot_slices_shape(
        empty_tool, empty_tool.tool_status.operations[0]
    )
    assert not empty_shape.valid
    assert empty_shape.panel_count == 0

    with monkeypatch.context() as context:
        context.setattr(
            tool,
            "_editable_operations",
            lambda: ((0, first_operation), (1, second_operation)),
        )
        assert (
            figurecomposer_plot_slices._plot_source_check_state(
                tool, first_operation, "first"
            )
            == QtCore.Qt.CheckState.PartiallyChecked
        )

    tool._update_source_section()
    selector = tool.step_source_controls.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesSourceSelector"
    )
    assert selector is not None
    checks = selector.findChildren(QtWidgets.QCheckBox)
    assert len(checks) == 2
    first_check = next(
        check for check in checks if check.property("figure_source_name") == "first"
    )
    second_check = next(
        check for check in checks if check.property("figure_source_name") == "second"
    )
    assert first_check.checkState() == QtCore.Qt.CheckState.Checked
    assert second_check.checkState() == QtCore.Qt.CheckState.Unchecked

    second_check.setCheckState(QtCore.Qt.CheckState.Checked)
    figurecomposer_plot_slices._plot_source_check_changed(
        tool, "second", second_check, ("first", "second")
    )
    assert tool.tool_status.operations[0].sources == ("first", "second")

    figurecomposer_plot_slices._plot_source_move(tool, "first", 1)
    assert tool.tool_status.operations[0].sources[:2] == ("second", "first")


def test_figure_composer_plot_slices_image_panel_style_editor_updates_styles(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(
        update={
            "cmap": "viridis",
            "norm_name": "PowerNorm",
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="magma",
                    norm_name="Normalize",
                    vmin=0.0,
                    vmax=1.0,
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="plasma_r",
                    norm_name="TwoSlopeNorm",
                    vcenter=0.0,
                    norm_kwargs={"clip": False},
                ),
            ),
        }
    )
    keys = (
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, "panel 1"),
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, 1, "panel 2"),
    )
    editor = figurecomposer_plot_slices._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
    )
    qtbot.addWidget(editor)
    emitted: list[tuple[FigurePlotSlicesPanelStyleState, ...]] = []
    editor.sigPanelStylesChanged.connect(emitted.append)

    for row in range(editor.panel_list.count()):
        item = editor.panel_list.item(row)
        assert item is not None
        item.setSelected(True)
    editor._sync_controls()
    assert not editor.cmap_override_check.isTristate()
    assert editor.cmap_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.cmap_combo.currentData() is figurecomposer_plot_slices._MISSING
    assert not editor.norm_override_check.isTristate()
    assert editor.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.norm_combo.currentData() is figurecomposer_plot_slices._MISSING
    assert editor.norm_kwargs_edit.placeholderText() == "(multiple values)"

    editor.norm_kwargs_edit.editingFinished.emit()
    assert emitted == []
    editor.vmin_edit.setText("0.2")
    editor.vmin_edit.setModified(True)
    editor.vmin_edit.editingFinished.emit()
    assert emitted[-1][0].vmin == pytest.approx(0.2)
    assert emitted[-1][1].vmin == pytest.approx(0.2)

    editor.norm_override_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert all(style.norm_name is None for style in emitted[-1])
    assert all(style.norm_kwargs == {} for style in emitted[-1])

    editor.cmap_override_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert emitted[-1] == ()
    assert not editor.cmap_override_check.isTristate()
    assert editor.cmap_override_check.checkState() == QtCore.Qt.CheckState.Unchecked
    assert not editor.cmap_combo.isEnabled()
    assert all("magma" not in editor.panel_list.item(row).text() for row in range(2))


def test_figure_composer_plot_slices_panel_override_controls_stay_live(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(update={"cmap": "viridis", "norm_name": "PowerNorm"})
    keys = (figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, "panel 1"),)
    editor = figurecomposer_plot_slices._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
    )
    qtbot.addWidget(editor)
    emitted: list[tuple[FigurePlotSlicesPanelStyleState, ...]] = []
    editor.sigPanelStylesChanged.connect(emitted.append)

    editor.cmap_combo.activated.emit(editor.cmap_combo.currentIndex())
    editor.cmap_reverse_check.setCheckState(QtCore.Qt.CheckState.Checked)
    editor.norm_combo.activated.emit(editor.norm_combo.currentIndex())
    editor.gamma_edit.setText("2")
    editor.gamma_edit.setModified(True)
    editor.gamma_edit.editingFinished.emit()
    editor.clip_combo.activated.emit(editor.clip_combo.currentIndex())
    editor.norm_kwargs_edit.setText("clip=False")
    editor.norm_kwargs_edit.setModified(True)
    editor.norm_kwargs_edit.editingFinished.emit()
    assert emitted == []

    assert not editor.cmap_override_check.isTristate()
    editor.cmap_override_check.click()
    assert editor.cmap_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.cmap_combo.isEnabled()
    assert emitted[-1] == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="viridis",
        ),
    )

    editor.norm_override_check.click()
    assert editor.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.norm_combo.isEnabled()
    assert emitted[-1] == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="viridis",
            norm_name="PowerNorm",
        ),
    )


def test_figure_composer_plot_slices_line_panel_style_editor_updates_styles(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="line",
        sources=("data",),
    ).model_copy(
        update={
            "line_kw": {"linewidth": 1.0},
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    line_kw={"color": "red", "linestyle": "-"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"color": "blue", "marker": "o", "alpha": 0.5},
                ),
            ),
        }
    )
    keys = (
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, "panel 1"),
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, 1, "panel 2"),
    )
    editor = figurecomposer_plot_slices._PanelLineStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
    )
    qtbot.addWidget(editor)
    emitted: list[tuple[FigurePlotSlicesPanelStyleState, ...]] = []
    editor.sigPanelStylesChanged.connect(emitted.append)

    for row in range(editor.panel_list.count()):
        item = editor.panel_list.item(row)
        assert item is not None
        item.setSelected(True)
    editor._sync_controls()
    assert editor.color_edit.line_edit.placeholderText() == "(multiple values)"
    assert editor.style_combo.currentData() is figurecomposer_plot_slices._MISSING
    assert editor.line_kwargs_edit.placeholderText() == "(multiple values)"

    editor.line_kwargs_edit.editingFinished.emit()
    assert emitted == []
    editor.line_kwargs_edit.setText("alpha=0.25, linewidth=9")
    editor.line_kwargs_edit.setModified(True)
    editor.line_kwargs_edit.editingFinished.emit()
    assert emitted[-1][0].line_kw == {
        "color": "red",
        "linestyle": "-",
        "alpha": 0.25,
    }
    assert emitted[-1][1].line_kw == {
        "color": "blue",
        "marker": "o",
        "alpha": 0.25,
    }

    editor._line_kw_changed("linewidth", 2.5, aliases=("lw",))
    assert all(style.line_kw["linewidth"] == 2.5 for style in emitted[-1])

    editor._line_kw_changed("color", None, aliases=("c",))
    assert all("color" not in style.line_kw for style in emitted[-1])
    editor._update_selected_extra_line_kw({})
    assert all("alpha" not in style.line_kw for style in emitted[-1])
    assert all(
        "red" not in editor.panel_list.item(row).text()
        and "blue" not in editor.panel_list.item(row).text()
        for row in range(2)
    )


def test_figure_composer_color_widgets_parse_and_sync(qtbot, monkeypatch) -> None:
    opaque = QtGui.QColor(1, 2, 3)
    translucent = QtGui.QColor(1, 2, 3, 4)
    grayscale = figurecomposer_widgets._qcolor_from_mpl_color_text("0.5")
    assert grayscale is not None
    assert grayscale.getRgb() == (128, 128, 128, 255)
    cycle_color = figurecomposer_widgets._qcolor_from_mpl_color_text("C1")
    assert cycle_color is not None
    assert (
        cycle_color.getRgb()
        == QtGui.QColor.fromRgbF(*mpl.colors.to_rgba("C1")).getRgb()
    )
    tuple_alpha = figurecomposer_widgets._qcolor_from_mpl_color_text("(1, 0, 0, 0.5)")
    assert tuple_alpha is not None
    assert tuple_alpha.getRgb() == (255, 0, 0, 128)
    hex_alpha = figurecomposer_widgets._qcolor_from_mpl_color_text("#01020304")
    assert hex_alpha is not None
    assert hex_alpha.getRgb() == (1, 2, 3, 4)
    assert figurecomposer_widgets._qcolor_to_mpl_color_text(opaque) == "#010203"
    assert figurecomposer_widgets._qcolor_to_mpl_color_text(translucent) == "#01020304"
    assert (
        figurecomposer_widgets._qcolor_from_mpl_color_text("(1.0, 0.0, 0.0)")
        is not None
    )
    assert (
        figurecomposer_widgets._qcolor_from_mpl_color_text("[1.0, 0.0, 0.0]")
        is not None
    )
    assert figurecomposer_widgets._qcolor_from_mpl_color_text("") is None
    assert figurecomposer_widgets._qcolor_from_mpl_color_text("[bad") is None
    assert figurecomposer_widgets._top_level_comma_parts(
        "red, (0, 1, 0), 'blue, still blue', [0, 0, 1]"
    ) == ("red", "(0, 1, 0)", "'blue, still blue'", "[0, 0, 1]")
    assert figurecomposer_widgets._top_level_comma_parts(r"'red\', blue', green") == (
        r"'red\', blue'",
        "green",
    )
    assert figurecomposer_widgets._color_tuple_from_text("") == ()
    assert figurecomposer_widgets._color_tuple_from_text("['red', 'blue']") == (
        "red",
        "blue",
    )
    assert figurecomposer_widgets._color_tuple_from_text("[bad") == ("[bad",)
    assert figurecomposer_widgets._color_tuple_from_text("[1]") == ("1",)
    with monkeypatch.context() as context:
        context.setattr(
            figurecomposer_widgets, "_qcolor_from_mpl_color_value", lambda _value: None
        )
        assert figurecomposer_widgets._default_mpl_line_color().getRgb() == (
            0,
            0,
            0,
            255,
        )

    inherited_edit = figurecomposer_widgets._ColorLineEditWidget(
        "",
        inherited_color="C1",
    )
    qtbot.addWidget(inherited_edit)
    assert (
        inherited_edit.color_button.color().getRgb()
        == QtGui.QColor.fromRgbF(*mpl.colors.to_rgba("C1")).getRgb()
    )
    inherited_edit.setText("0.5")
    assert inherited_edit.color_button.color().getRgb() == (128, 128, 128, 255)
    inherited_edit.setText("")
    assert (
        inherited_edit.color_button.color().getRgb()
        == QtGui.QColor.fromRgbF(*mpl.colors.to_rgba("C1")).getRgb()
    )
    inherited_edit.setInheritedColor("#01020304")
    assert inherited_edit.color_button.color().getRgb() == (1, 2, 3, 4)

    color_edit = figurecomposer_widgets._ColorLineEditWidget("tab:blue")
    qtbot.addWidget(color_edit)
    color_edit.setPlaceholderText("pick one")
    color_edit.setToolTip("Line color")
    color_edit.setLineEditObjectName("lineColorText")
    color_edit.setColorButtonObjectName("lineColorButton")
    color_edit.setModified(True)
    assert color_edit.isModified()
    assert color_edit.line_edit.placeholderText() == "pick one"
    assert color_edit.line_edit.objectName() == "lineColorText"
    color_edit.setText("not-a-color")
    color_edit._syncing = True
    color_edit._button_color_changed(translucent)
    color_edit._syncing = False
    color_edit._button_color_changed(typing.cast("QtGui.QColor", object()))
    color_edit._button_color_changed(translucent)
    assert color_edit.text() == "#01020304"

    with monkeypatch.context() as context:
        context.setattr(
            QtWidgets.QColorDialog,
            "getColor",
            lambda *_args, **_kwargs: opaque,
        )
        color_edit.color_button._choose_color()
    assert color_edit.text() == "#010203"
    color_edit.color_button.setColor(typing.cast("QtGui.QColor", object()))
    assert color_edit.color_button.color().getRgb() == opaque.getRgb()
    with monkeypatch.context() as context:
        context.setattr(
            QtWidgets.QColorDialog,
            "getColor",
            lambda *_args, **_kwargs: QtGui.QColor(),
        )
        color_edit.color_button._choose_color()
    assert color_edit.text() == "#010203"
    color_edit.color_button._dialog_open = True
    color_edit.color_button._choose_color()
    color_edit.color_button._dialog_open = False

    button_pixmap = QtGui.QPixmap(color_edit.color_button.sizeHint())
    button_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    color_edit.color_button.render(button_pixmap)

    deleted_edit = figurecomposer_widgets._ColorLineEditWidget("tab:red")
    qtbot.addWidget(deleted_edit)
    deleted_button = deleted_edit.color_button

    def delete_during_dialog(*_args, **_kwargs) -> QtGui.QColor:
        deleted_edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        return QtGui.QColor("blue")

    with monkeypatch.context() as context:
        context.setattr(QtWidgets.QColorDialog, "getColor", delete_during_dialog)
        deleted_button._choose_color()

    color_list = figurecomposer_widgets._ColorListEditorWidget(("red", "blue"))
    qtbot.addWidget(color_list)
    changed: list[tuple[str, ...]] = []
    color_list.colorsChanged.connect(changed.append)
    assert color_list.colors() == ("red", "blue")
    color_list.setToolTip("Profile colors")
    assert all(edit.toolTip() == "Profile colors" for edit in color_list._row_editors())
    color_list.setMixedPlaceholder("(multiple values)")
    assert color_list.batchUnchanged()
    color_list._syncing = True
    color_list._main_text_finished()
    color_list._set_colors_from_structure_change(("yellow",))
    color_list._syncing = False
    assert changed == []
    color_list.main_edit.setText("green, (0, 0, 1)")
    color_list.main_edit.setModified(True)
    color_list._main_text_finished()
    assert changed[-1] == ("green", "(0, 0, 1)")
    color_list._add_color()
    assert changed[-1] == ("green", "(0, 0, 1)", "")
    color_list._remove_color(1)
    assert changed[-1] == ("green",)
    first_editor = color_list._row_editors()[0]
    first_row = first_editor.parentWidget()
    first_editor.setText("black")
    first_editor.editingFinished.emit()
    assert changed[-1][0] == "black"
    assert erlab.interactive.utils.qt_is_valid(first_editor, first_row)
    assert color_list._row_editors()[0] is first_editor

    with monkeypatch.context() as context:
        context.setattr(
            QtWidgets.QApplication,
            "focusWidget",
            staticmethod(lambda: first_editor.line_edit),
        )
        color_list._set_colors_from_structure_change(("black", "white"))
        assert color_list._row_rebuild_pending
        assert color_list._row_editors()[0] is first_editor
        color_list._queue_rebuild_rows(("black", "white", "red"))
        assert color_list._pending_row_rebuild_colors == ("black", "white", "red")
    color_list._run_pending_row_rebuild()
    assert len(color_list._row_editors()) == 3
    color_list._pending_row_rebuild_colors = None
    color_list._run_pending_row_rebuild()
    assert len(color_list._row_editors()) == 3
    color_list._rows_layout.addSpacerItem(QtWidgets.QSpacerItem(1, 1))
    assert len(color_list._row_editors()) == 3

    color_list._syncing = True
    color_list._set_colors_from_rows(("white",))
    color_list._syncing = False
    assert changed[-1][0] == "black"


def test_figure_composer_axes_selector_widget_mouse_selection(qtbot) -> None:
    selector = figurecomposer_widgets._AxesSelectorWidget()
    qtbot.addWidget(selector)
    selector.set_grid(
        2,
        3,
        labels={(0, 0): "left", (0, 1): "middle", (0, 2): "right"},
    )
    selector.resize(selector.sizeHint())
    selector.show()
    assert selector.cell_rect((5, 5)).isNull()
    assert selector._axis_at(QtCore.QPoint(-10, -10)) is None
    assert selector._axis_label((1, 2)) == "1, 2"

    selected: list[tuple[tuple[int, int], ...]] = []
    selector.sigSelectionChanged.connect(selected.append)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector.cell_rect((0, 1)).center(),
    )
    assert selected[-1] == ((0, 1),)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=selector.cell_rect((1, 2)).center(),
    )
    assert selected[-1] == ((0, 1), (0, 2), (1, 1), (1, 2))
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        pos=selector.cell_rect((0, 1)).center(),
    )
    assert (0, 1) not in selected[-1]
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        pos=selector.cell_rect((0, 0)).center(),
    )
    assert (0, 0) in selected[-1]
    before_outside_click = selected[-1]
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(-10, -10),
    )
    assert selected[-1] == before_outside_click
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.RightButton,
        pos=selector.cell_rect((0, 0)).center(),
    )

    _drag_widget(
        selector,
        selector.cell_rect((0, 0)).center(),
        selector.cell_rect((1, 1)).center(),
        modifiers=QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert selected[-1] == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
    selector.mousePressEvent(None)
    selector.mouseMoveEvent(None)
    selector.leaveEvent(None)

    add_requests: list[str] = []
    selector.sigAddRowRequested.connect(lambda: add_requests.append("row"))
    selector.sigAddColumnRequested.connect(lambda: add_requests.append("column"))
    _send_mouse_move(selector, selector._add_pill_rect("row").center())
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector._add_pill_rect("row").center(),
    )
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector._add_pill_rect("column").center(),
    )
    assert add_requests == ["row", "column"]

    pixmap = QtGui.QPixmap(selector.size())
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    selector.render(pixmap)


def test_figure_composer_selector_colors_use_widget_palette_group(qtbot) -> None:
    selector = figurecomposer_widgets._AxesSelectorWidget()
    qtbot.addWidget(selector)
    palette = selector.palette()
    palette.setColor(
        QtGui.QPalette.ColorGroup.Active,
        QtGui.QPalette.ColorRole.Highlight,
        QtGui.QColor("red"),
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Inactive,
        QtGui.QPalette.ColorRole.Highlight,
        QtGui.QColor("blue"),
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.Highlight,
        QtGui.QColor("green"),
    )
    selector.setPalette(palette)

    assert figurecomposer_widgets._selector_colors(selector).selection == QtGui.QColor(
        "blue"
    )
    selector.setEnabled(False)
    assert figurecomposer_widgets._selector_colors(selector).selection == QtGui.QColor(
        "green"
    )


def test_figure_composer_axes_selector_add_pills_use_selector_color_roles(
    qtbot, monkeypatch
) -> None:
    selector = figurecomposer_widgets._AxesSelectorWidget()
    qtbot.addWidget(selector)
    selector.resize(selector.sizeHint())
    colors = figurecomposer_widgets._selector_colors(selector)

    captured_rects: list[tuple[QtGui.QColor, QtGui.QColor]] = []

    def record_selector_rect(
        _painter: QtGui.QPainter,
        _rect: QtCore.QRect,
        *,
        facecolor: QtGui.QColor,
        edgecolor: QtGui.QColor,
        linewidth: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        del linewidth, radius
        captured_rects.append((QtGui.QColor(facecolor), QtGui.QColor(edgecolor)))

    monkeypatch.setattr(
        figurecomposer_widgets, "_draw_selector_rect", record_selector_rect
    )
    pixmap = QtGui.QPixmap(selector.size())
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        selector._draw_add_pill(painter, colors, "row")
        selector._hovered_add_control = "row"
        selector._draw_add_pill(painter, colors, "row")
    finally:
        painter.end()

    idle_face = QtGui.QColor(colors.face)
    idle_face.setAlpha(70)
    idle_edge = QtGui.QColor(colors.border)
    idle_edge.setAlpha(95)
    hover_face = QtGui.QColor(colors.hover_face)
    hover_face.setAlpha(190)
    hover_edge = QtGui.QColor(colors.selection)
    hover_edge.setAlpha(210)
    assert [(color.getRgb(), edge.getRgb()) for color, edge in captured_rects] == [
        (idle_face.getRgb(), idle_edge.getRgb()),
        (hover_face.getRgb(), hover_edge.getRgb()),
    ]


def test_figure_composer_gridspec_view_widget_selection_and_editing(qtbot) -> None:
    main_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    child_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=2,
        col_start=1,
        col_stop=3,
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=2,
        ncols=3,
        axes=(
            FigureGridSpecAxesState(
                axes_id="main-axis",
                label="main",
                span=main_span,
            ),
        ),
        child_grids=(
            FigureGridSpecGridState(
                grid_id="child-grid",
                label="child",
                nrows=1,
                ncols=2,
                span=child_span,
                axes=(
                    FigureGridSpecAxesState(
                        axes_id="child-axis",
                        label="child axis",
                        span=FigureGridSpecSpanState(
                            row_start=0,
                            row_stop=1,
                            col_start=0,
                            col_stop=1,
                        ),
                    ),
                ),
            ),
        ),
    )
    selector = figurecomposer_widgets._GridSpecViewWidget(mode="select")
    qtbot.addWidget(selector)
    selector.set_layout(root, labels={"main-axis": "Main", "child-axis": "Child"})
    selector.resize(selector.sizeHint())
    selector.show()
    assert selector.axes_ids() == ("main-axis", "child-axis")
    assert not selector.axis_rect("main-axis").isNull()
    assert selector.axis_rect("missing").isNull()
    assert selector._range_axes_ids("missing", "child-axis") == ("child-axis",)
    assert selector._axis_edges(0, 100, 2, ()) == (0.0, 50.0, 100.0)

    selected: list[tuple[str, ...]] = []
    selector.sigSelectionChanged.connect(selected.append)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(-10, -10),
    )
    assert selected == []
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.RightButton,
        pos=selector.axis_rect("main-axis").center(),
    )
    assert selected == []
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector.axis_rect("main-axis").center(),
    )
    assert selected[-1] == ("main-axis",)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=selector.axis_rect("child-axis").center(),
    )
    assert selected[-1] == ("main-axis", "child-axis")
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        pos=selector.axis_rect("main-axis").center(),
    )
    assert selected[-1] == ("child-axis",)
    _drag_widget(
        selector,
        selector.axis_rect("main-axis").center(),
        selector.axis_rect("child-axis").center(),
        modifiers=QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert selected[-1] == ("main-axis", "child-axis")
    selector.mouseMoveEvent(None)
    selector.leaveEvent(None)

    class _RecordingGridSpecViewWidget(figurecomposer_widgets._GridSpecViewWidget):
        def __init__(self) -> None:
            super().__init__(mode="edit")
            self.cursor_shapes: list[QtCore.Qt.CursorShape] = []
            self.unset_count = 0

        def setCursor(self, cursor: QtGui.QCursor | QtCore.Qt.CursorShape) -> None:
            shape = cursor.shape() if isinstance(cursor, QtGui.QCursor) else cursor
            self.cursor_shapes.append(shape)
            super().setCursor(cursor)

        def unsetCursor(self) -> None:
            self.unset_count += 1
            super().unsetCursor()

    editor = _RecordingGridSpecViewWidget()
    qtbot.addWidget(editor)
    region = figurecomposer_widgets._GridSpecRegionInfo(
        region_id="main-axis",
        kind="axes",
        span=main_span,
        label="main",
    )
    child_region = figurecomposer_widgets._GridSpecRegionInfo(
        region_id="child-grid",
        kind="grid",
        span=child_span,
        label="child",
    )
    editor.set_edit_grid(root, (region, child_region), labels={"main-axis": "Main"})
    editor.resize(editor.sizeHint())
    editor.show()
    editor.set_selected_region("main-axis")
    assert editor.selected_region_id() == "main-axis"
    editor.set_selected_region("missing")
    assert editor.selected_region_id() == ""
    editor.set_selected_region("main-axis")
    assert editor._region_label("unknown") == "unknown"
    assert editor._cell_at(QtCore.QPoint(-100, -100), clamp_to_grid=True) is not None
    assert editor._cell_at(QtCore.QPoint(-100, -100), clamp_to_grid=False) is None
    assert editor._occupied_grid_cells(root) >= {(0, 0), (0, 1), (1, 1)}
    assert editor._axis_edges(0, 100, 0, ()) == (0.0,)
    assert editor._axis_edges(0, 100, 2, (2.0, 1.0)) == (
        0.0,
        200.0 / 3.0,
        100.0,
    )
    assert (
        editor.span_rect(
            FigureGridSpecSpanState(row_start=2, row_stop=3, col_start=0, col_stop=1)
        )
        == QtCore.QRect()
    )
    assert editor.cell_rect((5, 5)) == QtCore.QRect()
    assert editor._handle_at(QtCore.QPoint(-100, -100)) is None
    editor._set_region_handles_visible(True)
    axis_rect = editor.span_rect(main_span)
    for handle, handle_rect in editor._handle_rects(axis_rect, hit=True):
        assert editor._handle_at(handle_rect.center()) == handle
    editor._update_hover_cursor(
        editor._handle_rects(axis_rect, hit=True)[0][1].center()
    )
    assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeFDiagCursor
    assert editor.cursor_shapes == [QtCore.Qt.CursorShape.SizeFDiagCursor]
    editor._update_hover_cursor(
        editor._handle_rects(axis_rect, hit=True)[0][1].center()
    )
    assert editor.cursor_shapes == [QtCore.Qt.CursorShape.SizeFDiagCursor]
    editor._update_hover_cursor(
        editor._handle_rects(axis_rect, hit=True)[1][1].center()
    )
    assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeBDiagCursor
    assert editor.cursor_shapes == [
        QtCore.Qt.CursorShape.SizeFDiagCursor,
        QtCore.Qt.CursorShape.SizeBDiagCursor,
    ]
    editor._set_region_handles_visible(False)
    editor._update_hover_cursor(axis_rect.center())
    assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeAllCursor
    editor._update_hover_cursor(QtCore.QPoint(-100, -100))
    assert editor.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
    assert editor.unset_count == 1
    editor._update_hover_cursor(QtCore.QPoint(-100, -100))
    assert editor.unset_count == 1
    assert editor._active_preview_span() is None
    editor._drag_mode = "create"
    editor._drag_origin_cell = (0, 0)
    editor._drag_current_cell = (1, 1)
    assert editor._active_preview_span() == FigureGridSpecSpanState(
        row_start=0,
        row_stop=2,
        col_start=0,
        col_stop=2,
    )
    editor._reset_edit_drag()
    editor._drag_mode = "move"
    editor._resize_preview_span = main_span
    assert editor._active_preview_span() == main_span
    editor._reset_edit_drag()
    assert editor._span_from_resize(main_span, None, (1, 1)) == main_span
    assert editor._span_from_resize(main_span, "se", (1, 2)) == FigureGridSpecSpanState(
        row_start=0,
        row_stop=2,
        col_start=0,
        col_stop=3,
    )
    assert editor._span_from_resize(
        child_span, "nw", (1, 0)
    ) == FigureGridSpecSpanState(
        row_start=1,
        row_stop=2,
        col_start=0,
        col_stop=3,
    )
    assert editor._span_from_move(
        main_span,
        origin=(0, 0),
        current=(2, 2),
    ) == FigureGridSpecSpanState(row_start=1, row_stop=2, col_start=2, col_stop=3)

    created: list[FigureGridSpecSpanState] = []
    editor.sigRegionCreated.connect(lambda span, _kind: created.append(span))
    qtbot.mousePress(
        editor,
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(-10, -10),
    )
    assert created == []
    _drag_widget(
        editor,
        editor.cell_rect((1, 0)).center(),
        editor.cell_rect((1, 0)).center(),
    )
    assert created[-1] == FigureGridSpecSpanState(
        row_start=1,
        row_stop=2,
        col_start=0,
        col_stop=1,
    )
    editor._edit_mouse_release(None)

    activated: list[str] = []
    editor.sigNestedGridActivated.connect(activated.append)
    selector.mouseDoubleClickEvent(None)
    qtbot.mouseDClick(
        editor,
        QtCore.Qt.MouseButton.LeftButton,
        pos=editor.span_rect(child_span).center(),
    )
    assert activated == ["child-grid"]
    pixmap = QtGui.QPixmap(editor.size())
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    editor.render(pixmap)
    editor._handle_application_event(QtCore.QEvent(QtCore.QEvent.Type.WindowDeactivate))
    assert not editor._region_handles_visible
    editor._set_region_handles_visible(True)
    editor._handle_application_event(QtCore.QEvent(QtCore.QEvent.Type.MouseMove))
    assert editor._region_handles_visible
    editor._handle_application_event(QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress))
    assert editor._region_handles_visible


def test_figure_composer_line_source_combo_uses_alias_data_and_updates_recipe(
    qtbot,
) -> None:
    first = _figure_composer_profile_source("first")
    second = _figure_composer_profile_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: first"),
                FigureSourceState(name="data_1", label="ImageTool 1: second"),
            ),
            operations=(FigureOperationState.line(label="line", source="data_0"),),
            primary_source="data_0",
        ),
        source_data={"data_0": first, "data_1": second},
    )
    qtbot.addWidget(tool)
    tool._select_step_section("sources")

    source_combos = tool.step_source_controls.findChildren(
        QtWidgets.QComboBox, "figureComposerLineSourceCombo"
    )
    source_combo = next(
        (
            combo
            for combo in source_combos
            if combo.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
        ),
        None,
    )
    assert source_combo is not None
    first_index = source_combo.findData("data_0")
    second_index = source_combo.findData("data_1")
    assert first_index >= 0
    assert second_index >= 0
    assert source_combo.itemData(first_index) == "data_0"
    assert source_combo.itemData(second_index) == "data_1"

    _activate_combo_index(source_combo, second_index)

    assert tool.tool_status.operations[0].line_source == "data_1"


def test_figure_composer_plot_slices_source_selector_batch_toggles_sources(
    qtbot,
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="first_source", label="first"),
                FigureSourceState(name="second_source", label="second"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="first",
                    sources=("first_source",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.plot_slices(
                    label="second",
                    sources=("second_source",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="first_source",
        ),
        source_data={"first_source": first, "second_source": second},
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("sources")

    checks = _plot_source_checks(tool)
    assert checks["first_source"].checkState() == (
        QtCore.Qt.CheckState.PartiallyChecked
    )
    assert checks["second_source"].checkState() == (
        QtCore.Qt.CheckState.PartiallyChecked
    )

    checks["first_source"].setCheckState(QtCore.Qt.CheckState.Checked)

    assert tool.tool_status.operations[0].sources == ("first_source",)
    assert tool.tool_status.operations[1].sources == (
        "first_source",
        "second_source",
    )


def test_figure_composer_subplots_plot_roundtrip_restores_exact_render(
    qtbot,
    tmp_path,
) -> None:
    image = _figure_composer_image_source("image")
    profile = _figure_composer_profile_source("profile")
    source_data = {"image_source": image, "profile_source": profile}
    setup = FigureSubplotsState(
        nrows=2,
        ncols=2,
        figsize=(3.4, 2.1),
        dpi=80,
        layout=None,
        sharex=False,
        sharey=False,
    )
    plot_operation = FigureOperationState.plot_slices(
        label="constant energy selection",
        sources=("image_source",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(-0.5, 0.5),
    ).model_copy(
        update={
            "annotate": False,
            "cmap": "viridis",
            "same_limits": True,
            "transpose": True,
        }
    )
    line_operation = FigureOperationState.line(
        label="profile",
        source="profile_source",
        axes=FigureAxesSelectionState(axes=((1, 0), (1, 1))),
    ).model_copy(
        update={
            "line_colors": ("black",),
            "line_labels": ("restored profile",),
            "line_normalize": "max",
            "line_x": "kx",
        }
    )
    axes_method = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_title",
        axes=FigureAxesSelectionState(axes=((1, 0), (1, 1))),
        args=("Profile",),
    )
    figure_method = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="supxlabel",
        args=("Shared coordinate",),
    )
    recipe = FigureRecipeState(
        setup=setup,
        sources=(
            FigureSourceState(name="image_source", label="image"),
            FigureSourceState(name="profile_source", label="profile"),
        ),
        operations=(plot_operation, line_operation, axes_method, figure_method),
        primary_source="image_source",
    )
    tool = FigureComposerTool(image, recipe=recipe, source_data=source_data)
    qtbot.addWidget(tool)

    restored = _assert_serialized_plot_restores_exactly(tool, qtbot, tmp_path)
    xr.testing.assert_identical(restored.source_data()["profile_source"], profile)


def test_figure_composer_gridspec_plot_roundtrip_restores_exact_render(
    qtbot,
    tmp_path,
) -> None:
    image = _figure_composer_image_source("image")
    line_map = _figure_composer_line_slice_source("line_map")
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        figsize=(3.4, 2.1),
        dpi=80,
        layout=None,
        sharex=False,
        sharey=False,
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                label="Root",
                nrows=1,
                ncols=2,
                width_ratios=(2.0, 1.0),
                axes=(
                    FigureGridSpecAxesState(
                        axes_id="image-axis",
                        label="image",
                        span=FigureGridSpecSpanState(
                            row_start=0,
                            row_stop=1,
                            col_start=0,
                            col_stop=1,
                        ),
                    ),
                ),
                child_grids=(
                    FigureGridSpecGridState(
                        grid_id="line-grid",
                        label="line grid",
                        nrows=2,
                        ncols=1,
                        span=FigureGridSpecSpanState(
                            row_start=0,
                            row_stop=1,
                            col_start=1,
                            col_stop=2,
                        ),
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="line-top",
                                label="top",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                            FigureGridSpecAxesState(
                                axes_id="line-bottom",
                                label="bottom",
                                span=FigureGridSpecSpanState(
                                    row_start=1,
                                    row_stop=2,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
    )
    image_operation = FigureOperationState.plot_slices(
        label="image cut",
        sources=("image_source",),
        axes=FigureAxesSelectionState(axes_ids=("image-axis",)),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"annotate": False, "cmap": "magma"})
    line_slices_operation = FigureOperationState.plot_slices(
        label="line selection",
        sources=("line_source",),
        axes=FigureAxesSelectionState(axes_ids=("line-top", "line-bottom")),
        slice_dim="eV",
        slice_values=(-0.5, 0.5),
    ).model_copy(
        update={
            "annotate": False,
            "colorbar": "none",
            "gradient": True,
        }
    )
    erlab_method = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="clean_labels",
        axes=FigureAxesSelectionState(
            axes_ids=("image-axis", "line-top", "line-bottom")
        ),
    )
    figure_method = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="suptitle",
        args=("Nested GridSpec",),
    )
    recipe = FigureRecipeState(
        setup=setup,
        sources=(
            FigureSourceState(name="image_source", label="image"),
            FigureSourceState(name="line_source", label="line map"),
        ),
        operations=(
            image_operation,
            line_slices_operation,
            erlab_method,
            figure_method,
        ),
        primary_source="image_source",
    )
    tool = FigureComposerTool(
        image,
        recipe=recipe,
        source_data={"image_source": image, "line_source": line_map},
    )
    qtbot.addWidget(tool)

    restored = _assert_serialized_plot_restores_exactly(tool, qtbot, tmp_path)
    xr.testing.assert_identical(restored.source_data()["line_source"], line_map)


def test_figure_composer_rebases_source_node_uids(qtbot) -> None:
    data = _figure_composer_image_source("data")
    nested_spec = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="nested",
        active_name="nested",
        script_inputs=(
            provenance.ScriptInput(
                name="nested",
                label="nested",
                node_uid="old-nested",
            ),
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(
                    name="data",
                    label="data",
                    node_uid="old-source",
                    provenance_spec=nested_spec.model_dump(mode="json"),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.rebase_source_node_uids(
        {"old-source": "new-source", "old-nested": "new-nested"}
    )

    source = tool.tool_status.sources[0]
    assert source.node_uid == "new-source"
    rebased_spec = provenance.parse_tool_provenance_spec(source.provenance_spec)
    assert rebased_spec is not None
    assert rebased_spec.script_inputs[0].node_uid == "new-nested"


def test_figure_composer_provenance_includes_sources_and_build_step(
    qtbot, tmp_path: Path
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
        name="map",
    )
    file_path = tmp_path / "map.nc"
    data.to_netcdf(file_path)
    source_spec = _file_load_provenance(file_path)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"annotate": False})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="test source",
                    provenance_spec=source_spec.model_dump(mode="json"),
                ),
            ),
            operations=(operation,),
            primary_source="data_0",
        ),
        source_data={"data_0": data},
    )
    qtbot.addWidget(tool)

    spec = tool.current_provenance_spec()
    assert spec is not None
    assert spec.active_name == "fig"
    assert tuple(script_input.name for script_input in spec.script_inputs) == (
        "data_0",
    )
    entries = spec.display_entries()
    assert len(entries) == 3
    assert entries[1].code is None
    assert not entries[1].copyable
    assert entries[2].code is not None
    assert entries[2].copyable

    code = spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["fig"], Figure)


def test_axes_selector_size_hint_tracks_grid(qtbot):
    selector = figurecomposer_widgets._AxesSelectorWidget()
    qtbot.addWidget(selector)

    selector.set_grid(1, 4)
    one_by_four_hint = selector.sizeHint()
    assert one_by_four_hint.width() < 220
    assert one_by_four_hint.height() <= 52
    assert (
        selector.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Maximum
    )
    assert selector.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Policy.Fixed

    selector.set_grid(2, 4)
    two_by_four_hint = selector.sizeHint()
    assert two_by_four_hint.width() == one_by_four_hint.width()
    assert one_by_four_hint.height() < two_by_four_hint.height() < 82

    selector.resize(two_by_four_hint + QtCore.QSize(80, 40))
    grid_rect = QtCore.QRect(
        selector.cell_rect((0, 0)).topLeft(),
        selector.cell_rect((1, 3)).bottomRight(),
    )
    available_rect = selector.rect().adjusted(
        selector._GRID_MARGIN,
        selector._GRID_MARGIN,
        -(
            selector._GRID_MARGIN
            + selector._ADD_PILL_GAP
            + selector._ADD_PILL_THICKNESS
        ),
        -(
            selector._GRID_MARGIN
            + selector._ADD_PILL_GAP
            + selector._ADD_PILL_THICKNESS
        ),
    )
    assert (grid_rect.center() - available_rect.center()).manhattanLength() <= 1


def test_figure_composer_axes_selector_add_pills_grow_subplots(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool._select_step_section("axes")
    selector = tool.axes_selector
    selector.resize(selector.sizeHint())

    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector._add_pill_rect("row").center(),
    )
    assert tool.tool_status.setup.nrows == 2
    assert tool.tool_status.setup.ncols == 1

    selector.resize(selector.sizeHint())
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=selector._add_pill_rect("column").center(),
    )
    assert tool.tool_status.setup.nrows == 2
    assert tool.tool_status.setup.ncols == 2


def test_figure_display_window_uses_safe_resize_callbacks(qtbot, monkeypatch) -> None:
    calls: list[tuple[QtCore.QObject, int, str, tuple[QtCore.QObject, ...]]] = []

    def record_single_shot(
        receiver: QtCore.QObject,
        msec: int,
        callback: Callable[[], None],
        *guards: QtCore.QObject,
    ) -> None:
        callback_name = getattr(
            getattr(callback, "func", callback), "__name__", type(callback).__name__
        )
        calls.append((receiver, msec, callback_name, guards))

    monkeypatch.setattr(erlab.interactive.utils, "single_shot", record_single_shot)
    window = figurecomposer_widgets._FigureComposerDisplayWindow(FigureSubplotsState())

    window.resize_to_setup(FigureSubplotsState())
    window._suppress_resize_signal = False
    window.resizeEvent(None)

    assert (window, 0, "_allow_resize_signal", ()) in calls
    assert (window, 0, "_emit_canvas_size_changed", (window.canvas,)) in calls
    window.close_from_owner()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_figure_display_window_skips_stale_resize_callbacks(qtbot) -> None:
    window = figurecomposer_widgets._FigureComposerDisplayWindow(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0)
    )
    emitted_sizes: list[tuple[float, float]] = []
    window.sigCanvasSizeChanged.connect(
        lambda width, height: emitted_sizes.append((width, height))
    )

    window.resizeEvent(None)
    generation = window._resize_signal_generation
    canvas = window.canvas
    window.close_from_owner()

    window._emit_canvas_size_changed(generation, canvas)

    assert emitted_sizes == []
    assert not window._resize_signal_pending
    assert window._suppress_resize_signal
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_figure_display_window_close_and_canvas_size_contracts(qtbot) -> None:
    window = figurecomposer_widgets._FigureComposerDisplayWindow(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0)
    )

    emitted_sizes: list[tuple[float, float]] = []
    window.sigCanvasSizeChanged.connect(
        lambda width, height: emitted_sizes.append((width, height))
    )
    typing.cast("typing.Any", window.figure)._original_dpi = 100.0
    window.canvas.resize(250, 125)
    window._emit_canvas_size_changed()
    assert emitted_sizes[-1] == (2.5, 1.25)

    window._suppress_resize_signal = True
    window._emit_canvas_size_changed()
    assert emitted_sizes == [(2.5, 1.25)]
    window._suppress_resize_signal = False
    typing.cast("typing.Any", window.figure)._original_dpi = 0.0
    window._emit_canvas_size_changed()
    assert emitted_sizes == [(2.5, 1.25)]

    assert not window._is_close_shortcut_event(None)
    close_key = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_W,
        QtCore.Qt.KeyboardModifier.MetaModifier,
    )
    assert window._is_close_shortcut_event(close_key)

    window.show_for_setup(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0),
        "detached figure",
        activate=False,
    )
    assert window.eventFilter(window.canvas, close_key)
    assert not window.isVisible()

    window.show()
    assert not window.close()
    assert not window.isVisible()

    window.show()
    window.close_from_owner()
    assert not window.isVisible()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    app_quit_window = figurecomposer_widgets._FigureComposerDisplayWindow(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0)
    )
    app_quit_window.show()
    erlab.interactive.utils._set_application_quit_requested(True)
    try:
        assert app_quit_window.close()
    finally:
        erlab.interactive.utils._set_application_quit_requested(False)
    assert not app_quit_window.isVisible()
    app_quit_window.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_figure_composer_managed_display_window_configures_save_shortcut(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *_args, **_kwargs: None,
    )
    save_calls: list[bool] = []
    controller = object.__new__(manager_workspace_io._WorkspaceIOController)
    controller._manager = types.SimpleNamespace(
        save=lambda *, native=True: save_calls.append(native) or True
    )
    tool._set_managed_secondary_window_callback(
        controller._install_workspace_save_shortcut
    )

    tool.show_figure_window(activate=False)
    figure_window = tool.figure_window

    save_shortcuts = [
        shortcut
        for shortcut in figure_window.findChildren(QtWidgets.QShortcut)
        if shortcut.objectName() == "managerWorkspaceSaveShortcut"
    ]
    assert len(save_shortcuts) == 1
    save_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.ShortcutOverride,
        QtCore.Qt.Key.Key_S,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert figure_window.eventFilter(figure_window.canvas, save_event)
    assert save_event.isAccepted()
    assert save_calls == [True]

    save_shortcuts[0].activated.emit()
    assert save_calls == [True, True]


def test_figure_composer_manual_redraw_controls(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)

    assert tool.show_figure_button.text() == "Show Plot"
    assert tool.auto_redraw_check.isChecked()
    assert tool.redraw_plot_button.text() == ""
    assert tool.redraw_plot_button.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
    )
    assert not tool.redraw_plot_button.icon().isNull()

    render_calls: list[tuple[object, dict[str, object]]] = []

    def record_render(*args, **kwargs) -> None:
        render_calls.append((args, kwargs))

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", record_render)

    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    tool.auto_redraw_check.setChecked(False)
    tool._update_current_operation(label="manual")

    assert tool.tool_status.operations[0].label == "manual"
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert tool._auto_redraw_dirty
    assert info_changed == [None]

    tool.redraw_plot_button.click()

    assert render_calls == [((tool,), {})]
    assert not tool._auto_redraw_dirty
    assert info_changed == [None, None]

    render_calls.clear()
    info_changed.clear()
    tool.auto_redraw_check.setChecked(False)
    tool._update_current_operation(label="catch up")
    tool.auto_redraw_check.setChecked(True)

    assert render_calls == [((tool,), {})]
    assert not tool._auto_redraw_dirty
    assert info_changed == [None, None]


def test_workspace_modified_state_updates_figure_display_window(qtbot) -> None:
    primary_window = QtWidgets.QMainWindow()
    secondary_window = QtWidgets.QMainWindow()
    qtbot.addWidget(primary_window)
    qtbot.addWidget(secondary_window)
    tool = types.SimpleNamespace(
        tool_name="Figure Composer",
        _tool_display_name="Figure 1",
        _managed_secondary_windows=lambda: (
            (secondary_window, "Figure Composer: Figure 1"),
        ),
    )
    node = types.SimpleNamespace(window=primary_window, tool_window=tool)
    controller = object.__new__(manager_workspace_io._WorkspaceIOController)
    controller._manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes={"figure_uid": node})
    )
    controller._node_window_state_applied = {}
    controller._pending_node_window_modified = {}

    controller._set_node_window_modified("figure_uid", True)

    assert primary_window.isWindowModified()
    assert secondary_window.isWindowModified()
    if sys.platform != "darwin":
        assert "[*]" in secondary_window.windowTitle()

    controller._set_node_window_modified("figure_uid", False)

    assert not primary_window.isWindowModified()
    assert not secondary_window.isWindowModified()


def test_figure_composer_canvas_resize_defers_draw(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)

    draw_idle_calls: list[object] = []
    info_changes: list[tuple[float, float]] = []

    def record_draw_idle(canvas) -> None:
        draw_idle_calls.append(canvas)

    monkeypatch.setattr(
        figurecomposer_widgets.FigureCanvas, "draw_idle", record_draw_idle
    )
    tool.sigInfoChanged.connect(
        lambda: info_changes.append(tool.tool_status.setup.figsize)
    )

    tool._figure_window_canvas_size_changed(4.0, 2.5)
    assert tool.tool_status.setup.figsize == (4.0, 2.5)
    assert np.isclose(tool.width_spin.value(), 4.0)
    assert np.isclose(tool.height_spin.value(), 2.5)
    assert draw_idle_calls == []
    assert info_changes == []

    tool._figure_window_canvas_size_changed(4.5, 3.0)
    assert tool.tool_status.setup.figsize == (4.5, 3.0)
    qtbot.waitUntil(lambda: len(draw_idle_calls) == 1, timeout=1000)

    assert info_changes == [(4.5, 3.0)]


def test_figure_composer_plot_window_undo_redo_resizes_canvas(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    figure_window = tool.figure_window
    qtbot.wait_until(lambda: figure_window.isVisible(), timeout=5000)
    tool._reset_history_stack()

    initial_size = tool.tool_status.setup.figsize
    base_dpi = float(typing.cast("typing.Any", figure_window.figure)._original_dpi)
    target_size = (initial_size[0] + 0.75, initial_size[1] + 0.5)
    size_delta = figure_window.size() - figure_window.canvas.size()
    figure_window.resize(
        round(target_size[0] * base_dpi) + size_delta.width(),
        round(target_size[1] * base_dpi) + size_delta.height(),
    )
    qtbot.wait_until(
        lambda: (
            np.isclose(tool.tool_status.setup.figsize[0], target_size[0], atol=0.03)
            and np.isclose(tool.tool_status.setup.figsize[1], target_size[1], atol=0.03)
        ),
        timeout=5000,
    )
    resized_size = tool.tool_status.setup.figsize

    figure_window.toolbar.back()
    qtbot.wait_until(
        lambda: (
            np.isclose(tool.tool_status.setup.figsize[0], initial_size[0], atol=0.03)
            and np.isclose(
                tool.tool_status.setup.figsize[1], initial_size[1], atol=0.03
            )
            and abs(figure_window.canvas.width() - round(initial_size[0] * base_dpi))
            <= 2
            and abs(figure_window.canvas.height() - round(initial_size[1] * base_dpi))
            <= 2
        ),
        timeout=5000,
    )

    figure_window.toolbar.forward()
    qtbot.wait_until(
        lambda: (
            np.isclose(tool.tool_status.setup.figsize[0], resized_size[0], atol=0.03)
            and np.isclose(
                tool.tool_status.setup.figsize[1], resized_size[1], atol=0.03
            )
            and abs(figure_window.canvas.width() - round(resized_size[0] * base_dpi))
            <= 2
            and abs(figure_window.canvas.height() - round(resized_size[1] * base_dpi))
            <= 2
        ),
        timeout=5000,
    )


def test_figure_composer_resize_render_is_cancelled_on_close(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    info_changes: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changes.append(None))

    tool._queue_figure_resize_render()
    generation = tool._figure_resize_render_generation
    tool.close()
    tool._run_queued_figure_resize_render(generation)

    assert info_changes == []


def test_figure_composer_show_defers_figure_window(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    calls: list[bool] = []

    def record_show_figure_window(*, activate: bool = True) -> None:
        calls.append(activate)

    monkeypatch.setattr(tool, "show_figure_window", record_show_figure_window)

    tool.show()

    assert calls == []
    qtbot.waitUntil(lambda: calls == [False], timeout=1000)


def test_figure_composer_hide_cancels_deferred_figure_window(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    calls: list[bool] = []

    def record_show_figure_window(*, activate: bool = True) -> None:
        calls.append(activate)

    monkeypatch.setattr(tool, "show_figure_window", record_show_figure_window)

    tool.show()
    tool.hide()
    qtbot.wait(20)

    assert calls == []


def test_figure_composer_show_composer_from_figure_window(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)

    calls: list[str] = []
    monkeypatch.setattr(tool, "isMinimized", lambda: False)
    monkeypatch.setattr(tool, "show", lambda: calls.append("show"))
    monkeypatch.setattr(tool, "raise_", lambda: calls.append("raise"))
    monkeypatch.setattr(tool, "activateWindow", lambda: calls.append("activate"))

    tool._show_composer_from_figure_window()

    assert calls == ["show", "raise", "activate"]

    calls.clear()
    monkeypatch.setattr(tool, "isMinimized", lambda: True)
    monkeypatch.setattr(tool, "showNormal", lambda: calls.append("showNormal"))

    tool._show_composer_from_figure_window()

    assert calls == ["showNormal", "raise", "activate"]

    calls.clear()
    with monkeypatch.context() as context:
        context.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: False)
        tool._show_composer_from_figure_window()
    assert calls == []


def test_figure_composer_tool_edge_state_contracts(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(
        "erlab.interactive._figurecomposer._tool._render_preview",
        lambda *_args, **_kwargs: None,
    )
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")

    with pytest.raises(ValueError, match="At least one source"):
        FigureComposerTool.from_sources({}, sources=())

    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="known", label="duplicate"),
                FigureSourceState(name="missing", label="duplicate"),
            ),
            operations=(),
            primary_source="known",
        ),
        source_data={"known": data, "extra": data},
    )
    qtbot.addWidget(tool)

    assert tool._source_names() == ("known", "missing")
    assert tool._source_display_names(("known", "missing"))
    assert tool._source_tooltip("known")
    tool._refresh_source_list()
    source_names = {
        tool.source_list.topLevelItem(row).data(0, QtCore.Qt.ItemDataRole.UserRole)
        for row in range(tool.source_list.topLevelItemCount())
    }
    assert source_names == {"known", "extra", "missing"}

    assert not tool._set_recipe_figsize_from_canvas(
        *tool.tool_status.setup.figsize,
        draw=False,
        emit_info=False,
    )
    tool._updating_controls = True
    try:
        tool._figure_window_canvas_size_changed(8.0, 6.0)
    finally:
        tool._updating_controls = False
    assert tool.tool_status.setup.figsize != (8.0, 6.0)
    typing.cast("typing.Any", tool.figure)._original_dpi = 0.0
    assert not tool._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
    window = tool.figure_window
    tool._hide_figure_window()
    tool._close_figure_window()
    assert tool._figure_window is None
    window.sigCanvasSizeChanged.emit(8.0, 6.0)
    assert tool.tool_status.setup.figsize != (8.0, 6.0)
    tool._close_figure_window()

    assert not tool.remove_operation_button.isEnabled()
    tool._target_current_operation_all_axes()
    tool._target_current_operation_valid_axes()
    tool._remove_current_operation()
    tool._duplicate_current_operation()
    tool._move_current_operation(-1)

    custom_operation = FigureOperationState.custom(
        label="custom",
        code="pass",
        trusted=True,
    )
    tool.add_operation(custom_operation)
    tool._target_current_operation_all_axes()
    tool._target_current_operation_valid_axes()
    assert tool.tool_status.operations[0].axes.axes == ()
    item = tool.operation_list.item(0)
    assert item is not None
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not tool.tool_status.operations[0].enabled
    tool.operation_list.clearSelection()
    tool.operation_list.setCurrentRow(-1)
    tool._update_step_action_buttons()
    assert not tool.remove_operation_button.isEnabled()

    tool.axes_selector.set_selected_axes((), emit=False)
    assert tool._selected_axes_state().axes == ((0, 0),)
    old_setup = tool.tool_status.setup
    tool.width_ratios_edit.setText("0")
    tool._setup_controls_changed()
    assert tool.tool_status.setup == old_setup
    tool._updating_controls = True
    try:
        tool._size_mm_controls_changed()
    finally:
        tool._updating_controls = False

    span_00 = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    span_01 = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )
    outside_span = FigureGridSpecSpanState(
        row_start=1,
        row_stop=2,
        col_start=0,
        col_stop=1,
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        label="Root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(axes_id="ax0", span=span_00),
            FigureGridSpecAxesState(axes_id="ax1", span=span_01),
            FigureGridSpecAxesState(axes_id="outside", span=outside_span),
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes_ids=("missing",)),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(grid_tool)

    grid_tool._active_gridspec_grid_id = "missing"
    grid_tool._sync_active_grid_controls(grid_tool.tool_status.setup)
    assert grid_tool._active_gridspec_grid_id == "root"
    grid_tool._active_gridspec_grid_id = "missing"
    grid_tool._refresh_gridspec_editor()
    assert grid_tool._active_gridspec_grid_id == "root"
    grid_tool._refresh_gridspec_status(root)
    assert grid_tool.gridspec_status_label.text()
    grid_tool._gridspec_open_grid("missing")
    assert grid_tool._active_gridspec_grid_id == "root"
    grid_tool.gridspec_layout_widget.set_selected_region("")
    grid_tool._gridspec_open_selected_grid()
    grid_tool._gridspec_open_parent_grid()

    grid_tool._gridspec_region_changed("ax0", outside_span)
    grid_tool._gridspec_region_changed("ax0", span_01)
    grid_tool._add_gridspec_region(outside_span, "axes")
    grid_tool._add_gridspec_region(span_00, "axes")
    grid_tool._add_gridspec_region(span_01, "grid")

    grid_tool.gridspec_axes_selector.set_selected_axes_ids((), emit=False)
    assert grid_tool._selected_axes_state().axes_ids == ("ax0",)
    grid_tool._target_current_operation_valid_axes()
    assert grid_tool.tool_status.operations[0].axes.axes_ids == ("ax0",)
    grid_tool._target_current_operation_all_axes()
    assert set(grid_tool.tool_status.operations[0].axes.axes_ids) == {
        "ax0",
        "ax1",
        "outside",
    }
    grid_tool._active_gridspec_grid_id = "missing"
    assert grid_tool._nearest_gridspec_axes_after_delete("ax0") == ""
    grid_tool._active_gridspec_grid_id = "root"
    assert grid_tool._nearest_gridspec_axes_after_delete("missing") == ""

    child_with_invalid_axis = FigureGridSpecGridState(
        grid_id="child",
        label="child",
        nrows=1,
        ncols=1,
        span=span_00,
        axes=(FigureGridSpecAxesState(axes_id="child-outside", span=outside_span),),
    )
    nested_root = FigureGridSpecGridState(
        grid_id="root",
        label="Root",
        nrows=1,
        ncols=1,
        child_grids=(child_with_invalid_axis,),
    )
    grid_tool._recipe = grid_tool.tool_status.model_copy(
        update={
            "setup": FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=nested_root),
            )
        }
    )
    grid_tool._refresh_gridspec_status(nested_root)
    assert grid_tool.gridspec_status_label.text()


def test_figure_composer_editor_factories_preserve_mixed_and_missing_values(
    qtbot,
) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)
    committed: list[typing.Any] = []

    source_combo = tool._source_combo(
        ("data",),
        "stale_source",
        committed.append,
    )
    assert source_combo.findData("stale_source") >= 0
    assert source_combo.findData("data") >= 0
    _activate_combo_index(source_combo, source_combo.findData("data"))
    assert committed[-1] == "data"

    mixed_source_combo = tool._source_combo(
        ("data",),
        "data",
        committed.append,
        mixed=True,
    )
    before = list(committed)
    assert mixed_source_combo.currentData() is _editor_controls.MIXED_VALUE
    _activate_combo_index(mixed_source_combo, mixed_source_combo.currentIndex())
    assert committed == before

    name_combo = tool._optional_name_combo(
        ("kx",),
        "missing_dim",
        "auto",
        committed.append,
    )
    assert name_combo.findData("missing_dim") >= 0
    _activate_combo_index(name_combo, name_combo.findData(None))
    assert committed[-1] is None

    mixed_name_combo = tool._optional_name_combo(
        ("kx",),
        None,
        "auto",
        committed.append,
        mixed=True,
    )
    before = list(committed)
    assert mixed_name_combo.currentData() is _editor_controls.MIXED_VALUE
    _activate_combo_index(mixed_name_combo, mixed_name_combo.currentIndex())
    assert committed == before

    mixed_check = tool._check_box(False, committed.append, mixed=True)
    assert mixed_check.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    before = list(committed)
    mixed_check.stateChanged.emit(int(QtCore.Qt.CheckState.PartiallyChecked.value))
    assert committed == before
    mixed_check.setCheckState(QtCore.Qt.CheckState.Checked)
    assert committed[-1] is True


def test_figure_composer_axes_selection_guards_recipe_updates(
    qtbot,
    monkeypatch,
) -> None:
    data = _figure_composer_profile_source("data")
    line_operation = FigureOperationState.line(
        label="profile",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(line_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    tool._updating_controls = True
    try:
        tool._axes_selection_changed(((0, 1),))
    finally:
        tool._updating_controls = False
    assert tool.tool_status.operations[0].axes.axes == ((0, 0),)

    tool._axes_selection_changed(())
    assert tool.tool_status.operations[0].axes.axes == ((0, 0),)
    assert render_calls == []

    tool._axes_selection_changed(((0, 1),))
    assert tool.tool_status.operations[0].axes.axes == ((0, 1),)
    assert render_calls == []
    assert tool._preview_render_update_pending
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)

    tool.axes_expression_edit.setText("axs[0, :]")
    tool._axes_expression_changed()
    assert tool.tool_status.operations[0].axes.expression == "axs[0, :]"
    assert len(render_calls) == 1
    qtbot.waitUntil(lambda: len(render_calls) == 2, timeout=1000)

    tool.add_operation(
        FigureOperationState.custom(label="code", code="pass", trusted=True)
    )
    tool.operation_list.setCurrentRow(1)
    tool._axes_selection_changed(((0, 0),))
    assert tool.tool_status.operations[1].axes.axes == ()
    tool.axes_expression_edit.setText("axs")
    tool._axes_expression_changed()
    assert tool.tool_status.operations[1].axes.expression == ""


def test_figure_composer_gridspec_axes_selection_guards_recipe_updates(
    qtbot,
    monkeypatch,
) -> None:
    data = _figure_composer_profile_source("data")
    root = FigureGridSpecGridState(
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="left",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="right",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    line_operation = FigureOperationState.line(
        label="profile",
        source="data",
        axes=FigureAxesSelectionState(axes_ids=("left",)),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(line_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    tool.gridspec_axes_selector.set_selected_axes_ids(("right",), emit=False)
    tool._updating_controls = True
    try:
        tool._gridspec_axes_selection_changed()
    finally:
        tool._updating_controls = False
    assert tool.tool_status.operations[0].axes.axes_ids == ("left",)

    tool._gridspec_axes_selection_changed()
    assert tool.tool_status.operations[0].axes.axes_ids == ("right",)
    assert render_calls == []
    assert tool._preview_render_update_pending
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)

    tool.add_operation(
        FigureOperationState.custom(label="code", code="pass", trusted=True)
    )
    tool.operation_list.setCurrentRow(1)
    tool.gridspec_axes_selector.set_selected_axes_ids(("left",), emit=False)
    tool._gridspec_axes_selection_changed()
    assert tool.tool_status.operations[1].axes.axes_ids == ()


def test_figure_composer_recipe_codegen_and_loaded_custom_code_trust(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    assert tool.tool_status.setup.figsize == FigureSubplotsState().figsize
    code = tool.generated_code()
    assert "tools[" not in code
    assert "_manager" not in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert "fig" in namespace

    status = tool.tool_status
    restored = FigureComposerTool(data)
    qtbot.addWidget(restored)
    restored.tool_status = status
    assert restored.tool_status.model_dump() == status.model_dump()

    custom_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=status.sources,
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code="ax.set_title('trusted')",
                    trusted=True,
                ),
            ),
            primary_source=status.primary_source,
        ),
    )
    qtbot.addWidget(custom_tool)

    loaded = erlab.interactive.utils.ToolWindow.from_dataset(custom_tool.to_dataset())
    qtbot.addWidget(loaded)
    assert loaded.tool_status.operations[0].trusted is False


def test_figure_composer_undo_redo_setup_state(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    initial = tool.tool_status

    tool.nrows_spin.setValue(initial.setup.nrows + 1)

    assert tool.undoable
    assert tool.tool_status.setup.nrows == initial.setup.nrows + 1

    tool.undo()
    assert tool.tool_status.model_dump(mode="json") == initial.model_dump(mode="json")
    assert tool.redoable

    tool.redo()
    assert tool.tool_status.setup.nrows == initial.setup.nrows + 1


def test_figure_composer_subplot_controls_accept_more_than_twelve(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)

    assert np.isinf(tool.nrows_spin.maximum())
    assert np.isinf(tool.ncols_spin.maximum())

    tool.nrows_spin.setValue(13)
    assert tool.nrows_spin.value() == 13
    assert tool.tool_status.setup.nrows == 13

    tool.nrows_spin.setValue(1)
    tool.ncols_spin.setValue(13)
    assert tool.ncols_spin.value() == 13
    assert tool.tool_status.setup.ncols == 13


def test_figure_composer_gridspec_controls_accept_more_than_twelve(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")

    tool.nrows_spin.setValue(13)
    assert tool.nrows_spin.value() == 13
    assert tool.tool_status.setup.gridspec.root.nrows == 13

    tool.nrows_spin.setValue(1)
    tool.ncols_spin.setValue(13)
    assert tool.ncols_spin.value() == 13
    assert tool.tool_status.setup.gridspec.root.ncols == 13


def test_figure_composer_navigation_updates_recipe_limits(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    qtbot.addWidget(tool.figure_window)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool.canvas.draw()
    tool._reset_history_stack()

    initial = tool.tool_status
    layout_axes = figurecomposer_rendering._live_layout_axes(tool)
    assert layout_axes is not None
    axis = figurecomposer_rendering._iter_axes(layout_axes)[0]
    axis.set_xlim(0.25, 0.75)

    tool._figure_window_navigation_changed({axis: (True, False)})

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.METHOD
    assert operation.method_family == FigureMethodFamily.AXES
    assert operation.method_name == "set_xlim"
    assert operation.method_args == pytest.approx((0.25, 0.75))
    assert operation.axes == FigureAxesSelectionState(axes=((0, 0),))

    toolbar = tool.figure_window.toolbar
    assert toolbar._actions["back"].isEnabled()
    assert not toolbar._actions["forward"].isEnabled()

    toolbar.back()

    assert tool.tool_status == initial
    assert not toolbar._actions["back"].isEnabled()
    assert toolbar._actions["forward"].isEnabled()

    toolbar.forward()

    assert tool.tool_status.operations[-1].method_name == "set_xlim"
    assert tool.tool_status.operations[-1].method_args == pytest.approx((0.25, 0.75))

    axis.set_xlim(0.25, 0.75)
    unchanged_status = tool.tool_status
    tool._figure_window_navigation_changed({axis: (True, False)})
    assert tool.tool_status == unchanged_status

    axis.set_xlim(0.1, 0.9)
    tool._figure_window_navigation_changed({axis: (True, False)})
    assert tool.tool_status.operations[-1].method_name == "set_xlim"
    assert tool.tool_status.operations[-1].method_args == pytest.approx((0.1, 0.9))

    axis.set_ylim(0.2, 0.8)
    tool._figure_window_navigation_changed({axis: (False, True)})
    assert tool.tool_status.operations[-1].method_name == "set_ylim"
    assert tool.tool_status.operations[-1].method_args == pytest.approx((0.2, 0.8))

    status = tool.tool_status
    tool._updating_controls = True
    try:
        tool._figure_window_navigation_changed({axis: (True, True)})
    finally:
        tool._updating_controls = False
    assert tool.tool_status == status

    with pytest.MonkeyPatch.context() as context:
        context.setattr(
            figurecomposer_tool_module, "_live_layout_axes", lambda _tool: None
        )
        tool._figure_window_navigation_changed({axis: (True, True)})
    assert tool.tool_status == status


def test_figure_composer_navigation_helper_edges(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)

    axis = object()
    assert tool._navigation_axis_selection({"main": axis}, axis) == (
        FigureAxesSelectionState(axes_ids=("main",))
    )
    assert tool._navigation_axis_selection({"main": object()}, axis) is None
    assert tool._navigation_axis_selection(object(), axis) is None

    layout_axes = np.empty((1, 2), dtype=object)
    layout_axes[0, 0] = object()
    layout_axes[0, 1] = axis
    assert tool._navigation_axis_selection(layout_axes, axis) == (
        FigureAxesSelectionState(axes=((0, 1),))
    )

    class AxisWithLimits:
        def get_xlim(self) -> tuple[float, float]:
            return (0.1, 0.9)

        def get_ylim(self) -> tuple[float, float]:
            return (0.2, 0.8)

    assert tool._navigation_axis_limit_updates(
        AxisWithLimits(), x_changed=True, y_changed=True
    ) == (("set_xlim", (0.1, 0.9)), ("set_ylim", (0.2, 0.8)))
    assert (
        tool._navigation_axis_limit_updates(object(), x_changed=True, y_changed=True)
        == ()
    )

    matching_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_xlim",
        axes=FigureAxesSelectionState(axes_ids=("main",)),
    )
    assert (
        tool._matching_navigation_limit_operation_index(
            (matching_operation,),
            "set_xlim",
            FigureAxesSelectionState(axes_ids=("main",)),
        )
        == 0
    )
    assert (
        tool._matching_navigation_limit_operation_index(
            (matching_operation,),
            "set_ylim",
            FigureAxesSelectionState(axes_ids=("main",)),
        )
        is None
    )
    tool._apply_live_figure_operation_updates((), set())
    assert tool.tool_status.operations == ()


def test_figure_composer_navigation_ignores_colorbar_axes(qtbot) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_slices(
        label="data",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"colorbar": "right"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    qtbot.addWidget(tool.figure_window)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool.canvas.draw()
    tool._reset_history_stack()

    colorbar_axes = [axis for axis in tool.figure.axes if hasattr(axis, "_colorbar")]
    assert colorbar_axes

    initial = tool.tool_status
    colorbar_axes[0].set_ylim(0.2, 0.8)
    tool._figure_window_navigation_changed({colorbar_axes[0]: (False, True)})

    assert tool.tool_status == initial
    assert not tool.undoable


def test_figure_composer_colorbar_drag_updates_recipe_limits(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": np.arange(3), "x": np.arange(4)},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="data",
        sources=("data",),
    ).model_copy(update={"colorbar": "right"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    qtbot.addWidget(tool.figure_window)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool.canvas.draw()
    tool._reset_history_stack()

    colorbar_axis = next(
        axis for axis in tool.figure.axes if hasattr(axis, "_colorbar")
    )
    mappable = typing.cast("typing.Any", colorbar_axis)._colorbar.mappable
    previous_clim = mappable.get_clim()

    operations = tool.tool_status.operations
    assert tool._plot_slices_mappable_target(object(), operations) is None
    bad_panel_mappable = type("BadPanelMappable", (), {})()
    setattr(
        bad_panel_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        operation.operation_id,
    )
    setattr(
        bad_panel_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
        ("bad", 0),
    )
    assert tool._plot_slices_mappable_target(bad_panel_mappable, operations) is None
    missing_operation_mappable = type("MissingOperationMappable", (), {})()
    setattr(
        missing_operation_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        "missing",
    )
    setattr(
        missing_operation_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
        (0, 0),
    )
    assert (
        tool._plot_slices_mappable_target(missing_operation_mappable, operations)
        is None
    )
    assert (
        tool._operation_with_colorbar_clim(operation, (99, 99), (1.0, 5.0)) == operation
    )

    initial = tool.tool_status
    tool._figure_window_colorbar_changed({object(): (0.0, 1.0)})
    assert tool.tool_status == initial
    tool._updating_controls = True
    try:
        tool._figure_window_colorbar_changed({mappable: (1.0, 5.0)})
    finally:
        tool._updating_controls = False
    assert tool.tool_status == initial

    mappable.set_clim(1.0, 5.0)
    tool.figure_window.toolbar._commit_colorbar_clims({mappable: previous_clim})

    updated = tool.tool_status.operations[0]
    assert updated.vmin == pytest.approx(1.0)
    assert updated.vmax == pytest.approx(5.0)
    assert tool.undoable

    unchanged = tool.tool_status
    tool._figure_window_colorbar_changed({mappable: (1.0, 5.0)})
    assert tool.tool_status == unchanged

    tool.figure_window.toolbar.back()

    assert tool.tool_status.operations[0].vmin is None
    assert tool.tool_status.operations[0].vmax is None
    assert tool.redoable

    tool.figure_window.toolbar.forward()

    assert tool.tool_status.operations[0].vmin == pytest.approx(1.0)
    assert tool.tool_status.operations[0].vmax == pytest.approx(5.0)


def test_figure_composer_colorbar_drag_updates_panel_style_limits(qtbot) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_slices(
        label="data",
        sources=("data",),
        slice_dim="eV",
        slice_values=(-0.5, 0.0),
        axes=FigureAxesSelectionState(expression="axs[0, :]"),
    ).model_copy(update={"colorbar": "right"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    qtbot.addWidget(tool.figure_window)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool.canvas.draw()
    tool._reset_history_stack()

    colorbar_axis = next(
        axis for axis in tool.figure.axes if hasattr(axis, "_colorbar")
    )
    mappable = typing.cast("typing.Any", colorbar_axis)._colorbar.mappable
    previous_clim = mappable.get_clim()

    mappable.set_clim(0.25, 0.75)
    tool.figure_window.toolbar._commit_colorbar_clims({mappable: previous_clim})

    updated = tool.tool_status.operations[0]
    assert updated.vmin is None
    assert updated.vmax is None
    assert updated.panel_styles_enabled
    assert len(updated.panel_styles) == 1
    style = updated.panel_styles[0]
    assert style.map_index == 0
    assert style.slice_index == 0
    assert style.vmin == pytest.approx(0.25)
    assert style.vmax == pytest.approx(0.75)


def test_figure_composer_plot_slices_mappable_tagging_edges() -> None:
    operation = FigureOperationState.plot_slices(label="data", sources=("data",))
    key = figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, "panel")
    figure = Figure()
    axis = figure.subplots()
    old_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    old_ids = figurecomposer_plot_slices._axis_mappable_ids((axis,))

    figurecomposer_plot_slices._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key, figurecomposer_plot_slices._PlotSlicesPanelKey(0, 1, "extra")),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )

    new_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    figurecomposer_plot_slices._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key,),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )
    assert (
        getattr(
            new_mappable,
            figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        )
        == operation.operation_id
    )
    assert getattr(
        new_mappable,
        figurecomposer_plot_slices._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
    ) == (0, 0)


def test_figure_composer_custom_code_codegen_namespace(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "ax.set_title(str(np.array([1])[0]))\n"
                        "fig.__dict__['_eplt_name'] = eplt.__name__"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import numpy as np" in code
    assert "import erlab.plotting as eplt" in code
    assert "ax = axs[0, 0]" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_title() == "1"
    assert namespace["fig"].__dict__["_eplt_name"] == "erlab.plotting"


def test_figure_composer_custom_code_codegen_gridspec_axes_alias(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="main-axis",
                                label="main_axis",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "ax.set_title('main')\naxs['main-axis'].set_xlabel('energy')"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "axs = {" in code
    assert "'main-axis': main_axis" in code
    assert "ax = main_axis" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_title() == "main"
    assert namespace["fig"].axes[0].get_xlabel() == "energy"


def test_figure_composer_reports_and_clears_render_errors(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    operation = FigureOperationState.custom(
        label="custom",
        code="raise RuntimeError('boom')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" in item.text()
    assert "RuntimeError: boom" in item.toolTip()
    assert "Render error: RuntimeError: boom" in tool.source_status_label.text()

    tool._replace_operation(
        0,
        operation.model_copy(update={"code": "ax.set_title('ok')"}),
    )

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" not in item.text()
    assert "RuntimeError: boom" not in item.toolTip()
    assert "Render error" not in tool.source_status_label.text()


def test_figure_composer_untrusted_custom_code_reports_render_error(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    operation = FigureOperationState.custom(
        label="custom",
        code="ax.set_title('loaded')",
        trusted=False,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" in item.text()
    assert "Custom code is not trusted" in item.toolTip()
    assert "Enable Trusted to render it" in tool.source_status_label.text()

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")
    trusted_check = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QCheckBox, "figureComposerCustomCodeTrustedCheck"
    )
    assert trusted_check is not None
    assert not trusted_check.isChecked()

    trusted_check.setChecked(True)

    assert tool.tool_status.operations[0].trusted is True
    qtbot.waitUntil(lambda: not tool._operation_render_errors, timeout=1000)
    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" not in item.text()
    assert "Custom code is not trusted" not in item.toolTip()
    assert "Render error" not in tool.source_status_label.text()


def test_figure_composer_editor_input_errors_mark_and_clear_invalid_steps(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    operation = tool.tool_status.operations[0]
    edit = QtWidgets.QLineEdit(tool)
    tool._connect_line_edit_finished(
        edit,
        lambda text: tool._update_current_operation(
            extra_kwargs=figurecomposer_text._dict_from_text(text)
        ),
    )

    edit.setText("{alpha: 0.5}")
    edit.editingFinished.emit()

    assert tool._operation_has_invalid_input(operation)
    item = tool.operation_list.item(0)
    assert item is not None
    assert "Invalid input:" in item.toolTip()
    assert "Invalid input:" in tool.source_status_label.text()
    with pytest.raises(ValueError, match="invalid step inputs"):
        tool.generated_code()

    check = QtWidgets.QCheckBox(tool)
    tool._connect_editor_signal(
        check,
        check.toggled,
        lambda checked: tool._update_current_operation(transpose=checked),
    )
    check.setChecked(not operation.transpose)

    assert tool._operation_has_invalid_input(operation)

    edit.setText("alpha=0.5")
    edit.editingFinished.emit()

    assert not tool._operation_has_invalid_input(operation)
    assert tool.tool_status.operations[0].extra_kwargs == {"alpha": 0.5}
    item = tool.operation_list.item(0)
    assert item is not None
    assert "Invalid input:" not in item.toolTip()
    assert "Invalid input:" not in tool.source_status_label.text()


def test_figure_composer_editor_signal_allows_callback_to_delete_sender(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    operation = tool.tool_status.operations[0]
    edit = QtWidgets.QLineEdit(tool)
    edit.setObjectName("figureComposerDeletedSenderEdit")
    input_key = edit.objectName()
    tool._set_operation_input_errors({operation.operation_id: {input_key: "old error"}})

    def delete_sender() -> None:
        edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    tool._connect_editor_signal(edit, edit.editingFinished, delete_sender)
    edit.editingFinished.emit()

    assert not erlab.interactive.utils.qt_is_valid(edit)
    assert tool._editor_input_error_key(edit) == f"anonymous:{id(edit)}"
    assert not tool._operation_has_invalid_input(operation)

    error_edit = QtWidgets.QLineEdit(tool)
    error_edit.setObjectName("figureComposerDeletedErrorSenderEdit")

    def delete_sender_with_error() -> None:
        error_edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        raise figurecomposer_text.FigureComposerInputError("new error")

    tool._connect_editor_signal(
        error_edit, error_edit.editingFinished, delete_sender_with_error
    )
    error_edit.editingFinished.emit()

    assert not erlab.interactive.utils.qt_is_valid(error_edit)
    assert tool._operation_has_invalid_input(operation)
    assert tool._operation_input_error_text(operation) == "new error"


def test_figure_composer_editor_control_changes_defer_preview_render(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    plain = QtWidgets.QPlainTextEdit(tool)
    tool._connect_plain_text_changed(
        plain,
        lambda text: tool._update_current_operation(label=text),
    )
    spin = QtWidgets.QSpinBox(tool)
    tool._connect_value_signal(
        spin,
        spin.valueChanged,
        int,
        lambda value: tool._update_current_operation(label=f"value {value}"),
    )

    plain.setPlainText("typed")
    spin.setValue(1)

    assert tool.tool_status.operations[0].label == "value 1"
    assert render_calls == []
    assert tool._preview_render_update_pending

    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert not tool._preview_render_update_pending


def test_figure_composer_disabled_step_edits_do_not_render_or_queue(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    operation_item = tool.operation_list.item(0)
    assert operation_item is not None
    operation_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].enabled is False
    assert render_calls == [(tool,)]
    assert info_changed == [None]

    render_calls.clear()
    info_changed.clear()
    edit = QtWidgets.QLineEdit(tool)
    tool._connect_line_edit_finished(
        edit,
        lambda text: tool._update_current_operation(label=text),
    )
    edit.setText("disabled edit")
    edit.editingFinished.emit()

    assert tool.tool_status.operations[0].label == "disabled edit"
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None]

    tool._update_current_operation(label="programmatic disabled edit")

    assert tool.tool_status.operations[0].label == "programmatic disabled edit"
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None, None]

    operation_item = tool.operation_list.item(0)
    assert operation_item is not None
    operation_item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].enabled is True
    assert render_calls == [(tool,)]
    assert info_changed == [None, None, None]


def test_figure_composer_defaults_follow_stylesheet_rcparams(
    monkeypatch,
    restore_interactive_options,
) -> None:
    _set_figure_stylesheets(["classic"])
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("classic",),
    )

    with mpl_style.context(["classic"]):
        expected_figsize = tuple(
            float(value) for value in mpl.rcParams["figure.figsize"]
        )
        expected_dpi = float(mpl.rcParams["figure.dpi"])
        expected_layout = _expected_layout_from_rcparams()
        expected_export_dpi = mpl.rcParams["savefig.dpi"]
        if expected_export_dpi != "figure":
            expected_export_dpi = float(expected_export_dpi)
        expected_transparent = bool(mpl.rcParams["savefig.transparent"])
        expected_bbox = mpl.rcParams["savefig.bbox"]

    setup = FigureSubplotsState()
    export = FigureExportState()

    assert setup.figsize == expected_figsize
    assert setup.dpi == expected_dpi
    assert setup.layout == expected_layout
    assert setup.sharex == "col"
    assert setup.sharey == "row"
    assert setup.width_ratios == ()
    assert setup.height_ratios == ()
    assert export.dpi == expected_export_dpi
    assert export.transparent is expected_transparent
    assert export.bbox_inches == expected_bbox


def test_figure_composer_defaults_skip_unavailable_stylesheets(
    monkeypatch,
    restore_interactive_options,
) -> None:
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("classic", "missing-style"),
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "available_stylesheets",
        lambda names=(): frozenset({"classic"}),
    )
    with mpl_style.context(["classic"]):
        expected_figsize = tuple(
            float(value) for value in mpl.rcParams["figure.figsize"]
        )

    assert figurecomposer_defaults._available_configured_stylesheets() == ("classic",)
    assert figurecomposer_defaults._unavailable_configured_stylesheets() == (
        "missing-style",
    )
    assert figurecomposer_defaults._default_figsize() == expected_figsize


@pytest.mark.parametrize("dpi", [0, -1])
def test_figure_composer_export_dpi_must_be_positive(dpi: int) -> None:
    with pytest.raises(ValueError, match="export dpi must be positive"):
        FigureExportState(dpi=dpi)


def test_figure_composer_generated_code_uses_available_stylesheets(
    qtbot,
    monkeypatch,
    restore_interactive_options,
    tmp_path: Path,
) -> None:
    style_name = "erlab-test-available-style"
    style_dir = tmp_path / "stylelib"
    style_dir.mkdir()
    (style_dir / f"{style_name}.mplstyle").write_text("axes.facecolor: white\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
        import matplotlib.style.core as mpl_style_core

    mpl_style_core.USER_LIBRARY_PATHS.append(str(style_dir))
    try:
        mpl_style.reload_library()
        monkeypatch.setattr(
            figurecomposer_defaults,
            "_configured_stylesheets",
            lambda: (style_name, "missing-style"),
        )
        data = xr.DataArray(
            np.arange(4.0),
            dims=("x",),
            coords={"x": np.arange(4.0)},
            name="data",
        )
        tool = FigureComposerTool(data)
        qtbot.addWidget(tool)

        code = tool.generated_code()

        assert f"plt.style.use(['{style_name}'])" in code
        assert "# Skipped unavailable stylesheets: 'missing-style'" in code
        assert tool.preview_pixmap is None
        assert tool.refresh_preview_pixmap() is None
        assert tool.refresh_preview_pixmap(allow_offscreen=True) is not None
        assert tool.preview_pixmap is not None
        namespace = {"data": data}
        with mpl.rc_context():
            exec(code, namespace)  # noqa: S102
        namespace["plt"].close(namespace["fig"])
    finally:
        mpl_style_core.USER_LIBRARY_PATHS.remove(str(style_dir))
        mpl_style.reload_library()


def test_figure_composer_generated_code_loads_user_stylesheets(
    qtbot,
    monkeypatch,
    restore_interactive_options,
) -> None:
    @contextlib.contextmanager
    def style_context(_stylesheets):
        yield

    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("user-style", "missing-style"),
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "available_stylesheets",
        lambda names=(): frozenset({"user-style"}),
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "stylesheets_require_erlab_plotting",
        lambda names: False,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "stylesheets_require_user_stylesheets",
        lambda names: "user-style" in names,
    )
    monkeypatch.setattr(figurecomposer_defaults.mpl_style, "context", style_context)
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    code = tool.generated_code()

    user_import = "import erlab.interactive._stylesheets as _erlab_stylesheets"
    user_load = "_erlab_stylesheets.load_user_stylesheets()"
    style_use = "plt.style.use(['user-style'])"
    assert user_import in code
    assert user_load in code
    assert style_use in code
    assert code.index(user_import) < code.index(user_load) < code.index(style_use)
    assert "# Skipped unavailable stylesheets: 'missing-style'" in code


def test_figure_composer_preview_uses_live_canvas_without_rerender(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    live_figure = tool.figure
    live_canvas = tool.canvas

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("canvas preview refresh should not rerender the recipe")

    monkeypatch.setattr(figurecomposer_tool_module, "_render_into_figure", fail_render)

    preview = tool.refresh_preview_pixmap()

    assert preview is not None
    assert not preview.isNull()
    assert tool.figure is live_figure
    assert tool.canvas is live_canvas


def test_figure_composer_hidden_preview_does_not_create_window(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_preview(tool, show_window=False)

    assert tool._figure_window is None
    assert tool._operation_render_errors == {}


def test_figure_composer_rendering_helpers_cover_output_paths(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(2.0)},
        name="data",
    )
    invalid_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_xlabel",
        axes=FigureAxesSelectionState(axes=((1, 0),)),
    ).model_copy(update={"method_args": ("invalid",)})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1, layout="compressed"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(invalid_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._set_creation_layout_engine(figure, "compressed")
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    assert figure.axes[0].get_xlabel() == ""
    assert (
        figurecomposer_rendering._layout_axes_from_figure(
            tool,
            Figure(figsize=(1.0, 1.0)),
        )
        is None
    )

    assert tool._figure_window is None
    with figurecomposer_rendering._rendered_output_figure(tool) as output_figure:
        assert tool._figure_window is None
        assert output_figure.axes
    assert output_figure.axes == []
    assert isinstance(figurecomposer_rendering._make_axes(tool), np.ndarray)
    tool.show_figure_window(activate=False)
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)
    figurecomposer_rendering._render_preview(tool, show_window=False)
    assert tool._figure_window is not None

    root_grid = FigureGridSpecGridState(
        grid_id="root",
        nrows=2,
        ncols=1,
        height_ratios=(1.0, 2.0),
        wspace=0.1,
        hspace=0.2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
        child_grids=(
            FigureGridSpecGridState(
                grid_id="child",
                nrows=1,
                ncols=1,
                span=FigureGridSpecSpanState(
                    row_start=3,
                    row_stop=4,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root_grid),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_axes = figurecomposer_rendering._make_axes(
        grid_tool,
        Figure(figsize=(1.0, 1.0)),
        sync_visible=False,
    )
    assert isinstance(grid_axes, dict)
    assert set(grid_axes) == {"axis-a"}


def test_figure_composer_preview_update_is_cancelled_on_close(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    tool.request_preview_pixmap_update(delay_ms=0)
    assert tool._preview_pixmap_update_pending

    tool.close()
    QtWidgets.QApplication.processEvents()

    assert not tool._preview_pixmap_update_pending
    assert tool.preview_pixmap is None


def test_figure_composer_rechecks_configured_stylesheets_after_erlab_import(
    qtbot,
    monkeypatch,
    restore_interactive_options,
) -> None:
    available: list[str] = []
    monkeypatch.setattr("erlab.interactive._stylesheets.mpl_style.available", available)
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: available.append("classic"),
    )
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("classic",),
    )
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    code = tool.generated_code()

    assert "plt.style.use(['classic'])" in code
    assert "Skipped unavailable stylesheets" not in code


def test_figure_composer_generated_code_imports_erlab_for_erlab_stylesheet(
    qtbot,
    monkeypatch,
    restore_interactive_options,
) -> None:
    loaded = False

    def stylesheet_names() -> frozenset[str]:
        names = {"classic"}
        if loaded:
            names.add("erlab-test-style")
        return frozenset(names)

    def load_stylesheets() -> None:
        nonlocal loaded
        loaded = True

    @contextlib.contextmanager
    def style_context(_stylesheets):
        yield

    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_stylesheet_name_set",
        stylesheet_names,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        load_stylesheets,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "stylesheets_require_user_stylesheets",
        lambda names: False,
    )
    erlab.interactive._stylesheets._ERLAB_REGISTERED_STYLESHEETS.clear()
    monkeypatch.setattr(figurecomposer_defaults.mpl_style, "context", style_context)
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("erlab-test-style",),
    )
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    code = tool.generated_code()

    style_import = "import erlab.plotting  # registers ERLab matplotlib stylesheets"
    assert style_import in code
    assert code.index(style_import) < code.index("plt.style.use")
    assert "plt.style.use(['erlab-test-style'])" in code

    image_tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="image")
    )
    qtbot.addWidget(image_tool)
    image_code = image_tool.generated_code()

    assert "import erlab.plotting as eplt" in image_code
    assert style_import not in image_code


def test_figure_composer_generated_code_imports_erlab_for_preloaded_erlab_style(
    qtbot,
    monkeypatch,
    restore_interactive_options,
) -> None:
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "stylesheets_require_user_stylesheets",
        lambda names: False,
    )
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("nature",),
    )
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    code = tool.generated_code()

    style_import = "import erlab.plotting  # registers ERLab matplotlib stylesheets"
    assert style_import in code
    assert code.index(style_import) < code.index("plt.style.use")
    assert "plt.style.use(['nature'])" in code


def test_figure_composer_canvas_draw_and_print_use_style_context(
    monkeypatch, recwarn
) -> None:
    @contextlib.contextmanager
    def style_context():
        with mpl.rc_context({"font.family": ["serif"], "font.serif": ["DejaVu Serif"]}):
            yield

    draw_fonts: list[tuple[list[str], str]] = []
    print_fonts: list[tuple[list[str], str]] = []

    def draw(_self, *_args, **_kwargs) -> None:
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        draw_fonts.append(
            (list(mpl.rcParams["font.family"]), mpl.rcParams["font.serif"][0])
        )

    def print_figure(_self, *_args, **_kwargs) -> None:
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        print_fonts.append(
            (list(mpl.rcParams["font.family"]), mpl.rcParams["font.serif"][0])
        )

    monkeypatch.setattr(figurecomposer_defaults, "_figure_style_context", style_context)
    monkeypatch.setattr(figurecomposer_widgets.FigureCanvas, "draw", draw)
    monkeypatch.setattr(
        figurecomposer_widgets.FigureCanvas, "print_figure", print_figure
    )

    canvas = figurecomposer_widgets._StyledFigureCanvas(figurecomposer_widgets.Figure())
    canvas.draw()
    canvas.print_figure("unused.png")

    assert draw_fonts == [(["serif"], "DejaVu Serif")]
    assert print_fonts == [(["serif"], "DejaVu Serif")]
    assert not any(
        "constrained_layout not applied" in str(warning.message) for warning in recwarn
    )


def test_figure_composer_export_uses_style_context(qtbot, monkeypatch, recwarn) -> None:
    @contextlib.contextmanager
    def style_context():
        with mpl.rc_context({"font.family": ["serif"], "font.serif": ["DejaVu Serif"]}):
            yield

    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    savefig_fonts: list[tuple[list[str], str]] = []
    monkeypatch.setattr(figurecomposer_defaults, "_figure_style_context", style_context)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("figure.png", ""),
    )

    def savefig(*_args, **_kwargs) -> None:
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        savefig_fonts.append(
            (list(mpl.rcParams["font.family"]), mpl.rcParams["font.serif"][0])
        )

    monkeypatch.setattr(tool.figure, "savefig", savefig)

    tool.export_figure()

    assert savefig_fonts == [(["serif"], "DejaVu Serif")]
    assert not any(
        "constrained_layout not applied" in str(warning.message) for warning in recwarn
    )


def test_figure_composer_preview_suppresses_collapsed_layout_warning(
    qtbot, monkeypatch, recwarn
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    figurecomposer_rendering._render_preview(tool, show_window=False)
    original_draw = figurecomposer_widgets.FigureCanvas.draw

    def draw_with_warning(self, *args, **kwargs):
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        return original_draw(self, *args, **kwargs)

    monkeypatch.setattr(figurecomposer_widgets.FigureCanvas, "draw", draw_with_warning)

    assert tool.preview_pixmap is None
    assert tool.refresh_preview_pixmap() is None
    assert not any(
        "constrained_layout not applied" in str(warning.message) for warning in recwarn
    )


def test_figure_composer_duplicates_and_reorders_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.custom(
                    label="first",
                    code=(
                        "fig.__dict__['_order'] = "
                        "fig.__dict__.get('_order', []) + ['first']"
                    ),
                    trusted=True,
                ),
                FigureOperationState.custom(
                    label="second",
                    code=(
                        "fig.__dict__['_order'] = "
                        "fig.__dict__.get('_order', []) + ['second']"
                    ),
                    trusted=True,
                ),
                FigureOperationState.custom(
                    label="third",
                    code=(
                        "fig.__dict__['_order'] = "
                        "fig.__dict__.get('_order', []) + ['third']"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    duplicate_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerDuplicateStepButton"
    )
    move_up_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerMoveStepUpButton"
    )
    move_down_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerMoveStepDownButton"
    )
    delete_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerDeleteStepButton"
    )
    assert duplicate_button is tool.duplicate_operation_button
    assert move_up_button is tool.move_operation_up_button
    assert move_down_button is tool.move_operation_down_button
    assert delete_button is tool.remove_operation_button

    tool.operation_list.setCurrentRow(0)
    assert move_up_button.isEnabled() is False
    assert move_down_button.isEnabled() is True

    _select_operation_rows(tool, (1,))
    second = tool.tool_status.operations[1]
    duplicate_button.click()
    duplicate = tool.tool_status.operations[2]
    assert tool.operation_list.currentRow() == 2
    assert len(tool.tool_status.operations) == 4
    assert duplicate.operation_id != second.operation_id
    assert duplicate.model_dump(exclude={"operation_id"}) == second.model_dump(
        exclude={"operation_id"}
    )

    duplicate_id = duplicate.operation_id
    move_up_button.click()
    assert tool.operation_list.currentRow() == 1
    assert tool.tool_status.operations[1].operation_id == duplicate_id
    move_down_button.click()
    assert tool.operation_list.currentRow() == 2
    assert tool.tool_status.operations[2].operation_id == duplicate_id

    tool.operation_list.setCurrentRow(3)
    assert move_up_button.isEnabled() is True
    assert move_down_button.isEnabled() is False

    namespace: dict[str, typing.Any] = {}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["fig"].__dict__["_order"] == [
        "first",
        "second",
        "second",
        "third",
    ]


def test_figure_composer_batch_duplicates_reorders_and_removes_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )

    def custom_step(label: str) -> FigureOperationState:
        return FigureOperationState.custom(
            label=label,
            code=(
                f"fig.__dict__['_order'] = fig.__dict__.get('_order', []) + [{label!r}]"
            ),
            trusted=True,
        )

    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(custom_step(label) for label in "abcde"),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (1, 3))
    selected_originals = [
        tool.tool_status.operations[index].operation_id for index in (1, 3)
    ]
    tool.duplicate_operation_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "c",
        "d",
        "b",
        "d",
        "e",
    ]
    assert _selected_operation_rows(tool) == (4, 5)
    assert tool.operation_list.currentRow() == 4
    assert [
        tool.tool_status.operations[index].operation_id for index in (4, 5)
    ] != selected_originals

    duplicate_ids = {
        tool.tool_status.operations[index].operation_id for index in (4, 5)
    }
    tool.move_operation_up_button.click()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "c",
        "b",
        "d",
        "d",
        "e",
    ]
    assert _selected_operation_rows(tool) == (3, 4)
    assert {
        tool.tool_status.operations[index].operation_id for index in (3, 4)
    } == duplicate_ids

    _select_operation_rows(tool, (0, 2, 5))
    selected_ids = {
        tool.tool_status.operations[index].operation_id for index in (0, 2, 5)
    }
    tool.move_operation_down_button.click()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "b",
        "a",
        "b",
        "c",
        "d",
        "e",
        "d",
    ]
    assert _selected_operation_rows(tool) == (1, 3, 6)
    assert {
        tool.tool_status.operations[index].operation_id for index in (1, 3, 6)
    } == selected_ids

    tool.remove_operation_button.click()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "b",
        "b",
        "d",
        "e",
    ]
    assert _selected_operation_rows(tool) == (1,)
    assert tool.operation_list.currentRow() == 1


def test_figure_composer_copy_paste_steps_preserves_order_and_history(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(_custom_order_step(label) for label in "abcd"),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    _clear_clipboard()

    _select_operation_rows(tool, (1, 3))
    copied_ids = {tool.tool_status.operations[index].operation_id for index in (1, 3)}
    tool.copy_operation_button.click()
    _select_operation_rows(tool, (0,))
    tool.paste_operation_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "d",
        "b",
        "c",
        "d",
    ]
    assert _selected_operation_rows(tool) == (1, 2)
    assert tool.operation_list.currentRow() == 1
    pasted_ids = {tool.tool_status.operations[index].operation_id for index in (1, 2)}
    assert pasted_ids.isdisjoint(copied_ids)

    tool.undo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "c",
        "d",
    ]
    tool.redo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "d",
        "b",
        "c",
        "d",
    ]


def test_figure_composer_cut_paste_steps_preserves_order_and_history(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(_custom_order_step(label) for label in "abcd"),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    clipboard = _clear_clipboard()

    _select_operation_rows(tool, (1, 3))
    cut_ids = {tool.tool_status.operations[index].operation_id for index in (1, 3)}
    tool.cut_operation_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]
    assert _selected_operation_rows(tool) == (1,)
    assert tool.operation_list.currentRow() == 1
    mime = clipboard.mimeData()
    payload_text = bytes(
        mime.data(figurecomposer_tool_module._STEPS_CLIPBOARD_MIME).data()
    ).decode("utf-8")
    assert [
        operation["label"] for operation in json.loads(payload_text)["operations"]
    ] == ["b", "d"]
    assert clipboard.text().splitlines() == [
        "fig.__dict__['_order'] = fig.__dict__.get('_order', []) + ['b']",
        "fig.__dict__['_order'] = fig.__dict__.get('_order', []) + ['d']",
    ]
    with pytest.raises(json.JSONDecodeError):
        json.loads(clipboard.text())

    tool.paste_operation_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
        "b",
        "d",
    ]
    assert _selected_operation_rows(tool) == (2, 3)
    assert tool.operation_list.currentRow() == 2
    pasted_ids = {tool.tool_status.operations[index].operation_id for index in (2, 3)}
    assert pasted_ids.isdisjoint(cut_ids)

    tool.undo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]
    tool.undo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "c",
        "d",
    ]
    tool.redo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]
    tool.redo()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
        "b",
        "d",
    ]


def test_figure_composer_copy_paste_steps_carries_same_process_source_data(
    qtbot,
) -> None:
    source_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="map",
    )
    source_tool = FigureComposerTool(
        source_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="map", label="map"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("map",),
                ),
            ),
            primary_source="map",
        ),
        source_data={"map": source_data},
    )
    destination_data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="existing",
    )
    destination = FigureComposerTool(
        destination_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="existing", label="existing"),),
            operations=(),
            primary_source="existing",
        ),
        source_data={"existing": destination_data},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()
    data_changed: list[None] = []
    destination.sigDataChanged.connect(lambda: data_changed.append(None))

    _select_operation_rows(source_tool, (0,))
    source_tool.copy_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert data_changed == [None]
    assert [source.name for source in destination.tool_status.sources] == [
        "existing",
        "map",
    ]
    assert destination.tool_status.operations[-1].sources == ("map",)
    xr.testing.assert_identical(destination.source_data()["map"], source_data)

    destination.undo()
    assert [source.name for source in destination.tool_status.sources] == ["existing"]
    assert set(destination.source_data()) == {"existing"}

    destination.redo()
    assert [source.name for source in destination.tool_status.sources] == [
        "existing",
        "map",
    ]
    assert destination.tool_status.operations[-1].sources == ("map",)
    xr.testing.assert_identical(destination.source_data()["map"], source_data)


def test_figure_composer_cut_paste_steps_preserves_same_composer_sources(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)
    _clear_clipboard()

    _select_operation_rows(tool, (0,))
    original_id = tool.tool_status.operations[0].operation_id
    tool.cut_operation_button.click()
    tool.paste_operation_button.click()

    assert [source.name for source in tool.tool_status.sources] == ["data"]
    assert set(tool.source_data()) == {"data"}
    pasted_operation = tool.tool_status.operations[0]
    assert pasted_operation.sources == ("data",)
    assert pasted_operation.map_selections[0].source == "data"
    assert pasted_operation.operation_id != original_id


def test_figure_composer_cut_paste_steps_renames_cross_composer_sources(
    qtbot,
) -> None:
    image = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="incoming_image",
    )
    profile = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="incoming_profile",
    )
    source_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="image"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
                FigureOperationState.line(label="line", source="profile"),
            ),
            primary_source="data",
        ),
        source_data={"data": image, "profile": profile},
    )
    existing_image = image.copy(data=np.full((2, 2), -1.0))
    existing_profile = profile.copy(data=np.full(3, -1.0))
    destination = FigureComposerTool(
        existing_image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="existing image"),
                FigureSourceState(name="profile", label="existing profile"),
            ),
            operations=(),
            primary_source="data",
        ),
        source_data={"data": existing_image, "profile": existing_profile},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    _select_operation_rows(source_tool, (0, 1))
    source_tool.cut_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert [operation.label for operation in source_tool.tool_status.operations] == []
    assert [source.name for source in destination.tool_status.sources] == [
        "data",
        "profile",
        "data_copy",
        "profile_copy",
    ]
    pasted_plot, pasted_line = destination.tool_status.operations
    assert pasted_plot.sources == ("data_copy",)
    assert pasted_plot.map_selections[0].source == "data_copy"
    assert pasted_line.line_source == "profile_copy"
    xr.testing.assert_identical(destination.source_data()["data"], existing_image)
    xr.testing.assert_identical(destination.source_data()["profile"], existing_profile)
    xr.testing.assert_identical(destination.source_data()["data_copy"], image)
    xr.testing.assert_identical(destination.source_data()["profile_copy"], profile)


def test_figure_composer_copy_paste_steps_renames_source_conflicts(qtbot) -> None:
    image = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="incoming_image",
    )
    profile = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="incoming_profile",
    )
    source_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="image"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
                FigureOperationState.line(label="line", source="profile"),
            ),
            primary_source="data",
        ),
        source_data={"data": image, "profile": profile},
    )
    existing_image = image.copy(data=np.full((2, 2), -1.0))
    existing_profile = profile.copy(data=np.full(3, -1.0))
    destination = FigureComposerTool(
        existing_image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="existing image"),
                FigureSourceState(name="profile", label="existing profile"),
            ),
            operations=(),
            primary_source="data",
        ),
        source_data={"data": existing_image, "profile": existing_profile},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    _select_operation_rows(source_tool, (0, 1))
    source_tool.copy_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert [source.name for source in destination.tool_status.sources] == [
        "data",
        "profile",
        "data_copy",
        "profile_copy",
    ]
    pasted_plot, pasted_line = destination.tool_status.operations
    assert pasted_plot.sources == ("data_copy",)
    assert pasted_plot.map_selections[0].source == "data_copy"
    assert pasted_line.line_source == "profile_copy"
    xr.testing.assert_identical(destination.source_data()["data"], existing_image)
    xr.testing.assert_identical(destination.source_data()["profile"], existing_profile)
    xr.testing.assert_identical(destination.source_data()["data_copy"], image)
    xr.testing.assert_identical(destination.source_data()["profile_copy"], profile)


def test_figure_composer_copy_paste_steps_typed_payload_has_missing_source(
    qtbot,
) -> None:
    remote_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="remote",
    )
    source_tool = FigureComposerTool(
        remote_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="remote", label="remote"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("remote",),
                ),
            ),
            primary_source="remote",
        ),
        source_data={"remote": remote_data},
    )
    destination_data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="local",
    )
    destination = FigureComposerTool(
        destination_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="local", label="local"),),
            operations=(),
            primary_source="local",
        ),
        source_data={"local": destination_data},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    clipboard = _clear_clipboard()

    _select_operation_rows(source_tool, (0,))
    source_tool.copy_operation_button.click()
    copied_mime = clipboard.mimeData()
    payload_text = bytes(
        copied_mime.data(figurecomposer_tool_module._STEPS_CLIPBOARD_MIME).data()
    ).decode("utf-8")
    assert payload_text.startswith(
        '{\n  "type": "erlab.figure_composer.steps",\n  "version": 1,'
    )
    assert json.loads(payload_text)["type"] == "erlab.figure_composer.steps"
    assert copied_mime.text()
    assert not copied_mime.text().lstrip().startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(copied_mime.text())
    mime = QtCore.QMimeData()
    mime.setData(
        figurecomposer_tool_module._STEPS_CLIPBOARD_MIME,
        payload_text.encode("utf-8"),
    )
    mime.setText(copied_mime.text())
    clipboard.setMimeData(mime)
    destination._paste_operations_from_clipboard()

    assert [source.name for source in destination.tool_status.sources] == [
        "local",
        "remote",
    ]
    assert destination.tool_status.operations[-1].sources == ("remote",)
    assert "remote" not in destination.source_data()


def test_figure_composer_copy_steps_keeps_payload_when_code_text_fails(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    clipboard = _clear_clipboard()
    disabled = tool.tool_status.operations[0].model_copy(update={"enabled": False})
    assert figurecomposer_tool_module._step_clipboard_code_text(tool, (disabled,)) == ""

    class BrokenSpec:
        @staticmethod
        def code_lines(
            _tool: FigureComposerTool,
            _operation: FigureOperationState,
        ) -> list[str]:
            raise ValueError("bad axes")

    _select_operation_rows(tool, (0,))
    with monkeypatch.context() as patch:
        patch.setattr(
            figurecomposer_tool_module._registry,
            "spec_for",
            lambda _kind: BrokenSpec(),
        )
        failure_text = figurecomposer_tool_module._step_clipboard_code_text(
            tool,
            tool.tool_status.operations,
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            figurecomposer_tool_module,
            "_step_clipboard_code_text",
            lambda _tool, _operations: failure_text,
        )
        tool._write_selected_operations_to_clipboard()

    copied_mime = clipboard.mimeData()
    payload_text = bytes(
        copied_mime.data(figurecomposer_tool_module._STEPS_CLIPBOARD_MIME).data()
    ).decode("utf-8")
    assert [
        operation["label"] for operation in json.loads(payload_text)["operations"]
    ] == ["a"]
    assert failure_text.startswith(
        "# Could not generate Python code for the copied Figure Composer steps:"
    )
    assert copied_mime.text() == failure_text
    assert not copied_mime.text().lstrip().startswith("{")


def test_figure_composer_copy_paste_steps_ignores_invalid_payload(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    clipboard = _clear_clipboard()
    clipboard.setText("{not valid json")
    before = tool.tool_status

    tool._update_step_action_buttons()
    assert not tool.paste_operation_button.isEnabled()
    tool._paste_operations_from_clipboard()
    assert tool.tool_status == before

    clipboard.setText("fig.suptitle('Copied step code')")
    tool._update_step_action_buttons()
    assert not tool.paste_operation_button.isEnabled()
    tool._paste_operations_from_clipboard()
    assert tool.tool_status == before

    clipboard.setText(
        json.dumps(
            {
                "version": figurecomposer_tool_module._STEPS_CLIPBOARD_PAYLOAD_VERSION,
                "operations": [_custom_order_step("b").model_dump(mode="json")],
                "sources": [],
            }
        )
    )
    tool._update_step_action_buttons()
    assert not tool.paste_operation_button.isEnabled()
    tool._paste_operations_from_clipboard()
    assert tool.tool_status == before


def test_figure_composer_copy_paste_steps_shortcuts_and_context_menu(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(_custom_order_step(label) for label in ("a", "b")),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    _clear_clipboard()

    _select_operation_rows(tool, (0,))
    copy_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_C,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    tool.operation_list.keyPressEvent(copy_event)
    if not copy_event.isAccepted():
        copy_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        )
        tool.operation_list.keyPressEvent(copy_event)
    assert copy_event.isAccepted()

    tool._show_operation_context_menu(QtCore.QPoint(0, 0))
    assert tool._operation_context_menu is not None
    paste_action = next(
        action
        for action in tool._operation_context_menu.actions()
        if action.objectName() == "figureComposerContextPasteStepsAction"
    )
    _select_operation_rows(tool, (1,))
    paste_action.trigger()
    tool._operation_context_menu.close()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "a",
    ]


def test_figure_composer_cut_paste_steps_shortcuts_and_context_menu(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(_custom_order_step(label) for label in ("a", "b", "c")),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    _clear_clipboard()

    _select_operation_rows(tool, (1,))
    cut_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_X,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    tool.operation_list.keyPressEvent(cut_event)
    if not cut_event.isAccepted():
        cut_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_X,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        )
        tool.operation_list.keyPressEvent(cut_event)
    assert cut_event.isAccepted()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]

    _select_operation_rows(tool, (0,))
    tool._show_operation_context_menu(QtCore.QPoint(0, 0))
    assert tool._operation_context_menu is not None
    cut_action = next(
        action
        for action in tool._operation_context_menu.actions()
        if action.objectName() == "figureComposerContextCutStepsAction"
    )
    assert cut_action.isEnabled()
    cut_action.trigger()
    tool._operation_context_menu.close()

    assert [operation.label for operation in tool.tool_status.operations] == ["c"]


def test_figure_composer_step_payload_rejects_malformed_clipboard_data() -> None:
    mime = QtCore.QMimeData()
    mime.setData(figurecomposer_tool_module._STEPS_CLIPBOARD_MIME, b"\xff")
    assert figurecomposer_tool_module._step_clipboard_payload(mime) is None

    assert (
        figurecomposer_tool_module._step_clipboard_payload(QtCore.QMimeData()) is None
    )

    def payload_with(**updates: typing.Any) -> str:
        payload = {
            "type": figurecomposer_tool_module._STEPS_CLIPBOARD_PAYLOAD_TYPE,
            "version": figurecomposer_tool_module._STEPS_CLIPBOARD_PAYLOAD_VERSION,
            "operations": [_custom_order_step("a").model_dump(mode="json")],
            "sources": [],
        }
        payload.update(updates)
        return json.dumps(payload)

    invalid_payloads = (
        "[]",
        payload_with(version=2),
        payload_with(operations=[]),
        payload_with(operations={}),
        payload_with(sources={}),
    )
    for payload_text in invalid_payloads:
        mime = QtCore.QMimeData()
        mime.setText(payload_text)
        assert figurecomposer_tool_module._step_clipboard_payload(mime) is None

    mime = QtCore.QMimeData()
    mime.setText(payload_with())
    mime.figure_composer_source_data = object()
    operations, sources, source_data = (
        figurecomposer_tool_module._step_clipboard_payload(mime)
    )
    assert [operation.label for operation in operations] == ["a"]
    assert sources == ()
    assert source_data == {}


def test_figure_composer_operation_list_keypress_defensive_paths(qtbot) -> None:
    operation_list = figurecomposer_tool_module._FigureComposerOperationList()
    qtbot.addWidget(operation_list)
    pasted: list[bool] = []
    operation_list.paste_requested.connect(lambda: pasted.append(True))

    operation_list.keyPressEvent(None)

    paste_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_V,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    operation_list.keyPressEvent(paste_event)
    if not paste_event.isAccepted():
        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        )
        operation_list.keyPressEvent(paste_event)
    assert paste_event.isAccepted()
    assert pasted == [True]

    fallback_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    operation_list.keyPressEvent(fallback_event)


def test_figure_composer_copy_paste_defensive_paths(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    with monkeypatch.context() as patch:
        patch.setattr(
            figurecomposer_tool_module.QtWidgets.QApplication,
            "instance",
            staticmethod(lambda: None),
        )
        disconnected_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                sources=(FigureSourceState(name="data", label="data"),),
                operations=(_custom_order_step("a"),),
                primary_source="data",
            ),
        )
    qtbot.addWidget(disconnected_tool)
    assert disconnected_tool._connected_step_clipboard is None

    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert tool._clipboard() is not None
    tool.operation_list.setCurrentRow(-1)
    tool.operation_list.clearSelection()
    tool._copy_selected_operations()

    _select_operation_rows(tool, (0,))
    before = tool.tool_status
    with monkeypatch.context() as patch:
        patch.setattr(
            figurecomposer_tool_module.QtWidgets.QApplication,
            "instance",
            staticmethod(lambda: None),
        )
        assert tool._clipboard() is None
        assert tool._clipboard_step_payload() is None
        tool._copy_selected_operations()
        tool._cut_selected_operations()
    assert tool.tool_status == before

    tool._connected_step_clipboard = None
    tool._disconnect_step_clipboard()


def test_figure_composer_source_data_history_helpers(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    tool._reset_history_stack()
    assert list(tool._prev_source_data_states[-1]) == ["data"]
    assert not tool.undoable
    assert not tool.redoable
    tool.undo()
    tool.redo()
    tool._write_state()
    assert len(tool._prev_source_data_states) == 1

    with tool._history_suppressed():
        tool._write_state()
        tool._replace_last_state()
    assert len(tool._prev_source_data_states) == 1

    replacement = data.copy(data=np.full(3, 2.0))
    tool.set_source_data({"data": replacement})
    tool._replace_last_state()
    xr.testing.assert_identical(tool._prev_source_data_states[-1]["data"], replacement)

    tool._prev_states.clear()
    tool._prev_source_data_states.clear()
    tool._replace_last_state()
    assert len(tool._prev_states) == 1
    xr.testing.assert_identical(tool._prev_source_data_states[-1]["data"], replacement)


def test_figure_composer_copy_paste_source_and_insert_fallbacks(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    operation = FigureOperationState.plot_slices(
        label="plot",
        sources=("data",),
        map_selections=(FigureDataSelectionState(source="extra"),),
    )
    assert tool._operation_source_names(operation) == ("data", "extra")

    renamed_sources, _, _ = tool._renamed_pasted_sources(
        (
            FigureSourceState(name="extra", label="extra"),
            FigureSourceState(name="extra", label="extra"),
        ),
        {},
    )
    assert [source.name for source in renamed_sources] == ["extra"]

    clipboard = _clear_clipboard()
    clipboard.setText(
        figurecomposer_tool_module._step_clipboard_payload_text(
            (_custom_order_step("b"),), ()
        )
    )
    tool.operation_list.setCurrentRow(0)
    with monkeypatch.context() as patch:
        patch.setattr(tool, "_operation_id_for_item", lambda item: "missing")
        tool._paste_operations_from_clipboard()
    assert [operation.label for operation in tool.tool_status.operations] == ["a", "b"]

    def existing_source_payload(sources, source_data, *, preserve_existing=False):
        return (FigureSourceState(name="data", label="data"),), {}, {}

    with monkeypatch.context() as patch:
        patch.setattr(tool, "_renamed_pasted_sources", existing_source_payload)
        clipboard.setText(
            figurecomposer_tool_module._step_clipboard_payload_text(
                (_custom_order_step("c"),), ()
            )
        )
        tool._paste_operations_from_clipboard()
    assert [source.name for source in tool.tool_status.sources] == ["data"]

    tool._source_data["data_copy"] = data
    renamed_sources, rename_map, _ = tool._renamed_pasted_sources(
        (FigureSourceState(name="data", label="data"),),
        {},
    )
    assert [source.name for source in renamed_sources] == ["data_copy_2"]
    assert rename_map == {"data": "data_copy_2"}

    tool._source_data["extra"] = data
    renamed_sources, rename_map, renamed_source_data = tool._renamed_pasted_sources(
        (FigureSourceState(name="extra", label="extra"),),
        {"extra": data},
        preserve_existing=True,
    )
    assert [source.name for source in renamed_sources] == ["extra"]
    assert rename_map == {"extra": "extra"}
    assert renamed_source_data == {}

    metadata_source = FigureSourceState(name="metadata_only", label="metadata")
    tool._recipe = tool._recipe.model_copy(
        update={"sources": (*tool.tool_status.sources, metadata_source)}
    )
    renamed_sources, rename_map, renamed_source_data = tool._renamed_pasted_sources(
        (metadata_source,),
        {"metadata_only": data},
        preserve_existing=True,
    )
    assert renamed_sources == ()
    assert rename_map == {"metadata_only": "metadata_only"}
    xr.testing.assert_identical(renamed_source_data["metadata_only"], data)


def test_figure_composer_axes_code_compacts_contiguous_selections(qtbot) -> None:
    data = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=3, ncols=4),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    def axes_code(axes: tuple[tuple[int, int], ...]) -> str:
        return figurecomposer_code._axes_code(
            tool, FigureAxesSelectionState(axes=axes), for_plot_slices=False
        )

    def axes_sequence_code(axes: tuple[tuple[int, int], ...]) -> str:
        return figurecomposer_code._axes_sequence_code(
            tool, FigureAxesSelectionState(axes=axes)
        )

    all_axes = tuple((row, col) for row in range(3) for col in range(4))
    assert axes_code(((0, 1),)) == "axs[0, 1]"
    assert axes_sequence_code(((0, 1),)) == "(axs[0, 1],)"
    assert axes_code(all_axes) == "axs"
    assert axes_sequence_code(all_axes) == "axs.flat"
    assert axes_code(((0, 0), (0, 1), (0, 2), (0, 3))) == "axs[0, :]"
    assert axes_code(((0, 3), (0, 1), (0, 2), (0, 0))) == "axs[0, :]"
    assert axes_code(((0, 0), (0, 1), (0, 1), (0, 2))) == "axs[0, :3]"
    assert axes_sequence_code(((0, 0), (0, 1), (0, 2), (0, 3))) == ("axs[0, :].flat")
    assert axes_code(((0, 2), (1, 2), (2, 2))) == "axs[:, 2]"
    assert axes_code(((0, 0), (0, 1), (0, 2))) == "axs[0, :3]"
    assert axes_code(((0, 1), (0, 2), (0, 3))) == "axs[0, 1:4]"
    assert axes_code(((1, 0), (2, 0))) == "axs[1:3, 0]"
    assert axes_code(((1, 1), (1, 2), (2, 1), (2, 2))) == "axs[1:3, 1:3]"
    assert axes_code(((2, 2), (1, 1), (1, 2), (2, 1))) == "axs[1:3, 1:3]"
    assert axes_sequence_code(((1, 1), (1, 2), (2, 1), (2, 2))) == (
        "axs[1:3, 1:3].flat"
    )
    assert axes_code(((0, 0), (0, 2))) == "[axs[0, 0], axs[0, 2]]"
    assert axes_sequence_code(((0, 0), (0, 2))) == "(axs[0, 0], axs[0, 2])"
    assert (
        figurecomposer_code._axes_code(
            tool,
            FigureAxesSelectionState(axes=((0, 1),)),
            for_plot_slices=True,
        )
        == "[axs[0, 1]]"
    )
    assert (
        figurecomposer_code._axes_code(
            tool, FigureAxesSelectionState(axes=all_axes), for_plot_slices=True
        )
        == "axs"
    )


def test_figure_composer_code_helpers_cover_selection_and_layout_edges(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expression_selection = FigureAxesSelectionState(expression="custom_axes")
    assert (
        figurecomposer_code._axes_code(
            tool, expression_selection, for_plot_slices=False
        )
        == "custom_axes"
    )
    assert figurecomposer_code._axes_sequence_code(tool, expression_selection) == (
        "custom_axes"
    )
    with pytest.raises(ValueError, match="outside the current layout"):
        figurecomposer_code._axes_code(
            tool,
            FigureAxesSelectionState(axes=((2, 0),)),
            for_plot_slices=False,
        )

    selection_code = figurecomposer_code._selection_code(
        FigureDataSelectionState(
            source="data",
            isel={"x": {"kind": "slice", "start": 0, "stop": 1}},
            qsel={"y": 20.0},
            mean_dims=("x", "y"),
        )
    )
    assert selection_code == (
        'data.isel(x=slice(0, 1)).qsel(y=20.0).qsel.mean(("x", "y"))'
    )
    assert (
        figurecomposer_code._selection_code(
            FigureDataSelectionState(source="data", mean_dims=("x",))
        )
        == 'data.qsel.mean("x")'
    )

    invalid_grid = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=1,
        axes=(
            FigureGridSpecAxesState(
                axes_id="outside",
                span=FigureGridSpecSpanState(
                    row_start=1,
                    row_stop=2,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    invalid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=invalid_grid),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(invalid_tool)
    with pytest.raises(ValueError, match="outside their grids"):
        figurecomposer_code._setup_code(invalid_tool)

    child_without_span = FigureGridSpecGridState(
        grid_id="child",
        nrows=1,
        ncols=1,
        span=None,
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=2,
        ncols=2,
        width_ratios=(2.0, 1.0),
        height_ratios=(1.0, 3.0),
        wspace=0.25,
        hspace=0.5,
        axes=(
            FigureGridSpecAxesState(
                axes_id="main",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=2,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
        child_grids=(child_without_span,),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                layout="compressed",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    setup_code = "\n".join(figurecomposer_code._gridspec_setup_code_lines(grid_tool))
    assert 'layout="compressed"' in setup_code
    assert "width_ratios=(2.0, 1.0)" in setup_code
    assert "height_ratios=(1.0, 3.0)" in setup_code
    assert "wspace=0.25" in setup_code
    assert "hspace=0.5" in setup_code
    assert "subgridspec" not in setup_code


def test_figure_composer_gridspec_helpers_cover_names_and_invalid_regions() -> None:
    span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    duplicate_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )
    child_span = FigureGridSpecSpanState(
        row_start=1,
        row_stop=2,
        col_start=0,
        col_stop=2,
    )
    child_grid = FigureGridSpecGridState(
        grid_id="child-grid",
        nrows=1,
        ncols=1,
        span=child_span,
        axes=(
            FigureGridSpecAxesState(
                axes_id="child-axis",
                label="1 child",
                span=span,
            ),
        ),
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=2,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(axes_id="first", label="peak", span=span),
            FigureGridSpecAxesState(
                axes_id="second",
                label="peak",
                span=duplicate_span,
            ),
            FigureGridSpecAxesState(
                axes_id="keyword",
                label="class",
                span=FigureGridSpecSpanState(
                    row_start=1,
                    row_stop=2,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
        child_grids=(child_grid,),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(root=root),
    )

    assert figurecomposer_gridspec._gridspec_axis_code_names(setup) == {
        "first": "peak",
        "second": "ax1",
        "keyword": "ax2",
        "child-axis": "ax3",
    }
    assert figurecomposer_gridspec._gridspec_axis_code_names(
        setup, reserved_names=("peak", "data")
    ) == {
        "first": "ax0",
        "second": "ax1",
        "keyword": "ax2",
        "child-axis": "ax3",
    }
    underscore_setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                nrows=1,
                ncols=1,
                axes=(
                    FigureGridSpecAxesState(
                        axes_id="underscore",
                        label="_line",
                        span=span,
                    ),
                ),
            )
        ),
    )
    assert figurecomposer_gridspec._gridspec_axis_code_names(underscore_setup) == {
        "underscore": "ax0",
    }
    assert (
        figurecomposer_gridspec._gridspec_axis_variable_name_error(
            setup, "first", "valid_name"
        )
        == ""
    )
    assert "identifier" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "1 child"
    )
    assert "keyword" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "class"
    )
    assert figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "_line"
    )
    assert "unique" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "peak"
    )
    assert "reserved" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "fig"
    )
    assert "reserved" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "plt"
    )
    assert "reserved" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "gs0"
    )
    assert "reserved" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "gs0_0"
    )
    assert "reserved" in figurecomposer_gridspec._gridspec_axis_variable_name_error(
        setup, "first", "data", reserved_names=("data",)
    )
    assert figurecomposer_gridspec._gridspec_grid_path(setup, "missing") == (root,)
    assert (
        figurecomposer_gridspec._gridspec_region_label(setup, root, "child-grid")
        == "Grid 1"
    )
    assert (
        figurecomposer_gridspec._gridspec_region_label(setup, root, "missing")
        == "removed region"
    )
    assert (
        figurecomposer_gridspec._gridspec_axis_display_name(setup, "missing")
        == "removed axis"
    )
    assert (
        figurecomposer_gridspec._gridspec_grid_display_name(setup, "missing")
        == "removed grid"
    )
    assert (
        figurecomposer_gridspec._gridspec_region_overlaps(
            root, span, ignore_axes_id="first"
        )
        is False
    )
    assert figurecomposer_gridspec._gridspec_region_overlaps(root, span) is True
    assert (
        figurecomposer_gridspec._gridspec_region_overlaps(
            root,
            child_span,
            ignore_grid_id="child-grid",
        )
        is True
    )
    assert (
        figurecomposer_gridspec._gridspec_region_overlaps(
            root.model_copy(update={"axes": ()}),
            child_span,
            ignore_grid_id="child-grid",
        )
        is False
    )

    invalid_child = child_grid.model_copy(update={"span": None})
    invalid_setup = setup.model_copy(
        update={
            "gridspec": setup.gridspec.model_copy(
                update={
                    "root": root.model_copy(update={"child_grids": (invalid_child,)})
                }
            )
        }
    )
    assert figurecomposer_gridspec._gridspec_has_invalid_regions(
        invalid_setup.gridspec.root
    )
    assert figurecomposer_gridspec._gridspec_region_valid(root, child_span)
    assert not figurecomposer_gridspec._gridspec_region_valid(
        root,
        FigureGridSpecSpanState(
            row_start=0,
            row_stop=3,
            col_start=0,
            col_stop=1,
        ),
    )
    assert figurecomposer_gridspec._slice_code(0, 1, 3) == "0"
    assert figurecomposer_gridspec._slice_code(0, 3, 3) == ":"
    assert figurecomposer_gridspec._slice_code(0, 2, 3) == ":2"
    assert figurecomposer_gridspec._slice_code(1, 3, 3) == "1:3"


def test_figure_composer_subplots_codegen_regression(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                nrows=2,
                ncols=2,
                layout="compressed",
                width_ratios=(2.0, 1.0),
                height_ratios=(1.0, 3.0),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(
                        axes=((0, 0), (0, 1), (1, 0), (1, 1))
                    ),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._select_step_section("axes")
    assert not tool.axes_selector.isHidden()
    assert tool.gridspec_axes_selector.isHidden()
    assert not tool.axes_expression_edit.isHidden()

    code = tool.generated_code()
    assert "fig, axs = plt.subplots(" in code
    assert "squeeze=False" in code
    assert 'layout="compressed"' in code
    assert "width_ratios=(2.0, 1.0)" in code
    assert "height_ratios=(1.0, 3.0)" in code
    assert "fig.add_gridspec" not in code
    assert "subgridspec" not in code
    assert "layout_mode" not in code
    assert "gridspec=" not in code
    assert "for ax in axs.flat:" in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["axs"].shape == (2, 2)
    empty_selection = FigureAxesSelectionState(axes=())
    assert tool._axes_selection_has_invalid_target(empty_selection)
    with pytest.raises(ValueError, match="No axes are selected"):
        figurecomposer_code._axes_code(tool, empty_selection, for_plot_slices=False)


def test_figure_composer_axes_status_uses_compact_labels(qtbot) -> None:
    data = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=4),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(
                        axes=((0, 0), (0, 1), (0, 2), (0, 3))
                    ),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("axes")
    assert tool.target_axes_status_label.text() == "Targets: axs[0, :]"
    assert (
        tool._axes_target_text(FigureAxesSelectionState(axes=((0, 1), (0, 2), (0, 3))))
        == "axs[0, 1:4]"
    )

    tool._target_current_operation_all_axes()
    assert tool.target_axes_status_label.text() == "Targets: axs"


def test_figure_composer_gridspec_codegen_executes(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    main_axis = FigureGridSpecAxesState(
        axes_id="main-axis",
        label="main",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=2,
            col_start=0,
            col_stop=1,
        ),
    )
    child_grid = FigureGridSpecGridState(
        grid_id="child-grid",
        label="Cuts",
        nrows=2,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
        axes=(
            FigureGridSpecAxesState(
                axes_id="cut-0",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="cut-1",
                span=FigureGridSpecSpanState(
                    row_start=1,
                    row_stop=2,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        layout=None,
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                label="Root",
                nrows=2,
                ncols=2,
                axes=(main_axis,),
                child_grids=(child_grid,),
            )
        ),
    )
    recipe = FigureRecipeState(
        setup=setup,
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="grid",
                axes=FigureAxesSelectionState(axes_ids=("main-axis", "cut-0")),
            ),
        ),
        primary_source="data",
    )
    tool = FigureComposerTool(data, recipe=recipe)
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "fig = plt.figure" in code
    assert "fig.add_gridspec" in code
    assert ".subgridspec" in code
    assert "main = fig.add_subplot" in code
    assert "for ax in (main, ax1):" in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes) == 3


def test_figure_composer_gridspec_codegen_reserves_live_names(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    source_axis = FigureGridSpecAxesState(
        axes_id="source-axis",
        label="data",
        span=span,
    )
    figure_axis = FigureGridSpecAxesState(
        axes_id="figure-axis",
        label="fig",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
    )
    gridspec_axis = FigureGridSpecAxesState(
        axes_id="gridspec-axis",
        label="gs0",
        span=FigureGridSpecSpanState(
            row_start=1,
            row_stop=2,
            col_start=0,
            col_stop=1,
        ),
    )
    child_grid = FigureGridSpecGridState(
        grid_id="child-grid",
        nrows=1,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=1,
            row_stop=2,
            col_start=1,
            col_stop=2,
        ),
        axes=(
            FigureGridSpecAxesState(
                axes_id="nested-helper-axis",
                label="gs0_0",
                span=span,
            ),
        ),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        layout=None,
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                nrows=2,
                ncols=2,
                axes=(source_axis, figure_axis, gridspec_axis),
                child_grids=(child_grid,),
            )
        ),
    )
    recipe = FigureRecipeState(
        setup=setup,
        sources=(FigureSourceState(name="data", label="data"),),
        primary_source="data",
    )
    tool = FigureComposerTool(data, recipe=recipe)
    qtbot.addWidget(tool)

    code = tool.generated_code()
    lines = code.splitlines()
    assert "data = fig.add_subplot(gs0[0, 0])" not in lines
    assert "fig = fig.add_subplot(gs0[0, 1])" not in lines
    assert "gs0 = fig.add_subplot(gs0[1, 0])" not in lines
    assert "gs0_0 = fig.add_subplot(gs0_0[0, 0])" not in lines
    assert "ax0 = fig.add_subplot(gs0[0, 0])" in lines
    assert "ax1 = fig.add_subplot(gs0[0, 1])" in lines
    assert "ax2 = fig.add_subplot(gs0[1, 0])" in lines
    assert "gs0_0 = gs0[1, 1].subgridspec(nrows=1, ncols=1)" in lines
    assert "ax3 = fig.add_subplot(gs0_0[0, 0])" in lines

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["data"] is data
    assert len(namespace["fig"].axes) == 4


def test_figure_composer_gridspec_axis_code_and_selector(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    left_axis = FigureGridSpecAxesState(
        axes_id="left-axis",
        label="left_panel",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    right_axis = FigureGridSpecAxesState(
        axes_id="right-axis",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        label="Root",
                        nrows=1,
                        ncols=2,
                        axes=(left_axis, right_axis),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(axes_ids=("left-axis",)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._select_step_section("axes")
    assert tool.axes_selector.isHidden()
    assert not tool.gridspec_axes_selector.isHidden()
    assert tool.axes_expression_edit.isHidden()
    assert tool.gridspec_axes_selector.axes_ids() == ("left-axis", "right-axis")
    assert not tool.gridspec_axes_selector.axis_rect("left-axis").isNull()
    assert not tool.gridspec_axes_selector.axis_rect("right-axis").isNull()
    assert (
        figurecomposer_code._axes_code(
            tool,
            FigureAxesSelectionState(axes_ids=("left-axis",)),
            for_plot_slices=False,
        )
        == "left_panel"
    )
    assert (
        figurecomposer_code._axes_code(
            tool,
            FigureAxesSelectionState(axes_ids=("left-axis",)),
            for_plot_slices=True,
        )
        == "[left_panel]"
    )
    assert (
        figurecomposer_code._axes_sequence_code(
            tool, FigureAxesSelectionState(axes_ids=("left-axis", "right-axis"))
        )
        == "(left_panel, ax1)"
    )

    qtbot.mouseClick(
        tool.gridspec_axes_selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        tool.gridspec_axes_selector.axis_rect("right-axis").center(),
    )
    assert tool.tool_status.operations[0].axes.axes_ids == (
        "left-axis",
        "right-axis",
    )
    tool.gridspec_axes_selector.set_selected_axes_ids((), emit=True)
    assert tool.tool_status.operations[0].axes.axes_ids == ()
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    with pytest.raises(ValueError, match="Cannot generate code"):
        tool.generated_code()

    tool._target_current_operation_all_axes()
    assert tool.tool_status.operations[0].axes.axes_ids == (
        "left-axis",
        "right-axis",
    )
    code = tool.generated_code()
    assert "fig = plt.figure" in code
    assert "fig, axs = plt.subplots" not in code
    assert "left_panel = fig.add_subplot(gs0[0, 0])" in code
    assert "ax1 = fig.add_subplot(gs0[0, 1])" in code
    assert "for ax in (left_panel, ax1):" in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes) == 2
    empty_selection = FigureAxesSelectionState(axes_ids=())
    assert tool._axes_selection_has_invalid_target(empty_selection)
    with pytest.raises(ValueError, match="No axes are selected"):
        figurecomposer_code._axes_code(tool, empty_selection, for_plot_slices=False)
    invalid_selection = FigureAxesSelectionState(
        axes_ids=("left-axis", "removed-internal-axis")
    )
    with pytest.raises(
        ValueError, match="1 selected GridSpec axis outside the current layout"
    ) as excinfo:
        figurecomposer_code._axes_code(tool, invalid_selection, for_plot_slices=False)
    assert "removed-internal-axis" not in str(excinfo.value)
    with pytest.raises(
        ValueError, match="1 selected GridSpec axis outside the current layout"
    ) as excinfo:
        figurecomposer_code._axes_sequence_code(tool, invalid_selection)
    assert "removed-internal-axis" not in str(excinfo.value)
    display_names = figurecomposer_gridspec._gridspec_axis_display_names(
        tool.tool_status.setup,
        invalid_selection.axes_ids,
    )
    assert "removed-internal-axis" not in display_names
    assert display_names[0] == "left_panel"


def test_figure_composer_gridspec_axes_selector_inlines_nested_grids(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    main_axis = FigureGridSpecAxesState(
        axes_id="main-axis",
        label="main",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=2,
            col_start=0,
            col_stop=1,
        ),
    )
    child_grid = FigureGridSpecGridState(
        grid_id="child-grid",
        nrows=2,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=2,
            col_start=1,
            col_stop=2,
        ),
        axes=(
            FigureGridSpecAxesState(
                axes_id="child-top",
                label="top",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="child-bottom",
                label="bottom",
                span=FigureGridSpecSpanState(
                    row_start=1,
                    row_stop=2,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=2,
                        ncols=2,
                        axes=(main_axis,),
                        child_grids=(child_grid,),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(axes_ids=("main-axis",)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool._select_step_section("axes")
    selector = tool.gridspec_axes_selector
    selector.resize(selector.sizeHint())

    root_min_width = 2 * selector._GRID_MARGIN + 2 * selector._CELL_WIDTH
    root_min_width += selector._CELL_GAP
    assert selector.sizeHint().width() > root_min_width
    assert selector.axes_ids() == ("main-axis", "child-top", "child-bottom")
    main_rect = selector.axis_rect("main-axis")
    top_rect = selector.axis_rect("child-top")
    bottom_rect = selector.axis_rect("child-bottom")
    assert not main_rect.isNull()
    assert top_rect.left() > main_rect.right()
    assert bottom_rect.top() > top_rect.bottom()

    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        top_rect.center(),
    )
    assert tool.tool_status.operations[0].axes.axes_ids == ("child-top",)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        bottom_rect.center(),
    )
    assert tool.tool_status.operations[0].axes.axes_ids == (
        "child-top",
        "child-bottom",
    )


def test_figure_composer_gridspec_to_subplots_preserves_targets(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    main_axis = FigureGridSpecAxesState(
        axes_id="main-axis",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=2,
            col_start=0,
            col_stop=1,
        ),
    )
    child_grid = FigureGridSpecGridState(
        grid_id="child-grid",
        nrows=2,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
        axes=(
            FigureGridSpecAxesState(
                axes_id="child-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=2,
                        ncols=2,
                        axes=(main_axis,),
                        child_grids=(child_grid,),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(axes_ids=("main-axis", "child-axis")),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.layout_mode_combo.setCurrentText("subplots")
    operation = tool.tool_status.operations[0]
    assert operation.axes.axes == ((0, 0), (1, 0), (0, 1))
    assert not tool._operation_has_invalid_axes(operation)
    namespace: dict[str, typing.Any] = {}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"].shape == (2, 2)


def test_figure_composer_gridspec_widget_creates_nested_regions(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool.nrows_spin.setValue(2)
    tool.ncols_spin.setValue(2)
    tool._setup_controls_changed()
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())

    tool.gridspec_region_kind_combo.setCurrentIndex(
        tool.gridspec_region_kind_combo.findData("grid")
    )
    qtbot.mousePress(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    tool.gridspec_region_kind_combo.setCurrentIndex(
        tool.gridspec_region_kind_combo.findData("axes")
    )
    qtbot.mouseRelease(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    child_grid = tool.tool_status.setup.gridspec.root.child_grids[0]
    assert child_grid.label == ""
    assert tool.gridspec_region_label_edit.isHidden()
    assert tool.gridspec_region_name_label.isHidden()
    assert child_grid.span == FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )

    qtbot.mouseDClick(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.span_rect(child_grid.span).center(),
    )
    assert tool._active_gridspec_grid_id == child_grid.grid_id
    assert [button.text() for button in tool._gridspec_breadcrumb_buttons] == [
        "Root",
        "Grid 1",
    ]

    tool.gridspec_region_kind_combo.setCurrentIndex(
        tool.gridspec_region_kind_combo.findData("axes")
    )
    child_widget = tool.gridspec_layout_widget
    child_widget.resize(child_widget.sizeHint())
    qtbot.mousePress(
        child_widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=child_widget.cell_rect((0, 0)).center(),
    )
    qtbot.mouseRelease(
        child_widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=child_widget.cell_rect((0, 0)).center(),
    )
    active_grid = tool.tool_status.setup.gridspec.root.child_grids[0]
    assert len(active_grid.axes) == 1
    assert not tool.gridspec_region_label_edit.isHidden()
    assert not tool.gridspec_region_name_label.isHidden()
    tool._gridspec_open_parent_grid()
    assert tool._active_gridspec_grid_id == "root"
    widget.resize(widget.sizeHint())
    child_axis_id = active_grid.axes[0].axes_id
    assert child_axis_id in widget.axes_ids()
    assert (
        widget.axis_rect(child_axis_id).center().x() > widget.cell_rect((0, 0)).right()
    )


def test_figure_composer_gridspec_widget_resizes_selected_region(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool.nrows_spin.setValue(2)
    tool.ncols_spin.setValue(3)
    tool._setup_controls_changed()

    original_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=2,
        col_start=0,
        col_stop=3,
    )
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool._gridspec_region_changed(axis.axes_id, original_span)
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())
    widget.set_selected_region(axis.axes_id)

    handle_pos = widget.span_rect(original_span).bottomRight() - QtCore.QPoint(2, 2)
    end_pos = widget.cell_rect((0, 1)).center()
    _drag_widget(widget, handle_pos, end_pos)

    assert len(tool.tool_status.setup.gridspec.root.axes) == 1
    assert tool.tool_status.setup.gridspec.root.axes[0].span == (
        FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=2,
        )
    )


def test_figure_composer_gridspec_widget_moves_selected_region(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool.nrows_spin.setValue(2)
    tool.ncols_spin.setValue(3)
    tool._setup_controls_changed()

    original_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool._gridspec_region_changed(axis.axes_id, original_span)
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())

    _drag_widget(
        widget,
        widget.span_rect(original_span).center(),
        widget.cell_rect((1, 2)).center(),
    )

    assert len(tool.tool_status.setup.gridspec.root.axes) == 1
    assert tool.tool_status.setup.gridspec.root.axes[0].span == (
        FigureGridSpecSpanState(
            row_start=1,
            row_stop=2,
            col_start=2,
            col_stop=3,
        )
    )


def test_figure_composer_gridspec_widget_hides_handles_after_outside_click(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool._setup_controls_changed()
    tool.show()
    qtbot.wait_until(lambda: tool.isVisible(), timeout=5000)

    axis = tool.tool_status.setup.gridspec.root.axes[0]
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())
    widget.set_selected_region(axis.axes_id)
    assert widget.selected_region_id() == axis.axes_id
    assert widget._region_handles_visible

    outside_global_pos = tool.gridspec_status_label.mapToGlobal(QtCore.QPoint(1, 1))
    widget._handle_application_event(
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(1.0, 1.0),
            QtCore.QPointF(outside_global_pos),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )

    assert widget.selected_region_id() == axis.axes_id
    assert not widget._region_handles_visible
    tool.gridspec_region_label_edit.setText("1 renamed")
    tool._gridspec_region_label_changed()
    assert tool.gridspec_region_label_edit.property("invalid") is True
    assert tool.tool_status.setup.gridspec.root.axes[0].label == ""
    tool.gridspec_region_label_edit.setText("renamed")
    tool._gridspec_region_label_changed()
    assert tool.gridspec_region_label_edit.property("invalid") is False
    assert tool.tool_status.setup.gridspec.root.axes[0].label == "renamed"


def test_figure_composer_gridspec_occupied_cells_cover_spans(qtbot) -> None:
    widget = figurecomposer_widgets._GridSpecViewWidget(mode="edit")
    qtbot.addWidget(widget)
    axes = (
        FigureGridSpecAxesState(
            axes_id="top-left",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=0,
                col_stop=1,
            ),
        ),
        FigureGridSpecAxesState(
            axes_id="bottom-span",
            span=FigureGridSpecSpanState(
                row_start=1,
                row_stop=2,
                col_start=1,
                col_stop=3,
            ),
        ),
    )
    grid = FigureGridSpecGridState(
        grid_id="root",
        nrows=2,
        ncols=3,
        axes=axes,
    )

    assert widget._occupied_grid_cells(grid) == {(0, 0), (1, 1), (1, 2)}


def test_figure_composer_gridspec_shrink_marks_invalid_regions(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool.ncols_spin.setValue(2)
    assert tool.tool_status.setup.gridspec.root.ncols == 2
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())

    qtbot.mousePress(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    qtbot.mouseRelease(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    assert len(tool.tool_status.setup.gridspec.root.axes) == 2

    tool.ncols_spin.setValue(1)
    assert tool.editor_tabs.currentWidget() is tool.layout_page
    invalid_axes_id = tool.tool_status.setup.gridspec.root.axes[1].axes_id
    assert invalid_axes_id not in tool.gridspec_axes_selector.axes_ids()
    assert any(not region.valid for region in tool.gridspec_layout_widget._regions)


def test_figure_composer_gridspec_row_shrink_ignores_invalid_regions(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.layout_mode_combo.setCurrentText("gridspec")
    tool.nrows_spin.setValue(2)
    widget = tool.gridspec_layout_widget
    widget.resize(widget.sizeHint())

    qtbot.mousePress(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((1, 0)).center(),
    )
    qtbot.mouseRelease(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((1, 0)).center(),
    )
    assert len(tool.tool_status.setup.gridspec.root.axes) == 2
    removed_span = tool.tool_status.setup.gridspec.root.axes[1].span

    tool.nrows_spin.setValue(1)
    assert any(not region.valid for region in widget._regions)
    assert widget.span_rect(removed_span) == QtCore.QRect()
    assert widget._region_at(widget.cell_rect((0, 0)).center()) is not None
    _send_mouse_move(widget, widget.cell_rect((0, 0)).center())


def test_figure_composer_gridspec_axes_targets_survive_region_delete(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.layout_mode_combo.setCurrentText("gridspec")
    first_axes_id = tool.tool_status.setup.gridspec.root.axes[0].axes_id
    tool._select_step_section("axes")
    tool._sync_axes_selector()
    assert not tool.gridspec_axes_selector.isHidden()
    tool.gridspec_axes_selector.set_selected_axes_ids((first_axes_id,), emit=True)
    assert tool.tool_status.operations[0].axes.axes_ids == (first_axes_id,)

    tool.gridspec_layout_widget.set_selected_region(first_axes_id)
    tool._gridspec_delete_selected_region()
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert tool.tool_status.operations[0].axes.axes_ids == (first_axes_id,)
    target_text = tool._axes_target_text(tool.tool_status.operations[0].axes)
    assert target_text == "1 target axis removed"
    assert first_axes_id not in target_text
    tool._sync_axes_selector()
    status_text = tool.target_axes_status_label.text()
    assert status_text == "1 target axis was removed by the current GridSpec layout."
    assert first_axes_id not in status_text


def test_figure_composer_gridspec_axes_selector_click_keeps_surviving_removed_target(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    span_00 = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    span_01 = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=2,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="ax0",
                                span=span_00,
                            ),
                            FigureGridSpecAxesState(
                                axes_id="ax1",
                                span=span_01,
                            ),
                        ),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes_ids=("ax0", "ax1")),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.recipe_page)
    tool._select_step_section("axes")
    tool._sync_axes_selector()

    tool.gridspec_layout_widget.set_selected_region("ax1")
    tool._gridspec_delete_selected_region()
    tool._sync_axes_selector()
    tool.gridspec_axes_selector.resize(tool.gridspec_axes_selector.sizeHint())

    assert tool.tool_status.operations[0].axes.axes_ids == ("ax0", "ax1")
    assert tool.gridspec_axes_selector.selected_axes_ids() == ("ax0",)
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])

    qtbot.mouseClick(
        tool.gridspec_axes_selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=tool.gridspec_axes_selector.axis_rect("ax0").center(),
    )

    assert tool.tool_status.operations[0].axes.axes_ids == ("ax0",)
    assert not tool._operation_has_invalid_axes(tool.tool_status.operations[0])


def test_figure_composer_gridspec_delete_selects_nearby_axes(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    left_axis = FigureGridSpecAxesState(
        axes_id="left-axis",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    middle_axis = FigureGridSpecAxesState(
        axes_id="middle-axis",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
    )
    far_axis = FigureGridSpecAxesState(
        axes_id="far-axis",
        span=FigureGridSpecSpanState(
            row_start=1,
            row_stop=2,
            col_start=2,
            col_stop=3,
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=2,
                        ncols=3,
                        axes=(left_axis, middle_axis, far_axis),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.gridspec_layout_widget.set_selected_region(middle_axis.axes_id)
    tool._gridspec_delete_selected_region()

    axes_ids = tuple(axis.axes_id for axis in tool.tool_status.setup.gridspec.root.axes)
    assert axes_ids == (left_axis.axes_id, far_axis.axes_id)
    assert tool.gridspec_layout_widget.selected_region_id() == left_axis.axes_id
    assert tool.gridspec_delete_region_button.isEnabled()


def test_figure_composer_plot_slices_operation_uses_separate_window(
    qtbot, monkeypatch, recwarn
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(0.0, 1.0),
                ).model_copy(
                    update={
                        "axis": "image",
                        "cmap": "viridis_r",
                        "gamma": 0.5,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    tool._update_operation_editor()

    assert tool.findChildren(FigureCanvasQTAgg) == []
    assert tool.findChildren(NavigationToolbar2QT) == []
    assert tool.findChildren(QtWidgets.QSplitter) == [tool.recipe_splitter]
    assert tool.recipe_splitter.widget(0) is tool.operation_list
    assert tool.recipe_splitter.widget(1) is tool.step_inspector
    editor_tabs = tool.findChild(QtWidgets.QTabWidget, "figureComposerEditorTabs")
    assert editor_tabs is tool.editor_tabs
    assert [
        editor_tabs.widget(index).objectName() for index in range(editor_tabs.count())
    ] == ["figureComposerLayoutPage", "figureComposerRecipePage"]
    assert editor_tabs.currentWidget() is tool.recipe_page
    assert isinstance(tool.layout_page.layout(), QtWidgets.QGridLayout)
    layout_grid = typing.cast("QtWidgets.QGridLayout", tool.layout_page.layout())
    assert layout_grid.rowCount() == 10
    assert layout_grid.columnCount() == 5
    assert (
        tool.findChild(QtWidgets.QWidget, "figureComposerLayoutModeControls")
        is not None
    )
    assert tool.findChild(QtWidgets.QWidget, "figureComposerGridControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerSizeControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerSizeMmControls") is not None
    dpi_label = tool.findChild(QtWidgets.QLabel, "figureComposerDpiControls")
    assert dpi_label is not None
    assert dpi_label.buddy() is tool.dpi_spin
    assert tool.findChild(QtWidgets.QWidget, "figureComposerShareControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerRatioControls") is not None
    assert layout_grid.getItemPosition(layout_grid.indexOf(dpi_label)) == (
        5,
        0,
        1,
        2,
    )
    assert layout_grid.getItemPosition(layout_grid.indexOf(tool.dpi_spin)) == (
        5,
        2,
        1,
        3,
    )
    assert tool.gridspec_editor_widget.isHidden()
    gridspec_container = tool.findChild(
        QtWidgets.QWidget, "figureComposerGridSpecEditorContainer"
    )
    assert gridspec_container is tool.gridspec_editor_container
    assert tool.findChild(QtWidgets.QFrame, "figureComposerGridSpecEditorTopLine")
    assert tool.findChild(QtWidgets.QFrame, "figureComposerGridSpecEditorBottomLine")
    assert layout_grid.getItemPosition(layout_grid.indexOf(gridspec_container)) == (
        2,
        0,
        1,
        5,
    )
    layout_label = tool.findChild(QtWidgets.QLabel, "figureComposerLayoutControls")
    assert layout_label is not None
    assert layout_grid.getItemPosition(layout_grid.indexOf(layout_label)) == (
        6,
        0,
        1,
        2,
    )
    assert layout_grid.getItemPosition(layout_grid.indexOf(tool.layout_combo)) == (
        6,
        2,
        1,
        3,
    )
    add_step_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerAddStepButton"
    )
    assert add_step_button is tool.add_step_button
    assert add_step_button.parent() is tool.recipe_page
    assert add_step_button.menu() is None
    assert add_step_button.property("uses_inline_menu_arrow") is True
    assert tool.add_step_menu.parent() is add_step_button
    assert add_step_button.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
    )
    step_toolbar_buttons = (
        tool.add_step_button,
        tool.duplicate_operation_button,
        tool.move_operation_up_button,
        tool.move_operation_down_button,
        tool.remove_operation_button,
    )
    assert all(button.styleSheet() == "" for button in step_toolbar_buttons)
    assert {button.toolButtonStyle() for button in step_toolbar_buttons} == {
        QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
    }
    assert {
        button.sizePolicy().horizontalPolicy() for button in step_toolbar_buttons
    } == {QtWidgets.QSizePolicy.Policy.Minimum}
    assert {
        button.sizePolicy().verticalPolicy() for button in step_toolbar_buttons
    } == {QtWidgets.QSizePolicy.Policy.Fixed}
    assert len({button.sizeHint().height() for button in step_toolbar_buttons}) == 1
    assert [action.data() for action in tool.add_step_menu.actions()] == [
        "set_palette",
        "plot_slices",
        "line",
        "bz_overlay",
        "photon_energy_overlay",
        "method:erlab",
        "method:axes",
        "method:figure",
        "custom",
    ]
    assert [action.text() for action in tool.add_step_menu.actions()] == [
        "Set Palette",
        "Slice Plot",
        "Line/Profile",
        "BZ Overlay",
        "Photon Energy Overlay",
        "ERLab Method",
        "Axes Method",
        "Figure Method",
        "Python",
    ]
    assert tool.findChild(QtWidgets.QTabWidget, "figureComposerInspectorTabs") is None
    assert tool.findChild(QtWidgets.QToolBox) is None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerStepNavigator") is not None
    assert tool.step_editor_stack.objectName() == "figureComposerStepSectionStack"
    assert tool.step_section_keys == [
        "sources",
        "axes",
        "selection",
        "view",
        "colors",
        "advanced",
    ]
    assert [
        tool.step_editor_stack.widget(index).objectName()
        for index in range(tool.step_editor_stack.count())
    ] == [
        "figureComposerStepSourcesPage",
        "figureComposerTargetAxesPage",
        "figureComposerPlotSlicesSelectionPage",
        "figureComposerPlotSlicesViewPage",
        "figureComposerPlotSlicesColorsPage",
        "figureComposerPlotSlicesAdvancedPage",
    ]
    assert tool.findChild(QtWidgets.QTabWidget, "figureComposerPlotSlicesTabs") is None
    colors_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesColorsPage"
    )
    selection_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesSelectionPage"
    )
    view_page = tool.findChild(QtWidgets.QWidget, "figureComposerPlotSlicesViewPage")
    crop_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesCropCheck"
    )
    order_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerOrderCombo")
    transpose_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerTransposeCheck"
    )
    same_limits_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerSameLimitsCombo"
    )
    axis_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerAxisCombo")
    annotate_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAnnotateKwEdit"
    )
    colorbar_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerColorbarKwEdit"
    )
    assert colors_page is not None
    assert selection_page is not None
    assert view_page is not None
    assert crop_check is not None
    assert order_combo is not None
    assert transpose_check is not None
    assert same_limits_combo is not None
    assert axis_combo is not None
    assert annotate_kwargs_edit is not None
    assert colorbar_kwargs_edit is not None
    assert view_page.isAncestorOf(crop_check)
    assert view_page.isAncestorOf(order_combo)
    assert view_page.isAncestorOf(transpose_check)
    assert not selection_page.isAncestorOf(crop_check)
    assert same_limits_combo.parent() is colors_page
    assert axis_combo.parent() is view_page
    assert annotate_kwargs_edit.parent() is view_page
    assert colorbar_kwargs_edit.parent() is colors_page
    assert all(
        isinstance(button.property("section_title"), str)
        for button in tool.step_section_buttons.values()
    )
    assert tool.axes_selector.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    annotate_kwargs_edit.setFocus()
    annotate_kwargs_edit.setText("fontsize=8, color='black'")
    annotate_kwargs_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].annotate_kw == {
        "fontsize": 8,
        "color": "black",
    }
    colorbar_kwargs_edit.setFocus()
    colorbar_kwargs_edit.setText("fraction=0.05, pad=0.02")
    colorbar_kwargs_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].colorbar_kw == {
        "fraction": 0.05,
        "pad": 0.02,
    }
    if tool._preview_render_update_pending:
        qtbot.waitUntil(lambda: not tool._preview_render_update_pending, timeout=1000)
    tool.dpi_spin.setValue(180.0)
    tool._setup_controls_changed()
    assert tool.tool_status.setup.dpi == 180.0
    tool._update_operation_editor()
    annotate_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAnnotateKwEdit"
    )
    colorbar_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerColorbarKwEdit"
    )
    assert annotate_kwargs_edit is not None
    assert colorbar_kwargs_edit is not None
    assert annotate_kwargs_edit.text() == 'fontsize=8, color="black"'
    assert colorbar_kwargs_edit.text() == "fraction=0.05, pad=0.02"
    assert not any(
        spinbox.keyboardTracking()
        for spinbox in (
            tool.nrows_spin,
            tool.ncols_spin,
            tool.width_spin,
            tool.height_spin,
            tool.width_mm_spin,
            tool.height_mm_spin,
            tool.dpi_spin,
        )
    )
    tool._select_step_section("colors")
    tool._update_current_operation(axis="equal")
    assert tool.findChild(QtWidgets.QToolBox) is None
    assert tool._current_step_section_key == "colors"
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    tool._select_step_section("view")
    view_page = tool.step_editor_stack.currentWidget()
    xlim_edit = view_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesXLimEdit"
    )
    ylim_edit = view_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesYLimEdit"
    )
    assert xlim_edit is not None
    assert ylim_edit is not None
    assert xlim_edit.text() == ""
    assert "," in xlim_edit.placeholderText()
    assert ylim_edit.text() == ""
    assert "," in ylim_edit.placeholderText()
    xlim_edit.setFocus()
    xlim_edit.setText("0, 1")
    xlim_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].xlim == (0.0, 1.0)
    ylim_edit.setFocus()
    ylim_edit.setText("2.5")
    ylim_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].ylim == 2.5
    restored_status = FigureRecipeState.model_validate(tool.tool_status.model_dump())
    assert restored_status.operations[0].ylim == 2.5
    assert "ylim=2.5" in tool.generated_code()
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesViewPage"
    )
    qtbot.mouseClick(
        tool.step_section_buttons["colors"], QtCore.Qt.MouseButton.LeftButton
    )
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    operation_item = tool.operation_list.item(0)
    operation_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].enabled is False
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    operation_item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].enabled is True
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    tool._select_step_section("axes")
    tool.axes_expression_edit.setFocus()
    tool.axes_expression_edit.setText("axs[:, 0]")
    tool.axes_expression_edit.editingFinished.emit()
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.expression == "axs[:, 0]"
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerTargetAxesPage"
    )
    tool._target_current_operation_all_axes()
    selector = tool.axes_selector
    selector.resize(selector.sizeHint())
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        selector.cell_rect((0, 1)).center(),
    )
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.axes == ((0, 0),)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        selector.cell_rect((0, 1)).center(),
    )
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.axes == ((0, 1),)
    qtbot.mouseClick(
        selector,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        selector.cell_rect((0, 0)).center(),
    )
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.axes == ((0, 0), (0, 1))
    tool._target_current_operation_valid_axes()
    start = selector.cell_rect((0, 0)).center()
    end = selector.cell_rect((0, 1)).center()
    _drag_widget(selector, start, end)
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.axes == ((0, 0), (0, 1))
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerTargetAxesPage"
    )
    tool._target_current_operation_all_axes()
    tool._select_step_section("colors")
    cmap_combo = tool.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
    )
    cmap_reverse_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerCmapReverseCheck"
    )
    gamma_widget = tool.findChild(
        erlab.interactive.colors.ColorMapGammaWidget, "figureComposerGammaWidget"
    )
    norm_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert cmap_combo is not None
    assert cmap_reverse_check is not None
    assert gamma_widget is not None
    assert norm_combo is not None
    assert cmap_combo.toolTip()
    assert cmap_reverse_check.toolTip()
    assert gamma_widget.toolTip()
    assert norm_combo.currentText() == "PowerNorm"
    assert "Default" not in [
        norm_combo.itemText(index) for index in range(norm_combo.count())
    ]
    assert cmap_combo.currentText() == "viridis"
    assert cmap_reverse_check.isChecked()
    assert gamma_widget.value() == 0.5
    cmap_reverse_check.setChecked(False)
    assert tool.tool_status.operations[0].cmap == "viridis"
    cmap_combo = typing.cast(
        "erlab.interactive.colors.ColorMapComboBox",
        tool.findChild(
            erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
        ),
    )
    cmap_reverse_check = typing.cast(
        "QtWidgets.QCheckBox",
        tool.findChild(QtWidgets.QCheckBox, "figureComposerCmapReverseCheck"),
    )
    _activate_combo_text(cmap_combo, "magma")
    assert tool.tool_status.operations[0].cmap == "magma"
    cmap_reverse_check = typing.cast(
        "QtWidgets.QCheckBox",
        tool.findChild(QtWidgets.QCheckBox, "figureComposerCmapReverseCheck"),
    )
    cmap_reverse_check.setChecked(True)
    assert tool.tool_status.operations[0].cmap == "magma_r"
    gamma_widget = typing.cast(
        "erlab.interactive.colors.ColorMapGammaWidget",
        tool.findChild(
            erlab.interactive.colors.ColorMapGammaWidget, "figureComposerGammaWidget"
        ),
    )
    gamma_widget.setValue(0.75)
    assert tool.tool_status.operations[0].norm_gamma == 0.75
    assert tool.tool_status.operations[0].gamma is None
    current_fig = plt.figure()
    try:
        tool._update_current_operation(colorbar="right")
    finally:
        plt.close(current_fig)
    assert tool.tool_status.operations[0].colorbar == "right"
    assert not any(
        "Adding colorbar to a different Figure" in str(warning.message)
        for warning in recwarn
    )
    tool._update_current_operation(colorbar="none")
    assert tool.tool_status.operations[0].colorbar == "none"
    assert not any(
        "constrained_layout not applied" in str(warning.message) for warning in recwarn
    )
    assert tool._figure_window is not None
    preview = tool.refresh_preview_pixmap()
    assert preview is not None
    assert not preview.isNull()
    assert preview.width() > 0
    assert preview.height() > 0
    assert tool._figure_window is not None

    setup_before = tool.tool_status.setup.model_copy()
    code_before = tool.generated_code()
    tool.resize(240, 360)
    assert tool.tool_status.setup == setup_before
    assert tool.generated_code() == code_before

    exported: dict[str, tuple[float, float]] = {}
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("figure.png", ""),
    )

    original_savefig = Figure.savefig

    def record_savefig(fig: Figure, filename: str, **kwargs: object) -> None:
        exported["figsize"] = tuple(float(value) for value in fig.get_size_inches())

    monkeypatch.setattr(
        Figure,
        "savefig",
        record_savefig,
    )
    tool.export_figure()
    assert exported["figsize"] == setup_before.figsize
    assert tool._figure_window is not None
    monkeypatch.setattr(Figure, "savefig", original_savefig)

    show_activations: list[bool] = []
    figure_window = tool.figure_window
    original_show_for_setup = figure_window.show_for_setup

    def record_show_for_setup(*args, activate: bool) -> None:
        show_activations.append(activate)
        original_show_for_setup(*args, activate=activate)

    monkeypatch.setattr(figure_window, "show_for_setup", record_show_for_setup)

    tool._hide_figure_window()
    tool.show()
    qtbot.wait_until(lambda: bool(show_activations), timeout=5000)
    qtbot.wait_until(lambda: tool.figure_window.isVisible(), timeout=5000)
    assert show_activations[-1] is False
    activation_count = len(show_activations)
    tool._update_current_operation(axis="auto")
    assert len(show_activations) == activation_count
    figure_window = tool.figure_window
    figure_window.canvas.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
    qtbot.keyClick(
        figure_window.canvas,
        QtCore.Qt.Key.Key_W,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    qtbot.wait_until(lambda: not figure_window.isVisible(), timeout=5000)
    assert len(tool.tool_status.operations) == 1
    activation_count = len(show_activations)
    tool.show_figure_window()
    qtbot.wait_until(lambda: figure_window.isVisible(), timeout=5000)
    qtbot.wait_until(lambda: len(show_activations) > activation_count, timeout=5000)
    assert True in show_activations[activation_count:]
    activation_count = len(show_activations)
    tool.width_spin.setValue(7.0)
    tool.height_spin.setValue(5.0)
    tool._setup_controls_changed()
    assert len(show_activations) == activation_count
    base_dpi = float(figure_window.figure._original_dpi)
    qtbot.wait_until(
        lambda: (
            abs(figure_window.canvas.width() - round(7.0 * base_dpi)) <= 2
            and abs(figure_window.canvas.height() - round(5.0 * base_dpi)) <= 2
        ),
        timeout=5000,
    )
    assert tool.tool_status.setup.figsize == (7.0, 5.0)
    assert np.isclose(tool.width_mm_spin.value(), 7.0 * 25.4, atol=0.01)
    assert np.isclose(tool.height_mm_spin.value(), 5.0 * 25.4, atol=0.01)

    tool.width_mm_spin.setValue(127.0)
    tool.height_mm_spin.setValue(76.2)
    tool._size_mm_controls_changed()
    assert tool.tool_status.setup.figsize == (5.0, 3.0)
    assert np.isclose(tool.width_spin.value(), 5.0)
    assert np.isclose(tool.height_spin.value(), 3.0)

    size_delta = figure_window.size() - figure_window.canvas.size()
    target_width = 6.25
    target_height = 4.5
    figure_window.resize(
        round(target_width * base_dpi) + size_delta.width(),
        round(target_height * base_dpi) + size_delta.height(),
    )
    qtbot.wait_until(
        lambda: (
            np.isclose(tool.tool_status.setup.figsize[0], target_width, atol=0.03)
            and np.isclose(tool.tool_status.setup.figsize[1], target_height, atol=0.03)
        ),
        timeout=5000,
    )
    assert np.isclose(tool.width_spin.value(), target_width, atol=0.03)
    assert np.isclose(tool.height_spin.value(), target_height, atol=0.03)
    assert np.isclose(tool.width_mm_spin.value(), target_width * 25.4, atol=0.8)
    assert np.isclose(tool.height_mm_spin.value(), target_height * 25.4, atol=0.8)
    typing.cast("typing.Any", figure_window.canvas)._set_device_pixel_ratio(2.0)
    assert float(figure_window.figure.dpi) == base_dpi * 2.0
    tool._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
    assert np.isclose(tool.tool_status.setup.figsize[0], target_width, atol=0.03)
    assert np.isclose(tool.tool_status.setup.figsize[1], target_height, atol=0.03)
    tool.figure.set_size_inches((2.0, 2.0), forward=False)
    figurecomposer_rendering._render_preview(tool)
    canvas_size = figure_window.canvas.size()
    assert np.isclose(
        tool.figure.get_size_inches()[0],
        canvas_size.width() / base_dpi,
        atol=0.01,
    )
    assert np.isclose(
        tool.figure.get_size_inches()[1],
        canvas_size.height() / base_dpi,
        atol=0.01,
    )

    code = tool.generated_code()
    assert "squeeze=False" in code
    assert "axes=axs" in code
    assert "eplt.plot_slices" in code
    assert "annotate_kw" in code
    assert "colorbar_kw" in code
    assert "tools[" not in code
    assert "_manager" not in code


def test_figure_composer_line_profile_operation_uses_semantic_sections(
    qtbot,
) -> None:
    data = _figure_composer_line_slice_source("profile_data")
    operation = FigureOperationState.line(
        label="profiles",
        source="profile_data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_selection": {"eV": slice(-0.5, 0.5)},
            "line_reduce": "both",
            "line_labels": ("left", "right"),
            "line_colors": ("C0", "C1"),
            "line_kw": {"linestyle": "--", "marker": "o"},
            "line_normalize": "max",
            "line_scales": (2.0,),
            "line_offsets": (0.0, 1.0),
            "xlim": (-0.5, 0.5),
            "ylim": (0.0, None),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(tool)

    assert tool.step_section_keys == [
        "sources",
        "axes",
        "selection",
        "view",
        "style",
        "other",
    ]
    assert tool.step_section_buttons["other"].property("section_title") == "Transform"
    assert [
        tool.step_editor_stack.widget(index).objectName()
        for index in range(tool.step_editor_stack.count())
    ] == [
        "figureComposerStepSourcesPage",
        "figureComposerTargetAxesPage",
        "figureComposerLineSelectionPage",
        "figureComposerLineViewPage",
        "figureComposerLineStylePage",
        "figureComposerLineOtherPage",
    ]

    selection_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerLineSelectionPage"
    )
    view_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineViewPage")
    style_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineStylePage")
    other_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineOtherPage")
    assert selection_page is not None
    assert view_page is not None
    assert style_page is not None
    assert other_page is not None

    section_controls = (
        (
            selection_page,
            (
                "figureComposerProfileCoordinateCombo",
                "figureComposerProfileValuesCombo",
                "figureComposerLineSelectionEdit",
                "figureComposerProfileIterDimCombo",
                "figureComposerProfileReduceCombo",
            ),
        ),
        (
            view_page,
            (
                "figureComposerProfilePlacementCombo",
                "figureComposerDataValuesAxisCombo",
                "figureComposerLineXLimEdit",
                "figureComposerLineYLimEdit",
            ),
        ),
        (
            style_page,
            (
                "figureComposerLineLabelsEdit",
                "figureComposerLineColorModeCombo",
                "figureComposerLineColorsEdit",
                "figureComposerLineStyleCombo",
                "figureComposerLineWidthSpin",
                "figureComposerLineMarkerCombo",
                "figureComposerLineMarkerSizeSpin",
                "figureComposerLineMarkerFaceColorEdit",
                "figureComposerLineMarkerEdgeColorEdit",
                "figureComposerLineGradientCheck",
            ),
        ),
        (
            other_page,
            (
                "figureComposerLineNormalizeCombo",
                "figureComposerLineScalesEdit",
                "figureComposerLineOffsetSourceCombo",
                "figureComposerLineOffsetsEdit",
            ),
        ),
    )
    pages = (selection_page, view_page, style_page, other_page)
    for expected_page, control_names in section_controls:
        for name in control_names:
            widget = tool.findChild(QtWidgets.QWidget, name)
            assert widget is not None
            for page in pages:
                if page is expected_page:
                    assert page.isAncestorOf(widget)
                else:
                    assert not page.isAncestorOf(widget)

    for key, page in (
        ("selection", selection_page),
        ("view", view_page),
        ("style", style_page),
        ("other", other_page),
    ):
        tool._select_step_section(key)
        assert tool.step_editor_stack.currentWidget() is page
        assert tool.tool_status.operations[0] == operation


def test_figure_composer_step_section_buttons_are_tab_focusable(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    qtbot.waitUntil(lambda: not tool._step_tab_order_update_pending, timeout=1000)

    buttons = tuple(tool.step_section_buttons.values())
    assert len(buttons) > 1
    assert all(
        button.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus for button in buttons
    )
    step_action_buttons = (
        tool.add_step_button,
        tool.copy_operation_button,
        tool.cut_operation_button,
        tool.paste_operation_button,
        tool.duplicate_operation_button,
        tool.move_operation_up_button,
        tool.move_operation_down_button,
        tool.remove_operation_button,
        tool.operation_list,
    )
    for index, button in enumerate(step_action_buttons[:-1]):
        assert button.nextInFocusChain() is step_action_buttons[index + 1]
    for index, button in enumerate(buttons[:-1]):
        assert button.nextInFocusChain() is buttons[index + 1]

    tool._select_step_section(tool.step_section_keys[-1])
    qtbot.waitUntil(lambda: not tool._step_tab_order_update_pending, timeout=1000)
    buttons = tuple(tool.step_section_buttons.values())
    for index, button in enumerate(buttons[:-1]):
        assert button.nextInFocusChain() is buttons[index + 1]


def test_figure_composer_toolbar_uses_composer_actions(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    toolbar = tool.figure_window.toolbar

    assert toolbar.objectName() == "figureComposerNavigationToolbar"
    assert "home" not in toolbar._actions
    for action_id in (
        "show_composer",
        "back",
        "forward",
        "pan",
        "zoom",
        "configure_subplots",
        "edit_parameters",
        "copy_figure_to_clipboard",
        "save_figure",
    ):
        action = toolbar._actions[action_id]
        assert action.objectName() == f"figureComposerToolbar_{action_id}"
        assert not action.icon().isNull()

    undo_action = toolbar._actions["back"]
    redo_action = toolbar._actions["forward"]
    assert undo_action.shortcut() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Undo
    )
    assert redo_action.shortcut() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Redo
    )
    assert undo_action.shortcutContext() == QtCore.Qt.ShortcutContext.WindowShortcut
    assert redo_action.shortcutContext() == QtCore.Qt.ShortcutContext.WindowShortcut

    calls: list[str] = []
    monkeypatch.setattr(tool, "export_figure", lambda: calls.append("export"))
    monkeypatch.setattr(
        tool, "_show_subplot_adjust_dialog", lambda: calls.append("subplots")
    )
    monkeypatch.setattr(
        tool, "_show_axes_customize_dialog", lambda: calls.append("axes")
    )
    monkeypatch.setattr(
        tool,
        "_show_composer_from_figure_window",
        lambda: calls.append("composer"),
    )

    toolbar._actions["show_composer"].trigger()
    toolbar._actions["save_figure"].trigger()
    toolbar._actions["configure_subplots"].trigger()
    toolbar._actions["edit_parameters"].trigger()

    assert calls == ["composer", "export", "subplots", "axes"]
    assert figurecomposer_widgets._noop_toolbar_callback() is None
    toolbar.changeEvent(QtCore.QEvent(QtCore.QEvent.Type.PaletteChange))
    for action_id in ("show_composer", "pan", "zoom", "copy_figure_to_clipboard"):
        assert not toolbar._actions[action_id].icon().isNull()


def test_figure_composer_figure_window_refreshes_toolbar_icons_on_palette_change(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    toolbar = tool.figure_window.toolbar
    icon_names: list[str] = []
    reset_calls: list[None] = []

    def record_icon(name: str) -> QtGui.QIcon:
        icon_names.append(name)
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        return QtGui.QIcon(pixmap)

    monkeypatch.setattr(toolbar, "_icon", record_icon)
    monkeypatch.setattr(
        erlab.interactive.utils.qtawesome,
        "reset_cache",
        lambda: reset_calls.append(None),
    )

    tool.figure_window.changeEvent(
        QtCore.QEvent(QtCore.QEvent.Type.ApplicationPaletteChange)
    )

    assert reset_calls == [None]
    assert "figure_composer" in icon_names
    assert "move" in icon_names
    assert "zoom_to_rect" in icon_names
    assert "copy_figure_to_clipboard" in icon_names
    assert not toolbar._actions["pan"].icon().isNull()


def test_figure_composer_toolbar_navigation_helper_edges(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    toolbar = tool.figure_window.toolbar

    limited_toolitems = [
        item
        for item in figurecomposer_widgets._FigureComposerNavigationToolbar.toolitems
        if item[3] not in {"back", "forward"}
    ]
    with monkeypatch.context() as context:
        context.setattr(
            figurecomposer_widgets._FigureComposerNavigationToolbar,
            "toolitems",
            limited_toolitems,
        )
        limited_window = figurecomposer_widgets._FigureComposerDisplayWindow(
            FigureSubplotsState()
        )
        qtbot.addWidget(limited_window)
    assert "back" not in limited_window.toolbar._actions
    assert "forward" not in limited_window.toolbar._actions

    assert (
        figurecomposer_widgets._noop_navigation_callback({object(): (True, False)})
        is None
    )
    assert (
        figurecomposer_widgets._noop_colorbar_callback({object(): (0.0, 1.0)}) is None
    )

    class Axis:
        def __init__(
            self,
            xlim: tuple[float, float] = (0.0, 1.0),
            ylim: tuple[float, float] = (0.0, 1.0),
        ) -> None:
            self.xlim = xlim
            self.ylim = ylim

        def get_xlim(self) -> tuple[float, float]:
            return self.xlim

        def get_ylim(self) -> tuple[float, float]:
            return self.ylim

    class RaisingAxis:
        def get_xlim(self) -> tuple[float, float]:
            raise ValueError

        def get_ylim(self) -> tuple[float, float]:
            raise TypeError

    class Mappable:
        def __init__(self, clim: tuple[float | None, float | None]) -> None:
            self.clim = clim

        def get_clim(self) -> tuple[float | None, float | None]:
            return self.clim

    class RaisingMappable:
        def get_clim(self) -> tuple[float, float]:
            raise ValueError

    axis = Axis()
    assert figurecomposer_widgets._axis_limit_pair(axis, "get_xlim") == (0.0, 1.0)
    assert figurecomposer_widgets._axis_limit_pair(object(), "get_xlim") is None
    assert figurecomposer_widgets._axis_limit_pair(RaisingAxis(), "get_xlim") is None
    assert figurecomposer_widgets._axis_view(axis) == ((0.0, 1.0), (0.0, 1.0))
    assert figurecomposer_widgets._axis_view(object()) is None

    mappable = Mappable((0.0, 1.0))
    colorbar_axis = Axis()
    colorbar_axis._colorbar = type("Colorbar", (), {"mappable": mappable})()
    colorbar_without_mappable = Axis()
    colorbar_without_mappable._colorbar = None
    invalid_clim_axis = Axis()
    invalid_clim_axis._colorbar = type(
        "InvalidColorbar",
        (),
        {"mappable": Mappable((None, 1.0))},
    )()

    assert figurecomposer_widgets._is_colorbar_axis(colorbar_axis)
    assert not figurecomposer_widgets._is_colorbar_axis(axis)
    assert figurecomposer_widgets._colorbar_mappable(colorbar_axis) is mappable
    assert figurecomposer_widgets._colorbar_mappable(colorbar_without_mappable) is None
    assert figurecomposer_widgets._mappable_clim(mappable) == (0.0, 1.0)
    assert figurecomposer_widgets._mappable_clim(object()) is None
    assert figurecomposer_widgets._mappable_clim(RaisingMappable()) is None
    assert figurecomposer_widgets._mappable_clim(Mappable((None, 1.0))) is None
    assert figurecomposer_widgets._mappable_clim(Mappable(("bad", 1.0))) is None

    navigation_views = toolbar._capture_navigation_views(
        (axis, colorbar_axis, object(), RaisingAxis())
    )
    assert navigation_views == {axis: ((0.0, 1.0), (0.0, 1.0))}
    assert toolbar._capture_colorbar_clims(
        (
            axis,
            colorbar_axis,
            colorbar_without_mappable,
            invalid_clim_axis,
            object(),
            RaisingAxis(),
        )
    ) == {mappable: (0.0, 1.0)}

    navigation_calls: list[dict[object, tuple[bool, bool]]] = []
    colorbar_calls: list[dict[object, tuple[float, float]]] = []
    toolbar._navigation_callback = navigation_calls.append
    toolbar._colorbar_callback = colorbar_calls.append

    axis.xlim = (0.2, 0.8)
    toolbar._commit_navigation_views(
        {
            axis: ((0.0, 1.0), (0.0, 1.0)),
            object(): ((0.0, 1.0), (0.0, 1.0)),
        }
    )
    assert navigation_calls == [{axis: (True, False)}]

    toolbar._commit_navigation_views({axis: ((0.2, 0.8), (0.0, 1.0))})
    assert navigation_calls == [{axis: (True, False)}]

    toolbar._commit_colorbar_clims({})
    mappable.clim = (0.25, 0.75)
    toolbar._commit_colorbar_clims({mappable: (0.0, 1.0), object(): (0.0, 1.0)})
    assert colorbar_calls == [{mappable: (0.25, 0.75)}]
    toolbar._commit_colorbar_clims({mappable: (0.25, 0.75)})
    assert colorbar_calls == [{mappable: (0.25, 0.75)}]

    back_action = toolbar._actions.pop("back")
    forward_action = toolbar._actions.pop("forward")
    try:
        toolbar.set_history_buttons()
    finally:
        toolbar._actions["back"] = back_action
        toolbar._actions["forward"] = forward_action

    pan_axis = Axis()
    zoom_axis = Axis()

    class ToolbarInfo:
        def __init__(self, axes: tuple[object, ...]) -> None:
            self.axes = axes

    def fake_press_pan(self, _event):
        self._pan_info = ToolbarInfo((pan_axis, colorbar_axis))

    def fake_press_zoom(self, _event):
        self._zoom_info = ToolbarInfo((zoom_axis, colorbar_axis))

    monkeypatch.setattr(
        figurecomposer_widgets.NavigationToolbar, "press_pan", fake_press_pan
    )
    monkeypatch.setattr(
        figurecomposer_widgets.NavigationToolbar,
        "release_pan",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        figurecomposer_widgets.NavigationToolbar,
        "press_zoom",
        fake_press_zoom,
    )
    monkeypatch.setattr(
        figurecomposer_widgets.NavigationToolbar,
        "release_zoom",
        lambda *_args: None,
    )

    toolbar.press_pan(object())
    assert pan_axis in toolbar._navigation_press_views
    assert mappable in toolbar._colorbar_press_clims
    toolbar.release_pan(object())
    assert toolbar._navigation_press_views == {}
    assert toolbar._colorbar_press_clims == {}

    toolbar.press_zoom(object())
    assert zoom_axis in toolbar._navigation_press_views
    assert mappable in toolbar._colorbar_press_clims
    toolbar.release_zoom(object())
    assert toolbar._navigation_press_views == {}
    assert toolbar._colorbar_press_clims == {}


def test_figure_composer_toolbar_copies_canvas_to_clipboard(qtbot, monkeypatch) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)
    tool.figure_window.canvas.draw()

    clipboard = QtWidgets.QApplication.clipboard()
    clipboard.clear()
    tool.figure_window.toolbar._actions["copy_figure_to_clipboard"].trigger()

    copied = clipboard.pixmap()
    assert not copied.isNull()
    assert copied.size().width() > 0
    assert copied.size().height() > 0

    clipboard.clear()
    with monkeypatch.context() as context:
        context.setattr(tool.figure_window.canvas, "grab", lambda: QtGui.QPixmap())
        tool.figure_window.toolbar._actions["copy_figure_to_clipboard"].trigger()
    assert clipboard.pixmap().isNull()

    with monkeypatch.context() as context:
        context.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: False)
        tool.figure_window.toolbar._actions["copy_figure_to_clipboard"].trigger()


def test_figure_composer_subplots_adjust_helpers_repair_invalid_pairs() -> None:
    left_min, left_max = figurecomposer_subplot_adjust.subplots_adjust_spinbox_range(
        "left",
        {"right": 0.0},
    )
    right_min, right_max = figurecomposer_subplot_adjust.subplots_adjust_spinbox_range(
        "right",
        {"left": 1.0},
    )
    top_min, top_max = figurecomposer_subplot_adjust.subplots_adjust_spinbox_range(
        "top",
        {"bottom": 1.0},
    )

    assert left_min == pytest.approx(left_max)
    assert right_min == pytest.approx(right_max)
    assert top_min == pytest.approx(top_max)

    repaired_left = figurecomposer_subplot_adjust.normalize_subplots_adjust_kwargs(
        {"left": 1.0, "right": 1.0},
        changed_key="left",
    )
    repaired_right = figurecomposer_subplot_adjust.normalize_subplots_adjust_kwargs(
        {"left": 0.0, "right": 0.0},
        changed_key="right",
    )
    repaired_left_at_min = (
        figurecomposer_subplot_adjust.normalize_subplots_adjust_kwargs(
            {"left": 0.0, "right": 0.0},
            changed_key="left",
        )
    )
    repaired_right_at_max = (
        figurecomposer_subplot_adjust.normalize_subplots_adjust_kwargs(
            {"left": 1.0, "right": 1.0},
            changed_key="right",
        )
    )

    assert repaired_left["left"] < repaired_left["right"]
    assert repaired_right["left"] < repaired_right["right"]
    assert repaired_left_at_min["left"] < repaired_left_at_min["right"]
    assert repaired_right_at_max["left"] < repaired_right_at_max["right"]


def test_figure_composer_toolbar_subplot_dialog_updates_recipe(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._show_subplot_adjust_dialog()
    dialog = tool._subplot_adjust_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    engine_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarLayoutEngineCombo"
    )
    top_spin = dialog.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerToolbarSubplotAdjust_top"
    )
    bottom_spin = dialog.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerToolbarSubplotAdjust_bottom"
    )
    assert engine_combo is not None
    assert top_spin is not None
    assert bottom_spin is not None
    assert engine_combo.currentText() == "none"
    assert top_spin.isEnabled()

    tool._updating_controls = True
    try:
        top_spin.setValue(0.92)
        _activate_combo_text(engine_combo, "compressed")
    finally:
        tool._updating_controls = False
    assert _method_operations(tool, FigureMethodFamily.FIGURE, "subplots_adjust") == ()
    assert tool.tool_status.setup.layout == "none"
    engine_combo.setCurrentText("none")

    top_spin.setValue(0.91)

    adjust_ops = _method_operations(tool, FigureMethodFamily.FIGURE, "subplots_adjust")
    assert len(adjust_ops) == 1
    assert adjust_ops[0].enabled
    assert adjust_ops[0].method_kwargs["top"] == pytest.approx(0.91)

    bottom_spin.setValue(0.99)
    adjust_ops = _method_operations(tool, FigureMethodFamily.FIGURE, "subplots_adjust")
    assert adjust_ops[0].method_kwargs["bottom"] < adjust_ops[0].method_kwargs["top"]

    _activate_combo_text(engine_combo, "compressed")

    engine_ops = _method_operations(
        tool, FigureMethodFamily.FIGURE, "set_layout_engine"
    )
    adjust_ops = _method_operations(tool, FigureMethodFamily.FIGURE, "subplots_adjust")
    assert engine_ops == ()
    assert tool.tool_status.setup.layout == "compressed"
    assert len(adjust_ops) == 1
    assert not adjust_ops[0].enabled
    assert not top_spin.isEnabled()

    _activate_combo_text(engine_combo, "none")

    engine_ops = _method_operations(
        tool, FigureMethodFamily.FIGURE, "set_layout_engine"
    )
    adjust_ops = _method_operations(tool, FigureMethodFamily.FIGURE, "subplots_adjust")
    assert engine_ops == ()
    assert tool.tool_status.setup.layout == "none"
    assert adjust_ops[0].enabled
    assert top_spin.isEnabled()


def test_figure_composer_subplots_adjust_pairs_stay_valid(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.2, "right": 0.8, "bottom": 0.2, "top": 0.8},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    left_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustLeftEdit"
    )
    right_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustRightEdit"
    )
    assert left_spin is not None
    assert right_spin is not None

    left_spin.setValue(0.95)

    operation = tool.tool_status.operations[0]
    assert operation.method_kwargs["left"] < operation.method_kwargs["right"]

    invalid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.95, "right": 0.2, "bottom": 0.9, "top": 0.1},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(invalid_tool)
    fig = invalid_tool.figure
    figurecomposer_rendering._render_into_figure(invalid_tool, fig, sync_visible=False)
    assert fig.subplotpars.left < fig.subplotpars.right
    assert fig.subplotpars.bottom < fig.subplotpars.top

    namespace = {"data": data}
    exec(invalid_tool.generated_code(), namespace)  # noqa: S102
    generated_fig = namespace["fig"]
    assert generated_fig.subplotpars.left < generated_fig.subplotpars.right
    assert generated_fig.subplotpars.bottom < generated_fig.subplotpars.top

    for partial_kwargs in (
        {"left": 0.95, "bottom": 0.95},
        {"right": 0.05, "top": 0.05},
    ):
        partial_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(layout="none"),
                sources=(FigureSourceState(name="data", label="data"),),
                operations=(
                    FigureOperationState.method(
                        family=FigureMethodFamily.FIGURE,
                        name="subplots_adjust",
                        kwargs=partial_kwargs,
                    ),
                ),
                primary_source="data",
            ),
        )
        qtbot.addWidget(partial_tool)
        fig = partial_tool.figure
        figurecomposer_rendering._render_into_figure(
            partial_tool, fig, sync_visible=False
        )
        assert fig.subplotpars.left < fig.subplotpars.right
        assert fig.subplotpars.bottom < fig.subplotpars.top

        namespace = {"data": data}
        exec(partial_tool.generated_code(), namespace)  # noqa: S102
        generated_fig = namespace["fig"]
        assert generated_fig.subplotpars.left < generated_fig.subplotpars.right
        assert generated_fig.subplotpars.bottom < generated_fig.subplotpars.top


def test_figure_composer_toolbar_subplot_dialog_reuses_live_dialog(qtbot) -> None:
    tool = FigureComposerTool(
        _figure_composer_image_source("data"),
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._show_subplot_adjust_dialog()
    first_dialog = tool._subplot_adjust_dialog
    assert isinstance(first_dialog, QtWidgets.QDialog)

    tool._show_subplot_adjust_dialog()

    assert tool._subplot_adjust_dialog is first_dialog


def test_figure_composer_toolbar_axis_state_helpers(qtbot) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    form_layout = QtWidgets.QFormLayout(parent)
    edit = QtWidgets.QLineEdit(parent)
    check = QtWidgets.QCheckBox(parent)

    figurecomposer_toolbar_dialogs._add_axis_compound_form_row(
        form_layout,
        "Compound",
        "compoundRow",
        (
            ("edit", edit, "edit-tip"),
            ("check", check, "check-tip"),
        ),
        "row-tip",
    )
    row_widget = parent.findChild(QtWidgets.QWidget, "compoundRow")
    assert row_widget is not None
    assert form_layout.labelForField(row_widget) is not None
    assert edit.toolTip() == "edit-tip"
    assert check.toolTip() == "check-tip"

    line_edit = QtWidgets.QLineEdit(parent)
    unavailable = figurecomposer_toolbar_dialogs._AxisValueState()
    mixed = figurecomposer_toolbar_dialogs._AxisValueState(
        mixed=True,
        available=True,
    )
    available = figurecomposer_toolbar_dialogs._AxisValueState(
        value="linear",
        available=True,
    )

    figurecomposer_toolbar_dialogs._apply_axis_line_edit_state(line_edit, unavailable)
    assert not line_edit.isEnabled()

    figurecomposer_toolbar_dialogs._apply_axis_line_edit_state(line_edit, mixed)
    assert line_edit.isEnabled()
    assert line_edit.placeholderText() == _editor_controls.MIXED_VALUES_TEXT
    assert figurecomposer_toolbar_dialogs._line_edit_mixed_unchanged(line_edit)
    line_edit.setText("changed")
    line_edit.setModified(True)
    assert not figurecomposer_toolbar_dialogs._line_edit_mixed_unchanged(line_edit)

    figurecomposer_toolbar_dialogs._apply_axis_line_edit_state(line_edit, available)
    assert line_edit.text() == "linear"

    combo = QtWidgets.QComboBox(parent)
    combo.addItems(("linear", "log"))
    figurecomposer_toolbar_dialogs._apply_axis_combo_state(combo, mixed)
    assert combo.currentData() is _editor_controls.MIXED_VALUE
    figurecomposer_toolbar_dialogs._apply_axis_combo_state(combo, available)
    assert combo.currentText() == "linear"
    assert all(
        combo.itemData(index) is not _editor_controls.MIXED_VALUE
        for index in range(combo.count())
    )
    figurecomposer_toolbar_dialogs._apply_axis_combo_state(combo, unavailable)
    assert not combo.isEnabled()

    figurecomposer_toolbar_dialogs._apply_axis_check_state(check, mixed)
    assert check.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    figurecomposer_toolbar_dialogs._apply_axis_check_state(
        check,
        figurecomposer_toolbar_dialogs._AxisValueState(
            value=True,
            available=True,
        ),
    )
    assert check.isChecked()
    figurecomposer_toolbar_dialogs._apply_axis_check_state(check, unavailable)
    assert not check.isEnabled()

    assert figurecomposer_toolbar_dialogs._float_pair_text((1, 2)) == "1, 2"
    assert figurecomposer_toolbar_dialogs._float_pair_text((1,)) == ""
    assert figurecomposer_toolbar_dialogs._float_pair_from_text("1, 2") == (
        1.0,
        2.0,
    )
    assert figurecomposer_toolbar_dialogs._float_pair_from_text("1") is None
    assert figurecomposer_text._limit_pair_from_text("1, None") == (1.0, None)
    assert figurecomposer_toolbar_dialogs._aspect_text(2.0) == "2"
    assert figurecomposer_toolbar_dialogs._aspect_value("equal") == "equal"
    assert figurecomposer_toolbar_dialogs._aspect_value("2.5") == 2.5

    class DummyAxis:
        def __init__(self, value: str | None) -> None:
            self.value = value

    same_state = figurecomposer_toolbar_dialogs._axis_value_state(
        (DummyAxis("same"), DummyAxis("same")),
        lambda axis: axis.value,
    )
    assert same_state.value == "same"
    assert same_state.available
    mixed_state = figurecomposer_toolbar_dialogs._axis_value_state(
        (DummyAxis("left"), DummyAxis("right")),
        lambda axis: axis.value,
    )
    assert mixed_state.mixed
    none_state = figurecomposer_toolbar_dialogs._axis_value_state(
        (DummyAxis(None),),
        lambda axis: axis.value,
    )
    assert none_state.mixed

    assert figurecomposer_toolbar_dialogs._merge_grid_states(()) is None
    assert figurecomposer_toolbar_dialogs._merge_grid_states((True, True)) is True
    assert figurecomposer_toolbar_dialogs._merge_grid_states((True, False)) is None


def test_figure_composer_toolbar_line_style_widget_updates_all_controls(qtbot) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(
        update={
            "line_colors": ("tab:blue",),
            "line_kw": {
                "lw": 1.0,
                "mfc": "white",
                "custom": 3,
            },
        }
    )
    widget = figurecomposer_toolbar_dialogs._LineOperationStyleWidget(tool, operation)
    qtbot.addWidget(widget)
    updates: list[FigureOperationState] = []
    widget.sigOperationChanged.connect(updates.append)

    widget.colors_widget.main_edit.setText("tab:red, tab:green")
    widget.colors_widget.main_edit.setModified(True)
    widget.colors_widget.main_edit.editingFinished.emit()
    widget.marker_size_spin.setValue(4.5)
    widget.marker_face_edit.setText("none")
    widget.marker_face_edit.editingFinished.emit()
    widget.marker_edge_edit.setText("tab:orange")
    widget.marker_edge_edit.editingFinished.emit()
    widget.extra_kwargs_edit.setText("alpha=0.5, dash_capstyle='round'")
    widget.extra_kwargs_edit.editingFinished.emit()

    assert updates
    updated = updates[-1]
    assert updated.line_colors == ("tab:red", "tab:green")
    assert updated.line_kw["markersize"] == 4.5
    assert updated.line_kw["markerfacecolor"] == "none"
    assert updated.line_kw["markeredgecolor"] == "tab:orange"
    assert updated.line_kw["alpha"] == 0.5
    assert updated.line_kw["dash_capstyle"] == "round"
    assert "mfc" not in updated.line_kw
    assert "custom" not in updated.line_kw


def test_figure_composer_toolbar_operation_helpers_update_recipe(qtbot) -> None:
    data = _figure_composer_image_source("data")
    method = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_title",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        args=("old",),
    )
    slices = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(-0.5, 0.5),
    ).model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="viridis",
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(method, slices),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    figure_window = tool._figure_window
    tool._figure_window = None
    assert figurecomposer_toolbar_dialogs._dialog_parent(tool) is tool
    tool._figure_window = figure_window
    tool.show_figure_window(activate=False)

    updated_index = figurecomposer_toolbar_dialogs._upsert_method_operation(
        tool,
        FigureMethodFamily.AXES,
        "set_title",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        args=("updated",),
    )
    assert updated_index == 0
    assert tool.tool_status.operations[0].method_args == ("updated",)

    new_index = figurecomposer_toolbar_dialogs._upsert_method_operation(
        tool,
        FigureMethodFamily.AXES,
        "set_title",
        axes=FigureAxesSelectionState(axes=((0, 1),)),
        args=("right",),
        enabled=False,
    )
    assert new_index == 2
    assert not tool.tool_status.operations[new_index].enabled

    tool._reset_history_stack()
    figurecomposer_toolbar_dialogs._set_method_operation_enabled(
        tool,
        FigureMethodFamily.AXES,
        "set_title",
        axes=FigureAxesSelectionState(axes=((0, 1),)),
        enabled=True,
    )
    assert tool.tool_status.operations[new_index].enabled
    assert tool.undoable

    tool.undo()
    assert not tool.tool_status.operations[new_index].enabled
    assert tool.redoable

    tool.redo()
    assert tool.tool_status.operations[new_index].enabled

    slices_id = slices.operation_id
    panel_keys = figurecomposer_toolbar_dialogs._selected_plot_slices_panel_keys(
        tool,
        tool.tool_status.operations[1],
        {id(tool.figure.axes[0])},
    )
    panel_update = figurecomposer_toolbar_dialogs._update_plot_slices_panel_styles(
        tool,
        slices_id,
        panel_keys,
        (
            FigurePlotSlicesPanelStyleState(
                map_index=0,
                slice_index=0,
                cmap="magma",
            ),
        ),
    )
    assert panel_update is not None
    updated_slices = tool.tool_status.operations[1]
    assert updated_slices.panel_styles == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="magma",
        ),
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=1,
            cmap="viridis",
        ),
    )

    stale_update = figurecomposer_toolbar_dialogs._update_plot_slices_panel_styles(
        tool,
        slices_id,
        panel_keys,
        (
            FigurePlotSlicesPanelStyleState(
                map_index=0,
                slice_index=0,
                cmap="plasma",
            ),
        ),
        expected_operation=slices,
    )
    assert stale_update is None
    assert tool.tool_status.operations[1].panel_styles == updated_slices.panel_styles

    figurecomposer_toolbar_dialogs._update_plot_slices_panel_styles(
        tool,
        tool.tool_status.operations[0].operation_id,
        panel_keys,
        (),
    )
    assert tool.tool_status.operations[0].kind == FigureOperationKind.METHOD

    figurecomposer_toolbar_dialogs._replace_operation_by_id(
        tool,
        "missing",
        tool.tool_status.operations[0],
    )
    figurecomposer_toolbar_dialogs._replace_recipe_operation(
        tool,
        -1,
        tool.tool_status.operations[0],
    )


def test_figure_composer_toolbar_selector_helpers_default_to_available_axes(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    selector = figurecomposer_toolbar_dialogs._selector_widget(tool, tool)
    assert isinstance(selector, figurecomposer_widgets._AxesSelectorWidget)
    assert selector.selected_axes() == ((0, 0),)
    selector.set_selected_axes((), emit=False)
    assert figurecomposer_toolbar_dialogs._selector_selection(tool, selector).axes == (
        (0, 0),
    )

    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        nrows=1,
                        ncols=2,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="main",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                            FigureGridSpecAxesState(
                                axes_id="side",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=1,
                                    col_stop=2,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_selector = figurecomposer_toolbar_dialogs._selector_widget(
        grid_tool,
        grid_tool,
    )
    assert isinstance(grid_selector, figurecomposer_widgets._GridSpecViewWidget)
    assert grid_selector.selected_axes_ids() == ("main",)
    grid_selector.set_selected_axes_ids(())
    assert figurecomposer_toolbar_dialogs._selector_selection(
        grid_tool,
        grid_selector,
    ).axes_ids == ("main",)


def test_figure_composer_toolbar_misc_helper_edge_paths(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="data",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    args=("Peak",),
                ),
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(update={"enabled": False}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    no_axes = FigureAxesSelectionState(axes=((9, 9),))
    assert figurecomposer_toolbar_dialogs._curve_style_targets(tool, no_axes) == []
    assert figurecomposer_toolbar_dialogs._image_style_targets(tool, no_axes) == []
    assert (
        figurecomposer_toolbar_dialogs._selected_plot_slices_panel_keys(
            tool,
            tool.tool_status.operations[2],
            {id(tool.figure.axes[0])},
        )
        == ()
    )
    assert (
        figurecomposer_toolbar_dialogs._operation_by_id(tool, "missing-operation")
        is None
    )
    assert figurecomposer_toolbar_dialogs._dialog_parent(tool) is tool.figure_window
    curve_targets = figurecomposer_toolbar_dialogs._curve_style_targets(
        tool,
        FigureAxesSelectionState(axes=((0, 0),)),
    )
    assert len(curve_targets) == 1

    stale_dialog = QtWidgets.QDialog(tool)
    stale_dialog.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(
        stale_dialog,
        int(QtCore.QEvent.Type.DeferredDelete.value),
    )
    tool._axes_customize_dialog = stale_dialog
    assert (
        figurecomposer_toolbar_dialogs._show_existing_dialog(
            tool,
            "_axes_customize_dialog",
        )
        is False
    )

    live_dialog = QtWidgets.QDialog(tool)
    figurecomposer_toolbar_dialogs._show_toolbar_dialog(
        tool,
        "_axes_customize_dialog",
        live_dialog,
    )
    assert tool._axes_customize_dialog is live_dialog
    live_dialog.close()
    live_dialog.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(
        live_dialog,
        int(QtCore.QEvent.Type.DeferredDelete.value),
    )
    assert tool._axes_customize_dialog is None

    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    layout = QtWidgets.QVBoxLayout(parent)
    child = QtWidgets.QWidget(parent)
    layout.addWidget(child)
    nested = QtWidgets.QHBoxLayout()
    nested_child = QtWidgets.QWidget(parent)
    nested.addWidget(nested_child)
    layout.addLayout(nested)
    figurecomposer_toolbar_dialogs._clear_layout(layout)
    assert layout.count() == 0

    tool.figure.clear()
    assert figurecomposer_toolbar_dialogs._layout_axes(tool) is None
    assert (
        figurecomposer_toolbar_dialogs._axes_for_selection(
            tool,
            FigureAxesSelectionState(axes=((0, 0),)),
        )
        == ()
    )

    assert not figurecomposer_toolbar_dialogs._axis_value_state(
        (),
        lambda axis: axis,
    ).available
    assert figurecomposer_toolbar_dialogs._aspect_text(("custom",)) == "('custom',)"

    multi_panel = FigureOperationState.plot_slices(
        label="multi_panel",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(-0.5, 0.5),
    )
    multi_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(multi_panel,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(multi_tool)
    multi_tool.show_figure_window(activate=False)
    assert (
        len(
            figurecomposer_toolbar_dialogs._selected_plot_slices_panel_keys(
                multi_tool,
                multi_panel,
                {id(multi_tool.figure.axes[0])},
            )
        )
        == 2
    )

    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        nrows=1,
                        ncols=1,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="main",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_tool.figure.clear()
    assert figurecomposer_toolbar_dialogs._layout_axes(grid_tool) is None

    fig, ax = plt.subplots()
    try:
        ax.grid(True, which="minor")
        assert figurecomposer_toolbar_dialogs._axis_direction_grid_visible(
            ax.xaxis,
            "minor",
        ) in {True, False}
    finally:
        plt.close(fig)


def test_figure_composer_provenance_code_detection_edges() -> None:
    assert figurecomposer_provenance._code_assigns_name("fig, axs = make()", "fig")
    assert figurecomposer_provenance._code_assigns_name("fig: object = make()", "fig")
    assert figurecomposer_provenance._code_assigns_name("fig += update()", "fig")
    assert not figurecomposer_provenance._code_assigns_name("axs = make()", "fig")


def test_figure_composer_provenance_build_code_handles_invalid_recipes() -> None:
    class FakeTool:
        def __init__(self, code: str | None = None, exc: Exception | None = None):
            self._code = code
            self._exc = exc

        def generated_code(self) -> str:
            if self._exc is not None:
                raise self._exc
            if self._code is None:
                raise RuntimeError
            return self._code

    assert (
        figurecomposer_provenance._figure_build_code(
            typing.cast(
                "FigureComposerTool",
                FakeTool(exc=RuntimeError("invalid recipe")),
            )
        )
        is None
    )
    assert (
        figurecomposer_provenance._figure_build_code(
            typing.cast("FigureComposerTool", FakeTool("if"))
        )
        is None
    )
    assert (
        figurecomposer_provenance._figure_build_code(
            typing.cast("FigureComposerTool", FakeTool("axs = object()"))
        )
        is None
    )

    operation = figurecomposer_provenance._figure_build_operation(
        typing.cast("FigureComposerTool", FakeTool("fig = object()"))
    )
    assert operation.copyable
    assert operation.code == "fig = object()"


def test_figure_composer_toolbar_axes_dialog_updates_recipe(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    tool._show_axes_customize_dialog()
    assert tool._axes_customize_dialog is dialog
    title_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesTitleEdit"
    )
    xlim_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesXLimEdit"
    )
    xlabel_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesXLabelEdit"
    )
    ylabel_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesYLabelEdit"
    )
    aspect_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesAspectEdit"
    )
    xscale_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesXScaleCombo"
    )
    grid_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerToolbarAxesGridCheck"
    )
    grid_axis_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesGridAxisCombo"
    )
    tick_editor = dialog.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerToolbarAxesTickParamsEditor",
    )
    assert title_edit is not None
    assert xlim_edit is not None
    assert xlabel_edit is not None
    assert ylabel_edit is not None
    assert aspect_edit is not None
    assert xscale_combo is not None
    assert grid_check is not None
    assert grid_axis_combo is not None
    assert tick_editor is not None
    labels_row = dialog.findChild(
        QtWidgets.QWidget, "figureComposerToolbarAxesLabelsRow"
    )
    limits_row = dialog.findChild(
        QtWidgets.QWidget, "figureComposerToolbarAxesLimitsRow"
    )
    scales_row = dialog.findChild(
        QtWidgets.QWidget, "figureComposerToolbarAxesScalesRow"
    )
    grid_row = dialog.findChild(QtWidgets.QWidget, "figureComposerToolbarAxesGridRow")
    assert labels_row is not None
    assert limits_row is not None
    assert scales_row is not None
    assert grid_row is not None
    assert (
        labels_row.findChild(QtWidgets.QLineEdit, xlabel_edit.objectName()) is not None
    )
    assert (
        labels_row.findChild(QtWidgets.QLineEdit, ylabel_edit.objectName()) is not None
    )
    assert limits_row.findChild(QtWidgets.QLineEdit, xlim_edit.objectName()) is not None
    assert (
        scales_row.findChild(QtWidgets.QComboBox, xscale_combo.objectName()) is not None
    )
    assert grid_row.findChild(QtWidgets.QCheckBox, grid_check.objectName()) is not None
    grid_row_layout = grid_row.layout()
    assert grid_row_layout is not None
    grid_sublabels: list[str] = []
    for i in range(grid_row_layout.count()):
        widget = grid_row_layout.itemAt(i).widget()
        if isinstance(widget, QtWidgets.QLabel):
            grid_sublabels.append(widget.text())
    assert "Visible" not in grid_sublabels
    assert grid_check.text() == "Show"
    assert (
        dialog.findChild(QtWidgets.QComboBox, "figureComposerToolbarAxesStyleStepCombo")
        is None
    )
    assert (
        dialog.findChild(
            QtWidgets.QPushButton, "figureComposerToolbarAxesStyleStepButton"
        )
        is None
    )

    title_edit.setText("Peak")
    title_edit.setModified(True)
    title_edit.editingFinished.emit()
    title_edit.editingFinished.emit()
    title_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_title")
    assert len(title_ops) == 1
    assert title_ops[0].axes.axes == ((0, 0),)
    assert title_ops[0].method_args == ("Peak",)

    xlabel_edit.setText("Energy")
    xlabel_edit.setModified(True)
    xlabel_edit.editingFinished.emit()
    ylabel_edit.setText("Intensity")
    ylabel_edit.setModified(True)
    ylabel_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_xlabel")[
        0
    ].method_args == ("Energy",)
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_ylabel")[
        0
    ].method_args == ("Intensity",)

    xlim_edit.setText("-0.5, 0.5")
    xlim_edit.setModified(True)
    xlim_edit.editingFinished.emit()
    xlim_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_xlim")
    assert len(xlim_ops) == 1
    assert xlim_ops[0].method_args == (-0.5, 0.5)
    xlim_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_xlim") == xlim_ops

    ylim_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesYLimEdit"
    )
    assert ylim_edit is not None
    ylim_edit.setText("-1, 1")
    ylim_edit.setModified(True)
    ylim_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_ylim")[
        0
    ].method_args == (-1.0, 1.0)

    aspect_edit.setText("equal")
    aspect_edit.setModified(True)
    aspect_edit.editingFinished.emit()
    aspect_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_aspect")
    assert len(aspect_ops) == 1
    assert aspect_ops[0].method_args == ("equal",)
    aspect_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_aspect") == (
        aspect_ops
    )

    xlim_edit.setText("invalid")
    xlim_edit.setModified(True)
    xlim_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_xlim") == xlim_ops

    aspect_edit.setText("1, 2")
    aspect_edit.setModified(True)
    aspect_edit.editingFinished.emit()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_aspect") == (
        aspect_ops
    )

    _activate_combo_text(xscale_combo, "linear")
    xscale_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_xscale")
    assert len(xscale_ops) == 1
    assert xscale_ops[0].method_args == ("linear",)

    grid_check.setChecked(True)
    grid_ops = _method_operations(tool, FigureMethodFamily.AXES, "grid")
    assert len(grid_ops) == 1
    assert grid_ops[0].method_args == (True,)
    assert grid_ops[0].method_kwargs == {"which": "major", "axis": "both"}

    _activate_combo_text(grid_axis_combo, "x")
    grid_ops = _method_operations(tool, FigureMethodFamily.AXES, "grid")
    assert len(grid_ops) == 1
    assert grid_ops[0].method_kwargs == {"which": "major", "axis": "x"}

    _click_tick_params_segment(
        tick_editor,
        "figureComposerToolbarAxesTickParamsAxisCombo",
        "y",
    )
    _finish_tick_params_edit(
        tick_editor,
        "figureComposerToolbarAxesTickParamsLengthEdit",
        "5",
    )
    tick_ops = _method_operations(tool, FigureMethodFamily.AXES, "tick_params")
    assert len(tick_ops) == 1
    assert tick_ops[0].axes.axes == ((0, 0),)
    assert tick_ops[0].method_kwargs == {"axis": "y", "length": 5.0}

    selector = dialog.findChild(figurecomposer_widgets._AxesSelectorWidget)
    assert selector is not None
    selector.set_selected_axes(((0, 1),), emit=True)


def test_figure_composer_toolbar_axes_dialog_shows_mixed_axis_selection(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.axes_selector.set_selected_axes(((0, 0), (0, 1)), emit=False)
    tool.show_figure_window(activate=False)
    left_axis, right_axis = tool.figure.axes[:2]
    left_axis.set_title("Left")
    right_axis.set_title("Right")
    left_axis.set_xlim(0.1, 1.0)
    right_axis.set_xlim(0.1, 10.0)
    right_axis.set_xscale("log")
    left_axis.grid(True)
    right_axis.grid(False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    title_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesTitleEdit"
    )
    xlim_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesXLimEdit"
    )
    xscale_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesXScaleCombo"
    )
    grid_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerToolbarAxesGridCheck"
    )
    assert title_edit is not None
    assert xlim_edit is not None
    assert xscale_combo is not None
    assert grid_check is not None
    assert title_edit.text() == ""
    assert title_edit.placeholderText() == _editor_controls.MIXED_VALUES_TEXT
    assert xlim_edit.text() == ""
    assert xlim_edit.placeholderText() == _editor_controls.MIXED_VALUES_TEXT
    assert xscale_combo.currentText() == _editor_controls.MIXED_VALUES_TEXT
    assert grid_check.checkState() == QtCore.Qt.CheckState.PartiallyChecked

    xscale_combo.activated.emit(xscale_combo.currentIndex())
    yscale_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesYScaleCombo"
    )
    assert yscale_combo is not None
    figurecomposer_toolbar_dialogs._apply_axis_combo_state(
        yscale_combo,
        figurecomposer_toolbar_dialogs._AxisValueState(mixed=True, available=True),
    )
    yscale_combo.activated.emit(yscale_combo.currentIndex())
    grid_check.stateChanged.emit(int(QtCore.Qt.CheckState.PartiallyChecked.value))
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_xscale") == ()
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_yscale") == ()
    assert _method_operations(tool, FigureMethodFamily.AXES, "grid") == ()

    title_edit.setText("Shared")
    title_edit.setModified(True)
    title_edit.editingFinished.emit()

    title_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_title")
    assert len(title_ops) == 1
    assert title_ops[0].axes.axes == ((0, 0), (0, 1))
    assert title_ops[0].method_args == ("Shared",)

    _activate_combo_text(xscale_combo, "linear")
    xscale_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_xscale")
    assert len(xscale_ops) == 1
    assert xscale_ops[0].axes.axes == ((0, 0), (0, 1))
    assert xscale_ops[0].method_args == ("linear",)

    grid_check.setCheckState(QtCore.Qt.CheckState.Checked)
    grid_ops = _method_operations(tool, FigureMethodFamily.AXES, "grid")
    assert len(grid_ops) == 1
    assert grid_ops[0].axes.axes == ((0, 0), (0, 1))
    assert grid_ops[0].method_args == (True,)


def test_figure_composer_toolbar_axes_dialog_disables_unavailable_axes(
    qtbot,
    monkeypatch,
) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    monkeypatch.setattr(
        figurecomposer_toolbar_dialogs,
        "_axes_for_selection",
        lambda _tool, _selection: (),
    )

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    title_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesTitleEdit"
    )
    xscale_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesXScaleCombo"
    )
    grid_axis_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarAxesGridAxisCombo"
    )
    grid_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerToolbarAxesGridCheck"
    )
    tick_editor = dialog.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerToolbarAxesTickParamsEditor",
    )
    assert title_edit is not None
    assert xscale_combo is not None
    assert grid_axis_combo is not None
    assert grid_check is not None
    assert tick_editor is not None
    assert not title_edit.isEnabled()
    assert not xscale_combo.isEnabled()
    assert not grid_axis_combo.isEnabled()
    assert not grid_check.isEnabled()
    assert not tick_editor.isEnabled()


def test_figure_composer_toolbar_axes_dialog_handles_stale_style_targets(
    qtbot,
    monkeypatch,
) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="data",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    monkeypatch.setattr(
        figurecomposer_toolbar_dialogs,
        "_operation_by_id",
        lambda _tool, _operation_id: None,
    )

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    placeholder = dialog.findChild(
        QtWidgets.QLabel, "figureComposerToolbarStylePlaceholder"
    )
    assert placeholder is not None


def test_figure_composer_toolbar_axes_dialog_updates_curve_style(qtbot) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="data",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveTargetCombo"
    )
    colors_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarCurveColorsEdit"
    )
    style_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveLineStyleCombo"
    )
    width_spin = dialog.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerToolbarCurveLineWidthSpin"
    )
    marker_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveMarkerCombo"
    )
    assert target_combo is not None
    assert colors_edit is not None
    assert style_combo is not None
    assert width_spin is not None
    assert marker_combo is not None
    assert target_combo.count() == 1

    colors_edit.setText("tab:red, tab:blue")
    colors_edit.setModified(True)
    colors_edit.editingFinished.emit()
    _activate_combo_text(style_combo, "--")
    width_spin.setValue(2.5)
    _activate_combo_text(marker_combo, "o")

    operation = tool.tool_status.operations[0]
    assert operation.line_colors == ("tab:red", "tab:blue")
    assert operation.line_kw["linestyle"] == "--"
    assert operation.line_kw["linewidth"] == 2.5
    assert operation.line_kw["marker"] == "o"


def test_figure_composer_toolbar_axes_dialog_updates_curve_color_mode(qtbot) -> None:
    data = _figure_composer_line_slice_source("data")
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_iter_dim": "eV"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    mode_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveColorModeCombo"
    )
    coord_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveColorCoordCombo"
    )
    cmap_combo = dialog.findChild(
        erlab.interactive.colors.ColorMapComboBox,
        "figureComposerToolbarCurveColorCmapCombo",
    )
    reverse_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerToolbarCurveColorCmapReverseCheck"
    )
    trim_lower_spin = dialog.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerToolbarCurveColorCmapTrimLowerSpin"
    )
    trim_upper_spin = dialog.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerToolbarCurveColorCmapTrimUpperSpin"
    )
    colors_widget = dialog.findChild(
        QtWidgets.QWidget, "figureComposerToolbarCurveColorsWidget"
    )
    assert mode_combo is not None
    assert coord_combo is not None
    assert cmap_combo is not None
    assert reverse_check is not None
    assert trim_lower_spin is not None
    assert trim_upper_spin is not None
    assert colors_widget is not None
    assert mode_combo.findData("coordinate") >= 0
    assert coord_combo.findData("eV") >= 0
    assert not colors_widget.isHidden()

    _activate_combo_index(mode_combo, mode_combo.findData("coordinate"))
    cmap_combo.setCurrentText("plasma")
    cmap_combo.activated.emit(cmap_combo.currentIndex())
    reverse_check.setCheckState(QtCore.Qt.CheckState.Checked)
    trim_lower_spin.setValue(0.1)
    trim_upper_spin.setValue(0.2)

    operation = tool.tool_status.operations[0]
    assert operation.line_color_mode == "coordinate"
    assert operation.line_color_coord == "eV"
    assert operation.line_color_cmap == "plasma"
    assert operation.line_color_cmap_reverse
    assert operation.line_color_cmap_trim_lower == 0.1
    assert operation.line_color_cmap_trim_upper == 0.2
    assert colors_widget.isHidden()


def test_figure_composer_toolbar_axes_dialog_updates_plot_slices_curve_style(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(-0.5, 0.5),
                ).model_copy(update={"slice_kwargs": {"beta": 0.0}}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveTargetCombo"
    )
    panel_list = dialog.findChild(
        QtWidgets.QListWidget, "figureComposerPlotSlicesPanelLineStyleList"
    )
    color_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerPanelLineColorEdit"
    )
    style_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerPanelLineStyleCombo"
    )
    width_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerPanelLineWidthEdit"
    )
    marker_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerPanelLineMarkerCombo"
    )
    marker_size_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerPanelLineMarkerSizeEdit"
    )
    line_kwargs_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerPanelLineKwEdit"
    )
    mode_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveColorModeCombo"
    )
    coord_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarCurveColorCoordCombo"
    )
    cmap_combo = dialog.findChild(
        erlab.interactive.colors.ColorMapComboBox,
        "figureComposerToolbarCurveColorCmapCombo",
    )
    assert target_combo is not None
    assert panel_list is not None
    assert color_edit is not None
    assert style_combo is not None
    assert width_edit is not None
    assert marker_combo is not None
    assert marker_size_edit is not None
    assert line_kwargs_edit is not None
    assert mode_combo is not None
    assert coord_combo is not None
    assert cmap_combo is not None
    assert target_combo.count() == 1
    assert panel_list.count() == 2
    assert mode_combo.findData("coordinate") >= 0
    assert coord_combo.findData("eV") >= 0
    main_panel_styles_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesPanelStylesCheck"
    )
    assert main_panel_styles_check is not None
    assert main_panel_styles_check.checkState() == QtCore.Qt.CheckState.Unchecked

    _activate_combo_index(mode_combo, mode_combo.findData("coordinate"))
    cmap_combo.setCurrentText("plasma")
    cmap_combo.activated.emit(cmap_combo.currentIndex())

    panel_list.clearSelection()
    first_panel = panel_list.item(0)
    assert first_panel is not None
    first_panel.setSelected(True)
    color_edit.setText("tab:red")
    color_edit.editingFinished.emit()
    _activate_combo_text(style_combo, "--")
    width_edit.setText("2.5")
    width_edit.editingFinished.emit()
    _activate_combo_text(marker_combo, "o")
    marker_size_edit.setText("4")
    marker_size_edit.editingFinished.emit()
    line_kwargs_edit.setText("alpha=0.5")
    line_kwargs_edit.setModified(True)
    line_kwargs_edit.editingFinished.emit()

    operation = tool.tool_status.operations[0]
    assert operation.line_color_mode == "coordinate"
    assert operation.line_color_coord == "eV"
    assert operation.line_color_cmap == "plasma"
    assert operation.panel_styles_enabled
    assert operation.panel_styles == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            line_kw={
                "color": "tab:red",
                "linestyle": "--",
                "linewidth": 2.5,
                "marker": "o",
                "markersize": 4.0,
                "alpha": 0.5,
            },
        ),
    )
    main_panel_styles_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesPanelStylesCheck"
    )
    assert main_panel_styles_check is not None
    assert main_panel_styles_check.checkState() == QtCore.Qt.CheckState.Checked


def test_figure_composer_toolbar_axes_dialog_updates_image_style(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(-0.5, 0.5),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    selector = dialog.findChild(figurecomposer_widgets._AxesSelectorWidget)
    assert selector is not None
    selector.set_selected_axes(((0, 0), (0, 1)), emit=True)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarImageTargetCombo"
    )
    panel_list = dialog.findChild(
        QtWidgets.QListWidget, "figureComposerPlotSlicesPanelStyleList"
    )
    cmap_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelCmapOverrideCheck"
    )
    cmap_combo = dialog.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerPanelCmapCombo"
    )
    norm_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelNormOverrideCheck"
    )
    norm_combo = dialog.findChild(QtWidgets.QComboBox, "figureComposerPanelNormCombo")
    assert target_combo is not None
    assert panel_list is not None
    assert cmap_check is not None
    assert cmap_combo is not None
    assert norm_check is not None
    assert norm_combo is not None
    assert target_combo.count() == 1
    assert panel_list.count() == 2

    panel_list.clearSelection()
    first_panel = panel_list.item(0)
    assert first_panel is not None
    first_panel.setSelected(True)
    cmap_check.setCheckState(QtCore.Qt.CheckState.Checked)
    assert cmap_check.checkState() == QtCore.Qt.CheckState.Checked
    assert cmap_combo.isEnabled()
    cmap_combo.setCurrentText("magma")
    cmap_combo.activated.emit(cmap_combo.currentIndex())
    norm_check.setCheckState(QtCore.Qt.CheckState.Checked)
    assert norm_check.checkState() == QtCore.Qt.CheckState.Checked
    assert norm_combo.isEnabled()
    _activate_combo_text(norm_combo, "Normalize")

    operation = tool.tool_status.operations[0]
    assert operation.panel_styles_enabled
    assert operation.panel_styles == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="magma",
            norm_name="Normalize",
        ),
    )
    main_panel_styles_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesPanelStylesCheck"
    )
    assert main_panel_styles_check is not None
    assert main_panel_styles_check.checkState() == QtCore.Qt.CheckState.Checked


def test_figure_composer_toolbar_axes_dialog_updates_single_image_style_directly(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarImageTargetCombo"
    )
    cmap_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelCmapOverrideCheck"
    )
    norm_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelNormOverrideCheck"
    )
    cmap_combo = dialog.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerPanelCmapCombo"
    )
    cmap_reverse_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelCmapReverseCheck"
    )
    norm_combo = dialog.findChild(QtWidgets.QComboBox, "figureComposerPanelNormCombo")
    gamma_edit = dialog.findChild(QtWidgets.QLineEdit, "figureComposerPanelGammaEdit")
    vmin_edit = dialog.findChild(QtWidgets.QLineEdit, "figureComposerPanelVminEdit")
    vmax_edit = dialog.findChild(QtWidgets.QLineEdit, "figureComposerPanelVmaxEdit")
    assert target_combo is not None
    assert cmap_check is None
    assert norm_check is None
    assert cmap_combo is not None
    assert cmap_reverse_check is not None
    assert norm_combo is not None
    assert gamma_edit is not None
    assert vmin_edit is not None
    assert vmax_edit is not None
    assert target_combo.count() == 1

    cmap_combo.setCurrentText("magma")
    cmap_combo.activated.emit(cmap_combo.currentIndex())
    cmap_reverse_check.setCheckState(QtCore.Qt.CheckState.Checked)
    _activate_combo_text(norm_combo, "Normalize")
    vmin_edit.setText("-1")
    vmin_edit.editingFinished.emit()
    vmax_edit.setText("1")
    vmax_edit.editingFinished.emit()

    operation = tool.tool_status.operations[0]
    assert operation.cmap == "magma_r"
    assert operation.norm_name == "Normalize"
    assert operation.vmin == -1.0
    assert operation.vmax == 1.0
    assert operation.panel_styles == ()
    assert not operation.panel_styles_enabled
    assert not gamma_edit.isEnabled()
    assert vmin_edit.isEnabled()
    assert vmax_edit.isEnabled()


def test_figure_composer_toolbar_image_target_combo_elides_long_sources(qtbot) -> None:
    source_count = 6
    source_names = tuple(
        f"source_{index}_with_an_intentionally_long_alias_name"
        for index in range(source_count)
    )
    source_labels = tuple(
        f"Long data label {index} with additional descriptive text"
        for index in range(source_count)
    )
    source_data = {name: _figure_composer_image_source(name) for name in source_names}
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=source_names,
        axes=FigureAxesSelectionState(
            axes=tuple((0, index) for index in range(source_count))
        ),
    )
    tool = FigureComposerTool(
        source_data[source_names[0]],
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=source_count),
            sources=tuple(
                FigureSourceState(name=name, label=label)
                for name, label in zip(source_names, source_labels, strict=True)
            ),
            operations=(operation,),
            primary_source=source_names[0],
        ),
        source_data=source_data,
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    selector = dialog.findChild(figurecomposer_widgets._AxesSelectorWidget)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarImageTargetCombo"
    )
    assert selector is not None
    assert target_combo is not None
    selector.set_selected_axes(
        tuple((0, index) for index in range(source_count)), emit=True
    )

    assert target_combo.count() == 1
    visible_text = target_combo.itemText(0)
    assert "Image slices" in visible_text
    assert f"{source_count} panels" in visible_text
    for text in (*source_names, *source_labels):
        assert text not in visible_text

    tooltip = target_combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)
    assert isinstance(tooltip, str)
    assert all(label in tooltip for label in source_labels)
    assert target_combo.toolTip() == tooltip
    assert target_combo.view().textElideMode() == QtCore.Qt.TextElideMode.ElideMiddle
    assert (
        target_combo.sizeAdjustPolicy()
        == QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert (
        target_combo.sizeHint().width()
        < target_combo.fontMetrics().horizontalAdvance(tooltip)
    )
    custom_label = figurecomposer_toolbar_dialogs._image_style_target_label(
        1,
        operation.model_copy(update={"label": "Custom image step"}),
        (figurecomposer_plot_slices._PlotSlicesPanelKey(0, 1, "ignored"),),
    )
    assert custom_label == "Step 2: Custom image step: panel 1.2"


def test_figure_composer_toolbar_axes_dialog_keeps_cleared_image_styles_on_close(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(-0.5, 0.5),
                ).model_copy(
                    update={
                        "panel_styles_enabled": True,
                        "panel_styles": (
                            FigurePlotSlicesPanelStyleState(
                                map_index=0,
                                slice_index=0,
                                cmap="magma",
                            ),
                            FigurePlotSlicesPanelStyleState(
                                map_index=0,
                                slice_index=1,
                                cmap="plasma",
                            ),
                        ),
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    selector = dialog.findChild(figurecomposer_widgets._AxesSelectorWidget)
    assert selector is not None
    selector.set_selected_axes(((0, 0), (0, 1)), emit=True)

    panel_list = dialog.findChild(
        QtWidgets.QListWidget, "figureComposerPlotSlicesPanelStyleList"
    )
    cmap_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelCmapOverrideCheck"
    )
    assert panel_list is not None
    assert cmap_check is not None
    assert panel_list.count() == 2
    for row in range(panel_list.count()):
        item = panel_list.item(row)
        assert item is not None
        item.setSelected(True)

    cmap_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    operation = tool.tool_status.operations[0]
    assert not operation.panel_styles_enabled
    assert operation.panel_styles == ()

    dialog.close()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: tool._axes_customize_dialog is None, timeout=1000)

    operation = tool.tool_status.operations[0]
    assert not operation.panel_styles_enabled
    assert operation.panel_styles == ()


def test_figure_composer_toolbar_axes_dialog_uses_gridspec_selector(qtbot) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        nrows=1,
                        ncols=1,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="main",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    selector = dialog.findChild(figurecomposer_widgets._GridSpecViewWidget)
    title_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesTitleEdit"
    )
    assert selector is not None
    assert selector.selected_axes_ids() == ("main",)
    assert title_edit is not None

    title_edit.setText("GridSpec title")
    title_edit.setModified(True)
    title_edit.editingFinished.emit()

    title_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_title")
    assert len(title_ops) == 1
    assert title_ops[0].axes.axes_ids == ("main",)
    assert title_ops[0].method_args == ("GridSpec title",)


def test_figure_composer_layout_ratios_update_subplots_kwargs(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                nrows=2,
                ncols=3,
                width_ratios=(1.0, 2.0, 3.0),
                height_ratios=(2.0, 1.0),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert tool.width_ratios_edit.text() == "1, 2, 3"
    assert tool.height_ratios_edit.text() == "2, 1"

    tool.width_ratios_edit.setText("3, 2, 1")
    tool.height_ratios_edit.setText("4, 1")
    tool._setup_controls_changed()

    assert tool.tool_status.setup.width_ratios == (3.0, 2.0, 1.0)
    assert tool.tool_status.setup.height_ratios == (4.0, 1.0)
    setup_kwargs = figurecomposer_rendering._setup_kwargs(tool)
    assert setup_kwargs["width_ratios"] == (3.0, 2.0, 1.0)
    assert setup_kwargs["height_ratios"] == (4.0, 1.0)

    code = tool.generated_code()
    assert "width_ratios" in code
    assert "height_ratios" in code
    assert "gridspec_kw" not in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["axs"].shape == (2, 3)


def test_figure_composer_pipeline_codegen_executes(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    profile = xr.DataArray(
        np.arange(2.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(
                FigureSourceState(name="data", label="data"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="left",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.plot_slices(
                    label="right",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(1.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"line_x": "kx", "xlim": (0.25, 0.75)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "profile": profile},
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (2,))
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.itemData(0) is None
    assert profile_values_combo.itemData(0) is None
    assert profile_coordinate_combo.findData("kx") >= 0
    assert profile_values_combo.findData("kx") >= 0
    _activate_combo_index(
        profile_coordinate_combo, profile_coordinate_combo.findData("kx")
    )
    assert tool.tool_status.operations[2].line_x == "kx"
    _activate_combo_index(profile_coordinate_combo, 0)
    assert tool.tool_status.operations[2].line_x is None
    _activate_combo_index(profile_values_combo, profile_values_combo.findData("kx"))
    assert tool.tool_status.operations[2].line_y == "kx"
    selection_page = tool.step_editor_stack.currentWidget()
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_values_combo is not None
    _activate_combo_index(profile_values_combo, 0)
    assert tool.tool_status.operations[2].line_y is None
    selection_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.toolTip()
    assert profile_values_combo.toolTip()
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QCheckBox)
    )
    tool._select_step_section("view")
    view_page = tool.step_editor_stack.currentWidget()
    data_values_axis_combo = view_page.findChild(
        QtWidgets.QComboBox, "figureComposerDataValuesAxisCombo"
    )
    assert data_values_axis_combo is not None
    assert data_values_axis_combo.toolTip()
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QCheckBox)
    )

    _select_operation_rows(tool, (3,))
    assert tool.operation_list.item(3).text() == "eplt.clean_labels"
    assert tool.step_section_buttons["method"].text() == "eplt.clean_labels"
    tool._select_step_section("method")
    erlab_method_page = tool.step_editor_stack.currentWidget()
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QComboBox)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QPlainTextEdit)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QLineEdit)
    )

    namespace = {"data": data, "profile": profile}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"].shape == (1, 2)
    assert namespace["axs"][0, 0].get_xlim() == pytest.approx((0.25, 0.75))


def test_figure_composer_method_doc_url_uses_family_templates() -> None:
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.AXES_METHODS["text"]
    ) == ("https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html")
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.FIGURE_METHODS["supxlabel"]
    ) == (
        "https://matplotlib.org/stable/api/_as_gen/"
        "matplotlib.figure.Figure.supxlabel.html"
    )
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.ERLAB_METHODS["clean_labels"]
    ) == (
        "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
        "#erlab.plotting.clean_labels"
    )
    spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="call_name",
        label="call_name",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.FIGURE,
        call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
        doc_name="documented_name",
    )
    assert figurecomposer_method._method_doc_url(spec) == (
        "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
        "#erlab.plotting.documented_name"
    )


def test_figure_composer_method_docs_button_opens_current_url(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="text",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supxlabel",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    opened_urls: list[str] = []

    def record_url(url: QtCore.QUrl) -> bool:
        opened_urls.append(url.toString())
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", record_url)

    for row, expected in (
        (
            0,
            "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html",
        ),
        (
            1,
            "https://matplotlib.org/stable/api/_as_gen/"
            "matplotlib.figure.Figure.supxlabel.html",
        ),
        (
            2,
            "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
            "#erlab.plotting.clean_labels",
        ),
    ):
        tool.operation_list.setCurrentRow(row)
        tool._select_step_section("method")
        button = tool.step_editor_stack.currentWidget().findChild(
            QtWidgets.QToolButton, "figureComposerMethodDocsButton"
        )
        assert button is not None
        assert button.isEnabled()
        assert not button.autoRaise()
        assert button.property("figure_method_doc_url") == expected
        button.click()

    assert opened_urls == [
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html",
        (
            "https://matplotlib.org/stable/api/_as_gen/"
            "matplotlib.figure.Figure.supxlabel.html"
        ),
        (
            "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
            "#erlab.plotting.clean_labels"
        ),
    ]


def test_figure_composer_method_helper_edge_contracts(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2, layout=None),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlim",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("compressed",),
                    kwargs={"hspace": 0.2, "pad": 0.1},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="Unsupported axes method"):
        figurecomposer_method._method_spec(
            FigureOperationState.method(family=FigureMethodFamily.AXES, name="missing")
        )
    text_spec = figurecomposer_method.AXES_METHODS["set_xlim"]
    assert figurecomposer_method._method_selector_text(text_spec) == text_spec.name

    colorbar_operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="nice_colorbar",
    )
    colorbar_spec = figurecomposer_method._method_spec(colorbar_operation)
    with pytest.raises(ValueError, match="Unsupported call policy"):
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(update={"method_call_policy": "bad-policy"}),
            colorbar_spec,
        )
    with pytest.raises(ValueError, match="not available"):
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        figurecomposer_method.MethodCallPolicy.PLAIN_CALL.value
                    )
                }
            ),
            colorbar_spec,
        )
    assert (
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
                    )
                }
            ),
            colorbar_spec,
        )
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD
    )

    assert figurecomposer_method._live_layout_axes(
        tool, render_if_missing=True
    ).shape == (1, 2)
    assert (
        figurecomposer_method._first_live_axis(
            tool,
            FigureAxesSelectionState(expression="axs[3, 3]"),
        )
        is None
    )
    assert (
        figurecomposer_method._method_float_pair_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="set_xlim",
                axes=FigureAxesSelectionState(expression="axs[3, 3]"),
            ),
            figurecomposer_method.AXES_METHODS["set_xlim"],
        )
        is None
    )

    grid_axis = FigureGridSpecAxesState(
        axes_id="axis-a",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(grid_axis,),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_tool.show_figure_window(activate=False)
    grid_axes = figurecomposer_method._live_layout_axes(grid_tool)
    assert isinstance(grid_axes, dict)
    assert set(grid_axes) == {"axis-a"}
    assert (
        figurecomposer_method._first_live_axis(
            grid_tool,
            FigureAxesSelectionState(axes=(), axes_ids=()),
        )
        is grid_axes["axis-a"]
    )
    grid_tool.figure.clear()
    assert figurecomposer_method._live_layout_axes(grid_tool) is None
    assert (
        figurecomposer_method._limit_method_default_args(
            grid_tool,
            figurecomposer_method.AXES_METHODS["set_xlim"],
            FigureAxesSelectionState(axes=(), axes_ids=("missing-axis",)),
        )
        == ()
    )

    int_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.INT_ARG,
        label="Count",
        tooltip="count tooltip",
        object_name="count",
        default=None,
        step=2,
    )
    int_spin = figurecomposer_method._int_spinbox(None, int_control, parent=tool)
    assert int_spin.value() == 0
    assert int_spin.singleStep() == 2
    float_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_ARG,
        label="Value",
        tooltip="value tooltip",
        object_name="value",
        default=None,
        decimals=2,
        step=0.25,
    )
    float_spin = figurecomposer_method._float_spinbox(None, float_control, parent=tool)
    assert float_spin.value() == pytest.approx(0.0)
    assert float_spin.decimals() == 2
    assert float_spin.singleStep() == pytest.approx(0.25)
    assert "multiple values" in figurecomposer_method._numeric_control_tooltip(
        float_control,
        mixed=True,
    )

    default_from_window = figurecomposer_method._subplots_adjust_default(tool, "left")
    assert default_from_window == pytest.approx(
        tool.figure_window.figure.subplotpars.left
    )
    spin_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="subplots_adjust",
        kwargs={"left": "bad"},
    )
    adjust_spin = figurecomposer_method._subplots_adjust_spinbox(
        tool,
        spin_operation,
        "left",
        mixed=False,
        parent=tool,
    )
    assert adjust_spin.value() == pytest.approx(default_from_window)
    adjust_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.2},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(adjust_tool)
    figurecomposer_method._update_current_subplots_adjust_kwarg(
        adjust_tool,
        "left",
        None,
    )
    assert "left" not in adjust_tool.tool_status.operations[0].method_kwargs

    layout_spec = figurecomposer_method.FIGURE_METHODS["set_layout_engine"]
    assert (
        figurecomposer_method._layout_engine_name(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="set_layout_engine",
            ),
            layout_spec,
        )
        == "none"
    )
    assert figurecomposer_method._filter_layout_engine_kwargs(
        (),
        {"pad": 0.1},
    ) == {"pad": 0.1}
    assert figurecomposer_method._filter_layout_engine_kwargs(
        ("tight",),
        {"pad": 0.1, "hspace": 0.2},
    ) == {"pad": 0.1}

    with pytest.raises(ValueError, match="argument index"):
        figurecomposer_method._control_arg_index(
            figurecomposer_method.MethodControlSpec(
                kind=figurecomposer_method.MethodControlKind.TEXT_ARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )
    with pytest.raises(ValueError, match="keyword name"):
        figurecomposer_method._control_key(
            figurecomposer_method.MethodControlSpec(
                kind=figurecomposer_method.MethodControlKind.TEXT_KWARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )

    assert figurecomposer_method._empty_text_as_none("") is None
    assert figurecomposer_method._empty_text_as_none("title") == "title"
    assert figurecomposer_method._string_tuple_from_text_or_none("") is None
    assert figurecomposer_method._string_tuple_from_text_or_none("a, b") == ("a", "b")
    assert figurecomposer_method._format_int_value(None) == ""
    assert figurecomposer_method._format_int_value(2.0) == "2"
    assert figurecomposer_method._format_float_value(None) == ""
    assert figurecomposer_method._format_float_value(1.25) == "1.25"
    assert figurecomposer_method._format_literal_value(None) == ""
    assert figurecomposer_method._format_literal_value({"alpha": 0.5}) == "alpha=0.5"
    assert figurecomposer_method._format_literal_value((1, 2)) == "1, 2"
    assert figurecomposer_method._format_literal_value("literal") == '"literal"'
    assert figurecomposer_method._format_plot_sequence(3.0) == "3.0"
    assert figurecomposer_method._plot_sequence_from_text("") == ()
    assert figurecomposer_method._plot_sequence_from_text("1, 2") == (1, 2)
    assert figurecomposer_method._format_aspect_value(None) == ""
    assert figurecomposer_method._format_aspect_value("equal") == "equal"
    assert figurecomposer_method._format_aspect_value(2) == "2"
    assert figurecomposer_method._format_aspect_value(("custom",)) == '("custom",)'
    assert figurecomposer_method._literal_value_from_text("alpha=0.5") == {"alpha": 0.5}
    assert figurecomposer_method._literal_value_from_text("") is None
    assert figurecomposer_method._literal_value_from_text("[1, 2]") == [1, 2]
    assert figurecomposer_method._aspect_value_from_text("") is None
    assert figurecomposer_method._aspect_value_from_text("equal") == "equal"
    assert figurecomposer_method._aspect_value_from_text("2") == 2.0
    assert figurecomposer_method._aspect_value_from_text("custom") == "custom"
    assert figurecomposer_method._aspect_value_from_text("'manual'") == "manual"
    assert figurecomposer_method._aspect_value_from_text("[1, 2]") == "[1, 2]"
    assert figurecomposer_method._optional_literal_from_text("") is None
    assert figurecomposer_method._optional_literal_from_text("alpha=0.5") == {
        "alpha": 0.5
    }
    assert figurecomposer_method._optional_float_from_text("") is None
    assert figurecomposer_method._optional_float_from_text("1.5") == 1.5
    assert figurecomposer_method._optional_int_from_text("") is None
    assert figurecomposer_method._optional_int_from_text("3") == 3

    assert figurecomposer_method._plot_y_arg_value(
        FigureOperationState.method(family=FigureMethodFamily.AXES, name="plot"),
        figurecomposer_method.AXES_METHODS["plot"],
    ) == (0.0, 1.0)
    empty_default_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="empty_default",
        label="empty_default",
        tooltip="empty",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
    )
    assert (
        figurecomposer_method._plot_y_arg_value(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="empty_default",
            ),
            empty_default_spec,
        )
        == ()
    )

    assert figurecomposer_method._family_from_label("bad") == FigureMethodFamily.ERLAB
    assert (
        figurecomposer_method._family_from_label("Figure Method")
        == FigureMethodFamily.FIGURE
    )
    assert (
        figurecomposer_method._call_policy_from_label("Selected axes together")
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD
    )
    assert (
        figurecomposer_method._call_policy_from_label("plain_call")
        == figurecomposer_method.MethodCallPolicy.PLAIN_CALL
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.AXES_METHODS["set_xlabel"]
        )
        == "set_xlabel"
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.FIGURE_METHODS["supxlabel"]
        )
        == "supxlabel"
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.ERLAB_METHODS["clean_labels"]
        )
        == "clean_labels"
    )
    assert (
        figurecomposer_method._method_combo_object_name(FigureMethodFamily.FIGURE)
        == "figureComposerFigureMethodCombo"
    )
    assert (
        figurecomposer_method._method_kwargs_object_name(FigureMethodFamily.ERLAB)
        == "figureComposerERLabMethodKwEdit"
    )
    assert (
        figurecomposer_method._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="supxlabel",
            )
        )
        == "fig.supxlabel"
    )
    assert (
        figurecomposer_method._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.ERLAB,
                name="clean_labels",
            )
        )
        == "eplt.clean_labels"
    )
    assert (
        figurecomposer_method._callable_display(
            figurecomposer_method.FIGURE_METHODS["supxlabel"]
        )
        == "fig.supxlabel"
    )
    assert (
        figurecomposer_method._callable_display(colorbar_spec)
        == "erlab.plotting.nice_colorbar"
    )
    assert (
        figurecomposer_method._method_doc_url(
            figurecomposer_method.MethodSpec(
                family=FigureMethodFamily.ERLAB,
                name="custom_doc",
                label="custom_doc",
                tooltip="custom",
                target_domain=figurecomposer_method.MethodTargetDomain.NONE,
                call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
                doc_url="https://example.test/docs",
            )
        )
        == "https://example.test/docs"
    )

    transform_figure, transform_axis = plt.subplots()
    try:
        assert (
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "figure",
            )
            is transform_figure.transFigure
        )
        assert (
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "dpi",
            )
            is transform_figure.dpi_scale_trans
        )
        assert figurecomposer_method._transform_component_code("figure") == (
            "fig.transFigure"
        )
        assert (
            figurecomposer_method._transform_component_code("dpi")
            == "fig.dpi_scale_trans"
        )
        with pytest.raises(ValueError, match="Unknown transform component"):
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "bad",
            )
        with pytest.raises(ValueError, match="Unknown transform component"):
            figurecomposer_method._transform_component_code("bad")
        assert (
            figurecomposer_method._render_method_transform(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                figurecomposer_method.AXES_METHODS["plot"],
                figure=transform_figure,
                axis=transform_axis,
            )
            is None
        )
        assert (
            figurecomposer_method._method_transform_code(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                figurecomposer_method.AXES_METHODS["plot"],
            )
            is None
        )
    finally:
        plt.close(transform_figure)

    positional_text_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_positional",
        label="text_values_positional",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    keyword_text_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_keyword",
        label="text_values_keyword",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.KWARG,
        text_values_kwarg="labels",
    )
    text_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="text_values",
    ).model_copy(update={"text_values": ("A", "B")})
    assert figurecomposer_method._render_args_kwargs(
        tool,
        text_operation,
        positional_text_spec,
    )[0] == (["A", "B"],)
    assert figurecomposer_method._render_args_kwargs(
        tool,
        text_operation,
        keyword_text_spec,
    )[1] == {"labels": ["A", "B"]}
    assert figurecomposer_method._code_args_kwargs(
        tool,
        text_operation,
        keyword_text_spec,
    )[1] == {"labels": ["A", "B"]}

    no_operation_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(),
            primary_source="data",
        ),
    )
    qtbot.addWidget(no_operation_tool)
    figurecomposer_method._update_current_method_name(no_operation_tool, "plot")

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    figure_method_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodCombo"
    )
    assert figure_method_combo is not None
    assert figure_method_combo.currentText() == "set_layout_engine"
    assert figure_method_combo.currentData() == "set_layout_engine"

    figurecomposer_method._update_current_method_args(tool, ("none",))
    assert tool.tool_status.operations[1].method_args == ("none",)
    figurecomposer_method._update_current_layout_engine(tool, 0, "tight")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.1}
    figurecomposer_method._update_current_method_arg(tool, 2, "third")
    assert tool.tool_status.operations[1].method_args == ("tight", None, "third")
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 4, "tail")
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        None,
        "third",
        (),
        ("tail",),
    )
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 1, "a, b")
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        ("a", "b"),
        "third",
        (),
        ("tail",),
    )
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 1, "")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    figurecomposer_method._update_current_method_kwarg(tool, "pad", None)
    assert tool.tool_status.operations[1].method_kwargs == {}
    figurecomposer_method._update_current_method_kwarg(tool, "pad", 0.3)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.3}
    figurecomposer_method._operation_trust_update_callback(tool)(True)
    assert tool.tool_status.operations[1].trusted is True
    figurecomposer_method._method_float_pair_args_update_callback(tool)("0, 1")
    assert tool.tool_status.operations[1].method_args == (0.0, 1.0)
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.AXES)
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.AXES
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.FIGURE)
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.FIGURE
    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    figurecomposer_method._update_current_method_name(tool, "set_layout_engine")
    assert tool.tool_status.operations[1].method_name == "set_layout_engine"
    figurecomposer_method._update_current_method_call_policy(
        tool,
        figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
    )
    assert tool.tool_status.operations[1].method_call_policy == "plain_call"
    figurecomposer_method._update_current_method_call_policy(
        tool,
        figurecomposer_method.MethodCallPolicy.BOUND_FIGURE,
    )
    assert tool.tool_status.operations[1].method_call_policy is None
    figurecomposer_method._update_current_method_text_values(tool, "\nlabel\n")
    assert tool.tool_status.operations[1].text_values == ("label",)


def test_figure_composer_method_framework_dispatch_policies(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    calls: list[tuple[str, object, dict[str, object]]] = []

    def ax_keyword_helper(*args: object, ax: object, **kwargs: object) -> None:
        calls.append(("ax_keyword", ax, kwargs))

    def fig_keyword_helper(*args: object, fig: object, **kwargs: object) -> None:
        calls.append(("fig_keyword", fig, kwargs))

    def each_axis_helper(*args: object, ax: object, **kwargs: object) -> None:
        calls.append(("each_axis", ax, kwargs))

    def plain_helper(*args: object, **kwargs: object) -> None:
        calls.append(("plain", args, kwargs))

    monkeypatch.setattr(
        eplt, "composer_ax_keyword_test", ax_keyword_helper, raising=False
    )
    monkeypatch.setattr(
        eplt, "composer_fig_keyword_test", fig_keyword_helper, raising=False
    )
    monkeypatch.setattr(
        eplt, "composer_each_axis_test", each_axis_helper, raising=False
    )
    monkeypatch.setattr(eplt, "composer_plain_test", plain_helper, raising=False)
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_ax_keyword_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_ax_keyword_test",
            label="composer_ax_keyword_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.AXES,
            call_policy=figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_fig_keyword_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_fig_keyword_test",
            label="composer_fig_keyword_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.FIGURE,
            call_policy=figurecomposer_method.MethodCallPolicy.FIG_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_each_axis_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_each_axis_test",
            label="composer_each_axis_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.AXES,
            call_policy=figurecomposer_method.MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_plain_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_plain_test",
            label="composer_plain_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.NONE,
            call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
            default_args=("value",),
        ),
    )
    operations = (
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_ax_keyword_test",
            axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            kwargs={"alpha": 0.5},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_fig_keyword_test",
            kwargs={"label": "figure"},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_each_axis_test",
            axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            kwargs={"color": "red"},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_plain_test",
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=operations,
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    assert [call[0] for call in calls] == [
        "ax_keyword",
        "fig_keyword",
        "each_axis",
        "each_axis",
        "plain",
    ]
    assert isinstance(calls[0][1], np.ndarray)
    assert calls[0][2] == {"alpha": 0.5}
    assert calls[1][1] is figure
    assert calls[1][2] == {"label": "figure"}
    assert calls[2][2] == {"color": "red"}
    assert calls[3][2] == {"color": "red"}
    assert calls[4][1] == ("value",)

    lines = [
        line
        for operation in operations
        for line in figurecomposer_method._method_code(tool, operation)
    ]
    assert lines == [
        "eplt.composer_ax_keyword_test(alpha=0.5, ax=axs)",
        'eplt.composer_fig_keyword_test(label="figure", fig=fig)',
        "for ax in axs.flat:",
        '    eplt.composer_each_axis_test(color="red", ax=ax)',
        'eplt.composer_plain_test("value")',
    ]


def test_figure_composer_axes_plot_data_update_helper(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    args=((0.0, 1.0),),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_method._update_current_plot_data_arg(tool, "x", "0, 1")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (0.0, 1.0))

    figurecomposer_method._update_current_plot_data_arg(tool, "y", "2, 3")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (2, 3))

    figurecomposer_method._update_current_plot_data_arg(tool, "x", "")
    assert tool.tool_status.operations[0].method_args == ((2, 3),)


def test_figure_composer_tick_params_controls_update_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="tick_params",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0,))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    assert method_page is not None
    editor = method_page.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerAxesMethodTickParamsEditor",
    )
    assert editor is not None
    assert editor.tick_params() == {}
    editor.setToolTip("generic tick params tooltip")
    assert editor.toolTip() == ""
    axis_control = editor.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTickParamsAxisCombo"
    )
    reset_control = editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsResetCombo"
    )
    bottom_control = editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsBottomCombo"
    )
    length_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLengthEdit"
    )
    colors_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    grid_disclosure = editor.findChild(
        QtWidgets.QWidget, "figureComposerAxesMethodTickParamsGridDisclosure"
    )
    assert axis_control is not None
    assert reset_control is not None
    assert bottom_control is not None
    assert length_edit is not None
    assert colors_edit is not None
    assert grid_disclosure is not None
    tick_tooltips = (
        axis_control.toolTip(),
        reset_control.toolTip(),
        bottom_control.toolTip(),
        length_edit.toolTip(),
        colors_edit.toolTip(),
        grid_disclosure.toolTip(),
    )
    assert all(tick_tooltips)
    assert len(set(tick_tooltips)) == len(tick_tooltips)
    assert tool.tool_status.operations[0].method_kwargs == {}

    _click_tick_params_segment(
        editor,
        "figureComposerAxesMethodTickParamsAxisCombo",
        "x",
    )
    _click_tick_params_segment(
        editor,
        "figureComposerAxesMethodTickParamsDirectionCombo",
        "inout",
    )
    _set_tick_params_button(
        editor,
        "figureComposerAxesMethodTickParamsBottomCombo",
        False,
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsLengthEdit",
        "4.5",
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsLabelSizeEdit",
        "9",
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsColorsEdit",
        "tab:red",
    )

    assert tool.tool_status.operations[0].method_kwargs == {
        "axis": "x",
        "direction": "inout",
        "bottom": False,
        "length": 4.5,
        "labelsize": 9,
        "colors": "tab:red",
    }

    _set_tick_params_button(
        editor,
        "figureComposerAxesMethodTickParamsBottomCombo",
        None,
    )
    _finish_tick_params_edit(editor, "figureComposerAxesMethodTickParamsLengthEdit", "")

    assert tool.tool_status.operations[0].method_kwargs == {
        "axis": "x",
        "direction": "inout",
        "labelsize": 9,
        "colors": "tab:red",
    }


def test_figure_composer_axes_methods_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="text",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (0.1, 0.9, "Panel"),
                        "method_transform": "axes",
                        "method_kwargs": {"ha": "left", "va": "top"},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="axvline",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(
                    update={
                        "method_args": (0.5,),
                        "method_kwargs": {
                            "color": "red",
                            "linestyle": "--",
                        },
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="axvspan",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (0.2, 0.4),
                        "method_kwargs": {"alpha": 0.25},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xticks",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ((0.0, 1.0), ("left", "right")),
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (True,),
                        "method_kwargs": {"which": "major", "axis": "x"},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_axis_off",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xscale",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_yscale",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(update={"method_args": ("linear",)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Left title",),
                        "method_kwargs": {"loc": "left", "pad": 2.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Momentum",),
                        "method_kwargs": {"loc": "right", "labelpad": 3.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_ylabel",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ("Energy",),
                        "method_kwargs": {"loc": "top", "labelpad": 4.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="margins",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_kwargs": {"x": 0.1, "y": 0.2, "tight": False},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_aspect",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (2.0,),
                        "method_kwargs": {"share": True},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="tick_params",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_kwargs": {
                            "axis": "x",
                            "which": "both",
                            "direction": "in",
                            "length": 6.0,
                            "width": 1.5,
                            "colors": "red",
                            "pad": 3.0,
                            "labelrotation": 45.0,
                            "labelsize": 8,
                            "bottom": True,
                            "top": True,
                            "labelbottom": True,
                            "labeltop": False,
                            "grid_color": "blue",
                            "grid_alpha": 0.25,
                            "grid_linewidth": 0.75,
                            "grid_linestyle": "--",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    method_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerAxesMethodCombo")
    transform_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    text_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodTextEdit")
    kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit")
    assert method_combo is not None
    assert transform_combo is not None
    assert text_edit is not None
    assert kwargs_edit is not None
    assert method_combo.currentText() == "text"
    assert method_combo.currentData() == "text"
    assert transform_combo.currentText() == "axes"
    assert text_edit.text() == "Panel"
    assert kwargs_edit.text() == 'ha="left", va="top"'
    assert tool.step_section_buttons["method"].text() == "ax.text"

    tool.operation_list.setCurrentRow(4)
    tool._select_step_section("method")
    grid_visible_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodGridVisibleCombo"
    )
    grid_which_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodWhichCombo"
    )
    grid_axis_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodAxisCombo"
    )
    assert grid_visible_combo is not None
    assert grid_which_combo is not None
    assert grid_axis_combo is not None
    assert grid_visible_combo.currentText() == "True"
    assert grid_which_combo.currentText() == "major"
    assert grid_axis_combo.currentText() == "x"

    scale_names = tuple(mscale.get_scale_names())
    assert "log" in scale_names
    tool.operation_list.setCurrentRow(6)
    tool._select_step_section("method")
    xscale_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodXScaleCombo"
    )
    assert xscale_combo is not None
    assert (
        tuple(xscale_combo.itemText(index) for index in range(xscale_combo.count()))
        == scale_names
    )
    assert xscale_combo.currentText() == "log"
    assert tool.tool_status.operations[6].method_args == ()

    tool.operation_list.setCurrentRow(7)
    tool._select_step_section("method")
    yscale_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodYScaleCombo"
    )
    assert yscale_combo is not None
    assert (
        tuple(yscale_combo.itemText(index) for index in range(yscale_combo.count()))
        == scale_names
    )
    y_scale = "linear" if "linear" in scale_names else scale_names[0]
    _activate_combo_text(yscale_combo, y_scale)
    assert tool.tool_status.operations[7].method_args == (y_scale,)

    tool.operation_list.setCurrentRow(8)
    tool._select_step_section("method")
    title_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTitleEdit"
    )
    title_loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTitleLocCombo"
    )
    title_pad_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodTitlePadEdit"
    )
    assert title_edit is not None
    assert title_loc_combo is not None
    assert title_pad_edit is not None
    assert title_edit.text() == "Left title"
    assert title_loc_combo.currentText() == "left"
    assert title_pad_edit.value() == 2.0

    tool.operation_list.setCurrentRow(11)
    tool._select_step_section("method")
    x_margin_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodXMarginEdit"
    )
    y_margin_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodYMarginEdit"
    )
    tight_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodMarginsTightCombo"
    )
    assert x_margin_edit is not None
    assert y_margin_edit is not None
    assert tight_combo is not None
    assert x_margin_edit.value() == pytest.approx(0.1)
    assert y_margin_edit.value() == pytest.approx(0.2)
    assert tight_combo.currentText() == "False"

    tool.operation_list.setCurrentRow(12)
    tool._select_step_section("method")
    aspect_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodAspectEdit"
    )
    aspect_share_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodAspectShareCombo"
    )
    assert aspect_edit is not None
    assert aspect_share_combo is not None
    assert aspect_edit.text() == "2"
    assert aspect_share_combo.currentText() == "True"
    aspect_edit.setText("2.5")
    aspect_edit.editingFinished.emit()
    assert tool.tool_status.operations[12].method_args == (2.5,)

    tool.operation_list.setCurrentRow(13)
    tool._select_step_section("method")
    tick_editor = tool.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerAxesMethodTickParamsEditor",
    )
    assert tick_editor is not None
    tick_top_button = tick_editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsTopCombo"
    )
    tick_labeltop_button = tick_editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsLabelTopCombo"
    )
    tick_length_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLengthEdit"
    )
    tick_label_size_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLabelSizeEdit"
    )
    tick_colors_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    assert tick_top_button is not None
    assert tick_labeltop_button is not None
    assert tick_length_edit is not None
    assert tick_label_size_edit is not None
    assert tick_colors_edit is not None
    assert tick_editor.tick_params()["axis"] == "x"
    assert tick_editor.tick_params()["which"] == "both"
    assert tick_editor.tick_params()["direction"] == "in"
    assert tick_top_button.property("tick_params_value") is True
    assert tick_labeltop_button.property("tick_params_value") is False
    assert tick_length_edit.text() == "6"
    assert tick_label_size_edit.text() == "8"
    assert tick_colors_edit.text() == "red"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    fig.canvas.draw()
    assert fig.axes[0].texts[0].get_text() == "Panel"
    assert fig.axes[0].texts[0].get_transform() == fig.axes[0].transAxes
    assert fig.axes[0].lines[0].get_color() == "red"
    assert fig.axes[0].axison is False
    assert fig.axes[1].lines[0].get_color() == "red"
    assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == [
        "left",
        "right",
    ]
    assert len(fig.axes[1].patches) == 1
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[1].get_yscale() == y_scale
    assert fig.axes[0].get_title(loc="left") == "Left title"
    assert fig.axes[0].get_xlabel() == "Momentum"
    assert fig.axes[1].get_ylabel() == "Energy"
    assert fig.axes[0].margins() == (0.1, 0.2)
    assert fig.axes[1].get_aspect() == 2.5
    tick = fig.axes[1].xaxis.get_major_ticks()[0]
    assert tick.tick1line.get_markersize() == 6.0
    assert tick.tick1line.get_markeredgewidth() == 1.5
    assert tick.tick1line.get_color() == "red"
    assert tick.tick2line.get_visible() is True
    assert tick.label1.get_fontsize() == 8.0
    assert tick.label1.get_rotation() == 45.0
    assert tick.label1.get_color() == "red"
    assert tick.label2.get_visible() is False

    code = tool.generated_code()
    assert (
        'axs[0, 0].text(0.1, 0.9, "Panel", ha="left", va="top", '
        "transform=axs[0, 0].transAxes)"
    ) in code
    assert (
        'for ax in axs.flat:\n    ax.axvline(0.5, color="red", linestyle="--")' in code
    )
    assert "axs[0, 1].axvspan(0.2, 0.4, alpha=0.25)" in code
    assert 'axs[0, 1].set_xticks((0.0, 1.0), ("left", "right"))' in code
    assert 'axs[0, 1].grid(True, which="major", axis="x")' in code
    assert "axs[0, 0].set_axis_off()" in code
    assert 'axs[0, 0].set_xscale("log")' in code
    assert f'axs[0, 1].set_yscale("{y_scale}")' in code
    assert 'axs[0, 0].set_title("Left title", loc="left", pad=2.0)' in code
    assert 'axs[0, 0].set_xlabel("Momentum", loc="right", labelpad=3.0)' in code
    assert 'axs[0, 1].set_ylabel("Energy", loc="top", labelpad=4.0)' in code
    assert "axs[0, 0].margins(x=0.1, y=0.2, tight=False)" in code
    assert "axs[0, 1].set_aspect(2.5, share=True)" in code
    assert (
        'axs[0, 1].tick_params(axis="x", which="both", direction="in", '
        'length=6.0, width=1.5, colors="red", pad=3.0, labelrotation=45.0, '
        "labelsize=8, bottom=True, top=True, labelbottom=True, "
        'labeltop=False, grid_color="blue", grid_alpha=0.25, '
        'grid_linewidth=0.75, grid_linestyle="--")'
    ) in code
    assert "for ax in (axs[0, 0],):" not in code
    assert "for ax in (axs[0, 1],):" not in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    namespace["fig"].canvas.draw()
    axs = namespace["axs"]
    assert axs[0, 0].texts[0].get_text() == "Panel"
    assert axs[0, 0].axison is False
    assert [tick.get_text() for tick in axs[0, 1].get_xticklabels()] == [
        "left",
        "right",
    ]
    assert axs[0, 0].get_xscale() == "log"
    assert axs[0, 1].get_yscale() == y_scale
    assert axs[0, 0].get_title(loc="left") == "Left title"
    assert axs[0, 0].get_xlabel() == "Momentum"
    assert axs[0, 1].get_ylabel() == "Energy"
    assert axs[0, 0].margins() == (0.1, 0.2)
    assert axs[0, 1].get_aspect() == 2.5
    tick = axs[0, 1].xaxis.get_major_ticks()[0]
    assert tick.tick1line.get_markersize() == 6.0
    assert tick.tick1line.get_markeredgewidth() == 1.5
    assert tick.tick1line.get_color() == "red"
    assert tick.tick2line.get_visible() is True
    assert tick.label1.get_fontsize() == 8.0
    assert tick.label1.get_rotation() == 45.0
    assert tick.label1.get_color() == "red"
    assert tick.label2.get_visible() is False


def test_figure_composer_axes_plot_method_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (
                            (0.0, 0.5, 1.0),
                            (1.0, 0.5, 0.0),
                        ),
                        "method_kwargs": {
                            "color": "C1",
                            "linestyle": "--",
                            "linewidth": 2.5,
                            "marker": "o",
                            "markersize": 4.0,
                            "markerfacecolor": "white",
                            "markeredgecolor": "black",
                            "alpha": 0.75,
                            "label": "manual",
                            "zorder": 4.0,
                            "clip_on": False,
                            "transform": "ignored",
                        },
                        "method_transform": "blend",
                        "method_transform_x": "data",
                        "method_transform_y": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ((0.0, 0.25, 1.0),),
                        "method_transform": "custom",
                        "method_transform_expression": "ax.transAxes",
                        "trusted": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    x_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit"
    )
    y_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit"
    )
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    color_button = method_page.findChild(
        QtWidgets.QWidget, "figureComposerAxesMethodPlotColorEditButton"
    )
    style_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotLineStyleCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    marker_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotMarkerCombo"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    transform_x_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformXCombo"
    )
    transform_y_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformYCombo"
    )
    kwargs_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit"
    )
    assert method_combo is not None
    assert x_edit is not None
    assert y_edit is not None
    assert color_edit is not None
    assert color_button is not None
    assert style_combo is not None
    assert width_spin is not None
    assert marker_combo is not None
    assert transform_combo is not None
    assert transform_x_combo is not None
    assert transform_y_combo is not None
    assert kwargs_edit is not None
    assert method_combo.currentText() == "plot"
    assert method_combo.currentData() == "plot"
    assert x_edit.text() == "0.0, 0.5, 1.0"
    assert y_edit.text() == "1.0, 0.5, 0.0"
    assert color_edit.text() == "C1"
    assert style_combo.currentText() == "--"
    assert width_spin.value() == pytest.approx(2.5)
    assert marker_combo.currentText() == "o"
    assert transform_combo.currentText() == "blend"
    assert transform_x_combo.currentText() == "data"
    assert transform_y_combo.currentText() == "axes"
    assert kwargs_edit.text() == "clip_on=False"

    color_edit.setText("tab:blue")
    color_edit.setModified(True)
    color_widget = color_edit.parentWidget()
    assert isinstance(color_widget, figurecomposer_widgets._ColorLineEditWidget)
    color_widget.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:blue"
    kwargs_edit.setText('clip_on=True, transform="ax.transData"')
    kwargs_edit.setModified(True)
    kwargs_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["clip_on"] is True
    assert tool.tool_status.operations[0].method_kwargs["transform"] == "ignored"

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    custom_transform_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerMethodTransformExpressionEdit"
    )
    custom_transform_trusted = method_page.findChild(
        QtWidgets.QCheckBox, "figureComposerMethodTransformTrustedCheck"
    )
    assert custom_transform_edit is not None
    assert custom_transform_trusted is not None
    assert custom_transform_edit.text() == "ax.transAxes"
    assert custom_transform_trusted.isChecked()

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), (0.0, 0.5, 1.0))
    np.testing.assert_allclose(line.get_ydata(), (1.0, 0.5, 0.0))
    assert line.get_color() == "tab:blue"
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == pytest.approx(2.5)
    assert line.get_marker() == "o"
    assert line.get_markersize() == pytest.approx(4.0)
    assert line.get_markerfacecolor() == "white"
    assert line.get_markeredgecolor() == "black"
    assert line.get_alpha() == pytest.approx(0.75)
    assert line.get_label() == "manual"
    assert line.get_zorder() == pytest.approx(4.0)
    assert line.get_clip_on() is True
    assert isinstance(line.get_transform(), mtransforms.BlendedGenericTransform)
    assert fig.axes[1].lines[0].get_transform() == fig.axes[1].transAxes

    code = tool.generated_code()
    assert "import matplotlib.transforms as mtransforms" in code
    assert "axs[0, 0].plot((0.0, 0.5, 1.0), (1.0, 0.5, 0.0)" in code
    assert 'color="tab:blue"' in code
    assert "clip_on=True" in code
    assert "transform=ax.transData" not in code
    assert (
        "transform=mtransforms.blended_transform_factory("
        "axs[0, 0].transData, axs[0, 0].transAxes)"
    ) in code
    assert "ax.plot((0.0, 0.25, 1.0), transform=ax.transAxes)" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    assert len(axs[0, 0].lines) == 1
    assert axs[0, 0].lines[0].get_color() == "tab:blue"
    assert axs[0, 1].lines[0].get_transform() == axs[0, 1].transAxes


def test_figure_composer_axes_plot_data_mode_switches_editor(qtbot) -> None:
    data = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    mode_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotDataModeCombo"
    )
    assert mode_combo is not None
    assert mode_combo.currentData() == "entered"
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is not None
    )
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit")
        is not None
    )

    _activate_combo_index(mode_combo, mode_combo.findData("from_data"))
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
            )
            is not None
        ),
        timeout=5000,
    )

    operation = tool.tool_status.operations[0]
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_page = tool.step_editor_stack.currentWidget()
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is None
    )
    x_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXSourceCombo"
    )
    x_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXValuesCombo"
    )
    y_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
    )
    y_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYValuesCombo"
    )
    assert x_source_combo is not None
    assert x_values_combo is not None
    assert y_source_combo is not None
    assert y_values_combo is not None
    assert x_source_combo.property("figureComposerPlotDataRole") == "x_source"
    assert x_values_combo.property("figureComposerPlotDataRole") == "x_values"
    assert y_source_combo.property("figureComposerPlotDataRole") == "y_source"
    assert y_values_combo.property("figureComposerPlotDataRole") == "y_values"
    assert x_source_combo.toolTip()
    assert x_values_combo.toolTip()
    assert y_source_combo.toolTip()
    assert y_values_combo.toolTip()
    assert x_source_combo.toolTip() != x_values_combo.toolTip()
    assert y_source_combo.toolTip() != y_values_combo.toolTip()
    assert x_source_combo.itemText(0) != x_values_combo.itemText(0)
    assert x_source_combo.currentData() is None
    assert y_source_combo.currentData() == "data"
    y_value_options = {
        y_values_combo.itemData(index) for index in range(y_values_combo.count())
    }
    assert y_value_options >= {
        ("data", None),
        ("coord", "kx"),
    }
    assert all(
        y_values_combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
        for index in range(y_values_combo.count())
        if y_values_combo.itemData(index) is not None
    )


def test_figure_composer_axes_plot_method_picks_data_render_and_codegen(qtbot) -> None:
    x_source = xr.DataArray(
        np.array([10.0, 20.0, 30.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="x_source",
    )
    y_source = xr.DataArray(
        np.array([1.0, 4.0, 9.0]),
        dims=("point",),
        coords={"point": [0, 1, 2]},
        name="y_source",
    )
    tool = FigureComposerTool(
        y_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="x_data", label="x data"),
                FigureSourceState(name="y_data", label="y data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="x_data", kind="coord", name="kx"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="y_data", kind="data"
                        ),
                        "method_kwargs": {"color": "C2", "label": "picked"},
                    }
                ),
            ),
            primary_source="y_data",
        ),
        source_data={"x_data": x_source, "y_data": y_source},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = tool.figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), y_source.values)
    assert line.get_color() == "C2"
    assert line.get_label() == "picked"

    code = tool.generated_code()
    assert 'x_data.coords["kx"].values' in code
    assert "y_data.values" in code
    namespace: dict[str, typing.Any] = {
        "x_data": x_source,
        "y_data": y_source,
    }
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["axs"][0, 0].lines[0]
    np.testing.assert_allclose(generated_line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(generated_line.get_ydata(), y_source.values)


@pytest.mark.parametrize(
    ("source_data", "x_state", "y_state", "message"),
    [
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="data", kind="coord", name="missing"),
            "Coordinate 'missing' is not available",
        ),
        (
            {"image": xr.DataArray(np.ones((2, 2)), dims=("x", "y"), name="image")},
            None,
            FigureMethodPlotValueState(source="image", kind="data"),
            "one-dimensional",
        ),
        (
            {
                "x": xr.DataArray([0.0, 1.0, 2.0], dims=("x",), name="x"),
                "y": xr.DataArray([1.0, 2.0], dims=("y",), name="y"),
            },
            FigureMethodPlotValueState(source="x", kind="data"),
            FigureMethodPlotValueState(source="y", kind="data"),
            "same length",
        ),
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="missing", kind="data"),
            "is not available",
        ),
    ],
)
def test_figure_composer_axes_plot_picked_data_invalid_inputs(
    qtbot,
    source_data: dict[str, xr.DataArray],
    x_state: FigureMethodPlotValueState | None,
    y_state: FigureMethodPlotValueState,
    message: str,
) -> None:
    primary_source = next(iter(source_data))
    tool = FigureComposerTool(
        source_data[primary_source],
        recipe=FigureRecipeState(
            sources=tuple(
                FigureSourceState(name=name, label=name) for name in source_data
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": x_state,
                        "method_plot_y": y_state,
                    }
                ),
            ),
            primary_source=primary_source,
        ),
        source_data=source_data,
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert message in errors[0]
    with pytest.raises(ValueError, match=message):
        tool.generated_code()


def test_figure_composer_axes_plot_picked_data_sources_are_renamed() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(
                source="x_data", kind="coord", name="kx"
            ),
            "method_plot_y": FigureMethodPlotValueState(source="y_data", kind="data"),
        }
    )

    assert figurecomposer_method._source_names(operation) == ("x_data", "y_data")
    renamed = FigureComposerTool._operation_with_renamed_sources(
        operation, {"x_data": "x_copy", "y_data": "y_copy"}
    )

    assert renamed.method_plot_x == FigureMethodPlotValueState(
        source="x_copy", kind="coord", name="kx"
    )
    assert renamed.method_plot_y == FigureMethodPlotValueState(
        source="y_copy", kind="data"
    )


def test_figure_composer_axes_plot_data_helper_edges() -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={"scan": [0], "point": np.linspace(0.0, 1.0, 6)},
        name="source",
    )
    coord_grid = xr.DataArray(
        np.ones((1, 6)),
        dims=("scan", "point"),
        coords=source.coords,
        name="coord_grid",
    )
    source_with_grid_coord = source.assign_coords(grid=coord_grid)
    source_with_2d_coord = xr.DataArray(
        np.ones((2, 6)),
        dims=("scan", "point"),
        coords={"scan": [0, 1], "point": np.linspace(0.0, 1.0, 6)},
        name="source_with_2d_coord",
    ).assign_coords(
        grid=(
            ("scan", "point"),
            np.arange(12.0).reshape(2, 6),
        )
    )
    tool = types.SimpleNamespace(
        _source_data={"source": source_with_grid_coord},
        _source_display_name=lambda name: f"display {name}",
        _source_names=lambda: ("source",),
        _source_tooltip=lambda name: f"tooltip {name}",
    )

    assert figurecomposer_method._plot_data_mode_text("unknown") == "Enter values"
    assert figurecomposer_method._plot_value_display(None) == "Choose values"
    assert figurecomposer_method._plot_value_options(tool, None) == ()
    assert figurecomposer_method._plot_value_options(tool, "missing") == ()
    option_tool = types.SimpleNamespace(
        _source_data={"source": source_with_2d_coord},
        _source_display_name=lambda name: f"display {name}",
        _source_names=lambda: ("source",),
        _source_tooltip=lambda name: f"tooltip {name}",
    )
    assert ("coord", "grid") not in {
        value
        for _text, value in figurecomposer_method._plot_value_options(
            option_tool, "source"
        )
    }
    assert figurecomposer_method._plot_coord_by_name(source, "point") is not None
    assert figurecomposer_method._plot_source_combo_tooltip("x", has_sources=False)
    assert figurecomposer_method._plot_values_combo_tooltip(
        "y", source="source", value_options_match=False
    )
    with pytest.raises(ValueError, match="Unknown plot value selection"):
        figurecomposer_method._plot_value_combo_data_parts(("bad", None))

    data_state = FigureMethodPlotValueState(source="source", kind="data")
    code, value = figurecomposer_method._plot_value_code_and_data(tool, data_state)
    assert str(code) == "source.squeeze(drop=True).values"
    assert value.dims == ("point",)

    coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="point"
    )
    coord_code, coord_value = figurecomposer_method._plot_value_code_and_data(
        tool, coord_state
    )
    assert str(coord_code) == 'source.coords["point"].values'
    assert coord_value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_method._plot_value_data(
            tool, FigureMethodPlotValueState(source="source", kind="coord")
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_method._plot_value_data(
            option_tool,
            FigureMethodPlotValueState(source="source", kind="coord", name="grid"),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        figurecomposer_method._picked_plot_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
        )

    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_y": data_state,
        }
    )
    args = figurecomposer_method._picked_plot_args(tool, operation)
    assert len(args) == 1
    np.testing.assert_allclose(args[0], source.values.reshape(-1))
    code_args = figurecomposer_method._picked_plot_code_args(tool, operation)
    assert tuple(str(arg) for arg in code_args) == ("source.squeeze(drop=True).values",)


def test_figure_composer_axes_plot_data_update_helpers() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "entered",
            "method_plot_x": FigureMethodPlotValueState(source="old", kind="data"),
        }
    )
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    calls: list[bool] = []

    class UpdateTool:
        def __init__(self) -> None:
            self._source_data = {"data": data}
            self.operation = operation

        def _source_names(self) -> tuple[str, ...]:
            return ("data",)

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
            *,
            rebuild_editor: bool = False,
        ) -> None:
            calls.append(rebuild_editor)
            self.operation = updater(0, self.operation)

    tool = UpdateTool()
    figurecomposer_method._update_current_plot_data_mode(tool, "invalid")
    assert calls == []

    figurecomposer_method._update_current_plot_data_mode(tool, "from_data")
    assert calls[-1] is True
    assert tool.operation.method_plot_data_mode == "from_data"
    assert tool.operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )

    figurecomposer_method._update_current_plot_value_source(tool, "x", None)
    assert tool.operation.method_plot_x is None
    figurecomposer_method._update_current_plot_value_source(tool, "x", "data")
    assert tool.operation.method_plot_x == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    figurecomposer_method._update_current_plot_value_selection(
        tool, "x", ("coord", "x")
    )
    assert tool.operation.method_plot_x == FigureMethodPlotValueState(
        source="data", kind="coord", name="x"
    )
    figurecomposer_method._update_current_plot_value_selection(tool, "x", None)
    assert tool.operation.method_plot_x is None

    assert (
        figurecomposer_method._source_names(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="missing",
            )
        )
        == ()
    )


def test_figure_composer_axes_plot_optional_bool_editor_branch(qtbot) -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="tick_params",
        kwargs={"reset": True},
    )
    layout_parent = QtWidgets.QWidget()
    qtbot.addWidget(layout_parent)
    layout = QtWidgets.QFormLayout(layout_parent)
    updates: list[FigureOperationState] = []

    class EditorTool:
        def _batch_is_mixed(
            self,
            _operation: FigureOperationState,
            _getter: Callable[[FigureOperationState], object],
        ) -> bool:
            return False

        def _optional_name_combo(
            self,
            names: Sequence[str],
            current: str | None,
            none_label: str,
            callback: Callable[[str | None], None],
            *,
            parent: QtWidgets.QWidget | None = None,
            mixed: bool = False,
        ) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox(parent)
            combo.addItem(none_label, None)
            for name in names:
                combo.addItem(name, name)
            if current is not None:
                combo.setCurrentText(current)
            combo.setProperty("figureComposerMixedValue", mixed)
            combo.activated.connect(lambda _index: callback(combo.currentData()))
            return combo

        def _add_form_row(
            self,
            form: QtWidgets.QFormLayout,
            _label: str,
            widget: QtWidgets.QWidget,
            _tooltip: str,
        ) -> None:
            form.addRow(widget)

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
        ) -> None:
            updates.append(updater(0, operation))

    control = figurecomposer_method._optional_bool_kwarg_combo(
        "Reset",
        "reset",
        "figureComposerTestResetCombo",
        "tooltip",
    )
    figurecomposer_method._add_method_control_row(
        EditorTool(),
        layout,
        operation,
        figurecomposer_method.AXES_METHODS["tick_params"],
        control,
    )
    combo = layout_parent.findChild(QtWidgets.QComboBox, "figureComposerTestResetCombo")
    assert combo is not None
    assert combo.currentText() == "True"

    combo.setCurrentText("False")
    combo.activated.emit(combo.currentIndex())
    assert updates[-1].method_kwargs["reset"] is False


def test_figure_composer_axes_plot_combo_helper_edges(qtbot) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={
            "scan": [0],
            "point": np.linspace(0.0, 1.0, 6),
            "scan_coord": ("scan", [10.0]),
            "row_coord": (("scan", "point"), np.arange(6.0).reshape(1, 6)),
        },
        name="source",
    )

    class ComboTool:
        def __init__(self) -> None:
            self._source_data = {"source": source}
            self.operation_editor = QtWidgets.QWidget()
            self.operation = FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            )

        def _mark_editor_control(self, _widget: QtWidgets.QWidget) -> None:
            return None

        def _connect_editor_signal(
            self,
            _widget: QtWidgets.QWidget,
            signal: typing.Any,
            callback: Callable[..., None],
        ) -> None:
            signal.connect(callback)

        def _source_names(self) -> tuple[str, ...]:
            return ("source",)

        def _source_display_name(self, name: str) -> str:
            return f"display {name}"

        def _source_tooltip(self, name: str) -> str:
            return f"tooltip {name}"

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
        ) -> None:
            self.operation = updater(0, self.operation)

    tool = ComboTool()
    qtbot.addWidget(tool.operation_editor)
    changed: list[object] = []

    mode_combo = figurecomposer_method._plot_data_mode_combo(
        tool,
        current=None,
        changed=changed.append,
        parent=tool.operation_editor,
        mixed=True,
    )
    assert mode_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", mode_combo.model()).item(0).isEnabled()

    source_combo = figurecomposer_method._plot_source_combo(
        tool,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert source_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", source_combo.model()).item(0).isEnabled()

    values_combo = figurecomposer_method._plot_values_combo(
        tool,
        source="source",
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert values_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", values_combo.model()).item(0).isEnabled()

    no_source_values = figurecomposer_method._plot_values_combo(
        tool,
        source=None,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
    )
    assert no_source_values.itemData(0) is None
    assert not no_source_values.isEnabled()
    assert figurecomposer_method._plot_values_combo_tooltip(
        "y", source=None, value_options_match=True
    )

    squeezed_coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="row_coord"
    )
    code, value = figurecomposer_method._plot_value_code_and_data(
        tool, squeezed_coord_state
    )
    assert str(code) == 'source.coords["row_coord"].squeeze(drop=True).values'
    assert value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_method._plot_value_code_and_data(
            tool, FigureMethodPlotValueState(source="source", kind="coord")
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_method._plot_value_code_and_data(
            tool,
            FigureMethodPlotValueState(
                source="source", kind="coord", name="scan_coord"
            ),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        figurecomposer_method._picked_plot_code_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
        )

    callback = figurecomposer_method._method_optional_bool_kwarg_callback(
        tool, "visible"
    )
    callback(None)
    assert "visible" not in tool.operation.method_kwargs
    callback("True")
    assert tool.operation.method_kwargs["visible"] is True

    optional_bool_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO,
        label="Visible",
        tooltip="",
        object_name="visible",
    )
    assert figurecomposer_method._control_accepts_value(optional_bool_control, None)
    assert figurecomposer_method._control_accepts_value(optional_bool_control, False)
    assert not figurecomposer_method._control_accepts_value(
        optional_bool_control, "False"
    )


@pytest.mark.parametrize(
    ("mode", "code_fragment"),
    [
        ("figure", "fig.transFigure"),
        ("dpi", "fig.dpi_scale_trans"),
        ("xaxis", "ax.get_xaxis_transform()"),
        ("yaxis", "ax.get_yaxis_transform()"),
    ],
)
def test_figure_composer_method_transform_presets(
    mode: str, code_fragment: str
) -> None:
    figure, axis = plt.subplots()
    spec = figurecomposer_method.AXES_METHODS["plot"]
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(update={"method_transform": mode})

    transform = figurecomposer_method._render_method_transform(
        operation,
        spec,
        figure=figure,
        axis=axis,
    )
    code = figurecomposer_method._method_transform_code(operation, spec)

    assert isinstance(transform, mtransforms.Transform)
    assert repr(code) == code_fragment
    plt.close(figure)


def test_figure_composer_method_custom_transform_errors() -> None:
    figure, axis = plt.subplots()
    spec = figurecomposer_method.AXES_METHODS["plot"]
    untrusted = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_transform": "custom",
            "method_transform_expression": "ax.transAxes",
        }
    )
    empty = untrusted.model_copy(
        update={"trusted": True, "method_transform_expression": ""}
    )
    not_transform = untrusted.model_copy(
        update={"trusted": True, "method_transform_expression": "1"}
    )

    with pytest.raises(ValueError, match="not trusted"):
        figurecomposer_method._method_transform_code(untrusted, spec)
    with pytest.raises(ValueError, match="empty"):
        figurecomposer_method._method_transform_code(empty, spec)
    with pytest.raises(ValueError, match="empty"):
        figurecomposer_method._render_method_transform(
            empty,
            spec,
            figure=figure,
            axis=axis,
        )
    with pytest.raises(TypeError, match="Transform"):
        figurecomposer_method._render_method_transform(
            not_transform,
            spec,
            figure=figure,
            axis=axis,
        )
    plt.close(figure)


def test_figure_composer_loaded_custom_method_transform_requires_trust(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_args": ((0.0, 1.0),),
                        "method_transform": "custom",
                        "method_transform_expression": "ax.transAxes",
                        "trusted": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    loaded = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(loaded)
    assert isinstance(loaded, FigureComposerTool)
    operation = loaded.tool_status.operations[0]
    assert operation.method_transform == "custom"
    assert operation.method_transform_expression == "ax.transAxes"
    assert operation.trusted is False

    figurecomposer_rendering._render_into_figure(
        loaded, loaded.figure, sync_visible=False
    )
    assert "not trusted" in loaded._operation_render_errors[operation.operation_id]


def test_figure_composer_limit_methods_default_to_current_axis_limits(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_profile_source("data"))
    qtbot.addWidget(tool)
    layout_axes = figurecomposer_rendering._live_layout_axes(
        tool, render_if_missing=True
    )
    assert layout_axes is not None

    axis = figurecomposer_rendering._iter_axes(layout_axes)[0]
    expected_xlim = tuple(float(value) for value in axis.get_xlim())
    expected_ylim = tuple(float(value) for value in axis.get_ylim())
    assert tool._figure_window is None

    tool._add_operation("method:axes")
    figurecomposer_method._update_current_method_name(tool, "set_xlim")
    tool._select_step_section("method")
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
    )
    assert limits_edit is not None
    assert tool.tool_status.operations[-1].method_args == pytest.approx(expected_xlim)
    assert limits_edit.text() == f"{expected_xlim[0]:g}, {expected_xlim[1]:g}"

    tool._add_operation("method:axes")
    figurecomposer_method._update_current_method_name(tool, "set_ylim")
    tool._select_step_section("method")
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
    )
    assert limits_edit is not None
    assert tool.tool_status.operations[-1].method_args == pytest.approx(expected_ylim)
    assert limits_edit.text() == f"{expected_ylim[0]:g}, {expected_ylim[1]:g}"


def test_figure_composer_method_selector_preserves_compatible_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Momentum",),
                        "method_kwargs": {
                            "loc": "right",
                            "labelpad": 3.0,
                            "fontsize": 12,
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")

    method_page = tool.step_editor_stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    assert method_combo is not None
    initial_operation = tool.tool_status.operations[0]
    method_combo.activated.emit(method_combo.currentIndex())
    assert tool.tool_status.operations[0] == initial_operation

    ylabel_index = method_combo.findData("set_ylabel")
    assert ylabel_index >= 0
    _activate_combo_index(method_combo, ylabel_index)
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "set_ylabel"
    assert operation.method_args == ("Momentum",)
    assert operation.method_kwargs == {
        "loc": "top",
        "labelpad": 3.0,
        "fontsize": 12,
    }
    assert operation.axes == FigureAxesSelectionState(axes=((0, 0),))

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert not tool._operation_render_errors
    assert tool.figure.axes[0].get_ylabel() == "Momentum"

    namespace: dict[str, typing.Any] = {}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"][0, 0].get_ylabel() == "Momentum"

    figurecomposer_method._update_current_method_name(tool, "set_xlim")
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "set_xlim"
    assert operation.method_args != ("Momentum",)
    assert operation.method_kwargs == {}


def test_figure_composer_method_transfer_edge_contracts(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlim",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    initial_operation = tool.tool_status.operations[0]
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.AXES)
    assert tool.tool_status.operations[0] == initial_operation

    tool._updating_controls = True
    try:
        figurecomposer_method._update_current_method_name(tool, "set_ylim")
    finally:
        tool._updating_controls = False
    assert tool.tool_status.operations[0] == initial_operation

    combo_no_none = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
    )
    combo_with_none = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
        none_label="None",
    )
    bool_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.BOOL_KWARG_COMBO,
        label="Flag",
        tooltip="flag",
        object_name="flag",
        key="flag",
    )
    assert not figurecomposer_method._control_accepts_value(combo_no_none, None)
    assert figurecomposer_method._control_accepts_value(combo_with_none, None)
    assert figurecomposer_method._control_accepts_value(bool_combo, True)
    assert not figurecomposer_method._control_accepts_value(bool_combo, "True")

    assert (
        figurecomposer_method._transfer_axis_label_loc(
            figurecomposer_method.AXES_METHODS["set_ylabel"],
            figurecomposer_method.AXES_METHODS["set_xlabel"],
            "top",
        )
        == "right"
    )
    assert (
        figurecomposer_method._transfer_axis_label_loc(
            figurecomposer_method.AXES_METHODS["set_title"],
            figurecomposer_method.AXES_METHODS["set_xlabel"],
            "unchanged",
        )
        == "unchanged"
    )

    float_pair_updates = figurecomposer_method._method_transfer_updates(
        tool,
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlim",
            args=(1.0, 2.0),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        figurecomposer_method.AXES_METHODS["set_ylim"],
    )
    assert float_pair_updates["method_args"] == (1.0, 2.0)

    default_arg_updates = figurecomposer_method._method_transfer_updates(
        tool,
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlabel",
            args=("x",),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        figurecomposer_method.AXES_METHODS["set_ylabel"],
    )
    assert default_arg_updates["method_args"] == ("y",)

    plot_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
        args=((0.0, 1.0), (1.0, 2.0)),
        kwargs={"custom": "value"},
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_transform": "custom",
            "method_transform_x": "figure",
            "method_transform_y": "data",
            "method_transform_expression": "ax.transData",
        }
    )
    plot_updates = figurecomposer_method._method_transfer_updates(
        tool,
        plot_operation,
        figurecomposer_method.AXES_METHODS["plot"],
    )
    assert plot_updates["method_args"] == plot_operation.method_args
    assert plot_updates["method_kwargs"] == {"custom": "value"}
    assert plot_updates["method_transform"] == "custom"
    assert plot_updates["method_transform_x"] == "figure"
    assert plot_updates["method_transform_y"] == "data"
    assert plot_updates["method_transform_expression"] == "ax.transData"

    text_arg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TEXT_ARG,
        label="Text",
        tooltip="text",
        object_name="text",
        arg_index=0,
    )
    accepted_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Accepted",
        tooltip="accepted",
        object_name="accepted",
        arg_index=2,
        options=("keep",),
    )
    source_rejected_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("bad",),
    )
    target_rejected_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("ok",),
    )
    same_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_KWARG,
        label="Same",
        tooltip="same",
        object_name="same",
        key="same",
    )
    source_mismatch_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TEXT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    target_mismatch_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    transform_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TRANSFORM,
        label="Transform",
        tooltip="transform",
        object_name="transform",
    )
    source_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        label="transfer_source",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
            figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
        default_args=("default",),
        controls=(
            text_arg,
            accepted_combo,
            source_rejected_combo,
            same_kwarg,
            source_mismatch_kwarg,
            transform_control,
        ),
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    target_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_target",
        label="transfer_target",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
            figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
        controls=(
            text_arg,
            accepted_combo,
            target_rejected_combo,
            same_kwarg,
            target_mismatch_kwarg,
            transform_control,
        ),
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    monkeypatch.setitem(
        figurecomposer_method.AXES_METHODS, "transfer_source", source_spec
    )
    monkeypatch.setitem(
        figurecomposer_method.AXES_METHODS, "transfer_target", target_spec
    )
    transfer_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        args=("default", "ignored", "keep", "bad"),
        kwargs={"same": 2.0, "mismatch": "skip"},
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_call_policy": (
                figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
            ),
            "text_values": ("A", "B"),
            "method_transform": "custom",
            "method_transform_x": "figure",
            "method_transform_y": "data",
            "method_transform_expression": "ax.transData",
        }
    )

    transfer_updates = figurecomposer_method._method_transfer_updates(
        tool,
        transfer_operation,
        target_spec,
    )
    assert transfer_updates["method_args"] == (None, None, "keep")
    assert transfer_updates["method_kwargs"] == {"same": 2.0}
    assert (
        transfer_updates["method_call_policy"]
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
    )
    assert transfer_updates["text_values"] == ("A", "B")
    assert transfer_updates["method_transform"] == "custom"
    assert transfer_updates["method_transform_x"] == "figure"
    assert transfer_updates["method_transform_y"] == "data"
    assert transfer_updates["method_transform_expression"] == "ax.transData"


def test_figure_composer_batch_same_method_edits_selected_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("left",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("right",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    args=("unchanged",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    title_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTitleEdit"
    )
    assert title_edit is not None
    assert title_edit.text() == ""
    assert title_edit.placeholderText() == "(multiple values)"

    title_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_args == ("left",)
    assert tool.tool_status.operations[1].method_args == ("right",)

    title_edit.setText("shared")
    title_edit.setModified(True)
    title_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_args == ("shared",)
    assert tool.tool_status.operations[1].method_args == ("shared",)
    assert tool.tool_status.operations[2].method_args == ("unchanged",)
    assert _selected_operation_rows(tool) == (0, 1)


def test_figure_composer_batch_same_plot_method_edits_selected_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "red", "linewidth": 1.0},
                        "method_transform": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "blue", "linewidth": 3.0},
                        "method_transform": "figure",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("unchanged",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    assert color_edit is not None
    assert transform_combo is not None
    assert width_spin is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"
    assert transform_combo.currentText() == "(multiple values)"
    assert width_spin.value() == pytest.approx(float(mpl.rcParams["lines.linewidth"]))
    width_spin_container = width_spin.parentWidget()
    assert width_spin_container is not None
    assert width_spin_container.findChild(
        QtWidgets.QLabel, "figureComposerMixedValueMarker"
    )

    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "red"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "blue"
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        1.0
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        3.0
    )

    color_edit.setText("tab:green")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:green"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "tab:green"
    width_spin.setValue(4.5)
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )

    _activate_combo_text(transform_combo, "blend")
    assert tool.tool_status.operations[0].method_transform == "blend"
    assert tool.tool_status.operations[1].method_transform == "blend"
    assert tool.tool_status.operations[2].method_args == ("unchanged",)
    assert _selected_operation_rows(tool) == (0, 1)


def test_figure_composer_batch_incompatible_methods_disable_editor(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_ylabel",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))

    assert tool.step_editor_stack.currentWidget().objectName() == (
        "figureComposerIncompatibleBatchPage"
    )
    assert (
        tool.step_editor_stack.currentWidget().findChild(
            QtWidgets.QLineEdit, "figureComposerAxesMethodTitleEdit"
        )
        is None
    )


def test_figure_composer_figure_method_has_no_axes_target(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supxlabel",
                    args=("Momentum",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    assert "axes" not in tool.step_section_buttons
    assert tool.operation_list.item(0).text() == "fig.supxlabel"
    assert tool.step_section_buttons["method"].text() == "fig.supxlabel"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert fig._supxlabel is not None
    assert fig._supxlabel.get_text() == "Momentum"

    code = tool.generated_code()
    assert 'fig.supxlabel("Momentum")' in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"]._supxlabel.get_text() == "Momentum"


def test_figure_composer_figure_layout_methods_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    adjust_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={
                        "left": 0.2,
                        "bottom": 0.15,
                        "right": 0.8,
                        "top": 0.85,
                        "wspace": 0.4,
                        "hspace": 0.3,
                    },
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(adjust_tool)

    adjust_tool.operation_list.setCurrentRow(0)
    adjust_tool._select_step_section("method")
    adjust_page = adjust_tool.step_editor_stack.currentWidget()
    left_spin = adjust_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustLeftEdit"
    )
    top_spin = adjust_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustTopEdit"
    )
    assert left_spin is not None
    assert top_spin is not None
    assert left_spin.value() == pytest.approx(0.2)
    assert left_spin.minimum() == pytest.approx(0.0)
    assert left_spin.maximum() == pytest.approx(0.8 - 10.0 ** -left_spin.decimals())
    assert left_spin.decimals() == 3
    assert left_spin.singleStep() == pytest.approx(0.005)
    assert not left_spin.keyboardTracking()
    top_spin.setValue(0.9)
    assert adjust_tool.tool_status.operations[0].method_kwargs["top"] == 0.9

    default_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(default_tool)
    default_tool.operation_list.setCurrentRow(0)
    default_tool._select_step_section("method")
    default_page = default_tool.step_editor_stack.currentWidget()
    default_left_spin = default_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustLeftEdit"
    )
    assert default_left_spin is not None
    default_figure = Figure(
        figsize=default_tool.tool_status.setup.figsize,
        dpi=default_tool.tool_status.setup.dpi,
        layout=default_tool.tool_status.setup.layout,
    )
    assert default_left_spin.value() == pytest.approx(default_figure.subplotpars.left)

    fig = adjust_tool.figure
    figurecomposer_rendering._render_into_figure(adjust_tool, fig, sync_visible=False)
    assert fig.subplotpars.left == pytest.approx(0.2)
    assert fig.subplotpars.bottom == pytest.approx(0.15)
    assert fig.subplotpars.right == pytest.approx(0.8)
    assert fig.subplotpars.top == pytest.approx(0.9)
    assert fig.subplotpars.wspace == pytest.approx(0.4)
    assert fig.subplotpars.hspace == pytest.approx(0.3)

    code = adjust_tool.generated_code()
    assert (
        "fig.subplots_adjust(left=0.2, bottom=0.15, right=0.8, "
        "top=0.9, wspace=0.4, hspace=0.3)"
    ) in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].subplotpars.top == pytest.approx(0.9)

    engine_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("tight",),
                    kwargs={"pad": 0.5, "hspace": 0.2},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(engine_tool)

    engine_tool.operation_list.setCurrentRow(0)
    engine_tool._select_step_section("method")
    engine_page = engine_tool.step_editor_stack.currentWidget()
    engine_combo = engine_page.findChild(
        QtWidgets.QComboBox, "figureComposerFigureLayoutEngineCombo"
    )
    pad_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEnginePadEdit"
    )
    hspace_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
    )
    assert engine_combo is not None
    assert pad_edit is not None
    assert hspace_edit is None
    assert engine_combo.currentText() == "tight"
    assert pad_edit.value() == pytest.approx(0.5)
    assert "hspace" not in engine_tool.generated_code()

    _activate_combo_text(engine_combo, "compressed")
    qtbot.waitUntil(
        lambda: (
            engine_tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    operation = engine_tool.tool_status.operations[0]
    assert operation.method_args == ("compressed",)
    assert operation.method_kwargs == {"hspace": 0.2}

    engine_page = engine_tool.step_editor_stack.currentWidget()
    assert (
        engine_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEnginePadEdit"
        )
        is None
    )
    hspace_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
    )
    rect_edit = engine_page.findChild(
        QtWidgets.QLineEdit, "figureComposerFigureLayoutEngineRectEdit"
    )
    assert hspace_edit is not None
    assert rect_edit is not None
    assert hspace_edit.value() == pytest.approx(0.2)
    rect_edit.setText("0, 0, 0.9, 1")
    rect_edit.editingFinished.emit()
    assert engine_tool.tool_status.operations[0].method_kwargs == {
        "hspace": 0.2,
        "rect": (0, 0, 0.9, 1),
    }

    fig = engine_tool.figure
    figurecomposer_rendering._render_into_figure(engine_tool, fig, sync_visible=False)
    assert fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine"

    code = engine_tool.generated_code()
    assert (
        'fig.set_layout_engine("compressed", hspace=0.2, rect=(0, 0, 0.9, 1))'
    ) in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].get_layout_engine().__class__.__name__ == (
        "ConstrainedLayoutEngine"
    )


def test_figure_composer_layout_engine_none_is_post_creation_method(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("none",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.25},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert not any(
        "incompatible with subplots_adjust" in str(item.message) for item in caught
    )
    assert fig.get_layout_engine() is None
    assert fig.subplotpars.left == pytest.approx(0.25)

    code = tool.generated_code()
    assert 'layout="none"' in code
    assert 'fig.set_layout_engine("none")' in code
    namespace = {"data": data}
    with warnings.catch_warnings(record=True) as generated_caught:
        warnings.simplefilter("always")
        exec(code, namespace)  # noqa: S102
    assert not any(
        "incompatible with subplots_adjust" in str(item.message)
        for item in generated_caught
    )
    assert namespace["fig"].get_layout_engine() is None
    assert namespace["fig"].subplotpars.left == pytest.approx(0.25)


def test_figure_composer_legend_methods_render_and_codegen(qtbot) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    line_operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_labels": ("profile",)})
    axes_legend_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="legend",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_kwargs": {
                "loc": "upper right",
                "ncols": 1,
                "title": "Axis legend",
                "frameon": False,
                "fontsize": "small",
                "title_fontsize": "medium",
                "markerscale": 1.5,
                "labelspacing": 0.2,
                "handlelength": 1.0,
                "handletextpad": 0.3,
                "columnspacing": 0.5,
                "bbox_to_anchor": (1.0, 1.0),
            }
        }
    )
    figure_legend_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="legend",
    ).model_copy(
        update={
            "method_kwargs": {
                "loc": "lower center",
                "ncols": 1,
                "title": "Figure legend",
                "frameon": True,
                "bbox_to_anchor": (0.5, 0.0),
            }
        }
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                line_operation,
                axes_legend_operation,
                figure_legend_operation,
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (1,))
    tool._select_step_section("method")
    loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodLegendLocCombo"
    )
    columns_edit = tool.findChild(
        QtWidgets.QSpinBox, "figureComposerAxesMethodLegendColumnsEdit"
    )
    title_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLegendTitleEdit"
    )
    assert loc_combo is not None
    assert columns_edit is not None
    assert title_edit is not None
    assert loc_combo.currentText() == "upper right"
    assert columns_edit.value() == 1
    assert title_edit.text() == "Axis legend"

    tool.operation_list.setCurrentRow(2)
    tool._select_step_section("method")
    assert "axes" not in tool.step_section_buttons
    figure_loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodLegendLocCombo"
    )
    assert figure_loc_combo is not None
    assert figure_loc_combo.currentText() == "lower center"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    axis_legend = fig.axes[0].get_legend()
    assert axis_legend is not None
    assert axis_legend.get_title().get_text() == "Axis legend"
    assert axis_legend.get_frame_on() is False
    assert [text.get_text() for text in axis_legend.get_texts()] == ["profile"]
    figure_legend = fig.legends[0]
    assert figure_legend.get_title().get_text() == "Figure legend"
    assert figure_legend.get_frame_on() is True

    code = tool.generated_code()
    assert 'axs[0, 0].legend(loc="upper right", ncols=1' in code
    assert 'title="Axis legend"' in code
    assert "bbox_to_anchor=(1.0, 1.0)" in code
    assert 'fig.legend(loc="lower center", ncols=1' in code
    assert 'title="Figure legend"' in code

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_legend() is not None
    assert namespace["fig"].legends[0].get_title().get_text() == "Figure legend"


def test_figure_composer_label_generated_code_escapes_backslashes(qtbot) -> None:
    label = "$\\Delta y$"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supylabel",
                ).model_copy(update={"method_args": (label,)}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        compiled = compile(code, "<figure-composer-generated>", "exec")

    assert not any(warning.category is SyntaxWarning for warning in caught)
    namespace: dict[str, typing.Any] = {}
    exec(compiled, namespace)  # noqa: S102
    assert namespace["fig"].get_supylabel() == label


def test_figure_composer_colorbar_method_target_policy(qtbot) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="maps",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(0.0, 1.0),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="nice_colorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    policy_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerMethodCallPolicyCombo"
    )
    assert policy_combo is not None
    assert policy_combo.currentText() == "Each selected axis"
    assert "for ax in axs.flat:" in tool.generated_code()

    _activate_combo_text(policy_combo, "Selected axes together")
    assert tool.tool_status.operations[1].method_call_policy == "ax_keyword"
    assert "eplt.nice_colorbar(ax=axs)" in tool.generated_code()


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


def test_figure_composer_bz_overlay_editor_updates_state(qtbot) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    tool._select_step_section("slice")
    page = tool.step_editor_stack.currentWidget()
    assert page is not None
    mode_combo = page.findChild(QtWidgets.QComboBox, "figureComposerBZModeCombo")
    angle_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAngleSpin")
    kz_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZKzSpin")
    kz_absolute_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZKzAbsoluteSpin"
    )
    k_parallel_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZKParallelSpin"
    )
    bounds_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZBoundsEdit")
    assert mode_combo is not None
    assert angle_spin is not None
    assert kz_spin is not None
    assert kz_absolute_spin is not None
    assert k_parallel_spin is not None
    assert bounds_edit is not None
    assert angle_spin.suffix() == "°"
    assert kz_spin.suffix() == " π/c"
    assert kz_absolute_spin.suffix() == " Å⁻¹"
    assert k_parallel_spin.suffix() == " Å⁻¹"

    _activate_combo_text(mode_combo, "Out-of-plane")
    angle_spin.setValue(30.0)
    kz_spin.setValue(0.5)
    assert np.isclose(kz_absolute_spin.value(), np.pi / 2.0)
    kz_absolute_spin.setValue(0.25)
    assert np.isclose(kz_spin.value(), 0.25 / np.pi, atol=5e-5)
    k_parallel_spin.setValue(0.25)
    bounds_edit.setText("-2, 2, -3, 3")
    bounds_edit.editingFinished.emit()

    tool._select_step_section("lattice")
    page = tool.step_editor_stack.currentWidget()
    assert page is not None
    a_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAEdit")
    b_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZBEdit")
    c_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZCEdit")
    alpha_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAlphaEdit")
    beta_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZBetaEdit")
    gamma_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZGammaEdit")
    centering_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerBZCenteringCombo"
    )
    assert a_spin is not None
    assert b_spin is not None
    assert c_spin is not None
    assert alpha_spin is not None
    assert beta_spin is not None
    assert gamma_spin is not None
    assert centering_combo is not None
    assert a_spin.suffix() == " Å"
    assert b_spin.suffix() == " Å"
    assert c_spin.suffix() == " Å"
    assert alpha_spin.suffix() == "°"
    assert beta_spin.suffix() == "°"
    assert gamma_spin.suffix() == "°"
    c_spin.setValue(4.5)
    _activate_combo_text(centering_combo, "F")

    tool._select_step_section("style")
    page = tool.step_editor_stack.currentWidget()
    assert page is not None
    color_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZColorEdit")
    line_style_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerBZLineStyleCombo"
    )
    line_width_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZLineWidthSpin"
    )
    vertices_check = page.findChild(
        QtWidgets.QCheckBox, "figureComposerBZVerticesCheck"
    )
    vertex_kw_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZVertexKwEdit")
    assert color_edit is not None
    assert line_style_combo is not None
    assert line_width_spin is not None
    assert vertices_check is not None
    assert vertex_kw_edit is not None

    color_edit.setText("tab:red")
    color_edit.editingFinished.emit()
    _activate_combo_text(line_style_combo, "-.")
    line_width_spin.setValue(1.5)
    vertices_check.setChecked(True)
    vertex_kw_edit.setText("s=12, color='tab:green'")
    vertex_kw_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.bz_mode == "out_of_plane"
    assert updated.bz_angle == 30.0
    assert np.isclose(updated.bz_kz_pi_over_c, 0.25 * 4.5 / np.pi)
    assert updated.bz_k_parallel == 0.25
    assert updated.bz_bounds == (-2.0, 2.0, -3.0, 3.0)
    assert updated.bz_c == 4.5
    assert updated.bz_centering_type == "F"
    assert updated.bz_vertices is True
    assert updated.bz_vertex_kw == {"s": 12, "color": "tab:green"}
    assert (
        figurecomposer_bz_overlay._section_summary(tool, "slice", updated)
        == "OOP, k=0.25 Å⁻¹"
    )
    assert updated.line_kw == {
        "color": "tab:red",
        "linestyle": "-.",
        "linewidth": 1.5,
    }


def test_figure_composer_bz_overlay_state_round_trip() -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        mode="out_of_plane",
    ).model_copy(
        update={
            "bz_a": 2.0,
            "bz_b": 3.0,
            "bz_c": 4.0,
            "bz_centering_type": "I",
            "bz_angle": 15.0,
            "bz_k_parallel": 0.25,
            "bz_bounds": (-1.0, 1.0, -2.0, 2.0),
            "bz_vertices": True,
            "bz_midpoint_kw": {"color": "tab:blue"},
            "line_kw": {"color": "tab:red"},
        }
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )

    restored = FigureRecipeState.model_validate(recipe.model_dump(mode="json"))

    assert restored.operations[0].kind == FigureOperationKind.BZ_OVERLAY
    assert restored.operations[0].bz_centering_type == "I"
    assert restored.operations[0].bz_bounds == (-1.0, 1.0, -2.0, 2.0)
    assert restored.operations[0].bz_midpoint_kw == {"color": "tab:blue"}
    assert restored.operations[0].line_kw == {"color": "tab:red"}


def test_figure_composer_bz_overlay_render_matches_plotting_helper(qtbot) -> None:
    bounds = (-4.0, 4.0, -4.0, 4.0)
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(
        update={
            "bz_bounds": bounds,
            "bz_angle": 45.0,
            "bz_vertices": True,
            "line_kw": {"color": "tab:purple", "linewidth": 1.25},
        }
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    bvec = erlab.lattice.to_reciprocal(erlab.lattice.abc2avec(1, 1, 1, 90, 90, 90))
    segments, vertices, _midpoints = erlab.lattice.get_in_plane_bz(
        bvec,
        kz=0.0,
        angle=45.0,
        bounds=bounds,
        return_midpoints=True,
    )
    axis = figure.axes[0]
    _assert_bz_lines_match_segments(axis.lines, segments)
    assert axis.collections
    np.testing.assert_allclose(axis.collections[0].get_offsets(), vertices)
    assert all(line.get_color() == "tab:purple" for line in axis.lines)
    assert all(line.get_linewidth() == 1.25 for line in axis.lines)


def test_figure_composer_bz_overlay_out_of_plane_render_and_code(qtbot) -> None:
    bounds = (-4.0, 4.0, -4.0, 4.0)
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        mode="out_of_plane",
    ).model_copy(
        update={
            "bz_bounds": bounds,
            "bz_k_parallel": 0.25,
            "bz_midpoints": True,
            "bz_vertex_kw": {"s": 11},
            "bz_midpoint_kw": {"s": 13},
            "line_kw": {"color": "tab:orange"},
        }
    )
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-1.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure(figsize=(4, 2))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    bvec = erlab.lattice.to_reciprocal(erlab.lattice.abc2avec(1, 1, 1, 90, 90, 90))
    segments, _vertices, midpoints = erlab.lattice.get_out_of_plane_bz(
        bvec,
        k_parallel=0.25,
        angle=0.0,
        bounds=bounds,
        return_midpoints=True,
    )
    for axis in figure.axes:
        _assert_bz_lines_match_segments(axis.lines, segments)
        assert all(line.get_color() == "tab:orange" for line in axis.lines)
        assert len(axis.collections) == 1
        np.testing.assert_allclose(axis.collections[0].get_offsets(), midpoints)
        assert axis.collections[0].get_sizes()[0] == 13

    code = tool.generated_code()
    assert "for ax in axs.flat:" in code
    namespace = _exec_generated_code(code, {"data": tool.source_data()["data"]})
    assert len(namespace["fig"].axes) == 2
    assert all(axis.lines for axis in namespace["fig"].axes)


def test_figure_composer_bz_overlay_text_parsers() -> None:
    assert figurecomposer_bz_overlay._mode_from_text("not a mode") == "in_plane"
    assert figurecomposer_bz_overlay._format_bounds(None) == ""
    assert figurecomposer_bz_overlay._bounds_from_text("") is None
    assert figurecomposer_bz_overlay._bounds_from_text("-1, 1, -2, 2") == (
        -1.0,
        1.0,
        -2.0,
        2.0,
    )

    for text in ("'not bounds'", "1, 2, 3", "1, 2, 3, object()"):
        with pytest.raises(ValueError, match="four comma-separated numbers"):
            figurecomposer_bz_overlay._bounds_from_text(text)
    with pytest.raises(ValueError, match="four comma-separated numbers"):
        figurecomposer_bz_overlay._bounds_from_text("1, 2, 3, 'bad'")

    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(update={"bz_kz_pi_over_c": 0.5, "bz_c": 4.0})
    tool = _bz_tool(operation)
    assert figurecomposer_bz_overlay._section_summary(tool, "slice", operation) == (
        f"IP, kz=0.5 π/c; {np.pi / 8:g} Å⁻¹"
    )


def test_figure_composer_bz_overlay_helper_edges(qtbot, monkeypatch) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)
    empty_tool = FigureComposerTool.from_sources(
        {"data": xr.DataArray(np.zeros((2, 2)), dims=("kx", "ky"), name="data")},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(empty_tool)
    assert (
        figurecomposer_bz_overlay._current_bz_operation(empty_tool, operation)
        is operation
    )

    nonprimitive = operation.model_copy(update={"bz_centering_type": "I"})
    assert figurecomposer_bz_overlay._bz_bvec(nonprimitive).shape == (3, 3)
    assert (
        figurecomposer_bz_overlay._single_axis_code(
            tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(expression="axs[:1]")}
            ),
        )
        is None
    )

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool(
        xr.DataArray(np.zeros((2, 2)), dims=("kx", "ky"), name="data"),
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    assert (
        figurecomposer_bz_overlay._single_axis_code(
            grid_tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(axes_ids=("axis-a", "axis-b"))}
            ),
        )
        is None
    )

    monkeypatch.setattr(
        figurecomposer_bz_overlay,
        "_axes_from_selection",
        lambda *_args, **_kwargs: (),
    )
    figurecomposer_bz_overlay._render_bz_overlay(tool, operation, None)

    tool.operation_list.setCurrentRow(0)
    created = figurecomposer_bz_overlay._create_operation(tool)
    assert created.kind == FigureOperationKind.BZ_OVERLAY
    assert figurecomposer_bz_overlay._section_summary(tool, "unknown", operation) == ""


def test_figure_composer_bz_overlay_generated_code_static_centering(qtbot) -> None:
    primitive = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(update={"bz_bounds": (-20.0, 20.0, -20.0, 20.0)})
    primitive_tool = _bz_tool(primitive)
    qtbot.addWidget(primitive_tool)

    primitive_code = primitive_tool.generated_code()
    assert "to_primitive" not in primitive_code
    primitive_namespace = _exec_generated_code(
        primitive_code, {"data": primitive_tool.source_data()["data"]}
    )
    assert primitive_namespace["fig"].axes[0].lines

    nonprimitive = primitive.model_copy(update={"bz_centering_type": "F"})
    nonprimitive_tool = _bz_tool(nonprimitive)
    qtbot.addWidget(nonprimitive_tool)

    nonprimitive_code = nonprimitive_tool.generated_code()
    assert (
        'avec = erlab.lattice.to_primitive(avec, centering_type="F")'
        in nonprimitive_code
    )
    assert 'if centering_type != "P"' not in nonprimitive_code
    assert "if " not in "\n".join(
        line for line in nonprimitive_code.splitlines() if "to_primitive" in line
    )
    nonprimitive_namespace = _exec_generated_code(
        nonprimitive_code, {"data": nonprimitive_tool.source_data()["data"]}
    )
    assert nonprimitive_namespace["fig"].axes[0].lines


def test_figure_composer_bz_overlay_invalid_axes_block_generated_code(qtbot) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 1),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="target axes"):
        tool.generated_code()


def test_figure_composer_bz_overlay_plain_momentum_seeding() -> None:
    in_plane = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 0.0, 1.0], "ky": [-2.0, -1.0, 0.0, 1.0]},
    )
    kx_kz = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kx", "kz"),
        coords={"kx": [-0.5, 0.0, 0.5], "kz": [-2.0, -1.0, 0.0, 1.0]},
    )
    ky_kz = xr.DataArray(
        np.zeros((2, 3)),
        dims=("ky", "kz"),
        coords={"ky": [0.25, 0.75], "kz": [-1.0, 0.0, 1.0]},
    )

    in_plane_operation = bz_overlay_operation_from_momentum_data(in_plane)
    kx_kz_operation = bz_overlay_operation_from_momentum_data(kx_kz)
    ky_kz_operation = bz_overlay_operation_from_momentum_data(ky_kz)

    assert in_plane_operation is not None
    assert in_plane_operation.bz_mode == "in_plane"
    assert in_plane_operation.bz_bounds == (-1.0, 1.0, -2.0, 1.0)
    assert kx_kz_operation is not None
    assert kx_kz_operation.bz_mode == "out_of_plane"
    assert kx_kz_operation.bz_bounds == (-0.5, 0.5, -2.0, 1.0)
    assert ky_kz_operation is not None
    assert ky_kz_operation.bz_mode == "out_of_plane"
    assert ky_kz_operation.bz_bounds == (0.25, 0.75, -1.0, 1.0)


def test_figure_composer_bz_overlay_seeding_edge_cases() -> None:
    no_momentum = xr.DataArray(
        np.zeros(2),
        dims=("x",),
        coords={"x": [0.0, 1.0]},
    )
    bad_coord = xr.DataArray(
        np.zeros(2),
        dims=("kx",),
        coords={"kx": ["bad", "coord"]},
    )
    nan_coord = xr.DataArray(
        np.zeros(2),
        dims=("kx",),
        coords={"kx": [np.nan, np.nan]},
    )

    assert figurecomposer_seeding._coordinate_bounds(no_momentum, "kx") is None
    assert figurecomposer_seeding._coordinate_bounds(bad_coord, "kx") is None
    assert figurecomposer_seeding._coordinate_bounds(nan_coord, "kx") is None
    assert figurecomposer_seeding._momentum_bounds(nan_coord, "kx", "ky") is None
    assert figurecomposer_seeding._representative_coord_value(no_momentum, "kx") is None
    assert figurecomposer_seeding._representative_coord_value(bad_coord, "kx") is None
    assert figurecomposer_seeding._representative_coord_value(nan_coord, "kx") is None

    in_plane_bad_bounds = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "ky"),
        coords={"kx": [np.nan, np.nan], "ky": [0.0, 1.0]},
    )
    out_of_plane_bad_bounds = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "kz"),
        coords={"kx": [np.nan, np.nan], "kz": [0.0, 1.0]},
    )
    kz_only = xr.DataArray(
        np.zeros(2),
        dims=("kz",),
        coords={"kz": [0.0, 1.0]},
    )

    assert bz_overlay_operation_from_momentum_data(in_plane_bad_bounds) is None
    assert bz_overlay_operation_from_momentum_data(no_momentum) is None
    assert bz_overlay_operation_from_momentum_data(out_of_plane_bad_bounds) is None
    assert bz_overlay_operation_from_momentum_data(kz_only) is None
    assert (
        figurecomposer_seeding._flat_ktool_out_of_plane_value(
            kz_only, FigureOperationState.bz_overlay(mode="out_of_plane")
        )
        is None
    )

    class DisabledStatus:
        bz_enabled = False

    class EnabledStatus:
        bz_enabled = True

    class DisabledKtool:
        tool_status = DisabledStatus()

    class EnabledKtool:
        tool_status = EnabledStatus()

    assert bz_overlay_operation_from_ktool(DisabledKtool(), no_momentum) is None
    assert bz_overlay_operation_from_ktool(EnabledKtool(), no_momentum) is None


def test_figure_composer_bz_overlay_ktool_seeding_snapshot() -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kz", "kx"),
        coords={"kz": [-1.0, 0.0, 1.0], "kx": [-0.5, 0.0, 0.5, 1.0], "ky": 0.25},
    )

    class FakeStatus:
        bz_enabled = True
        lattice_params = (2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        centering = "I"
        rot = 15.0
        kz = 0.5
        points = True

    class FakeKtool:
        tool_status = FakeStatus()

    operation = bz_overlay_operation_from_ktool(FakeKtool(), data)

    assert operation is not None
    assert operation.bz_mode == "out_of_plane"
    assert operation.bz_a == 2.0
    assert operation.bz_b == 3.0
    assert operation.bz_c == 4.0
    assert operation.bz_centering_type == "I"
    assert operation.bz_angle == 15.0
    assert operation.bz_kz_pi_over_c == 0.5
    assert operation.bz_k_parallel == 0.25
    assert operation.bz_vertices is True
    assert operation.bz_midpoints is True


def test_figure_composer_bz_overlay_ktool_seeding_uses_fixed_coord_median() -> None:
    ky_values = np.array(
        [
            [np.nan, -0.25, 0.0, 0.25],
            [0.5, 0.75, 1.0, 1.25],
            [1.5, 1.75, 2.0, 2.25],
        ]
    )
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kz", "kx"),
        coords={
            "kz": [-1.0, 0.0, 1.0],
            "kx": [-0.5, 0.0, 0.5, 1.0],
            "ky": (("kz", "kx"), ky_values),
        },
    )

    class FakeStatus:
        bz_enabled = True
        lattice_params = (2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        centering = "I"
        rot = 15.0
        kz = 0.5
        points = True

    class FakeKtool:
        tool_status = FakeStatus()

    operation = bz_overlay_operation_from_ktool(FakeKtool(), data)

    assert operation is not None
    assert operation.bz_mode == "out_of_plane"
    assert operation.bz_k_parallel == np.nanmedian(ky_values)


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


def test_figure_composer_photon_energy_overlay_editor_updates_state(qtbot) -> None:
    data = _photon_energy_source()
    other_data = _photon_energy_source(name="other_kconv")
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    tool = _photon_energy_tool(
        operation,
        data,
        extra_source_data={"other_kconv": other_data},
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    source_combo = next(
        (
            combo
            for combo in tool.step_source_controls.findChildren(
                QtWidgets.QComboBox, "figureComposerPhotonEnergySourceCombo"
            )
            if combo.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
        ),
        None,
    )
    assert source_combo is not None
    other_index = source_combo.findData("other_kconv")
    assert other_index >= 0
    _activate_combo_index(source_combo, other_index)

    tool._select_step_section("photon")
    page = tool.step_editor_stack.currentWidget()
    assert page is not None
    energies_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyValuesEdit"
    )
    binding_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyBindingEnergyEdit"
    )
    assert energies_edit is not None
    assert binding_edit is not None
    assert energies_edit.placeholderText() == "Enter photon energies"
    energies_edit.setText("30, 45, 60")
    energies_edit.editingFinished.emit()
    binding_edit.setText("-0.3")
    binding_edit.editingFinished.emit()

    tool._select_step_section("style")
    page = tool.step_editor_stack.currentWidget()
    assert page is not None
    color_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyColorEdit"
    )
    line_style_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerPhotonEnergyLineStyleCombo"
    )
    line_width_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerPhotonEnergyLineWidthSpin"
    )
    legend_check = page.findChild(
        QtWidgets.QCheckBox, "figureComposerPhotonEnergyLegendCheck"
    )
    legend_kw_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyLegendKwEdit"
    )
    label_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyLabelTemplateEdit"
    )
    assert color_edit is not None
    assert line_style_combo is not None
    assert line_width_spin is not None
    assert legend_check is not None
    assert legend_kw_edit is not None
    assert label_edit is not None

    color_edit.setText("tab:red")
    color_edit.editingFinished.emit()
    _activate_combo_text(line_style_combo, "--")
    line_width_spin.setValue(1.5)
    legend_check.setChecked(False)
    legend_kw_edit.setText("title='Photon energy', frameon=False")
    legend_kw_edit.editingFinished.emit()
    label_edit.setText("hv={hv:g} eV")
    label_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.hv_overlay_source == "other_kconv"
    assert updated.photon_energies == (30.0, 45.0, 60.0)
    assert updated.binding_energy == -0.3
    assert updated.show_legend is False
    assert updated.legend_kw == {"title": "Photon energy", "frameon": False}
    assert updated.label_template == "hv={hv:g} eV"
    assert tool.step_section_buttons["photon"].text() == "hν: 30, 45, 60; eV=-0.3"
    assert updated.line_kw == {
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    }


def test_figure_composer_photon_energy_overlay_state_round_trip() -> None:
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(
        update={
            "photon_energies": (30.0, 45.0, 60.0),
            "show_legend": False,
            "legend_kw": {"title": "Photon energy", "frameon": False},
            "label_template": "hv={hv:g}",
            "line_kw": {"color": "tab:green", "linewidth": 1.25},
        }
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(operation,),
        primary_source="hvdep_kconv",
    )

    restored = FigureRecipeState.model_validate(recipe.model_dump(mode="json"))

    assert restored.operations[0].kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY
    assert restored.operations[0].hv_overlay_source == "hvdep_kconv"
    assert restored.operations[0].photon_energies == (30.0, 45.0, 60.0)
    assert restored.operations[0].binding_energy == -0.3
    assert restored.operations[0].show_legend is False
    assert restored.operations[0].legend_kw == {
        "title": "Photon energy",
        "frameon": False,
    }
    assert restored.operations[0].label_template == "hv={hv:g}"
    assert restored.operations[0].line_kw == {
        "color": "tab:green",
        "linewidth": 1.25,
    }


@pytest.mark.parametrize(("configuration", "x_dim"), [(1, "kx"), (2, "ky")])
def test_figure_composer_photon_energy_overlay_render_and_code(
    qtbot, configuration: int, x_dim: str
) -> None:
    data = _photon_energy_source(configuration)
    hv_values = (30.0, 45.0, 60.0)
    binding_energy = -0.3
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=binding_energy,
    ).model_copy(
        update={
            "photon_energies": hv_values,
            "line_kw": {"color": "tab:blue", "linewidth": 1.5},
            "legend_kw": {"title": "Photon energy", "frameon": False},
        }
    )
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    expected = data.kspace.hv_to_kz(list(hv_values)).qsel(eV=binding_energy)
    axis = figure.axes[0]
    _assert_photon_energy_lines_match(axis.lines, expected, x_dim)
    assert axis.get_legend() is not None
    assert axis.get_legend().get_title().get_text() == "Photon energy"
    assert axis.get_legend().get_frame_on() is False
    assert all(line.get_color() == "tab:blue" for line in axis.lines)
    assert all(line.get_linewidth() == 1.5 for line in axis.lines)

    code = tool.generated_code()
    assert "import erlab" in code
    assert "kz_values = hvdep_kconv.kspace.hv_to_kz([30, 45, 60]).qsel(eV=-0.3)" in code
    assert "for i in range(len(kz_values.hv)):" in code
    assert f"axs[0, 0].plot(kz.{x_dim}, kz, label=rf" in code
    assert "ax.legend()" not in code
    assert 'axs[0, 0].legend(title="Photon energy", frameon=False)' in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    generated_axis = namespace["fig"].axes[0]
    _assert_photon_energy_lines_match(generated_axis.lines, expected, x_dim)
    assert generated_axis.get_legend() is not None
    assert generated_axis.get_legend().get_title().get_text() == "Photon energy"
    assert generated_axis.get_legend().get_frame_on() is False


def test_figure_composer_photon_energy_overlay_multiple_axes_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0, 45.0), "show_legend": False})
    tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(nrows=1, ncols=2),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "for ax in axs.flat:\n    for i in range(len(kz_values.hv)):" in code
    assert "        ax.plot(kz.kx, kz, label=rf" in code
    assert "legend()" not in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    for axis in namespace["fig"].axes:
        _assert_photon_energy_lines_match(axis.lines, expected, "kx")
        assert axis.get_legend() is None


def test_figure_composer_photon_energy_overlay_custom_label_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(
        update={
            "photon_energies": (30.0, 45.0),
            "label_template": "hv={hv:g} eV",
            "show_legend": False,
        }
    )
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert ".format(hv=kz.hv)" in code
    assert 'rf"$h\\nu' not in code
    assert "legend()" not in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    axis = namespace["fig"].axes[0]
    assert axis.get_legend() is None
    assert [line.get_label() for line in axis.lines] == [
        f"hv={expected.isel(hv=index).hv:g} eV" for index in range(expected.sizes["hv"])
    ]


def test_figure_composer_photon_energy_overlay_expression_axes_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(expression="axs[0, :]"),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0, 45.0)})
    tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(nrows=1, ncols=2),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "for ax in axs[0, :]:\n    for i in range(len(kz_values.hv)):" in code
    assert "    ax.legend()" in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    for axis in namespace["fig"].axes:
        _assert_photon_energy_lines_match(axis.lines, expected, "kx")
        assert axis.get_legend() is not None


def test_figure_composer_photon_energy_overlay_blocks_invalid_inputs(qtbot) -> None:
    data = _photon_energy_source()
    missing_values = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    )
    missing_values_tool = _photon_energy_tool(missing_values, data)
    qtbot.addWidget(missing_values_tool)
    with pytest.raises(ValueError, match="at least one photon energy"):
        missing_values_tool.generated_code()

    unselected_energy = missing_values.model_copy(
        update={"photon_energies": (30.0,), "binding_energy": None}
    )
    unselected_tool = _photon_energy_tool(unselected_energy, data)
    qtbot.addWidget(unselected_tool)
    with pytest.raises(ValueError, match="Select a binding energy"):
        unselected_tool.generated_code()

    invalid_axes = missing_values.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 1),))}
    )
    invalid_axes_tool = _photon_energy_tool(invalid_axes, data)
    qtbot.addWidget(invalid_axes_tool)
    with pytest.raises(ValueError, match="target axes"):
        invalid_axes_tool.generated_code()

    invalid_source = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-1.0, 1.0]},
        name="hvdep_kconv",
    )
    invalid_source_operation = missing_values.model_copy(
        update={"photon_energies": (30.0,)}
    )
    invalid_source_tool = _photon_energy_tool(invalid_source_operation, invalid_source)
    qtbot.addWidget(invalid_source_tool)
    with pytest.raises(ValueError, match="kx-kz or ky-kz"):
        invalid_source_tool.generated_code()

    no_source_operation = missing_values.model_copy(update={"hv_overlay_source": None})
    no_source_tool = _photon_energy_tool(no_source_operation, data)
    qtbot.addWidget(no_source_tool)
    with pytest.raises(ValueError, match="Select a source"):
        no_source_tool.generated_code()

    missing_source_operation = missing_values.model_copy(
        update={"hv_overlay_source": "missing"}
    )
    missing_source_tool = _photon_energy_tool(missing_source_operation, data)
    qtbot.addWidget(missing_source_tool)
    with pytest.raises(ValueError, match="Source 'missing' is missing"):
        missing_source_tool.generated_code()

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(
        missing_values_tool, figure, sync_visible=False
    )
    assert (
        "at least one photon energy"
        in missing_values_tool._operation_render_errors[missing_values.operation_id]
    )


def test_figure_composer_photon_energy_overlay_helper_edges(qtbot, monkeypatch) -> None:
    assert figurecomposer_photon_energy._optional_float_from_text("") is None
    assert figurecomposer_photon_energy._optional_float_from_text(" -0.3 ") == -0.3
    with pytest.raises(ValueError, match="one number"):
        figurecomposer_photon_energy._optional_float_from_text("bad")
    assert figurecomposer_photon_energy._photon_energies_from_text("30, 45") == (
        30.0,
        45.0,
    )

    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source=None,
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    assert (
        figurecomposer_photon_energy._section_summary(tool, "sources", operation)
        == "none"
    )
    with pytest.raises(ValueError, match="Select a source"):
        figurecomposer_photon_energy._source_data(tool, operation)

    kz_values = xr.DataArray(
        np.zeros((2, 3)),
        dims=("hv", "angle"),
        coords={"hv": [30.0, 45.0], "angle": [0.0, 1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="kx-kz or ky-kz"):
        figurecomposer_photon_energy._photon_x_dim(data, kz_values)

    class FakeKspace:
        slit_axis = "kx"

        @staticmethod
        def hv_to_kz(values: Sequence[float]) -> xr.DataArray:
            return xr.DataArray(
                np.zeros((len(values), 2, 2)),
                dims=("hv", "kx", "temperature"),
                coords={
                    "hv": list(values),
                    "kx": [-1.0, 1.0],
                    "temperature": [20.0, 30.0],
                },
            )

    class FakeKParallelKzData:
        dims = ("kx", "kz")
        kspace = FakeKspace()

        def squeeze(self, *, drop: bool):
            return self

    extra_dim_operation = operation.model_copy(update={"binding_energy": None})
    with pytest.raises(ValueError, match="hv and one momentum dimension"):
        figurecomposer_photon_energy._kz_values(
            FakeKParallelKzData(), extra_dim_operation
        )

    assert (
        figurecomposer_photon_energy._single_axis_code(
            tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(expression="axs[:1]")}
            ),
        )
        is None
    )

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(
            layout_mode="gridspec",
            gridspec=FigureGridSpecLayoutState(root=root),
        ),
    )
    qtbot.addWidget(grid_tool)
    assert (
        figurecomposer_photon_energy._single_axis_code(
            grid_tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(axes_ids=("axis-a", "axis-b"))}
            ),
        )
        is None
    )

    render_operation = operation.model_copy(update={"hv_overlay_source": "hvdep_kconv"})
    monkeypatch.setattr(
        figurecomposer_photon_energy,
        "_axes_from_selection",
        lambda *_args, **_kwargs: (),
    )
    render_tool = _photon_energy_tool(render_operation, data)
    qtbot.addWidget(render_tool)
    figurecomposer_photon_energy._render_photon_energy_overlay(
        render_tool, render_operation, None
    )


def test_figure_composer_photon_energy_overlay_add_step_seeds_source_and_binding(
    qtbot,
) -> None:
    data = _photon_energy_source()
    plot_operation = FigureOperationState.plot_slices(
        label="hvdep",
        sources=("hvdep_kconv",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(-0.3,),
    )
    tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(plot_operation,),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    action = next(
        action
        for action in tool.add_step_menu.actions()
        if action.data() == FigureOperationKind.PHOTON_ENERGY_OVERLAY.value
    )
    action.trigger()

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY
    assert operation.hv_overlay_source == "hvdep_kconv"
    assert operation.binding_energy == -0.3
    assert operation.photon_energies == ()


def test_figure_composer_photon_energy_overlay_seed_fallbacks(qtbot) -> None:
    data = _photon_energy_source()
    unused_data = _photon_energy_source(name="unused")
    tool = FigureComposerTool.from_sources(
        {"unused": unused_data, "hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="unused",
    )
    qtbot.addWidget(tool)

    assert figurecomposer_photon_energy._seed_source(tool) == "hvdep_kconv"
    assert figurecomposer_photon_energy._seed_binding_energy(tool) is None

    primary_tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(primary_tool)
    assert figurecomposer_photon_energy._seed_source(primary_tool) == "hvdep_kconv"

    line_operation = FigureOperationState.line(
        label="line", source="hvdep_kconv", axes=FigureAxesSelectionState()
    )
    line_tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(line_operation,),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(line_tool)
    line_tool.operation_list.setCurrentRow(0)

    assert figurecomposer_photon_energy._seed_source(line_tool) == "hvdep_kconv"
    assert figurecomposer_photon_energy._seed_binding_energy(line_tool) is None


def test_figure_composer_photon_energy_overlay_tracks_source_ownership(
    qtbot,
) -> None:
    data = _photon_energy_source()
    existing_data = _photon_energy_source(name="hvdep_kconv")
    unused_data = _photon_energy_source(name="unused")
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    source_tool = _photon_energy_tool(
        operation,
        data,
        extra_source_data={"unused": unused_data},
    )
    destination = _photon_energy_tool(
        FigureOperationState.photon_energy_overlay(
            source="hvdep_kconv",
            axes=FigureAxesSelectionState(axes=((0, 0),)),
            binding_energy=-0.3,
        ).model_copy(update={"photon_energies": (45.0,)}),
        existing_data,
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    assert not source_tool._source_removable("hvdep_kconv")
    assert source_tool._source_removable("unused")

    _select_operation_rows(source_tool, (0,))
    source_tool.copy_operation_button.click()
    payload = source_tool._clipboard_step_payload()
    assert payload is not None
    _operations, sources, source_data = payload
    assert [source.name for source in sources] == ["hvdep_kconv"]
    xr.testing.assert_identical(source_data["hvdep_kconv"], data)

    destination._paste_operations_from_clipboard()

    pasted = destination.tool_status.operations[-1]
    assert pasted.hv_overlay_source == "hvdep_kconv_copy"
    assert [source.name for source in destination.tool_status.sources] == [
        "hvdep_kconv",
        "hvdep_kconv_copy",
    ]
    xr.testing.assert_identical(destination.source_data()["hvdep_kconv"], existing_data)
    xr.testing.assert_identical(destination.source_data()["hvdep_kconv_copy"], data)


def test_figure_composer_erlab_method_controls_update_recipe(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    axes = FigureAxesSelectionState(axes=((0, 0), (0, 1)))
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="label_subplots",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="label_subplot_properties",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="nice_colorbar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="proportional_colorbar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="fermiline",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="mark_points",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="scale_units",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="fancy_labels",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="integer_ticks",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="sizebar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="unify_clim",
                    axes=axes,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    def select_method(row: int) -> QtWidgets.QWidget:
        tool.operation_list.setCurrentRow(row)
        tool._select_step_section("method")
        page = tool.step_editor_stack.currentWidget()
        assert page is not None
        return page

    def line_edit(page: QtWidgets.QWidget, name: str) -> QtWidgets.QLineEdit:
        widget = page.findChild(QtWidgets.QLineEdit, name)
        assert widget is not None
        return widget

    def combo_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QComboBox:
        widget = page.findChild(QtWidgets.QComboBox, name)
        assert widget is not None
        return widget

    def spin_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QSpinBox:
        widget = page.findChild(QtWidgets.QSpinBox, name)
        assert widget is not None
        return widget

    def double_spin_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QDoubleSpinBox:
        widget = page.findChild(QtWidgets.QDoubleSpinBox, name)
        assert widget is not None
        return widget

    def set_line_edit(page: QtWidgets.QWidget, name: str, text: str) -> None:
        edit = line_edit(page, name)
        edit.setText(text)
        edit.editingFinished.emit()

    def set_combo(page: QtWidgets.QWidget, name: str, text: str) -> None:
        _activate_combo_text(combo_box(page, name), text)

    def operation(row: int) -> FigureOperationState:
        return tool.tool_status.operations[row]

    page = select_method(0)
    set_combo(page, "figureComposerERLabCleanLabelsRemoveInnerTicksCombo", "True")
    assert operation(0).method_args == (True,)

    page = select_method(1)
    spin_box(page, "figureComposerERLabLabelSubplotsStartEdit").setValue(3)
    set_combo(page, "figureComposerERLabLabelSubplotsOrderCombo", "F")
    set_combo(page, "figureComposerERLabLabelSubplotsLocCombo", "lower right")
    set_line_edit(page, "figureComposerERLabLabelSubplotsOffsetEdit", "1, 2")
    set_line_edit(page, "figureComposerERLabLabelSubplotsPrefixEdit", "(")
    set_line_edit(page, "figureComposerERLabLabelSubplotsSuffixEdit", ")")
    set_combo(page, "figureComposerERLabLabelSubplotsNumericCombo", "True")
    set_combo(page, "figureComposerERLabLabelSubplotsCapitalCombo", "True")
    set_combo(page, "figureComposerERLabLabelSubplotsFontWeightCombo", "bold")
    set_line_edit(page, "figureComposerERLabLabelSubplotsFontSizeEdit", "8")
    assert operation(1).method_kwargs == {
        "startfrom": 3,
        "order": "F",
        "loc": "lower right",
        "offset": (1.0, 2.0),
        "prefix": "(",
        "suffix": ")",
        "numeric": True,
        "capital": True,
        "fontweight": "bold",
        "fontsize": 8,
    }

    page = select_method(2)
    set_line_edit(page, "figureComposerERLabLabelPropertiesValuesEdit", "eV=[0, 1]")
    set_line_edit(page, "figureComposerERLabLabelPropertiesDecimalsEdit", "2")
    spin_box(page, "figureComposerERLabLabelPropertiesSiEdit").setValue(-3)
    set_line_edit(page, "figureComposerERLabLabelPropertiesNameEdit", "Energy")
    set_line_edit(page, "figureComposerERLabLabelPropertiesUnitEdit", "eV")
    set_combo(page, "figureComposerERLabLabelPropertiesOrderCombo", "F")
    set_combo(page, "figureComposerERLabLabelPropertiesLocCombo", "lower left")
    set_line_edit(page, "figureComposerERLabLabelPropertiesOffsetEdit", "3, 4")
    set_line_edit(page, "figureComposerERLabLabelPropertiesPrefixEdit", "[")
    set_line_edit(page, "figureComposerERLabLabelPropertiesSuffixEdit", "]")
    set_combo(page, "figureComposerERLabLabelPropertiesFontWeightCombo", "bold")
    set_line_edit(page, "figureComposerERLabLabelPropertiesFontSizeEdit", "9")
    assert operation(2).method_args == ({"eV": [0, 1]},)
    assert operation(2).method_kwargs == {
        "decimals": 2,
        "si": -3,
        "name": "Energy",
        "unit": "eV",
        "order": "F",
        "loc": "lower left",
        "offset": (3.0, 4.0),
        "prefix": "[",
        "suffix": "]",
        "fontweight": "bold",
        "fontsize": 9,
    }

    page = select_method(3)
    double_spin_box(page, "figureComposerERLabNiceColorbarWidthEdit").setValue(10.0)
    double_spin_box(page, "figureComposerERLabNiceColorbarAspectEdit").setValue(4.0)
    double_spin_box(page, "figureComposerERLabNiceColorbarPadEdit").setValue(2.0)
    set_combo(page, "figureComposerERLabNiceColorbarMinMaxCombo", "True")
    set_combo(page, "figureComposerERLabNiceColorbarOrientationCombo", "horizontal")
    set_combo(page, "figureComposerERLabNiceColorbarFloatingCombo", "True")
    set_line_edit(page, "figureComposerERLabNiceColorbarTicksEdit", "0, 0.5, 1")
    set_line_edit(
        page, "figureComposerERLabNiceColorbarTickLabelsEdit", "low, mid, high"
    )
    assert operation(3).method_kwargs == {
        "width": 10.0,
        "aspect": 4.0,
        "pad": 2.0,
        "minmax": True,
        "orientation": "horizontal",
        "floating": True,
        "ticks": (0, 0.5, 1),
        "ticklabels": ("low", "mid", "high"),
    }

    page = select_method(4)
    spin_box(page, "figureComposerERLabProportionalColorbarIndexEdit").setValue(0)
    set_combo(page, "figureComposerERLabProportionalColorbarImageOnlyCombo", "True")
    set_line_edit(page, "figureComposerERLabProportionalColorbarTicksEdit", "[0, 1]")
    assert operation(4).method_kwargs == {
        "index": 0,
        "image_only": True,
        "ticks": [0, 1],
    }

    page = select_method(5)
    set_combo(page, "figureComposerERLabSetTitlesOrderCombo", "F")
    assert operation(5).method_kwargs == {"order": "F"}

    page = select_method(6)
    double_spin_box(page, "figureComposerERLabFermilineValueEdit").setValue(0.1)
    set_combo(page, "figureComposerERLabFermilineOrientationCombo", "v")
    assert operation(6).method_kwargs == {"value": 0.1, "orientation": "v"}
    set_line_edit(page, "figureComposerERLabFermilineColorEdit", "tab:red")
    set_combo(page, "figureComposerERLabFermilineLineStyleCombo", "--")
    set_line_edit(page, "figureComposerERLabFermilineLineWidthEdit", "1.5")
    assert operation(6).method_kwargs == {
        "value": 0.1,
        "orientation": "v",
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    }

    page = select_method(7)
    set_line_edit(page, "figureComposerERLabMarkPointsPointsEdit", "0, 1")
    set_line_edit(page, "figureComposerERLabMarkPointsLabelsEdit", "G, M")
    set_line_edit(page, "figureComposerERLabMarkPointsYEdit", "0.25, 0.5")
    set_line_edit(page, "figureComposerERLabMarkPointsPadEdit", "1, 2")
    set_combo(page, "figureComposerERLabMarkPointsLiteralCombo", "True")
    set_combo(page, "figureComposerERLabMarkPointsRomanCombo", "False")
    set_combo(page, "figureComposerERLabMarkPointsBarCombo", "True")
    assert operation(7).method_args == ((0, 1), ("G", "M"))
    assert operation(7).method_kwargs == {
        "y": (0.25, 0.5),
        "pad": (1.0, 2.0),
        "literal": True,
        "roman": False,
        "bar": True,
    }

    page = select_method(8)
    set_combo(page, "figureComposerERLabScaleUnitsAxisCombo", "y")
    spin_box(page, "figureComposerERLabScaleUnitsSiEdit").setValue(3)
    set_combo(page, "figureComposerERLabScaleUnitsPrefixCombo", "False")
    set_combo(page, "figureComposerERLabScaleUnitsPowerCombo", "True")
    assert operation(8).method_args == ("y", 3)
    assert operation(8).method_kwargs == {"prefix": False, "power": True}

    page = select_method(9)
    set_combo(page, "figureComposerERLabFancyLabelsRadiansCombo", "True")
    assert operation(9).method_kwargs == {"radians": True}

    page = select_method(10)
    assert (
        page.findChild(QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit") is None
    )

    page = select_method(11)
    assert operation(11).method_kwargs == {}
    assert double_spin_box(page, "figureComposerERLabSizebarValueEdit").value() == 1.0
    assert line_edit(page, "figureComposerERLabSizebarUnitEdit").text() == "m"
    double_spin_box(page, "figureComposerERLabSizebarValueEdit").setValue(2.0)
    set_line_edit(page, "figureComposerERLabSizebarUnitEdit", "m")
    spin_box(page, "figureComposerERLabSizebarSiEdit").setValue(-6)
    double_spin_box(page, "figureComposerERLabSizebarResolutionEdit").setValue(0.001)
    spin_box(page, "figureComposerERLabSizebarDecimalsEdit").setValue(1)
    set_line_edit(page, "figureComposerERLabSizebarLabelEdit", "200 um")
    set_combo(page, "figureComposerERLabSizebarLocCombo", "lower left")
    double_spin_box(page, "figureComposerERLabSizebarPadEdit").setValue(0.2)
    double_spin_box(page, "figureComposerERLabSizebarBorderPadEdit").setValue(0.6)
    double_spin_box(page, "figureComposerERLabSizebarSepEdit").setValue(4.0)
    set_combo(page, "figureComposerERLabSizebarFrameCombo", "True")
    assert operation(11).method_kwargs == {
        "value": 2.0,
        "unit": "m",
        "si": -6,
        "resolution": 0.001,
        "decimals": 1,
        "label": "200 um",
        "loc": "lower left",
        "pad": 0.2,
        "borderpad": 0.6,
        "sep": 4.0,
        "frameon": True,
    }

    page = select_method(12)
    set_combo(page, "figureComposerERLabUnifyClimImageOnlyCombo", "True")
    set_combo(page, "figureComposerERLabUnifyClimAutoscaleCombo", "True")
    set_line_edit(page, "figureComposerERLabUnifyClimVminEdit", "0")
    set_line_edit(page, "figureComposerERLabUnifyClimVmaxEdit", "1")
    assert operation(12).method_kwargs == {
        "image_only": True,
        "autoscale": True,
        "vmin": 0.0,
        "vmax": 1.0,
    }

    code = tool.generated_code()
    assert "ticks=(0, 0.5, 1)" in code
    assert "ticks=[0, 1]" in code
    assert "eplt.fancy_labels(axs, radians=True)" in code
    assert "eplt.integer_ticks(axs)" in code
    assert "eplt.label_subplot_properties(" in code
    assert 'loc="lower left"' in code
    assert "offset=(3.0, 4.0)" in code
    assert 'prefix="["' in code
    assert 'suffix="]"' in code
    assert 'fontweight="bold"' in code
    assert "fontsize=9" in code
    assert "eplt.sizebar(" in code
    assert "value=2.0" in code
    assert 'unit="m"' in code
    assert "eplt.unify_clim(" in code
    assert "image_only=True" in code
    assert "autoscale=True" in code


def test_figure_composer_line_values_axis_swaps_regular_profile(qtbot) -> None:
    profile = xr.DataArray(
        np.array([2.0, 4.0, 8.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_values_axis": "x", "gradient": True})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile.values)
    np.testing.assert_allclose(line.get_ydata(), profile["kx"].values)
    assert len(fig.axes[0].images) == 1
    np.testing.assert_allclose(
        fig.axes[0].images[0].cmap(1.0),
        mcolors.to_rgba(line.get_color()),
        rtol=1e-7,
    )

    code = tool.generated_code()
    assert "import erlab.plotting as eplt" in code
    assert "_line = axs[0, 0].plot(profile, profile['kx'])[0]" in code
    assert "transpose=True" in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "    _line = axs[0, 0].plot(profile, profile['kx'])[0]"
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile.values)
    np.testing.assert_allclose(line.get_ydata(), profile["kx"].values)
    assert len(namespace["fig"].axes[0].images) == 1
    np.testing.assert_allclose(
        namespace["fig"].axes[0].images[0].cmap(1.0),
        mcolors.to_rgba(line.get_color()),
        rtol=1e-7,
    )


def test_figure_composer_line_mean_normalization_executes(qtbot) -> None:
    profile = xr.DataArray(
        np.array([2.0, 4.0, 6.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_normalize": "mean"})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    expected = profile / profile.mean(skipna=True)
    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), expected.values)

    code = tool.generated_code()
    assert "profile.mean(skipna=True)" in code
    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), expected.values)


def test_figure_composer_line_normalization_reports_zero_scale(
    qtbot, monkeypatch
) -> None:
    profile = xr.DataArray(
        np.array([0.0, 0.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_normalize": "max"})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    error_text = tool._operation_render_errors[operation.operation_id]
    assert "Cannot normalize profile by max" in error_text
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        tool.generated_code()

    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog:
        def __init__(self, parent=None, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            dialogs.append(self)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )
    tool.copy_code()
    assert dialogs
    assert dialogs[0].kwargs["title"] == "Cannot Copy Figure Code"
    assert "Cannot normalize profile by max" in dialogs[0].kwargs["text"]
    assert "Traceback" in dialogs[0].kwargs["detailed_text"]


def test_figure_composer_line_profile_helper_contracts(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("cut", "kx"),
        coords={
            "cut": [0.0, 1.0],
            "kx": [-1.0, 0.0, 1.0],
            "temperature": ("cut", [20.0, 30.0]),
            "signal": (("cut", "kx"), np.arange(6.0).reshape(2, 3) + 10.0),
        },
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_y": "signal",
            "line_iter_dim": "cut",
            "line_selection": {"kx": slice(-1.0, 1.0)},
            "line_values_axis": "x",
            "line_labels": ("a", "b"),
            "line_colors": ("red", "blue"),
            "line_kw": {"lw": 2.0, "c": "black", "marker": "o"},
            "xlim": (-2.0, 2.0),
            "ylim": (0.0, 20.0),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)

    figurecomposer_line_profile._line_limit_update_callback(tool, "xlim")("0, 1")
    assert tool.tool_status.operations[0].xlim == (0.0, 1.0)
    figurecomposer_line_profile._line_limit_update_callback(tool, "ylim")("")
    assert tool.tool_status.operations[0].ylim is None

    assert (
        figurecomposer_line_profile._line_placement_text("one_per_axis")
        == "One profile per axis"
    )
    assert (
        figurecomposer_line_profile._line_placement_from_text("One profile per axis")
        == "one_per_axis"
    )
    assert (
        figurecomposer_line_profile._line_placement_from_text("anything else")
        == "all_axes"
    )
    assert (
        figurecomposer_line_profile._line_choice_data(
            tool, operation.model_copy(update={"line_source": None}), values=False
        )
        is None
    )
    assert (
        figurecomposer_line_profile._available_line_value_names(
            tool, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    assert set(
        figurecomposer_line_profile._available_line_value_names(tool, operation)
    ) >= {"cut", "kx", "temperature", "signal"}
    assert figurecomposer_line_profile._available_line_coordinate_names(
        tool, operation
    ) == ["kx"]
    assert figurecomposer_line_profile._available_line_offset_coords(
        tool, operation
    ) == ["temperature"]

    profiles = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(profiles) == 2
    assert all(profile.dims == ("kx",) for profile in profiles)
    selected_operation = operation.model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="profile", qsel={"cut": 0.0}),
            ),
            "line_labels": (),
        }
    )
    assert (
        len(figurecomposer_line_profile._line_data_items(tool, selected_operation)) == 1
    )
    assert (
        figurecomposer_line_profile._line_data_items(
            tool, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    assert (
        figurecomposer_line_profile._line_data_items(
            tool, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_line_profile._line_coordinate(data, None)

    assert figurecomposer_line_profile._line_text_values((), 0, default=None) == ()
    assert figurecomposer_line_profile._line_text_values(
        ("shared",), 2, default=None
    ) == (
        "shared",
        "shared",
    )
    with pytest.raises(ValueError, match="one value or one per profile"):
        figurecomposer_line_profile._line_text_values(("a", "b", "c"), 2, default=None)
    assert figurecomposer_line_profile._line_profile_style_kwargs(operation) == {
        "linewidth": 2.0,
        "marker": "o",
    }

    loop_names = ["profile"]
    loop_values = ["profiles"]
    style_lines, kwargs_text = figurecomposer_line_profile._line_style_code(
        operation,
        profiles=profiles,
        sources=("profile", "profile"),
        loop_names=loop_names,
        loop_values=loop_values,
    )
    assert loop_names == ["profile", "label", "color"]
    assert loop_values == ["profiles", "['a', 'b']", "['red', 'blue']"]
    assert style_lines == []
    assert "linewidth=2.0" in kwargs_text
    assert "label=label" in kwargs_text
    assert "color=color" in kwargs_text

    assert (
        figurecomposer_line_profile._line_code(
            tool, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    selection_lines = figurecomposer_line_profile._line_code(tool, selected_operation)
    assert selection_lines[0] == "profiles = ["
    assert any(".qsel(cut=0.0)" in line for line in selection_lines)
    one_per_axis_lines = figurecomposer_line_profile._line_code(
        tool, operation.model_copy(update={"line_placement": "one_per_axis"})
    )
    assert not any("if len(target_axes)" in line for line in one_per_axis_lines)
    assert not any(line.startswith("target_axes =") for line in one_per_axis_lines)
    assert any(
        "axs[0, 0].plot(profile, profile['kx']" in line for line in one_per_axis_lines
    )
    assert one_per_axis_lines[-1] == "axs[0, 0].set(xlim=(-2.0, 2.0), ylim=(0.0, 20.0))"


def test_figure_composer_profile_lines_support_per_profile_style_and_offsets(
    qtbot,
) -> None:
    profile_data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("cut", "kx"),
        coords={
            "cut": [0.0, 1.0, 2.0],
            "kx": [-1.0, 0.0, 1.0, 2.0],
            "temperature": ("cut", [10.0, 20.0, 30.0]),
        },
        name="profile_data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="profile_data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_labels": ("a", "b", "c"),
            "line_colors": ("red", "green", "blue"),
            "line_kw": {
                "linestyle": "--",
                "linewidth": 1.5,
                "marker": "o",
                "markersize": 6.0,
                "markerfacecolor": "yellow",
                "markeredgecolor": "black",
            },
            "line_offset_source": "associated",
            "line_offset_coord": "temperature",
            "line_offset_scale": 0.01,
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        profile_data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(tool)
    profiles = figurecomposer_line_profile._line_data_items(tool, operation)

    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    reduce_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileReduceCombo"
    )
    assert reduce_combo is not None
    assert reduce_combo.currentText() == "Disabled"
    assert (
        selection_page.findChild(
            QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
        )
        is None
    )
    _activate_combo_text(reduce_combo, "Both")
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
            )
            is not None
        ),
        timeout=1000,
    )
    selection_page = tool.step_editor_stack.currentWidget()
    coarsen_spin = selection_page.findChild(
        QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
    )
    thin_spin = selection_page.findChild(
        QtWidgets.QSpinBox, "figureComposerProfileReduceThinSpin"
    )
    assert tool.tool_status.operations[0].line_reduce == "both"
    assert coarsen_spin is not None
    assert coarsen_spin.value() == 2
    assert thin_spin is not None
    thin_spin.setValue(3)
    assert tool.tool_status.operations[0].line_reduce_thin == 3
    tool._replace_operation(0, operation)
    tool._select_step_section("other")
    other_page = tool.step_editor_stack.currentWidget()
    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is not None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is not None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )
    tool._select_step_section("style")
    style_page = tool.step_editor_stack.currentWidget()
    line_style_combo = style_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineStyleCombo"
    )
    line_width_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineWidthSpin"
    )
    marker_combo = style_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineMarkerCombo"
    )
    marker_size_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineMarkerSizeSpin"
    )
    marker_face_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineMarkerFaceColorEdit"
    )
    marker_edge_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineMarkerEdgeColorEdit"
    )
    gradient_check = style_page.findChild(
        QtWidgets.QCheckBox, "figureComposerLineGradientCheck"
    )
    marker_face_button = style_page.findChild(
        figurecomposer_widgets._ColorPickerButton,
        "figureComposerLineMarkerFaceColorButton",
    )
    assert line_style_combo is not None
    assert line_style_combo.currentText() == "--"
    assert line_width_spin is not None
    assert line_width_spin.value() == 1.5
    assert marker_combo is not None
    assert marker_combo.currentText() == "o"
    assert marker_size_spin is not None
    assert marker_size_spin.value() == 6.0
    assert marker_face_edit is not None
    assert marker_face_edit.text() == "yellow"
    assert marker_face_button is not None
    assert marker_edge_edit is not None
    assert marker_edge_edit.text() == "black"
    assert gradient_check is not None
    assert gradient_check.isChecked()

    color_text_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorsEdit"
    )
    first_color_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorItemEdit_0"
    )
    assert color_text_edit is not None
    assert color_text_edit.text() == "red, green, blue"
    assert first_color_edit is not None
    first_color_edit.setText("tab:blue")
    first_color_edit.setModified(True)
    first_color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].line_colors == (
        "tab:blue",
        "green",
        "blue",
    )
    assert color_text_edit.text() == "tab:blue, green, blue"
    color_text_edit.setText("C0, C1, C2")
    color_text_edit.setModified(True)
    color_text_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].line_colors == ("C0", "C1", "C2")

    operation = tool.tool_status.operations[0]

    tool._select_step_section("other")
    other_page = tool.step_editor_stack.currentWidget()
    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    _activate_combo_text(offset_source_combo, "manual")
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
            )
            is None
        ),
        timeout=1000,
    )
    other_page = tool.step_editor_stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "manual"
    assert tool.tool_status.operations[0].line_offset_scale == 1.0
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is not None
    )

    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    _activate_combo_text(offset_source_combo, "index")
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    other_page = tool.step_editor_stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "index"
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is not None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )
    tool._replace_operation(0, operation)

    assert figurecomposer_line_profile._available_line_offset_coords(
        tool, operation
    ) == ["temperature"]
    assert _line_transform.line_offsets_for_profiles(
        operation.model_copy(
            update={"line_offset_source": "index", "line_offset_scale": 2.0}
        ),
        profiles,
    ) == (0.0, 2.0, 4.0)
    assert _line_transform.line_offsets_for_profiles(
        operation.model_copy(
            update={"line_offset_source": "coordinate", "line_offset_scale": 0.5}
        ),
        profiles,
    ) == (0.0, 0.5, 1.0)
    assert _line_transform.line_offsets_for_profiles(operation, profiles) == (
        0.1,
        0.2,
        0.3,
    )

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert fig.axes[0].get_legend() is None
    assert len(fig.axes[0].images) == 3
    for index, line in enumerate(fig.axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("C0", "C1", "C2")[index]
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 1.5
        assert line.get_marker() == "o"
        assert line.get_markersize() == 6.0
        assert line.get_markerfacecolor() == "yellow"
        assert line.get_markeredgecolor() == "black"
        np.testing.assert_allclose(
            fig.axes[0].images[index].cmap(1.0),
            mcolors.to_rgba(line.get_color()),
            rtol=1e-7,
        )

    namespace: dict[str, typing.Any] = {"profile_data": profile_data}
    code = tool.generated_code()
    assert "import erlab.plotting as eplt" in code
    assert "ax.legend()" not in code
    assert "profile_offsets =" not in code
    assert "0.01 * profile_data['temperature'] + profile_data" in code
    assert "for ax in" not in code
    assert "_line = axs[0, 0].plot(profile['kx'], profile" in code
    assert "for _line in" not in code
    assert "ax.lines" not in code
    code_lines = code.splitlines()
    plot_index = next(
        index
        for index, line in enumerate(code_lines)
        if "_line = axs[0, 0].plot(" in line
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes[0].images) == 3
    for index, line in enumerate(namespace["fig"].axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("C0", "C1", "C2")[index]
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 1.5
        assert line.get_marker() == "o"
        assert line.get_markersize() == 6.0
        assert line.get_markerfacecolor() == "yellow"
        assert line.get_markeredgecolor() == "black"
        np.testing.assert_allclose(
            namespace["fig"].axes[0].images[index].cmap(1.0),
            mcolors.to_rgba(line.get_color()),
            rtol=1e-7,
        )

    shared_label_operation = operation.model_copy(
        update={"line_labels": ("shared",), "line_colors": ()}
    )
    shared_label_tool = FigureComposerTool(
        profile_data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(shared_label_operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(shared_label_tool)

    shared_fig = shared_label_tool.figure
    figurecomposer_rendering._render_into_figure(
        shared_label_tool, shared_fig, sync_visible=False
    )
    assert [line.get_label() for line in shared_fig.axes[0].lines] == [
        "shared",
        "shared",
        "shared",
    ]

    namespace = {"profile_data": profile_data}
    shared_code = shared_label_tool.generated_code()
    assert "profile_label =" not in shared_code
    exec(shared_code, namespace)  # noqa: S102
    assert [line.get_label() for line in namespace["fig"].axes[0].lines] == [
        "shared",
        "shared",
        "shared",
    ]


def test_figure_composer_line_label_placeholders_render_and_codegen(qtbot) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_label_text": r"$k_{F}$, $E-E_F = {eV:g}$ eV",
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_labels = [
        r"$k_{F}$, $E-E_F = -0.1$ eV",
        r"$k_{F}$, $E-E_F = 0.2$ eV",
    ]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [line.get_label() for line in tool.figure.axes[0].lines] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert 'profile.coords["eV"].values.item()' in code
    assert "profile_labels" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        line.get_label() for line in namespace["fig"].axes[0].lines
    ] == expected_labels


def test_figure_composer_line_label_missing_placeholder_errors(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_label_text": "eV={eV:g}, temperature={temperature:g}",
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    render_error = tool._operation_render_errors[operation.operation_id]
    assert "temperature" in render_error
    assert "Available placeholders" in render_error
    with pytest.raises(ValueError, match=r"temperature.*Available placeholders"):
        tool.generated_code()


def test_figure_composer_line_profile_coordinate_colormap_render_and_codegen(
    qtbot,
) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_color_mode": "coordinate",
            "line_color_cmap": "plasma",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.2,
            "line_colors": ("black",),
            "line_kw": {"color": "red", "linestyle": "--"},
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_colors = _expected_line_colormap_colors(eV, "plasma", trim=(0.1, 0.2))
    assert figurecomposer_line_profile._available_line_color_coords(
        tool, operation
    ) == ["eV"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    rendered_lines = tool.figure.axes[0].lines
    assert [line.get_linestyle() for line in rendered_lines] == ["--", "--"]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in rendered_lines]),
        expected_colors,
    )

    tool._select_step_section("style")
    style_page = tool.step_editor_stack.currentWidget()
    assert (
        style_page.findChild(QtWidgets.QComboBox, "figureComposerLineColorModeCombo")
        is not None
    )
    assert (
        style_page.findChild(QtWidgets.QComboBox, "figureComposerLineColorCoordCombo")
        is not None
    )
    assert (
        style_page.findChild(
            erlab.interactive.colors.ColorMapComboBox,
            "figureComposerLineColorCmapCombo",
        )
        is not None
    )
    trim_lower_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineColorCmapTrimLowerSpin"
    )
    trim_upper_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineColorCmapTrimUpperSpin"
    )
    assert trim_lower_spin is not None
    assert trim_upper_spin is not None
    assert trim_lower_spin.value() == pytest.approx(0.1)
    assert trim_upper_spin.value() == pytest.approx(0.2)
    assert not trim_lower_spin.keyboardTracking()
    assert not trim_upper_spin.keyboardTracking()
    assert trim_lower_spin.toolTip()
    assert trim_upper_spin.toolTip()
    assert (
        style_page.findChild(QtWidgets.QLineEdit, "figureComposerLineColorsEdit")
        is None
    )

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "line_color_values =" in code
    assert "line_colors = plt.get_cmap('plasma')(" in code
    assert "0.1 + 0.7 * line_color_values_norm(line_color_values)" in code
    assert "black" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    generated_lines = namespace["fig"].axes[0].lines
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in generated_lines]),
        expected_colors,
    )
    trim_lower_spin.setValue(0.25)
    trim_upper_spin.setValue(0.3)
    assert tool.tool_status.operations[0].line_color_cmap_trim_lower == pytest.approx(
        0.25
    )
    assert tool.tool_status.operations[0].line_color_cmap_trim_upper == pytest.approx(
        0.3
    )


def test_figure_composer_line_colormap_equal_values_use_midpoint() -> None:
    colors = figurecomposer_line_colormap.colors_from_values(
        (2.0, 2.0), "viridis", trim=(0.2, 0.3)
    )
    expected = plt.get_cmap("viridis")([0.45, 0.45])
    np.testing.assert_allclose(colors, expected)


def test_figure_composer_line_colormap_rejects_invalid_trim() -> None:
    with pytest.raises(ValueError, match="trim"):
        figurecomposer_line_colormap.colors_from_values(
            (1.0,), "viridis", trim=(0.5, 0.5)
        )
    with pytest.raises(ValueError, match="trim"):
        figurecomposer_line_colormap.colormap_code_lines(
            "[1.0]", "viridis", trim=(-0.1, 0.0)
        )
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={"line_color_cmap_trim_lower": 0.5, "line_color_cmap_trim_upper": 0.5}
    )
    assert figurecomposer_line_colormap.line_color_cmap_trim_control_values(
        operation
    ) == (0.0, 0.0)


def test_figure_composer_line_colormap_helper_edges() -> None:
    contexts = (
        {"numeric": 1.0, "text": "a", "nan": np.nan, "index": 0},
        {"numeric": 2.0, "text": "b", "nan": 1.0, "index": 1},
    )
    assert figurecomposer_line_colormap.numeric_context_field_names(contexts) == (
        "numeric",
    )
    assert figurecomposer_line_colormap.values_from_contexts(
        contexts, "numeric", item_name="line"
    ) == (1.0, 2.0)
    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, None, item_name="line"
        )
    with pytest.raises(ValueError, match="missing"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "missing", item_name="line"
        )
    with pytest.raises(ValueError, match="numeric scalars"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "text", item_name="line"
        )
    with pytest.raises(ValueError, match="finite numeric scalars"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "nan", item_name="line"
        )
    assert figurecomposer_line_colormap.colors_from_values((), "viridis") == ()
    assert figurecomposer_line_colormap.colormap_code_lines("[0.0, 1.0]", "viridis")[
        -2:
    ] == ["else:", "    line_colors = []"]


def test_figure_composer_line_color_mode_widget_updates_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
    ).model_copy(
        update={
            "line_iter_dim": "eV",
            "line_color_mode": "manual",
            "line_color_coord": "missing",
            "line_color_cmap": "viridis_r",
        }
    )
    widget = figurecomposer_toolbar_dialogs._LineColorModeWidget(
        tool, operation, line_kind="profile"
    )
    qtbot.addWidget(widget)
    emitted: list[FigureOperationState] = []
    widget.sigOperationChanged.connect(emitted.append)

    assert widget._mode_from_text("By coordinate") == "manual"
    widget.mode_combo.setItemData(widget.mode_combo.currentIndex(), None)
    assert widget._mode_from_text("By coordinate") == "coordinate"
    widget.mode_combo.setItemData(widget.mode_combo.currentIndex(), "manual")

    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("coordinate"))
    widget.mode_combo.activated.emit(widget.mode_combo.currentIndex())
    assert emitted[-1].line_color_mode == "coordinate"
    assert emitted[-1].line_color_coord == "missing"
    assert not widget.coordinate_page.isHidden()

    widget.coord_combo.setCurrentIndex(widget.coord_combo.findData("eV"))
    widget.coord_combo.activated.emit(widget.coord_combo.currentIndex())
    assert emitted[-1].line_color_coord == "eV"

    widget.cmap_combo.setCurrentText("plasma")
    widget.cmap_combo.activated.emit(widget.cmap_combo.currentIndex())
    assert emitted[-1].line_color_cmap == "plasma"
    assert emitted[-1].line_color_cmap_reverse is True

    widget.reverse_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert emitted[-1].line_color_cmap_reverse is False
    widget.trim_lower_spin.setValue(0.1)
    widget.trim_upper_spin.setValue(0.2)
    assert emitted[-1].line_color_cmap_trim_lower == pytest.approx(0.1)
    assert emitted[-1].line_color_cmap_trim_upper == pytest.approx(0.2)

    emitted.clear()
    widget._updating = True
    widget._mode_changed(0)
    widget._coord_changed(0)
    widget._cmap_changed(0)
    widget._reverse_changed(QtCore.Qt.CheckState.Checked.value)
    widget._trim_lower_changed(0.3)
    widget._trim_upper_changed(0.4)
    assert emitted == []
    widget._updating = False

    plot_slices_operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.1, 0.2),
    ).model_copy(update={"line_color_mode": "manual"})
    plot_slices_widget = figurecomposer_toolbar_dialogs._LineColorModeWidget(
        tool, plot_slices_operation, line_kind="plot_slices"
    )
    qtbot.addWidget(plot_slices_widget)
    plot_slices_widget.mode_combo.setItemData(
        plot_slices_widget.mode_combo.currentIndex(), None
    )
    assert plot_slices_widget._mode_from_text("By coordinate") == "coordinate"


def test_figure_composer_image_operation_style_widget_updates_state(qtbot) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(
        update={"cmap": "viridis_r", "norm_name": "PowerNorm", "norm_gamma": 0.5}
    )
    widget = figurecomposer_toolbar_dialogs._ImageOperationStyleWidget(operation)
    qtbot.addWidget(widget)
    emitted: list[FigureOperationState] = []
    widget.sigOperationChanged.connect(emitted.append)

    widget.cmap_combo.setCurrentText("magma")
    widget.cmap_combo.activated.emit(widget.cmap_combo.currentIndex())
    assert emitted[-1].cmap == "magma_r"

    widget.cmap_reverse_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert emitted[-1].cmap == "magma"

    widget.norm_combo.setCurrentText("Normalize")
    widget.norm_combo.activated.emit(widget.norm_combo.currentIndex())
    assert emitted[-1].norm_name == "Normalize"

    widget.vmin_edit.setText("2.5")
    widget.vmin_edit.editingFinished.emit()
    assert emitted[-1].vmin == pytest.approx(2.5)

    widget.vmin_edit.setText("")
    widget.vmin_edit.editingFinished.emit()
    assert emitted[-1].vmin is None

    widget.clip_combo.setCurrentText("True")
    widget.clip_combo.activated.emit(widget.clip_combo.currentIndex())
    assert emitted[-1].norm_clip is True

    widget.norm_kwargs_edit.setText("{'vmin': -1.0}")
    widget.norm_kwargs_edit.editingFinished.emit()
    assert emitted[-1].norm_kwargs == {"vmin": -1.0}

    emitted.clear()
    widget._updating = True
    widget._cmap_changed(0)
    widget._cmap_reverse_changed(QtCore.Qt.CheckState.Checked.value)
    widget._norm_changed(0)
    widget._number_changed("vmin", widget.vmin_edit)
    widget._clip_changed(0)
    widget._norm_kwargs_changed()
    assert emitted == []
    widget._updating = False


def test_figure_composer_line_profile_helper_edges() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={
            "eV": [0.1, 0.2],
            "kx": [-1.0, 0.0, 1.0],
            "offset": ("eV", [10.0, 20.0]),
        },
        name="data",
    )
    tool = types.SimpleNamespace(_source_data={"data": data})
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={"line_iter_dim": "eV", "line_x": "kx"}
    )

    assert figurecomposer_line_profile._line_reduce_text("unknown") == "Disabled"
    assert figurecomposer_line_profile._line_reduce_from_text("unknown") == "disabled"
    assert figurecomposer_line_profile._line_color_mode_text(
        "unknown"
    ) == figurecomposer_line_profile._line_color_mode_text("manual")
    assert (
        figurecomposer_line_profile._line_color_mode_from_text("By coordinate")
        == "coordinate"
    )
    assert figurecomposer_line_profile._line_color_mode_from_text("Manual") == "manual"
    assert (
        figurecomposer_line_profile._available_line_value_names(
            tool, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    assert (
        figurecomposer_line_profile._available_line_value_names(
            tool, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    assert set(
        figurecomposer_line_profile._available_line_coordinate_names(tool, operation)
    ) >= {"kx"}
    assert figurecomposer_line_profile._available_line_offset_coords(
        tool, operation
    ) == ["offset"]

    updates: list[dict[str, object]] = []
    cmap_tool = types.SimpleNamespace(
        _updating_controls=True,
        _update_current_operation=lambda **kwargs: updates.append(kwargs),
    )
    figurecomposer_line_profile._update_current_line_color_cmap(
        cmap_tool, "viridis", True
    )
    assert updates == []
    cmap_tool._updating_controls = False
    figurecomposer_line_profile._update_current_line_color_cmap(
        cmap_tool, "viridis_r", True
    )
    assert updates == [{"line_color_cmap": "viridis", "line_color_cmap_reverse": True}]


def test_figure_composer_line_profile_style_codegen_helper_edges() -> None:
    profiles = [
        xr.DataArray(
            [1.0, 2.0],
            dims=("kx",),
            coords={"kx": [0.0, 1.0], "eV": value},
            name="profile",
        )
        for value in (0.1, 0.2)
    ]
    sources = ["first", "second"]
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={
            "line_iter_dim": "eV",
            "line_label_text": "{source}:{number}:{dim}:{value:g}:{eV:g}",
            "line_color_mode": "coordinate",
            "line_color_cmap": "plasma",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.2,
        }
    )
    loop_names = ["profile"]
    loop_values = ["profiles"]
    lines, kwargs = figurecomposer_line_profile._line_style_code(
        operation,
        profiles=profiles,
        sources=sources,
        loop_names=loop_names,
        loop_values=loop_values,
    )
    assert loop_names == ["profile", "index", "source", "color"]
    assert loop_values[1] == "range(len(profiles))"
    assert loop_values[2] == "['first', 'second']"
    assert loop_values[3] == "line_colors"
    assert any("line_color_values =" in line for line in lines)
    assert "label=f'{source}:{index + 1}" in kwargs
    assert "color=color" in kwargs

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_line_profile._line_style_code(
            operation.model_copy(update={"line_iter_dim": None, "line_label_text": ""}),
            profiles=profiles,
            sources=sources,
            loop_names=["profile"],
            loop_values=["profiles"],
        )

    single_color_operation = FigureOperationState.line(
        label="profiles", source="data"
    ).model_copy(update={"line_colors": ("red",)})
    _lines, single_kwargs = figurecomposer_line_profile._line_style_code(
        single_color_operation,
        profiles=profiles,
        sources=sources,
        loop_names=["profile"],
        loop_values=["profiles"],
    )
    assert "color='red'" in single_kwargs

    multi_color_loop_names = ["profile"]
    multi_color_loop_values = ["profiles"]
    _lines, multi_kwargs = figurecomposer_line_profile._line_style_code(
        single_color_operation.model_copy(update={"line_colors": ("red", "blue")}),
        profiles=profiles,
        sources=sources,
        loop_names=multi_color_loop_names,
        loop_values=multi_color_loop_values,
    )
    assert multi_color_loop_names == ["profile", "color"]
    assert multi_color_loop_values[1] == "['red', 'blue']"
    assert "color=color" in multi_kwargs


def test_figure_composer_default_line_labels_use_property_labels() -> None:
    from erlab.interactive._figurecomposer import _labels as figurecomposer_labels

    assert (
        figurecomposer_labels.default_label_text(
            "sample_temp", (20.0,), fallback="profile {number}"
        )
        == r"$T = {sample_temp:g}$ K"
    )
    assert (
        figurecomposer_labels.default_label_text(
            "eV", (-0.1,), fallback="profile {number}"
        )
        == r"$E-E_F = {eV:g}$ eV"
    )
    kx_label = figurecomposer_labels.default_label_text(
        "kx", (1.0,), fallback="profile {number}"
    )
    assert figurecomposer_labels.labels_from_text(
        kx_label,
        ({"kx": 1.25, "index": 0, "number": 1},),
    ) == (r"$k_x = 1.25$ Å${}^{-1}$",)
    phase_label = figurecomposer_labels.default_label_text(
        "phase", fallback="profile {number}"
    )
    assert phase_label == r"$phase = {phase}$"
    assert figurecomposer_labels.labels_from_text(
        phase_label,
        ({"phase": "A", "index": 0, "number": 1},),
    ) == (r"$phase = A$",)


def test_figure_composer_label_helper_edges() -> None:
    from erlab.interactive._figurecomposer import _labels as figurecomposer_labels

    profile = xr.DataArray(
        [1.0],
        dims=("x",),
        coords={"x": [0.0], "sample_temp": 20.0},
        name="profile",
    )
    context = figurecomposer_labels.label_context(
        profile,
        index=2,
        source="map",
        dim="sample_temp",
        value=np.asarray([20.0]),
    )
    assert context["number"] == 3
    assert context["source"] == "map"
    assert context["sample_temp"] == 20.0
    assert context["value"] == 20.0
    assert figurecomposer_labels.labels_from_text("", (), default="fallback") == ()
    assert figurecomposer_labels.labels_from_text(
        "", (context, context), literal_values=("one",), default=None
    ) == ("one", "one")
    with pytest.raises(ValueError, match="one value per profile"):
        figurecomposer_labels.labels_from_text(
            "", (context, context), literal_values=("a", "b", "c")
        )
    assert figurecomposer_labels.labels_from_text(
        "$k_{F}$", (context,), default=None
    ) == (r"$k_{F}$",)
    assert figurecomposer_labels.labels_from_text(
        "{source!r}:{sample_temp:.1f}", (context,)
    ) == ("'map':20.0",)
    with pytest.raises(ValueError, match="not available"):
        figurecomposer_labels.labels_from_text("{missing:g}", (context,))
    with pytest.raises(ValueError, match="Could not format"):
        figurecomposer_labels.labels_from_text("{source:g}", (context,))

    assert not figurecomposer_labels.label_text_uses_placeholders("", (context,))
    assert not figurecomposer_labels.label_text_uses_placeholders("plain", (context,))
    assert (
        figurecomposer_labels.label_fstring_code(
            r"{source}\{missing}", {"source": "source"}
        )
        == r"f'{source}\\{{missing}}'"
    )
    with pytest.raises(ValueError, match="not available"):
        figurecomposer_labels.label_fstring_code("{missing:g}", {"source": "source"})
    assert figurecomposer_labels.coord_value_expression("sample_temp") == (
        'profile.coords["sample_temp"].values.item()'
    )
    assert figurecomposer_labels.string_literal_expression("a'b") == '"a\'b"'
    assert (
        figurecomposer_labels.default_label_text(
            None, fallback="profile {number}", source_count=2
        )
        == "{source}, profile {number}"
    )
    assert (
        figurecomposer_labels.default_label_text(
            "phase", ("A",), fallback="profile {number}", source_count=2
        )
        == r"{source}, $phase = {phase}$"
    )
    assert "comma-separated" in figurecomposer_labels.label_text_tooltip(
        (), item_name="profile"
    )
    assert "Available" in figurecomposer_labels.label_text_tooltip(
        (context,), item_name="profile"
    )
    assert (
        figurecomposer_labels.label_editor_text(
            FigureOperationState.line(label="line", source="data").model_copy(
                update={"line_labels": ("A", "B")}
            )
        )
        == "A, B"
    )


def test_figure_composer_plot_slices_line_label_placeholders_codegen(
    qtbot,
) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=tuple(float(value) for value in eV),
    ).model_copy(update={"line_label_text": r"$E-E_F = {eV:g}$ eV"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_labels = [r"$E-E_F = -0.1$ eV", r"$E-E_F = 0.2$ eV"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [axis.lines[0].get_label() for axis in tool.figure.axes] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert "slice_value" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        axis.lines[0].get_label() for axis in namespace["fig"].axes
    ] == expected_labels


def test_figure_composer_plot_slices_line_coordinate_colormap_codegen(
    qtbot,
) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=tuple(float(value) for value in eV),
    ).model_copy(
        update={
            "line_color_mode": "coordinate",
            "line_color_cmap": "viridis",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.15,
            "line_kw": {"color": "red", "linestyle": "--"},
            "line_label_text": r"$E-E_F = {eV:g}$ eV",
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"linewidth": 2.5, "color": "black"},
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_colors = _expected_line_colormap_colors(eV, "viridis", trim=(0.1, 0.15))
    assert figurecomposer_plot_slices._available_plot_slices_line_color_coords(
        tool, operation
    ) == ["eV"]
    line_kw = figurecomposer_plot_slices._panel_line_kw_argument(tool, operation)
    assert isinstance(line_kw, list)
    assert line_kw[0][0]["linestyle"] == "--"
    assert line_kw[0][1]["linewidth"] == 2.5
    assert "c" not in line_kw[0][0]
    np.testing.assert_allclose(
        [line_kw[0][0]["color"], line_kw[0][1]["color"]],
        expected_colors,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    rendered_lines = [axis.lines[0] for axis in tool.figure.axes]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in rendered_lines]),
        expected_colors,
    )
    assert [line.get_linestyle() for line in rendered_lines] == ["--", "--"]
    assert rendered_lines[1].get_linewidth() == 2.5

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineColorModeCombo"
        )
        is not None
    )
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineColorCoordCombo"
        )
        is not None
    )
    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapComboBox,
            "figureComposerPlotSlicesLineColorCmapCombo",
        )
        is not None
    )
    trim_lower_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox,
        "figureComposerPlotSlicesLineColorCmapTrimLowerSpin",
    )
    trim_upper_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox,
        "figureComposerPlotSlicesLineColorCmapTrimUpperSpin",
    )
    assert trim_lower_spin is not None
    assert trim_upper_spin is not None
    assert trim_lower_spin.value() == pytest.approx(0.1)
    assert trim_upper_spin.value() == pytest.approx(0.15)
    assert not trim_lower_spin.keyboardTracking()
    assert not trim_upper_spin.keyboardTracking()
    assert trim_lower_spin.toolTip()
    assert trim_upper_spin.toolTip()
    assert (
        colors_page.findChild(
            QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
        )
        is None
    )

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "line_color_values =" in code
    assert "line_colors = plt.get_cmap('viridis')(" in code
    assert "0.1 + 0.75 * line_color_values_norm(line_color_values)" in code
    assert "red" not in code
    assert "black" not in code
    assert "line_colors[0]" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    generated_lines = [axis.lines[0] for axis in namespace["fig"].axes]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in generated_lines]),
        expected_colors,
    )
    assert [line.get_linestyle() for line in generated_lines] == ["--", "--"]
    assert generated_lines[1].get_linewidth() == 2.5
    trim_lower_spin.setValue(0.2)
    trim_upper_spin.setValue(0.25)
    assert tool.tool_status.operations[0].line_color_cmap_trim_lower == pytest.approx(
        0.2
    )
    assert tool.tool_status.operations[0].line_color_cmap_trim_upper == pytest.approx(
        0.25
    )


def test_figure_composer_plot_slices_all_coordinate_helper_edges() -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(5, 3),
        dims=("eV", "kx"),
        coords={"eV": np.linspace(-0.2, 0.2, 5), "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    tool = types.SimpleNamespace(
        _source_data={"data": data},
        _source_display_name=lambda name: name,
        _editable_operations=lambda: (),
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
    ).model_copy(update={"slice_values_mode": "all"})

    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values(tool, operation) == ()
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_code(operation) is None
    )
    assert (
        figurecomposer_plot_slices._slice_values_mode_from_text("not a mode")
        == "manual"
    )
    assert (
        figurecomposer_plot_slices._line_color_mode_from_text("By coordinate")
        == "coordinate"
    )
    assert figurecomposer_plot_slices._line_color_mode_from_text("Manual") == "manual"
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_error(
            tool, operation, data.dims
        )
        == "Choose a dimension before using all coordinate values."
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(tool, operation)
        == "Choose a dimension."
    )

    missing_dim = operation.model_copy(update={"slice_dim": "missing"})
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values(tool, missing_dim) == ()
    )
    assert "'missing' is not an input dimension" in (
        figurecomposer_plot_slices._all_coordinate_slice_values_error(
            tool, missing_dim, data.dims
        )
    )
    assert "'missing' is not an input dimension" in (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(
            tool, missing_dim
        )
    )

    missing_source_tool = types.SimpleNamespace(
        _source_data={},
        _source_display_name=lambda name: name,
        _editable_operations=lambda: (),
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values(
            missing_source_tool, operation.model_copy(update={"slice_dim": "eV"})
        )
        == ()
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(
            missing_source_tool, missing_dim
        )
        == "Select at least one valid source."
    )

    string_coord = xr.DataArray(
        np.ones((2, 3)),
        dims=("label", "kx"),
        coords={"label": ["a", "b"], "kx": [-1.0, 0.0, 1.0]},
        name="string_coord",
    )
    string_tool = types.SimpleNamespace(
        _source_data={"data": string_coord},
        _source_display_name=lambda name: name,
        _editable_operations=lambda: (),
    )
    string_operation = operation.model_copy(update={"slice_dim": "label"})
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values(
            string_tool, string_operation
        )
        == ()
    )
    assert "numeric and non-empty" in (
        figurecomposer_plot_slices._all_coordinate_slice_values_error(
            string_tool, string_operation, string_coord.dims
        )
    )
    assert "numeric and non-empty" in (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(
            string_tool, string_operation
        )
    )

    thinned = operation.model_copy(update={"slice_dim": "eV", "slice_values_thin": 2})
    assert figurecomposer_plot_slices._all_coordinate_slice_values(
        tool, thinned
    ) == pytest.approx(tuple(data.thin({"eV": 2}).coords["eV"].values))
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(tool, thinned)
        == "eV: 5 values, 3 plotted"
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_code(thinned)
        == 'data.thin({"eV": 2}).coords["eV"].values'
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_code(
            thinned.model_copy(update={"slice_values_thin": 1})
        )
        == 'data.coords["eV"].values'
    )
    assert (
        figurecomposer_plot_slices._all_coordinate_slice_values_summary(
            tool, thinned.model_copy(update={"slice_values_thin": 1})
        )
        == "eV: 5 values"
    )
    assert (
        figurecomposer_plot_slices._plot_slices_slice_values_code(
            tool, operation.model_copy(update={"slice_values_mode": "manual"})
        )
        == "[None]"
    )
    assert (
        figurecomposer_plot_slices._plot_slices_slice_values_code(
            tool,
            operation.model_copy(
                update={
                    "slice_values_mode": "manual",
                    "slice_dim": "eV",
                    "slice_values": (0.1,),
                }
            ),
        )
        == "[0.1]"
    )
    assert figurecomposer_plot_slices._plot_slices_panel_qsel_kwargs(
        operation.model_copy(
            update={"slice_dim": "eV", "slice_values": (0.1,), "slice_width": 0.2}
        ),
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, ""),
    ) == {"eV": 0.1, "eV_width": 0.2}


def test_figure_composer_tick_params_editor_edge_commits(qtbot) -> None:
    editor = figurecomposer_tick_params.TickParamsEditorWidget(
        {"length": 1.0, "grid_alpha": 0.5, "grid_linestyle": "--"}
    )
    qtbot.addWidget(editor)
    emitted: list[dict[str, typing.Any]] = []
    editor.sigTickParamsChanged.connect(emitted.append)

    editor.setToolTip("ignored")
    assert editor.toolTip() == ""

    _finish_tick_params_edit(editor, "figureComposerAxesMethodTickParamsLengthEdit", "")
    assert "length" not in editor.tick_params()
    assert emitted[-1]["grid_alpha"] == 0.5

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "not-a-number"
    )
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "-1"
    )
    assert emitted == []

    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "2.5"
    )
    assert emitted[-1]["width"] == pytest.approx(2.5)

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsGridAlphaEdit", "2"
    )
    assert emitted == []
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsGridAlphaEdit", "0.25"
    )
    assert emitted[-1]["grid_alpha"] == pytest.approx(0.25)

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", "not valid("
    )
    assert emitted == []
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", "'small'"
    )
    assert emitted[-1]["labelsize"] == "small"
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", ""
    )
    assert "labelsize" not in emitted[-1]

    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelFontEdit", "Arial"
    )
    assert emitted[-1]["labelfontfamily"] == "Arial"
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelFontEdit", ""
    )
    assert "labelfontfamily" not in emitted[-1]

    colors_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    assert colors_edit is not None
    colors_edit.setText("")
    colors_edit.editingFinished.emit()
    assert "colors" not in emitted[-1]

    combo = editor.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTickParamsGridLineStyleCombo"
    )
    assert combo is not None
    figurecomposer_tick_params.TickParamsEditorWidget._set_combo_value(
        combo, "not-present"
    )
    assert combo.currentIndex() == 0

    editor._updating = True
    editor._set_kwarg("axis", "x")
    assert emitted[-1].get("axis") != "x"
    editor._updating = False

    tristate = figurecomposer_tick_params._TriStateCheckBox("test")
    qtbot.addWidget(tristate)
    tristate.set_value("bad")
    assert tristate.value() is None


def test_figure_composer_plot_slices_label_codegen_helper_variants() -> None:
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("a", "b"),
        slice_dim="eV",
        slice_values=(0.1, 0.2),
    ).model_copy(update={"line_label_text": "{source}:{number}:{eV:g}"})
    fields = {"source", "number", "eV"}
    single_key = (figurecomposer_plot_slices._PlotSlicesPanelKey(0, 0, "A"),)
    by_slice_keys = tuple(
        figurecomposer_plot_slices._PlotSlicesPanelKey(0, index, f"A {index}")
        for index in range(2)
    )
    by_source_keys = tuple(
        figurecomposer_plot_slices._PlotSlicesPanelKey(index, 0, f"S {index}")
        for index in range(2)
    )
    grid_keys = tuple(
        figurecomposer_plot_slices._PlotSlicesPanelKey(map_index, slice_index, "")
        for map_index in range(2)
        for slice_index in range(2)
    )
    namespace = {"slice_values": [0.1, 0.2]}

    single = figurecomposer_plot_slices._plot_slices_label_line_kw_comprehension_code(
        operation, single_key, ("alpha",), "slice_values", fields
    )
    assert eval(single, {}, namespace) == {"label": "alpha:1:0.1"}  # noqa: S307

    by_slice = figurecomposer_plot_slices._plot_slices_label_line_kw_comprehension_code(
        operation, by_slice_keys, ("alpha",), "slice_values", fields
    )
    assert eval(by_slice, {}, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "alpha:2:0.2"},
    ]

    by_source = (
        figurecomposer_plot_slices._plot_slices_label_line_kw_comprehension_code(
            operation, by_source_keys, ("alpha", "beta"), "slice_values", fields
        )
    )
    assert eval(by_source, {}, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "beta:2:0.1"},
    ]

    grid = figurecomposer_plot_slices._plot_slices_label_line_kw_comprehension_code(
        operation, grid_keys, ("alpha", "beta"), "slice_values", fields
    )
    assert eval(grid, {}, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "alpha:2:0.2"}],
        [{"label": "beta:3:0.1"}, {"label": "beta:4:0.2"}],
    ]

    fortran_grid = (
        figurecomposer_plot_slices._plot_slices_label_line_kw_comprehension_code(
            operation.model_copy(update={"order": "F"}),
            grid_keys,
            ("alpha", "beta"),
            "slice_values",
            fields,
        )
    )
    assert eval(fortran_grid, {}, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "beta:2:0.1"}],
        [{"label": "alpha:3:0.2"}, {"label": "beta:4:0.2"}],
    ]

    styled = figurecomposer_plot_slices._plot_slices_styled_label_line_kw_code(
        operation.model_copy(
            update={
                "line_kw": {"linewidth": 1.5},
                "panel_styles": (
                    FigurePlotSlicesPanelStyleState(
                        map_index=1, slice_index=1, line_kw={"color": "red"}
                    ),
                ),
            }
        ),
        grid_keys,
        ("alpha", "beta"),
        "slice_values",
        {
            (1, 1): FigurePlotSlicesPanelStyleState(
                map_index=1, slice_index=1, line_kw={"color": "red"}
            )
        },
        fields,
    )
    styled_value = eval(styled, {}, namespace)  # noqa: S307
    assert styled_value[1][1]["label"] == "beta:4:0.2"
    assert styled_value[1][1]["color"] == "red"
    assert styled_value[0][0]["linewidth"] == 1.5

    assert (
        figurecomposer_plot_slices._plot_slices_panel_index_expr(
            "F",
            map_count=2,
            slice_count=3,
            map_index_expr="map_index",
            slice_index_expr="slice_index",
        )
        == "slice_index * 2 + map_index"
    )
    assert (
        figurecomposer_plot_slices._line_kw_dict_code({"alpha": 0.5}, "label_code")
        == "{**{'alpha': 0.5}, 'label': label_code}"
    )


def test_figure_composer_plot_slices_line_color_codegen_helper_variants() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    tool = types.SimpleNamespace(
        _source_data={
            "a": data,
            "b": data + 10.0,
            "line": xr.DataArray([1.0, 2.0], dims=("kx",), name="line"),
        },
        _source_display_name=lambda name: name.upper(),
        _editable_operations=lambda: (),
    )

    assert (
        figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="manual",
                sources=("a",),
                slice_dim="eV",
                slice_values=(0.1,),
            ),
        )
        == []
    )

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="missing", sources=("line",)
            ).model_copy(update={"line_color_mode": "coordinate"}),
        )
    with pytest.raises(ValueError, match="Cannot color slices"):
        figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="bad",
                sources=("a",),
                slice_dim="eV",
                slice_values=(0.1,),
            ).model_copy(
                update={"line_color_mode": "coordinate", "line_color_coord": "kx"}
            ),
        )

    one_slice = FigureOperationState.plot_slices(
        label="one",
        sources=("a", "b"),
        slice_dim="eV",
        slice_values=(0.1,),
    ).model_copy(update={"line_color_mode": "coordinate"})
    assert any(
        "for _ in range(2)" in line
        for line in figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool, one_slice
        )
    )

    fortran = one_slice.model_copy(update={"slice_values": (0.1, 0.2), "order": "F"})
    assert any(
        "for slice_value in [0.1, 0.2] for _ in range(2)" in line
        for line in figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool, fortran
        )
    )

    c_order = fortran.model_copy(update={"order": "C"})
    assert any(
        "for _ in range(2) for slice_value in [0.1, 0.2]" in line
        for line in figurecomposer_plot_slices._plot_slices_line_color_code_lines(
            tool, c_order
        )
    )

    label_operation = FigureOperationState.plot_slices(
        label="labels",
        sources=("a",),
        slice_dim="eV",
        slice_values=(0.1,),
    ).model_copy(
        update={
            "line_color_mode": "coordinate",
            "line_labels": ("literal",),
        }
    )
    label_line_kw_code = figurecomposer_plot_slices._plot_slices_line_kw_code(
        tool, label_operation
    )
    assert label_line_kw_code is not None
    assert "'label': 'literal'" in label_line_kw_code
    assert "'color': line_colors[0]" in label_line_kw_code


def test_figure_composer_line_coordinate_colormap_rejects_bad_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, np.nan], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_color_mode": "coordinate",
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    render_error = tool._operation_render_errors[operation.operation_id]
    assert "finite numeric scalars" in render_error
    with pytest.raises(ValueError, match="finite numeric scalars"):
        tool.generated_code()

    plot_operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, np.nan),
    ).model_copy(update={"line_color_mode": "coordinate"})
    plot_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(plot_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(plot_tool)
    with pytest.raises(ValueError, match="finite numeric scalars"):
        figurecomposer_plot_slices._panel_line_kw_argument(plot_tool, plot_operation)
    with pytest.raises(ValueError, match="finite numeric scalars"):
        plot_tool.generated_code()


def test_figure_composer_one_profile_per_axis_codegen_executes(qtbot) -> None:
    cut_values = np.array([0.0, 1.0, 2.0])
    energy = np.array([-0.2, 0.0, 0.2])
    kx = np.array([-1.0, 0.0, 1.0, 2.0])
    values = np.arange(cut_values.size * energy.size * kx.size, dtype=float).reshape(
        cut_values.size, energy.size, kx.size
    )
    data = xr.DataArray(
        values,
        dims=("cut", "eV", "kx"),
        coords={"cut": cut_values, "eV": energy, "kx": kx},
        name="data",
    )
    profile_operation = FigureOperationState.line(
        label="mdc overlay",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_selection": {"cut": cut_values.tolist(), "eV": 0.0},
            "line_iter_dim": "cut",
            "line_normalize": "max",
            "line_colors": ("black",),
            "line_scales": (0.1, 0.2, 0.3),
            "line_offsets": (-0.2, 0.0, 0.2),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
                    slice_dim="cut",
                    slice_values=tuple(float(value) for value in cut_values),
                ),
                profile_operation,
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "if len(target_axes)" not in code
    assert "profiles =" in code
    assert "profile_scales =" not in code
    assert "profile_offsets =" not in code
    assert "[0.1, 0.2, 0.3]," in code
    assert "[-0.2, 0.0, 0.2]," in code
    assert (
        "profiles = [\n    offset + scale * (profile / profile.max(skipna=True))"
    ) in code
    assert "for profile, scale, offset in zip(" in code
    assert "for ax, profile in zip(" in code
    assert "ax.plot(profile['kx'], profile" in code
    assert "_line = " not in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    for index, axis in enumerate(axs.flat):
        line = axis.lines[0]
        profile = data.qsel(cut=float(cut_values[index]), eV=0.0).squeeze(drop=True)
        expected = profile_operation.line_offsets[
            index
        ] + profile_operation.line_scales[index] * (profile / profile.max(skipna=True))
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(line.get_ydata(), expected.values)


def test_figure_composer_line_gradient_fill_one_profile_per_axis_executes(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5]]),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_values_axis": "x",
            "line_colors": ("red", "blue"),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for index, axis in enumerate(fig.axes):
        profile = data.isel(cut=index)
        assert len(axis.lines) == 1
        assert len(axis.images) == 1
        np.testing.assert_allclose(axis.lines[0].get_xdata(), profile.values)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), kx)
        np.testing.assert_allclose(
            axis.images[0].cmap(1.0),
            mcolors.to_rgba(axis.lines[0].get_color()),
            rtol=1e-7,
        )

    code = tool.generated_code()
    assert "for ax, profile, color in zip(" in code
    assert "_line = ax.plot(profile, profile['kx'], color=color)[0]" in code
    assert "transpose=True" in code
    assert "for _line in" not in code
    assert "ax.lines" not in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "    _line = ax.plot(profile, profile['kx'], color=color)[0]"
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for index, axis in enumerate(namespace["fig"].axes):
        profile = data.isel(cut=index)
        assert len(axis.lines) == 1
        assert len(axis.images) == 1
        np.testing.assert_allclose(axis.lines[0].get_xdata(), profile.values)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), kx)
        np.testing.assert_allclose(
            axis.images[0].cmap(1.0),
            mcolors.to_rgba(axis.lines[0].get_color()),
            rtol=1e-7,
        )


def test_figure_composer_line_gradient_fill_codegen_preserves_line_source(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.array([1.0, 2.0, 1.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="_line",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="_line", label="_line"),),
            operations=(
                FigureOperationState.line(
                    label="gradient",
                    source="_line",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"gradient": True}),
                FigureOperationState.line(
                    label="later",
                    source="_line",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="_line",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    code_lines = code.splitlines()
    assert not any(line.startswith("    _line = profile.plot(") for line in code_lines)
    assert "_line_2 = axs[0, 0].plot(profile[profile.dims[0]], profile)[0]" in code
    assert "profile_data = _line" in code

    namespace: dict[str, typing.Any] = {"_line": data}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes[0].lines) == 2
    assert len(namespace["fig"].axes[0].images) == 1


def test_figure_composer_profile_reduce_codegen_executes(qtbot) -> None:
    cut_values = np.arange(6.0)
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="reduced profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_reduce": "both",
            "line_reduce_coarsen": 2,
            "line_reduce_thin": 2,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    reduced = data.coarsen(cut=2, boundary="trim").mean().thin(cut=2)
    profiles = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(profiles) == 2
    xr.testing.assert_identical(profiles[0], reduced.isel(cut=0))
    xr.testing.assert_identical(profiles[1], reduced.isel(cut=1))

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for index, axis in enumerate(fig.axes):
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profiles[index].values)

    code = tool.generated_code()
    assert "if len(target_axes)" not in code
    assert (
        'profile_data = data.coarsen(cut=2, boundary="trim").mean().thin(cut=2)' in code
    )
    assert "profile_data = profile_data." not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for index, axis in enumerate(namespace["axs"].flat):
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profiles[index].values)


def test_figure_composer_one_profile_per_axis_codegen_broadcasts_profiles(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0, 2.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )

    many_profiles_operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_offset_source": "index",
            "xlim": (-0.5, 0.5),
        }
    )
    many_profiles_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(many_profiles_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(many_profiles_tool)

    namespace: dict[str, typing.Any] = {"data": data}
    many_profiles_code = many_profiles_tool.generated_code()
    assert "if len(target_axes)" not in many_profiles_code
    assert "target_axes =" not in many_profiles_code
    assert "for profile in profiles:" in many_profiles_code
    assert "axs[0, 0].plot(profile['kx'], profile)" in many_profiles_code
    assert "axs[0, 0].set(xlim=(-0.5, 0.5))" in many_profiles_code
    exec(many_profiles_code, namespace)  # noqa: S102
    lines = namespace["fig"].axes[0].lines
    assert len(lines) == 3
    assert namespace["fig"].axes[0].get_xlim() == pytest.approx((-0.5, 0.5))
    for index, line in enumerate(lines):
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(
            line.get_ydata(), data.isel(cut=index).values + index
        )
    figurecomposer_rendering._render_into_figure(
        many_profiles_tool, many_profiles_tool.figure, sync_visible=False
    )
    rendered_lines = many_profiles_tool.figure.axes[0].lines
    assert len(rendered_lines) == 3
    for index, line in enumerate(rendered_lines):
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(
            line.get_ydata(), data.isel(cut=index).values + index
        )

    single_profile_operation = FigureOperationState.line(
        label="profile",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_selection": {"cut": 1.0},
            "line_offset_source": "index",
            "xlim": (-0.5, 0.5),
        }
    )
    single_profile_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(single_profile_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(single_profile_tool)

    namespace = {"data": data}
    single_profile_code = single_profile_tool.generated_code()
    assert "if len(target_axes)" not in single_profile_code
    assert "target_axes =" not in single_profile_code
    assert "profiles * 3," in single_profile_code
    exec(single_profile_code, namespace)  # noqa: S102
    profile = data.qsel(cut=1.0).squeeze(drop=True)
    for index, axis in enumerate(namespace["axs"].flat):
        assert len(axis.lines) == 1
        assert axis.get_xlim() == pytest.approx((-0.5, 0.5))
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profile.values + index)
    figurecomposer_rendering._render_into_figure(
        single_profile_tool, single_profile_tool.figure, sync_visible=False
    )
    for index, axis in enumerate(single_profile_tool.figure.axes):
        assert len(axis.lines) == 1
        assert axis.get_xlim() == pytest.approx((-0.5, 0.5))
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profile.values + index)


def test_figure_composer_line_action_seeds_from_selected_slice_step(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(18.0).reshape(3, 2, 3),
        dims=("cut", "eV", "kx"),
        coords={
            "cut": [0.0, 1.0, 2.0],
            "eV": [-0.1, 0.1],
            "kx": [-1.0, 0.0, 1.0],
        },
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
                    slice_dim="cut",
                    slice_values=(0.0, 1.0, 2.0),
                    slice_width=0.25,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    profile_action = next(
        action for action in tool.add_step_menu.actions() if action.data() == "line"
    )
    profile_action.trigger()

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_placement == "one_per_axis"
    assert operation.line_normalize == "max"
    assert operation.line_colors == ("black",)
    assert operation.line_x == "kx"
    assert operation.line_iter_dim == "cut"
    assert operation.line_selection == {"cut": [0.0, 1.0, 2.0], "cut_width": 0.25}
    assert operation.axes.axes == ((0, 0), (0, 1), (0, 2))

    tool._select_step_section("view")
    placement_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerProfilePlacementCombo"
    )
    assert placement_combo is not None
    assert placement_combo.currentText() == "One profile per axis"

    tool._select_step_section("other")
    normalize_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert normalize_combo is not None
    assert normalize_combo.currentText() == "Each profile by maximum"
    assert "each extracted 1D profile independently" in normalize_combo.toolTip()

    unseeded_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(unseeded_tool)
    line_action = next(
        action
        for action in unseeded_tool.add_step_menu.actions()
        if action.data() == "line"
    )
    line_action.trigger()
    assert unseeded_tool.tool_status.operations[-1].line_placement == "all_axes"


def test_figure_composer_regular_line_profiles_render_on_each_selected_axis(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_values_axis": "x",
            "line_labels": ("low", "high"),
            "line_colors": ("red", "blue"),
            "line_scales": (2.0,),
            "ylim": (-1.5, 1.5),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for axis in fig.axes:
        assert len(axis.lines) == 2
        assert axis.get_ylim() == pytest.approx((-1.5, 1.5))
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(
                line.get_xdata(), 2.0 * data.isel(cut=index).values
            )
            np.testing.assert_allclose(line.get_ydata(), kx)
            assert line.get_label() == ("low", "high")[index]
            assert line.get_color() == ("red", "blue")[index]

    code = tool.generated_code()
    assert "for ax in" in code
    assert "ax.plot(profile, profile['kx']" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for axis in namespace["fig"].axes:
        assert len(axis.lines) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(
                line.get_xdata(), 2.0 * data.isel(cut=index).values
            )
            np.testing.assert_allclose(line.get_ydata(), kx)

    test_fig = Figure()
    test_axis = test_fig.subplots()
    with pytest.warns(UserWarning, match="identical low and high xlims"):
        figurecomposer_line_profile._set_axis_xlim(test_axis, 1.0)
    figurecomposer_line_profile._set_axis_ylim(test_axis, 2.0)
    assert test_axis.get_xlim()[0] < 1.0 < test_axis.get_xlim()[1]
    assert test_axis.get_ylim()[0] == pytest.approx(2.0)


def test_figure_composer_regular_line_gradient_fill_codegen_executes(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_colors": ("red", "blue"),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for axis in fig.axes:
        assert len(axis.lines) == 2
        assert len(axis.images) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(line.get_xdata(), kx)
            np.testing.assert_allclose(line.get_ydata(), data.isel(cut=index).values)
            assert line.get_color() == ("red", "blue")[index]

    code = tool.generated_code()
    assert "_line = ax.plot(profile['kx'], profile, color=color)[0]" in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "        _line = ax.plot(profile['kx'], profile, color=color)[0]"
    )
    assert code_lines[plot_index + 1] == "        eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for axis in namespace["fig"].axes:
        assert len(axis.lines) == 2
        assert len(axis.images) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(line.get_xdata(), kx)
            np.testing.assert_allclose(line.get_ydata(), data.isel(cut=index).values)


def test_figure_composer_line_gradient_fill_uses_new_line_after_existing_line(
    qtbot,
) -> None:
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.array([1.0, 2.0, 1.0]),
        dims=("kx",),
        coords={"kx": kx},
        name="data",
    )
    base_operation = FigureOperationState.line(
        label="base",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_colors": ("black",)})
    gradient_operation = FigureOperationState.line(
        label="gradient",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_colors": ("red",),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(base_operation, gradient_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    axis = fig.axes[0]
    assert len(axis.lines) == 2
    assert len(axis.images) == 1
    assert axis.lines[0].get_color() == "black"
    assert axis.lines[1].get_color() == "red"
    np.testing.assert_allclose(
        axis.images[0].cmap(1.0),
        mcolors.to_rgba(axis.lines[1].get_color()),
        rtol=1e-7,
    )

    code = tool.generated_code()
    assert "profile.plot(" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axis = namespace["fig"].axes[0]
    assert len(axis.lines) == 2
    assert len(axis.images) == 1
    assert axis.lines[0].get_color() == "black"
    assert axis.lines[1].get_color() == "red"
    np.testing.assert_allclose(
        axis.images[0].cmap(1.0),
        mcolors.to_rgba(axis.lines[1].get_color()),
        rtol=1e-7,
    )


def test_figure_composer_line_labels_auto_add_axes_legend_step(
    qtbot, monkeypatch
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("style")
    labels_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None

    rebuild_calls: list[None] = []
    monkeypatch.setattr(
        tool,
        "_update_operation_editor",
        lambda: rebuild_calls.append(None),
    )
    labels_edit.setText("profile A")
    labels_edit.editingFinished.emit()

    assert rebuild_calls == []
    assert tool.operation_list.currentRow() == 0
    assert len(tool.tool_status.operations) == 2
    line_operation, legend_operation = tool.tool_status.operations
    assert line_operation.line_label_text == "profile A"
    assert line_operation.line_labels == ()
    assert legend_operation.kind == FigureOperationKind.METHOD
    assert legend_operation.method_family == FigureMethodFamily.AXES
    assert legend_operation.method_name == "legend"
    assert legend_operation.axes == line_operation.axes

    labels_edit.setText("profile B")
    labels_edit.editingFinished.emit()

    assert len(tool.tool_status.operations) == 2
    assert tool.tool_status.operations[0].line_label_text == "profile B"
    assert tool.tool_status.operations[0].line_labels == ()


def test_figure_composer_disabled_line_labels_do_not_add_legend_or_render(
    qtbot,
    monkeypatch,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"enabled": False}),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("style")
    labels_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None
    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    labels_edit.setText("disabled profile")
    labels_edit.editingFinished.emit()

    assert len(tool.tool_status.operations) == 1
    assert tool.tool_status.operations[0].line_label_text == "disabled profile"
    assert tool.tool_status.operations[0].line_labels == ()
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None]


def test_figure_composer_batch_line_edits_update_selected_steps(qtbot) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(label="first", source="profile"),
                FigureOperationState.line(label="second", source="profile"),
                FigureOperationState.line(label="third", source="profile"),
                FigureOperationState.line(label="unselected", source="profile"),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1, 2))
    tool._select_step_section("other")
    other_page = tool.step_editor_stack.currentWidget()
    normalize_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert normalize_combo is not None
    _activate_combo_text(normalize_combo, "Each profile by mean")

    assert [operation.line_normalize for operation in tool.tool_status.operations] == [
        "mean",
        "mean",
        "mean",
        "none",
    ]
    assert _selected_operation_rows(tool) == (0, 1, 2)


def test_figure_composer_batch_line_mixed_values_do_not_overwrite_on_blur(
    qtbot,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(label="first", source="profile").model_copy(
                    update={"line_colors": ("C0",)}
                ),
                FigureOperationState.line(label="second", source="profile").model_copy(
                    update={"line_colors": ("C1",)}
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("style")
    style_page = tool.step_editor_stack.currentWidget()
    color_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorsEdit"
    )
    assert color_edit is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"

    color_edit.editingFinished.emit()
    assert [operation.line_colors for operation in tool.tool_status.operations] == [
        ("C0",),
        ("C1",),
    ]

    color_edit.setText("black")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert [operation.line_colors for operation in tool.tool_status.operations] == [
        ("black",),
        ("black",),
    ]


def test_figure_composer_batch_line_source_dependent_combos_disable(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("kx",),
        coords={"kx": [0.0, 1.0]},
        name="first",
    )
    second = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("ky",),
        coords={"ky": [0.0, 1.0]},
        name="second",
    )
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="first", label="first"),
                FigureSourceState(name="second", label="second"),
            ),
            operations=(
                FigureOperationState.line(label="first", source="first"),
                FigureOperationState.line(label="second", source="second"),
            ),
            primary_source="first",
        ),
        source_data={"first": first, "second": second},
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    assert coordinate_combo is not None
    assert coordinate_combo.isEnabled() is False
    assert "different valid choices" in coordinate_combo.toolTip()


def test_figure_composer_batch_line_labels_add_one_legend_per_axes_group(
    qtbot,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    first_axes = FigureAxesSelectionState(axes=((0, 0),))
    second_axes = FigureAxesSelectionState(axes=((0, 1),))
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="first",
                    source="profile",
                    axes=first_axes,
                ),
                FigureOperationState.line(
                    label="second",
                    source="profile",
                    axes=first_axes,
                ),
                FigureOperationState.line(
                    label="third",
                    source="profile",
                    axes=second_axes,
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1, 2))
    tool._select_step_section("style")
    labels_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None
    labels_edit.setText("profile")
    labels_edit.editingFinished.emit()

    operations = tool.tool_status.operations
    assert len(operations) == 5
    assert [
        operation.line_label_text
        for operation in operations
        if operation.kind == FigureOperationKind.LINE
    ] == ["profile", "profile", "profile"]
    assert operations[2].method_name == "legend"
    assert operations[2].axes == first_axes
    assert operations[4].method_name == "legend"
    assert operations[4].axes == second_axes


def test_figure_composer_editor_widget_rebuilds_are_deferred(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")
    values_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    assert values_edit is not None

    rebuild_calls: list[None] = []
    monkeypatch.setattr(
        tool,
        "_update_operation_editor",
        lambda: rebuild_calls.append(None),
    )
    values_edit.setText("0, 1")
    values_edit.editingFinished.emit()

    assert tool.tool_status.operations[0].slice_values == (0.0, 1.0)
    assert rebuild_calls == []
    assert tool._operation_editor_update_pending is True
    qtbot.waitUntil(lambda: rebuild_calls == [None], timeout=1000)
    assert tool._operation_editor_update_pending is False

    active_popup: list[QtWidgets.QWidget | None] = [QtWidgets.QMenu(tool)]
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "activePopupWidget",
        staticmethod(lambda: active_popup[0]),
    )
    rebuild_calls.clear()
    values_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    assert values_edit is not None
    values_edit.setText("1")
    values_edit.editingFinished.emit()

    assert tool.tool_status.operations[0].slice_values == (1.0,)
    assert tool._operation_editor_update_pending is True
    qtbot.wait(100)
    assert rebuild_calls == []

    active_popup[0] = None
    qtbot.waitUntil(lambda: rebuild_calls == [None], timeout=1000)
    assert tool._operation_editor_update_pending is False

    dimension_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    assert dimension_combo is not None
    rebuild_calls.clear()
    tool.eventFilter(
        dimension_combo,
        QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
    )
    assert tool._operation_editor_rebuild_must_wait()
    tool._update_current_operation_rebuild(slice_values=(0.0, 1.0))

    assert tool._operation_editor_update_pending is True
    qtbot.wait(100)
    assert rebuild_calls == []
    qtbot.waitUntil(lambda: rebuild_calls == [None], timeout=1000)
    assert tool._operation_editor_update_pending is False

    dimension_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    assert dimension_combo is not None
    rebuild_calls.clear()
    _activate_combo_text(dimension_combo, "kx")

    assert tool.tool_status.operations[0].slice_dim == "kx"
    assert rebuild_calls == []
    assert tool._operation_editor_update_pending is True
    qtbot.waitUntil(lambda: rebuild_calls == [None], timeout=1000)
    assert tool._operation_editor_update_pending is False


def test_figure_composer_plot_slices_qsel_kwargs_display_in_selection(qtbot) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(
        update={
            "extra_kwargs": {
                "eV": 0.0,
                "eV_width": 0.1,
                "beta": slice(-0.5, 0.5),
                "zorder": 2,
            },
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    dimension_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    values_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    width_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesWidthEdit"
    )
    slice_kwargs_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert dimension_combo is not None
    assert dimension_combo.currentText() == "eV"
    assert values_edit is not None
    assert values_edit.text() == "0"
    assert width_edit is not None
    assert width_edit.text() == "0.1"
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"

    tool._select_step_section("advanced")
    extra_kwargs_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerExtraKwEdit"
    )
    assert extra_kwargs_edit is not None
    assert extra_kwargs_edit.text() == "zorder=2"
    code = tool.generated_code()
    assert "eV=[0.0]" in code
    assert "eV_width=0.1" in code
    assert "beta=slice(-0.5, 0.5)" in code


def test_figure_composer_plot_slices_advanced_qsel_kwargs_move_to_selection(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="image",
                    sources=("data",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("advanced")
    extra_kwargs_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerExtraKwEdit"
    )
    assert extra_kwargs_edit is not None
    extra_kwargs_edit.setText("eV=0.5, eV_width=0.25, beta=slice(-0.5, 0.5), zorder=2")
    extra_kwargs_edit.editingFinished.emit()

    operation = tool.tool_status.operations[0]
    assert operation.slice_dim == "eV"
    assert operation.slice_values == (0.5,)
    assert operation.slice_width == pytest.approx(0.25)
    assert operation.extra_kwargs == {"zorder": 2}
    beta_slice = operation.slice_kwargs["beta"]
    assert isinstance(beta_slice, slice)
    assert beta_slice.start == pytest.approx(-0.5)
    assert beta_slice.stop == pytest.approx(0.5)
    assert beta_slice.step is None

    qtbot.waitUntil(
        lambda: not tool._operation_editor_update_pending,
        timeout=1000,
    )
    tool._select_step_section("selection")
    slice_kwargs_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"


def test_figure_composer_retired_editor_widgets_drain_after_popup(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")
    old_page = tool.step_editor_stack.currentWidget()
    active_popup: list[QtWidgets.QWidget | None] = [None]
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "activePopupWidget",
        staticmethod(lambda: active_popup[0]),
    )

    tool._update_current_operation_rebuild(slice_values=(0.0, 1.0))
    qtbot.waitUntil(lambda: old_page in tool._retired_editor_widgets, timeout=1000)
    active_popup[0] = QtWidgets.QMenu(tool)

    qtbot.wait(150)
    assert old_page in tool._retired_editor_widgets
    assert erlab.interactive.utils.qt_is_valid(old_page)

    active_popup[0] = None
    qtbot.waitUntil(lambda: not tool._retired_editor_widgets, timeout=1000)
    qtbot.waitUntil(
        lambda: not erlab.interactive.utils.qt_is_valid(old_page),
        timeout=1000,
    )


def test_figure_composer_operation_list_event_filter_is_removed_on_close(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    viewport = tool._operation_list_viewport
    assert viewport is not None
    assert erlab.interactive.utils.qt_is_valid(viewport)

    tool._operation_multi_select_event = True
    erlab.interactive.utils.single_shot(
        tool, 0, tool._clear_operation_multi_select_event
    )
    tool.close()
    qtbot.wait(10)

    assert tool._operation_list_viewport is None
    assert tool._operation_multi_select_event is False
    assert erlab.interactive.utils.qt_is_valid(viewport)


def test_figure_composer_toolbar_panel_signal_skips_destroyed_owner(qtbot) -> None:
    class SignalSender(QtCore.QObject):
        sigChanged = QtCore.Signal(object)

    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    sender = SignalSender()
    calls: list[object] = []

    _connect_panel_editor_signal(owner, sender.sigChanged, calls.append)

    sender.sigChanged.emit("live")
    assert calls == ["live"]

    owner.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qtbot.waitUntil(
        lambda: not erlab.interactive.utils.qt_is_valid(owner),
        timeout=1000,
    )
    sender.sigChanged.emit("stale")

    assert calls == ["live"]


def test_figure_composer_erlab_method_allows_empty_text_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(update={"text_values": ("Left", "Right")}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_xlabels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(update={"text_values": ("initial",)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_ylabels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    title_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit
    )
    assert title_edit is not None
    title_edit.setPlainText("Left\n")
    assert tool.tool_status.operations[0].text_values == ("Left", "")

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    xlabel_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit
    )
    assert xlabel_edit is not None
    xlabel_edit.setPlainText("")
    assert tool.tool_status.operations[1].text_values == ("",)

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    axs = namespace["axs"]
    assert axs[0, 0].get_title() == "Left"
    assert axs[0, 1].get_title() == ""
    assert axs[0, 0].get_xlabel() == ""
    assert axs[0, 1].get_xlabel() == ""
    assert axs[0, 0].get_ylabel() == ""
    assert axs[0, 1].get_ylabel() == ""


def test_figure_composer_norm_controls_are_dynamic_and_split_kwargs(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    tool._update_operation_editor()
    tool._select_step_section("colors")

    colors_page = tool.step_editor_stack.currentWidget()
    norm_combo = colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert norm_combo is not None
    assert norm_combo.currentText() == "PowerNorm"
    assert "Default" not in [
        norm_combo.itemText(index) for index in range(norm_combo.count())
    ]
    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapGammaWidget,
            "figureComposerGammaWidget",
        )
        is not None
    )
    vmin_edit = colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVminNormEdit")
    vmax_edit = colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVmaxNormEdit")
    assert vmin_edit is not None
    assert vmax_edit is not None
    assert vmin_edit.text() == ""
    assert vmax_edit.text() == ""
    assert vmin_edit.placeholderText() == "0"
    assert vmax_edit.placeholderText() == "3"
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit")
        is None
    )

    _activate_combo_text(norm_combo, "CenteredInversePowerNorm")
    assert tool.tool_status.operations[0].norm_name == "CenteredInversePowerNorm"
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerVcenterNormEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.step_editor_stack.currentWidget()
    vcenter_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerVcenterNormEdit"
    )
    assert vcenter_edit is not None
    assert vcenter_edit.text() == ""
    assert vcenter_edit.placeholderText() == "0"

    norm_combo = colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert norm_combo is not None
    _activate_combo_text(norm_combo, "CenteredPowerNorm")
    assert tool.tool_status.operations[0].norm_name == "CenteredPowerNorm"
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapGammaWidget,
            "figureComposerGammaWidget",
        )
        is not None
    )
    vcenter_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerVcenterNormEdit"
    )
    assert vcenter_edit is not None
    assert vcenter_edit.text() == ""
    assert vcenter_edit.placeholderText() == "0"
    assert colors_page.findChild(QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit")
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVminNormEdit") is None
    )

    norm_kwargs_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
    )
    assert norm_kwargs_edit is not None
    norm_kwargs_edit.setText("halfrange=1.0, custom='extra'")
    norm_kwargs_edit.editingFinished.emit()

    assert tool.tool_status.operations[0].halfrange == 1.0
    assert tool.tool_status.operations[0].norm_kwargs == {"custom": "extra"}

    def norm_kwargs_text_updated() -> bool:
        refreshed_edit = tool.step_editor_stack.currentWidget().findChild(
            QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
        )
        return refreshed_edit is not None and refreshed_edit.text() == 'custom="extra"'

    qtbot.waitUntil(
        norm_kwargs_text_updated,
        timeout=1000,
    )
    colors_page = tool.step_editor_stack.currentWidget()
    norm_kwargs_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
    )
    assert norm_kwargs_edit is not None
    assert norm_kwargs_edit.text() == 'custom="extra"'

    colors_page = tool.step_editor_stack.currentWidget()
    norm_combo = colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert norm_combo is not None
    _activate_combo_text(norm_combo, "Normalize")
    assert tool.tool_status.operations[0].norm_name == "Normalize"
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerVminNormEdit"
            )
            is not None
            and tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
            )
            is None
        ),
        timeout=1000,
    )
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapGammaWidget,
            "figureComposerGammaWidget",
        )
        is None
    )
    assert colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVminNormEdit")
    assert colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVmaxNormEdit")
    assert colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormClipCombo")
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVcenterNormEdit")
        is None
    )


def test_figure_composer_plot_slices_color_controls_do_not_commit_on_rebuild(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="first",
                    sources=("data",),
                ).model_copy(
                    update={
                        "cmap": "viridis",
                        "norm_name": "CenteredPowerNorm",
                        "halfrange": 1.0,
                    }
                ),
                FigureOperationState.plot_slices(
                    label="second",
                    sources=("data",),
                ).model_copy(
                    update={
                        "cmap": "magma_r",
                        "norm_name": "CenteredPowerNorm",
                        "halfrange": 2.0,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("colors")
    first_page = tool.step_editor_stack.currentWidget()
    first_cmap_combo = first_page.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
    )
    assert first_cmap_combo is not None
    assert first_cmap_combo.currentText() == "viridis"

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("colors")

    assert tool.tool_status.operations[0].cmap == "viridis"
    assert tool.tool_status.operations[1].cmap == "magma_r"

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    cmap_combo = colors_page.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
    )
    halfrange_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
    )
    assert cmap_combo is not None
    assert cmap_combo.currentText() == "(multiple values)"
    assert halfrange_edit is not None
    assert halfrange_edit.text() == ""
    assert halfrange_edit.placeholderText() == "(multiple values)"
    assert [operation.cmap for operation in tool.tool_status.operations] == [
        "viridis",
        "magma_r",
    ]
    assert [operation.halfrange for operation in tool.tool_status.operations] == [
        1.0,
        2.0,
    ]

    _activate_combo_text(cmap_combo, "plasma")

    assert [operation.cmap for operation in tool.tool_status.operations] == [
        "plasma",
        "plasma_r",
    ]
    halfrange_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
    )
    assert halfrange_edit is not None
    halfrange_edit.setText("3.5")
    halfrange_edit.setModified(True)
    halfrange_edit.editingFinished.emit()
    assert [operation.halfrange for operation in tool.tool_status.operations] == [
        3.5,
        3.5,
    ]


def test_figure_composer_plot_slices_gamma_queues_preview_render(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(label="plot", sources=("data",)),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    figurecomposer_plot_slices._update_current_norm_gamma(tool, 0.75)

    assert tool.tool_status.operations[0].norm_gamma == 0.75
    assert render_calls == []
    assert tool._preview_render_update_pending

    figurecomposer_plot_slices._update_current_norm_gamma(tool, 0.5)

    assert tool.tool_status.operations[0].norm_gamma == 0.5
    assert render_calls == []
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert not tool._preview_render_update_pending


def test_figure_composer_plot_slices_line_panels_use_line_controls(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="line_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(0.0, 1.0),
                ).model_copy(
                    update={
                        "line_kw": {
                            "color": "C1",
                            "linestyle": "--",
                            "linewidth": 1.5,
                            "marker": "o",
                            "markersize": 6.0,
                            "markerfacecolor": "yellow",
                            "markeredgecolor": "black",
                            "alpha": 0.75,
                        },
                        "colorbar": "right",
                        "colorbar_kw": {"pad": 0.01},
                        "same_limits": True,
                        "norm_name": "CenteredPowerNorm",
                        "norm_gamma": 0.5,
                        "gradient": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    shape_summary = tool.findChild(
        QtWidgets.QLabel, "figureComposerPlotSlicesShapeSummary"
    )
    order_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerOrderCombo")
    assert shape_summary is not None
    assert "Input dims: eV, kx" in shape_summary.text()
    assert "Plotted dims: kx (1D line)" in shape_summary.text()
    assert "Targets:" not in shape_summary.text()
    assert "Selection:" not in shape_summary.text()
    assert order_combo is not None

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    line_color_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
    )
    line_style_combo = colors_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesLineStyleCombo"
    )
    line_width_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerPlotSlicesLineWidthSpin"
    )
    marker_combo = colors_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesMarkerCombo"
    )
    marker_size_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerPlotSlicesMarkerSizeSpin"
    )
    marker_face_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesMarkerFaceColorEdit"
    )
    marker_edge_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesMarkerEdgeColorEdit"
    )
    line_kwargs_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineKwEdit"
    )
    gradient_check = colors_page.findChild(
        QtWidgets.QCheckBox, "figureComposerGradientCheck"
    )
    assert line_color_edit is not None
    assert line_color_edit.text() == "C1"
    assert line_style_combo is not None
    assert line_style_combo.currentText() == "--"
    assert line_width_spin is not None
    assert line_width_spin.value() == 1.5
    assert marker_combo is not None
    assert marker_combo.currentText() == "o"
    assert marker_size_spin is not None
    assert marker_size_spin.value() == 6.0
    assert marker_face_edit is not None
    assert marker_face_edit.text() == "yellow"
    assert marker_edge_edit is not None
    assert marker_edge_edit.text() == "black"
    assert line_kwargs_edit is not None
    assert line_kwargs_edit.text() == "alpha=0.75"
    assert gradient_check is not None
    assert gradient_check.isChecked()
    assert colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo") is None
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerColorbarKwEdit")
        is None
    )

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(QtWidgets.QComboBox, "figureComposerSameLimitsCombo")
        is None
    )

    code = tool.generated_code()
    assert "line_kw" in code
    assert "cmap=" not in code
    assert "gradient=True" in code
    assert "colorbar" not in code
    assert "same_limits" not in code
    assert "norm=" not in code
    assert "gamma=" not in code
    assert "import matplotlib.colors as mcolors" not in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    assert len(axs[0, 0].lines) == 1
    assert len(axs[0, 1].lines) == 1
    line = axs[0, 0].lines[0]
    assert line.get_color() == "C1"
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == 1.5
    assert line.get_marker() == "o"
    assert line.get_markersize() == 6.0
    assert line.get_markerfacecolor() == "yellow"
    assert line.get_markeredgecolor() == "black"
    assert line.get_alpha() == 0.75


def test_figure_composer_plot_slices_line_transforms_codegen_executes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
        slice_width=0.1,
    ).model_copy(
        update={
            "line_normalize": "max",
            "line_scales": (0.5, 2.0),
            "line_offsets": (1.0, -1.0),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert "transform" in tool.step_section_keys
    tool._select_step_section("transform")
    transform_page = tool.step_editor_stack.currentWidget()
    assert transform_page.objectName() == "figureComposerPlotSlicesTransformPage"
    assert (
        transform_page.findChild(
            QtWidgets.QWidget, "figureComposerPlotSlicesLineTransformGroup"
        )
        is None
    )
    assert transform_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
    )
    assert transform_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineScalesEdit"
    )
    assert transform_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineOffsetsEdit"
    )
    assert (
        transform_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")
        is None
    )

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
        )
        is None
    )
    assert colors_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")

    kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool, tool.tool_status.operations[0]
    )
    assert "line_normalize" not in kwargs
    assert "line_scales" not in kwargs
    assert "line_offsets" not in kwargs

    code = tool.generated_code()
    assert "import xarray as xr" in code
    assert "data.qsel(eV=0.0, eV_width=0.1)" in code
    assert "data.qsel(eV=0.0, eV_width=0.1).squeeze(drop=True)" not in code
    assert "profile_scales =" not in code
    assert "profile_offsets =" not in code
    assert "[0.5, 2.0]," in code
    assert "[1.0, -1.0]," in code
    assert (
        "profiles = [\n    offset + scale * (profile / profile.max(skipna=True))"
    ) in code
    assert "xr.IndexVariable" not in code
    assert "plot_map =" not in code
    assert "plot_maps" not in code
    assert "eplt.plot_slices(\n    xr.concat(" in code
    assert 'dim="eV"' in code
    assert 'coords="different"' in code
    assert 'compat="equals"' in code
    assert '.assign_coords({"eV": [0.0, 1.0]})' in code
    assert "eV_width" not in code.split("eplt.plot_slices(", maxsplit=1)[1]

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    np.testing.assert_allclose(axs[0, 0].lines[0].get_ydata(), [1.25, 1.5])
    np.testing.assert_allclose(axs[0, 1].lines[0].get_ydata(), [0.0, 1.0])


def test_figure_composer_plot_slices_line_transform_rejects_zero_scale(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.zeros((2, 2)),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"line_normalize": "max"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        tool.generated_code()


def test_figure_composer_plot_slices_line_panels_ignore_image_cmap(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"cmap": "magma"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    line_color_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
    )
    assert line_color_edit is not None
    assert line_color_edit.text() == ""
    kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool, tool.tool_status.operations[0]
    )
    assert "cmap" not in kwargs
    assert "line_kw" not in kwargs


def test_figure_composer_plot_slices_mixed_image_line_batch_hides_color_controls(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "kx", "ky"),
        coords={
            "eV": [0.0, 1.0],
            "kx": [0.0, 1.0, 2.0],
            "ky": [0.0, 1.0, 2.0, 3.0],
        },
        name="data",
    )
    image_operation = FigureOperationState.plot_slices(
        label="image_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"cmap": "magma", "same_limits": True})
    line_operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(
        update={
            "slice_kwargs": {"kx": 1.0},
            "line_kw": {"color": "C1"},
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(image_operation, line_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()

    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
        )
        is None
    )
    assert (
        colors_page.findChild(
            QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
        )
        is None
    )
    assert colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo") is None
    assert (
        colors_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")
        is None
    )
    mixed_label = colors_page.findChild(
        QtWidgets.QLabel, "figureComposerPlotSlicesMixedColorsLabel"
    )
    assert mixed_label is not None

    tool._select_step_section("view")
    view_page = tool.step_editor_stack.currentWidget()
    assert view_page.findChild(QtWidgets.QComboBox, "figureComposerAxisCombo")
    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(QtWidgets.QComboBox, "figureComposerSameLimitsCombo")
        is None
    )

    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    assert selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )


def test_figure_composer_plot_slices_image_panels_hide_line_transforms(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="image_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(
        update={
            "line_normalize": "max",
            "line_scales": (2.0,),
            "line_offsets": (1.0,),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert "transform" not in tool.step_section_keys
    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
        )
        is None
    )
    code = tool.generated_code()
    assert "profile_scales" not in code
    assert "profile_offsets" not in code
    assert "plot_maps" not in code


def test_figure_composer_plot_slices_image_panel_styles_codegen_executes(
    qtbot,
) -> None:
    import matplotlib.colors as mcolors

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="image_panels",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="viridis",
                    norm_name="Normalize",
                    vmin=0.0,
                    vmax=5.0,
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="magma_r",
                    norm_name="CenteredPowerNorm",
                    norm_gamma=0.5,
                    halfrange=1.0,
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool, tool.tool_status.operations[0]
    )
    assert kwargs["cmap"] == [["viridis", "magma_r"]]
    assert isinstance(kwargs["norm"][0][0], mcolors.Normalize)
    assert isinstance(kwargs["norm"][0][1], eplt.CenteredPowerNorm)

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    axs = namespace["axs"]
    assert axs[0, 0].images[0].cmap.name == "viridis"
    assert axs[0, 1].images[0].cmap.name == "magma_r"
    assert isinstance(axs[0, 0].images[0].norm, mcolors.Normalize)
    assert isinstance(axs[0, 1].images[0].norm, eplt.CenteredPowerNorm)
    assert axs[0, 1].images[0].norm.gamma == 0.5
    assert axs[0, 1].images[0].norm.halfrange == 1.0


def test_figure_composer_plot_slices_line_panel_styles_codegen_executes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_panels",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (1, 0))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "order": "F",
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    line_kw={"color": "red", "linestyle": "--"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"color": "blue", "marker": "o", "linewidth": 2.0},
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    kwargs = figurecomposer_plot_slices._plot_slices_kwargs(
        tool, tool.tool_status.operations[0]
    )
    assert kwargs["line_kw"] == [
        [{"color": "red", "linestyle": "--"}],
        [{"color": "blue", "marker": "o", "linewidth": 2.0}],
    ]
    assert kwargs["line_order"] == "F"

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    first_line = namespace["axs"][0, 0].lines[0]
    second_line = namespace["axs"][1, 0].lines[0]
    assert first_line.get_color() == "red"
    assert first_line.get_linestyle() == "--"
    assert second_line.get_color() == "blue"
    assert second_line.get_marker() == "o"
    assert second_line.get_linewidth() == 2.0


def test_figure_composer_plot_slices_line_panel_style_editor_updates_recipe(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="line_panels",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(0.0, 1.0),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool._select_step_section("colors")

    colors_page = tool.step_editor_stack.currentWidget()
    panel_check = colors_page.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesPanelStylesCheck"
    )
    assert panel_check is not None
    panel_check.setChecked(True)
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QWidget, "figureComposerPlotSlicesPanelLineStyleEditor"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.step_editor_stack.currentWidget()
    panel_list = colors_page.findChild(
        QtWidgets.QListWidget, "figureComposerPlotSlicesPanelLineStyleList"
    )
    color_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPanelLineColorEdit"
    )
    style_combo = colors_page.findChild(
        QtWidgets.QComboBox, "figureComposerPanelLineStyleCombo"
    )
    assert panel_list is not None
    assert color_edit is not None
    assert style_combo is not None
    panel_list.setCurrentRow(1)
    color_edit.setText("tab:blue")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    _activate_combo_text(style_combo, "--")

    styles = tool.tool_status.operations[0].panel_styles
    assert styles == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=1,
            line_kw={"color": "tab:blue", "linestyle": "--"},
        ),
    )


def test_figure_composer_dict_inputs_prefer_keyword_form(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="image",
                    sources=("data",),
                ).model_copy(
                    update={
                        "annotate_kw": {"fontsize": 8, "color": "black"},
                        "colorbar_kw": {"fraction": 0.05, "pad": 0.02},
                        "norm_kwargs": {"custom": "extra"},
                        "extra_kwargs": {"alpha": 0.5, "zorder": 2},
                    }
                ),
                FigureOperationState.plot_slices(
                    label="line",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "gradient_kw": {"color": "C0", "alpha": 0.25},
                    }
                ),
                FigureOperationState.line(
                    label="profile",
                    source="data",
                ).model_copy(update={"line_selection": {"eV": 0.0, "eV_width": 0.1}}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                ).model_copy(update={"method_kwargs": {"fontsize": 9}}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("view")
    annotate_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAnnotateKwEdit"
    )
    assert annotate_kwargs_edit is not None
    assert annotate_kwargs_edit.text() == 'fontsize=8, color="black"'

    tool._select_step_section("colors")
    colorbar_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerColorbarKwEdit"
    )
    norm_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
    )
    assert colorbar_kwargs_edit is not None
    assert colorbar_kwargs_edit.text() == "fraction=0.05, pad=0.02"
    assert norm_kwargs_edit is not None
    assert norm_kwargs_edit.text() == 'custom="extra"'

    tool._select_step_section("advanced")
    extra_kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerExtraKwEdit")
    assert extra_kwargs_edit is not None
    assert extra_kwargs_edit.text() == "alpha=0.5, zorder=2"

    _select_operation_rows(tool, (1,))
    tool._select_step_section("colors")
    gradient_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerGradientKwEdit"
    )
    assert gradient_kwargs_edit is not None
    assert gradient_kwargs_edit.text() == 'color="C0", alpha=0.25'

    _select_operation_rows(tool, (2,))
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    assert (
        selection_page.findChild(
            QtWidgets.QComboBox, "figureComposerProfileReduceCombo"
        )
        is None
    )
    line_selection_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineSelectionEdit"
    )
    assert line_selection_edit is not None
    assert line_selection_edit.text() == "eV=0.0, eV_width=0.1"
    line_selection_edit.setText("eV=slice(0.0, 1.0), kx=0.0")
    line_selection_edit.editingFinished.emit()
    assert tool.tool_status.operations[2].line_selection == {
        "eV": slice(0.0, 1.0),
        "kx": 0.0,
    }

    tool.operation_list.setCurrentRow(3)
    tool._select_step_section("method")
    erlab_method_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit"
    )
    assert erlab_method_kwargs_edit is not None
    assert erlab_method_kwargs_edit.text() == "fontsize=9"


def test_figure_composer_imagetool_norm_parser_uses_structured_fields() -> None:
    from erlab.interactive.imagetool.plot_items import ItoolPlotItem

    updates = ItoolPlotItem._figure_composer_norm_updates(
        "|eplt.CenteredPowerNorm(0.5, vcenter=0.0, halfrange=1.0)|"
    )

    assert updates == {
        "norm_kwargs": {},
        "norm_name": "CenteredPowerNorm",
        "norm_gamma": 0.5,
        "vcenter": 0.0,
        "halfrange": 1.0,
    }


def test_figure_composer_imagetool_value_and_norm_parsers_cover_edges() -> None:
    from erlab.interactive.imagetool.plot_items import ItoolPlotItem

    class Floatable:
        def __float__(self) -> float:
            return 1.5

    fallback = object()
    assert ItoolPlotItem._figure_composer_plain_value(None) is None
    assert ItoolPlotItem._figure_composer_plain_value(np.bool_(True)) is True
    assert ItoolPlotItem._figure_composer_plain_value([np.int64(1)]) == [1]
    assert ItoolPlotItem._figure_composer_plain_value((np.float64(2.0),)) == (2.0,)
    assert ItoolPlotItem._figure_composer_plain_value({"a": np.int64(3)}) == {"a": 3}
    assert ItoolPlotItem._figure_composer_plain_value(np.int64(4)) == 4
    assert ItoolPlotItem._figure_composer_plain_value(np.float64(5.0)) == 5.0
    assert ItoolPlotItem._figure_composer_plain_value(Floatable()) == 1.5
    assert ItoolPlotItem._figure_composer_plain_value(fallback) is fallback
    assert ItoolPlotItem._figure_composer_indexer_state(slice(1, 3, 2)) == {
        "kind": "slice",
        "start": 1,
        "stop": 3,
        "step": 2,
    }
    assert ItoolPlotItem._figure_composer_indexer_state(2) == 2

    invalid_norms = (
        "not valid python",
        "1",
        "eplt",
        "mcolors.PowerNorm(1)",
        "eplt.PowerNorm(1)",
        "eplt.CenteredPowerNorm(**kwargs)",
    )
    for norm_code in invalid_norms:
        assert ItoolPlotItem._figure_composer_norm_updates(norm_code) is None

    assert ItoolPlotItem._figure_composer_operation_updates({"norm": object()}) is None
    assert (
        ItoolPlotItem._figure_composer_operation_updates({"norm": "eplt.PowerNorm(1)"})
        is None
    )
    assert (
        ItoolPlotItem._figure_composer_operation_updates({"cmap": "|dynamic_cmap|"})
        is None
    )
    assert ItoolPlotItem._figure_composer_operation_updates(
        {"gamma": np.float64(0.5), "alpha": np.int64(2)}
    ) == {
        "norm_name": "PowerNorm",
        "norm_gamma": 0.5,
        "extra_kwargs": {"alpha": 2},
    }


def test_figure_composer_imagetool_operation_seed_helpers_cover_branches() -> None:
    import types

    from erlab.interactive.imagetool.plot_items import ItoolPlotItem

    fake = types.SimpleNamespace(
        slicer_area=types.SimpleNamespace(current_cursor=0, n_cursors=2),
        slicer_data_items=[types.SimpleNamespace(normalize=True)],
        _crop_indexers={"kx": slice(-1.0, 1.0)},
    )
    fake._figure_composer_operation_updates = (
        ItoolPlotItem._figure_composer_operation_updates
    )
    fake._figure_composer_plain_value = ItoolPlotItem._figure_composer_plain_value
    fake._figure_composer_plot_slices_kwargs = lambda _dim_order_plot: {
        "gamma": 0.5,
        "colorbar": "right",
    }
    fake._figure_composer_line_style_updates = lambda: {"line_colors": ("red", "blue")}
    fake._figure_composer_line_limit_updates = lambda x_dim: (
        {"xlim": (-1.0, 1.0)} if x_dim == "kx" else {}
    )

    assert ItoolPlotItem._plot_slices_qsel_key_is_editable("eV")
    assert ItoolPlotItem._plot_slices_qsel_key_is_editable("eV_width")
    assert ItoolPlotItem._plot_slices_qsel_key_is_editable("Track Shift")
    assert not ItoolPlotItem._plot_slices_qsel_key_is_editable("sample_temp_idx")
    assert not ItoolPlotItem._plot_slices_qsel_key_is_editable("sample_temp_idx_width")
    assert not ItoolPlotItem._plot_slices_qsel_key_is_editable(("bad", "key"))

    selected_maps_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim=None,
        dim_order_plot=["kx", "ky"],
        selected_maps=["data"],
        map_selections=(FigureDataSelectionState(source="data", qsel={"eV": 0.0}),),
    )
    assert selected_maps_operation.map_selections == (
        FigureDataSelectionState(source="data", qsel={"eV": 0.0}),
    )
    assert selected_maps_operation.norm_name == "PowerNorm"
    assert selected_maps_operation.norm_gamma == 0.5
    assert selected_maps_operation.colorbar == "right"

    no_selection_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim=None,
        dim_order_plot=["kx", "ky"],
        qsel_kwargs=None,
    )
    assert no_selection_operation.map_selections == ()

    invalid_key_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        dim_order_plot=["kx", "ky"],
        qsel_kwargs={("bad", "key"): [0.0, 1.0]},
    )
    assert tuple(
        selection.qsel for selection in invalid_key_operation.map_selections
    ) == (
        {"('bad', 'key')": 0.0},
        {"('bad', 'key')": 1.0},
    )

    non_identifier_key_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim=None,
        dim_order_plot=["kx", "ky"],
        qsel_kwargs={"Track Shift": 2.0},
    )
    assert non_identifier_key_operation.map_selections == ()
    assert non_identifier_key_operation.slice_dim == "Track Shift"
    assert non_identifier_key_operation.slice_values == (2.0,)

    slice_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        dim_order_plot=["kx", "ky"],
        qsel_kwargs={"eV": [0.0, 1.0], "eV_width": [0.1, 0.1], "beta": 2.0},
    )
    assert slice_operation.slice_dim == "eV"
    assert slice_operation.slice_values == (0.0, 1.0)
    assert slice_operation.slice_width == 0.1
    assert slice_operation.slice_kwargs == {"beta": 2.0}

    unequal_width_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        dim_order_plot=["kx", "ky"],
        qsel_kwargs={"eV": [0.0, 1.0], "eV_width": [0.1, 0.2]},
    )
    assert unequal_width_operation.slice_width is None
    assert unequal_width_operation.slice_kwargs == {"eV_width": [0.1, 0.2]}

    inferred_slice_operation = ItoolPlotItem._figure_composer_plot_slices_operation(
        fake,
        source_name="data",
        variable_dim=None,
        dim_order_plot=["kx", "ky"],
        qsel_kwargs={"beta": [0.0, 1.0], "beta_width": [0.2, 0.2]},
    )
    assert inferred_slice_operation.slice_dim == "beta"
    assert inferred_slice_operation.slice_values == (0.0, 1.0)
    assert inferred_slice_operation.slice_width == 0.2
    assert inferred_slice_operation.slice_kwargs == {}

    selected_lines_operation = ItoolPlotItem._figure_composer_line_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        x_dim="kx",
        selected_lines=["data"],
        map_selections=(FigureDataSelectionState(source="data", qsel={"eV": 0.0}),),
    )
    assert selected_lines_operation.line_x == "kx"
    assert selected_lines_operation.map_selections == (
        FigureDataSelectionState(source="data", qsel={"eV": 0.0}),
    )
    assert selected_lines_operation.line_normalize == "mean"
    assert selected_lines_operation.line_colors == ("red", "blue")
    assert selected_lines_operation.xlim == (-1.0, 1.0)

    no_qsel_line_operation = ItoolPlotItem._figure_composer_line_operation(
        fake,
        source_name="data",
        variable_dim=None,
        x_dim="kx",
        qsel_kwargs=None,
    )
    assert no_qsel_line_operation.line_selection == {}
    assert no_qsel_line_operation.line_iter_dim is None

    qsel_line_operation = ItoolPlotItem._figure_composer_line_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        x_dim="kx",
        qsel_kwargs={"eV": [0.0, 1.0], "beta": 2.0},
    )
    assert qsel_line_operation.line_selection == {"eV": [0.0, 1.0], "beta": 2.0}
    assert qsel_line_operation.line_iter_dim == "eV"

    non_identifier_key_line_operation = ItoolPlotItem._figure_composer_line_operation(
        fake,
        source_name="data",
        variable_dim="Track Shift",
        x_dim="kx",
        qsel_kwargs={"Track Shift": [0.0, 1.0], "beta": 2.0},
    )
    assert non_identifier_key_line_operation.map_selections == ()
    assert non_identifier_key_line_operation.line_selection == {
        "Track Shift": [0.0, 1.0],
        "beta": 2.0,
    }
    assert non_identifier_key_line_operation.line_iter_dim == "Track Shift"

    invalid_key_line_operation = ItoolPlotItem._figure_composer_line_operation(
        fake,
        source_name="data",
        variable_dim="eV",
        x_dim="kx",
        qsel_kwargs={("bad", "key"): [0.0, 1.0]},
    )
    assert tuple(
        selection.qsel for selection in invalid_key_line_operation.map_selections
    ) == (
        {"('bad', 'key')": 0.0},
        {"('bad', 'key')": 1.0},
    )


def test_figure_composer_plot_slices_spaced_qsel_dimension_codegen_executes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("Track Shift", "kx", "ky"),
        coords={
            "Track Shift": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0],
            "ky": [0.0, 1.0],
        },
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="spaced",
        sources=("data",),
        slice_dim="Track Shift",
        slice_values=(1.0,),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    _render_figure_composer_rgba(tool)
    code = tool.generated_code()
    assert "Track Shift" in code
    assert "**{" in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["axs"][0, 0].images) == 1


def test_figure_composer_powernorm_codegen_uses_plot_kwargs(qtbot) -> None:
    import matplotlib.colors as mcolors

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="power",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "norm_name": "PowerNorm",
                        "norm_gamma": 0.5,
                        "vmin": 0.0,
                        "vmax": 10.0,
                    }
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" not in code
    assert "norm=mcolors.PowerNorm" not in code
    assert "gamma=0.5" in code
    assert "vmin=0.0" in code
    assert "vmax=10.0" in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    norm = namespace["axs"][0, 0].images[0].norm
    assert isinstance(norm, mcolors.PowerNorm)
    assert norm.gamma == 0.5
    assert norm.vmin == 0.0
    assert norm.vmax == 10.0


def test_figure_composer_explicit_norm_codegen_executes(qtbot) -> None:
    import matplotlib.colors as mcolors

    import erlab.plotting as eplt

    data = xr.DataArray(
        np.arange(24.0).reshape(3, 4, 2),
        dims=("eV", "kx", "ky"),
        coords={
            "eV": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0, 2.0, 3.0],
            "ky": [0.0, 1.0],
        },
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="power",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "norm_name": "PowerNorm",
                        "norm_gamma": 0.5,
                        "vmin": 0.0,
                        "vmax": 10.0,
                        "norm_clip": True,
                    }
                ),
                FigureOperationState.plot_slices(
                    label="inverse",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(1.0,),
                ).model_copy(
                    update={
                        "norm_name": "InversePowerNorm",
                        "norm_gamma": 0.75,
                        "vmin": 0.0,
                        "vmax": 20.0,
                    }
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "norm=mcolors.PowerNorm" in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102

    left_norm = namespace["axs"][0, 0].images[0].norm
    right_norm = namespace["axs"][0, 1].images[0].norm
    assert isinstance(left_norm, mcolors.PowerNorm)
    assert left_norm.gamma == 0.5
    assert left_norm.vmin == 0.0
    assert left_norm.vmax == 10.0
    assert left_norm.clip is True
    assert isinstance(right_norm, eplt.InversePowerNorm)
    assert right_norm.gamma == 0.75
    assert right_norm.vmin == 0.0
    assert right_norm.vmax == 20.0


def test_figure_composer_layout_change_marks_removed_axes(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((1, 1),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.editor_tabs.setCurrentWidget(tool.layout_page)
    tool.nrows_spin.setValue(1)

    assert tool.editor_tabs.currentWidget() is tool.layout_page
    assert tool.tool_status.setup.nrows == 1
    assert tool.tool_status.operations[0].axes.axes == ((1, 1),)
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert tool._current_step_section_key == "sources"
    tool._select_step_section("axes")
    assert tool.keep_valid_axes_button.isEnabled()
    with pytest.raises(ValueError, match="Cannot generate code"):
        tool.generated_code()

    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append(text),
    )
    tool.copy_code()
    assert warnings

    tool.use_all_axes_button.click()

    assert tool.tool_status.operations[0].axes.axes == ((0, 0), (0, 1))
    assert not tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert "axes=axs" in tool.generated_code()


def test_figure_composer_axes_selector_click_keeps_surviving_removed_axes_target(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (1, 1))),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.nrows_spin.setValue(1)
    tool.editor_tabs.setCurrentWidget(tool.recipe_page)
    tool._select_step_section("axes")
    tool.axes_selector.resize(tool.axes_selector.sizeHint())

    assert tool.tool_status.operations[0].axes.axes == ((0, 0), (1, 1))
    assert tool.axes_selector.selected_axes() == ((0, 0),)
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])

    qtbot.mouseClick(
        tool.axes_selector,
        QtCore.Qt.MouseButton.LeftButton,
        pos=tool.axes_selector.cell_rect((0, 0)).center(),
    )

    assert tool.tool_status.operations[0].axes.axes == ((0, 0),)
    assert not tool._operation_has_invalid_axes(tool.tool_status.operations[0])


def test_manager_figures_ui_is_lazy_and_figures_survive_source_removal(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        assert not manager.left_tabs.tabBar().isVisible()
        assert not manager.left_tabs.isTabVisible(1)
        assert not hasattr(manager, "figure_tab")
        assert not hasattr(manager, "figure_view_controls")

        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager.create_figure_action.trigger()
        assert len(manager._tool_graph.figure_uids) == 1
        figure_uid = manager._tool_graph.figure_uids[0]
        assert manager.left_tabs.tabBar().isVisible()
        assert manager.left_tabs.indexOf(manager.figure_tab) == 1
        assert manager.left_tabs.isTabVisible(1)
        assert figure_uid in manager._tool_graph.figure_uids
        assert figure_uid not in manager._tool_graph.root_wrappers[0]._childtool_indices

        manager.remove_imagetool(0)
        assert figure_uid in manager._tool_graph.nodes
        assert manager.figure_list.count() == 1
        assert manager.dependency_status_for_uid(figure_uid) == "missing"
        assert manager.left_tabs.tabBar().isVisible()

        manager._remove_childtool(figure_uid)
        assert figure_uid not in manager._tool_graph.nodes
        assert not manager.left_tabs.tabBar().isVisible()
        assert not manager.left_tabs.isTabVisible(1)
        assert not hasattr(manager, "figure_tab")


def test_manager_tool_selection_clears_stale_figure_selection_details(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None

        manager._select_figure_uid(figure_uid)

        assert manager._selected_tool_uids() == [figure_uid]

        select_tools(manager, [0])

        root_uid = manager._tool_graph.root_wrappers[0].uid
        assert manager._selected_imagetool_targets() == [0]
        assert manager._selected_tool_uids() == []
        assert manager._selected_figure_uids() == []
        assert manager._metadata_node_uid == root_uid


def test_imagetool_plot_with_matplotlib_warns_for_uneditable_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(tool)

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        tool.slicer_area.images[0].plot_with_matplotlib()

        assert warnings
        assert len(manager._tool_graph.figure_uids) == 0


def test_manager_figure_action_warns_for_uneditable_plot_slices_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(tool)
        select_tools(manager, [0])

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        manager.create_figure_action.trigger()

        assert warnings
        assert len(manager._tool_graph.figure_uids) == 0


def test_manager_append_figure_warns_for_uneditable_plot_slices_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        source_tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(source_tool)

        figure_data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="existing",
        )
        figure_tool = FigureComposerTool(
            figure_data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="existing", label="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        assert appended is False
        assert warnings
        assert figure_tool.tool_status.operations == ()


def test_manager_figures_tab_does_not_set_empty_minimum_width(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        manager.show()
        empty_width = manager.left_tabs.minimumSizeHint().width()

        assert manager.left_tabs.count() == 1
        assert not hasattr(manager, "figure_tab")

        figure_uid = manager.add_figuretool(FigureComposerTool(data), show=False)

        assert manager.left_tabs.count() == 2
        assert manager.left_tabs.indexOf(manager.figure_tab) == 1
        manager._show_figure_menu(QtCore.QPoint())
        menu = manager._figure_menu
        assert isinstance(menu, QtWidgets.QMenu)
        qtbot.wait_until(menu.isVisible, timeout=5000)

        manager._remove_childtool(figure_uid)
        qtbot.wait_until(lambda: manager._figure_menu is None, timeout=5000)

        assert manager.left_tabs.count() == 1
        assert not hasattr(manager, "figure_tab")
        assert manager.left_tabs.minimumSizeHint().width() == empty_width

        manager._clear_figure_selection_from_tree()
        manager._figure_selection_changed()
        manager._show_figure_item(QtWidgets.QListWidgetItem("removed"))
        manager._show_figure_menu(QtCore.QPoint())


def test_manager_figure_menu_helpers_release_stale_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._close_figure_menu()
        assert manager._figure_menu is None

        menu = QtWidgets.QMenu(manager)
        manager._figure_menu = menu
        manager._close_figure_menu()
        assert manager._figure_menu is None

        stale_menu = QtWidgets.QMenu(manager)
        manager._figure_menu = stale_menu
        original_qt_is_valid = erlab.interactive.utils.qt_is_valid

        def fake_qt_is_valid(*objects: object) -> bool:
            if any(obj is stale_menu for obj in objects):
                return False
            return original_qt_is_valid(*objects)

        with monkeypatch.context() as patch:
            patch.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
            manager._close_figure_menu()
            assert manager._figure_menu is None
            manager._release_figure_menu(stale_menu)


def test_manager_figure_menu_releases_menu_without_viewport(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        manager.add_figuretool(FigureComposerTool(data), show=False)
        monkeypatch.setattr(type(manager.figure_list), "viewport", lambda _self: None)

        manager._show_figure_menu(QtCore.QPoint())

        assert manager._figure_menu is None


def _selection_shortcut_sequences(widget: QtWidgets.QWidget) -> set[str]:
    return {
        shortcut.key().toString(QtGui.QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QtWidgets.QShortcut)
        if shortcut.parent() is widget
    }


@pytest.mark.parametrize(
    ("platform", "rename_shortcut", "show_shortcut", "expected_shortcuts"),
    [
        (
            "darwin",
            "Return",
            "Ctrl+Down",
            {"Return", "Enter", "Ctrl+Down"},
        ),
        (
            "linux",
            "F2",
            "Return",
            {"F2", "Return", "Enter"},
        ),
    ],
)
def test_manager_figure_list_selection_shortcuts_are_platform_native(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    platform: str,
    rename_shortcut: str,
    show_shortcut: str,
    expected_shortcuts: set[str],
) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", platform)
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )

    with manager_context() as manager:
        figure_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        manager._select_figure_uid(figure_uid)

        assert manager.show_action.shortcut().isEmpty()
        assert _selection_shortcut_sequences(manager.figure_list) == expected_shortcuts

        activate_widget_shortcut(manager.figure_list, rename_shortcut)
        qtbot.wait_until(
            lambda: (
                manager.figure_list.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        editor = manager.figure_list.findChild(QtWidgets.QLineEdit)
        assert editor is not None
        editor.setText(f"{platform}_figure")
        qtbot.keyClick(editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._child_node(figure_uid).name == f"{platform}_figure",
            timeout=5000,
        )

        shown: list[str] = []
        monkeypatch.setattr(manager, "show_childtool", shown.append)
        activate_widget_shortcut(manager.figure_list, show_shortcut)
        qtbot.wait_until(lambda: shown == [figure_uid], timeout=5000)


def test_manager_figures_gallery_view_preserves_selection_and_persists(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        first_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        second_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        manager._select_figure_uid(first_uid)

        assert manager.figure_list.viewMode() == QtWidgets.QListView.ViewMode.IconMode
        assert (
            manager.figure_list.verticalScrollMode()
            == QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        assert manager.figure_gallery_size_combo.isVisible()
        assert manager.figure_view_list_button.text() == ""
        assert manager.figure_view_gallery_button.text() == ""
        assert not manager.figure_view_list_button.icon().isNull()
        assert not manager.figure_view_gallery_button.icon().isNull()

        manager.figure_view_list_button.click()
        assert manager.figure_list.viewMode() == QtWidgets.QListView.ViewMode.ListMode
        assert manager.figure_gallery_size_combo.isHidden()
        manager.figure_view_gallery_button.click()

        assert manager.figure_list.viewMode() == QtWidgets.QListView.ViewMode.IconMode
        assert (
            manager.figure_list.verticalScrollMode()
            == QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        assert manager.figure_gallery_size_combo.isVisible()
        assert manager._selected_figure_uids() == [first_uid]
        for row, uid in enumerate((first_uid, second_uid)):
            item = manager.figure_list.item(row)
            assert item is not None
            assert item.data(QtCore.Qt.ItemDataRole.UserRole) == uid
            assert not item.icon().isNull()

        old_grid_size = manager.figure_list.gridSize()
        large_index = manager.figure_gallery_size_combo.findData("large")
        assert large_index >= 0
        manager.figure_gallery_size_combo.setCurrentIndex(large_index)
        assert manager.figure_list.gridSize().width() > old_grid_size.width()
        assert manager._selected_figure_uids() == [first_uid]

        shown: list[str] = []
        monkeypatch.setattr(manager, "show_childtool", shown.append)
        manager._show_figure_item(manager.figure_list.item(0))
        assert shown == [first_uid]

        manager.figure_view_list_button.click()
        assert manager.figure_list.viewMode() == QtWidgets.QListView.ViewMode.ListMode
        assert manager.figure_gallery_size_combo.isHidden()
        manager.figure_view_gallery_button.click()

    with manager_context() as restored_manager:
        assert restored_manager._figure_view_mode == "gallery"
        assert restored_manager._figure_gallery_thumbnail_size_name == "large"
        restored_uid = restored_manager.add_figuretool(
            FigureComposerTool(data), show=False
        )
        assert (
            restored_manager.figure_list.viewMode()
            == QtWidgets.QListView.ViewMode.IconMode
        )
        item = restored_manager.figure_list.item(0)
        assert item is not None
        assert item.data(QtCore.Qt.ItemDataRole.UserRole) == restored_uid
        assert not item.icon().isNull()


def test_manager_figures_gallery_reuses_cached_preview_for_size_changes(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        manager.add_figuretool(figure_tool, show=False)
        assert figure_tool.refresh_preview_pixmap(allow_offscreen=True) is not None

        def fail_preview_update(*_args, **_kwargs) -> None:
            pytest.fail("gallery thumbnail updates must not render the recipe")

        monkeypatch.setattr(
            figure_tool, "request_preview_pixmap_update", fail_preview_update
        )
        monkeypatch.setattr(figure_tool, "refresh_preview_pixmap", fail_preview_update)

        manager.figure_view_gallery_button.click()
        old_grid_size = manager.figure_list.gridSize()
        size_name = (
            "large"
            if manager._figure_gallery_thumbnail_size_name != "large"
            else "small"
        )
        size_index = manager.figure_gallery_size_combo.findData(size_name)
        assert size_index >= 0
        manager.figure_gallery_size_combo.setCurrentIndex(size_index)

        assert manager.figure_list.gridSize() != old_grid_size


def test_figure_composer_persists_compact_preview_cache_without_rendering(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    first = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(4.0, 8.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="First"),
            FigureSourceState(name="second", label="Second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    assert tool.refresh_preview_pixmap(allow_offscreen=True) is not None

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("saving a preview cache must not render the figure")

    monkeypatch.setattr(tool, "refresh_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_canvas_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_fallback_preview_pixmap", fail_render)

    ds = tool.to_dataset()
    encoded_cache = ds.attrs.get(
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR
    )
    assert isinstance(encoded_cache, str)
    max_encoded_cache_size = 4 * (
        (figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_MAX_BYTES + 2) // 3
    )
    assert len(encoded_cache) <= max_encoded_cache_size

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", fail_render)
    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)
    restored_preview = restored.preview_pixmap
    assert restored_preview is not None
    assert not restored.preview_pixmap_stale
    xr.testing.assert_identical(restored.source_data()["second"], second)
    assert (
        restored_preview.width()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width()
    )
    assert (
        restored_preview.height()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.height()
    )
    thumbnail = restored.preview_thumbnail_pixmap(QtCore.QSize(64, 64))
    assert thumbnail is not None
    assert not thumbnail.isNull()


def test_figure_composer_visible_restore_queues_auto_redraw(
    qtbot,
    monkeypatch,
) -> None:
    first = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(4.0, 8.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="First"),
            FigureSourceState(name="second", label="Second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    render_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_render(*args, **kwargs) -> None:
        render_calls.append((args, kwargs))

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", record_render)

    ds = tool.to_dataset()
    window_state = json.loads(ds.attrs["tool_window_state"])
    window_state["visible"] = True
    ds.attrs["tool_window_state"] = json.dumps(window_state)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(ds)
    qtbot.addWidget(restored)

    qtbot.wait_until(lambda: bool(render_calls), timeout=5000)
    assert render_calls == [((restored,), {"show_window": True})]


def test_figure_composer_skips_preview_cache_when_unrendered(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    assert tool.preview_pixmap is None

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("saving without a preview cache must not render the figure")

    monkeypatch.setattr(tool, "refresh_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_canvas_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_fallback_preview_pixmap", fail_render)

    ds = tool.to_dataset()

    assert figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR not in ds.attrs
    assert tool.preview_pixmap is None


def test_manager_workspace_restores_figure_gallery_preview_cache(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    workspace_path = tmp_path / "figure-preview-cache.itws"
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        assert figure_tool.refresh_preview_pixmap(allow_offscreen=True) is not None

        manager._save_workspace_document(workspace_path, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        assert loaded_tool.preview_pixmap is not None
        assert not loaded_tool.preview_pixmap_stale

        manager.figure_view_gallery_button.click()
        item = manager._figure_list_item_for_uid(figure_uid)
        assert item is not None
        assert not item.icon().isNull()


def test_manager_figures_gallery_helpers_handle_invalid_sources(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )

    class FakeSettings:
        def value(self, key: str, default: str) -> object:
            if key.endswith("view_mode"):
                return "invalid"
            if key.endswith("gallery_thumbnail_size"):
                return "huge"
            return 1

        def setValue(self, key: str, value: str) -> None:
            pass

    with manager_context() as manager:
        monkeypatch.setattr(
            manager_mainwindow, "_manager_settings", lambda: FakeSettings()
        )
        assert manager._settings_string("unknown", "fallback") == "fallback"
        assert manager._read_figure_view_mode_setting() == "gallery"
        assert manager._read_figure_gallery_size_setting() == "medium"

        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        manager.figure_view_gallery_button.click()

        assert not manager._figure_gallery_icon("missing").isNull()
        assert manager._figure_gallery_tool_thumbnail_pixmap(object()) is None

        fallback = QtGui.QPixmap(40, 20)
        fallback.fill(QtGui.QColor("red"))

        high_dpi_preview = QtGui.QPixmap(400, 100)
        high_dpi_preview.setDevicePixelRatio(2.0)
        high_dpi_preview.fill(QtGui.QColor("red"))
        high_dpi_thumbnail = manager._figure_gallery_thumbnail_pixmap(high_dpi_preview)
        high_dpi_image = high_dpi_thumbnail.toImage().convertToFormat(
            QtGui.QImage.Format.Format_ARGB32
        )
        red_pixels: list[tuple[int, int]] = []
        for y_pos in range(high_dpi_image.height()):
            for x_pos in range(high_dpi_image.width()):
                color = high_dpi_image.pixelColor(x_pos, y_pos)
                if color.red() > 220 and color.green() < 40 and color.blue() < 40:
                    red_pixels.append((x_pos, y_pos))
        assert red_pixels
        min_x = min(x_pos for x_pos, _y_pos in red_pixels)
        max_x = max(x_pos for x_pos, _y_pos in red_pixels)
        min_y = min(y_pos for _x_pos, y_pos in red_pixels)
        max_y = max(y_pos for _x_pos, y_pos in red_pixels)
        assert high_dpi_thumbnail.size() == manager._figure_gallery_thumbnail_size()
        assert max_x - min_x + 1 == high_dpi_thumbnail.width()
        assert abs((max_y - min_y + 1) - round(high_dpi_thumbnail.width() / 4)) <= 1
        assert abs(((min_y + max_y) / 2) - ((high_dpi_thumbnail.height() - 1) / 2)) <= 1

        class NullThumbnailProvider:
            preview_pixmap = fallback

            def preview_thumbnail_pixmap(self, _size: QtCore.QSize) -> QtGui.QPixmap:
                return QtGui.QPixmap()

        assert (
            manager._figure_gallery_tool_thumbnail_pixmap(NullThumbnailProvider())
            is not None
        )

        class DirectThumbnailProvider:
            preview_pixmap = QtGui.QPixmap()

            def preview_thumbnail_pixmap(self, _size: QtCore.QSize) -> QtGui.QPixmap:
                pixmap = QtGui.QPixmap(12, 8)
                pixmap.fill(QtGui.QColor("blue"))
                return pixmap

        assert (
            manager._figure_gallery_tool_thumbnail_pixmap(DirectThumbnailProvider())
            is not None
        )

        requested: list[None] = []
        monkeypatch.setattr(
            figure_tool,
            "request_preview_pixmap_update",
            lambda: requested.append(None),
        )
        figure_tool._preview_pixmap_stale = True
        assert not manager._figure_gallery_icon(figure_uid).isNull()
        assert requested == []


def test_manager_figures_gallery_updates_one_icon_from_preview_signal(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        manager.figure_view_gallery_button.click()
        item = manager._figure_list_item_for_uid(figure_uid)
        assert item is not None
        old_cache_key = item.icon().cacheKey()

        pixmap = QtGui.QPixmap(32, 24)
        pixmap.fill(QtGui.QColor("red"))
        figure_tool._preview_pixmap_cache = pixmap
        figure_tool._preview_pixmap_generation += 1
        figure_tool._preview_thumbnail_cache.clear()
        figure_tool._preview_pixmap_stale = False

        def fail_sync(*_args, **_kwargs) -> None:
            pytest.fail("preview updates should update the changed gallery item only")

        with monkeypatch.context() as patch:
            patch.setattr(manager, "_sync_figures_ui", fail_sync)
            figure_tool.sigInfoChanged.emit()

        qtbot.wait_until(lambda: item.icon().cacheKey() != old_cache_key, timeout=1000)


def test_manager_figure_selection_defers_preview_generation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        refresh_calls: list[None] = []
        request_calls: list[int] = []

        def record_refresh() -> None:
            refresh_calls.append(None)

        def record_request(*, delay_ms: int = 250) -> None:
            request_calls.append(delay_ms)

        monkeypatch.setattr(figure_tool, "refresh_preview_pixmap", record_refresh)
        monkeypatch.setattr(
            figure_tool, "request_preview_pixmap_update", record_request
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        manager._select_figure_uid(figure_uid)

        assert refresh_calls == []
        assert request_calls == []
        assert not manager.preview_widget.isVisible()

        qtbot.waitUntil(lambda: request_calls == [0], timeout=1000)
        assert refresh_calls == []


def test_manager_copy_full_code_for_file_backed_figure_composer_sources(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
        name="first",
    )
    second = first + 10.0
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first.to_netcdf(first_path, engine="h5netcdf")
    second.to_netcdf(second_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            first,
            manager=True,
            file_path=first_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        itool(
            second,
            manager=True,
            file_path=second_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        manager.tree_view.clearSelection()
        select_child_tool(manager, figure_uid)
        manager._update_info(uid=figure_uid)
        assert manager.metadata_derivation_list.topLevelItemCount() == 4
        assert manager.metadata_derivation_list.count() == 6
        child_counts: list[int] = []
        for row in range(manager.metadata_derivation_list.topLevelItemCount()):
            item = manager.metadata_derivation_list.topLevelItem(row)
            assert item is not None
            child_counts.append(item.childCount())
        assert child_counts == [0, 1, 1, 0]

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed figure replay should not prompt"),
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        namespace = _exec_generated_code(copied[-1], {})
        assert isinstance(namespace["fig"], Figure)


def test_manager_copy_full_code_for_memory_figure_reports_unavailable(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog:
        def __init__(self, parent=None, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            dialogs.append(self)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        manager.tree_view.clearSelection()
        select_child_tool(manager, figure_uid)
        manager._update_info(uid=figure_uid)
        assert manager.metadata_derivation_list.count() == 3

        copied: list[str] = []
        monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", copied.append)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert not copied
        assert len(dialogs) == 1
        assert (
            dialogs[0].kwargs["icon_pixmap"]
            == QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        )
        assert dialogs[0].kwargs["text"]
        assert dialogs[0].kwargs["informative_text"]
        assert dialogs[0].kwargs["detailed_text"]


def test_manager_figure_action_new_target_creates_second_figure(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        first_uid = manager.create_figure_from_targets((0,), show=False)
        assert first_uid is not None
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (first_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_NEW

            def is_new_figure(self) -> bool:
                return True

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeFigureDialog
        )

        manager.create_figure_action.trigger()

        assert len(manager._tool_graph.figure_uids) == 2
        assert first_uid in manager._tool_graph.figure_uids


def test_manager_figure_action_appends_to_selected_subplots_axes(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0),
            dims=("x",),
            coords={"x": np.arange(4.0)},
            name="line",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_ADD_STEP

            def is_new_figure(self) -> bool:
                return False

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 1),))

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeFigureDialog
        )

        manager.create_figure_action.trigger()

        assert len(manager._tool_graph.figure_uids) == 1
        assert len(figure_tool.tool_status.operations) == 1
        assert figure_tool.tool_status.operations[0].axes.axes == ((0, 1),)


def test_manager_figure_action_appends_to_selected_gridspec_axes(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0),
            dims=("x",),
            coords={"x": np.arange(4.0)},
            name="line",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(
                    layout_mode="gridspec",
                    gridspec=FigureGridSpecLayoutState(
                        root=FigureGridSpecGridState(
                            grid_id="root",
                            nrows=1,
                            ncols=1,
                            axes=(
                                FigureGridSpecAxesState(
                                    axes_id="axis-a",
                                    span=FigureGridSpecSpanState(
                                        row_start=0,
                                        row_stop=1,
                                        col_start=0,
                                        col_stop=1,
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_ADD_STEP

            def is_new_figure(self) -> bool:
                return False

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes_ids=("axis-a",))

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeFigureDialog
        )

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == 1
        assert figure_tool.tool_status.operations[0].axes.axes_ids == ("axis-a",)


def test_manager_auto_names_figures_numerically(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager._tool_graph.root_wrappers[0].slicer_area.axes[0].plot_with_matplotlib()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 1, timeout=5000
        )
        first_uid = manager._tool_graph.figure_uids[0]
        assert manager._child_node(first_uid).display_text == "Figure 1"

        second_uid = manager.create_figure_from_targets((0,), show=False)
        assert second_uid is not None
        assert manager._child_node(second_uid).display_text == "Figure 2"

        manager._child_node(first_uid).name = "Published figure"
        assert manager._child_node(first_uid).display_text == "Published figure"

        third_uid = manager.create_figure_from_targets((0,), show=False)
        assert third_uid is not None
        assert manager._child_node(third_uid).display_text == "Figure 3"

        manager._remove_childtool(second_uid)
        fourth_uid = manager.create_figure_from_targets((0,), show=False)
        assert fourth_uid is not None
        assert manager._child_node(fourth_uid).display_text == "Figure 4"

        preserved_tool = FigureComposerTool(data)
        preserved_tool._tool_display_name = "ImageTool plot"
        preserved_uid = manager.add_figuretool(preserved_tool, show=False)
        assert manager._child_node(preserved_uid).display_text == "ImageTool plot"

        explicit_uid = manager.create_figure_from_targets(
            (0,), title="Custom figure", show=False
        )
        assert explicit_uid is not None
        assert manager._child_node(explicit_uid).display_text == "Custom figure"

        fifth_uid = manager.create_figure_from_targets((0,), show=False)
        assert fifth_uid is not None
        assert manager._child_node(fifth_uid).display_text == "Figure 5"

        unnamed_tool = FigureComposerTool(data)
        unnamed_uid = manager.add_figuretool(unnamed_tool, show=False)
        assert manager._child_node(unnamed_uid).display_text == "Figure 6"


def test_manager_duplicate_figure_assigns_unique_display_name_and_keeps_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def uid_for_name(
        manager: erlab.interactive.imagetool.manager.ImageToolManager, name: str
    ) -> str:
        matches = [
            uid
            for uid in manager._tool_graph.figure_uids
            if manager._child_node(uid).display_text == name
        ]
        assert len(matches) == 1
        return matches[0]

    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        first_uid = manager.create_figure_from_targets((0,), show=False)
        assert first_uid is not None
        assert manager._child_node(first_uid).display_text == "Figure 1"

        original_tool = manager._child_node(first_uid).tool_window
        assert isinstance(original_tool, FigureComposerTool)
        original_status = original_tool.tool_status

        manager._select_figure_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 2, timeout=5000
        )

        auto_copy_uid = uid_for_name(manager, "Figure 2")
        assert manager._selected_figure_uids() == [auto_copy_uid]
        auto_copy_tool = manager._child_node(auto_copy_uid).tool_window
        assert isinstance(auto_copy_tool, FigureComposerTool)
        assert auto_copy_tool.tool_status == original_status
        assert auto_copy_tool.tool_data.identical(original_tool.tool_data)

        manager._child_node(first_uid).name = "Band map"

        manager._select_figure_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 3, timeout=5000
        )

        first_custom_copy_uid = uid_for_name(manager, "Band map copy")
        assert manager._selected_figure_uids() == [first_custom_copy_uid]

        manager._select_figure_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 4, timeout=5000
        )

        second_custom_copy_uid = uid_for_name(manager, "Band map copy 2")
        assert manager._selected_figure_uids() == [second_custom_copy_uid]


def test_manager_create_figure_uses_first_selected_main_image_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(20.0).reshape(2, 2, 5) - 4.0,
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
            name="first",
        )
        second = xr.DataArray(
            np.arange(20.0).reshape(2, 2, 5),
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        first_tool = manager.get_imagetool(0)
        first_tool.slicer_area.set_value(axis=2, value=1.0, cursor=0)
        first_tool.slicer_area.set_bin(axis=2, value=3, cursor=0)
        first_tool.slicer_area.set_colormap(
            "magma",
            gamma=0.75,
            reverse=True,
            high_contrast=True,
            zero_centered=True,
            levels_locked=True,
            levels=(-2.0, 4.0),
        )
        manager.get_imagetool(1).slicer_area.set_colormap("viridis", gamma=0.25)
        vmin, vmax = first_tool.slicer_area.colormap_properties["levels"]
        expected = first_tool.slicer_area.images[0].figure_composer_operation(
            source_name="data_0"
        )

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.sources == ("data_0", "data_1")
        assert operation.order == "F"
        assert figure_tool.tool_status.setup.nrows == 1
        assert figure_tool.tool_status.setup.ncols == 2
        assert operation.slice_dim == expected.slice_dim
        assert operation.slice_values == expected.slice_values
        assert operation.slice_width == expected.slice_width
        assert operation.slice_kwargs == expected.slice_kwargs
        assert operation.transpose == expected.transpose
        assert operation.xlim == expected.xlim
        assert operation.ylim == expected.ylim
        assert operation.crop == expected.crop
        assert operation.axis == expected.axis
        assert operation.cmap is None
        assert operation.same_limits is False
        assert operation.norm_name == "PowerNorm"
        assert operation.norm_gamma is None
        assert operation.vcenter is None
        assert operation.halfrange is None
        assert operation.panel_styles_enabled
        styles = {
            (style.map_index, style.slice_index): style
            for style in operation.panel_styles
        }
        assert set(styles) == {(0, 0), (1, 0)}
        assert styles[(0, 0)].cmap == "magma_r"
        assert styles[(0, 0)].norm_name == "CenteredInversePowerNorm"
        assert styles[(0, 0)].norm_gamma == pytest.approx(0.75)
        assert styles[(0, 0)].vcenter == pytest.approx(0.5 * (vmin + vmax))
        assert styles[(0, 0)].halfrange == pytest.approx(0.5 * (vmax - vmin))
        assert styles[(1, 0)].cmap == "viridis"
        assert styles[(1, 0)].norm_name is None
        assert styles[(1, 0)].norm_gamma == pytest.approx(0.25)


def test_manager_create_figure_from_2d_data_ignores_autorange_startup_limits(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25.0).reshape(5, 5),
        dims=("eV", "alpha"),
        coords={
            "eV": np.linspace(10.0, 14.0, 5),
            "alpha": np.linspace(20.0, 24.0, 5),
        },
        name="map",
    )

    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.xlim is None
        assert operation.ylim is None

        figure = plt.figure()
        try:
            figurecomposer_rendering._render_into_figure(
                figure_tool, figure, sync_visible=False
            )
            assert figure_tool._operation_render_errors == {}
            assert any(axis.images for axis in figure.axes)
        finally:
            plt.close(figure)


def test_manager_workspace_figure_sources_save_as_references(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="first",
    )
    second = xr.DataArray(
        np.arange(9.0, 18.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="second",
    )

    with manager_context() as manager:
        for data in (first, second):
            tool = erlab.interactive.itool(data, manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(tool, show=False)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None

        tree = manager._to_datatree()
        try:
            ds = typing.cast(
                "xr.DataTree", tree[f"figures/{figure_uid}/tool"]
            ).to_dataset(inherit=False)
            references = json.loads(
                ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
            )

            assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME in references
            assert "data_1" in references
            assert ds[erlab.interactive.utils._SAVED_TOOL_DATA_NAME].size == 0
            assert ds["data_1"].size == 0
            assert manager_workspace._workspace_dataset_can_write_h5py(ds)

            source_data_by_uid = {
                references[erlab.interactive.utils._SAVED_TOOL_DATA_NAME][
                    "node_uid"
                ]: first,
                references["data_1"]["node_uid"]: second,
            }
            corrupt_ds = ds.drop_vars("data_1")
            restored_from_reference = erlab.interactive.utils.ToolWindow.from_dataset(
                corrupt_ds,
                _tool_data_reference_resolver=lambda reference: source_data_by_uid.get(
                    reference.get("node_uid")
                ),
            )
            qtbot.addWidget(restored_from_reference)
            assert isinstance(restored_from_reference, FigureComposerTool)
            xr.testing.assert_identical(
                restored_from_reference.source_data()["data_1"], second
            )

            fname = tmp_path / "figure-source-references.itws"
            manager._save_workspace_document(fname, force_full=True)
            saved_ds = manager_workspace._read_workspace_dataset_group_h5py(
                fname,
                f"figures/{figure_uid}/tool",
                preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
            )
            assert saved_ds is not None
            assert "data_1" in saved_ds

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            assert manager._from_datatree(
                tree, replace=True, mark_dirty=False, select=False
            )
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
            assert manager._load_workspace_file(
                fname,
                replace=True,
                associate=False,
                mark_dirty=False,
                select=False,
            )
            loaded_tool = manager._tool_graph.nodes[figure_uid].tool_window
            assert isinstance(loaded_tool, FigureComposerTool)
            xr.testing.assert_identical(loaded_tool._source_data["data_1"], second)
        finally:
            tree.close()

        restored = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        source_data = restored.source_data()
        xr.testing.assert_identical(source_data["data_0"], first)
        xr.testing.assert_identical(source_data["data_1"], second)


def test_manager_plot_slices_setup_honors_order_for_horizontal_seeding() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0, 1.0, 2.0),
    )
    assert manager_mainwindow.ImageToolManager._figure_plot_slices_grid_shape(
        operation
    ) == (1, 3)

    multi_source_operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0", "data_1", "data_2"),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"order": "F"})
    assert manager_mainwindow.ImageToolManager._figure_plot_slices_grid_shape(
        multi_source_operation
    ) == (1, 3)


def test_plot_slices_source_styles_keep_default_cmap_panels() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0", "data_1"),
    )
    source_operations = (
        FigureOperationState.plot_slices(
            label="first",
            sources=("data_0",),
        ).model_copy(update={"cmap": "magma"}),
        FigureOperationState.plot_slices(
            label="second",
            sources=("data_1",),
        ),
    )

    seeded = plot_slices_operation_with_source_styles(
        operation,
        source_operations,
        selections_per_source=1,
    )

    assert seeded.cmap is None
    assert seeded.panel_styles_enabled
    assert {
        (style.map_index, style.slice_index): style.cmap
        for style in seeded.panel_styles
    } == {(0, 0): "magma"}


def test_manager_append_to_gridspec_figure_uses_axes_ids(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        axis_a = FigureGridSpecAxesState(
            axes_id="axis-a",
            label="panel",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=0,
                col_stop=1,
            ),
        )
        axis_b = FigureGridSpecAxesState(
            axes_id="axis-b",
            label="panel",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=1,
                col_stop=2,
            ),
        )
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(
                    layout_mode="gridspec",
                    gridspec=FigureGridSpecLayoutState(
                        root=FigureGridSpecGridState(
                            grid_id="root",
                            nrows=1,
                            ncols=2,
                            axes=(axis_a, axis_b),
                        )
                    ),
                ),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        operation = FigureOperationState.line(
            label="line",
            source="line",
            axes=FigureAxesSelectionState(),
        )

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager, (figure_uid,), operation
        )
        qtbot.addWidget(dialog)

        assert dialog.selector_stack.currentWidget() is dialog.gridspec_axes_selector
        assert dialog.gridspec_axes_selector.axes_ids() == ("axis-a", "axis-b")
        assert dialog.axes_selection() == FigureAxesSelectionState(axes_ids=("axis-a",))

        dialog.gridspec_axes_selector.set_selected_axes_ids(("axis-b",), emit=True)

        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes_ids=("axis-b",)),
        )


def test_manager_append_to_subplots_figure_uses_axes_selector(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        operation = FigureOperationState.plot_slices(
            label="plot_slices",
            sources=("line",),
            axes=FigureAxesSelectionState(),
        )

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager, (figure_uid,), operation
        )
        qtbot.addWidget(dialog)

        assert dialog.selector_stack.currentWidget() is dialog.axes_selector
        assert dialog.axes_selector.selected_axes() == ((0, 0), (0, 1))

        dialog.axes_selector.set_selected_axes(((0, 1),), emit=True)

        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 1),)),
        )

        dialog.axes_selector.resize(dialog.axes_selector.sizeHint())
        qtbot.mouseClick(
            dialog.axes_selector,
            QtCore.Qt.MouseButton.LeftButton,
            pos=dialog.axes_selector._add_pill_rect("row").center(),
        )
        assert figure_tool.tool_status.setup.nrows == 2
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 1),))

        dialog.axes_selector.resize(dialog.axes_selector.sizeHint())
        qtbot.mouseClick(
            dialog.axes_selector,
            QtCore.Qt.MouseButton.LeftButton,
            pos=dialog.axes_selector._add_pill_rect("column").center(),
        )
        assert figure_tool.tool_status.setup.ncols == 3
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 1),))


def test_manager_figure_target_dialog_defaults_to_add_step_without_selected_figure(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
        )
        qtbot.addWidget(dialog)

        assert dialog.selected_action() == manager_mainwindow._FIGURE_DIALOG_ADD_STEP
        assert not dialog.is_new_figure()
        assert not dialog.selector_stack.isHidden()
        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(manager_mainwindow._FIGURE_DIALOG_NEW)
        )

        assert dialog.is_new_figure()
        assert dialog.selector_stack.isHidden()
        assert dialog.selected_target() is None
        assert button.isEnabled()


def test_manager_figure_target_dialog_defaults_to_replace_selected_single_source(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="Line Source"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=1,
            selected_figure_uid=figure_uid,
        )
        qtbot.addWidget(dialog)

        assert (
            dialog.selected_action() == manager_mainwindow._FIGURE_DIALOG_REPLACE_SOURCE
        )
        assert dialog.is_replace_source()
        assert dialog.selected_source_alias() == "line"
        assert "line" in dialog.source_combo.currentText()
        assert dialog.selector_stack.isHidden()
        assert not dialog.source_combo.isHidden()
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()
        assert dialog._figure_source_count(None) == 0
        assert dialog._figure_source_count("missing") == 0

        class EmptyFigureNode:
            tool_window = None

        with monkeypatch.context() as context:
            context.setattr(manager, "_child_node", lambda _uid: EmptyFigureNode())
            assert dialog._figure_source_count(figure_uid) == 0


def test_manager_figure_target_dialog_switches_and_repairs_axes_selection(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        first_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        second_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        qtbot.addWidget(first_tool)
        qtbot.addWidget(second_tool)
        first_uid = manager.add_figuretool(first_tool, show=False)
        second_uid = manager.add_figuretool(second_tool, show=False)

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager,
            (first_uid, second_uid),
            FigureOperationState.line(label="line", source="line"),
            allow_new_figure=True,
        )
        qtbot.addWidget(dialog)

        assert dialog.selected_action() == manager_mainwindow._FIGURE_DIALOG_ADD_STEP
        assert not dialog.is_new_figure()
        assert not dialog.selector_stack.isHidden()
        ok_button = dialog.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button is not None
        assert ok_button.isEnabled()

        dialog.figure_combo.setCurrentIndex(dialog.figure_combo.findData(first_uid))
        assert dialog.figure_uid() == first_uid
        assert not dialog.is_new_figure()
        assert dialog.selector_stack.currentWidget() is dialog.axes_selector
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 0),))

        dialog._select_all_axes()
        assert dialog.axes_selection() == FigureAxesSelectionState(
            axes=((0, 0), (0, 1))
        )
        dialog._clear_axes()
        assert dialog.axes_selection() is None
        assert not ok_button.isEnabled()
        dialog._select_all_axes()
        assert ok_button.isEnabled()
        assert dialog.selected_target() == (
            first_uid,
            FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        )

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(manager_mainwindow._FIGURE_DIALOG_ADD_SOURCE)
        )
        assert dialog.is_add_source_only()
        assert dialog.selected_target() is None
        assert dialog.selector_stack.isHidden()
        assert ok_button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(
                manager_mainwindow._FIGURE_DIALOG_REPLACE_SOURCE
            )
        )
        assert dialog.is_replace_source()
        assert dialog.selected_source_alias() == "line"
        assert dialog.selector_stack.isHidden()
        assert not dialog.source_combo.isHidden()
        assert ok_button.isEnabled()
        dialog.source_combo.setCurrentIndex(-1)
        dialog._selection_changed()
        assert not ok_button.isEnabled()
        dialog.source_combo.setCurrentIndex(0)
        dialog._selection_changed()
        assert ok_button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(manager_mainwindow._FIGURE_DIALOG_ADD_STEP)
        )
        dialog.figure_combo.setCurrentIndex(dialog.figure_combo.findData(second_uid))
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 0),))

        dialog.figure_combo.setItemData(dialog.figure_combo.currentIndex(), "missing")
        dialog._figure_changed()
        assert dialog.axes_selection() is None
        assert not ok_button.isEnabled()
        dialog._select_all_axes()
        dialog._grow_subplot_grid("row")

        dialog.figure_combo.setItemData(dialog.figure_combo.currentIndex(), None)
        assert dialog.figure_uid() == first_uid


def test_manager_figure_target_dialog_disables_replace_for_multiple_sources(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = manager_mainwindow._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=2,
            selected_figure_uid=figure_uid,
        )
        qtbot.addWidget(dialog)

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(
                manager_mainwindow._FIGURE_DIALOG_REPLACE_SOURCE
            )
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert not button.isEnabled()
        assert "one ImageTool source" in dialog.status_label.text()


def test_manager_prompt_append_figure_target_auto_and_cancel_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        assert manager._prompt_append_figure_target(None) is None
        assert manager._prompt_append_figure_target(None, figure_uid="missing") is None

        single_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        qtbot.addWidget(single_tool)
        single_uid = manager.add_figuretool(single_tool, show=False)
        assert manager._append_single_axis_selection(single_uid) == (
            FigureAxesSelectionState(axes=((0, 0),))
        )
        assert manager._prompt_append_figure_target(None) == (
            single_uid,
            FigureAxesSelectionState(axes=((0, 0),)),
        )

        wide_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        qtbot.addWidget(wide_tool)
        wide_uid = manager.add_figuretool(wide_tool, show=False)

        class RejectDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
            ) -> None:
                assert figure_uids == (wide_uid,)
                assert allow_new_figure is False

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Rejected

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", RejectDialog
        )
        assert manager._prompt_append_figure_target(None, figure_uid=wide_uid) is None


def test_manager_child_imagetool_gets_figure_context_actions(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def action_names(tool: erlab.interactive.imagetool.ImageTool) -> set[str]:
        return {
            action.objectName()
            for plot in tool.slicer_area.axes
            for action in plot.vb.menu.actions()
        }

    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=False,
            execute=False,
        )
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        assert "itool_plot_with_matplotlib_action" not in action_names(child)

        manager.add_imagetool_child(child, 0, show=False)

        assert "itool_plot_with_matplotlib_action" in action_names(child)
        assert "itool_append_to_figure_action" in action_names(child)


def test_manager_workspace_restores_figures_ui(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        workspace = manager._to_datatree()
        try:
            manager.remove_all_tools()
            assert not manager.left_tabs.tabBar().isVisible()

            restored = manager._from_datatree(
                workspace, replace=True, mark_dirty=False, select=False
            )
            assert restored is True
            assert manager.figure_list.count() == 1
            assert manager.left_tabs.tabBar().isVisible()
            assert manager.left_tabs.isTabVisible(1)
        finally:
            workspace.close()


def test_manager_workspace_close_serializes_all_figures(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uids: list[str] = []
        for _ in range(2):
            uid = manager.create_figure_from_targets((0,), show=False)
            assert uid is not None
            figure_uids.append(uid)

        workspace = manager._to_datatree(close=True)
        try:
            assert "figures" in workspace
            figures = typing.cast("xr.DataTree", workspace["figures"])
            assert all(uid in figures for uid in figure_uids)
            assert manager.ntools == 0
            assert manager._tool_graph.figure_uids == []
        finally:
            workspace.close()


def test_manager_append_operation_to_existing_figure(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        operation_count = len(figure.tool_status.operations)
        source_name = figure.tool_status.sources[0].name

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            operation=FigureOperationState.line(label="overlay", source=source_name),
            show=False,
        )

        assert appended is True
        assert len(figure.tool_status.operations) == operation_count + 1
        assert figure.tool_status.operations[-1].kind.value == "line"


def test_manager_append_momentum_source_seeds_bz_overlay(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 0.0, 1.0], "ky": [-2.0, 0.0, 2.0]},
        name="momentum",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="existing",
        )
        figure_tool = FigureComposerTool(
            figure_data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="existing", label="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        assert appended is True
        assert [operation.kind for operation in figure_tool.tool_status.operations] == [
            FigureOperationKind.PLOT_SLICES,
            FigureOperationKind.BZ_OVERLAY,
        ]
        assert figure_tool.tool_status.operations[-1].axes.axes == ((0, 0),)


def test_manager_create_explicit_plot_slices_fills_axes(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="map",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets(
            (0,),
            operation=FigureOperationState.plot_slices(
                label="plot", sources=("map",), axes=FigureAxesSelectionState()
            ),
            show=False,
        )

        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        assert figure_tool.tool_status.operations[0].axes.axes == ((0, 0),)


def test_manager_ktool_output_figure_seeds_bz_overlay(
    qtbot,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    from erlab.interactive.kspace import KspaceTool

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap.qsel(eV=-0.1), link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, KspaceTool)
        child._avec = erlab.lattice.abc2avec(2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        child.centering_combo.setCurrentText("I")
        child.rot_spin.setValue(15.0)
        child.kz_spin.setValue(0.5)
        child.points_check.setChecked(True)
        qtbot.wait_until(lambda: child.bz_group.isEnabled(), timeout=5000)
        child.bz_group.setChecked(True)
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]
        manager._child_node(output_uid).name = "converted"

        figure_uid = manager.create_figure_from_targets((output_uid,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operations = figure_tool.tool_status.operations

        assert [operation.kind for operation in operations] == [
            FigureOperationKind.PLOT_SLICES,
            FigureOperationKind.BZ_OVERLAY,
        ]
        bz_operation = operations[1]
        assert np.isclose(bz_operation.bz_a, 2.0)
        assert np.isclose(bz_operation.bz_b, 3.0)
        assert np.isclose(bz_operation.bz_c, 4.0)
        assert bz_operation.bz_centering_type == "I"
        assert bz_operation.bz_angle == 15.0
        assert bz_operation.bz_kz_pi_over_c == 0.5
        assert bz_operation.bz_vertices is True
        assert bz_operation.bz_midpoints is True


def test_manager_bz_overlay_ignores_converted_output_without_ktool_parent() -> None:
    class FakeNode:
        output_id = "ktool.converted_output"

    class FakeParent:
        tool_window = object()

    class FakeManager:
        def _node_for_target(self, target: int | str) -> FakeNode:
            assert target == "converted"
            return FakeNode()

        def _parent_node(self, node: FakeNode) -> FakeParent:
            assert isinstance(node, FakeNode)
            return FakeParent()

    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-2.0, 2.0]},
        name="momentum",
    )
    axes = FigureAxesSelectionState(axes=((0, 0),))

    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_targets(
            FakeManager(),
            ("first", "second"),
            {"momentum": data},
            axes=axes,
        )
        is None
    )
    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_targets(
            FakeManager(),
            ("converted",),
            {"first": data, "second": data},
            axes=axes,
        )
        is None
    )

    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_target(
            FakeManager(),
            "converted",
            data,
            axes=axes,
        )
        is None
    )


def test_manager_append_operation_uses_axes_dialog_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_tool = FigureComposerTool(
            xr.DataArray(np.arange(4.0), dims=("x",), name="line"),
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        class FakeAppendDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState,
            ) -> None:
                assert figure_uids == (figure_uid,)

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 1),))

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeAppendDialog
        )

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            operation=FigureOperationState.line(label="overlay", source="line"),
            show=False,
        )

        assert appended is True
        assert figure_tool.tool_status.operations[-1].axes.axes == ((0, 1),)


def test_manager_figure_action_replace_source_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)

        select_tools(manager, [1])

        class FakeReplaceDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_REPLACE_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

            def selected_source_alias(self) -> str:
                return "data_0"

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeReplaceDialog
        )

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["data_0"], second)
        [source] = figure_tool.source_states()
        assert source.name == "data_0"
        assert source.node_uid == manager._node_for_target(1).uid
        assert "data_1" not in figure_tool.source_data()


def test_manager_figure_source_row_refresh_updates_only_selected_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        original_second = figure_tool.source_data()["data_1"]
        operations = figure_tool.tool_status.operations
        manager._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        buttons = _source_refresh_buttons(figure_tool)
        assert buttons["data_0"].isEnabled()
        assert buttons["data_1"].isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            buttons["data_0"].click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["data_0"], updated_first)
        xr.testing.assert_identical(source_data["data_1"], original_second)
        [source_0, source_1] = figure_tool.source_states()
        assert source_0.name == "data_0"
        assert source_0.node_uid == manager._node_for_target(0).uid
        assert source_1.name == "data_1"
        assert figure_tool.tool_status.operations == operations
        assert "data_0" in figure_tool.generated_code()

        _render_figure_composer_rgba(figure_tool)
        namespace = _exec_generated_code(
            figure_tool.generated_code(),
            {"data_0": updated_first, "data_1": original_second},
        )
        assert isinstance(namespace["fig"], Figure)
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager.remove_imagetool(0)
        buttons = _source_refresh_buttons(figure_tool)
        assert not buttons["data_0"].isEnabled()
        assert (
            buttons["data_0"].toolTip()
            == "This source is not linked to an open ImageTool"
        )


def test_manager_figure_refresh_sources_updates_live_sources_and_skips_detached(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        detached = xr.DataArray(
            np.array([1.0, 2.0]),
            dims=("x",),
            coords={"x": [0.0, 1.0]},
            name="detached",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.add_sources(
            (FigureSourceState(name="detached", label="Detached"),),
            {"detached": detached},
        )
        operations = figure_tool.tool_status.operations
        manager._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        buttons = _source_refresh_buttons(figure_tool)
        assert buttons["data_0"].isEnabled()
        assert buttons["data_1"].isEnabled()
        assert not buttons["detached"].isEnabled()
        assert figure_tool.refresh_sources_button.isEnabled()
        figure_tool.refresh_sources_button.click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["data_0"], updated_first)
        xr.testing.assert_identical(source_data["data_1"], updated_second)
        xr.testing.assert_identical(source_data["detached"], detached)
        assert figure_tool.tool_status.operations == operations
        assert figure_tool.source_status_label.text() == ""
        assert figure_tool.source_status_label.isHidden()
        assert figure_tool._operation_render_errors == {}
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_figure_source_helper_edge_contracts(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="data",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        image_uid = manager._node_for_target(0).uid
        child = itool(data.sum("y"), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        select_child_tool(manager, figure_uid)
        assert manager._selected_figure_uid_for_figure_dialog() == figure_uid

        source_states = figure_tool.source_states()
        source_data = figure_tool.source_data()
        replacement_source = FigureSourceState(
            name="replacement",
            label="replacement",
            node_uid=image_uid,
        )
        assert not manager._add_sources_to_figure(
            "missing",
            source_states,
            source_data,
            show=False,
        )
        assert manager._figure_source_state(figure_tool, "missing") is None
        assert manager._figure_source_live_node("missing", "data_0") is None
        assert not manager._refresh_figure_source(figure_uid, "missing")
        assert not manager._replace_figure_source(
            figure_uid,
            "data_0",
            (),
            {},
            show=False,
        )
        assert not manager._replace_figure_source(
            figure_uid,
            "data_0",
            (replacement_source,),
            {},
            show=False,
        )

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_is_figure_uid",
                lambda uid: uid in {figure_uid, child_uid},
            )
            assert not manager._add_sources_to_figure(
                child_uid,
                source_states,
                source_data,
                show=False,
            )
            assert not manager._replace_figure_source(
                child_uid,
                "data_0",
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert manager._figure_source_live_node(child_uid, "data_0") is None
            assert not manager._refresh_figure_source(child_uid, "data_0")

        with monkeypatch.context() as context:
            context.setattr(figure_tool, "replace_source", lambda *_args: False)
            assert not manager._replace_figure_source(
                figure_uid,
                "data_0",
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert not manager._refresh_figure_source(figure_uid, "data_0")

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_figure_source_live_node",
                lambda *_args: manager._node_for_target(0),
            )
            context.setattr(manager, "_is_figure_uid", lambda uid: uid == child_uid)
            assert not manager._refresh_figure_source(child_uid, "data_0")

        with monkeypatch.context() as context:
            context.setattr(manager, "_figure_uids", lambda: ("missing",))
            manager._refresh_figure_source_controls()

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(
                manager,
                "_figure_sources_from_targets",
                lambda _targets: ((), (), {}),
            )
            manager.create_figure_from_selection()

        dialog_events: list[str] = []

        class AliasNoneDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                _figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid == figure_uid
                dialog_events.append("init")

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_REPLACE_SOURCE

            def selected_source_alias(self) -> str | None:
                dialog_events.append("alias")
                return None

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(manager, "_figure_uids", lambda: (figure_uid,))
            context.setattr(
                manager,
                "_figure_sources_from_targets",
                lambda _targets: ((0,), source_states, source_data),
            )
            context.setattr(
                manager,
                "_selected_figure_uid_for_figure_dialog",
                lambda: figure_uid,
            )
            context.setattr(
                manager_mainwindow,
                "_AppendFigureTargetDialog",
                AliasNoneDialog,
            )
            manager.create_figure_from_selection()
        assert dialog_events == ["init", "alias"]


def test_manager_figure_action_add_source_only_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)
        workspace_path = tmp_path / "add-source-only-delta.itws"
        manager._save_workspace_document(workspace_path, force_full=True)

        select_tools(manager, [1])

        class FakeSourceOnlyDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_ADD_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeSourceOnlyDialog
        )

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["data_1"], second)
        assert {source.name for source in figure_tool.source_states()} == {
            "data_0",
            "data_1",
        }
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager._save_workspace_document(workspace_path)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        xr.testing.assert_identical(loaded_tool.source_data()["data_1"], second)


def test_manager_figure_remove_unused_source_persists_workspace(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        _resolved_targets, sources, source_data = manager._figure_sources_from_targets(
            (1,)
        )
        [source] = sources
        manager._add_sources_to_figure(figure_uid, sources, source_data, show=False)
        assert source.name in figure_tool.source_data()
        manager._mark_workspace_clean()

        buttons = _source_remove_buttons(figure_tool)
        assert buttons[source.name].isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            buttons[source.name].click()

        assert source.name not in figure_tool.source_data()
        assert source.name not in {
            source.name for source in figure_tool.source_states()
        }
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        workspace_path = tmp_path / "remove-unused-figure-source.itws"
        manager._save_workspace_document(workspace_path, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        assert source.name not in loaded_tool.source_data()
        assert source.name not in {
            source.name for source in loaded_tool.source_states()
        }


def test_manager_figure_action_multi_source_append_preserves_panel_colormaps(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area.set_colormap("magma")
        manager.get_imagetool(1).slicer_area.set_colormap("viridis", reverse=True)

        figure_tool = FigureComposerTool(
            first,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="seed", label="seed"),),
                operations=(),
                primary_source="seed",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0, 1])

        class FakeAppendDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 2
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return manager_mainwindow._FIGURE_DIALOG_ADD_STEP

            def is_new_figure(self) -> bool:
                return False

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 0), (0, 1)))

        monkeypatch.setattr(
            manager_mainwindow, "_AppendFigureTargetDialog", FakeAppendDialog
        )

        manager.create_figure_action.trigger()

        operation = figure_tool.tool_status.operations[-1]
        assert operation.kind == FigureOperationKind.PLOT_SLICES
        assert operation.sources == ("data_0", "data_1")
        assert operation.order == "F"
        assert operation.axes.axes == ((0, 0), (0, 1))
        assert operation.cmap is None
        assert operation.panel_styles_enabled
        assert operation.panel_styles == (
            FigurePlotSlicesPanelStyleState(
                map_index=0,
                slice_index=0,
                cmap="magma",
            ),
            FigurePlotSlicesPanelStyleState(
                map_index=1,
                slice_index=0,
                cmap="viridis_r",
            ),
        )
