import contextlib
import json
import sys
import types
import typing
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib import style as mpl_style
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer._code as figurecomposer_code
import erlab.interactive._figurecomposer._defaults as figurecomposer_defaults
import erlab.interactive._figurecomposer._model._gridspec as figurecomposer_gridspec
import erlab.interactive._figurecomposer._provenance as figurecomposer_provenance
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._text as figurecomposer_text
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._figurecomposer._ui._axes_widgets as axes_widgets
import erlab.interactive._figurecomposer._ui._color_widgets as color_widgets
import erlab.interactive._figurecomposer._ui._editor_controls as _editor_controls
import erlab.interactive._figurecomposer._ui._figure_window as figure_window_ui
import erlab.interactive._figurecomposer._ui._tick_params as figurecomposer_tick_params
import erlab.interactive._stylesheets
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
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer import (
    _subplot_adjust as figurecomposer_subplot_adjust,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _render as plot_slices_render,
)
from erlab.interactive._figurecomposer._ui import (
    _operation_panel as figurecomposer_operation_panel,
)
from erlab.interactive._figurecomposer._ui import (
    _reorder_list as figurecomposer_reorder_list,
)
from erlab.interactive._figurecomposer._ui import (
    _toolbar_dialogs as figurecomposer_toolbar_dialogs,
)
from erlab.interactive._figurecomposer._ui._operation_editor import (
    _FigureComposerStepEditorPage,
    _FigureComposerStepEditorScroll,
)
from erlab.interactive._figurecomposer._ui._toolbar_dialogs import (
    _connect_panel_editor_signal,
)
from erlab.interactive._options import options
from erlab.interactive._options.schema import FigureOptions
from erlab.interactive.imagetool.manager._workspace import (
    _controller as workspace_controller,
)

from ._common import (
    _COLLAPSED_LAYOUT_WARNING,
    _activate_combo_index,
    _activate_combo_text,
    _assert_figure_composer_provenance_replayable,
    _assert_serialized_plot_restores_exactly,
    _clear_clipboard,
    _click_tick_params_segment,
    _custom_order_step,
    _drag_widget,
    _expected_layout_from_rcparams,
    _figure_composer_image_source,
    _figure_composer_line_slice_source,
    _figure_composer_profile_source,
    _figure_composer_replay_source_state,
    _finish_tick_params_edit,
    _method_operations,
    _operation_section_buttons,
    _operation_source_status_label,
    _operation_status_codes,
    _photon_energy_source,
    _select_operation_rows,
    _selected_operation_rows,
    _send_mouse_move,
    _set_figure_stylesheets,
)


def _operation_context_action(
    tool: FigureComposerTool, object_name: str
) -> tuple[QtWidgets.QMenu, QtGui.QAction]:
    existing_menus = tool.operation_panel.operation_list.findChildren(QtWidgets.QMenu)
    tool.operation_panel._show_context_menu(QtCore.QPoint(0, 0))
    menu = next(
        menu
        for menu in tool.operation_panel.operation_list.findChildren(QtWidgets.QMenu)
        if all(menu is not existing_menu for existing_menu in existing_menus)
    )
    action = menu.findChild(QtGui.QAction, object_name)
    assert action is not None
    return menu, action


def test_figure_composer_color_widgets_parse_and_sync(qtbot, monkeypatch) -> None:
    opaque = QtGui.QColor(1, 2, 3)
    translucent = QtGui.QColor(1, 2, 3, 4)
    grayscale = color_widgets._qcolor_from_mpl_color_text("0.5")
    assert grayscale is not None
    assert grayscale.getRgb() == (128, 128, 128, 255)
    cycle_color = color_widgets._qcolor_from_mpl_color_text("C1")
    assert cycle_color is not None
    assert (
        cycle_color.getRgb()
        == QtGui.QColor.fromRgbF(*mpl.colors.to_rgba("C1")).getRgb()
    )
    tuple_alpha = color_widgets._qcolor_from_mpl_color_text("(1, 0, 0, 0.5)")
    assert tuple_alpha is not None
    assert tuple_alpha.getRgb() == (255, 0, 0, 128)
    hex_alpha = color_widgets._qcolor_from_mpl_color_text("#01020304")
    assert hex_alpha is not None
    assert hex_alpha.getRgb() == (1, 2, 3, 4)
    assert color_widgets._qcolor_to_mpl_color_text(opaque) == "#010203"
    assert color_widgets._qcolor_to_mpl_color_text(translucent) == "#01020304"
    assert color_widgets._qcolor_from_mpl_color_text("(1.0, 0.0, 0.0)") is not None
    assert color_widgets._qcolor_from_mpl_color_text("[1.0, 0.0, 0.0]") is not None
    assert color_widgets._qcolor_from_mpl_color_text("") is None
    assert color_widgets._qcolor_from_mpl_color_text("[bad") is None
    assert color_widgets._top_level_comma_parts(
        "red, (0, 1, 0), 'blue, still blue', [0, 0, 1]"
    ) == ("red", "(0, 1, 0)", "'blue, still blue'", "[0, 0, 1]")
    assert color_widgets._top_level_comma_parts(r"'red\', blue', green") == (
        r"'red\', blue'",
        "green",
    )
    assert color_widgets._color_tuple_from_text("") == ()
    assert color_widgets._color_tuple_from_text("['red', 'blue']") == (
        "red",
        "blue",
    )
    assert color_widgets._color_tuple_from_text("[bad") == ("[bad",)
    assert color_widgets._color_tuple_from_text("[1]") == ("1",)
    with monkeypatch.context() as context:
        context.setattr(
            color_widgets,
            "_qcolor_from_mpl_color_value",
            lambda _value: None,
        )
        assert color_widgets._default_mpl_line_color().getRgb() == (
            0,
            0,
            0,
            255,
        )

    inherited_edit = color_widgets._ColorLineEditWidget(
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

    color_edit = color_widgets._ColorLineEditWidget("tab:blue")
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

    deleted_edit = color_widgets._ColorLineEditWidget("tab:red")
    qtbot.addWidget(deleted_edit)
    deleted_button = deleted_edit.color_button

    def delete_during_dialog(*_args, **_kwargs) -> QtGui.QColor:
        deleted_edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        return QtGui.QColor("blue")

    with monkeypatch.context() as context:
        context.setattr(QtWidgets.QColorDialog, "getColor", delete_during_dialog)
        deleted_button._choose_color()

    color_list = color_widgets._ColorListEditorWidget(("red", "blue"))
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
    selector = axes_widgets._AxesSelectorWidget()
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
    selector = axes_widgets._AxesSelectorWidget()
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

    assert axes_widgets._selector_colors(selector).selection == QtGui.QColor("blue")
    selector.setEnabled(False)
    assert axes_widgets._selector_colors(selector).selection == QtGui.QColor("green")


def test_figure_composer_axes_selector_add_pills_use_selector_color_roles(
    qtbot, monkeypatch
) -> None:
    selector = axes_widgets._AxesSelectorWidget()
    qtbot.addWidget(selector)
    selector.resize(selector.sizeHint())
    colors = axes_widgets._selector_colors(selector)

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

    monkeypatch.setattr(axes_widgets, "_draw_selector_rect", record_selector_rect)
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
    selector = axes_widgets._GridSpecViewWidget(mode="select")
    qtbot.addWidget(selector)
    selector.set_layout(root, labels={"main-axis": "Main", "child-axis": "Child"})
    selector.resize(selector.sizeHint())
    selector.show()
    assert selector.axes_ids() == ("main-axis", "child-axis")
    assert not selector.axis_rect("main-axis").isNull()
    assert selector.axis_rect("missing").isNull()
    assert selector._range_axes_ids("missing", "child-axis") == ("child-axis",)
    assert axes_widgets._ratio_edges(0, 100, 2, ()) == (
        0.0,
        50.0,
        100.0,
    )

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

    class _RecordingGridSpecViewWidget(axes_widgets._GridSpecViewWidget):
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
    region = axes_widgets._GridSpecRegionInfo(
        region_id="main-axis",
        kind="axes",
        span=main_span,
        label="main",
    )
    child_region = axes_widgets._GridSpecRegionInfo(
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
    assert axes_widgets._ratio_edges(0, 100, 0, ()) == (0.0,)
    assert axes_widgets._ratio_edges(0, 100, 2, (2.0, 1.0)) == (
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
    native_hover_cursors = sys.platform != "darwin"
    if native_hover_cursors:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeFDiagCursor
        assert editor.cursor_shapes == [QtCore.Qt.CursorShape.SizeFDiagCursor]
    else:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
        assert editor.cursor_shapes == []
    editor._update_hover_cursor(
        editor._handle_rects(axis_rect, hit=True)[0][1].center()
    )
    if native_hover_cursors:
        assert editor.cursor_shapes == [QtCore.Qt.CursorShape.SizeFDiagCursor]
    else:
        assert editor.cursor_shapes == []
    editor._update_hover_cursor(
        editor._handle_rects(axis_rect, hit=True)[1][1].center()
    )
    if native_hover_cursors:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeBDiagCursor
        assert editor.cursor_shapes == [
            QtCore.Qt.CursorShape.SizeFDiagCursor,
            QtCore.Qt.CursorShape.SizeBDiagCursor,
        ]
    else:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
        assert editor.cursor_shapes == []
    editor._set_region_handles_visible(False)
    editor._update_hover_cursor(axis_rect.center())
    if native_hover_cursors:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.SizeAllCursor
    else:
        assert editor.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
    editor._update_hover_cursor(QtCore.QPoint(-100, -100))
    assert editor.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
    assert editor.unset_count == int(native_hover_cursors)
    editor._update_hover_cursor(QtCore.QPoint(-100, -100))
    assert editor.unset_count == int(native_hover_cursors)
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


def test_figure_composer_generated_provenance_covers_operation_kinds(qtbot) -> None:
    image = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 0.0, 1.0], "ky": [-1.0, 0.0, 1.0]},
        name="image",
    )
    volume = _figure_composer_image_source("volume")
    profile_map = _figure_composer_line_slice_source("profile_map")
    plotting_tool = FigureComposerTool.from_sources(
        {
            "image": image,
            "volume": volume,
            "profile_map": profile_map,
        },
        sources=(
            _figure_composer_replay_source_state("image"),
            _figure_composer_replay_source_state("volume"),
            _figure_composer_replay_source_state("profile_map"),
        ),
        setup=FigureSubplotsState(nrows=2, ncols=2),
        operations=(
            FigureOperationState.set_palette(),
            FigureOperationState.plot_array(
                label="image",
                source="image",
                axes=FigureAxesSelectionState(axes=((0, 0),)),
            ).model_copy(update={"cmap": "viridis", "colorbar": "right"}),
            FigureOperationState.plot_slices(
                label="cuts",
                sources=("volume",),
                axes=FigureAxesSelectionState(axes=((0, 1),)),
                slice_dim="eV",
                slice_values=(-0.5, 0.5),
            ).model_copy(
                update={
                    "annotate": False,
                    "cmap": "terrain_r",
                    "line_label_text": "{number}: {eV:g}",
                    "line_color_mode": "coordinate",
                    "line_color_coord": "eV",
                }
            ),
            FigureOperationState.line(
                label="profiles",
                source="profile_map",
                axes=FigureAxesSelectionState(axes=((1, 0),)),
            ).model_copy(
                update={
                    "line_iter_dim": "eV",
                    "line_color_mode": "coordinate",
                    "line_color_coord": "eV",
                    "line_color_cmap": "plasma",
                    "line_normalize": "mean",
                }
            ),
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="set_title",
                axes=FigureAxesSelectionState(axes=((1, 0),)),
                args=("Profiles",),
            ),
            FigureOperationState.method(
                family=FigureMethodFamily.ERLAB,
                name="clean_labels",
                axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (1, 0), (1, 1))),
            ),
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="supxlabel",
                args=("Shared coordinate",),
            ),
        ),
        primary_source="image",
    )
    qtbot.addWidget(plotting_tool)

    hvdep_kconv = _photon_energy_source()
    overlay_tool = FigureComposerTool.from_sources(
        {
            "image": image,
            "hvdep_kconv": hvdep_kconv,
        },
        sources=(
            _figure_composer_replay_source_state("image"),
            _figure_composer_replay_source_state("hvdep_kconv"),
        ),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        operations=(
            FigureOperationState.plot_array(
                label="momentum",
                source="image",
                axes=FigureAxesSelectionState(axes=((0, 0),)),
            ),
            FigureOperationState.bz_overlay(
                axes=FigureAxesSelectionState(axes=((0, 0),)),
                mode="out_of_plane",
            ).model_copy(
                update={
                    "bz_bounds": (-2.0, 2.0, -2.0, 2.0),
                    "bz_midpoints": True,
                    "line_kw": {"color": "tab:orange"},
                }
            ),
            FigureOperationState.photon_energy_overlay(
                source="hvdep_kconv",
                axes=FigureAxesSelectionState(axes=((0, 1),)),
                binding_energy=-0.3,
            ).model_copy(
                update={
                    "photon_energies": (30.0, 45.0),
                    "legend_kw": {"title": "Photon energy"},
                }
            ),
        ),
        primary_source="image",
    )
    qtbot.addWidget(overlay_tool)

    tools = {
        "core plotting": plotting_tool,
        "overlays": overlay_tool,
    }
    covered_kinds = {
        operation.kind
        for tool in tools.values()
        for operation in tool.tool_status.operations
    }
    assert covered_kinds == set(FigureOperationKind) - {FigureOperationKind.CUSTOM}

    for case_label, tool in tools.items():
        code = _assert_figure_composer_provenance_replayable(
            tool,
            case_label=case_label,
        )
        assert "fig, axs = plt.subplots" in code


def test_axes_selector_size_hint_tracks_grid(qtbot):
    selector = axes_widgets._AxesSelectorWidget()
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
    tool.operation_editor.select_section("axes")
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
    window = figure_window_ui._FigureComposerDisplayWindow(FigureSubplotsState())

    window.resize_to_setup(FigureSubplotsState())
    window._suppress_resize_signal = False
    window.resizeEvent(None)

    assert (window, 0, "_allow_resize_signal", ()) in calls
    assert (window, 0, "_emit_canvas_size_changed", (window.canvas,)) in calls
    window.close_from_owner()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_figure_display_window_skips_stale_resize_callbacks(qtbot) -> None:
    window = figure_window_ui._FigureComposerDisplayWindow(
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
    window = figure_window_ui._FigureComposerDisplayWindow(
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

    app_quit_window = figure_window_ui._FigureComposerDisplayWindow(
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


def test_figure_display_window_event_filter_accepts_source_drag(qtbot) -> None:
    window = figure_window_ui._FigureComposerDisplayWindow(FigureSubplotsState())
    qtbot.addWidget(window)
    mime = QtCore.QMimeData()
    window.set_source_drop_callbacks(can_drop=lambda data: data is mime)

    class _NoMimeDragEnterEvent(QtGui.QDragEnterEvent):
        def mimeData(self) -> None:
            return None

    no_mime_event = _NoMimeDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert not window._handle_source_drag_event(no_mime_event)

    event = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert window.eventFilter(window.canvas, event)
    assert event.isAccepted()
    assert event.dropAction() == QtCore.Qt.DropAction.CopyAction


def test_figure_display_window_show_for_setup_recalls_hidden_states(
    qtbot, monkeypatch
) -> None:
    class _FakeScreen:
        def availableGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(0, 0, 800, 600)

    class _MovedDisplayWindow(figure_window_ui._FigureComposerDisplayWindow):
        def __init__(self) -> None:
            super().__init__(FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0))
            self._test_frame = QtCore.QRect(5000, 5000, 120, 120)
            self.moved_to: list[QtCore.QPoint] = []

        def frameGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(self._test_frame)

        def move(self, point: QtCore.QPoint) -> None:
            self.moved_to.append(QtCore.QPoint(point))
            self._test_frame.moveTopLeft(point)

        def screen(self) -> _FakeScreen:
            return _FakeScreen()

    minimized = figure_window_ui._FigureComposerDisplayWindow(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0)
    )
    show_calls: list[str] = []
    monkeypatch.setattr(minimized, "isMinimized", lambda: True)
    monkeypatch.setattr(minimized, "showNormal", lambda: show_calls.append("normal"))
    monkeypatch.setattr(minimized, "show", lambda: show_calls.append("show"))

    minimized.show_for_setup(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0),
        "minimized",
        activate=False,
    )

    assert show_calls == ["normal"]
    minimized.close_from_owner()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    offscreen = _MovedDisplayWindow()
    offscreen.show_for_setup(
        FigureSubplotsState(figsize=(1.0, 1.0), dpi=100.0),
        "offscreen",
        activate=False,
    )

    assert offscreen.moved_to
    assert _FakeScreen().availableGeometry().intersects(offscreen.frameGeometry())
    offscreen.close_from_owner()
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
    controller = object.__new__(workspace_controller._WorkspaceController)
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

    assert tool.auto_redraw_check.isChecked()
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
    tool.operation_editor.request_update(label="manual")

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
    tool.operation_editor.request_update(label="catch up")
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
    controller = object.__new__(workspace_controller._WorkspaceController)
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

    monkeypatch.setattr(figure_window_ui.FigureCanvas, "draw_idle", record_draw_idle)
    tool.sigInfoChanged.connect(
        lambda: info_changes.append(tool.tool_status.setup.figsize)
    )

    tool._figure_window_canvas_size_changed(4.0, 2.5)
    assert tool.tool_status.setup.figsize == (4.0, 2.5)
    assert np.isclose(tool.layout_panel.width_spin.value(), 4.0)
    assert np.isclose(tool.layout_panel.height_spin.value(), 2.5)
    assert draw_idle_calls == []
    assert info_changes == []

    tool._figure_window_canvas_size_changed(4.5, 3.0)
    assert tool.tool_status.setup.figsize == (4.5, 3.0)
    qtbot.waitUntil(lambda: len(draw_idle_calls) == 1, timeout=1000)

    assert info_changes == [(4.5, 3.0)]


def test_figure_composer_canvas_resize_debounces_history(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool._reset_history_stack()

    initial_history_len = len(tool._prev_states)
    tool._figure_window_canvas_size_changed(4.0, 2.5)
    tool._figure_window_canvas_size_changed(4.5, 3.0)
    tool._figure_window_canvas_size_changed(5.0, 3.5)

    assert tool.tool_status.setup.figsize == (5.0, 3.5)
    assert tool._figure_resize_history_pending
    assert len(tool._prev_states) == initial_history_len

    assert tool._flush_pending_figure_resize_history_write()
    assert not tool._figure_resize_history_pending
    assert len(tool._prev_states) == initial_history_len + 1
    assert tool._prev_states[-1].setup.figsize == (5.0, 3.5)
    assert not tool._flush_pending_figure_resize_history_write()


def test_figure_composer_canvas_resize_undo_flushes_history(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool._reset_history_stack()

    initial_size = tool.tool_status.setup.figsize
    resized_size = (initial_size[0] + 0.75, initial_size[1] + 0.5)
    tool._figure_window_canvas_size_changed(*resized_size)

    assert tool._figure_resize_history_pending
    tool.undo()

    assert not tool._figure_resize_history_pending
    assert tool.tool_status.setup.figsize == initial_size
    assert not tool.undoable
    assert tool.redoable

    tool.redo()

    assert tool.tool_status.setup.figsize == tuple(
        round(value, 4) for value in resized_size
    )


def test_figure_composer_canvas_resize_save_flushes_history(qtbot) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    )
    qtbot.addWidget(tool)
    tool._reset_history_stack()

    tool._figure_window_canvas_size_changed(4.0, 2.5)
    assert tool._figure_resize_history_pending

    state = FigureRecipeState.model_validate_json(tool.to_dataset().attrs["tool_state"])

    assert not tool._figure_resize_history_pending
    assert state.setup.figsize == (4.0, 2.5)
    assert tool._prev_states[-1].setup.figsize == (4.0, 2.5)


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

    assert tool._document.source_names() == ("known", "missing")
    assert tuple(
        tool.operation_editor.source_display_name(name) for name in ("known", "missing")
    ) == ("known", "missing")
    assert tool.operation_editor.source_tooltip("known")
    tool._refresh_source_list()
    source_names = {
        tool.source_panel.source_list.topLevelItem(row).data(
            0, QtCore.Qt.ItemDataRole.UserRole
        )
        for row in range(tool.source_panel.source_list.topLevelItemCount())
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

    assert not tool.operation_panel.delete_button.isEnabled()
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
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    assert not tool.tool_status.operations[0].enabled
    tool.operation_panel.operation_list.clearSelection()
    tool.operation_panel.operation_list.setCurrentIndex(QtCore.QModelIndex())
    tool._update_step_action_buttons()
    assert not tool.operation_panel.delete_button.isEnabled()

    tool.axes_selector.set_selected_axes((), emit=False)
    assert tool._selected_axes_state().axes == ((0, 0),)
    old_setup = tool.tool_status.setup
    tool.layout_panel.width_ratios_edit.setText("0")
    tool.layout_panel.width_ratios_edit.editingFinished.emit()
    assert tool.tool_status.setup == old_setup

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

    grid_tool.layout_panel.set_setup(grid_tool.tool_status.setup)
    assert not grid_tool.layout_panel.gridspec_parent_grid_button.isEnabled()
    assert figurecomposer_gridspec._gridspec_has_invalid_regions(root)
    grid_tool.layout_panel.gridspec_layout_widget.set_selected_region("")
    grid_tool.layout_panel.gridspec_layout_widget.sigRegionChanged.emit(
        "ax0", outside_span
    )
    grid_tool.layout_panel.gridspec_layout_widget.sigRegionChanged.emit("ax0", span_01)
    grid_tool.layout_panel.gridspec_layout_widget.sigRegionCreated.emit(
        outside_span, "axes"
    )
    grid_tool.layout_panel.gridspec_layout_widget.sigRegionCreated.emit(span_00, "axes")
    grid_tool.layout_panel.gridspec_layout_widget.sigRegionCreated.emit(span_01, "grid")

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
    assert (
        figurecomposer_gridspec._gridspec_nearest_axes_after_region_delete(
            grid_tool.tool_status.setup, "missing", "ax0"
        )
        == ""
    )
    assert (
        figurecomposer_gridspec._gridspec_nearest_axes_after_region_delete(
            grid_tool.tool_status.setup, "root", "missing"
        )
        == ""
    )

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
    grid_tool.tool_status = grid_tool.tool_status.model_copy(
        update={
            "setup": FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=nested_root),
            )
        }
    )
    grid_tool.layout_panel.set_setup(grid_tool.tool_status.setup)
    assert figurecomposer_gridspec._gridspec_has_invalid_regions(nested_root)


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

    source_combo = tool.operation_editor.source_combo(
        ("data",),
        "stale_source",
        committed.append,
    )
    assert source_combo.findData("stale_source") >= 0
    assert source_combo.findData("data") >= 0
    _activate_combo_index(source_combo, source_combo.findData("data"))
    assert committed[-1] == "data"

    mixed_source_combo = tool.operation_editor.source_combo(
        ("data",),
        "data",
        committed.append,
        mixed=True,
    )
    before = list(committed)
    assert mixed_source_combo.currentData() is _editor_controls.MIXED_VALUE
    _activate_combo_index(mixed_source_combo, mixed_source_combo.currentIndex())
    assert committed == before

    name_combo = tool.operation_editor.optional_name_combo(
        ("kx",),
        "missing_dim",
        "auto",
        committed.append,
    )
    assert name_combo.findData("missing_dim") >= 0
    _activate_combo_index(name_combo, name_combo.findData(None))
    assert committed[-1] is None

    mixed_name_combo = tool.operation_editor.optional_name_combo(
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

    mixed_check = tool.operation_editor.check_box(False, committed.append, mixed=True)
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.gridspec_axes_selector.set_selected_axes_ids(("left",), emit=False)
    tool._gridspec_axes_selection_changed()
    assert tool.tool_status.operations[1].axes.axes_ids == ()


def test_figure_composer_undo_redo_setup_state(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    initial = tool.tool_status

    tool.layout_panel.nrows_spin.setValue(initial.setup.nrows + 1)

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

    assert np.isinf(tool.layout_panel.nrows_spin.maximum())
    assert np.isinf(tool.layout_panel.ncols_spin.maximum())

    tool.layout_panel.nrows_spin.setValue(13)
    assert tool.layout_panel.nrows_spin.value() == 13
    assert tool.tool_status.setup.nrows == 13

    tool.layout_panel.nrows_spin.setValue(1)
    tool.layout_panel.ncols_spin.setValue(13)
    assert tool.layout_panel.ncols_spin.value() == 13
    assert tool.tool_status.setup.ncols == 13


def test_figure_composer_gridspec_controls_accept_more_than_twelve(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")

    tool.layout_panel.nrows_spin.setValue(13)
    assert tool.layout_panel.nrows_spin.value() == 13
    assert tool.tool_status.setup.gridspec.root.nrows == 13

    tool.layout_panel.nrows_spin.setValue(1)
    tool.layout_panel.ncols_spin.setValue(13)
    assert tool.layout_panel.ncols_spin.value() == 13
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
    assert tool._image_mappable_target(object(), operations) is None
    bad_panel_mappable = type("BadPanelMappable", (), {})()
    setattr(
        bad_panel_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        operation.operation_id,
    )
    setattr(
        bad_panel_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
        ("bad", 0),
    )
    assert tool._image_mappable_target(bad_panel_mappable, operations) is None
    missing_operation_mappable = type("MissingOperationMappable", (), {})()
    setattr(
        missing_operation_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        "missing",
    )
    setattr(
        missing_operation_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
        (0, 0),
    )
    assert tool._image_mappable_target(missing_operation_mappable, operations) is None
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

    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ("render_error",)
    assert "RuntimeError: boom" in item.toolTip(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN
    )
    assert _operation_source_status_label(tool).isHidden()

    tool._replace_operation(
        0,
        operation.model_copy(update={"code": "ax.set_title('ok')"}),
    )

    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ()
    assert "RuntimeError: boom" not in item.toolTip(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN
    )
    assert _operation_source_status_label(tool).isHidden()


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
    tool.operation_editor.connect_line_edit_finished(
        edit,
        lambda text: tool.operation_editor.request_update(
            extra_kwargs=figurecomposer_text._dict_from_text(text),
        ),
    )

    edit.setText("{alpha: 0.5}")
    edit.editingFinished.emit()

    assert tool.operation_editor.has_input_error(operation)
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ("invalid_input",)
    assert "Invalid input:" in item.toolTip(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN
    )
    assert not _operation_source_status_label(tool).isHidden()
    with pytest.raises(ValueError, match="invalid step inputs"):
        tool.generated_code()

    check = QtWidgets.QCheckBox(tool)
    tool.operation_editor.connect_signal(
        check,
        check.toggled,
        lambda checked: tool.operation_editor.request_update(transpose=checked),
    )
    check.setChecked(not operation.transpose)

    assert tool.operation_editor.has_input_error(operation)

    edit.setText("alpha=0.5")
    edit.editingFinished.emit()

    assert not tool.operation_editor.has_input_error(operation)
    assert tool.tool_status.operations[0].extra_kwargs == {"alpha": 0.5}
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ()
    assert "Invalid input:" not in item.toolTip(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN
    )
    assert _operation_source_status_label(tool).isHidden()


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
    tool.operation_editor.set_input_errors(
        {operation.operation_id: {input_key: "old error"}}
    )

    def delete_sender() -> None:
        edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    tool.operation_editor.connect_signal(edit, edit.editingFinished, delete_sender)
    edit.editingFinished.emit()

    assert not erlab.interactive.utils.qt_is_valid(edit)
    assert not tool.operation_editor.has_input_error(operation)

    error_edit = QtWidgets.QLineEdit(tool)
    error_edit.setObjectName("figureComposerDeletedErrorSenderEdit")

    def delete_sender_with_error() -> None:
        error_edit.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        raise FigureComposerInputError("new error")

    tool.operation_editor.connect_signal(
        error_edit, error_edit.editingFinished, delete_sender_with_error
    )
    error_edit.editingFinished.emit()

    assert not erlab.interactive.utils.qt_is_valid(error_edit)
    assert tool.operation_editor.has_input_error(operation)
    assert tool.operation_editor.input_error_text(operation) == "new error"


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
    tool.operation_editor.connect_plain_text_changed(
        plain,
        lambda text: tool.operation_editor.request_update(label=text),
    )
    spin = QtWidgets.QSpinBox(tool)
    tool.operation_editor.connect_value_signal(
        spin,
        spin.valueChanged,
        int,
        lambda value: tool.operation_editor.request_update(label=f"value {value}"),
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    operation_item = tool.operation_panel.operation_list.topLevelItem(0)
    assert operation_item is not None
    operation_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].enabled is False
    assert render_calls == [(tool,)]
    assert info_changed == [None]

    render_calls.clear()
    info_changed.clear()
    edit = QtWidgets.QLineEdit(tool)
    tool.operation_editor.connect_line_edit_finished(
        edit,
        lambda text: tool.operation_editor.request_update(label=text),
    )
    edit.setText("disabled edit")
    edit.editingFinished.emit()

    assert tool.tool_status.operations[0].label == "disabled edit"
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None]

    tool.operation_editor.request_update(label="programmatic disabled edit")

    assert tool.tool_status.operations[0].label == "programmatic disabled edit"
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None, None]

    operation_item = tool.operation_panel.operation_list.topLevelItem(0)
    assert operation_item is not None
    operation_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
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


def test_figure_composer_default_dpi_option_overrides_stylesheet(
    monkeypatch,
    restore_interactive_options,
) -> None:
    options.model = options.model.model_copy(
        update={"figure": FigureOptions(stylesheets=["classic"], dpi=180.0)}
    )
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_configured_stylesheets",
        lambda: ("classic",),
    )

    assert figurecomposer_defaults._default_figure_dpi() == 180.0
    assert FigureSubplotsState().dpi == 180.0


def test_figure_composer_dpi_option_affects_only_new_recipes(
    qtbot,
    restore_interactive_options,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    options.model = options.model.model_copy(
        update={"figure": FigureOptions(dpi=120.0)}
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    assert tool.tool_status.setup.dpi == 120.0

    options.model = options.model.model_copy(
        update={"figure": FigureOptions(dpi=220.0)}
    )
    new_tool = FigureComposerTool(data)
    qtbot.addWidget(new_tool)

    assert tool.tool_status.setup.dpi == 120.0
    assert new_tool.tool_status.setup.dpi == 220.0


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


def test_figure_composer_defaults_avoid_stylesheet_registry_when_unconfigured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(figurecomposer_defaults, "_configured_stylesheets", tuple)

    def unexpected_stylesheet_lookup(*_args, **_kwargs):
        raise AssertionError("empty stylesheet settings must not load the registry")

    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "available_stylesheets",
        unexpected_stylesheet_lookup,
    )

    assert figurecomposer_defaults._available_configured_stylesheets() == ()
    assert figurecomposer_defaults._unavailable_configured_stylesheets() == ()
    assert figurecomposer_defaults._style_required_imports() == ()


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


def test_figure_composer_visible_redraw_caches_drawn_canvas(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)
    tool._clear_preview_pixmap_cache(stale=True)
    previous_generation = tool.preview_pixmap_generation

    draw_calls = 0
    original_draw = tool.canvas.draw

    def counted_draw() -> None:
        nonlocal draw_calls
        draw_calls += 1
        original_draw()

    monkeypatch.setattr(tool.canvas, "draw", counted_draw)

    figurecomposer_rendering._render_preview(tool, show_window=True)

    assert draw_calls == 1
    assert tool.preview_pixmap is not None
    assert not tool.preview_pixmap.isNull()
    assert tool.preview_pixmap_generation == previous_generation + 1
    assert not tool.preview_pixmap_stale

    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda *_args, **_kwargs: pytest.fail(
            "a current canvas preview should not schedule another draw"
        ),
    )
    tool.request_preview_pixmap_update(delay_ms=0)


def test_figure_composer_post_draw_invalidation_keeps_preview_stale(
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
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)
    tool._clear_preview_pixmap_cache(stale=True)

    flush_calls = 0

    def flush_events() -> None:
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 2:
            tool._mark_preview_pixmap_stale()

    monkeypatch.setattr(tool.canvas, "flush_events", flush_events)

    figurecomposer_rendering._render_preview(tool, show_window=True)

    assert flush_calls == 2
    assert tool.preview_pixmap is not None
    assert tool.preview_pixmap_stale


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
    monkeypatch.setattr(figure_window_ui.FigureCanvas, "draw", draw)
    monkeypatch.setattr(figure_window_ui.FigureCanvas, "print_figure", print_figure)

    canvas = figure_window_ui._StyledFigureCanvas(figure_window_ui.Figure())
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


def test_figure_composer_exports_pdf(qtbot, monkeypatch, tmp_path) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    export_path = tmp_path / "figure.pdf"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), ""),
    )

    tool.export_figure()

    assert export_path.read_bytes().startswith(b"%PDF")


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
    original_draw = figure_window_ui.FigureCanvas.draw

    def draw_with_warning(self, *args, **kwargs):
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        return original_draw(self, *args, **kwargs)

    monkeypatch.setattr(figure_window_ui.FigureCanvas, "draw", draw_with_warning)

    assert tool.preview_pixmap is None
    assert tool.refresh_preview_pixmap() is None
    assert not any(
        "constrained_layout not applied" in str(warning.message) for warning in recwarn
    )


def test_figure_composer_operation_table_presents_targets_and_selects_rows(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.set_palette(),
                FigureOperationState.plot_array(
                    label="image",
                    source="data",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ),
                FigureOperationState.plot_slices(
                    label="slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(expression="axs[:, 0]"),
                ),
                FigureOperationState.custom(label="custom", code="pass", trusted=True),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    operation_list = tool.operation_panel.operation_list
    assert isinstance(operation_list, QtWidgets.QTreeWidget)
    assert operation_list.columnCount() == 3
    assert (
        operation_list.selectionBehavior()
        == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    assert isinstance(
        operation_list.itemDelegateForColumn(
            figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN
        ),
        axes_widgets._AxesTargetItemDelegate,
    )
    header = operation_list.header()
    assert (
        header.sectionSize(figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN)
        <= 80
    )
    assert (
        header.sectionSize(figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN)
        <= 96
    )

    palette_item = operation_list.topLevelItem(0)
    image_item = operation_list.topLevelItem(1)
    expression_item = operation_list.topLevelItem(2)
    custom_item = operation_list.topLevelItem(3)
    assert all(
        item is not None
        for item in (palette_item, image_item, expression_item, custom_item)
    )
    palette_descriptor = palette_item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    custom_descriptor = custom_item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    assert palette_descriptor[0] == custom_descriptor[0] == "text"
    assert palette_descriptor != custom_descriptor

    image_descriptor = image_item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    expression_descriptor = expression_item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    assert image_descriptor[0] == expression_descriptor[0] == "layout"
    assert [entry[-1] for entry in image_descriptor[2]] == [False, True, False, False]
    assert [entry[-1] for entry in expression_descriptor[2]] == [
        True,
        False,
        True,
        False,
    ]
    unresolved_descriptor = tool._operation_target_preview_descriptor(
        tool.tool_status.operations[2].model_copy(
            update={
                "axes": FigureAxesSelectionState(expression="axs + axs"),
            }
        )
    )
    assert unresolved_descriptor[-1] is True
    expression_item.setData(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
        unresolved_descriptor,
    )
    assert (
        expression_item.data(
            figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
        )
        == "axs[:, 0]"
    )
    for row in range(operation_list.topLevelItemCount()):
        item = operation_list.topLevelItem(row)
        assert item is not None
        assert item.sizeHint(0).height() <= 24
        for column in range(operation_list.columnCount()):
            assert operation_list.itemWidget(item, column) is None

    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.show()
    operation_list.scrollToItem(expression_item)
    initial_list_height = operation_list.height()
    for row in range(operation_list.topLevelItemCount()):
        operation_list.setCurrentItem(operation_list.topLevelItem(row))
        QtWidgets.QApplication.processEvents()
        assert operation_list.height() == initial_list_height
        assert (
            tool.operation_editor.scroll_area.minimumSizeHint().width()
            >= tool.operation_editor.stack.minimumSizeHint().width()
        )
    operation_list.setCurrentItem(palette_item)
    QtWidgets.QApplication.processEvents()
    target_index = operation_list.indexFromItem(
        expression_item,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
    )
    target_rect = operation_list.visualRect(target_index)
    assert not target_rect.isEmpty()
    current_page = tool.operation_editor.stack.currentWidget()
    assert current_page is not None
    current_control = next(
        widget
        for widget in current_page.findChildren(QtWidgets.QWidget)
        if tool.operation_editor.control_signal_allowed(widget)
    )
    qtbot.mouseClick(
        operation_list.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=target_rect.center(),
    )
    assert operation_list.indexOfTopLevelItem(operation_list.currentItem()) == 2
    assert expression_item.isSelected()
    assert tool.operation_editor.control_signal_allowed(current_control)
    qtbot.waitUntil(
        lambda: not tool.operation_editor.control_signal_allowed(current_control),
        timeout=1000,
    )
    assert operation_list.height() == initial_list_height


def test_figure_operation_panel_emits_stable_identity_intentions(qtbot) -> None:
    tabs = QtWidgets.QTabWidget()
    qtbot.addWidget(tabs)
    panel = figurecomposer_operation_panel.FigureOperationPanel(
        tabs,
        (
            figurecomposer_operation_panel.FigureOperationAction(
                "custom", "Python", "Add Python code."
            ),
        ),
    )
    tabs.addTab(panel, "Recipe")
    rows = (
        figurecomposer_operation_panel.FigureOperationRow(
            "first",
            "First",
            True,
            "First step",
            ("text", "—"),
            "No target",
            "",
            (),
            "",
        ),
        figurecomposer_operation_panel.FigureOperationRow(
            "second",
            "Second",
            True,
            "Second step",
            ("text", "—"),
            "No target",
            "Invalid input",
            ("invalid_input",),
            "Invalid input: bad value",
        ),
    )
    panel.set_rows(rows, selected_ids={"first"}, current_id="first")
    assert panel.current_id() == "first"
    assert panel.selected_ids() == frozenset({"first"})
    assert panel.operation_list._operation_ids() == ("first", "second")

    enabled_requests: list[tuple[str, bool]] = []
    selection_requests: list[tuple[str | None, frozenset[str]]] = []
    added: list[str] = []
    panel.enabled_requested.connect(
        lambda operation_id, enabled: enabled_requests.append((operation_id, enabled))
    )
    panel.selection_changed.connect(
        lambda current_id, selected_ids: selection_requests.append(
            (current_id, selected_ids)
        )
    )
    panel.add_requested.connect(added.append)

    second = panel.operation_list.topLevelItem(1)
    second.setCheckState(
        figurecomposer_operation_panel._OPERATION_LIST_STEP_COLUMN,
        QtCore.Qt.CheckState.Unchecked,
    )
    assert enabled_requests == [("second", False)]
    panel.operation_list.setCurrentItem(second)
    assert selection_requests[-1] == ("second", frozenset({"second"}))

    panel.set_current_row(1, preserve_selection=False)
    panel.set_selected_ids({"second"})
    previous_request_count = len(selection_requests)
    panel.select_row(0)
    assert len(selection_requests) == previous_request_count + 1
    assert selection_requests[-1] == ("first", frozenset({"first"}))
    panel.add_step_menu.actions()[0].trigger()
    assert added == ["custom"]

    previous_request_count = len(selection_requests)
    panel._selection_input_event = True
    panel.operation_list.setCurrentItem(second)
    assert panel._selection_notification_pending
    assert len(selection_requests) == previous_request_count
    panel.release()
    qtbot.wait(1)
    assert len(selection_requests) == previous_request_count


def test_figure_composer_operation_table_uses_text_for_single_axes(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"))
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(FigureOperationState.plot_array(label="image", source="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    ) == ("single_axes",)
    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.show()
    qtbot.wait(1)

    root = FigureGridSpecGridState(
        grid_id="root",
        axes=(
            FigureGridSpecAxesState(
                axes_id="only-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    assert axes_widgets._gridspec_target_preview_descriptor(root, ("only-axis",)) == (
        "single_axes",
    )
    assert axes_widgets._gridspec_target_preview_descriptor(root, ())[0] == ("layout")

    thin_child = FigureGridSpecGridState(
        grid_id="thin-child",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
        axes=(
            FigureGridSpecAxesState(
                axes_id="thin-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    thin_root = FigureGridSpecGridState(
        grid_id="thin-root",
        ncols=400,
        child_grids=(thin_child,),
    )
    thin_descriptor = axes_widgets._gridspec_target_preview_descriptor(
        thin_root, ("thin-axis",)
    )
    assert [entry[0] for entry in thin_descriptor[2]].count("grid") == 1
    assert all(entry[0] != "axis" for entry in thin_descriptor[2])


def test_figure_composer_operation_row_target_and_source_drag_edges(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.plot_array(
                    label="image", source="data", axes=FigureAxesSelectionState()
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool._set_current_operation_row_silent(99, preserve_selection=False)
    assert not tool.operation_panel.operation_list.currentIndex().isValid()

    empty_expression = tool.tool_status.operations[0].model_copy(
        update={"axes": FigureAxesSelectionState(expression="axs[:0, :0]")}
    )
    descriptor = tool._operation_target_preview_descriptor(empty_expression)
    assert descriptor[-1] is True
    assert not any(entry[-1] for entry in descriptor[2])

    mime = QtCore.QMimeData()
    tool.source_panel.set_drop_handlers(
        lambda data: data is mime, lambda data: data is mime
    )

    def drag_enter_event() -> QtGui.QDragEnterEvent:
        return QtGui.QDragEnterEvent(
            QtCore.QPoint(),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def drag_move_event() -> QtGui.QDragMoveEvent:
        return QtGui.QDragMoveEvent(
            QtCore.QPoint(),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def drop_event() -> QtGui.QDropEvent:
        return QtGui.QDropEvent(
            QtCore.QPointF(),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    filtered_event = drag_enter_event()
    assert tool.source_panel.eventFilter(tool, filtered_event)
    assert filtered_event.isAccepted()

    for handler, event in (
        (tool.source_panel.dragEnterEvent, drag_enter_event()),
        (tool.source_panel.dragMoveEvent, drag_move_event()),
        (tool.source_panel.dropEvent, drop_event()),
    ):
        handler(event)
        assert event.isAccepted()

    tool.source_panel.set_drop_handlers(lambda _data: False, lambda _data: False)
    tool.source_panel.dragEnterEvent(drag_enter_event())
    tool.source_panel.dragMoveEvent(drag_move_event())
    tool.source_panel.dropEvent(drop_event())


def test_figure_composer_operation_target_preview_uses_axes_selector_palette(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"))
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.plot_array(
                    label="image",
                    source="data",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    index = tool.operation_panel.operation_list.indexFromItem(
        item, figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN
    )
    observed_sources: list[QtWidgets.QWidget] = []
    selector_colors = axes_widgets._selector_colors

    def record_color_source(
        widget: QtWidgets.QWidget,
    ) -> axes_widgets._SelectorColors:
        observed_sources.append(widget)
        return selector_colors(widget)

    monkeypatch.setattr(axes_widgets, "_selector_colors", record_color_source)
    option = QtWidgets.QStyleOptionViewItem()
    option.initFrom(tool.operation_panel.operation_list)
    option.widget = tool.operation_panel.operation_list
    option.rect = QtCore.QRect(0, 0, 80, 24)
    pixmap = QtGui.QPixmap(option.rect.size())
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    delegate = tool.operation_panel.findChild(axes_widgets._AxesTargetItemDelegate)
    assert delegate is not None
    try:
        delegate.paint(painter, option, index)
    finally:
        painter.end()

    assert observed_sources == [tool.axes_selector]


def test_figure_composer_operation_target_delegate_handles_empty_previews(
    qtbot, monkeypatch
) -> None:
    view = QtWidgets.QTreeWidget()
    view.setColumnCount(1)
    qtbot.addWidget(view)
    color_source = QtWidgets.QWidget()
    qtbot.addWidget(color_source)
    item = QtWidgets.QTreeWidgetItem(view)
    descriptor_role = int(QtCore.Qt.ItemDataRole.UserRole) + 77
    delegate = axes_widgets._AxesTargetItemDelegate(descriptor_role, color_source, view)
    index = view.indexFromItem(item, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.initFrom(view)
    option.widget = view
    draw_calls: list[QtCore.QRectF] = []
    draw_selector_rect = axes_widgets._draw_selector_rect

    def record_selector_rect(
        painter: QtGui.QPainter,
        rect: QtCore.QRect | QtCore.QRectF,
        **kwargs: typing.Any,
    ) -> None:
        draw_calls.append(QtCore.QRectF(rect))
        draw_selector_rect(painter, rect, **kwargs)

    monkeypatch.setattr(axes_widgets, "_draw_selector_rect", record_selector_rect)

    def paint(descriptor: object, rect: QtCore.QRect) -> None:
        item.setData(0, descriptor_role, descriptor)
        option.rect = rect
        pixmap = QtGui.QPixmap(max(1, rect.width()), max(1, rect.height()))
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        try:
            delegate.paint(painter, option, index)
        finally:
            painter.end()

    paint(None, QtCore.QRect(0, 0, 80, 24))
    assert draw_calls == []

    descriptor = (
        "layout",
        1.0,
        (("axis", 0.0, 0.0, 1.0, 1.0, False),),
        False,
    )
    paint(descriptor, QtCore.QRect(0, 0, 7, 3))
    assert draw_calls == []

    paint(
        (
            "layout",
            1.0,
            (("axis", 0.0, 0.0, 0.0, 0.0, False),),
            False,
        ),
        QtCore.QRect(0, 0, 80, 24),
    )
    assert len(draw_calls) == 1

    observed_sources: list[QtWidgets.QWidget] = []
    selector_colors = axes_widgets._selector_colors

    def record_color_source(
        widget: QtWidgets.QWidget,
    ) -> axes_widgets._SelectorColors:
        observed_sources.append(widget)
        return selector_colors(widget)

    monkeypatch.setattr(axes_widgets, "_selector_colors", record_color_source)
    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: False)
    draw_calls.clear()
    paint(descriptor, QtCore.QRect(0, 0, 80, 24))

    assert observed_sources == [view]
    assert len(draw_calls) == 2


def test_figure_composer_operation_table_presents_nested_gridspec_target(
    qtbot,
) -> None:
    child_axis = FigureGridSpecAxesState(
        axes_id="child-axis",
        span=FigureGridSpecSpanState(
            row_start=1,
            row_stop=2,
            col_start=0,
            col_stop=1,
        ),
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="root-axis",
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
                nrows=2,
                ncols=1,
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
                axes=(child_axis,),
            ),
            FigureGridSpecGridState(
                grid_id="outside-root",
                nrows=1,
                ncols=1,
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=2,
                    col_stop=3,
                ),
            ),
        ),
    )
    data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"))
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="slices",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes_ids=("child-axis",)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    descriptor = item.data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    assert descriptor[0] == "layout"
    assert any(entry[0] == "grid" for entry in descriptor[2])
    assert sum(entry[0] == "axis" and entry[-1] for entry in descriptor[2]) == 1
    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.show()
    qtbot.wait(10)


def test_figure_composer_operation_table_centralizes_status_and_layout_refresh(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"))
    operation = FigureOperationState.plot_array(
        label="image",
        source="missing",
        axes=FigureAxesSelectionState(axes=((0, 1),)),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data"),
                FigureSourceState(name="missing"),
            ),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    assert _operation_status_codes(tool, 0) == (
        "invalid_target",
        "missing_source",
    )
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    assert item.text(
        figurecomposer_operation_panel._OPERATION_LIST_STEP_COLUMN
    ) == tool._operation_display_text(tool.tool_status.operations[0])

    tool.operation_editor.set_input_errors(
        {operation.operation_id: {"test": "invalid value"}}
    )
    tool._set_operation_render_errors({operation.operation_id: "render failed"})
    assert _operation_status_codes(tool, 0) == (
        "invalid_target",
        "missing_source",
        "invalid_input",
        "render_error",
    )
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None
    status_description = item.data(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN,
        QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
    )
    assert "invalid value" in status_description
    assert "render failed" in status_description

    tool.operation_editor.set_input_errors({})
    tool._set_operation_render_errors({})
    tool.layout_panel.ncols_spin.setValue(2)
    assert _operation_status_codes(tool, 0) == ("missing_source",)
    descriptor = tool.operation_panel.operation_list.topLevelItem(0).data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    assert [entry[-1] for entry in descriptor[2]] == [False, True]

    tool.undo()
    assert "invalid_target" in _operation_status_codes(tool, 0)
    descriptor = tool.operation_panel.operation_list.topLevelItem(0).data(
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
        figurecomposer_operation_panel._OPERATION_LIST_TARGET_ROLE,
    )
    assert [entry[-1] for entry in descriptor[2]] == [False]


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

    delete_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerDeleteStepButton"
    )
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerDuplicateStepButton")
        is None
    )
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerMoveStepUpButton") is None
    )
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerMoveStepDownButton")
        is None
    )
    assert delete_button is tool.operation_panel.delete_button

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    menu, move_up_action = _operation_context_action(
        tool, "figureComposerContextMoveStepUpAction"
    )
    move_down_action = next(
        action
        for action in menu.actions()
        if action.objectName() == "figureComposerContextMoveStepDownAction"
    )
    assert move_up_action.isEnabled() is False
    assert move_down_action.isEnabled() is True
    menu.close()

    _select_operation_rows(tool, (1,))
    second = tool.tool_status.operations[1]
    menu, duplicate_action = _operation_context_action(
        tool, "figureComposerContextDuplicateStepAction"
    )
    duplicate_action.trigger()
    menu.close()
    duplicate = tool.tool_status.operations[2]
    assert tool._current_operation_index() == 2
    assert len(tool.tool_status.operations) == 4
    assert duplicate.operation_id != second.operation_id
    assert duplicate.model_dump(exclude={"operation_id"}) == second.model_dump(
        exclude={"operation_id"}
    )

    duplicate_id = duplicate.operation_id
    menu, move_up_action = _operation_context_action(
        tool, "figureComposerContextMoveStepUpAction"
    )
    move_up_action.trigger()
    menu.close()
    assert tool._current_operation_index() == 1
    assert tool.tool_status.operations[1].operation_id == duplicate_id
    menu, move_down_action = _operation_context_action(
        tool, "figureComposerContextMoveStepDownAction"
    )
    move_down_action.trigger()
    menu.close()
    assert tool._current_operation_index() == 2
    assert tool.tool_status.operations[2].operation_id == duplicate_id

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(3)
    )
    menu, move_up_action = _operation_context_action(
        tool, "figureComposerContextMoveStepUpAction"
    )
    move_down_action = next(
        action
        for action in menu.actions()
        if action.objectName() == "figureComposerContextMoveStepDownAction"
    )
    assert move_up_action.isEnabled() is True
    assert move_down_action.isEnabled() is False
    menu.close()

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
    menu, duplicate_action = _operation_context_action(
        tool, "figureComposerContextDuplicateStepAction"
    )
    duplicate_action.trigger()
    menu.close()

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
    assert tool._current_operation_index() == 4
    assert [
        tool.tool_status.operations[index].operation_id for index in (4, 5)
    ] != selected_originals

    operation_list = tool.operation_panel.operation_list
    with QtCore.QSignalBlocker(operation_list):
        operation_list.setCurrentItem(
            operation_list.topLevelItem(5),
            0,
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )
    tool.operation_panel._synchronize_selection_cache()
    current_duplicate_id = tool.operation_panel.current_id()
    duplicate_ids = {
        tool.tool_status.operations[index].operation_id for index in (4, 5)
    }
    menu, move_up_action = _operation_context_action(
        tool, "figureComposerContextMoveStepUpAction"
    )
    move_up_action.trigger()
    menu.close()
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
    assert tool.operation_panel.current_id() == current_duplicate_id

    _select_operation_rows(tool, (0, 2, 5))
    selected_ids = {
        tool.tool_status.operations[index].operation_id for index in (0, 2, 5)
    }
    menu, move_down_action = _operation_context_action(
        tool, "figureComposerContextMoveStepDownAction"
    )
    move_down_action.trigger()
    menu.close()
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

    tool.operation_panel.delete_button.click()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "b",
        "b",
        "d",
        "e",
    ]
    assert _selected_operation_rows(tool) == (1,)
    assert tool._current_operation_index() == 1


def test_figure_composer_drag_reorders_steps_and_history(qtbot) -> None:
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

    assert (
        tool.operation_panel.operation_list.dragDropMode()
        == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
    )
    assert (
        tool.operation_panel.operation_list.defaultDropAction()
        == QtCore.Qt.DropAction.MoveAction
    )
    assert tool.operation_panel.operation_list.showDropIndicator()

    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.show()
    qtbot.waitUntil(tool.isVisible)

    _select_operation_rows(tool, (1, 2))
    first_moved = tool.operation_panel.operation_list.takeTopLevelItem(1)
    second_moved = tool.operation_panel.operation_list.takeTopLevelItem(1)
    assert first_moved is not None
    assert second_moved is not None
    tool.operation_panel.operation_list.insertTopLevelItem(2, first_moved)
    tool.operation_panel.operation_list.insertTopLevelItem(3, second_moved)
    tool.operation_panel.operation_list.setCurrentItem(first_moved)
    first_moved.setSelected(True)
    second_moved.setSelected(True)
    tool.operation_panel.operation_list._queue_rows_reordered()

    assert tool.operation_panel.operation_list.topLevelItemCount() == 4
    assert [operation.label for operation in tool.tool_status.operations] == list(
        "abcd"
    )
    qtbot.waitUntil(
        lambda: (
            [operation.label for operation in tool.tool_status.operations]
            == ["a", "d", "b", "c"]
        )
    )

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "d",
        "b",
        "c",
    ]
    assert all(
        tool.operation_panel.operation_list.topLevelItem(row).childCount() == 0
        for row in range(tool.operation_panel.operation_list.topLevelItemCount())
    )
    assert _selected_operation_rows(tool) == (2, 3)
    assert tool._current_operation_index() == 2

    namespace: dict[str, typing.Any] = {}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["fig"].__dict__["_order"] == ["a", "d", "b", "c"]

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
        "d",
        "b",
        "c",
    ]


def test_figure_composer_operation_list_keyboard_context_menu(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=tuple(_custom_order_step(label) for label in "ab"),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )

    event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Menu,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    tool.operation_panel.operation_list.keyPressEvent(event)

    assert event.isAccepted()
    move_up_action = tool.operation_panel.operation_list.findChild(
        QtGui.QAction, "figureComposerContextMoveStepUpAction"
    )
    assert move_up_action is not None
    menu = move_up_action.parent()
    assert isinstance(menu, QtWidgets.QMenu)
    action_names = {
        action.objectName() for action in menu.actions() if action.objectName()
    }
    assert "figureComposerContextMoveStepUpAction" in action_names
    assert "figureComposerContextMoveStepDownAction" in action_names
    menu.close()


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
    tool.operation_panel.copy_button.click()
    _select_operation_rows(tool, (0,))
    tool.operation_panel.paste_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "b",
        "d",
        "b",
        "c",
        "d",
    ]
    assert _selected_operation_rows(tool) == (1, 2)
    assert tool._current_operation_index() == 1
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
    tool.operation_panel.cut_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]
    assert _selected_operation_rows(tool) == (1,)
    assert tool._current_operation_index() == 1
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

    tool.operation_panel.paste_button.click()

    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
        "b",
        "d",
    ]
    assert _selected_operation_rows(tool) == (2, 3)
    assert tool._current_operation_index() == 2
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
    assert not tool.operation_panel.paste_button.isEnabled()
    tool._paste_operations_from_clipboard()
    assert tool.tool_status == before

    clipboard.setText("fig.suptitle('Copied step code')")
    tool._update_step_action_buttons()
    assert not tool.operation_panel.paste_button.isEnabled()
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
    assert not tool.operation_panel.paste_button.isEnabled()
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
    tool.operation_panel.operation_list.keyPressEvent(copy_event)
    if not copy_event.isAccepted():
        copy_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        )
        tool.operation_panel.operation_list.keyPressEvent(copy_event)
    assert copy_event.isAccepted()

    menu, paste_action = _operation_context_action(
        tool, "figureComposerContextPasteStepsAction"
    )
    _select_operation_rows(tool, (1,))
    paste_action.trigger()
    menu.close()

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
    tool.operation_panel.operation_list.keyPressEvent(cut_event)
    if not cut_event.isAccepted():
        cut_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_X,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        )
        tool.operation_panel.operation_list.keyPressEvent(cut_event)
    assert cut_event.isAccepted()
    assert [operation.label for operation in tool.tool_status.operations] == [
        "a",
        "c",
    ]

    _select_operation_rows(tool, (0,))
    menu, cut_action = _operation_context_action(
        tool, "figureComposerContextCutStepsAction"
    )
    assert cut_action.isEnabled()
    cut_action.trigger()
    menu.close()

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
    operations, sources, source_data, selection_base_data = (
        figurecomposer_tool_module._step_clipboard_payload(mime)
    )
    assert [operation.label for operation in operations] == ["a"]
    assert sources == ()
    assert source_data == {}
    assert selection_base_data == {}


def test_figure_composer_operation_list_keypress_defensive_paths(qtbot) -> None:
    operation_list = figurecomposer_operation_panel._FigureComposerOperationList()
    qtbot.addWidget(operation_list)
    pasted: list[bool] = []
    operation_list.paste_requested.connect(lambda: pasted.append(True))

    operation_list.keyPressEvent(None)
    operation_list.dropEvent(None)
    operation_list._emit_rows_reordered()

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


def test_figure_composer_step_editor_and_reorder_defensive_paths(
    qtbot, monkeypatch
) -> None:
    scroll = _FigureComposerStepEditorScroll()
    qtbot.addWidget(scroll)
    empty_hint = scroll.minimumSizeHint()

    class _HintWidget(QtWidgets.QWidget):
        def minimumSizeHint(self) -> QtCore.QSize:
            return QtCore.QSize(220, 20)

    content = _HintWidget()
    scroll.setWidget(content)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    assert scroll.minimumSizeHint().width() > empty_hint.width()

    tabs = QtWidgets.QTabWidget()
    qtbot.addWidget(tabs)
    page = _FigureComposerStepEditorPage(tabs, tabs)
    page._background_color = QtGui.QColor(0, 0, 0, 0)
    monkeypatch.setattr(tabs, "render", lambda *_args, **_kwargs: None)
    page.refresh_background()
    assert page._background_color.alpha() == 0
    page.paintEvent(None)
    page.changeEvent(None)

    scheduled: list[tuple[QtCore.QObject, object]] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda owner, _delay, callback, *_args: scheduled.append((owner, callback)),
    )
    page.changeEvent(QtCore.QEvent(QtCore.QEvent.Type.PaletteChange))
    assert scheduled == [(page, page.refresh_background)]

    rows = figurecomposer_reorder_list.ReorderList(0)
    qtbot.addWidget(rows)
    invalid_item = QtWidgets.QTreeWidgetItem(rows)
    invalid_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, 1)
    assert rows._row_ids() == ()
    rows._emit_rows_reordered()

    rows.clear()
    for row_id in ("first", "second"):
        item = QtWidgets.QTreeWidgetItem(rows)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, row_id)
    emitted: list[tuple[tuple[str, ...], frozenset[str], str | None]] = []
    rows.rows_reordered.connect(
        lambda row_ids, selected_ids, current_id: emitted.append(
            (row_ids, selected_ids, current_id)
        )
    )
    rows.setCurrentIndex(QtCore.QModelIndex())
    rows._emit_rows_reordered()
    assert emitted[-1] == (("first", "second"), frozenset(), None)

    with monkeypatch.context() as context:
        context.setattr(rows, "currentItem", lambda: QtWidgets.QTreeWidgetItem())
        rows._emit_rows_reordered()
    assert emitted[-1][-1] is None

    scheduled.clear()
    rows._rows_reordered_pending = True
    rows._queue_rows_reordered()
    assert scheduled == []
    rows._rows_reordered_pending = False
    rows._queue_rows_reordered()
    assert scheduled == [(rows, rows._emit_rows_reordered)]
    rows._rows_reordered_pending = False

    mime = QtCore.QMimeData()
    external_drop = QtGui.QDropEvent(
        QtCore.QPointF(),
        QtCore.Qt.DropAction.MoveAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    rows.dropEvent(external_drop)
    assert not external_drop.isAccepted()

    class _InternalDropEvent(QtGui.QDropEvent):
        def source(self) -> QtCore.QObject:
            return rows

    internal_drop = _InternalDropEvent(
        QtCore.QPointF(),
        QtCore.Qt.DropAction.MoveAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    queued: list[bool] = []
    monkeypatch.setattr(rows, "_queue_rows_reordered", lambda: queued.append(True))
    with monkeypatch.context() as context:
        context.setattr(
            QtWidgets.QTreeWidget,
            "dropEvent",
            lambda _tree, event: event.accept(),
        )
        rows.dropEvent(internal_drop)
    assert internal_drop.isAccepted()
    assert queued == [True]


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
    tool.operation_panel.operation_list.setCurrentIndex(QtCore.QModelIndex())
    tool.operation_panel.operation_list.clearSelection()
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
            tool._document, FigureAxesSelectionState(axes=axes), for_plot_slices=False
        )

    def axes_sequence_code(axes: tuple[tuple[int, int], ...]) -> str:
        return figurecomposer_code._axes_sequence_code(
            tool._document, FigureAxesSelectionState(axes=axes)
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
            tool._document,
            FigureAxesSelectionState(axes=((0, 1),)),
            for_plot_slices=True,
        )
        == "[axs[0, 1]]"
    )
    assert (
        figurecomposer_code._axes_code(
            tool._document,
            FigureAxesSelectionState(axes=all_axes),
            for_plot_slices=True,
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
            tool._document, expression_selection, for_plot_slices=False
        )
        == "custom_axes"
    )
    assert figurecomposer_code._axes_sequence_code(
        tool._document, expression_selection
    ) == ("custom_axes")
    with pytest.raises(ValueError, match="outside the current layout"):
        figurecomposer_code._axes_code(
            tool._document,
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
        figurecomposer_code._setup_code(invalid_tool._document)

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
    setup_code = "\n".join(
        figurecomposer_code._gridspec_setup_code_lines(grid_tool._document)
    )
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

    tool.operation_editor.select_section("axes")
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
    assert tool._document.axes_selection_has_invalid_target(empty_selection)
    with pytest.raises(ValueError, match="No axes are selected"):
        figurecomposer_code._axes_code(
            tool._document, empty_selection, for_plot_slices=False
        )


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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("axes")
    assert (
        tool._axes_target_text(FigureAxesSelectionState(axes=((0, 1), (0, 2), (0, 3))))
        == "axs[0, 1:4]"
    )

    tool._target_current_operation_all_axes()
    assert tool.tool_status.operations[0].axes.axes == tuple(np.ndindex(2, 4))


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

    tool.operation_editor.select_section("axes")
    assert tool.axes_selector.isHidden()
    assert not tool.gridspec_axes_selector.isHidden()
    assert tool.axes_expression_edit.isHidden()
    assert tool.gridspec_axes_selector.axes_ids() == ("left-axis", "right-axis")
    assert not tool.gridspec_axes_selector.axis_rect("left-axis").isNull()
    assert not tool.gridspec_axes_selector.axis_rect("right-axis").isNull()
    assert (
        figurecomposer_code._axes_code(
            tool._document,
            FigureAxesSelectionState(axes_ids=("left-axis",)),
            for_plot_slices=False,
        )
        == "left_panel"
    )
    assert (
        figurecomposer_code._axes_code(
            tool._document,
            FigureAxesSelectionState(axes_ids=("left-axis",)),
            for_plot_slices=True,
        )
        == "[left_panel]"
    )
    assert (
        figurecomposer_code._axes_sequence_code(
            tool._document,
            FigureAxesSelectionState(axes_ids=("left-axis", "right-axis")),
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
    assert tool._document.axes_selection_has_invalid_target(empty_selection)
    with pytest.raises(ValueError, match="No axes are selected"):
        figurecomposer_code._axes_code(
            tool._document, empty_selection, for_plot_slices=False
        )
    invalid_selection = FigureAxesSelectionState(
        axes_ids=("left-axis", "removed-internal-axis")
    )
    with pytest.raises(
        ValueError, match="1 selected GridSpec axis outside the current layout"
    ) as excinfo:
        figurecomposer_code._axes_code(
            tool._document, invalid_selection, for_plot_slices=False
        )
    assert "removed-internal-axis" not in str(excinfo.value)
    with pytest.raises(
        ValueError, match="1 selected GridSpec axis outside the current layout"
    ) as excinfo:
        figurecomposer_code._axes_sequence_code(tool._document, invalid_selection)
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
    tool.operation_editor.select_section("axes")
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

    tool.layout_panel.layout_mode_combo.setCurrentText("subplots")
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
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.layout_panel.nrows_spin.setValue(2)
    tool.layout_panel.ncols_spin.setValue(2)
    widget = tool.layout_panel.gridspec_layout_widget
    widget.resize(widget.sizeHint())

    tool.layout_panel.gridspec_region_kind_combo.setCurrentIndex(
        tool.layout_panel.gridspec_region_kind_combo.findData("grid")
    )
    qtbot.mousePress(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    tool.layout_panel.gridspec_region_kind_combo.setCurrentIndex(
        tool.layout_panel.gridspec_region_kind_combo.findData("axes")
    )
    qtbot.mouseRelease(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=widget.cell_rect((0, 1)).center(),
    )
    child_grid = tool.tool_status.setup.gridspec.root.child_grids[0]
    assert child_grid.label == ""
    assert tool.layout_panel.gridspec_region_label_edit.isHidden()
    assert tool.layout_panel.gridspec_region_name_label.isHidden()
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
    assert tool.layout_panel.ncols_spin.value() == child_grid.ncols
    assert tool.layout_panel.gridspec_parent_grid_button.isEnabled()

    tool.layout_panel.gridspec_region_kind_combo.setCurrentIndex(
        tool.layout_panel.gridspec_region_kind_combo.findData("axes")
    )
    child_widget = tool.layout_panel.gridspec_layout_widget
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
    assert not tool.layout_panel.gridspec_region_label_edit.isHidden()
    assert not tool.layout_panel.gridspec_region_name_label.isHidden()
    tool.layout_panel.gridspec_parent_grid_button.click()
    assert (
        tool.layout_panel.ncols_spin.value()
        == tool.tool_status.setup.gridspec.root.ncols
    )
    assert not tool.layout_panel.gridspec_parent_grid_button.isEnabled()
    widget.resize(widget.sizeHint())
    child_axis_id = active_grid.axes[0].axes_id
    assert child_axis_id in widget.axes_ids()
    assert (
        widget.axis_rect(child_axis_id).center().x() > widget.cell_rect((0, 0)).right()
    )


def test_figure_composer_layout_reserved_names_follow_source_structure(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool.layout_panel.gridspec_layout_widget.set_selected_region(axis.axes_id)
    tool.layout_panel.gridspec_layout_widget.sigRegionSelected.emit(
        axis.axes_id, "axes"
    )

    def name_is_rejected(name: str) -> bool:
        tool.layout_panel.gridspec_region_label_edit.setText(name)
        tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
        return tool.layout_panel.gridspec_region_label_edit.property("invalid") is True

    initial_axis_name = figurecomposer_gridspec._gridspec_axis_code_names(
        tool.tool_status.setup, reserved_names=tool._document.source_names()
    )[axis.axes_id]
    extra = data.rename(initial_axis_name)
    result = tool.add_sources(
        (FigureSourceState(name=initial_axis_name),),
        {initial_axis_name: extra},
    )
    assert result.added
    assert name_is_rejected(initial_axis_name)

    expected_axis_names = figurecomposer_gridspec._gridspec_axis_code_names(
        tool.tool_status.setup, reserved_names=tool._document.source_names()
    )
    assert tool.layout_panel.gridspec_layout_widget._labels == expected_axis_names
    assert tool.gridspec_axes_selector._labels == expected_axis_names
    operation_item = tool.operation_panel.operation_list.topLevelItem(0)
    assert operation_item is not None
    assert (
        operation_item.data(
            figurecomposer_operation_panel._OPERATION_LIST_TARGET_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
        )
        == expected_axis_names[axis.axes_id]
    )

    tool.source_panel.rename_requested.emit(initial_axis_name, "renamed")
    assert name_is_rejected("renamed")

    existing_names = {source.name for source in tool.tool_status.sources}
    tool.source_panel.duplicate_requested.emit(("renamed",))
    duplicate_name = next(
        source.name
        for source in tool.tool_status.sources
        if source.name not in existing_names
    )
    assert name_is_rejected(duplicate_name)

    assert tool.remove_source(duplicate_name)
    tool.layout_panel.gridspec_region_label_edit.setText(duplicate_name)
    tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
    assert tool.layout_panel.gridspec_region_label_edit.property("invalid") is False
    assert tool.tool_status.setup.gridspec.root.axes[0].label == duplicate_name


def test_figure_composer_public_source_data_updates_layout_reserved_names(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    setup = figurecomposer_gridspec._gridspec_setup_from_subplots(FigureSubplotsState())
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(setup=setup, sources=(), operations=()),
        source_data={"initial": data},
    )
    qtbot.addWidget(tool)
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool.layout_panel.gridspec_layout_widget.set_selected_region(axis.axes_id)
    tool.layout_panel.gridspec_layout_widget.sigRegionSelected.emit(
        axis.axes_id, "axes"
    )

    tool.set_source_data({"live_source": data})
    assert tool.source_panel.source_list.topLevelItemCount() == 1
    assert tool.source_panel.selected_names() == ("live_source",)
    assert tool.source_panel.current_name() == "live_source"
    expected_axis_names = figurecomposer_gridspec._gridspec_axis_code_names(
        tool.tool_status.setup, reserved_names=tool._document.source_names()
    )
    assert tool.layout_panel.gridspec_layout_widget._labels == expected_axis_names
    assert tool.gridspec_axes_selector._labels == expected_axis_names
    tool.layout_panel.gridspec_region_label_edit.setText("live_source")
    tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
    assert tool.layout_panel.gridspec_region_label_edit.property("invalid") is True

    tool.set_source_data({"replacement": data})
    assert tool.source_panel.source_list.topLevelItemCount() == 1
    assert tool.source_panel.selected_names() == ("replacement",)
    assert tool.source_panel.current_name() == "replacement"
    expected_axis_names = figurecomposer_gridspec._gridspec_axis_code_names(
        tool.tool_status.setup, reserved_names=tool._document.source_names()
    )
    assert tool.layout_panel.gridspec_layout_widget._labels == expected_axis_names
    assert tool.gridspec_axes_selector._labels == expected_axis_names
    tool.layout_panel.gridspec_region_label_edit.setText("live_source")
    tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
    assert tool.layout_panel.gridspec_region_label_edit.property("invalid") is False
    assert tool.tool_status.setup.gridspec.root.axes[0].label == "live_source"


def test_figure_composer_gridspec_widget_resizes_selected_region(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.layout_panel.nrows_spin.setValue(2)
    tool.layout_panel.ncols_spin.setValue(3)

    original_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=2,
        col_start=0,
        col_stop=3,
    )
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool.layout_panel.gridspec_layout_widget.sigRegionChanged.emit(
        axis.axes_id, original_span
    )
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    widget = tool.layout_panel.gridspec_layout_widget
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
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.layout_panel.nrows_spin.setValue(2)
    tool.layout_panel.ncols_spin.setValue(3)

    original_span = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    axis = tool.tool_status.setup.gridspec.root.axes[0]
    tool.layout_panel.gridspec_layout_widget.sigRegionChanged.emit(
        axis.axes_id, original_span
    )
    widget = tool.layout_panel.gridspec_layout_widget
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
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.show()
    qtbot.wait_until(lambda: tool.isVisible(), timeout=5000)
    qtbot.wait(1)

    axis = tool.tool_status.setup.gridspec.root.axes[0]
    widget = tool.layout_panel.gridspec_layout_widget
    widget.resize(widget.sizeHint())
    widget.set_selected_region(axis.axes_id)
    assert widget.selected_region_id() == axis.axes_id
    assert widget._region_handles_visible

    outside_global_pos = tool.layout_panel.gridspec_status_label.mapToGlobal(
        QtCore.QPoint(1, 1)
    )
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
    tool.layout_panel.gridspec_region_label_edit.setText("1 renamed")
    tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
    assert tool.layout_panel.gridspec_region_label_edit.property("invalid") is True
    assert tool.tool_status.setup.gridspec.root.axes[0].label == ""
    tool.layout_panel.gridspec_region_label_edit.setText("renamed")
    tool.layout_panel.gridspec_region_label_edit.editingFinished.emit()
    assert tool.layout_panel.gridspec_region_label_edit.property("invalid") is False
    assert tool.tool_status.setup.gridspec.root.axes[0].label == "renamed"


def test_figure_composer_gridspec_occupied_cells_cover_spans(qtbot) -> None:
    widget = axes_widgets._GridSpecViewWidget(mode="edit")
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
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.layout_panel.ncols_spin.setValue(2)
    assert tool.tool_status.setup.gridspec.root.ncols == 2
    widget = tool.layout_panel.gridspec_layout_widget
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

    tool.layout_panel.ncols_spin.setValue(1)
    assert tool.editor_tabs.currentWidget() is tool.layout_panel
    root = tool.tool_status.setup.gridspec.root
    invalid_axes_id = root.axes[1].axes_id
    assert invalid_axes_id not in tool.gridspec_axes_selector.axes_ids()
    assert not figurecomposer_gridspec._gridspec_region_valid(root, root.axes[1].span)


def test_figure_composer_gridspec_row_shrink_ignores_invalid_regions(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    tool.layout_panel.nrows_spin.setValue(2)
    widget = tool.layout_panel.gridspec_layout_widget
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

    tool.layout_panel.nrows_spin.setValue(1)
    root = tool.tool_status.setup.gridspec.root
    assert not figurecomposer_gridspec._gridspec_region_valid(root, root.axes[1].span)
    assert widget.span_rect(removed_span) == QtCore.QRect()
    assert widget._region_at(widget.cell_rect((0, 0)).center()) is not None
    _send_mouse_move(widget, widget.cell_rect((0, 0)).center())


def test_figure_composer_gridspec_axes_targets_survive_region_delete(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.layout_panel.layout_mode_combo.setCurrentText("gridspec")
    first_axes_id = tool.tool_status.setup.gridspec.root.axes[0].axes_id
    tool.operation_editor.select_section("axes")
    tool._sync_axes_selector()
    assert not tool.gridspec_axes_selector.isHidden()
    tool.gridspec_axes_selector.set_selected_axes_ids((first_axes_id,), emit=True)
    assert tool.tool_status.operations[0].axes.axes_ids == (first_axes_id,)

    tool.layout_panel.gridspec_layout_widget.sigRegionSelected.emit(
        first_axes_id, "axes"
    )
    tool.layout_panel.gridspec_delete_region_button.click()
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert tool.tool_status.operations[0].axes.axes_ids == (first_axes_id,)
    target_text = tool._axes_target_text(tool.tool_status.operations[0].axes)
    assert target_text == "1 target axis removed"
    assert first_axes_id not in target_text
    tool._sync_axes_selector()
    assert not tool.target_axes_status_label.isHidden()


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
    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.operation_editor.select_section("axes")
    tool._sync_axes_selector()

    tool.layout_panel.gridspec_layout_widget.sigRegionSelected.emit("ax1", "axes")
    tool.layout_panel.gridspec_delete_region_button.click()
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

    tool.layout_panel.gridspec_layout_widget.sigRegionSelected.emit(
        middle_axis.axes_id, "axes"
    )
    tool.layout_panel.gridspec_delete_region_button.click()

    axes_ids = tuple(axis.axes_id for axis in tool.tool_status.setup.gridspec.root.axes)
    assert axes_ids == (left_axis.axes_id, far_axis.axes_id)
    assert (
        tool.layout_panel.gridspec_layout_widget.selected_region_id()
        == left_axis.axes_id
    )
    assert tool.layout_panel.gridspec_delete_region_button.isEnabled()


def test_figure_composer_step_section_buttons_are_tab_focusable(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)

    buttons = _operation_section_buttons(tool)
    assert len(buttons) > 1
    assert all(
        button.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus for button in buttons
    )
    step_action_buttons = (
        tool.operation_panel.add_step_button,
        tool.operation_panel.copy_button,
        tool.operation_panel.cut_button,
        tool.operation_panel.paste_button,
        tool.operation_panel.delete_button,
        tool.operation_panel.operation_list,
    )

    def next_tab_focus(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        candidate = widget.nextInFocusChain()
        while not candidate.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus:
            candidate = candidate.nextInFocusChain()
        return candidate

    for index, button in enumerate(step_action_buttons[:-1]):
        assert next_tab_focus(button) is step_action_buttons[index + 1]
    for index, button in enumerate(buttons[:-1]):
        assert next_tab_focus(button) is buttons[index + 1]

    tool.operation_editor.select_section(tool.operation_editor.section_keys[-1])
    buttons = _operation_section_buttons(tool)
    for index, button in enumerate(buttons[:-1]):
        assert next_tab_focus(button) is buttons[index + 1]


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
    assert figure_window_ui._noop_toolbar_callback() is None
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
        for item in figure_window_ui._FigureComposerNavigationToolbar.toolitems
        if item[3] not in {"back", "forward"}
    ]
    with monkeypatch.context() as context:
        context.setattr(
            figure_window_ui._FigureComposerNavigationToolbar,
            "toolitems",
            limited_toolitems,
        )
        limited_window = figure_window_ui._FigureComposerDisplayWindow(
            FigureSubplotsState()
        )
        qtbot.addWidget(limited_window)
    assert "back" not in limited_window.toolbar._actions
    assert "forward" not in limited_window.toolbar._actions

    assert figure_window_ui._noop_navigation_callback({object(): (True, False)}) is None
    assert figure_window_ui._noop_colorbar_callback({object(): (0.0, 1.0)}) is None

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
    assert figure_window_ui._axis_limit_pair(axis, "get_xlim") == (0.0, 1.0)
    assert figure_window_ui._axis_limit_pair(object(), "get_xlim") is None
    assert figure_window_ui._axis_limit_pair(RaisingAxis(), "get_xlim") is None
    assert figure_window_ui._axis_view(axis) == ((0.0, 1.0), (0.0, 1.0))
    assert figure_window_ui._axis_view(object()) is None

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

    assert figure_window_ui._is_colorbar_axis(colorbar_axis)
    assert not figure_window_ui._is_colorbar_axis(axis)
    assert figure_window_ui._colorbar_mappable(colorbar_axis) is mappable
    assert figure_window_ui._colorbar_mappable(colorbar_without_mappable) is None
    assert figure_window_ui._mappable_clim(mappable) == (0.0, 1.0)
    assert figure_window_ui._mappable_clim(object()) is None
    assert figure_window_ui._mappable_clim(RaisingMappable()) is None
    assert figure_window_ui._mappable_clim(Mappable((None, 1.0))) is None
    assert figure_window_ui._mappable_clim(Mappable(("bad", 1.0))) is None

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

    monkeypatch.setattr(figure_window_ui.NavigationToolbar, "press_pan", fake_press_pan)
    monkeypatch.setattr(
        figure_window_ui.NavigationToolbar,
        "release_pan",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        figure_window_ui.NavigationToolbar,
        "press_zoom",
        fake_press_zoom,
    )
    monkeypatch.setattr(
        figure_window_ui.NavigationToolbar,
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
    assert tool.tool_status.setup.layout == "none"
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
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
    assert combo.currentIndex() == 0
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

    tool.operation_panel.select_row(1)
    current_id = tool.operation_panel.current_id()
    selected_ids = tool.operation_panel.selected_ids()
    section_keys = tool.operation_editor.section_keys
    updated_index = figurecomposer_toolbar_dialogs._upsert_method_operation(
        tool,
        FigureMethodFamily.AXES,
        "set_title",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        args=("updated",),
    )
    assert updated_index == 0
    assert tool.tool_status.operations[0].method_args == ("updated",)
    assert tool.operation_panel.current_id() == current_id
    assert tool.operation_panel.selected_ids() == selected_ids
    assert tool.operation_editor.section_keys == section_keys

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
    tool.operation_panel.select_row(0)
    current_id = tool.operation_panel.current_id()
    selected_ids = tool.operation_panel.selected_ids()
    section_keys = tool.operation_editor.section_keys
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
    assert tool.operation_panel.current_id() == current_id
    assert tool.operation_panel.selected_ids() == selected_ids
    assert tool.operation_editor.section_keys == section_keys
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
    assert isinstance(selector, axes_widgets._AxesSelectorWidget)
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
    assert isinstance(grid_selector, axes_widgets._GridSpecViewWidget)
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


def test_figure_composer_provenance_build_code_handles_invalid_recipes(
    monkeypatch,
) -> None:
    code_or_exc: Exception | str = RuntimeError("invalid recipe")

    def fake_generated_code(*_args, **_kwargs):
        if isinstance(code_or_exc, Exception):
            raise code_or_exc
        return code_or_exc

    monkeypatch.setattr(
        figurecomposer_provenance.erlab.interactive._figurecomposer._codegen,
        "generated_code",
        fake_generated_code,
    )
    fake_tool = typing.cast("FigureComposerTool", object())

    assert figurecomposer_provenance._figure_build_code(fake_tool) is None
    code_or_exc = "if"
    assert figurecomposer_provenance._figure_build_code(fake_tool) is None
    code_or_exc = "axs = object()"
    assert figurecomposer_provenance._figure_build_code(fake_tool) is None

    code_or_exc = "fig = object()"
    operation = figurecomposer_provenance._figure_build_operation(fake_tool)
    assert operation.copyable
    assert operation.code == "fig = object()"


def test_figure_composer_toolbar_axes_plain_text_handles_missing_document(
    qtbot, monkeypatch
) -> None:
    class _DocumentlessPlainTextEdit(QtWidgets.QPlainTextEdit):
        def document(self):
            return None

    edit = _DocumentlessPlainTextEdit()
    qtbot.addWidget(edit)
    figurecomposer_toolbar_dialogs._apply_axis_plain_text_edit_state(
        edit,
        figurecomposer_toolbar_dialogs._AxisValueState(value="Title", available=True),
    )
    assert edit.toPlainText() == "Title"

    tool = FigureComposerTool(_figure_composer_image_source("data"))
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    title_edit = dialog.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesTitleEdit"
    )
    assert title_edit is not None
    monkeypatch.setattr(title_edit, "document", lambda: None)

    title_edit.textChanged.emit()

    assert _method_operations(tool, FigureMethodFamily.AXES, "set_title") == ()


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
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesTitleEdit"
    )
    xlim_edit = dialog.findChild(
        QtWidgets.QLineEdit, "figureComposerToolbarAxesXLimEdit"
    )
    xlabel_edit = dialog.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesXLabelEdit"
    )
    ylabel_edit = dialog.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesYLabelEdit"
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
        labels_row.findChild(QtWidgets.QPlainTextEdit, xlabel_edit.objectName())
        is not None
    )
    assert (
        labels_row.findChild(QtWidgets.QPlainTextEdit, ylabel_edit.objectName())
        is not None
    )
    assert limits_row.findChild(QtWidgets.QLineEdit, xlim_edit.objectName()) is not None
    assert (
        scales_row.findChild(QtWidgets.QComboBox, xscale_combo.objectName()) is not None
    )
    assert grid_row.findChild(QtWidgets.QCheckBox, grid_check.objectName()) is not None
    grid_row_layout = grid_row.layout()
    assert grid_row_layout is not None
    assert grid_row_layout.indexOf(grid_check) >= 0
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

    title_edit.setPlainText("Peak")
    title_ops = _method_operations(tool, FigureMethodFamily.AXES, "set_title")
    assert len(title_ops) == 1
    assert title_ops[0].axes.axes == ((0, 0),)
    assert title_ops[0].method_args == ("Peak",)

    xlabel_edit.setPlainText("Energy\n(eV)")
    ylabel_edit.setPlainText("Intensity")
    assert _method_operations(tool, FigureMethodFamily.AXES, "set_xlabel")[
        0
    ].method_args == ("Energy\n(eV)",)
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

    selector = dialog.findChild(axes_widgets._AxesSelectorWidget)
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
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesTitleEdit"
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
    assert title_edit.toPlainText() == ""
    assert title_edit.placeholderText() == _editor_controls.MIXED_VALUES_TEXT
    assert xlim_edit.text() == ""
    assert xlim_edit.placeholderText() == _editor_controls.MIXED_VALUES_TEXT
    assert xscale_combo.currentData() is _editor_controls.MIXED_VALUE
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

    title_edit.setPlainText("Shared")

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
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesTitleEdit"
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
    selector = dialog.findChild(axes_widgets._AxesSelectorWidget)
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
    selector = dialog.findChild(axes_widgets._AxesSelectorWidget)
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
    selector = dialog.findChild(axes_widgets._GridSpecViewWidget)
    title_edit = dialog.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerToolbarAxesTitleEdit"
    )
    assert selector is not None
    assert selector.selected_axes_ids() == ("main",)
    assert title_edit is not None

    title_edit.setPlainText("GridSpec title")

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

    assert tool.layout_panel.width_ratios_edit.text() == "1, 2, 3"
    assert tool.layout_panel.height_ratios_edit.text() == "2, 1"

    tool.layout_panel.width_ratios_edit.setText("3, 2, 1")
    tool.layout_panel.height_ratios_edit.setText("4, 1")
    tool.layout_panel.height_ratios_edit.editingFinished.emit()

    assert tool.tool_status.setup.width_ratios == (3.0, 2.0, 1.0)
    assert tool.tool_status.setup.height_ratios == (4.0, 1.0)
    setup_kwargs = figurecomposer_rendering._setup_kwargs(tool._document)
    assert setup_kwargs["width_ratios"] == (3.0, 2.0, 1.0)
    assert setup_kwargs["height_ratios"] == (4.0, 1.0)

    code = tool.generated_code()
    assert "width_ratios" in code
    assert "height_ratios" in code
    assert "gridspec_kw" not in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["axs"].shape == (2, 3)


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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    values_edit = tool.operation_editor.stack.currentWidget().findChild(
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
    values_edit = tool.operation_editor.stack.currentWidget().findChild(
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

    dimension_combo = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    assert dimension_combo is not None
    rebuild_calls.clear()
    tool.operation_editor.eventFilter(
        dimension_combo,
        QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
    )
    assert tool.operation_editor.rebuild_must_wait()
    tool.operation_editor.request_update_rebuild(slice_values=(0.0, 1.0))

    assert tool._operation_editor_update_pending is True
    qtbot.wait(100)
    assert rebuild_calls == []
    qtbot.waitUntil(lambda: rebuild_calls == [None], timeout=1000)
    assert tool._operation_editor_update_pending is False

    dimension_combo = tool.operation_editor.stack.currentWidget().findChild(
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    old_page = tool.operation_editor.stack.currentWidget()
    active_popup: list[QtWidgets.QWidget | None] = [None]
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "activePopupWidget",
        staticmethod(lambda: active_popup[0]),
    )
    monkeypatch.setattr(tool.operation_editor, "_queue_retired_drain", lambda: None)

    tool._update_operation_editor()
    assert old_page in tool.operation_editor._retired_widgets
    assert erlab.interactive.utils.qt_is_valid(old_page)
    active_popup[0] = QtWidgets.QMenu(tool)

    tool.operation_editor._drain_retired_widgets()
    assert old_page in tool.operation_editor._retired_widgets
    assert erlab.interactive.utils.qt_is_valid(old_page)

    active_popup[0] = None
    tool.operation_editor._drain_retired_widgets()
    assert old_page not in tool.operation_editor._retired_widgets
    QtCore.QCoreApplication.sendPostedEvents(
        old_page,
        QtCore.QEvent.Type.DeferredDelete,
    )
    assert not erlab.interactive.utils.qt_is_valid(old_page)


def test_figure_composer_retired_editor_control_signal_is_ignored(
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
    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    old_page = tool.operation_editor.stack.currentWidget()
    old_values_edit = old_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    assert old_values_edit is not None
    tool.operation_editor.request_update_rebuild(slice_values=(0.0, 1.0))
    qtbot.waitUntil(
        lambda: not tool.operation_editor.control_signal_allowed(old_values_edit),
        timeout=1000,
    )
    assert old_page.isHidden()
    assert erlab.interactive.utils.qt_is_valid(old_values_edit)

    recipe_before_stale_signal = tool.tool_status
    history_before_stale_signal = (
        tuple(tool._prev_states),
        tuple(tool._next_states),
    )
    operation_before_stale_signal = tool.tool_status.operations[0]
    input_error_before_stale_signal = tool.operation_editor.input_error_text(
        operation_before_stale_signal
    )
    render_errors_before_stale_signal = dict(tool._operation_render_errors)
    status_before_stale_signal = _operation_status_codes(tool, 0)
    render_calls.clear()

    # Exercise the generation guard even though retirement also blocks signals.
    old_values_edit.blockSignals(False)
    old_values_edit.setText("1")
    old_values_edit.editingFinished.emit()
    qtbot.wait(350)

    assert tool.tool_status == recipe_before_stale_signal
    assert (tuple(tool._prev_states), tuple(tool._next_states)) == (
        history_before_stale_signal
    )
    assert (
        tool.operation_editor.input_error_text(tool.tool_status.operations[0])
        == input_error_before_stale_signal
    )
    assert tool._operation_render_errors == render_errors_before_stale_signal
    assert _operation_status_codes(tool, 0) == status_before_stale_signal
    assert render_calls == []


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
    viewport = tool.operation_panel._operation_viewport
    assert viewport is not None
    assert erlab.interactive.utils.qt_is_valid(viewport)

    tool.operation_panel._multi_select_event = True
    tool.operation_panel._selection_input_event = True
    erlab.interactive.utils.single_shot(
        tool.operation_panel, 0, tool.operation_panel._clear_selection_input_state
    )
    tool.close()
    qtbot.wait(10)

    assert tool.operation_panel._operation_viewport is None
    assert tool.operation_panel._multi_select_event is False
    assert tool.operation_panel._selection_input_event is False
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

    tool.editor_tabs.setCurrentWidget(tool.layout_panel)
    tool.layout_panel.nrows_spin.setValue(1)

    assert tool.editor_tabs.currentWidget() is tool.layout_panel
    assert tool.tool_status.setup.nrows == 1
    assert tool.tool_status.operations[0].axes.axes == ((1, 1),)
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert (
        tool.operation_editor.stack.currentWidget() is tool.operation_editor.source_page
    )
    tool.operation_editor.select_section("axes")
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

    tool.layout_panel.nrows_spin.setValue(1)
    tool.editor_tabs.setCurrentWidget(tool.operation_panel)
    tool.operation_editor.select_section("axes")
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
