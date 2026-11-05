import gc
import typing
import weakref

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._completion as completion_widgets
import erlab.interactive._figurecomposer._code as figurecomposer_code
import erlab.interactive._figurecomposer._line_style as figurecomposer_line_style
import erlab.interactive._figurecomposer._line_transform as _line_transform
import erlab.interactive._figurecomposer._model._axes as figurecomposer_axes
import erlab.interactive._figurecomposer._model._gridspec as figurecomposer_gridspec
import erlab.interactive._figurecomposer._norms as figurecomposer_norms
import erlab.interactive._figurecomposer._rendered_colors as _rendered_colors
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._ui._color_widgets as color_widgets
import erlab.interactive._figurecomposer._ui._editor_controls as _editor_controls
import erlab.interactive._figurecomposer._ui._line_style as figurecomposer_line_style_ui
import erlab.interactive._stylesheets
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureExportState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection

from ._common import (
    _assert_step_editor_section,
    _figure_composer_image_source,
    _figure_composer_line_slice_source,
    _operation_section_button,
)


def test_figure_composer_color_completion_commits_named_and_custom_colors(qtbot):
    color_edit = color_widgets._ColorLineEditWidget("red")
    qtbot.addWidget(color_edit)
    edit = color_edit.line_edit
    assert isinstance(edit, completion_widgets.CompletingLineEdit)
    for color in ("r", "tab:orange", "rebeccapurple", "C9", "none"):
        assert color in edit.completion_model.stringList()

    committed = []
    color_edit.editingFinished.connect(lambda: committed.append(color_edit.text()))
    edit.completion_completer.activated[str].emit("tab:orange")
    assert color_edit.text() == "tab:orange"
    assert color_edit.isModified()
    assert committed == ["tab:orange"]
    assert (
        color_edit.color_button.color().getRgb()
        == QtGui.QColor.fromRgbF(*mcolors.to_rgba("tab:orange")).getRgb()
    )

    for text, rgba in (
        ("#01020304", (1, 2, 3, 4)),
        ("(1, 0, 0, 0.5)", (255, 0, 0, 128)),
        ("0.5", (128, 128, 128, 255)),
    ):
        edit.selectAll()
        qtbot.keyClicks(edit, text)
        edit.editingFinished.emit()
        assert committed[-1] == text
        assert color_edit.color_button.color().getRgb() == rgba


@pytest.mark.parametrize("accept_with_mouse", [False, True])
def test_figure_composer_color_completion_accepts_popup(qtbot, accept_with_mouse):
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    layout = QtWidgets.QVBoxLayout(window)
    color_edit = color_widgets._ColorLineEditWidget()
    other_edit = QtWidgets.QLineEdit(window)
    layout.addWidget(color_edit)
    layout.addWidget(other_edit)
    committed = []
    color_edit.editingFinished.connect(
        lambda: committed.append(
            (
                color_edit.text(),
                color_edit.isModified(),
                color_edit.color_button.color().getRgb(),
            )
        )
    )
    window.show()
    window.activateWindow()
    edit = color_edit.line_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "tab:or")
    popup = edit.completer_popup
    qtbot.waitUntil(popup.isVisible)

    if accept_with_mouse:
        index = edit.completion_completer.completionModel().index(0, 0)
        qtbot.mouseClick(
            popup.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=popup.visualRect(index).center(),
        )
    else:
        qtbot.keyClick(popup, QtCore.Qt.Key.Key_Return)

    assert color_edit.text() == "tab:orange"
    assert committed == [
        (
            "tab:orange",
            True,
            QtGui.QColor.fromRgbF(*mcolors.to_rgba("tab:orange")).getRgb(),
        )
    ]
    assert not popup.isVisible()


def test_figure_composer_color_list_row_completion_preserves_other_colors(qtbot):
    color_list = color_widgets._ColorListEditorWidget(("red", "blue"))
    qtbot.addWidget(color_list)
    first, second = color_list._row_editors()
    committed = []
    color_list.colorsChanged.connect(committed.append)

    first.line_edit.completion_completer.activated[str].emit("rebeccapurple")

    assert committed == [("rebeccapurple", "blue")]
    assert color_list.colors() == ("rebeccapurple", "blue")
    assert color_list.main_edit.isModified()
    assert second.text() == "blue"
    assert color_list.main_edit.completer() is None


def test_figure_composer_color_widget_separates_default_and_auto(qtbot) -> None:
    colors = {"default": "red", "auto": "blue"}
    widget = color_widgets._ColorLineEditWidget(
        "auto",
        inherited_color=colors["default"],
        auto_color=colors["auto"],
        inherited_color_getter=lambda: colors["default"],
        auto_color_getter=lambda: colors["auto"],
    )
    qtbot.addWidget(widget)
    widget.show()
    assert widget.color_button.color() == QtGui.QColor("blue")
    widget.setText("")
    assert widget.color_button.color() == QtGui.QColor("red")
    widget.setText("auto")
    colors.update(default="purple", auto="green")
    widget.refreshInheritedColor()
    assert widget.color_button.color() == QtGui.QColor("green")
    widget.setText("")
    widget.refreshInheritedColor()
    assert widget.color_button.color() == QtGui.QColor("purple")


def test_figure_composer_color_widget_resolves_only_visible_defaults(qtbot) -> None:
    colors = {"default": "red", "auto": "blue"}
    reads: list[str] = []

    def read_color(key: str) -> str:
        reads.append(key)
        return colors[key]

    widget = color_widgets._ColorLineEditWidget(
        "",
        inherited_color="black",
        auto_color="black",
        inherited_color_getter=lambda: read_color("default"),
        auto_color_getter=lambda: read_color("auto"),
    )
    qtbot.addWidget(widget)
    widget.refreshInheritedColor()
    assert reads == []
    widget.show()
    assert reads == ["default"]
    assert widget.color_button.color() == QtGui.QColor("red")

    reads.clear()
    widget.setText("auto")
    assert reads == []
    widget.line_edit.editingFinished.emit()
    assert reads == ["auto"]
    assert widget.color_button.color() == QtGui.QColor("blue")
    reads.clear()
    widget.setText("orange")
    widget.line_edit.editingFinished.emit()
    widget.refreshInheritedColor()
    assert reads == []

    widget.hide()
    colors.update(default="purple", auto="green")
    widget.setText("")
    widget.refreshInheritedColor()
    assert reads == []
    widget.show()
    assert reads == ["default"]
    assert widget.color_button.color() == QtGui.QColor("purple")
    reads.clear()
    widget.hide()
    widget.setText("auto")
    widget.show()
    assert reads == ["auto"]
    assert widget.color_button.color() == QtGui.QColor("green")

    fallback = color_widgets._ColorLineEditWidget(
        "auto", inherited_color_getter=lambda: read_color("default")
    )
    qtbot.addWidget(fallback)
    reads.clear()
    fallback.show()
    assert reads == ["default"]
    assert fallback.color_button.color() == QtGui.QColor("purple")


def test_figure_composer_deferred_form_tracks_visible_current_state(qtbot) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="text", label="original"
    )
    other = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="text", label="other"
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation, other),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    editor = tool.operation_editor
    tool.operation_panel.select_row(0)
    editor.prepare_rebuild()
    idle, _idle_layout = editor.new_form_page("deferredIdlePage")
    unrelated = editor.line_edit("unrelated", parent=idle)
    unrelated.setObjectName("deferredUnrelatedEdit")
    _idle_layout.addRow(unrelated)
    page, _layout = editor.new_form_page("deferredTestPage")
    built: list[tuple[str, str, float, int]] = []

    def build(current, target_page, layout):
        source = editor.context.source_data["data"]
        built.append(
            (
                current.operation_id,
                current.label,
                float(source.values[0]),
                id(editor.context.source_payloads["data"]),
            )
        )
        edit = editor.line_edit(current.label, parent=target_page)
        edit.setObjectName("deferredTestEdit")
        layout.addRow(edit)

    editor.defer_form_page(page, build)
    editor.replace_sections(
        (
            StepSection("idle", "Idle", idle, ""),
            StepSection("deferred", "Deferred", page, ""),
        ),
        summaries={},
    )
    latest = operation.model_copy(update={"label": "latest"})
    tool._document.replace_operations((other, latest))
    refreshed_data = xr.DataArray([3.0, 4.0], dims="x", name="data")
    tool._document.replace_source_payloads({"data": refreshed_data}, {})
    payload_id = id(tool._document.source_payloads["data"])
    assert built == []
    tool.show()
    assert idle.isVisible()
    assert built == []

    editor.select_section("deferred")
    assert built == [(operation.operation_id, "latest", 3.0, payload_id)]
    first_edit = page.findChild(QtWidgets.QLineEdit, "deferredTestEdit")
    assert first_edit is not None
    first_edit.setText("invalid raw input")
    editor.set_input_errors(
        {
            operation.operation_id: {
                "deferredTestEdit": "invalid field",
                "deferredUnrelatedEdit": "unrelated error",
            }
        }
    )
    editor.select_section("idle")
    editor.select_section("deferred")
    assert len(built) == 1
    assert page.findChild(QtWidgets.QLineEdit, "deferredTestEdit") is first_edit
    assert first_edit.text() == "invalid raw input"
    assert editor._input_errors[operation.operation_id] == {
        "deferredTestEdit": "invalid field",
        "deferredUnrelatedEdit": "unrelated error",
    }

    tool.hide()
    latest = latest.model_copy(update={"label": "changed"})
    tool._document.replace_operations((other, latest))
    assert len(built) == 1
    tool.show()
    assert built[-1] == (operation.operation_id, "changed", 3.0, payload_id)
    assert len(built) == 2
    new_edit = page.findChild(QtWidgets.QLineEdit, "deferredTestEdit")
    assert new_edit is not first_edit
    assert new_edit is not None
    assert new_edit.text() == "changed"
    assert editor._input_errors[operation.operation_id] == {
        "deferredUnrelatedEdit": "unrelated error"
    }

    editor.select_section("idle")
    refreshed_data.values[0] = 5.0
    tool._document.touch_source_payloads()
    assert id(tool._document.source_payloads["data"]) == payload_id
    assert len(built) == 2
    editor.select_section("deferred")
    assert built[-1] == (operation.operation_id, "changed", 5.0, payload_id)
    assert len(built) == 3

    editor.select_section("idle")
    replacement = xr.DataArray([7.0, 8.0], dims="x", name="data")
    tool._document.replace_source_payloads({"data": replacement}, {})
    editor.select_section("deferred")
    assert built[-1] == (
        operation.operation_id,
        "changed",
        7.0,
        id(tool._document.source_payloads["data"]),
    )
    assert len(built) == 4

    editor.select_section("idle")
    tool._document.replace_operations((other,))
    editor.select_section("deferred")
    assert len(built) == 4
    editor.prepare_rebuild()
    tool._document.replace_operations((other, latest))
    page.show()
    assert len(built) == 4


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
    figurecomposer_line_style_ui.configure_style_combo(
        combo,
        figurecomposer_line_style.LINE_STYLE_OPTIONS,
        None,
    )
    assert combo.itemData(0) is None
    assert figurecomposer_line_style_ui.style_combo_value(combo) is None
    figurecomposer_line_style_ui.set_style_combo_value(combo, "")
    assert figurecomposer_line_style_ui.style_combo_value(combo) == "none"
    figurecomposer_line_style_ui.set_style_combo_value(combo, " ")
    assert figurecomposer_line_style_ui.style_combo_value(combo) == "none"
    figurecomposer_line_style_ui.set_style_combo_value(combo, "None")
    assert figurecomposer_line_style_ui.style_combo_value(combo) == "none"
    figurecomposer_line_style_ui.set_style_combo_value(combo, "custom-dash")
    assert figurecomposer_line_style_ui.style_combo_value(combo) == "custom-dash"
    assert combo.itemText(combo.count() - 1) == "custom-dash"

    spinbox = figurecomposer_line_style_ui.optional_positive_spinbox(
        None, parent=parent
    )
    assert (
        figurecomposer_line_style_ui.optional_positive_spinbox_value(spinbox.value())
        is None
    )
    spinbox.stepUp()
    assert spinbox.value() == spinbox.singleStep()
    spinbox.stepDown()
    assert spinbox.value() == 0.0
    spinbox.stepDown()
    assert spinbox.value() == spinbox.minimum()
    spinbox.setValue(0.0)
    assert spinbox.value() > spinbox.minimum()
    assert (
        figurecomposer_line_style_ui.optional_positive_spinbox_value(spinbox.value())
        == 0.0
    )
    spinbox.setValue(2.5)
    assert (
        figurecomposer_line_style_ui.optional_positive_spinbox_value(spinbox.value())
        == 2.5
    )

    operation = FigureOperationState.plot_slices(
        label="line-slices",
        sources=("data",),
    ).model_copy(
        update={
            "line_kw": {"lw": "2", "marker": "o", "custom": 1},
            "cmap": "magma",
        }
    )
    assert figurecomposer_line_style.line_kw_text(operation, "linewidth", "lw") == "2"
    assert figurecomposer_line_style.line_kw_float(operation, "linewidth", "lw") == 2.0
    assert figurecomposer_line_style.extra_line_kw(operation) == {"custom": 1}
    assert figurecomposer_line_style.extra_line_kw(
        operation,
        controlled_keys=figurecomposer_line_style.STROKE_LINE_KW_KEYS,
        reserved_keys=("custom",),
    ) == {"marker": "o"}
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    tool._updating_controls = True
    figurecomposer_line_style_ui.update_current_line_kw(
        tool.operation_editor, "color", "red", clear_stale_cmap=True
    )
    tool._updating_controls = False
    assert tool.tool_status.operations[0].cmap == "magma"

    figurecomposer_line_style_ui.update_current_line_kw(
        tool.operation_editor,
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
    figurecomposer_line_style_ui.update_current_extra_line_kw(
        tool.operation_editor, {"zorder": 5}
    )
    tool._updating_controls = False
    assert "zorder" not in tool.tool_status.operations[0].line_kw

    figurecomposer_line_style_ui.update_current_extra_line_kw(
        tool.operation_editor, {"zorder": 5}
    )
    assert tool.tool_status.operations[0].line_kw == {
        "lw": "2",
        "color": "red",
        "marker": "o",
        "zorder": 5,
    }
    with pytest.raises(ValueError, match=r"dedicated controls.*angle"):
        figurecomposer_line_style_ui.update_current_extra_line_kw(
            tool.operation_editor,
            {"angle": 30.0},
            controlled_keys=figurecomposer_line_style.STROKE_LINE_KW_KEYS,
            reserved_keys=("angle",),
        )
    with pytest.raises(ValueError, match=r"dedicated controls.*linewidth"):
        figurecomposer_line_style_ui.update_current_extra_line_kw(
            tool.operation_editor,
            {"linewidth": 3.0},
        )


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
            FigureOperationState.plot_array(
                label="image plot",
                source="image",
            ).model_copy(update={"colorbar": "right"}),
        ),
        primary_source="image",
    )
    tool = FigureComposerTool(
        image,
        recipe=recipe,
        source_data={"image": image, "line_map": line_map},
    )
    qtbot.addWidget(tool)
    tool.show()

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("selection")
    line_selection_page = tool.operation_editor.current_page
    assert line_selection_page is not None
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionDataSection"
    )
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionProfilesSection"
    )
    tool.operation_editor.select_section("view")
    line_view_page = tool.operation_editor.current_page
    assert line_view_page is not None
    assert (
        line_view_page.findChild(
            QtWidgets.QWidget, "figureComposerLineViewPlacementSection"
        )
        is None
    )
    tool.operation_editor.select_section("style")
    line_style_page = tool.operation_editor.current_page
    assert line_style_page is not None
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLegendSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleColorSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLineSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleFillSection")
    tool.operation_editor.select_section("other")
    line_other_page = tool.operation_editor.current_page
    assert line_other_page is not None
    assert (
        _operation_section_button(tool, "other").property("section_title")
        == "Transform"
    )
    assert (
        line_other_page.findChild(
            QtWidgets.QWidget, "figureComposerLineOtherTransformSection"
        )
        is None
    )

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("selection")
    line_slices_selection_page = tool.operation_editor.current_page
    assert line_slices_selection_page is not None
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionDimensionsSection",
    )
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionValuesSection",
    )
    tool.operation_editor.select_section("view")
    line_slices_view_page = tool.operation_editor.current_page
    assert line_slices_view_page is not None
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewPanelsSection"
    )
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewAxesSection"
    )
    tool.operation_editor.select_section("colors")
    line_slices_colors_page = tool.operation_editor.current_page
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
    tool.operation_editor.select_section("transform")
    line_slices_transform_page = tool.operation_editor.current_page
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(2)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("colors")
    image_slices_colors_page = tool.operation_editor.current_page
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(3)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.current_page
    assert method_page is not None
    _assert_step_editor_section(method_page, "figureComposerMethodCallSection")
    _assert_step_editor_section(method_page, "figureComposerMethodValuesSection")
    _assert_step_editor_section(method_page, "figureComposerMethodAdvancedSection")

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(4)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("view")
    plot_array_view_page = tool.operation_editor.current_page
    assert plot_array_view_page is not None
    _assert_step_editor_section(
        plot_array_view_page, "figureComposerPlotArrayViewImageSection"
    )
    _assert_step_editor_section(
        plot_array_view_page, "figureComposerPlotArrayViewAxesSection"
    )
    tool.operation_editor.select_section("colors")
    plot_array_colors_page = tool.operation_editor.current_page
    assert plot_array_colors_page is not None
    _assert_step_editor_section(
        plot_array_colors_page,
        "figureComposerPlotArrayColorsImageColorSection",
    )
    _assert_step_editor_section(
        plot_array_colors_page,
        "figureComposerPlotArrayColorsColorbarSection",
    )


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


def test_figure_composer_cmap_helpers_normalize_colorcet_names() -> None:
    pytest.importorskip("colorcet")

    erlab.interactive.colors.load_all_colormaps()

    assert figurecomposer_norms._cmap_base_and_reverse("CET_C1_r") == (
        "cet_CET_C1",
        True,
    )
    assert figurecomposer_norms._cmap_with_reverse("fire", True) == "cet_fire_r"


def test_figure_composer_axes_helpers_parse_safe_expressions() -> None:
    axs = np.empty((2, 2), dtype=object)
    axs[0, 0] = "ax00"
    axs[0, 1] = "ax01"
    axs[1, 0] = "ax10"
    axs[1, 1] = "ax11"

    assert figurecomposer_axes._compact_axes_code(()) is None
    assert figurecomposer_axes._compact_axes_iterable_code((), nrows=2, ncols=2) is None
    assert figurecomposer_axes._compact_axes_index(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        nrows=2,
        ncols=2,
    ) == (slice(None), slice(None))
    assert figurecomposer_axes._compact_axes_index(
        ((0, 0), (0, 1)),
        nrows=2,
        ncols=3,
    ) == (0, slice(None, 2))
    assert (
        figurecomposer_axes._compact_axes_index(
            ((0, 0), (1, 1)),
            nrows=2,
            ncols=2,
        )
        is None
    )
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
            tool._document, FigureAxesSelectionState(axes=((1, 0),))
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(
            tool._document, FigureAxesSelectionState(axes=())
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
            grid_tool._document,
            FigureAxesSelectionState(axes_ids=("axis-a", "axis-b")),
            for_plot_slices=False,
        )
        == "[ax0, ax1]"
    )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(
            grid_tool._document, FigureAxesSelectionState(axes_ids=())
        )


def test_figure_composer_gridspec_shared_axes_render_and_generate_code(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    top = FigureGridSpecAxesState(
        axes_id="top-id",
        label="top",
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=0, col_stop=1),
    )
    lower_left = FigureGridSpecAxesState(
        axes_id="lower-left-id",
        label="lower_left",
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=0, col_stop=1),
    )
    lower_right = FigureGridSpecAxesState(
        axes_id="lower-right-id",
        label="lower_right",
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=1, col_stop=2),
    )
    child = FigureGridSpecGridState(
        grid_id="child",
        nrows=1,
        ncols=2,
        span=FigureGridSpecSpanState(row_start=1, row_stop=2, col_start=0, col_stop=1),
        axes=(lower_left, lower_right),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                nrows=2,
                ncols=1,
                axes=(top,),
                child_grids=(child,),
            ),
            shared_x_axes=(("top-id", "lower-left-id", "lower-right-id"),),
            shared_y_axes=(("top-id", "lower-right-id"),),
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=setup,
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = plt.figure()
    rendered = figurecomposer_rendering._make_axes(tool, figure, sync_visible=False)
    assert isinstance(rendered, dict)
    shared_x = rendered["top-id"].get_shared_x_axes()
    shared_y = rendered["top-id"].get_shared_y_axes()
    assert shared_x.joined(rendered["top-id"], rendered["lower-left-id"])
    assert shared_x.joined(rendered["top-id"], rendered["lower-right-id"])
    assert not shared_y.joined(rendered["top-id"], rendered["lower-left-id"])
    assert shared_y.joined(rendered["top-id"], rendered["lower-right-id"])
    plt.close(figure)

    namespace = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    generated_top = namespace["top"]
    generated_left = namespace["lower_left"]
    generated_right = namespace["lower_right"]
    generated_shared_x = generated_top.get_shared_x_axes()
    generated_shared_y = generated_top.get_shared_y_axes()
    assert generated_shared_x.joined(generated_top, generated_left)
    assert generated_shared_x.joined(generated_top, generated_right)
    assert not generated_shared_y.joined(generated_top, generated_left)
    assert generated_shared_y.joined(generated_top, generated_right)
    plt.close(namespace["fig"])


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
    with pytest.raises(ValueError, match="at least two axes"):
        FigureGridSpecLayoutState(shared_x_axes=(("axis-a",),))
    with pytest.raises(ValueError, match="duplicate axes"):
        FigureGridSpecLayoutState(shared_x_axes=(("axis-a", "axis-a"),))
    with pytest.raises(ValueError, match="multiple shared-axis groups"):
        FigureGridSpecLayoutState(
            shared_x_axes=(("axis-a", "axis-b"), ("axis-b", "axis-c"))
        )
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
            setup=FigureSubplotsState(nrows=2, ncols=2, sharex=False, sharey=False),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    assert "sharex" not in figurecomposer_rendering._setup_kwargs(tool._document)
    assert "sharey" not in figurecomposer_rendering._setup_kwargs(tool._document)

    fig, axs = plt.subplots(2, 2, squeeze=False)
    rectangular = figurecomposer_rendering._axes_from_selection(
        tool,
        FigureAxesSelectionState(axes=((0, 0), (0, 1), (1, 0), (1, 1))),
        axs,
        for_plot_slices=False,
    )
    assert isinstance(rectangular, np.ndarray)
    assert rectangular.shape == (2, 2)
    assert rectangular[1, 0] is axs[1, 0]

    plot_slices_axes = figurecomposer_rendering._axes_from_selection(
        tool,
        FigureAxesSelectionState(axes=((0, 0), (0, 1), (1, 0), (1, 1))),
        axs,
        for_plot_slices=True,
    )
    assert isinstance(plot_slices_axes, np.ndarray)
    assert plot_slices_axes.shape == (4,)

    irregular = figurecomposer_rendering._axes_from_selection(
        tool,
        FigureAxesSelectionState(axes=((0, 0), (1, 1))),
        axs,
        for_plot_slices=False,
    )
    assert isinstance(irregular, np.ndarray)
    assert irregular.shape == (2,)

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


def test_gridspec_transformations_validate_edits_and_ignore_stale_ids() -> None:
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(grid_id="root", nrows=2, ncols=2)
        ),
    )
    top_left = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    top_right = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )
    bottom_left = FigureGridSpecSpanState(
        row_start=1,
        row_stop=2,
        col_start=0,
        col_stop=1,
    )

    assert (
        figurecomposer_gridspec._gridspec_update_grid_settings(
            setup, "missing", nrows=1, ncols=1
        )
        == setup
    )
    with pytest.raises(ValueError, match="at least one row"):
        figurecomposer_gridspec._gridspec_update_grid_settings(
            setup, "root", nrows=0, ncols=1
        )

    unchanged, region_id = figurecomposer_gridspec._gridspec_add_region(
        setup, "missing", top_left, "axes"
    )
    assert unchanged == setup
    assert region_id == ""
    with pytest.raises(ValueError, match="unknown GridSpec region kind"):
        figurecomposer_gridspec._gridspec_add_region(
            setup, "root", top_left, typing.cast("typing.Any", "unknown")
        )

    setup, first_axes_id = figurecomposer_gridspec._gridspec_add_region(
        setup, "root", top_left, "axes"
    )
    setup, second_axes_id = figurecomposer_gridspec._gridspec_add_region(
        setup, "root", top_right, "axes"
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        figurecomposer_gridspec._gridspec_add_region(setup, "root", top_left, "grid")
    with pytest.raises(ValueError, match="outside the active grid"):
        figurecomposer_gridspec._gridspec_update_region_span(
            setup,
            "root",
            first_axes_id,
            FigureGridSpecSpanState(
                row_start=2,
                row_stop=3,
                col_start=0,
                col_stop=1,
            ),
        )
    with pytest.raises(ValueError, match="cannot overlap"):
        figurecomposer_gridspec._gridspec_update_region_span(
            setup, "root", first_axes_id, top_right
        )
    assert (
        figurecomposer_gridspec._gridspec_update_region_span(
            setup, "missing", first_axes_id, bottom_left
        )
        == setup
    )
    assert (
        figurecomposer_gridspec._gridspec_update_region_span(
            setup, "root", "missing", bottom_left
        )
        == setup
    )
    assert (
        figurecomposer_gridspec._gridspec_nearest_axes_after_region_delete(
            setup, "root", second_axes_id
        )
        == first_axes_id
    )
    assert (
        figurecomposer_gridspec._gridspec_nearest_axes_after_region_delete(
            setup, "missing", first_axes_id
        )
        == ""
    )
    assert (
        figurecomposer_gridspec._gridspec_nearest_axes_after_region_delete(
            setup, "root", "missing"
        )
        == ""
    )
    assert (
        figurecomposer_gridspec._gridspec_remove_region(setup, "root", "missing")
        == setup
    )


def test_gridspec_shared_axes_groups_update_and_clean_up() -> None:
    axes = tuple(
        FigureGridSpecAxesState(
            axes_id=f"axis-{index}",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=index,
                col_stop=index + 1,
            ),
        )
        for index in range(4)
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(grid_id="root", nrows=1, ncols=4, axes=axes),
            shared_x_axes=(("axis-0", "axis-1", "axis-2"),),
            shared_y_axes=(("axis-0", "axis-1"),),
        ),
    )

    assert figurecomposer_gridspec._gridspec_shared_axes_group(
        setup, "axis-1", "x"
    ) == ("axis-0", "axis-1", "axis-2")
    assert figurecomposer_gridspec._gridspec_shared_axes_group(
        setup, "axis-3", "x"
    ) == ("axis-3",)
    assert (
        figurecomposer_gridspec._gridspec_shared_axes_group(setup, "missing", "x") == ()
    )
    assert figurecomposer_gridspec._gridspec_shared_axes_targets(setup, "x") == {
        "axis-1": "axis-0",
        "axis-2": "axis-0",
    }

    updated = figurecomposer_gridspec._gridspec_set_shared_axes_group(
        setup, "axis-0", "x", ("axis-0", "axis-3")
    )
    assert updated.gridspec.shared_x_axes == (
        ("axis-0", "axis-3"),
        ("axis-1", "axis-2"),
    )
    assert (
        figurecomposer_gridspec._gridspec_set_shared_axes_group(
            updated, "missing", "x", ("axis-0",)
        )
        == updated
    )
    assert (
        figurecomposer_gridspec._gridspec_set_shared_axes_group(
            updated, "axis-0", "x", ("axis-0", "axis-3")
        )
        == updated
    )
    independent = figurecomposer_gridspec._gridspec_set_shared_axes_group(
        updated, "axis-0", "x", ("axis-0",)
    )
    assert independent.gridspec.shared_x_axes == (("axis-1", "axis-2"),)
    with pytest.raises(ValueError, match="unknown shared-axis dimension"):
        figurecomposer_gridspec._gridspec_shared_axes_group(
            setup, "axis-0", typing.cast("typing.Any", "z")
        )

    invalid_root = updated.gridspec.root.model_copy(update={"ncols": 3})
    invalid_setup = updated.model_copy(
        update={"gridspec": updated.gridspec.model_copy(update={"root": invalid_root})}
    )
    assert figurecomposer_gridspec._gridspec_shared_axes_targets(
        invalid_setup, "x"
    ) == {"axis-2": "axis-1"}

    restored = FigureSubplotsState.model_validate_json(updated.model_dump_json())
    assert restored == updated
    assert (
        figurecomposer_gridspec._gridspec_remove_region(updated, "missing", "axis-1")
        == updated
    )
    removed = figurecomposer_gridspec._gridspec_remove_region(updated, "root", "axis-1")
    assert removed.gridspec.shared_x_axes == (("axis-0", "axis-3"),)
    assert removed.gridspec.shared_y_axes == ()

    nested_axes = tuple(
        axis.model_copy(
            update={
                "axes_id": f"nested-{index}",
                "span": FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=index,
                    col_stop=index + 1,
                ),
            }
        )
        for index, axis in enumerate(axes[:2])
    )
    child = FigureGridSpecGridState(
        grid_id="child",
        nrows=1,
        ncols=2,
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=0, col_stop=1),
        axes=nested_axes,
    )
    nested_setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root", nrows=1, ncols=1, child_grids=(child,)
            ),
            shared_x_axes=(("nested-0", "nested-1"),),
        ),
    )
    removed_child = figurecomposer_gridspec._gridspec_remove_region(
        nested_setup, "root", "child"
    )
    assert removed_child.gridspec.root.child_grids == ()
    assert removed_child.gridspec.shared_x_axes == ()


def test_subplots_shared_axes_groups_cover_boolean_modes() -> None:
    setup = FigureSubplotsState(nrows=1, ncols=2)
    axes_ids = ("axis-0", "axis-1")

    assert (
        figurecomposer_gridspec._subplots_shared_axes_groups(setup, axes_ids, False)
        == ()
    )
    assert figurecomposer_gridspec._subplots_shared_axes_groups(
        setup, axes_ids, True
    ) == (axes_ids,)
    assert (
        figurecomposer_gridspec._subplots_shared_axes_groups(
            FigureSubplotsState(), ("axis-0",), True
        )
        == ()
    )


def test_rendered_color_cache_reuses_frozen_colors_and_misses(monkeypatch) -> None:
    figure = Figure()
    axis = figure.subplots()
    [line] = axis.plot(
        [0.0, 1.0],
        color="red",
        markerfacecolor="green",
        markeredgecolor="blue",
    )
    setattr(line, figurecomposer_rendering._OPERATION_ID_ATTR, "line")
    calls = 0
    findobj = figure.findobj

    def counted_findobj(*args, **kwargs):
        nonlocal calls
        calls += 1
        return findobj(*args, **kwargs)

    monkeypatch.setattr(figure, "findobj", counted_findobj)
    cache = _rendered_colors._RenderedColorCache()
    assert cache.color(figure, "line", "color") is None
    assert calls == 0
    cache.mark_rendered(figure)
    assert cache.color(figure, "line", "color") == mcolors.to_rgba("red")
    assert cache.color(figure, "line", "markerfacecolor") == mcolors.to_rgba("green")
    assert cache.color(figure, "line", "markeredgecolor") == mcolors.to_rgba("blue")
    assert cache.color(figure, "line", "facecolor") is None
    assert cache.color(figure, "line", "facecolor") is None
    assert calls == 1
    assert len(cache._values) <= 5

    line.set_color("yellow")
    assert cache.color(figure, "line", "color") == mcolors.to_rgba("red")
    assert calls == 1
    cache.mark_rendered(figure)
    assert cache.color(figure, "line", "color") == mcolors.to_rgba("yellow")
    assert calls == 2

    assert cache.color(figure, "missing", "color") is None
    assert cache.color(figure, "missing", "markerfacecolor") is None
    assert calls == 3
    assert len(cache._values) <= 5
    assert cache.color(figure, "line", "color") == mcolors.to_rgba("yellow")
    assert calls == 4

    cache.invalidate()
    assert cache._values == {}
    assert cache.color(figure, "line", "color") is None
    assert calls == 4
    cache.mark_rendered(figure)
    assert cache.color(Figure(), "line", "color") is None
    assert cache.color(figure, "line", "alpha") is None
    assert cache.color(figure, "line", "color", field="unknown") is None
    assert calls == 4


def test_rendered_color_cache_bounds_fit_component_snapshots(monkeypatch) -> None:
    figure = Figure()
    axis = figure.subplots()
    operation_attr = figurecomposer_rendering._OPERATION_ID_ATTR
    style_attr = _rendered_colors._FIT_STYLE_ATTR

    def tag(artist, field, component=None):
        setattr(artist, operation_attr, "fit")
        setattr(artist, style_attr, (field, component))

    for field, color in (("fit_data_kw", "red"), ("fit_line_kw", "blue")):
        [line] = axis.plot([0.0, 1.0], color=color)
        tag(line, field)
    band = axis.fill_between([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    band.set_facecolor(["yellow", "green"])
    band.set_edgecolor("none")
    tag(band, "fit_band_kw")
    component_colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, 8))
    for index, color in enumerate(component_colors):
        [line] = axis.plot([0.0, 1.0], color=color)
        tag(line, "fit_component_kw", f"component-{index}")
        band = axis.fill_between([0.0, 1.0], [0.0, 1.0], [1.0, 2.0], facecolor=color)
        tag(band, "fit_component_band_kw", f"component-{index}")

    calls = 0
    findobj = figure.findobj

    def counted_findobj(*args, **kwargs):
        nonlocal calls
        calls += 1
        return findobj(*args, **kwargs)

    monkeypatch.setattr(figure, "findobj", counted_findobj)
    cache = _rendered_colors._RenderedColorCache()
    cache.mark_rendered(figure)
    assert cache.color(figure, "fit", "color", field="fit_line_kw") == (
        mcolors.to_rgba("blue")
    )
    assert cache.color(figure, "fit", "facecolor", field="fit_band_kw") == (
        mcolors.to_rgba("yellow")
    )
    assert cache.color(figure, "fit", "edgecolor", field="fit_band_kw") == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert calls == 1

    for index, color in enumerate(component_colors):
        component = f"component-{index}"
        assert cache.color(
            figure, "fit", "color", field="fit_component_kw", component=component
        ) == tuple(color)
        assert cache.color(
            figure,
            "fit",
            "facecolor",
            field="fit_component_band_kw",
            component=component,
        ) == tuple(color)
        assert cache.color(figure, "fit", "color", field="fit_data_kw") == (
            mcolors.to_rgba("red")
        )
        assert (
            cache.color(
                figure,
                "fit",
                "edgecolor",
                field="fit_component_kw",
                component=component,
            )
            is None
        )
        assert calls == index + 2
        assert len(cache._values) <= 25
        assert all(
            value is None or (isinstance(value, tuple) and len(value) == 4)
            for value in cache._values.values()
        )

    assert (
        cache.color(
            figure, "fit", "color", field="fit_component_kw", component="missing"
        )
        is None
    )
    assert (
        cache.color(
            figure,
            "fit",
            "facecolor",
            field="fit_component_band_kw",
            component="missing",
        )
        is None
    )
    assert calls == len(component_colors) + 2
    assert len(cache._values) <= 25


def test_rendered_color_cache_does_not_retain_figure_or_artists() -> None:
    cache = _rendered_colors._RenderedColorCache()

    def transient_figure():
        figure = Figure()
        [line] = figure.subplots().plot([0.0, 1.0], color="red")
        setattr(line, figurecomposer_rendering._OPERATION_ID_ATTR, "line")
        cache.mark_rendered(figure)
        assert cache.color(figure, "line", "color") == mcolors.to_rgba("red")
        return weakref.ref(figure), weakref.ref(line)

    figure_ref, artist_ref = transient_figure()
    gc.collect()
    assert figure_ref() is None
    assert artist_ref() is None
