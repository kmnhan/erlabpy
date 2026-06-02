import contextlib
import typing
import warnings
from collections.abc import Callable

import matplotlib as mpl
import matplotlib.scale as mscale
import numpy as np
import pytest
import xarray as xr
from matplotlib import style as mpl_style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive._figurecomposer._code as figurecomposer_code
import erlab.interactive._figurecomposer._defaults as figurecomposer_defaults
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._widgets as figurecomposer_widgets
import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureExportState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._operations import (
    _line_profile as figurecomposer_line_profile,
)
from erlab.interactive._options import options
from erlab.interactive._options.schema import AppOptions, FigureOptions
from erlab.interactive.imagetool import itool
from tests.interactive.imagetool.manager.helpers import select_tools

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


def _set_figure_stylesheets(stylesheets: list[str]) -> None:
    options.model = options.model.model_copy(
        update={"figure": FigureOptions(stylesheets=stylesheets)}
    )


def _expected_layout_from_rcparams() -> str | None:
    if mpl.rcParams["figure.constrained_layout.use"]:
        return "constrained"
    if mpl.rcParams["figure.autolayout"]:
        return "tight"
    return None


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


def test_figure_composer_defaults_follow_stylesheet_rcparams(
    restore_interactive_options,
) -> None:
    _set_figure_stylesheets(["classic"])

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


def test_figure_composer_generated_code_uses_available_stylesheets(
    qtbot,
    restore_interactive_options,
) -> None:
    _set_figure_stylesheets(["classic", "missing-style"])
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
    assert "# Skipped unavailable stylesheets: 'missing-style'" in code
    assert tool.preview_pixmap is not None
    namespace = {"data": data}
    with mpl.rc_context():
        exec(code, namespace)  # noqa: S102
    namespace["plt"].close(namespace["fig"])


def test_figure_composer_rechecks_configured_stylesheets_after_erlab_import(
    qtbot,
    monkeypatch,
    restore_interactive_options,
) -> None:
    available: list[str] = []
    monkeypatch.setattr("erlab.interactive._stylesheets.mpl_style.available", available)
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: available.append("classic"),
    )
    _set_figure_stylesheets(["classic"])
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
    original_draw = FigureCanvasAgg.draw

    def draw_with_warning(self, *args, **kwargs):
        warnings.warn(_COLLAPSED_LAYOUT_WARNING, UserWarning, stacklevel=2)
        return original_draw(self, *args, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, "draw", draw_with_warning)

    assert tool.preview_pixmap is not None
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

    tool.operation_list.setCurrentRow(1)
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
    assert axes_sequence_code(((0, 0), (0, 1), (0, 2), (0, 3))) == ("axs[0, :].flat")
    assert axes_code(((0, 2), (1, 2), (2, 2))) == "axs[:, 2]"
    assert axes_code(((0, 0), (0, 1), (0, 2))) == "axs[0, :3]"
    assert axes_code(((0, 1), (0, 2), (0, 3))) == "axs[0, 1:4]"
    assert axes_code(((1, 0), (2, 0))) == "axs[1:3, 0]"
    assert axes_code(((1, 1), (1, 2), (2, 1), (2, 2))) == "axs[1:3, 1:3]"
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

    assert tool.findChildren(FigureCanvasQTAgg) == []
    assert tool.findChildren(NavigationToolbar2QT) == []
    assert tool.findChildren(QtWidgets.QSplitter) == []
    editor_tabs = tool.findChild(QtWidgets.QTabWidget, "figureComposerEditorTabs")
    assert editor_tabs is tool.editor_tabs
    assert [
        editor_tabs.widget(index).objectName() for index in range(editor_tabs.count())
    ] == ["figureComposerLayoutPage", "figureComposerRecipePage"]
    assert editor_tabs.currentWidget() is tool.recipe_page
    assert isinstance(tool.layout_page.layout(), QtWidgets.QGridLayout)
    layout_grid = typing.cast("QtWidgets.QGridLayout", tool.layout_page.layout())
    assert layout_grid.rowCount() == 7
    assert layout_grid.columnCount() == 5
    assert tool.findChild(QtWidgets.QWidget, "figureComposerGridControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerSizeControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerSizeMmControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerShareControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerRatioControls") is not None
    layout_label = tool.findChild(QtWidgets.QLabel, "figureComposerLayoutControls")
    assert layout_label is not None
    assert layout_grid.getItemPosition(layout_grid.indexOf(layout_label)) == (
        3,
        0,
        1,
        2,
    )
    assert layout_grid.getItemPosition(layout_grid.indexOf(tool.layout_combo)) == (
        3,
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
        "plot_slices",
        "line",
        "method:erlab",
        "method:axes",
        "method:figure",
        "custom",
    ]
    assert [action.text() for action in tool.add_step_menu.actions()] == [
        "Slice Plot",
        "Line/Profile",
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
        "cuts",
        "limits",
        "colors",
        "style",
        "advanced",
    ]
    assert [
        tool.step_editor_stack.widget(index).objectName()
        for index in range(tool.step_editor_stack.count())
    ] == [
        "figureComposerStepSourcesPage",
        "figureComposerTargetAxesPage",
        "figureComposerPlotSlicesCutsPage",
        "figureComposerPlotSlicesLimitsPage",
        "figureComposerPlotSlicesColorsPage",
        "figureComposerPlotSlicesStylePage",
        "figureComposerPlotSlicesAdvancedPage",
    ]
    assert tool.findChild(QtWidgets.QTabWidget, "figureComposerPlotSlicesTabs") is None
    colors_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesColorsPage"
    )
    cuts_page = tool.findChild(QtWidgets.QWidget, "figureComposerPlotSlicesCutsPage")
    limits_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerPlotSlicesLimitsPage"
    )
    style_page = tool.findChild(QtWidgets.QWidget, "figureComposerPlotSlicesStylePage")
    crop_check = tool.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesCropCheck"
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
    assert cuts_page is not None
    assert limits_page is not None
    assert style_page is not None
    assert crop_check is not None
    assert same_limits_combo is not None
    assert axis_combo is not None
    assert annotate_kwargs_edit is not None
    assert colorbar_kwargs_edit is not None
    assert limits_page.isAncestorOf(crop_check)
    assert not cuts_page.isAncestorOf(crop_check)
    assert same_limits_combo.parent() is style_page
    assert axis_combo.parent() is style_page
    assert annotate_kwargs_edit.parent() is style_page
    assert colorbar_kwargs_edit.parent() is colors_page
    assert all(
        widget.toolTip()
        for widget in (
            tool.nrows_spin,
            tool.ncols_spin,
            tool.width_spin,
            tool.height_spin,
            tool.width_mm_spin,
            tool.height_mm_spin,
            tool.layout_combo,
            tool.sharex_combo,
            tool.sharey_combo,
            tool.width_ratios_edit,
            tool.height_ratios_edit,
            tool.operation_list,
            tool.source_list,
            tool.use_all_axes_button,
            tool.keep_valid_axes_button,
            tool.axes_expression_edit,
            same_limits_combo,
            axis_combo,
            annotate_kwargs_edit,
            colorbar_kwargs_edit,
        )
    )
    assert all(button.toolTip() for button in tool.step_section_buttons.values())
    assert all(
        isinstance(button.property("section_title"), str)
        for button in tool.step_section_buttons.values()
    )
    assert tool.axes_selector.toolTip()
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
    tool._select_step_section("colors")
    tool._update_current_operation(axis="equal")
    assert tool.findChild(QtWidgets.QToolBox) is None
    assert tool._current_step_section_key == "colors"
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    tool._select_step_section("limits")
    limit_edits = tool.step_editor_stack.currentWidget().findChildren(
        QtWidgets.QLineEdit
    )
    assert limit_edits[0].text() == ""
    assert "," in limit_edits[0].placeholderText()
    assert limit_edits[1].text() == ""
    assert "," in limit_edits[1].placeholderText()
    limit_edits[0].setFocus()
    limit_edits[0].setText("0, 1")
    limit_edits[0].editingFinished.emit()
    assert tool.tool_status.operations[0].xlim == (0.0, 1.0)
    limit_edits[1].setFocus()
    limit_edits[1].setText("2.5")
    limit_edits[1].editingFinished.emit()
    assert tool.tool_status.operations[0].ylim == 2.5
    restored_status = FigureRecipeState.model_validate(tool.tool_status.model_dump())
    assert restored_status.operations[0].ylim == 2.5
    assert "ylim=2.5" in tool.generated_code()
    assert (
        tool.step_editor_stack.currentWidget().objectName()
        == "figureComposerPlotSlicesLimitsPage"
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
    qtbot.mousePress(selector, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(selector, end)
    qtbot.mouseRelease(selector, QtCore.Qt.MouseButton.LeftButton, pos=end)
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
    cmap_combo.setCurrentText("magma")
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
    assert not tool.figure_window.isVisible()
    assert len(tool.figure.axes) == 2
    assert tool.figure_window.parent() is None
    assert tool.figure_window.canvas.figure is tool.figure
    live_figure = tool.figure
    live_canvas = tool.figure_window.canvas
    live_axes_count = len(live_figure.axes)
    preview = tool.preview_pixmap
    assert preview is not None
    assert not preview.isNull()
    assert preview.width() > 0
    assert preview.height() > 0
    assert tool.figure is live_figure
    assert tool.figure_window.canvas is live_canvas
    assert len(tool.figure.axes) == live_axes_count
    show_activations: list[bool] = []
    original_show_for_setup = tool.figure_window.show_for_setup

    def record_show_for_setup(*args, activate: bool) -> None:
        show_activations.append(activate)
        original_show_for_setup(*args, activate=activate)

    monkeypatch.setattr(tool.figure_window, "show_for_setup", record_show_for_setup)

    setup_before = tool.tool_status.setup.model_copy()
    code_before = tool.generated_code()
    tool.resize(240, 360)
    assert tool.tool_status.setup == setup_before
    assert tool.generated_code() == code_before

    exported: dict[str, tuple[float, float]] = {}
    tool.figure.set_size_inches((12.0, 9.0), forward=False)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("figure.png", ""),
    )
    monkeypatch.setattr(
        tool.figure,
        "savefig",
        lambda filename, **kwargs: exported.setdefault(
            "figsize", tuple(tool.figure.get_size_inches())
        ),
    )
    tool.export_figure()
    assert exported["figsize"] == setup_before.figsize

    tool.show()
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
                ),
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

    tool.operation_list.setCurrentRow(2)
    tool._select_step_section("line")
    line_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    data_values_axis_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerDataValuesAxisCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert data_values_axis_combo is not None
    assert profile_coordinate_combo.itemData(0) is None
    assert profile_values_combo.itemData(0) is None
    assert profile_coordinate_combo.findData("kx") >= 0
    assert profile_values_combo.findData("kx") >= 0
    profile_coordinate_combo.setCurrentIndex(profile_coordinate_combo.findData("kx"))
    assert tool.tool_status.operations[2].line_x == "kx"
    profile_coordinate_combo.setCurrentIndex(0)
    assert tool.tool_status.operations[2].line_x is None
    profile_values_combo.setCurrentIndex(profile_values_combo.findData("kx"))
    assert tool.tool_status.operations[2].line_y == "kx"
    line_page = tool.step_editor_stack.currentWidget()
    profile_values_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_values_combo is not None
    profile_values_combo.setCurrentIndex(0)
    assert tool.tool_status.operations[2].line_y is None
    line_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.toolTip()
    assert profile_values_combo.toolTip()
    assert data_values_axis_combo.toolTip()
    assert all(
        widget.toolTip() for widget in line_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in line_page.findChildren(QtWidgets.QCheckBox)
    )

    tool.operation_list.setCurrentRow(3)
    assert tool.operation_list.item(3).text() == "clean_labels"
    assert tool.step_section_buttons["method"].text() == "clean_labels"
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
                        "method_coordinate_system": "axes",
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
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("method")
    method_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerAxesMethodCombo")
    coord_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCoordCombo"
    )
    text_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodTextEdit")
    kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit")
    assert method_combo is not None
    assert coord_combo is not None
    assert text_edit is not None
    assert kwargs_edit is not None
    assert method_combo.currentText() == "Text"
    assert coord_combo.currentText() == "axes"
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
    yscale_combo.setCurrentText(y_scale)
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
        QtWidgets.QLineEdit, "figureComposerAxesMethodTitlePadEdit"
    )
    assert title_edit is not None
    assert title_loc_combo is not None
    assert title_pad_edit is not None
    assert title_edit.text() == "Left title"
    assert title_loc_combo.currentText() == "left"
    assert title_pad_edit.text() == "2"

    tool.operation_list.setCurrentRow(11)
    tool._select_step_section("method")
    x_margin_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodXMarginEdit"
    )
    y_margin_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodYMarginEdit"
    )
    tight_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodMarginsTightCombo"
    )
    assert x_margin_edit is not None
    assert y_margin_edit is not None
    assert tight_combo is not None
    assert x_margin_edit.text() == "0.1"
    assert y_margin_edit.text() == "0.2"
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

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
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

    code = tool.generated_code()
    assert (
        'ax.text(0.1, 0.9, "Panel", ha="left", va="top", transform=ax.transAxes)'
    ) in code
    assert 'ax.axvline(0.5, color="red", linestyle="--")' in code
    assert "ax.axvspan(0.2, 0.4, alpha=0.25)" in code
    assert 'ax.set_xticks((0.0, 1.0), ("left", "right"))' in code
    assert 'ax.grid(True, which="major", axis="x")' in code
    assert "ax.set_axis_off()" in code
    assert 'ax.set_xscale("log")' in code
    assert f'ax.set_yscale("{y_scale}")' in code
    assert 'ax.set_title("Left title", loc="left", pad=2.0)' in code
    assert 'ax.set_xlabel("Momentum", loc="right", labelpad=3.0)' in code
    assert 'ax.set_ylabel("Energy", loc="top", labelpad=4.0)' in code
    assert "ax.margins(x=0.1, y=0.2, tight=False)" in code
    assert "ax.set_aspect(2.5, share=True)" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
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

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("method")
    loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodLegendLocCombo"
    )
    columns_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLegendColumnsEdit"
    )
    title_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLegendTitleEdit"
    )
    assert loc_combo is not None
    assert columns_edit is not None
    assert title_edit is not None
    assert loc_combo.currentText() == "upper right"
    assert columns_edit.text() == "1"
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
    assert 'ax.legend(loc="upper right", ncols=1' in code
    assert 'title="Axis legend"' in code
    assert "bbox_to_anchor=(1.0, 1.0)" in code
    assert 'fig.legend(loc="lower center", ncols=1' in code
    assert 'title="Figure legend"' in code

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_legend() is not None
    assert namespace["fig"].legends[0].get_title().get_text() == "Figure legend"


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

    policy_combo.setCurrentText("Selected axes together")
    assert tool.tool_status.operations[1].method_call_policy == "ax_keyword"
    assert "eplt.nice_colorbar(ax=axs)" in tool.generated_code()


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

    def set_line_edit(page: QtWidgets.QWidget, name: str, text: str) -> None:
        edit = line_edit(page, name)
        edit.setText(text)
        edit.editingFinished.emit()

    def operation(row: int) -> FigureOperationState:
        return tool.tool_status.operations[row]

    page = select_method(0)
    combo_box(
        page, "figureComposerERLabCleanLabelsRemoveInnerTicksCombo"
    ).setCurrentText("True")
    assert operation(0).method_args == (True,)

    page = select_method(1)
    set_line_edit(page, "figureComposerERLabLabelSubplotsStartEdit", "3")
    combo_box(page, "figureComposerERLabLabelSubplotsOrderCombo").setCurrentText("F")
    combo_box(page, "figureComposerERLabLabelSubplotsLocCombo").setCurrentText(
        "lower right"
    )
    set_line_edit(page, "figureComposerERLabLabelSubplotsOffsetEdit", "1, 2")
    set_line_edit(page, "figureComposerERLabLabelSubplotsPrefixEdit", "(")
    set_line_edit(page, "figureComposerERLabLabelSubplotsSuffixEdit", ")")
    combo_box(page, "figureComposerERLabLabelSubplotsNumericCombo").setCurrentText(
        "True"
    )
    combo_box(page, "figureComposerERLabLabelSubplotsCapitalCombo").setCurrentText(
        "True"
    )
    combo_box(page, "figureComposerERLabLabelSubplotsFontWeightCombo").setCurrentText(
        "bold"
    )
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
    set_line_edit(page, "figureComposerERLabLabelPropertiesSiEdit", "-3")
    set_line_edit(page, "figureComposerERLabLabelPropertiesNameEdit", "Energy")
    set_line_edit(page, "figureComposerERLabLabelPropertiesUnitEdit", "eV")
    combo_box(page, "figureComposerERLabLabelPropertiesOrderCombo").setCurrentText("F")
    assert operation(2).method_args == ({"eV": [0, 1]},)
    assert operation(2).method_kwargs == {
        "decimals": 2,
        "si": -3,
        "name": "Energy",
        "unit": "eV",
        "order": "F",
    }

    page = select_method(3)
    set_line_edit(page, "figureComposerERLabNiceColorbarWidthEdit", "10")
    set_line_edit(page, "figureComposerERLabNiceColorbarAspectEdit", "4")
    set_line_edit(page, "figureComposerERLabNiceColorbarPadEdit", "2")
    combo_box(page, "figureComposerERLabNiceColorbarMinMaxCombo").setCurrentText("True")
    combo_box(page, "figureComposerERLabNiceColorbarOrientationCombo").setCurrentText(
        "horizontal"
    )
    combo_box(page, "figureComposerERLabNiceColorbarFloatingCombo").setCurrentText(
        "True"
    )
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
    set_line_edit(page, "figureComposerERLabProportionalColorbarIndexEdit", "0")
    combo_box(
        page, "figureComposerERLabProportionalColorbarImageOnlyCombo"
    ).setCurrentText("True")
    set_line_edit(page, "figureComposerERLabProportionalColorbarTicksEdit", "[0, 1]")
    assert operation(4).method_kwargs == {
        "index": 0,
        "image_only": True,
        "ticks": [0, 1],
    }

    page = select_method(5)
    combo_box(page, "figureComposerERLabSetTitlesOrderCombo").setCurrentText("F")
    assert operation(5).method_kwargs == {"order": "F"}

    page = select_method(6)
    set_line_edit(page, "figureComposerERLabFermilineValueEdit", "0.1")
    combo_box(page, "figureComposerERLabFermilineOrientationCombo").setCurrentText("v")
    assert operation(6).method_kwargs == {"value": 0.1, "orientation": "v"}

    page = select_method(7)
    set_line_edit(page, "figureComposerERLabMarkPointsPointsEdit", "0, 1")
    set_line_edit(page, "figureComposerERLabMarkPointsLabelsEdit", "G, M")
    set_line_edit(page, "figureComposerERLabMarkPointsYEdit", "0.25, 0.5")
    set_line_edit(page, "figureComposerERLabMarkPointsPadEdit", "1, 2")
    combo_box(page, "figureComposerERLabMarkPointsLiteralCombo").setCurrentText("True")
    combo_box(page, "figureComposerERLabMarkPointsRomanCombo").setCurrentText("False")
    combo_box(page, "figureComposerERLabMarkPointsBarCombo").setCurrentText("True")
    assert operation(7).method_args == ((0, 1), ("G", "M"))
    assert operation(7).method_kwargs == {
        "y": (0.25, 0.5),
        "pad": (1.0, 2.0),
        "literal": True,
        "roman": False,
        "bar": True,
    }

    page = select_method(8)
    combo_box(page, "figureComposerERLabScaleUnitsAxisCombo").setCurrentText("y")
    set_line_edit(page, "figureComposerERLabScaleUnitsSiEdit", "3")
    combo_box(page, "figureComposerERLabScaleUnitsPrefixCombo").setCurrentText("False")
    combo_box(page, "figureComposerERLabScaleUnitsPowerCombo").setCurrentText("True")
    assert operation(8).method_args == ("y", 3)
    assert operation(8).method_kwargs == {"prefix": False, "power": True}

    page = select_method(9)
    combo_box(page, "figureComposerERLabFancyLabelsRadiansCombo").setCurrentText("True")
    assert operation(9).method_kwargs == {"radians": True}

    page = select_method(10)
    assert (
        page.findChild(QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit") is None
    )

    page = select_method(11)
    assert operation(11).method_kwargs == {}
    assert line_edit(page, "figureComposerERLabSizebarValueEdit").text() == "1"
    assert line_edit(page, "figureComposerERLabSizebarUnitEdit").text() == "m"
    set_line_edit(page, "figureComposerERLabSizebarValueEdit", "2")
    set_line_edit(page, "figureComposerERLabSizebarUnitEdit", "m")
    set_line_edit(page, "figureComposerERLabSizebarSiEdit", "-6")
    set_line_edit(page, "figureComposerERLabSizebarResolutionEdit", "0.001")
    set_line_edit(page, "figureComposerERLabSizebarDecimalsEdit", "1")
    set_line_edit(page, "figureComposerERLabSizebarLabelEdit", "200 um")
    combo_box(page, "figureComposerERLabSizebarLocCombo").setCurrentText("lower left")
    set_line_edit(page, "figureComposerERLabSizebarPadEdit", "0.2")
    set_line_edit(page, "figureComposerERLabSizebarBorderPadEdit", "0.6")
    set_line_edit(page, "figureComposerERLabSizebarSepEdit", "4")
    combo_box(page, "figureComposerERLabSizebarFrameCombo").setCurrentText("True")
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
    combo_box(page, "figureComposerERLabUnifyClimImageOnlyCombo").setCurrentText("True")
    combo_box(page, "figureComposerERLabUnifyClimAutoscaleCombo").setCurrentText("True")
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
    ).model_copy(update={"line_x": "kx", "line_values_axis": "x"})
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

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(tool.generated_code(), namespace)  # noqa: S102
    line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile.values)
    np.testing.assert_allclose(line.get_ydata(), profile["kx"].values)


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
            "line_offset_source": "associated",
            "line_offset_coord": "temperature",
            "line_offset_scale": 0.01,
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

    tool._select_step_section("line")
    line_page = tool.step_editor_stack.currentWidget()
    offset_source_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    assert (
        line_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is not None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetScaleEdit")
        is not None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )

    offset_source_combo.setCurrentText("manual")
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
            )
            is None
        ),
        timeout=1000,
    )
    line_page = tool.step_editor_stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "manual"
    assert tool.tool_status.operations[0].line_offset_scale == 1.0
    assert (
        line_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetScaleEdit")
        is None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is not None
    )

    offset_source_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    offset_source_combo.setCurrentText("index")
    qtbot.waitUntil(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerLineOffsetScaleEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    line_page = tool.step_editor_stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "index"
    assert (
        line_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetScaleEdit")
        is not None
    )
    assert (
        line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )
    tool._replace_operation(0, operation)

    assert figurecomposer_line_profile._available_line_offset_coords(
        tool, operation
    ) == ["temperature"]
    assert figurecomposer_line_profile._line_offsets_for_profiles(
        tool,
        operation.model_copy(
            update={"line_offset_source": "index", "line_offset_scale": 2.0}
        ),
        profiles,
    ) == (0.0, 2.0, 4.0)
    assert figurecomposer_line_profile._line_offsets_for_profiles(
        tool,
        operation.model_copy(
            update={"line_offset_source": "coordinate", "line_offset_scale": 0.5}
        ),
        profiles,
    ) == (0.0, 0.5, 1.0)
    assert figurecomposer_line_profile._line_offsets_for_profiles(
        tool, operation, profiles
    ) == (0.1, 0.2, 0.3)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert fig.axes[0].get_legend() is None
    for index, line in enumerate(fig.axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("red", "green", "blue")[index]

    namespace: dict[str, typing.Any] = {"profile_data": profile_data}
    code = tool.generated_code()
    assert "ax.legend()" not in code
    exec(code, namespace)  # noqa: S102
    for index, line in enumerate(namespace["fig"].axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("red", "green", "blue")[index]

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
    exec(shared_label_tool.generated_code(), namespace)  # noqa: S102
    assert [line.get_label() for line in namespace["fig"].axes[0].lines] == [
        "shared",
        "shared",
        "shared",
    ]


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
            "line_color": "black",
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
                    label="cuts",
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
    assert "profiles =" in code
    assert "profile_scales = [0.1, 0.2, 0.3]" in code
    assert "profile_offsets = [-0.2, 0.0, 0.2]" in code
    assert "for ax, profile, scale, offset in zip(" in code
    assert "ax.plot(profile['kx'], offset + scale * profile" in code

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
                    label="cuts",
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
    assert operation.line_color == "black"
    assert operation.line_x == "kx"
    assert operation.line_iter_dim == "cut"
    assert operation.line_selection == {"cut": [0.0, 1.0, 2.0], "cut_width": 0.25}
    assert operation.axes.axes == ((0, 0), (0, 1), (0, 2))

    tool._select_step_section("line")
    placement_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerProfilePlacementCombo"
    )
    normalize_combo = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert placement_combo is not None
    assert normalize_combo is not None
    assert placement_combo.currentText() == "One profile per axis"
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
    tool._select_step_section("line")
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
    assert line_operation.line_labels == ("profile A",)
    assert legend_operation.kind == FigureOperationKind.METHOD
    assert legend_operation.method_family == FigureMethodFamily.AXES
    assert legend_operation.method_name == "legend"
    assert legend_operation.axes == line_operation.axes

    labels_edit.setText("profile B")
    labels_edit.editingFinished.emit()

    assert len(tool.tool_status.operations) == 2
    assert tool.tool_status.operations[0].line_labels == ("profile B",)


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
    tool._select_step_section("line")
    line_page = tool.step_editor_stack.currentWidget()
    normalize_combo = line_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert normalize_combo is not None
    normalize_combo.setCurrentText("Each profile by mean")

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
                    update={"line_color": "C0"}
                ),
                FigureOperationState.line(label="second", source="profile").model_copy(
                    update={"line_color": "C1"}
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("line")
    line_page = tool.step_editor_stack.currentWidget()
    color_edit = line_page.findChild(QtWidgets.QLineEdit, "figureComposerLineColorEdit")
    assert color_edit is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"

    color_edit.editingFinished.emit()
    assert [operation.line_color for operation in tool.tool_status.operations] == [
        "C0",
        "C1",
    ]

    color_edit.setText("black")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert [operation.line_color for operation in tool.tool_status.operations] == [
        "black",
        "black",
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
    tool._select_step_section("line")
    line_page = tool.step_editor_stack.currentWidget()
    coordinate_combo = line_page.findChild(
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
    tool._select_step_section("line")
    labels_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None
    labels_edit.setText("profile")
    labels_edit.editingFinished.emit()

    operations = tool.tool_status.operations
    assert len(operations) == 5
    assert [
        operation.line_labels
        for operation in operations
        if operation.kind == FigureOperationKind.LINE
    ] == [("profile",), ("profile",), ("profile",)]
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
                    label="cuts",
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
    tool._select_step_section("cuts")
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
    dimension_combo.setCurrentText("kx")

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
                    label="cuts",
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
    tool._select_step_section("cuts")
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
    assert colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVminNormEdit")
    assert colors_page.findChild(QtWidgets.QLineEdit, "figureComposerVmaxNormEdit")
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit")
        is None
    )

    norm_combo.setCurrentText("CenteredInversePowerNorm")
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
    norm_combo.setCurrentText("CenteredPowerNorm")
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
    norm_combo.setCurrentText("Normalize")
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
                        "cmap": "C1",
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
    assert "1D line over kx" in shape_summary.text()
    assert "2 selected for 2 panels" in shape_summary.text()
    assert order_combo is not None

    tool._select_step_section("colors")
    colors_page = tool.step_editor_stack.currentWidget()
    line_color_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorEdit"
    )
    gradient_check = colors_page.findChild(
        QtWidgets.QCheckBox, "figureComposerGradientCheck"
    )
    assert line_color_edit is not None
    assert line_color_edit.text() == "C1"
    assert gradient_check is not None
    assert gradient_check.isChecked()
    assert colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo") is None
    assert (
        colors_page.findChild(QtWidgets.QLineEdit, "figureComposerColorbarKwEdit")
        is None
    )

    tool._select_step_section("style")
    style_page = tool.step_editor_stack.currentWidget()
    assert (
        style_page.findChild(QtWidgets.QComboBox, "figureComposerSameLimitsCombo")
        is None
    )

    code = tool.generated_code()
    assert "cmap=" in code
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
    tool._select_step_section("style")
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

    tool.operation_list.setCurrentRow(1)
    tool._select_step_section("colors")
    gradient_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerGradientKwEdit"
    )
    assert gradient_kwargs_edit is not None
    assert gradient_kwargs_edit.text() == 'color="C0", alpha=0.25'

    tool.operation_list.setCurrentRow(2)
    tool._select_step_section("line")
    line_selection_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerLineSelectionEdit"
    )
    assert line_selection_edit is not None
    assert line_selection_edit.text() == "eV=0.0, eV_width=0.1"

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

    tool.nrows_spin.setValue(1)
    tool._setup_controls_changed()

    assert tool.tool_status.operations[0].axes.axes == ((1, 1),)
    assert tool._operation_has_invalid_axes(tool.tool_status.operations[0])
    assert tool.step_editor_stack.currentWidget() is tool.target_axes_page
    assert tool._current_step_section_key == "axes"
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
        qtbot.addWidget(preserved_tool)
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
        qtbot.addWidget(unnamed_tool)
        unnamed_uid = manager.add_figuretool(unnamed_tool, show=False)
        assert manager._child_node(unnamed_uid).display_text == "Figure 6"


def test_manager_create_figure_uses_first_selected_colormap(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(8.0).reshape(2, 2, 2) - 4.0,
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [0.0, 1.0],
            },
            name="first",
        )
        second = xr.DataArray(
            np.arange(8.0).reshape(2, 2, 2),
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [0.0, 1.0],
            },
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area.set_colormap(
            "magma",
            gamma=0.75,
            reverse=True,
            high_contrast=True,
            zero_centered=True,
            levels_locked=True,
            levels=(-2.0, 4.0),
        )
        manager.get_imagetool(1).slicer_area.set_colormap("viridis", gamma=0.25)
        vmin, vmax = manager.get_imagetool(0).slicer_area.colormap_properties["levels"]

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.sources == ("data_0", "data_1")
        assert operation.cmap == "magma_r"
        assert operation.same_limits is True
        assert operation.norm_name == "CenteredInversePowerNorm"
        assert operation.norm_gamma == pytest.approx(0.75)
        assert operation.vcenter == pytest.approx(0.5 * (vmin + vmax))
        assert operation.halfrange == pytest.approx(0.5 * (vmax - vmin))


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
