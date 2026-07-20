import typing
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._figurecomposer_adapter import (
    build_figure_composer_operation,
)
from erlab.interactive.imagetool.manager._figurecomposer import _dialogs
from tests.interactive.imagetool.manager.helpers import select_tools

from ._common import (
    _set_unsupported_plot_slices_cursor_state,
    _unsupported_plot_slices_data,
)


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
                return _dialogs._FIGURE_DIALOG_NEW

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

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
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 1),))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

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
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes_ids=("axis-a",))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

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

        manager._figure_collection.select_uid(first_uid)
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

        manager._figure_collection.select_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 3, timeout=5000
        )

        first_custom_copy_uid = uid_for_name(manager, "Band map copy")
        assert manager._selected_figure_uids() == [first_custom_copy_uid]

        manager._figure_collection.select_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 4, timeout=5000
        )

        second_custom_copy_uid = uid_for_name(manager, "Band map copy 2")
        assert manager._selected_figure_uids() == [second_custom_copy_uid]


def test_manager_duplicate_deferred_figure_materializes_saved_payload(
    qtbot,
    tmp_path,
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

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_node = manager._child_node(figure_uid)
        figure_node.name = "Band map"
        figure_tool = figure_node.tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        manager.get_imagetool(0).hide()

        workspace_path = tmp_path / "deferred-figure-duplicate.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )

        source_node = manager._tool_graph.root_wrappers[0]
        loaded_node = manager._child_node(figure_uid)
        assert source_node.pending_workspace_memory_payload is not None
        assert loaded_node.pending_workspace_tool_payload is not None
        assert loaded_node.tool_window is None

        manager._figure_collection.select_uid(figure_uid)
        manager.duplicate_selected()

        assert source_node.pending_workspace_memory_payload is not None
        assert loaded_node.pending_workspace_tool_payload is None
        loaded_tool = loaded_node.tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        duplicate_uid = next(
            uid for uid in manager._tool_graph.figure_uids if uid != figure_uid
        )
        duplicate_node = manager._child_node(duplicate_uid)
        duplicate_tool = duplicate_node.tool_window
        assert isinstance(duplicate_tool, FigureComposerTool)
        assert duplicate_node.display_text == "Band map copy"
        assert duplicate_tool.tool_status == loaded_tool.tool_status
        xr.testing.assert_identical(duplicate_tool.tool_data, loaded_tool.tool_data)
        assert manager._selected_figure_uids() == [duplicate_uid]
        assert duplicate_uid in manager._workspace_state.dirty_added
        assert figure_uid not in manager._workspace_state.dirty_added


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
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(first, _in_manager=True),
            show=False,
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(second, _in_manager=True),
            show=False,
        )

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
        second_tool = manager.get_imagetool(1)
        second_tool.slicer_area.set_colormap("viridis", gamma=0.25)
        vmin, vmax = first_tool.slicer_area.colormap_properties["levels"]
        expected_first = build_figure_composer_operation(
            first_tool.slicer_area.images[0], source_name="first"
        )
        expected_second = build_figure_composer_operation(
            second_tool.slicer_area.images[0], source_name="second"
        )

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.sources == ("first_selected", "second_selected")
        sources = {source.name: source for source in figure_tool.source_states()}
        assert sources["first_selected"].selection_source == "first"
        assert sources["first_selected"].qsel == expected_first.map_selections[0].qsel
        assert sources["second_selected"].selection_source == "second"
        assert sources["second_selected"].qsel == expected_second.map_selections[0].qsel
        assert operation.order == "F"
        assert figure_tool.tool_status.setup.nrows == 1
        assert figure_tool.tool_status.setup.ncols == 2
        assert operation.slice_dim is None
        assert operation.slice_values == ()
        assert operation.slice_width is None
        assert operation.slice_kwargs == {}
        assert operation.transpose == expected_first.transpose
        assert operation.xlim == expected_first.xlim
        assert operation.ylim == expected_first.ylim
        assert operation.crop == expected_first.crop
        assert operation.axis == expected_first.axis
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

        dialog = _dialogs._AppendFigureTargetDialog(manager, (figure_uid,), operation)

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

        dialog = _dialogs._AppendFigureTargetDialog(manager, (figure_uid,), operation)

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

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
        assert not dialog.selector_stack.isHidden()
        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_NEW)
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_NEW
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

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=1,
            selected_figure_uid=figure_uid,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE
        assert dialog.selected_source_alias() == "line"
        assert dialog.source_combo.currentData() == "line"
        assert dialog.selector_stack.isHidden()
        assert not dialog.source_combo.isHidden()
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()
        assert dialog._figure_source_count(None) == 0
        assert dialog._figure_source_count("missing") == 0

        class EmptyFigureNode:
            uid = "empty-figure"
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
        first_uid = manager.add_figuretool(first_tool, show=False)
        second_uid = manager.add_figuretool(second_tool, show=False)

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (first_uid, second_uid),
            FigureOperationState.line(label="line", source="line"),
            allow_new_figure=True,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
        assert not dialog.selector_stack.isHidden()
        ok_button = dialog.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button is not None
        assert ok_button.isEnabled()

        dialog.figure_combo.setCurrentIndex(dialog.figure_combo.findData(first_uid))
        assert dialog.figure_uid() == first_uid
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
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
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_ADD_SOURCE)
        )
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_SOURCE
        assert dialog.selected_target() is None
        assert dialog.selector_stack.isHidden()
        assert ok_button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_REPLACE_SOURCE)
        )
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE
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
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_ADD_STEP)
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

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=2,
            selected_figure_uid=figure_uid,
        )

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_REPLACE_SOURCE)
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert not button.isEnabled()
        assert not dialog.status_label.isHidden()


def test_manager_prompt_append_figure_target_auto_and_cancel_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        assert manager._figure_workflows._prompt_append_figure_target(None) is None
        assert (
            manager._figure_workflows._prompt_append_figure_target(
                None, figure_uid="missing"
            )
            is None
        )

        single_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        single_uid = manager.add_figuretool(single_tool, show=False)
        assert manager._figure_workflows._append_single_axis_selection(single_uid) == (
            FigureAxesSelectionState(axes=((0, 0),))
        )
        assert manager._figure_workflows._prompt_append_figure_target(None) == (
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

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", RejectDialog)
        assert (
            manager._figure_workflows._prompt_append_figure_target(
                None, figure_uid=wide_uid
            )
            is None
        )


def test_manager_child_imagetool_gets_figure_context_actions(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def action_names(tool: erlab.interactive.imagetool.ImageTool) -> set[str]:
        names: set[str] = set()
        for plot in tool.slicer_area.axes:
            menu = plot.vb.getMenu(None)
            assert menu is not None
            names.update(action.objectName() for action in menu.actions())
        return names

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
        assert all(plot.vb.menu is None for plot in child.slicer_area.axes)
        assert all(
            plot._plot_with_matplotlib_action is None for plot in child.slicer_area.axes
        )

        manager.add_imagetool_child(child, 0, show=False)

        assert all(plot.vb.menu is None for plot in child.slicer_area.axes)
        assert "itool_plot_with_matplotlib_action" in action_names(child)
        assert "itool_append_to_figure_action" in action_names(child)
        main_plot = child.slicer_area.axes[0]
        main_plot.vb.setMenuEnabled(False)
        assert main_plot.vb.menu is None
        main_plot.vb.setMenuEnabled(True)
        rebuilt_menu = main_plot.vb.getMenu(None)
        assert rebuilt_menu is not None
        rebuilt_action_names = {
            action.objectName() for action in rebuilt_menu.actions()
        }
        assert "itool_plot_with_matplotlib_action" in rebuilt_action_names
        assert "itool_append_to_figure_action" in rebuilt_action_names


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
        assert figure.tool_status.operations[-1].line_source == source_name


def test_manager_explicit_figure_operations_use_readable_source_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(8.0).reshape(2, 4),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
            name="sample map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        script_name = manager._script_input_name_for_node(manager._node_for_target(0))
        assert script_name != "sample_map"
        figure_uid = manager.create_figure_from_targets(
            (0,),
            operation=FigureOperationState.plot_array(label="plot", source=script_name),
            show=False,
        )

        assert figure_uid is not None
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        assert tuple(figure.source_data()) == ("sample_map",)
        assert figure.tool_status.operations[-1].sources == ("sample_map",)


def test_manager_custom_figure_code_uses_readable_source_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(8.0).reshape(2, 4),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
            name="sample map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        script_name = manager._script_input_name_for_node(manager._node_for_target(0))
        assert script_name != "sample_map"
        for argument in ("operation", "custom_code"):
            code = f"fig.__dict__['{argument}_total'] = float({script_name}.sum())"
            kwargs: dict[str, typing.Any]
            if argument == "operation":
                kwargs = {
                    "operation": FigureOperationState.custom(
                        label="summary",
                        code=code,
                        trusted=True,
                    )
                }
            else:
                kwargs = {"custom_code": code}

            figure_uid = manager.create_figure_from_targets((0,), show=False, **kwargs)

            assert figure_uid is not None
            figure = manager._child_node(figure_uid).tool_window
            assert isinstance(figure, FigureComposerTool)
            [custom_operation] = figure.tool_status.operations
            assert script_name not in custom_operation.code
            assert "sample_map" in custom_operation.code
            figurecomposer_rendering._render_into_figure(
                figure, figure.figure, sync_visible=False
            )
            assert figure._operation_render_errors == {}
            assert figure.figure.__dict__[f"{argument}_total"] == float(data.sum())

        ambiguous_code = f"{script_name} = {script_name}.mean()"
        for kwargs in (
            {
                "operation": FigureOperationState.custom(
                    label="ambiguous",
                    code=ambiguous_code,
                    trusted=True,
                )
            },
            {"custom_code": ambiguous_code},
        ):
            with pytest.raises(
                FigureComposerInputError,
                match="also binds",
            ):
                manager.create_figure_from_targets((0,), show=False, **kwargs)


def test_manager_append_explicit_operation_uses_conflict_free_source_alias(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        for offset in (0.0, 10.0):
            itool(
                xr.DataArray(
                    np.arange(8.0).reshape(2, 4) + offset,
                    dims=("x", "y"),
                    coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
                    name="sample map",
                ),
                manager=True,
            )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        assert tuple(figure.source_data()) == ("sample_map",)

        script_name = manager._script_input_name_for_node(manager._node_for_target(1))
        assert script_name != "sample_map_2"
        appended = manager.append_figure_from_targets(
            (1,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            operation=FigureOperationState.plot_array(
                label="overlay", source=script_name
            ),
            show=False,
        )

        assert appended is True
        assert tuple(figure.source_data()) == ("sample_map", "sample_map_2")
        assert figure.tool_status.operations[-1].sources == ("sample_map_2",)


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
            FigureOperationKind.PLOT_ARRAY,
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
            FigureOperationKind.PLOT_ARRAY,
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

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeAppendDialog)

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            operation=FigureOperationState.line(label="overlay", source="line"),
            show=False,
        )

        assert appended is True
        assert figure_tool.tool_status.operations[-1].axes.axes == ((0, 1),)


def test_manager_figure_action_multi_source_append_preserves_image_colormaps(
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
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 0), (0, 1)))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeAppendDialog)

        manager.create_figure_action.trigger()

        appended = figure_tool.tool_status.operations[-2:]
        assert [operation.kind for operation in appended] == [
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.PLOT_ARRAY,
        ]
        assert [operation.sources for operation in appended] == [
            ("first",),
            ("second",),
        ]
        assert [operation.axes.axes for operation in appended] == [
            ((0, 0),),
            ((0, 1),),
        ]
        assert [operation.cmap for operation in appended] == ["magma", "viridis_r"]
        assert all(not operation.panel_styles_enabled for operation in appended)
