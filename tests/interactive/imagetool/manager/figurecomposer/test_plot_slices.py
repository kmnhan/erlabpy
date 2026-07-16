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


def _plot_slices_selection_migration_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("x", "hv", "y"),
        coords={"x": [0.0, 1.0], "hv": [10.0, 20.0, 30.0], "y": [-1.0, 0.0, 1.0, 2.0]},
        name="first",
    )


def test_figure_composer_plot_slices_migrates_shared_map_selection_to_slice_state(
    qtbot,
) -> None:
    data = _plot_slices_selection_migration_data()
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("first",),
        map_selections=(FigureDataSelectionState(source="first", qsel={"y": 0.0}),),
        axes=FigureAxesSelectionState(),
    )
    tool = FigureComposerTool.from_sources(
        {"first": data},
        sources=(FigureSourceState(name="first", label="first"),),
        operations=(operation,),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first",)
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    assert loaded_operation.slice_kwargs == {}
    assert plot_slices_model._plot_slices_shape(tool._document, loaded_operation).valid

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    assert (
        tool.findChild(
            QtWidgets.QWidget,
            "figureComposerPlotSlicesInputSelectionSection",
        )
        is None
    )


def test_figure_composer_plot_slices_migrates_shared_multi_map_selection_to_slice_state(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 0.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first", "second")
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    assert loaded_operation.slice_kwargs == {}
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_migrates_shared_selection_before_sources_restore(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    primary = first.rename("primary")
    tool = FigureComposerTool.from_sources(
        {"primary": primary},
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 0.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="primary",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first", "second")
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    tool.set_source_data({"primary": primary, "first": first, "second": second})
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_migrates_per_source_map_selections_to_aliases(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("x", "hv", "y"),
        coords={"x": [0.0, 1.0], "hv": [10.0, 20.0, 30.0], "y": [-1.0, 0.0, 1.0, 2.0]},
        name="first",
    )
    second = first + 100.0
    second.name = "second"
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 2.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first_selected", "second_selected")
    assert loaded_operation.map_selections == ()
    source_by_name = {source.name: source for source in tool.source_states()}
    assert source_by_name["first_selected"].selection_source == "first"
    assert source_by_name["first_selected"].qsel == {"y": 0.0}
    assert source_by_name["second_selected"].selection_source == "second"
    assert source_by_name["second_selected"].qsel == {"y": 2.0}
    xr.testing.assert_identical(tool.source_data()["first_selected"], first.qsel(y=0.0))
    xr.testing.assert_identical(
        tool.source_data()["second_selected"], second.qsel(y=2.0)
    )


def test_figure_composer_plot_slices_restores_deferred_source_alias_data(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    primary = first.rename("primary")
    tool = FigureComposerTool.from_sources(
        {"primary": primary},
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 2.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="primary",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first_selected", "second_selected")
    assert "first_selected" not in tool.source_data()
    tool._restore_persistence_data_items(
        {"first": first, "second": second},
        xr.Dataset(),
    )
    xr.testing.assert_identical(tool.source_data()["first_selected"], first.qsel(y=0.0))
    xr.testing.assert_identical(
        tool.source_data()["second_selected"], second.qsel(y=2.0)
    )
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_source_selector_updates_sliced_sources(
    qtbot,
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first",),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"eV": 0.0}),
                ),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("sources")

    checks = _plot_source_checks(tool)
    checks["second"].setCheckState(QtCore.Qt.CheckState.Checked)

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("first", "second")
    assert updated.map_selections == ()

    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("second", "up")].isEnabled(),
        timeout=1000,
    )
    _plot_source_move_buttons(tool)[("second", "up")].click()
    qtbot.waitUntil(
        lambda: (
            tool.tool_status.operations[0].sources
            == (
                "second",
                "first",
            )
        ),
        timeout=1000,
    )

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("second", "first")
    assert updated.map_selections == ()


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
    tool.operation_editor.select_section("sources")

    assert not tool.operation_editor.source_controls.findChildren(QtWidgets.QLineEdit)
    checks = _plot_source_checks(tool)
    assert set(checks) == {"first_source", "second_source"}
    assert checks["first_source"].checkState() == QtCore.Qt.CheckState.Checked
    assert checks["second_source"].checkState() == QtCore.Qt.CheckState.Unchecked
    assert tool.source_panel.source_status_label.isHidden()

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


def test_figure_composer_plot_slices_default_colormap_editor_uses_stylesheet(
    qtbot,
    tmp_path: Path,
) -> None:
    style_name = "erlab-test-slices-image-cmap"
    style_dir = tmp_path / "stylelib"
    style_dir.mkdir()
    (style_dir / f"{style_name}.mplstyle").write_text(
        "image.cmap: plasma\n", encoding="utf-8"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
        import matplotlib.style.core as mpl_style_core

    mpl_style_core.USER_LIBRARY_PATHS.append(str(style_dir))
    try:
        mpl_style.reload_library()
        _set_figure_stylesheets([style_name])
        data = _figure_composer_image_source("data").isel(eV=0)
        operation = FigureOperationState.plot_slices(
            label="plot_slices",
            sources=("data",),
        )
        tool = FigureComposerTool.from_sources(
            {"data": data},
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        )
        qtbot.addWidget(tool)
        tool.operation_panel.operation_list.setCurrentItem(
            tool.operation_panel.operation_list.topLevelItem(0)
        )
        tool.operation_editor.select_section("colors")

        cmap_combo = next(
            (
                candidate
                for candidate in tool.findChildren(
                    erlab.interactive.colors.ColorMapComboBox,
                    "figureComposerCmapCombo",
                )
                if tool.operation_editor.control_signal_allowed(candidate)
            ),
            None,
        )
        cmap_reverse_check = next(
            (
                candidate
                for candidate in tool.findChildren(
                    QtWidgets.QCheckBox, "figureComposerCmapReverseCheck"
                )
                if tool.operation_editor.control_signal_allowed(candidate)
            ),
            None,
        )
        assert cmap_combo is not None
        assert cmap_reverse_check is not None
        assert cmap_combo.currentData() == "plasma"
        assert cmap_reverse_check.checkState() == QtCore.Qt.CheckState.Unchecked
        assert tool.tool_status.operations[0].cmap is None

        kwargs = plot_slices_model._plot_slices_kwargs(
            tool, tool.tool_status.operations[0]
        )
        assert "cmap" not in kwargs
        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )
        assert tool.figure.axes[0].images[-1].get_cmap().name == "plasma"

        cmap_reverse_check.setChecked(True)

        assert tool.tool_status.operations[0].cmap == "plasma_r"
    finally:
        mpl_style_core.USER_LIBRARY_PATHS.remove(str(style_dir))
        mpl_style.reload_library()


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


def test_figure_composer_plot_slices_selection_error_details() -> None:
    plain = str(FigureComposerPlotSlicesSelectionError())
    assert "Details:" not in plain

    detailed = str(FigureComposerPlotSlicesSelectionError("bad key"))
    assert plain in detailed
    assert "Details: bad key" in detailed


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
    internal = erlab.utils.array._make_dims_uniform(public)
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

    assert tool.operation_panel.operation_list.topLevelItemCount() == 1
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
    operation = figurecomposer_adapter.build_figure_composer_operation(
        tool.slicer_area.images[2], source_name="data"
    )

    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert len(operation.map_selections) == 1
    assert operation.map_selections[0].source == "data"
    assert "sample_temp_idx" not in operation.model_dump_json()


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
    operation = figurecomposer_adapter.build_figure_composer_operation(
        tool.slicer_area.images[2], source_name="data"
    )

    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert operation.map_selections == (
        FigureDataSelectionState(source="data", qsel={"Track Shift": 1.0}),
    )


def test_imagetool_rejects_uneditable_plot_slices_selection(qtbot) -> None:
    tool = erlab.interactive.imagetool.ImageTool(_unsupported_plot_slices_data())
    qtbot.addWidget(tool)

    _set_unsupported_plot_slices_cursor_state(tool)

    with pytest.raises(
        FigureComposerPlotSlicesSelectionError,
        match="Cannot plot when more than one dimension",
    ):
        figurecomposer_adapter.build_figure_composer_operation(
            tool.slicer_area.images[0], source_name="data"
        )


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

    figurecomposer_adapter.show_plot_slices_selection_error(
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


def test_figure_composer_plot_slices_kwargs_normalize_colorcet_colormaps(
    qtbot,
) -> None:
    pytest.importorskip("colorcet")

    erlab.interactive.colors.load_all_colormaps()
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"cmap": "CET_C1"})
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    assert plot_slices_model._plot_slices_kwargs(tool, operation)["cmap"] == (
        "cet_CET_C1"
    )

    panel_operation = operation.model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="fire",
                ),
            ),
        }
    )

    assert (
        plot_slices_model._plot_slices_kwargs(tool, panel_operation)["cmap"]
        == "cet_fire"
    )


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

    keys = plot_slices_model._plot_slices_panel_keys(
        tool._document, tool._source_display_name, operation
    )
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert plot_slices_model._plot_slices_slice_count(tool._document, operation) == 2
    assert plot_slices_model._plot_slices_slice_labels(operation, 2) == (
        "eV=0",
        "eV=1",
    )
    assert plot_slices_model._panel_cmap_argument(tool, operation) == [
        ["magma", "plasma"]
    ]
    assert plot_slices_model._effective_panel_cmap(
        FigureOperationState.plot_slices(label="default", sources=("data",)),
        FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0),
    ) == figurecomposer_norms._matplotlib_cmap_name(options.model.colors.cmap.name)
    assert plot_slices_model._panel_line_kw_argument(tool, operation) == [
        [{"linewidth": 1.5, "color": "red"}, {"linewidth": 1.5, "color": "blue"}]
    ]
    assert plot_slices_model._has_panel_line_kw_overrides(tool, operation)
    norm_argument = plot_slices_model._panel_norm_argument(tool, operation)
    assert isinstance(norm_argument, list)
    assert plot_slices_codegen._panel_norm_uses_matplotlib_colors(tool, operation)
    assert "mcolors.Normalize" in (
        plot_slices_codegen._panel_norm_code(tool, operation) or ""
    )

    profiles, profile_keys = plot_slices_model._plot_slices_line_profiles(
        tool._document,
        tool._source_display_name,
        operation,
        maps=(source,),
    )
    assert len(profiles) == 2
    assert [(key.map_index, key.slice_index) for key in profile_keys] == [
        (0, 0),
        (0, 1),
    ]
    assert plot_slices_model._plot_slices_uses_transformed_line_maps(tool, operation)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="In a future version of xarray the default value for .*",
            category=FutureWarning,
        )
        transformed_maps = plot_slices_model._plot_slices_transformed_maps(
            tool,
            operation,
            (source,),
        )
    assert len(transformed_maps) == 1
    assert transformed_maps[0].dims == ("eV", "kx")
    assert set(
        plot_slices_editor._available_plot_slices_offset_coords(
            tool.operation_editor, operation
        )
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
                plot_slices_codegen._plot_slices_transformed_code_lines(
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
        plot_slices_model._panel_cmap_argument(tool, single_panel_operation) == "magma"
    )
    single_norm = plot_slices_model._panel_norm_argument(tool, single_panel_operation)
    assert isinstance(single_norm, mpl.colors.Normalize)
    assert plot_slices_model._panel_line_kw_argument(tool, single_panel_operation) == {
        "linewidth": 1.5,
        "color": "red",
    }

    no_override_operation = operation.model_copy(
        update={"panel_styles_enabled": False, "panel_styles": ()}
    )
    assert plot_slices_model._panel_norm_argument(tool, no_override_operation) is None
    assert plot_slices_codegen._panel_norm_code(tool, no_override_operation) is None
    assert plot_slices_model._panel_line_kw_argument(tool, no_override_operation) == {
        "linewidth": 1.5
    }

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
    assert plot_slices_model._panel_cmap_argument(tool, same_cmap_operation) == "magma"
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
    assert plot_slices_model._panel_line_kw_argument(tool, same_line_operation) == {
        "linewidth": 1.5,
        "color": "red",
    }

    selection_operation = operation.model_copy(
        update={
            "slice_dim": None,
            "slice_values": (),
            "slice_kwargs": {"kx": [-1.0, 0.0], "kx_width": 0.1},
        }
    )
    assert plot_slices_model._plot_slices_slice_labels(selection_operation, 2) == (
        "kx[0]",
        "kx[1]",
    )

    no_override_operation = operation.model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0),
            ),
        }
    )
    assert (
        plot_slices_model._panel_cmap_argument(tool, no_override_operation) == "viridis"
    )
    assert plot_slices_model._panel_norm_argument(tool, no_override_operation) is None
    assert plot_slices_codegen._panel_norm_code(tool, no_override_operation) is None
    assert not plot_slices_codegen._panel_norm_uses_matplotlib_colors(
        tool, no_override_operation
    )
    assert plot_slices_model._panel_line_kw_argument(tool, no_override_operation) == {
        "linewidth": 1.5
    }


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

    assert (
        plot_slices_model._plot_slices_batch_panel_kind(
            tool._document,
            ((0, line_operation), (1, image_operation)),
            line_operation,
        )
        == "mixed"
    )
    assert (
        plot_slices_model._plot_slices_batch_panel_kind(
            tool._document, (), line_operation
        )
        == "line"
    )

    keys = plot_slices_model._plot_slices_panel_keys(
        tool._document, tool._source_display_name, line_operation
    )
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert plot_slices_model._plot_slices_slice_labels(
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
        plot_slices_model._plot_slices_slice_count(
            tool._document, slice_kwarg_operation
        )
        == 2
    )
    shape = plot_slices_model._plot_slices_shape(tool._document, slice_kwarg_operation)
    assert shape.valid
    assert shape.panel_count == 2
    range_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(
            update={
                "slice_dim": None,
                "slice_values": (),
                "slice_kwargs": {"kx": slice(-1.0, 1.0), "eV": 0.0},
            }
        ),
    )
    assert range_shape.valid

    missing_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        FigureOperationState.plot_slices(label="missing", sources=("missing",)),
    )
    assert not missing_shape.valid
    mismatched_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        FigureOperationState.plot_slices(label="mixed", sources=("line", "other")),
    )
    assert not mismatched_shape.valid
    invalid_cut_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(update={"slice_dim": "missing", "slice_values": ()}),
    )
    assert invalid_cut_shape.valid
    incomplete_cut_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(update={"slice_values": ()}),
    )
    assert incomplete_cut_shape.valid

    image_kwargs = plot_slices_model._plot_slices_kwargs(tool, image_operation)
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
    assert _method_float_pair_args(
        tool.operation_editor,
        set_xlim_operation,
        _method_spec(set_xlim_operation),
    ) == (0.0, None)
    assert image_kwargs["subplot_kw"] == {"sharex": True}
    assert image_kwargs["annotate_kw"] == {"fontsize": 8}
    assert image_kwargs["colorbar_kw"] == {"ticks": [0.0, 1.0]}
    assert image_kwargs["alpha"] == 0.9

    explicit_norm_kwargs = plot_slices_model._plot_slices_kwargs(
        tool,
        image_operation.model_copy(
            update={"norm_name": "Normalize", "norm_gamma": None}
        ),
    )
    assert "norm" in explicit_norm_kwargs
    panel_norm_kwargs = plot_slices_model._plot_slices_kwargs(
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

    line_kwargs = plot_slices_model._plot_slices_kwargs(tool, line_operation)
    assert line_kwargs["line_kw"] == [
        [{"linewidth": 1.5, "color": "red"}],
        [{"linewidth": 1.5, "color": "blue"}],
    ]
    assert line_kwargs["line_order"] == "F"
    assert line_kwargs["gradient"] is True
    assert line_kwargs["gradient_kw"] == {"alpha": 0.2}
    transformed_kwargs = plot_slices_model._plot_slices_transformed_kwargs(
        tool,
        line_operation,
    )
    assert "eV_width" not in transformed_kwargs
    assert transformed_kwargs["eV"] == [0.0, 1.0]

    flat_axes = np.empty(4, dtype=object)
    reshaped_axes = plot_slices_render._plot_slices_axes(
        line_operation.model_copy(update={"sources": ("line", "other")}),
        (line, other),
        flat_axes,
    )
    assert isinstance(reshaped_axes, np.ndarray)
    assert reshaped_axes.shape == (2, 2)
    mismatched_axes = np.empty(3, dtype=object)
    assert (
        plot_slices_render._plot_slices_axes(
            line_operation,
            (line,),
            mismatched_axes,
        )
        is mismatched_axes
    )
    assert (
        plot_slices_render._plot_slices_axes(line_operation, (line,), object())
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
        len(plot_slices_model._operation_maps(tool._document, selection_operation)) == 1
    )
    selection_lines = plot_slices_codegen._plot_slices_code_lines(
        tool,
        selection_operation,
    )
    assert all("selected_maps" not in line for line in selection_lines)
    assert any("eplt.plot_slices" in line for line in selection_lines)
    single_selection_operation = FigureOperationState.plot_slices(
        label="single_selection",
        sources=("image",),
        map_selections=(FigureDataSelectionState(source="image", qsel={"eV": 1.0}),),
    )
    single_selection_lines = plot_slices_codegen._plot_slices_code_lines(
        tool,
        single_selection_operation,
    )
    assert len(single_selection_lines) == 1
    assert "selected_maps" not in single_selection_lines[0]
    assert "eplt.plot_slices(image" in single_selection_lines[0]
    assert (
        plot_slices_codegen._plot_slices_code_lines(
            tool,
            FigureOperationState.plot_slices(label="empty", sources=()),
        )
        == []
    )

    transform_lines = plot_slices_codegen._plot_slices_transformed_code_lines(
        tool,
        line_operation,
    )
    assert transform_lines[0] == "profiles = ["
    assert any("eplt.plot_slices" in line for line in transform_lines)
    no_slice_map_lines, no_slice_maps_code = (
        plot_slices_codegen._plot_slices_transformed_maps_code(
            line_operation.model_copy(update={"slice_dim": None, "slice_values": ()}),
            keys[:1],
        )
    )
    assert no_slice_map_lines == []
    assert no_slice_maps_code == "profiles[0]"

    assert plot_slices_editor._bool_or_text("True") is True
    assert plot_slices_editor._bool_or_text("False") is False
    assert plot_slices_editor._bool_or_text("row") == "row"
    assert plot_slices_editor._optional_number_or_text("vmin", "") is None
    assert plot_slices_editor._optional_number_or_text("cmap", "magma") == "magma"
    assert plot_slices_editor._optional_number_or_text("vmax", "1.5") == 1.5
    assert (
        plot_slices_editor._norm_field_placeholder(
            image_operation.model_copy(update={"norm_name": "CenteredPowerNorm"}),
            "vcenter",
        )
        == "0"
    )
    assert (
        plot_slices_editor._norm_field_placeholder(
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
    assert plot_slices_editor._plot_slices_color_limit_placeholders(
        placeholder_tool.operation_editor,
        placeholder_operation,
    ) == {"vmin": "0", "vmax": "11"}
    assert (
        plot_slices_editor._norm_gamma_value(
            image_operation.model_copy(update={"norm_gamma": None, "gamma": None})
        )
        == 1.0
    )
    assert plot_slices_model._norm_clip_text(None) == "default"
    assert plot_slices_model._norm_clip_from_text("True") is True
    assert plot_slices_model._norm_clip_from_text("False") is False
    assert plot_slices_model._norm_clip_from_text("default") is None

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    plot_slices_editor._update_current_norm_name(
        tool.operation_editor, "CenteredPowerNorm"
    )
    assert tool.tool_status.operations[1].norm_name == "CenteredPowerNorm"
    plot_slices_editor._update_current_norm_gamma(tool.operation_editor, 0.75)
    assert tool.tool_status.operations[1].norm_gamma == 0.75
    plot_slices_editor._update_current_norm_kwargs(
        tool.operation_editor,
        "halfrange=2.0, clip=True, custom=1",
    )
    assert tool.tool_status.operations[1].halfrange == 2.0
    assert tool.tool_status.operations[1].norm_clip is True
    assert tool.tool_status.operations[1].norm_kwargs == {"custom": 1}
    plot_slices_editor._update_current_slice_kwargs(
        tool.operation_editor,
        "eV=[0, 1], eV_width=0.2",
    )
    assert tool.tool_status.operations[1].slice_dim == "eV"
    assert tool.tool_status.operations[1].slice_width == 0.2
    plot_slices_editor._update_current_extra_kwargs(
        tool.operation_editor,
        "kx=0.0, alpha=0.5",
    )
    assert tool.tool_status.operations[1].slice_kwargs["kx"] == 0.0
    assert tool.tool_status.operations[1].extra_kwargs == {"alpha": 0.5}
    plot_slices_editor._update_current_cmap(
        tool.operation_editor, base="viridis", reverse=True
    )
    assert tool.tool_status.operations[1].cmap == "viridis_r"
    plot_slices_editor._update_current_panel_styles_enabled(
        tool.operation_editor, False
    )
    assert not tool.tool_status.operations[1].panel_styles_enabled
    plot_slices_editor._update_current_panel_styles(
        tool.operation_editor,
        (FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0, cmap="plasma"),),
    )
    assert tool.tool_status.operations[1].panel_styles_enabled
    assert tool.tool_status.operations[1].panel_styles[0].cmap == "plasma"
    plot_slices_editor._update_current_panel_styles(tool.operation_editor, ())
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
    kwargs = plot_slices_model._plot_slices_kwargs(tool, operation)
    np.testing.assert_allclose(kwargs["eV"], expected_values)
    shape = plot_slices_model._plot_slices_shape(tool._document, operation)
    assert shape.panel_count == expected_values.size
    assert shape.plot_dims == ("kx", "ky")

    code_kwargs = plot_slices_codegen._plot_slices_code_kwargs(tool, operation)
    assert isinstance(code_kwargs["eV"], figurecomposer_text._RawCode)
    captured: list[dict[str, typing.Any]] = []

    class PlotSlicesCapture:
        @staticmethod
        def plot_slices(_maps, **plot_kwargs):
            captured.append(plot_kwargs)

    exec(  # noqa: S102
        "\n".join(plot_slices_codegen._plot_slices_code_lines(tool, operation)),
        {
            "data": data,
            "eplt": PlotSlicesCapture,
            "axs": object(),
        },
    )
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0]["eV"], expected_values)

    full_values_operation = operation.model_copy(update={"slice_values_thin": 1})
    full_kwargs = plot_slices_model._plot_slices_kwargs(
        tool,
        full_values_operation,
    )
    np.testing.assert_allclose(full_kwargs["eV"], data.coords["eV"].values)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
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

    shape = plot_slices_model._plot_slices_shape(tool._document, first_operation)
    assert shape.source_text == "eV, kx, ky"
    assert shape.panel_text == "eV (1D line)"
    assert shape.selection_text == ""
    assert shape.plot_ndim == 1
    assert shape.panel_count == 2
    assert shape.valid
    invalid_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        first_operation.model_copy(
            update={
                "slice_kwargs": {"eV": 0.0, "kx": 1.0, "ky": 2.0},
            }
        ),
    )
    assert invalid_shape.plot_ndim == 0
    assert not invalid_shape.valid
    assert (
        plot_slices_spec._section_summary(tool, "selection", first_operation)
        == "additional"
    )
    assert plot_slices_spec._section_summary(tool, "view", first_operation) == "auto"
    assert plot_slices_spec._section_summary(tool, "advanced", first_operation) == ""
    assert plot_slices_spec._section_summary(tool, "unknown", first_operation) == ""

    mixed_operation = first_operation.model_copy(
        update={"sources": ("first", "second")}
    )
    mixed_shape = plot_slices_model._plot_slices_shape(tool._document, mixed_operation)
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
    empty_tool._document.replace_source_payloads({}, {})
    empty_shape = plot_slices_model._plot_slices_shape(
        empty_tool._document, empty_tool.tool_status.operations[0]
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
            plot_slices_editor._plot_source_check_state(
                tool.operation_editor, first_operation, "first"
            )
            == QtCore.Qt.CheckState.PartiallyChecked
        )

    checks = _plot_source_checks(tool)
    assert set(checks) == {"first", "second"}
    assert checks["first"].checkState() == QtCore.Qt.CheckState.Checked
    assert checks["second"].checkState() == QtCore.Qt.CheckState.Unchecked

    checks["second"].setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].sources == ("first", "second")

    _plot_source_checks(tool)["first"].setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].sources == ("second",)

    _plot_source_checks(tool)["first"].setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].sources == ("first", "second")

    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("first", "down")].isEnabled(),
        timeout=1000,
    )
    _plot_source_move_buttons(tool)[("first", "down")].click()
    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].sources[:2] == ("second", "first"),
        timeout=1000,
    )


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
        plot_slices_model._PlotSlicesPanelKey(0, 0, "panel 1"),
        plot_slices_model._PlotSlicesPanelKey(0, 1, "panel 2"),
    )
    editor = plot_slices_panel_style_editor._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
        "viridis",
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
    assert editor.cmap_combo.currentData() is plot_slices_model._MISSING
    assert not editor.norm_override_check.isTristate()
    assert editor.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.norm_combo.currentData() is plot_slices_model._MISSING
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


def test_figure_composer_plot_slices_panel_override_controls_stay_live(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(update={"cmap": "viridis", "norm_name": "PowerNorm"})
    keys = (plot_slices_model._PlotSlicesPanelKey(0, 0, "panel 1"),)
    editor = plot_slices_panel_style_editor._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
        "viridis",
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


def test_figure_composer_plot_slices_panel_style_editor_reverses_mixed_cmap(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(
        update={
            "cmap": "viridis",
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
    )
    keys = (
        plot_slices_model._PlotSlicesPanelKey(0, 0, "panel 1"),
        plot_slices_model._PlotSlicesPanelKey(0, 1, "panel 2"),
    )
    editor = plot_slices_panel_style_editor._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
        "viridis",
    )
    qtbot.addWidget(editor)
    emitted: list[tuple[FigurePlotSlicesPanelStyleState, ...]] = []
    editor.sigPanelStylesChanged.connect(emitted.append)

    for row in range(editor.panel_list.count()):
        item = editor.panel_list.item(row)
        assert item is not None
        item.setSelected(True)
    editor._sync_controls()
    assert editor.cmap_combo.currentData() is plot_slices_model._MISSING
    with QtCore.QSignalBlocker(editor.cmap_override_check):
        editor.cmap_override_check.setCheckState(QtCore.Qt.CheckState.Checked)

    editor._cmap_reverse_changed(QtCore.Qt.CheckState.Checked.value)

    assert emitted
    assert tuple(style.cmap for style in emitted[-1]) == ("viridis_r", "viridis_r")


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
        plot_slices_model._PlotSlicesPanelKey(0, 0, "panel 1"),
        plot_slices_model._PlotSlicesPanelKey(0, 1, "panel 2"),
    )
    editor = plot_slices_panel_style_editor._PanelLineStyleEditorWidget(
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
    assert editor.style_combo.currentData() is plot_slices_model._MISSING
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
    tool.operation_editor.select_section("sources")

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


def test_figure_composer_plot_slices_mappable_tagging_edges() -> None:
    operation = FigureOperationState.plot_slices(label="data", sources=("data",))
    key = plot_slices_model._PlotSlicesPanelKey(0, 0, "panel")
    figure = Figure()
    axis = figure.subplots()
    old_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    old_ids = plot_slices_render._axis_mappable_ids((axis,))

    plot_slices_render._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key, plot_slices_model._PlotSlicesPanelKey(0, 1, "extra")),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )

    new_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    plot_slices_render._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key,),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )
    assert (
        getattr(
            new_mappable,
            plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        )
        == operation.operation_id
    )
    assert getattr(
        new_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
    ) == (0, 0)


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
    assert set(tool.findChildren(QtWidgets.QSplitter)) == {
        tool.source_panel.source_splitter,
        tool.operation_panel.splitter,
    }
    assert (
        tool.operation_panel.splitter.widget(0) is tool.operation_panel.operation_list
    )
    assert tool.operation_panel.splitter.widget(1) is tool.operation_editor
    assert tool.operation_editor.scroll_area.widget() is tool.operation_editor.stack
    assert not tool.operation_editor.scroll_area.isAncestorOf(
        tool.operation_editor.navigator
    )
    editor_tabs = tool.findChild(QtWidgets.QTabWidget, "figureComposerEditorTabs")
    assert editor_tabs is tool.editor_tabs
    assert [
        editor_tabs.widget(index).objectName() for index in range(editor_tabs.count())
    ] == [
        "figureComposerSourcesPage",
        "figureComposerLayoutPage",
        "figureComposerRecipePage",
    ]
    assert editor_tabs.currentWidget() is tool.operation_panel
    assert isinstance(tool.layout_panel.layout(), QtWidgets.QGridLayout)
    layout_grid = typing.cast("QtWidgets.QGridLayout", tool.layout_panel.layout())
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
    assert dpi_label.buddy() is tool.layout_panel.dpi_spin
    assert tool.findChild(QtWidgets.QWidget, "figureComposerShareControls") is not None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerRatioControls") is not None
    assert layout_grid.getItemPosition(layout_grid.indexOf(dpi_label)) == (
        5,
        0,
        1,
        2,
    )
    assert layout_grid.getItemPosition(
        layout_grid.indexOf(tool.layout_panel.dpi_spin)
    ) == (
        5,
        2,
        1,
        3,
    )
    assert tool.layout_panel.gridspec_editor_widget.isHidden()
    gridspec_container = tool.findChild(
        QtWidgets.QWidget, "figureComposerGridSpecEditorContainer"
    )
    assert gridspec_container is tool.layout_panel.gridspec_editor_container
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
    assert layout_grid.getItemPosition(
        layout_grid.indexOf(tool.layout_panel.layout_combo)
    ) == (
        6,
        2,
        1,
        3,
    )
    add_step_button = tool.findChild(
        QtWidgets.QToolButton, "figureComposerAddStepButton"
    )
    assert add_step_button is tool.operation_panel.add_step_button
    assert add_step_button.parent() is tool.operation_panel
    assert add_step_button.menu() is None
    assert add_step_button.property("uses_inline_menu_arrow") is True
    assert tool.operation_panel.add_step_menu.parent() is add_step_button
    assert add_step_button.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
    )
    step_toolbar_buttons = (
        tool.operation_panel.add_step_button,
        tool.operation_panel.copy_button,
        tool.operation_panel.cut_button,
        tool.operation_panel.paste_button,
        tool.operation_panel.delete_button,
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
    assert [
        action.data() for action in tool.operation_panel.add_step_menu.actions()
    ] == [
        "set_palette",
        "plot_array",
        "plot_slices",
        "line",
        "bz_overlay",
        "photon_energy_overlay",
        "method:erlab",
        "method:axes",
        "method:figure",
        "custom",
    ]
    assert tool.findChild(QtWidgets.QTabWidget, "figureComposerInspectorTabs") is None
    assert tool.findChild(QtWidgets.QToolBox) is None
    assert tool.findChild(QtWidgets.QWidget, "figureComposerStepNavigator") is not None
    assert tool.operation_editor.stack.objectName() == "figureComposerStepSectionStack"
    assert tool.operation_editor.section_keys == (
        "sources",
        "axes",
        "selection",
        "view",
        "colors",
        "advanced",
    )
    assert [
        tool.operation_editor.stack.widget(index).objectName()
        for index in range(tool.operation_editor.stack.count())
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
        for button in _operation_section_buttons(tool)
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
    tool.layout_panel.dpi_spin.setValue(180.0)
    tool.layout_panel.dpi_spin.editingFinished.emit()
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
            tool.layout_panel.nrows_spin,
            tool.layout_panel.ncols_spin,
            tool.layout_panel.width_spin,
            tool.layout_panel.height_spin,
            tool.layout_panel.width_mm_spin,
            tool.layout_panel.height_mm_spin,
            tool.layout_panel.dpi_spin,
        )
    )
    tool.operation_editor.select_section("colors")
    tool.operation_editor.request_update(axis="equal")
    assert tool.findChild(QtWidgets.QToolBox) is None
    assert (
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    tool.operation_editor.select_section("view")
    view_page = tool.operation_editor.stack.currentWidget()
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
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerPlotSlicesViewPage"
    )
    qtbot.mouseClick(
        _operation_section_button(tool, "colors"), QtCore.Qt.MouseButton.LeftButton
    )
    assert (
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    operation_item = tool.operation_panel.operation_list.topLevelItem(0)
    operation_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].enabled is False
    assert (
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    operation_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].enabled is True
    assert (
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerPlotSlicesColorsPage"
    )
    tool.operation_editor.select_section("axes")
    tool.axes_expression_edit.setFocus()
    tool.axes_expression_edit.setText("axs[:, 0]")
    tool.axes_expression_edit.editingFinished.emit()
    qtbot.wait(1)
    assert tool.tool_status.operations[0].axes.expression == "axs[:, 0]"
    assert (
        tool.operation_editor.stack.currentWidget().objectName()
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
        tool.operation_editor.stack.currentWidget().objectName()
        == "figureComposerTargetAxesPage"
    )
    tool._target_current_operation_all_axes()
    tool.operation_editor.select_section("colors")
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
    assert norm_combo.currentIndex() == figurecomposer_norms._NORM_CHOICES.index(
        "PowerNorm"
    )
    assert norm_combo.count() == len(figurecomposer_norms._NORM_CHOICES)
    assert cmap_combo.currentData() == "viridis"
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
        tool.operation_editor.request_update(colorbar="right")
    finally:
        plt.close(current_fig)
    assert tool.tool_status.operations[0].colorbar == "right"
    assert not any(
        "Adding colorbar to a different Figure" in str(warning.message)
        for warning in recwarn
    )
    tool.operation_editor.request_update(colorbar="none")
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
    tool.operation_editor.request_update(axis="auto")
    assert len(show_activations) == activation_count
    figure_window = tool.figure_window
    figure_window.canvas.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
    qtbot.keyClick(
        figure_window.canvas,
        QtCore.Qt.Key.Key_W,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    qtbot.keyRelease(tool, QtCore.Qt.Key.Key_Control)
    qtbot.wait_until(lambda: not figure_window.isVisible(), timeout=5000)
    assert len(tool.tool_status.operations) == 1
    activation_count = len(show_activations)
    tool.show_figure_window()
    qtbot.wait_until(lambda: figure_window.isVisible(), timeout=5000)
    qtbot.wait_until(lambda: len(show_activations) > activation_count, timeout=5000)
    assert True in show_activations[activation_count:]
    activation_count = len(show_activations)
    tool.layout_panel.width_spin.setValue(7.0)
    tool.layout_panel.height_spin.setValue(5.0)
    tool.layout_panel.height_spin.editingFinished.emit()
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
    assert np.isclose(tool.layout_panel.width_mm_spin.value(), 7.0 * 25.4, atol=0.01)
    assert np.isclose(tool.layout_panel.height_mm_spin.value(), 5.0 * 25.4, atol=0.01)

    tool.layout_panel.width_mm_spin.setValue(127.0)
    tool.layout_panel.height_mm_spin.setValue(76.2)
    tool.layout_panel.height_mm_spin.editingFinished.emit()
    assert tool.tool_status.setup.figsize == (5.0, 3.0)
    assert np.isclose(tool.layout_panel.width_spin.value(), 5.0)
    assert np.isclose(tool.layout_panel.height_spin.value(), 3.0)

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
    assert np.isclose(tool.layout_panel.width_spin.value(), target_width, atol=0.03)
    assert np.isclose(tool.layout_panel.height_spin.value(), target_height, atol=0.03)
    assert np.isclose(
        tool.layout_panel.width_mm_spin.value(), target_width * 25.4, atol=0.8
    )
    assert np.isclose(
        tool.layout_panel.height_mm_spin.value(), target_height * 25.4, atol=0.8
    )
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


def test_figure_composer_toolbar_plot_slices_panel_cmap_uses_stylesheet(
    qtbot,
    tmp_path: Path,
) -> None:
    style_name = "erlab-test-toolbar-panel-cmap"
    style_dir = tmp_path / "stylelib"
    style_dir.mkdir()
    (style_dir / f"{style_name}.mplstyle").write_text(
        "image.cmap: plasma\n", encoding="utf-8"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
        import matplotlib.style.core as mpl_style_core

    mpl_style_core.USER_LIBRARY_PATHS.append(str(style_dir))
    try:
        mpl_style.reload_library()
        _set_figure_stylesheets([style_name])
        data = _figure_composer_image_source("data")
        operation = FigureOperationState.plot_slices(
            label="plot_slices",
            sources=("data",),
            axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            slice_dim="eV",
            slice_values=(-0.5, 0.5),
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
        tool.show_figure_window(activate=False)

        figurecomposer_toolbar_dialogs.show_axes_customize_dialog(tool)
        dialog = typing.cast("QtWidgets.QDialog", tool._axes_customize_dialog)
        qtbot.addWidget(dialog)
        editor = dialog.findChild(
            plot_slices_panel_style_editor._PanelStyleEditorWidget
        )
        assert editor is not None
        assert editor.cmap_combo.currentData() == "plasma"
        assert editor.cmap_override_check.checkState() == QtCore.Qt.CheckState.Unchecked

        editor.cmap_override_check.click()

        assert tool.tool_status.operations[0].panel_styles == (
            FigurePlotSlicesPanelStyleState(
                map_index=0,
                slice_index=0,
                cmap="plasma",
            ),
        )
    finally:
        mpl_style_core.USER_LIBRARY_PATHS.remove(str(style_dir))
        mpl_style.reload_library()


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
    assert plot_slices_model._available_plot_slices_line_color_coords(
        tool._document, tool._source_display_name, operation
    ) == ["eV"]
    line_kw = plot_slices_model._panel_line_kw_argument(tool, operation)
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

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
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
    context = FigureDocument(
        FigureRecipeState(),
        source_data={"data": data},
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
    ).model_copy(update={"slice_values_mode": "all"})

    assert plot_slices_model._all_coordinate_slice_values(context, operation) == ()
    assert plot_slices_codegen._all_coordinate_slice_values_code(operation) is None
    assert (
        plot_slices_codegen._first_plot_slices_source_code(
            operation.model_copy(update={"sources": ()})
        )
        is None
    )
    selection_operation = operation.model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="data", qsel={"eV": 0.1}),
            )
        }
    )
    assert (
        plot_slices_codegen._first_plot_slices_source_code(selection_operation)
        == "data"
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(
            operation.model_copy(update={"sources": (), "slice_dim": "eV"})
        )
        is None
    )
    assert plot_slices_model._slice_values_mode_from_text("not a mode") == "manual"
    assert plot_slices_model._line_color_mode_from_text("By coordinate") == "coordinate"
    assert plot_slices_model._line_color_mode_from_text("Manual") == "manual"
    assert (
        plot_slices_model._all_coordinate_slice_values_error(
            context, operation, data.dims
        )
        == "Choose a dimension before using all coordinate values."
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(context, operation)
        == "Choose a dimension."
    )

    missing_dim = operation.model_copy(update={"slice_dim": "missing"})
    assert plot_slices_model._all_coordinate_slice_values(context, missing_dim) == ()
    assert "'missing' is not an input dimension" in (
        plot_slices_model._all_coordinate_slice_values_error(
            context, missing_dim, data.dims
        )
    )
    assert "'missing' is not an input dimension" in (
        plot_slices_model._all_coordinate_slice_values_summary(context, missing_dim)
    )

    missing_source_context = FigureDocument(FigureRecipeState())
    assert (
        plot_slices_model._all_coordinate_slice_values(
            missing_source_context,
            operation.model_copy(update={"slice_dim": "eV"}),
        )
        == ()
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(
            missing_source_context, missing_dim
        )
        == "Select at least one valid source."
    )

    string_coord = xr.DataArray(
        np.ones((2, 3)),
        dims=("label", "kx"),
        coords={"label": ["a", "b"], "kx": [-1.0, 0.0, 1.0]},
        name="string_coord",
    )
    string_context = FigureDocument(
        FigureRecipeState(),
        source_data={"data": string_coord},
    )
    string_operation = operation.model_copy(update={"slice_dim": "label"})
    assert (
        plot_slices_model._all_coordinate_slice_values(string_context, string_operation)
        == ()
    )
    assert "numeric and non-empty" in (
        plot_slices_model._all_coordinate_slice_values_error(
            string_context, string_operation, string_coord.dims
        )
    )
    assert "numeric and non-empty" in (
        plot_slices_model._all_coordinate_slice_values_summary(
            string_context, string_operation
        )
    )

    thinned = operation.model_copy(update={"slice_dim": "eV", "slice_values_thin": 2})
    assert plot_slices_model._all_coordinate_slice_values(
        context, thinned
    ) == pytest.approx(tuple(data.thin({"eV": 2}).coords["eV"].values))
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(context, thinned)
        == "eV: 5 values, 3 plotted"
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(thinned)
        == 'data.thin({"eV": 2}).coords["eV"].values'
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(
            thinned.model_copy(update={"slice_values_thin": 1})
        )
        == 'data.coords["eV"].values'
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(
            context, thinned.model_copy(update={"slice_values_thin": 1})
        )
        == "eV: 5 values"
    )
    assert (
        plot_slices_codegen._plot_slices_slice_values_code(
            context, operation.model_copy(update={"slice_values_mode": "manual"})
        )
        == "[None]"
    )
    assert (
        plot_slices_codegen._plot_slices_slice_values_code(
            context,
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
    assert plot_slices_model._plot_slices_panel_qsel_kwargs(
        operation.model_copy(
            update={"slice_dim": "eV", "slice_values": (0.1,), "slice_width": 0.2}
        ),
        plot_slices_model._PlotSlicesPanelKey(0, 0, ""),
    ) == {"eV": 0.1, "eV_width": 0.2}


def test_figure_composer_plot_slices_label_codegen_helper_variants(qtbot) -> None:
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("a", "b"),
        slice_dim="eV",
        slice_values=(0.1, 0.2),
    ).model_copy(update={"line_label_text": "{source}:{number}:{eV:g}"})
    fields = {"source", "number", "eV"}
    single_key = (plot_slices_model._PlotSlicesPanelKey(0, 0, "A"),)
    by_slice_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(0, index, f"A {index}")
        for index in range(2)
    )
    by_source_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(index, 0, f"S {index}")
        for index in range(2)
    )
    grid_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(map_index, slice_index, "")
        for map_index in range(2)
        for slice_index in range(2)
    )
    namespace = {"slice_values": [0.1, 0.2]}

    single = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, single_key, ("alpha",), "slice_values", fields
    )
    assert eval(single, namespace) == {"label": "alpha:1:0.1"}  # noqa: S307

    by_slice = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, by_slice_keys, ("alpha",), "slice_values", fields
    )
    assert eval(by_slice, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "alpha:2:0.2"},
    ]

    by_source = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, by_source_keys, ("alpha", "beta"), "slice_values", fields
    )
    assert eval(by_source, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "beta:2:0.1"},
    ]

    grid = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, grid_keys, ("alpha", "beta"), "slice_values", fields
    )
    assert eval(grid, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "alpha:2:0.2"}],
        [{"label": "beta:3:0.1"}, {"label": "beta:4:0.2"}],
    ]

    fortran_grid = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation.model_copy(update={"order": "F"}),
        grid_keys,
        ("alpha", "beta"),
        "slice_values",
        fields,
    )
    assert eval(fortran_grid, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "beta:2:0.1"}],
        [{"label": "alpha:3:0.2"}, {"label": "beta:4:0.2"}],
    ]
    for code in (by_slice, by_source, grid, fortran_grid):
        available_names = {
            *_SCRIPT_REPLAY_ALLOWED_BUILTINS,
            "slice_values",
        }
        _validate_script_code_names(
            f"line_kw = {code}",
            available_names,
            {},
        )

    styled = plot_slices_codegen._plot_slices_styled_label_line_kw_code(
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
    styled_value = eval(styled, namespace)  # noqa: S307
    assert styled_value[1][1]["label"] == "beta:4:0.2"
    assert styled_value[1][1]["color"] == "red"
    assert styled_value[0][0]["linewidth"] == 1.5

    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    styled_operation = operation.model_copy(
        update={
            "sources": ("a",),
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, line_kw={"color": "blue"}
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="a", label="a"),),
            operations=(styled_operation,),
            primary_source="a",
        ),
        source_data={"a": data},
    )
    qtbot.addWidget(tool)
    styled_line_kw = plot_slices_codegen._plot_slices_label_line_kw_code(
        tool, styled_operation
    )
    assert styled_line_kw is not None
    styled_line_kw_value = eval(styled_line_kw, {}, namespace)  # noqa: S307
    assert styled_line_kw_value[0][1]["label"] == "a:2:0.2"
    assert styled_line_kw_value[0][1]["color"] == "blue"

    assert (
        plot_slices_codegen._plot_slices_panel_index_expr(
            "F",
            map_count=2,
            slice_count=3,
            map_index_expr="map_index",
            slice_index_expr="slice_index",
        )
        == "slice_index * 2 + map_index"
    )
    assert (
        plot_slices_codegen._line_kw_dict_code({"alpha": 0.5}, "label_code")
        == "{**{'alpha': 0.5}, 'label': label_code}"
    )
    assert (
        plot_slices_codegen._line_kw_dict_code({}, "label_code")
        == "{'label': label_code}"
    )


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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
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
    assert values_edit is not None
    assert values_edit.text() == "0"
    assert width_edit is not None
    assert width_edit.text() == "0.1"
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"

    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
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
    tool.operation_editor.select_section("selection")
    slice_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"


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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("colors")
    first_page = tool.operation_editor.stack.currentWidget()
    first_cmap_combo = first_page.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
    )
    assert first_cmap_combo is not None
    assert first_cmap_combo.currentData() == "viridis"

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("colors")

    assert tool.tool_status.operations[0].cmap == "viridis"
    assert tool.tool_status.operations[1].cmap == "magma_r"

    _select_operation_rows(tool, (0, 1))
    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    cmap_combo = colors_page.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerCmapCombo"
    )
    halfrange_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
    )
    assert cmap_combo is not None
    assert cmap_combo.currentIndex() == 0
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
    halfrange_edit = tool.operation_editor.stack.currentWidget().findChild(
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

    plot_slices_editor._update_current_norm_gamma(tool.operation_editor, 0.75)

    assert tool.tool_status.operations[0].norm_gamma == 0.75
    assert render_calls == []
    assert tool._preview_render_update_pending

    plot_slices_editor._update_current_norm_gamma(tool.operation_editor, 0.5)

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

    order_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerOrderCombo")
    shape = plot_slices_model._plot_slices_shape(
        tool._document, tool.tool_status.operations[0]
    )
    assert shape.valid
    assert shape.plot_dims == ("kx",)
    assert shape.plot_ndim == 1
    assert order_combo is not None

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
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
    assert line_style_combo.currentData() == "--"
    assert line_width_spin is not None
    assert line_width_spin.value() == 1.5
    assert marker_combo is not None
    assert marker_combo.currentData() == "o"
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

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
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

    assert "transform" in tool.operation_editor.section_keys
    tool.operation_editor.select_section("transform")
    transform_page = tool.operation_editor.stack.currentWidget()
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

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
        )
        is None
    )
    assert colors_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
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

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    line_color_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
    )
    assert line_color_edit is not None
    assert line_color_edit.text() == ""
    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
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
    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()

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

    tool.operation_editor.select_section("view")
    view_page = tool.operation_editor.stack.currentWidget()
    assert view_page.findChild(QtWidgets.QComboBox, "figureComposerAxisCombo")
    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    assert (
        colors_page.findChild(QtWidgets.QComboBox, "figureComposerSameLimitsCombo")
        is None
    )

    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
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

    assert "transform" not in tool.operation_editor.section_keys
    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
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

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
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

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
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
    tool.operation_editor.select_section("colors")

    colors_page = tool.operation_editor.stack.currentWidget()
    panel_check = colors_page.findChild(
        QtWidgets.QCheckBox, "figureComposerPlotSlicesPanelStylesCheck"
    )
    assert panel_check is not None
    panel_check.setChecked(True)
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QWidget, "figureComposerPlotSlicesPanelLineStyleEditor"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.operation_editor.stack.currentWidget()
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("view")
    annotate_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAnnotateKwEdit"
    )
    assert annotate_kwargs_edit is not None
    assert annotate_kwargs_edit.text() == 'fontsize=8, color="black"'

    tool.operation_editor.select_section("colors")
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

    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerExtraKwEdit")
    assert extra_kwargs_edit is not None
    assert extra_kwargs_edit.text() == "alpha=0.5, zorder=2"

    _select_operation_rows(tool, (1,))
    tool.operation_editor.select_section("colors")
    gradient_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerGradientKwEdit"
    )
    assert gradient_kwargs_edit is not None
    assert gradient_kwargs_edit.text() == 'color="C0", alpha=0.25'

    _select_operation_rows(tool, (2,))
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(3)
    )
    tool.operation_editor.select_section("method")
    erlab_method_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit"
    )
    assert erlab_method_kwargs_edit is not None
    assert erlab_method_kwargs_edit.text() == "fontsize=9"


def test_figure_composer_imagetool_norm_parser_uses_structured_fields() -> None:
    updates = figurecomposer_adapter._norm_updates(
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
    class Floatable:
        def __float__(self) -> float:
            return 1.5

    fallback = object()
    assert figurecomposer_adapter._plain_value(None) is None
    assert figurecomposer_adapter._plain_value(np.bool_(True)) is True
    assert figurecomposer_adapter._plain_value([np.int64(1)]) == [1]
    assert figurecomposer_adapter._plain_value((np.float64(2.0),)) == (2.0,)
    assert figurecomposer_adapter._plain_value({"a": np.int64(3)}) == {"a": 3}
    assert figurecomposer_adapter._plain_value(np.int64(4)) == 4
    assert figurecomposer_adapter._plain_value(np.float64(5.0)) == 5.0
    assert figurecomposer_adapter._plain_value(Floatable()) == 1.5
    assert figurecomposer_adapter._plain_value(fallback) is fallback
    assert figurecomposer_adapter._indexer_state(slice(1, 3, 2)) == {
        "kind": "slice",
        "start": 1,
        "stop": 3,
        "step": 2,
    }
    assert figurecomposer_adapter._indexer_state(2) == 2

    invalid_norms = (
        "not valid python",
        "1",
        "eplt",
        "mcolors.PowerNorm(1)",
        "eplt.PowerNorm(1)",
        "eplt.CenteredPowerNorm(**kwargs)",
    )
    for norm_code in invalid_norms:
        assert figurecomposer_adapter._norm_updates(norm_code) is None

    assert figurecomposer_adapter._operation_updates({"norm": object()}) is None
    assert (
        figurecomposer_adapter._operation_updates({"norm": "eplt.PowerNorm(1)"}) is None
    )
    assert figurecomposer_adapter._operation_updates({"cmap": "|dynamic_cmap|"}) is None
    assert figurecomposer_adapter._operation_updates(
        {"gamma": np.float64(0.5), "alpha": np.int64(2)}
    ) == {
        "norm_name": "PowerNorm",
        "norm_gamma": 0.5,
        "extra_kwargs": {"alpha": 2},
    }


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
    tool.operation_editor.select_section("colors")

    colors_page = tool.operation_editor.stack.currentWidget()
    norm_combo = colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert norm_combo is not None
    assert norm_combo.currentIndex() == figurecomposer_norms._NORM_CHOICES.index(
        "PowerNorm"
    )
    assert norm_combo.count() == len(figurecomposer_norms._NORM_CHOICES)
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
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerVcenterNormEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.operation_editor.stack.currentWidget()
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
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    colors_page = tool.operation_editor.stack.currentWidget()
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
        refreshed_edit = tool.operation_editor.stack.currentWidget().findChild(
            QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
        )
        return refreshed_edit is not None and refreshed_edit.text() == 'custom="extra"'

    qtbot.waitUntil(
        norm_kwargs_text_updated,
        timeout=1000,
    )
    colors_page = tool.operation_editor.stack.currentWidget()
    norm_kwargs_edit = colors_page.findChild(
        QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
    )
    assert norm_kwargs_edit is not None
    assert norm_kwargs_edit.text() == 'custom="extra"'

    colors_page = tool.operation_editor.stack.currentWidget()
    norm_combo = colors_page.findChild(QtWidgets.QComboBox, "figureComposerNormCombo")
    assert norm_combo is not None
    _activate_combo_text(norm_combo, "Normalize")
    assert tool.tool_status.operations[0].norm_name == "Normalize"
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerVminNormEdit"
            )
            is not None
            and tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerHalfrangeNormEdit"
            )
            is None
        ),
        timeout=1000,
    )
    colors_page = tool.operation_editor.stack.currentWidget()
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
